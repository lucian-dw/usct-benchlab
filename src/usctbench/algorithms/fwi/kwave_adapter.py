"""Adapter for k-Wave/WaveformInversionUST-style MATLAB FWI results."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from usctbench.metrics.image import compute_baseline_improvement_metrics, compute_image_metrics
from usctbench.schema import AlgorithmConfig, ReconstructionResult, ResultStatus, USCTCase


class KWaveFWIAdapterAlgorithm:
    """Load or launch an external k-Wave frequency-domain FWI run.

    The default path is intentionally non-invasive: it reads an existing
    MATLAB result MAT file and converts it to the standard ReconstructionResult.
    Set `run_external: true` only in A100 configs that deliberately invoke the
    external `USCT_kwave` pipeline.
    """

    name = "fwi_kwave_adapter"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        result_path = _configured_path(config, "result_path")
        if result_path is None:
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.SKIPPED,
                failure_reason="fwi_kwave_adapter requires parameters.result_path",
            )

        run_external = bool(config.parameters.get("run_external", False))
        if run_external:
            launched = _run_external_pipeline(case, config, result_path)
            if launched.status != ResultStatus.SUCCESS:
                return launched

        if not result_path.exists():
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.SKIPPED,
                failure_reason=f"k-Wave FWI result file not found: {result_path}",
            )

        try:
            external = read_kwave_fwi_result(result_path)
        except Exception as exc:
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.FAILED,
                failure_reason=f"failed to read k-Wave FWI result: {type(exc).__name__}: {exc}",
            )

        sound_speed = _resize_to_shape(external["sound_speed_mps"], case.grid.shape)
        attenuation = _resize_to_shape(external["attenuation_np_per_m"], case.grid.shape) if external.get("attenuation_np_per_m") is not None else None
        c0 = float(config.parameters.get("baseline_sound_speed_mps", case.metadata.get("reference_sound_speed_mps", 1500.0)))
        metrics: dict[str, Any] = {
            "external_result_loaded": True,
            "external_result_path": str(result_path),
            "external_dataset_path": external.get("dataset_path") or "",
            "iterations": int(external.get("iterations", 0)),
            "initial_loss": external.get("initial_loss"),
            "final_loss": external.get("final_loss"),
            "loss_decreased": external.get("loss_decreased"),
            "matlab_psnr_value": external.get("psnr_value"),
            "matlab_ssim_value": external.get("ssim_value"),
        }
        if case.ground_truth.sound_speed_mps is not None:
            truth = np.asarray(case.ground_truth.sound_speed_mps, dtype=float)
            metrics.update(compute_image_metrics(sound_speed, truth, mask=case.grid.roi_mask))
            metrics.update(compute_baseline_improvement_metrics(sound_speed, truth, c0, mask=case.grid.roi_mask))
        if attenuation is not None and case.ground_truth.attenuation_np_per_m is not None:
            metrics.update(
                compute_image_metrics(
                    attenuation,
                    np.asarray(case.ground_truth.attenuation_np_per_m, dtype=float),
                    mask=case.grid.roi_mask,
                    prefix="attenuation_",
                )
            )
        return ReconstructionResult(
            algorithm=self.name,
            case_id=case.case_id,
            sound_speed_mps=sound_speed,
            attenuation_np_per_m=attenuation,
            metrics=metrics,
            artifacts={
                "external_result_path": str(result_path),
                "external_dataset_path": external.get("dataset_path") or "",
            },
        )


def read_kwave_fwi_result(path: str | Path) -> dict[str, Any]:
    h5py = _h5py()
    result_path = Path(path).expanduser().resolve()
    with h5py.File(result_path, "r") as handle:
        sound_speed = _require_dataset(handle, "VEL_ESTIM")
        attenuation = _read_dataset(handle, "ATTEN_ESTIM")
        losses = _read_vector(handle, "LOSS_ITER")
        return {
            "sound_speed_mps": np.asarray(sound_speed, dtype=float),
            "attenuation_np_per_m": np.asarray(attenuation, dtype=float) if attenuation is not None else None,
            "losses": losses.tolist(),
            "iterations": int(losses.size),
            "initial_loss": float(losses[0]) if losses.size else None,
            "final_loss": float(losses[-1]) if losses.size else None,
            "loss_decreased": bool(losses[-1] < losses[0]) if losses.size >= 2 else None,
            "psnr_value": _read_scalar(handle, "psnr_value"),
            "ssim_value": _read_scalar(handle, "ssim_value"),
            "dataset_path": _read_matlab_string(handle, "datasetPath"),
        }


def _run_external_pipeline(case: USCTCase, config: AlgorithmConfig, result_path: Path) -> ReconstructionResult:
    dataset_path = _configured_path(config, "dataset_path")
    if dataset_path is None:
        dataset_from_case = case.metadata.get("source_path")
        dataset_path = Path(str(dataset_from_case)).expanduser() if dataset_from_case else None
    if dataset_path is None:
        return ReconstructionResult(
            algorithm=KWaveFWIAdapterAlgorithm.name,
            case_id=case.case_id,
            status=ResultStatus.SKIPPED,
            failure_reason="run_external requires parameters.dataset_path or case.metadata.source_path",
        )

    usct_kwave_root = Path(
        os.path.expandvars(str(config.parameters.get("usct_kwave_root", os.environ.get("USCT_KWAVE_ROOT", "/home/wudalong/USCT_kwave"))))
    ).expanduser()
    python_bin = str(config.parameters.get("python_bin", sys.executable))
    module = str(config.parameters.get("pipeline_module", "openbreastus_diffusion.kwave_dps.run_full_pipeline"))
    extra_args = [str(value) for value in config.parameters.get("pipeline_args", [])]
    command = [
        python_bin,
        "-m",
        module,
        "--skip-siminfo",
        "--skip-rf",
        "--skip-assemble",
        "--dataset-path",
        str(dataset_path),
        "--result-path",
        str(result_path),
        *extra_args,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(usct_kwave_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log_path = _configured_path(config, "external_log_path")
    timeout_s = float(config.parameters.get("timeout_s", 3600.0))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_target = subprocess.PIPE
    stderr_target = subprocess.STDOUT
    log_handle = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8")
            log_handle.write("$ " + " ".join(command) + "\n\n")
            log_handle.flush()
            stdout_target = log_handle
            stderr_target = subprocess.STDOUT
        proc = subprocess.run(
            command,
            cwd=usct_kwave_root,
            env=env,
            text=True,
            stdout=stdout_target,
            stderr=stderr_target,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        return ReconstructionResult(
            algorithm=KWaveFWIAdapterAlgorithm.name,
            case_id=case.case_id,
            status=ResultStatus.FAILED,
            failure_reason=f"external k-Wave FWI launch failed: {type(exc).__name__}: {exc}",
        )
    finally:
        if log_handle is not None:
            log_handle.close()

    if proc.returncode != 0:
        detail = ""
        if log_path is not None:
            detail = f"; log={log_path}"
        elif proc.stdout:
            detail = f"; stdout={proc.stdout[-1000:]}"
        return ReconstructionResult(
            algorithm=KWaveFWIAdapterAlgorithm.name,
            case_id=case.case_id,
            status=ResultStatus.FAILED,
            failure_reason=f"external k-Wave FWI returned {proc.returncode}{detail}",
        )
    return ReconstructionResult(algorithm=KWaveFWIAdapterAlgorithm.name, case_id=case.case_id)


def _configured_path(config: AlgorithmConfig, key: str) -> Path | None:
    value = config.parameters.get(key)
    if not value:
        return None
    return Path(os.path.expandvars(str(value))).expanduser()


def _resize_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    if array.shape == shape:
        return array
    y_idx = np.linspace(0, array.shape[0] - 1, shape[0])
    x_idx = np.linspace(0, array.shape[1] - 1, shape[1])
    y0 = np.floor(y_idx).astype(int)
    x0 = np.floor(x_idx).astype(int)
    y1 = np.clip(y0 + 1, 0, array.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, array.shape[1] - 1)
    wy = (y_idx - y0)[:, None]
    wx = (x_idx - x0)[None, :]
    top = (1.0 - wx) * array[np.ix_(y0, x0)] + wx * array[np.ix_(y0, x1)]
    bottom = (1.0 - wx) * array[np.ix_(y1, x0)] + wx * array[np.ix_(y1, x1)]
    return (1.0 - wy) * top + wy * bottom


def _require_dataset(handle: Any, name: str) -> np.ndarray:
    value = _read_dataset(handle, name)
    if value is None:
        raise KeyError(f"missing required dataset {name}")
    return value


def _read_dataset(handle: Any, name: str) -> np.ndarray | None:
    if name not in handle:
        return None
    return np.asarray(handle[name][()])


def _read_vector(handle: Any, name: str) -> np.ndarray:
    value = _read_dataset(handle, name)
    if value is None:
        return np.asarray([], dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)


def _read_scalar(handle: Any, name: str) -> float | None:
    value = _read_vector(handle, name)
    if value.size == 0:
        return None
    return float(value[0])


def _read_matlab_string(handle: Any, name: str) -> str | None:
    if name not in handle:
        return None
    values = np.asarray(handle[name][()]).reshape(-1)
    try:
        return "".join(chr(int(value)) for value in values if int(value) != 0)
    except Exception:
        return None


def _h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required to read MATLAB v7.3 FWI result files") from exc
    return h5py
