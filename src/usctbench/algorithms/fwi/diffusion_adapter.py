"""Adapter for external diffusion-prior k-Wave/FWI DPS results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from usctbench.algorithms.fwi.adapter import (
    _as_bool,
    _configured_path,
    _expand_text,
    _resize_to_shape,
)
from usctbench.core.schema import (
    AlgorithmConfig,
    ReconstructionResult,
    ResultStatus,
    USCTCase,
)
from usctbench.metrics import (
    compute_baseline_improvement_metrics,
    compute_image_metrics,
)

_DPS_IMAGE_FIELDS = (
    "VEL_DPS_PHYS",
    "VEL_DPS_VIEW",
    "VEL_FINAL_PHYS",
    "VEL_FINAL_VIEW",
    "VEL_INIT_VIEW",
)
_DPS_GT_FIELDS = ("GT_VIEW", "C_INTERP")
_DEFAULT_FREQS_MHZ = [
    0.3,
    0.3,
    0.3,
    0.35,
    0.35,
    0.35,
    0.4,
    0.4,
    0.4,
    0.45,
    0.45,
    0.45,
]


class DiffusionKWaveFWIAdapterAlgorithm:
    """Load or launch an external diffusion-prior k-Wave/FWI result.

    The normal benchmark path is loader-only (`run_external: false`): the
    adapter reads an existing DPS `.mat` plus optional JSON summary, then
    returns a standard :class:`ReconstructionResult`. Set `run_external: true`
    only on machines with the external USCT-kwave environment configured.
    """

    name = "diffusion_fwi_kwave_adapter"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        result_path = _configured_path(config, "result_path") or _configured_path(
            config, "output_path"
        )
        if result_path is None:
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.SKIPPED,
                failure_reason=(
                    "diffusion_fwi_kwave_adapter requires "
                    "parameters.result_path or parameters.output_path"
                ),
            )

        run_external = _as_bool(config.parameters.get("run_external", False))
        if run_external:
            launched = _run_external_dps_pipeline(case, config, result_path)
            if launched.status != ResultStatus.SUCCESS:
                return launched

        if not result_path.exists():
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.SKIPPED,
                failure_reason=f"diffusion FWI result file not found: {result_path}",
            )

        summary_path = _summary_path_for(config, result_path)
        selected_field = str(config.parameters.get("selected_field", "auto"))
        try:
            external = read_diffusion_kwave_fwi_result(
                result_path, summary_path=summary_path, selected_field=selected_field
            )
        except Exception as exc:
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                status=ResultStatus.FAILED,
                failure_reason=(
                    "failed to read diffusion k-Wave FWI result: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        sound_speed = _resize_to_shape(external["sound_speed_mps"], case.grid.shape)
        metrics = _result_metrics(config, external, result_path, summary_path)
        _add_truth_metrics(metrics, sound_speed, external, case, config)

        artifacts = {
            "external_result_path": str(result_path),
            "external_summary_path": str(summary_path) if summary_path else "",
            "external_dataset_path": str(metrics.get("dataset_path") or ""),
            "external_checkpoint": str(metrics.get("checkpoint") or ""),
        }
        if external.get("png_path"):
            artifacts["external_preview_path"] = str(external["png_path"])

        return ReconstructionResult(
            algorithm=self.name,
            case_id=case.case_id,
            sound_speed_mps=sound_speed,
            metrics=metrics,
            artifacts=artifacts,
        )


def read_diffusion_kwave_fwi_result(
    path: str | Path,
    *,
    summary_path: str | Path | None = None,
    selected_field: str = "auto",
) -> dict[str, Any]:
    """Read a DPS result MAT file and optional JSON summary."""

    result_path = Path(path).expanduser().resolve()
    fields = _read_mat_fields(result_path)
    field_name = _select_dps_field(fields, selected_field)
    summary = _read_summary(summary_path, fields)
    ground_truth = _first_array(fields, _DPS_GT_FIELDS)
    initial = _array_from_fields(fields, "VEL_INIT_VIEW")
    return {
        "sound_speed_mps": np.asarray(fields[field_name], dtype=float),
        "selected_field": field_name,
        "ground_truth_sound_speed_mps": ground_truth,
        "initial_sound_speed_mps": initial,
        "summary": summary,
        "dataset_path": _summary_value(summary, "dataset_path"),
        "checkpoint": _summary_value(summary, "checkpoint"),
        "png_path": _summary_value(summary, "png_path"),
    }


def _read_mat_fields(path: Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat

        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
        return {
            key: _normalize_mat_value(value)
            for key, value in payload.items()
            if not key.startswith("__")
        }
    except NotImplementedError:
        return _read_hdf5_mat_fields(path)
    except ValueError as exc:
        if "Unknown mat file type" not in str(exc):
            raise
        return _read_hdf5_mat_fields(path)


def _read_hdf5_mat_fields(path: Path) -> dict[str, Any]:
    import h5py

    fields: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        for key in handle:
            fields[key] = np.asarray(handle[key][()])
    return fields


def _normalize_mat_value(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return value.item()
    return value


def _select_dps_field(fields: dict[str, Any], selected_field: str) -> str:
    if selected_field and selected_field != "auto":
        if selected_field not in fields:
            raise KeyError(f"selected DPS field not found: {selected_field}")
        return selected_field
    for field_name in _DPS_IMAGE_FIELDS:
        if _array_from_fields(fields, field_name) is not None:
            return field_name
    available = ", ".join(sorted(fields))
    raise KeyError(
        "missing DPS velocity field. Expected one of "
        f"{', '.join(_DPS_IMAGE_FIELDS)}; available fields: {available}"
    )


def _array_from_fields(fields: dict[str, Any], key: str) -> np.ndarray | None:
    value = fields.get(key)
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim < 2:
        return None
    return np.asarray(array, dtype=float)


def _first_array(fields: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        value = _array_from_fields(fields, key)
        if value is not None:
            return value
    return None


def _read_summary(
    summary_path: str | Path | None, fields: dict[str, Any]
) -> dict[str, Any]:
    if summary_path:
        path = Path(summary_path).expanduser()
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    history = fields.get("history_json")
    if history is None:
        return {}
    text = _mat_string(history)
    if not text:
        return {}
    return json.loads(text)


def _mat_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.shape == ():
        return _mat_string(array.item())
    if array.dtype.kind in {"U", "S"}:
        return "".join(array.astype(str).reshape(-1).tolist())
    if array.dtype == object and array.size == 1:
        return _mat_string(array.reshape(-1)[0])
    return ""


def _summary_path_for(config: AlgorithmConfig, result_path: Path) -> Path | None:
    configured = _configured_path(config, "summary_path")
    if configured is not None:
        return configured
    default = result_path.with_suffix(".json")
    return default if default.exists() else None


def _result_metrics(
    config: AlgorithmConfig,
    external: dict[str, Any],
    result_path: Path,
    summary_path: Path | None,
) -> dict[str, Any]:
    summary = external.get("summary", {})
    reference_hparams = summary.get("reference_hparams", {})
    selected_metrics = summary.get("selected_metrics", {})
    final_metrics = summary.get("final_metrics", {})
    freqs_mhz = _config_or_summary_sequence(
        config, summary, "freqs_mhz", default=_DEFAULT_FREQS_MHZ
    )
    steps = int(_config_or_summary_scalar(config, summary, "steps", default=12))
    output_selection = _config_or_summary_scalar(
        config, summary, "output_selection", default="final"
    )
    selected_step = _selected_step(summary, steps, output_selection)
    metrics: dict[str, Any] = {
        "external_result_loaded": True,
        "external_execution_mode": (
            "run_external"
            if _as_bool(config.parameters.get("run_external", False))
            else "loader_only"
        ),
        "external_result_path": str(result_path),
        "summary_path": str(summary_path) if summary_path else "",
        "checkpoint": str(
            _config_or_summary_scalar(config, summary, "checkpoint", default="")
            or external.get("checkpoint")
            or ""
        ),
        "dataset_path": str(
            _config_or_summary_scalar(config, summary, "dataset_path", default="")
            or external.get("dataset_path")
            or ""
        ),
        "score_reg_t": float(
            _config_or_summary_scalar(
                config, reference_hparams, "score_reg_t", default=0.10
            )
        ),
        "score_reg_lambda": float(
            _config_or_summary_scalar(
                config, reference_hparams, "score_reg_lambda", default=0.1
            )
        ),
        "steps": steps,
        "freqs_mhz": [float(value) for value in freqs_mhz],
        "array_mode": str(config.parameters.get("array_mode", "sparse64")),
        "selected_step": selected_step,
        "selected_field": external["selected_field"],
        "prior_mode": str(
            _config_or_summary_scalar(
                config, reference_hparams, "prior_mode", default="score_reg"
            )
        ),
        "prior_strength": float(
            _config_or_summary_scalar(
                config, reference_hparams, "prior_strength", default=1.0
            )
        ),
        "prior_mask_mode": str(
            _config_or_summary_scalar(
                config, reference_hparams, "prior_mask_mode", default="none"
            )
        ),
        "physics_position": str(
            _config_or_summary_scalar(
                config, reference_hparams, "physics_position", default="pre"
            )
        ),
        "physics_inner_steps": int(
            _config_or_summary_scalar(
                config, reference_hparams, "physics_inner_steps", default=1
            )
        ),
        "eta": float(
            _config_or_summary_scalar(config, reference_hparams, "eta", default=0.1)
        ),
        "guidance_gain": float(
            _config_or_summary_scalar(
                config, reference_hparams, "guidance_gain", default=1.15
            )
        ),
        "gradient_mode": str(
            _config_or_summary_scalar(
                config, reference_hparams, "gradient_mode", default="slowness_precond"
            )
        ),
        "step_strategy": str(
            _config_or_summary_scalar(
                config, reference_hparams, "step_strategy", default="line_search"
            )
        ),
        "mask_mode": str(config.parameters.get("mask_mode", "support_alpha")),
        "output_selection": str(output_selection),
        "final_prior_update": bool(
            _config_or_summary_scalar(
                config, reference_hparams, "final_prior_update", default=False
            )
        ),
    }
    metrics.update(_flatten_metric_dict(selected_metrics, "dps_selected_"))
    metrics.update(_flatten_metric_dict(final_metrics, "dps_final_"))
    return metrics


def _add_truth_metrics(
    metrics: dict[str, Any],
    sound_speed: np.ndarray,
    external: dict[str, Any],
    case: USCTCase,
    config: AlgorithmConfig,
) -> None:
    baseline = float(
        config.parameters.get(
            "baseline_sound_speed_mps",
            case.metadata.get("reference_sound_speed_mps", 1500.0),
        )
    )
    truth = case.ground_truth.sound_speed_mps
    prefix = ""
    if truth is None and external.get("ground_truth_sound_speed_mps") is not None:
        truth = external["ground_truth_sound_speed_mps"]
        prefix = "external_gt_"
    if truth is None:
        return
    target = _resize_to_shape(np.asarray(truth, dtype=float), case.grid.shape)
    metrics.update(
        compute_image_metrics(
            sound_speed, target, mask=case.grid.roi_mask, prefix=prefix
        )
    )
    metrics.update(
        compute_baseline_improvement_metrics(
            sound_speed, target, baseline, mask=case.grid.roi_mask, prefix=prefix
        )
    )
    if external.get("initial_sound_speed_mps") is not None:
        init = _resize_to_shape(external["initial_sound_speed_mps"], case.grid.shape)
        metrics.update(
            compute_image_metrics(init, target, mask=case.grid.roi_mask, prefix="init_")
        )


def _flatten_metric_dict(payload: Any, prefix: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            for child_key, child_value in _flatten_metric_dict(
                value, f"{prefix}{key}_"
            ).items():
                flattened[child_key] = child_value
        elif _json_metric_value(value) is not None:
            flattened[f"{prefix}{key}"] = _json_metric_value(value)
    return flattened


def _json_metric_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_metric_value(item) for item in value]
    return None


def _summary_value(summary: dict[str, Any], key: str) -> Any:
    if key in summary:
        return summary[key]
    reference = summary.get("reference_hparams")
    if isinstance(reference, dict) and key in reference:
        return reference[key]
    return None


def _config_or_summary_scalar(
    config: AlgorithmConfig, summary: dict[str, Any], key: str, *, default: Any
) -> Any:
    if key in config.parameters and config.parameters[key] not in (None, ""):
        return config.parameters[key]
    return summary.get(key, default)


def _config_or_summary_sequence(
    config: AlgorithmConfig,
    summary: dict[str, Any],
    key: str,
    *,
    default: list[float],
) -> list[Any]:
    value = config.parameters.get(key)
    if value not in (None, ""):
        return value if isinstance(value, list) else [value]
    if key in summary:
        summary_value = summary[key]
        return summary_value if isinstance(summary_value, list) else [summary_value]
    if "freqs_hz" in summary:
        return [float(value) / 1.0e6 for value in summary["freqs_hz"]]
    return default


def _selected_step(summary: dict[str, Any], steps: int, output_selection: Any) -> int:
    if "selected_step" in summary:
        return int(summary["selected_step"])
    if str(output_selection) == "best_object_ssim":
        value = summary.get("best_object_ssim_step_index")
        if value is not None:
            return int(value) + 1
    return int(steps)


def _run_external_dps_pipeline(
    case: USCTCase, config: AlgorithmConfig, result_path: Path
) -> ReconstructionResult:
    build = _build_external_dps_commands(case, config, result_path)
    if build["error"]:
        return ReconstructionResult(
            algorithm=DiffusionKWaveFWIAdapterAlgorithm.name,
            case_id=case.case_id,
            status=ResultStatus.SKIPPED,
            failure_reason=str(build["error"]),
        )

    root = Path(str(build["root"])).expanduser()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    log_path = _configured_path(config, "external_log_path")
    timeout_s = float(config.parameters.get("timeout_s", 21600.0))
    result_path.parent.mkdir(parents=True, exist_ok=True)

    for index, command in enumerate(build["commands"], start=1):
        log_handle = None
        stdout_target: Any = subprocess.PIPE
        stderr_target: Any = subprocess.STDOUT
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w" if index == 1 else "a", encoding="utf-8")
            log_handle.write(f"# diffusion FWI external step {index}\n")
            log_handle.write("$ " + " ".join(command) + "\n\n")
            log_handle.flush()
            stdout_target = log_handle
            stderr_target = subprocess.STDOUT
        try:
            proc = subprocess.run(
                command,
                cwd=root,
                env=env,
                text=True,
                stdout=stdout_target,
                stderr=stderr_target,
                timeout=timeout_s,
                check=False,
            )
        except Exception as exc:
            return ReconstructionResult(
                algorithm=DiffusionKWaveFWIAdapterAlgorithm.name,
                case_id=case.case_id,
                status=ResultStatus.FAILED,
                failure_reason=(
                    "external diffusion FWI step "
                    f"{index} failed: {type(exc).__name__}: {exc}"
                ),
            )
        finally:
            if log_handle is not None:
                log_handle.close()

        if proc.returncode != 0:
            detail = f"; log={log_path}" if log_path else ""
            if not detail and proc.stdout:
                detail = f"; stdout={proc.stdout[-1000:]}"
            return ReconstructionResult(
                algorithm=DiffusionKWaveFWIAdapterAlgorithm.name,
                case_id=case.case_id,
                status=ResultStatus.FAILED,
                failure_reason=(
                    f"external diffusion FWI step {index} returned "
                    f"{proc.returncode}{detail}"
                ),
            )

    return ReconstructionResult(
        algorithm=DiffusionKWaveFWIAdapterAlgorithm.name, case_id=case.case_id
    )


def _build_external_dps_commands(
    case: USCTCase, config: AlgorithmConfig, result_path: Path
) -> dict[str, Any]:
    root_text = _expand_text(
        config.parameters.get("usct_kwave_root", os.environ.get("USCT_KWAVE_ROOT", ""))
    ).strip()
    if not root_text or "$" in root_text:
        return {
            "root": "",
            "commands": [],
            "error": (
                "run_external requires parameters.usct_kwave_root " "or USCT_KWAVE_ROOT"
            ),
        }
    dataset_path = _configured_path(config, "dataset_path")
    if dataset_path is None and case.metadata.get("source_path"):
        dataset_path = Path(_expand_text(case.metadata["source_path"])).expanduser()
    if dataset_path is None:
        return {
            "root": root_text,
            "commands": [],
            "error": (
                "run_external requires parameters.dataset_path "
                "or case.metadata.source_path"
            ),
        }
    checkpoint = _expand_text(config.parameters.get("checkpoint", "")).strip()
    if not checkpoint or "$" in checkpoint:
        return {
            "root": root_text,
            "commands": [],
            "error": "run_external requires parameters.checkpoint",
        }

    python_bin = _configured_python_bin(config)
    summary_path = _configured_path(config, "summary_path") or result_path.with_suffix(
        ".json"
    )
    png_path = _configured_path(config, "png_path")
    commands: list[list[str]] = []
    init_mat = _configured_path(config, "init_mat")
    if _warm_start_builder(config) == "bulk_support" and init_mat is None:
        run_output_dir = (
            _configured_path(config, "_run_output_dir") or result_path.parent
        )
        init_mat = (
            run_output_dir / "warm_start" / f"{case.case_id}_bulk_support_init.mat"
        )
        commands.append(
            _bulk_support_init_command(
                python_bin=python_bin,
                dataset_path=dataset_path,
                init_mat=init_mat,
                config=config,
            )
        )

    command = [
        python_bin,
        "-m",
        str(
            config.parameters.get(
                "pipeline_module", "openbreastus_diffusion.kwave_dps.run_dps_kwave"
            )
        ),
        "--dataset-path",
        str(dataset_path),
        "--checkpoint",
        checkpoint,
        "--output-path",
        str(result_path),
        "--summary-path",
        str(summary_path),
        "--device",
        str(config.parameters.get("device", "cuda:0")),
        "--seed",
        str(config.parameters.get("seed", 1234)),
        "--steps",
        str(config.parameters.get("steps", 12)),
        "--crop-source-size",
        str(config.parameters.get("crop_source_size", 300)),
        "--source-size",
        str(config.parameters.get("source_size", 480)),
        "--sampler-mode",
        str(config.parameters.get("sampler_mode", "reference")),
        "--prior-mode",
        str(config.parameters.get("prior_mode", "score_reg")),
        "--t-start",
        str(config.parameters.get("t_start", 0.99)),
        "--t-end",
        str(config.parameters.get("t_end", 0.01)),
        "--eta",
        str(config.parameters.get("eta", 0.1)),
        "--guidance-gain",
        str(config.parameters.get("guidance_gain", 1.15)),
        "--prior-strength",
        str(config.parameters.get("prior_strength", 1.0)),
        "--prior-start-step",
        str(config.parameters.get("prior_start_step", 1)),
        "--prior-every",
        str(config.parameters.get("prior_every", 1)),
        "--prior-stop-step",
        str(config.parameters.get("prior_stop_step", 0)),
        "--prior-mask-mode",
        str(config.parameters.get("prior_mask_mode", "none")),
        "--score-reg-t",
        str(config.parameters.get("score_reg_t", 0.10)),
        "--score-reg-lambda",
        str(config.parameters.get("score_reg_lambda", 0.1)),
        "--physics-position",
        str(config.parameters.get("physics_position", "pre")),
        "--physics-inner-steps",
        str(config.parameters.get("physics_inner_steps", 1)),
        "--output-selection",
        str(config.parameters.get("output_selection", "final")),
        "--gradient-mode",
        str(config.parameters.get("gradient_mode", "slowness_precond")),
        "--step-strategy",
        str(config.parameters.get("step_strategy", "line_search")),
        "--tx-stride",
        str(config.parameters.get("tx_stride", 1)),
        "--mask-mode",
        str(config.parameters.get("mask_mode", "support_alpha")),
        "--sign-conv",
        str(config.parameters.get("sign_conv", -1)),
        "--perc-outliers",
        str(config.parameters.get("perc_outliers", 0.99)),
        "--target-max-abs",
        str(config.parameters.get("target_max_abs", 1.0)),
    ]
    if init_mat is not None:
        command.extend(["--init-mat", str(init_mat)])
    if png_path is not None:
        command.extend(["--png-path", str(png_path)])
    command.append("--freqs-mhz")
    command.extend(str(value) for value in _freqs_mhz(config))
    command.extend(_bool_flag(config, "final_prior_update", "--final-prior-update"))
    command.extend(_bool_flag(config, "support_guidance", "--support-guidance"))
    command.extend(_bool_flag(config, "normalize_observed", "--normalize-observed"))
    if _as_bool(config.parameters.get("start_matlab", False)):
        command.append("--start-matlab")
    commands.append(command)
    return {"root": root_text, "commands": commands, "error": None}


def _bulk_support_init_command(
    *,
    python_bin: str,
    dataset_path: Path,
    init_mat: Path,
    config: AlgorithmConfig,
) -> list[str]:
    prefix = init_mat.with_suffix("")
    summary_path = init_mat.with_suffix(".json")
    command = [
        python_bin,
        "-m",
        str(
            config.parameters.get(
                "warm_start_module",
                "openbreastus_diffusion.kwave_dps.make_bulk_support_init",
            )
        ),
        "--dataset-path",
        str(dataset_path),
        "--output-path",
        str(init_mat),
        "--summary-path",
        str(summary_path),
        "--diagnostic-prefix",
        str(prefix),
        "--init-mode",
        "bulk_support",
        "--background-speed",
        str(config.parameters.get("background_speed", 1500)),
        "--arrival-picker",
        str(config.parameters.get("arrival_picker", "hilbert")),
        "--min-confidence",
        str(config.parameters.get("min_confidence", 1.5)),
        "--residual-clip-us",
        str(config.parameters.get("residual_clip_us", 5)),
        "--residual-percentile",
        str(config.parameters.get("residual_percentile", 95)),
        "--max-rays",
        str(config.parameters.get("max_rays", 8000)),
        "--smooth-sigma-mm",
        str(config.parameters.get("smooth_sigma_mm", 10)),
        "--support-mode",
        str(config.parameters.get("support_mode", "backprojection")),
        "--support-percentile",
        str(config.parameters.get("support_percentile", 45)),
        "--support-sigma-mm",
        str(config.parameters.get("support_sigma_mm", 8)),
        "--support-alpha-sigma-mm",
        str(config.parameters.get("support_alpha_sigma_mm", 8)),
        "--support-dilate-mm",
        str(config.parameters.get("support_dilate_mm", 4)),
        "--bulk-update-scale",
        str(config.parameters.get("bulk_update_scale", 1.0)),
        "--bulk-min-path-mm",
        str(config.parameters.get("bulk_min_path_mm", 20)),
        "--bulk-stat",
        str(config.parameters.get("bulk_stat", "median")),
        "--velocity-bounds",
    ]
    command.extend(
        str(value)
        for value in config.parameters.get("velocity_bounds", [1408.692, 1595.1279])
    )
    if _as_bool(config.parameters.get("compare_gt", True)):
        command.append("--compare-gt")
    return command


def _warm_start_builder(config: AlgorithmConfig) -> str:
    value = str(config.parameters.get("warm_start_builder", "bulk_support"))
    normalized = value.strip().lower().replace("-", "_")
    return "bulk_support" if normalized in {"bulk_support", "bulk"} else normalized


def _configured_python_bin(config: AlgorithmConfig) -> str:
    value = _expand_text(config.parameters.get("python_bin", "")).strip()
    return value if value and "$" not in value else sys.executable


def _freqs_mhz(config: AlgorithmConfig) -> list[Any]:
    value = config.parameters.get("freqs_mhz", _DEFAULT_FREQS_MHZ)
    return value if isinstance(value, list) else [value]


def _bool_flag(config: AlgorithmConfig, key: str, flag: str) -> list[str]:
    value = config.parameters.get(key)
    if value is None:
        value = key != "final_prior_update"
    if _as_bool(value):
        return [flag]
    return [f"--no-{flag[2:]}"]
