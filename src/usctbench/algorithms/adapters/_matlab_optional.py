"""Shared optional-MATLAB adapter helpers."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

import numpy as np

from usctbench.adapters.matlab import (
    MatlabAdapter,
    MatlabUnavailable,
    read_matlab_adapter_result,
    write_matlab_adapter_contract,
    write_usct_case_mat,
)
from usctbench.algorithms.ray._common import (
    apply_mask,
    masked_norm,
    reference_sound_speed,
    residual_metrics,
    target_delta_tof,
)
from usctbench.algorithms.ray.straight_projector import StraightRayProjector
from usctbench.metrics.image import compute_baseline_improvement_metrics, compute_image_metrics
from usctbench.schema import AlgorithmConfig, ReconstructionResult, ResultStatus, USCTCase


def requests_matlab_backend(config: AlgorithmConfig) -> bool:
    params = config.parameters
    backend = str(params.get("backend", "")).strip().lower()
    return backend == "matlab" or any(params.get(key) for key in ("matlab_bin", "external_root", "entrypoint"))


def run_optional_matlab_backend(
    *,
    algorithm: str,
    case: USCTCase,
    config: AlgorithmConfig,
    native_runner: Callable[[], ReconstructionResult],
    missing_config_reason: str,
    missing_entrypoint_prefix: str,
    configured_message: str,
    log_name: str,
    unimplemented_reason: str,
) -> ReconstructionResult:
    if not requests_matlab_backend(config):
        from usctbench.algorithms.ray._common import run_with_failure_capture

        return run_with_failure_capture(algorithm, case, native_runner)
    return run_matlab_backend(
        algorithm=algorithm,
        case=case,
        config=config,
        missing_config_reason=missing_config_reason,
        missing_entrypoint_prefix=missing_entrypoint_prefix,
        configured_message=configured_message,
        log_name=log_name,
        unimplemented_reason=unimplemented_reason,
    )


def run_matlab_backend(
    *,
    algorithm: str,
    case: USCTCase,
    config: AlgorithmConfig,
    missing_config_reason: str,
    missing_entrypoint_prefix: str,
    configured_message: str,
    log_name: str,
    unimplemented_reason: str,
) -> ReconstructionResult:
    params = config.parameters
    work_dir = params.get("work_dir") or params.get("_run_output_dir") or "."
    try:
        adapter = MatlabAdapter.from_config(
            matlab_bin=params.get("matlab_bin"),
            work_dir=work_dir,
        )
    except MatlabUnavailable as exc:
        return skipped_matlab_adapter(algorithm, case, str(exc))

    entrypoint = params.get("entrypoint")
    external_root = params.get("external_root")
    if not entrypoint or not external_root:
        return skipped_matlab_adapter(algorithm, case, missing_config_reason)

    external_root_path = Path(external_root).expanduser()
    entry_path = _entrypoint_path(external_root_path, str(entrypoint))
    if not entry_path.exists():
        return skipped_matlab_adapter(algorithm, case, f"{missing_entrypoint_prefix}: {entry_path}")

    input_path = _adapter_input_path(params, adapter.work_dir, algorithm, case)
    output_path = _adapter_output_path(params, adapter.work_dir, algorithm, case)
    write_usct_case_mat(case, input_path)
    contract_dir = write_matlab_adapter_contract(adapter.work_dir)
    code = (
        f"addpath('{_matlab_string(contract_dir)}'); "
        f"addpath(genpath('{_matlab_string(external_root_path)}')); "
        f"usctbench_input_mat='{_matlab_string(input_path)}'; "
        f"usctbench_output_mat='{_matlab_string(output_path)}'; "
        f"usctbench_output_dir='{_matlab_string(adapter.work_dir)}'; "
        f"usctbench_parameters_json='{_matlab_string(_parameters_json(params))}'; "
        f"disp('{_matlab_string(configured_message)}'); "
        f"{_entrypoint_call(params, entry_path)}"
    )
    try:
        log_path = adapter.run_batch(code, log_name=log_name, timeout_s=int(params.get("timeout_s", 300)))
    except MatlabUnavailable as exc:
        return skipped_matlab_adapter(algorithm, case, str(exc))

    artifacts = {
        "adapter_input_mat": str(input_path),
        "adapter_output_mat": str(output_path),
        "adapter_contract_dir": str(contract_dir),
        "matlab_log": str(log_path),
        "external_entrypoint": str(entry_path),
    }
    if not output_path.exists():
        return ReconstructionResult(
            algorithm=algorithm,
            case_id=case.case_id,
            status=ResultStatus.SKIPPED,
            failure_reason=unimplemented_reason,
            artifacts=artifacts,
            metrics={"adapter_dependency_available": True, "external_adapter_output_loaded": False},
        )
    try:
        result = read_matlab_adapter_result(output_path, algorithm=algorithm, case_id=case.case_id)
    except Exception as exc:
        return ReconstructionResult(
            algorithm=algorithm,
            case_id=case.case_id,
            status=ResultStatus.FAILED,
            failure_reason=f"failed to read MATLAB adapter output: {type(exc).__name__}: {exc}",
            artifacts=artifacts,
            metrics={"adapter_dependency_available": True, "external_adapter_output_loaded": False},
        )
    result.artifacts.update(artifacts)
    result.metrics.setdefault("adapter_dependency_available", True)
    result.metrics["external_adapter_output_loaded"] = True
    _augment_matlab_log_metrics(result, log_path)
    _augment_external_result_metrics(result, case, config)
    return result


def skipped_matlab_adapter(algorithm: str, case: USCTCase, reason: str) -> ReconstructionResult:
    return ReconstructionResult(
        algorithm=algorithm,
        case_id=case.case_id,
        status=ResultStatus.SKIPPED,
        failure_reason=reason,
        metrics={"adapter_dependency_available": False},
        artifacts={"adapter_status": "skipped", "skip_reason": reason},
    )


def _matlab_string(value: object) -> str:
    return str(value).replace("'", "''")


def _adapter_input_path(params: dict, work_dir: Path, algorithm: str, case: USCTCase) -> Path:
    configured = params.get("adapter_input_path") or params.get("input_mat_path")
    if configured:
        return Path(str(configured)).expanduser()
    return work_dir / f"{_safe_stem(algorithm)}_{_safe_stem(case.case_id)}_input.mat"


def _adapter_output_path(params: dict, work_dir: Path, algorithm: str, case: USCTCase) -> Path:
    configured = params.get("adapter_output_path") or params.get("output_mat_path")
    if configured:
        return Path(str(configured)).expanduser()
    return work_dir / f"{_safe_stem(algorithm)}_{_safe_stem(case.case_id)}_output.mat"


def _entrypoint_call(params: dict, entry_path: Path) -> str:
    configured = params.get("entrypoint_call")
    if configured:
        return str(configured)
    return f"run('{_matlab_string(entry_path)}');"


def _entrypoint_path(external_root: Path, entrypoint: str) -> Path:
    path = Path(entrypoint).expanduser()
    if path.is_absolute():
        return path
    return external_root / path


def _safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value)).strip("._") or "case"


def _parameters_json(params: dict) -> str:
    public_params = {str(key): value for key, value in params.items() if not str(key).startswith("_")}
    return json.dumps(public_params, default=str, sort_keys=True)


def _augment_external_result_metrics(result: ReconstructionResult, case: USCTCase, config: AlgorithmConfig) -> None:
    """Add project-standard metrics to external MATLAB outputs when possible."""

    if result.sound_speed_mps is None:
        return
    try:
        sound_speed = np.asarray(result.sound_speed_mps, dtype=float)
        c0 = reference_sound_speed(case, config)
        if case.ground_truth.sound_speed_mps is not None and sound_speed.shape == case.grid.shape:
            truth = np.asarray(case.ground_truth.sound_speed_mps, dtype=float)
            for key, value in compute_image_metrics(sound_speed, truth, mask=case.grid.roi_mask).items():
                result.metrics.setdefault(key, value)
            for key, value in compute_baseline_improvement_metrics(sound_speed, truth, c0, mask=case.grid.roi_mask).items():
                result.metrics.setdefault(key, value)
        if sound_speed.shape == case.grid.shape:
            projector = StraightRayProjector.from_case(case)
            target, mask = target_delta_tof(case, projector)
            delta_slowness = (1.0 / np.clip(sound_speed, 1.0e-12, np.inf)) - (1.0 / c0)
            residual = apply_mask(target - projector.forward(delta_slowness), mask)
            initial_norm = masked_norm(target, mask)
            final_norm = masked_norm(residual, mask)
            for key, value in residual_metrics(initial_norm, final_norm).items():
                result.metrics.setdefault(key, value)
    except Exception as exc:
        result.metrics.setdefault("external_metric_augmentation_error", f"{type(exc).__name__}: {exc}")


def _augment_matlab_log_metrics(result: ReconstructionResult, log_path: Path) -> None:
    try:
        diagnostics = _parse_matlab_log_diagnostics(log_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # pragma: no cover - diagnostic only.
        result.metrics.setdefault("matlab_log_diagnostic_error", f"{type(exc).__name__}: {exc}")
        return
    result.metrics.update(diagnostics)


def _parse_matlab_log_diagnostics(text: str) -> dict[str, object]:
    """Extract lightweight numerical diagnostics from external MATLAB logs."""

    bad_linkings = _extract_numbers_after(text, r"The number of bad linkings:\s*([+-]?\d+(?:\.\d+)?)")
    sign_jacobian = _extract_numbers_after(
        text,
        r"The number of the rays for which the sign of the Jacobian is changed are:\s*([+-]?\d+(?:\.\d+)?)",
    )
    objectives = _extract_numbers_after(text, r"The objective function is:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
    relative_errors = _extract_numbers_after(text, r"The relative error is:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)%")
    frequency_levels = _extract_numbers_after(text, r"The frequency level \(linearised subproblem\) is:\s*([+-]?\d+(?:\.\d+)?)")
    finite_objectives = [value for value in objectives if math.isfinite(value)]
    nan_count = len(re.findall(r"(?<![A-Za-z])NaN(?![A-Za-z])", text, flags=re.IGNORECASE))
    inf_count = len(re.findall(r"(?<![A-Za-z])Inf(?![A-Za-z])", text, flags=re.IGNORECASE))

    diagnostics: dict[str, object] = {
        "matlab_log_bad_linking_reports": len(bad_linkings),
        "matlab_log_bad_linkings_total": float(np.sum(bad_linkings)) if bad_linkings else 0.0,
        "matlab_log_bad_linkings_max": float(np.max(bad_linkings)) if bad_linkings else 0.0,
        "matlab_log_sign_jacobian_reports": len(sign_jacobian),
        "matlab_log_sign_jacobian_changed_total": float(np.sum(sign_jacobian)) if sign_jacobian else 0.0,
        "matlab_log_sign_jacobian_changed_max": float(np.max(sign_jacobian)) if sign_jacobian else 0.0,
        "matlab_log_objective_reports": len(finite_objectives),
        "matlab_log_nan_token_count": nan_count,
        "matlab_log_inf_token_count": inf_count,
        "matlab_log_frequency_level_reports": len(frequency_levels),
        "matlab_log_frequency_level_max": float(np.max(frequency_levels)) if frequency_levels else 0.0,
        "matlab_log_relative_error_reports": len(relative_errors),
        "matlab_log_relative_error_percent_max": float(np.max(relative_errors)) if relative_errors else float("nan"),
        "matlab_log_relative_error_percent_last": float(relative_errors[-1]) if relative_errors else float("nan"),
    }
    if finite_objectives:
        initial = float(finite_objectives[0])
        final = float(finite_objectives[-1])
        diagnostics.update(
            {
                "matlab_log_objective_initial": initial,
                "matlab_log_objective_final": final,
                "matlab_log_objective_min": float(np.min(finite_objectives)),
                "matlab_log_objective_max": float(np.max(finite_objectives)),
                "matlab_log_objective_increased": final > initial,
            }
        )
    else:
        diagnostics.update(
            {
                "matlab_log_objective_initial": float("nan"),
                "matlab_log_objective_final": float("nan"),
                "matlab_log_objective_min": float("nan"),
                "matlab_log_objective_max": float("nan"),
                "matlab_log_objective_increased": False,
            }
        )
    diagnostics["matlab_log_likely_failure_mode"] = _classify_matlab_log_diagnostics(diagnostics)
    return diagnostics


def _extract_numbers_after(text: str, pattern: str) -> list[float]:
    return [float(match) for match in re.findall(pattern, text, flags=re.IGNORECASE)]


def _classify_matlab_log_diagnostics(metrics: dict[str, object]) -> str:
    nan_count = int(metrics.get("matlab_log_nan_token_count", 0))
    bad_linkings = float(metrics.get("matlab_log_bad_linkings_total", 0.0))
    sign_total = float(metrics.get("matlab_log_sign_jacobian_changed_total", 0.0))
    objective_increased = bool(metrics.get("matlab_log_objective_increased", False))
    if nan_count > 0:
        return "nan_or_nonfinite_gradient"
    if bad_linkings > 0:
        return "ray_linking_or_caustic_failure"
    if sign_total > 0 and objective_increased:
        return "sign_jacobian_or_update_direction"
    if objective_increased:
        return "source_frequency_or_update_model_mismatch"
    if sign_total > 0:
        return "sign_jacobian_present_but_objective_stable"
    return "no_log_failure_signal"
