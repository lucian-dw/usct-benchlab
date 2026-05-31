"""Shared optional-MATLAB adapter helpers."""

from __future__ import annotations

import json
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
    return run_matlab_placeholder(
        algorithm=algorithm,
        case=case,
        config=config,
        missing_config_reason=missing_config_reason,
        missing_entrypoint_prefix=missing_entrypoint_prefix,
        configured_message=configured_message,
        log_name=log_name,
        unimplemented_reason=unimplemented_reason,
    )


def run_matlab_placeholder(
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
