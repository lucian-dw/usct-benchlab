"""Shared optional-MATLAB adapter helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from usctbench.adapters.matlab import MatlabAdapter, MatlabUnavailable, write_usct_case_mat
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
    entry_path = external_root_path / str(entrypoint)
    if not entry_path.exists():
        return skipped_matlab_adapter(algorithm, case, f"{missing_entrypoint_prefix}: {entry_path}")

    input_path = _adapter_input_path(params, adapter.work_dir, algorithm, case)
    write_usct_case_mat(case, input_path)
    code = (
        f"addpath(genpath('{_matlab_string(external_root_path)}')); "
        f"usctbench_input_mat='{_matlab_string(input_path)}'; "
        f"usctbench_output_dir='{_matlab_string(adapter.work_dir)}'; "
        f"disp('{_matlab_string(configured_message)}');"
    )
    try:
        log_path = adapter.run_batch(code, log_name=log_name, timeout_s=int(params.get("timeout_s", 300)))
    except MatlabUnavailable as exc:
        return skipped_matlab_adapter(algorithm, case, str(exc))

    return ReconstructionResult(
        algorithm=algorithm,
        case_id=case.case_id,
        status=ResultStatus.SKIPPED,
        failure_reason=unimplemented_reason,
        artifacts={"adapter_input_mat": str(input_path), "matlab_log": str(log_path), "external_entrypoint": str(entry_path)},
    )


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


def _safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in str(value)).strip("._") or "case"
