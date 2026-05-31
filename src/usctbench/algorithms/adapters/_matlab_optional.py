"""Shared optional-MATLAB adapter helpers."""

from __future__ import annotations

from pathlib import Path

from usctbench.adapters.matlab import MatlabAdapter, MatlabUnavailable
from usctbench.schema import AlgorithmConfig, ReconstructionResult, ResultStatus, USCTCase


def requests_matlab_backend(config: AlgorithmConfig) -> bool:
    params = config.parameters
    backend = str(params.get("backend", "")).strip().lower()
    return backend == "matlab" or any(params.get(key) for key in ("matlab_bin", "external_root", "entrypoint"))


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
    try:
        adapter = MatlabAdapter.from_config(
            matlab_bin=params.get("matlab_bin"),
            work_dir=params.get("work_dir", "."),
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

    code = f"addpath(genpath('{_matlab_string(external_root_path)}')); disp('{_matlab_string(configured_message)}');"
    try:
        log_path = adapter.run_batch(code, log_name=log_name, timeout_s=int(params.get("timeout_s", 300)))
    except MatlabUnavailable as exc:
        return skipped_matlab_adapter(algorithm, case, str(exc))

    return ReconstructionResult(
        algorithm=algorithm,
        case_id=case.case_id,
        status=ResultStatus.SKIPPED,
        failure_reason=unimplemented_reason,
        artifacts={"matlab_log": str(log_path), "external_entrypoint": str(entry_path)},
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
