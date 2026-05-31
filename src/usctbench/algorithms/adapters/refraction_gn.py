"""Optional refraction-corrected Gauss-Newton MATLAB adapter."""

from __future__ import annotations

from pathlib import Path

from usctbench.adapters.matlab import MatlabAdapter, MatlabUnavailable
from usctbench.algorithms.adapters._travel_time import run_iterative_travel_time_solver
from usctbench.algorithms.ray._common import run_with_failure_capture
from usctbench.schema import AlgorithmConfig, ReconstructionResult, ResultStatus, USCTCase


class BentRayGNAdapter:
    """Callable placeholder for established MATLAB bent-ray GN code.

    The default v0.1 backend is a project-native regularized travel-time
    inversion that preserves the same I/O and metrics as the external adapter.
    Set `backend: matlab` or provide MATLAB-specific parameters to exercise the
    dependency-checking external path.
    """

    name = "bent_ray_gn"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        if not _requests_matlab_backend(config):
            return run_with_failure_capture(self.name, case, lambda: _run_python_bent_ray(self.name, case, config))
        return _run_matlab_placeholder(self.name, case, config)


def _run_python_bent_ray(algorithm: str, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
    return run_iterative_travel_time_solver(
        algorithm=algorithm,
        case=case,
        config=config,
        method_family="refraction_corrected_travel_time_gn_surrogate",
        default_outer_iterations=4,
        default_inner_iterations=16,
        default_regularization="laplacian",
        default_regularization_lambda=3.0e-5,
        default_smooth_sigma=0.6,
        extra_metrics={
            "refraction_correction_enabled": True,
            "external_reference": "rehmanali1994/refractionCorrectedUSCT.github.io",
        },
    )


def _requests_matlab_backend(config: AlgorithmConfig) -> bool:
    params = config.parameters
    backend = str(params.get("backend", "")).strip().lower()
    return backend == "matlab" or any(params.get(key) for key in ("matlab_bin", "external_root", "entrypoint"))


def _run_matlab_placeholder(algorithm: str, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
    params = config.parameters
    try:
        adapter = MatlabAdapter.from_config(
            matlab_bin=params.get("matlab_bin"),
            work_dir=params.get("work_dir", "."),
        )
    except MatlabUnavailable as exc:
        return _skipped(algorithm, case, str(exc))

    entrypoint = params.get("entrypoint")
    external_root = params.get("external_root")
    if not entrypoint or not external_root:
        return _skipped(
            algorithm,
            case,
            "MATLAB refraction GN adapter requires parameters.external_root and parameters.entrypoint",
        )

    entry_path = Path(external_root).expanduser() / str(entrypoint)
    if not entry_path.exists():
        return _skipped(algorithm, case, f"MATLAB refraction GN entrypoint not found: {entry_path}")

    # Execution is deliberately opt-in because external packages have
    # different input conventions. The generated log still proves the
    # adapter can call MATLAB when a user supplies an entrypoint.
    code = f"addpath(genpath('{Path(external_root).expanduser()}')); disp('USCT bent_ray_gn adapter configured');"
    try:
        log_path = adapter.run_batch(code, log_name="bent_ray_gn_matlab.log", timeout_s=int(params.get("timeout_s", 300)))
    except MatlabUnavailable as exc:
        return _skipped(algorithm, case, str(exc))

    return ReconstructionResult(
        algorithm=algorithm,
        case_id=case.case_id,
        status=ResultStatus.SKIPPED,
        failure_reason="External MATLAB entrypoint is configured, but data marshaling is not implemented yet.",
        artifacts={"matlab_log": str(log_path), "external_entrypoint": str(entry_path)},
    )


def _skipped(algorithm: str, case: USCTCase, reason: str) -> ReconstructionResult:
    return ReconstructionResult(
        algorithm=algorithm,
        case_id=case.case_id,
        status=ResultStatus.SKIPPED,
        failure_reason=reason,
        metrics={"adapter_dependency_available": False},
        artifacts={"adapter_status": "skipped", "skip_reason": reason},
    )
