"""Optional refraction-corrected Gauss-Newton MATLAB adapter."""

from __future__ import annotations

from usctbench.algorithms.adapters._matlab_optional import requests_matlab_backend, run_matlab_placeholder
from usctbench.algorithms.adapters._travel_time import run_iterative_travel_time_solver
from usctbench.algorithms.ray._common import run_with_failure_capture
from usctbench.schema import AlgorithmConfig, ReconstructionResult, USCTCase


class BentRayGNAdapter:
    """Callable placeholder for established MATLAB bent-ray GN code.

    The default v0.1 backend is a project-native regularized travel-time
    inversion that preserves the same I/O and metrics as the external adapter.
    Set `backend: matlab` or provide MATLAB-specific parameters to exercise the
    dependency-checking external path.
    """

    name = "bent_ray_gn"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        if not requests_matlab_backend(config):
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


def _run_matlab_placeholder(algorithm: str, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
    return run_matlab_placeholder(
        algorithm=algorithm,
        case=case,
        config=config,
        missing_config_reason="MATLAB refraction GN adapter requires parameters.external_root and parameters.entrypoint",
        missing_entrypoint_prefix="MATLAB refraction GN entrypoint not found",
        configured_message="USCT bent_ray_gn adapter configured",
        log_name="bent_ray_gn_matlab.log",
        unimplemented_reason="External MATLAB entrypoint is configured, but data marshaling is not implemented yet.",
    )
