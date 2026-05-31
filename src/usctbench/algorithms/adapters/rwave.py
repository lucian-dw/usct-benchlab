"""Optional ray-Born/r-Wave MATLAB adapter."""

from __future__ import annotations

from usctbench.algorithms.adapters._matlab_optional import run_optional_matlab_backend
from usctbench.algorithms.adapters._travel_time import run_iterative_travel_time_solver
from usctbench.schema import AlgorithmConfig, ReconstructionResult, USCTCase


class RWaveAdapter:
    """Callable adapter for ray-Born/r-Wave-style USCT reconstruction.

    The default v0.1 backend is a native regularized ray-Born surrogate over
    the benchmark travel-time features. The MATLAB path remains available for
    explicit external integration checks.
    """

    name = "rwave_adapter"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        return run_optional_matlab_backend(
            algorithm=self.name,
            case=case,
            config=config,
            native_runner=lambda: _run_python_rwave(self.name, case, config),
            missing_config_reason="r-Wave adapter requires parameters.external_root and parameters.entrypoint",
            missing_entrypoint_prefix="r-Wave entrypoint not found",
            configured_message="USCT rwave_adapter configured",
            log_name="rwave_matlab.log",
            unimplemented_reason="External r-Wave entrypoint is configured, but data marshaling is not implemented yet.",
        )


def _run_python_rwave(algorithm: str, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
    return run_iterative_travel_time_solver(
        algorithm=algorithm,
        case=case,
        config=config,
        method_family="ray_born_rwave_surrogate",
        default_outer_iterations=3,
        default_inner_iterations=20,
        default_regularization="laplacian",
        default_regularization_lambda=1.0e-5,
        default_smooth_sigma=0.35,
        extra_metrics={
            "ray_born_linearization": True,
            "external_reference": "Ash1362/ray-based-quantitative-ultrasound-tomography",
        },
    )
