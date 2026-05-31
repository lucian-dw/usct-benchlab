"""Optional refraction-corrected Gauss-Newton MATLAB adapter."""

from __future__ import annotations

from usctbench.algorithms.adapters._matlab_optional import run_optional_matlab_backend
from usctbench.algorithms.adapters._travel_time import run_iterative_travel_time_solver
from usctbench.schema import AlgorithmConfig, ReconstructionResult, USCTCase


class BentRayGNAdapter:
    """Callable adapter for refraction-corrected travel-time tomography.

    The default v0.1 backend is a project-native regularized travel-time
    inversion that preserves the same I/O and metrics as the external adapter.
    Set `backend: matlab` with `external_root` and `entrypoint` to run the
    public-package MATLAB bridge through the standard adapter MAT contract.
    """

    name = "bent_ray_gn"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        return run_optional_matlab_backend(
            algorithm=self.name,
            case=case,
            config=config,
            native_runner=lambda: _run_python_bent_ray(self.name, case, config),
            missing_config_reason="MATLAB refraction GN adapter requires parameters.external_root and parameters.entrypoint",
            missing_entrypoint_prefix="MATLAB refraction GN entrypoint not found",
            configured_message="USCT bent_ray_gn adapter configured",
            log_name="bent_ray_gn_matlab.log",
            unimplemented_reason="External MATLAB entrypoint completed but did not write usctbench_output_mat; configure it to write the standard adapter output file.",
        )


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
