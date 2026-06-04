"""rWave/ray-Born-style adapter baseline."""

from __future__ import annotations

from usctbench.algorithms.bent_ray import run_iterative_travel_time_solver
from usctbench.algorithms.ray import run_with_failure_capture
from usctbench.core.registry import register_algorithm
from usctbench.core.schema import AlgorithmConfig, ReconstructionResult, USCTCase


class RWaveAdapter:
    """Adapter-style rWave baseline using the common travel-time case contract."""

    name = "rwave_adapter"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        def _run() -> ReconstructionResult:
            return run_iterative_travel_time_solver(
                algorithm=self.name,
                case=case,
                config=config,
                method_family="rwave_travel_time_adapter",
                default_outer_iterations=3,
                default_inner_iterations=20,
                default_regularization="laplacian",
                default_regularization_lambda=1.0e-5,
                default_smooth_sigma=0.35,
                extra_metrics={
                    "ray_born_inspired": True,
                    "adapter_style": True,
                    "full_ray_born_solver": False,
                    "backend": "adapter_style_travel_time_baseline",
                    "surrogate_travel_time_backend": True,
                    "external_reference": "ray-Born/rWave literature",
                },
            )

        return run_with_failure_capture(self.name, case, _run)


def register_rwave_algorithm(*, replace: bool = False) -> None:
    register_algorithm(
        "rwave_adapter",
        RWaveAdapter,
        description="rWave/ray-Born-style adapter baseline.",
        tags=("adapter", "rwave"),
        replace=replace,
    )


__all__ = ["RWaveAdapter", "register_rwave_algorithm"]
