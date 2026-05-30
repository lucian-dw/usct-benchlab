"""Straight-ray CGLS sound-speed reconstruction."""

from __future__ import annotations

import numpy as np

from usctbench.algorithms.ray._common import (
    cgls_solve,
    reference_sound_speed,
    run_with_failure_capture,
    slowness_to_sound_speed,
    speed_bounds,
    target_delta_tof,
)
from usctbench.algorithms.ray.straight_projector import StraightRayProjector
from usctbench.metrics.image import compute_baseline_improvement_metrics, compute_image_metrics
from usctbench.schema import AlgorithmConfig, ReconstructionResult, USCTCase


class StraightRayCGLSAlgorithm:
    name = "straight_cgls"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        def _run() -> ReconstructionResult:
            projector = StraightRayProjector.from_case(case)
            target, mask = target_delta_tof(case, projector)
            iterations = int(config.parameters.get("iterations", 30))
            damping = float(config.parameters.get("damping", 0.0))
            c0 = reference_sound_speed(case, config)
            delta_slowness, residual_norms = cgls_solve(projector, target, mask, iterations=iterations, damping=damping)
            sound_speed = slowness_to_sound_speed(delta_slowness, c0, speed_bounds(config))
            metrics = {
                "data_residual_norm": residual_norms[-1],
                "initial_data_residual_norm": residual_norms[0],
                "iterations": len(residual_norms) - 1,
            }
            if case.ground_truth.sound_speed_mps is not None:
                metrics.update(
                    compute_image_metrics(
                        sound_speed,
                        np.asarray(case.ground_truth.sound_speed_mps, dtype=float),
                        mask=case.grid.roi_mask,
                    )
                )
                metrics.update(
                    compute_baseline_improvement_metrics(
                        sound_speed,
                        np.asarray(case.ground_truth.sound_speed_mps, dtype=float),
                        c0,
                        mask=case.grid.roi_mask,
                    )
                )
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                sound_speed_mps=sound_speed,
                metrics=metrics,
            )

        return run_with_failure_capture(self.name, case, _run)
