"""Straight-ray SART sound-speed reconstruction."""

from __future__ import annotations

import numpy as np

from usctbench.algorithms.ray._common import (
    reference_sound_speed,
    run_with_failure_capture,
    sart_solve,
    slowness_to_sound_speed,
    speed_bounds,
    target_delta_tof,
)
from usctbench.algorithms.ray.straight_projector import StraightRayProjector
from usctbench.metrics.image import compute_image_metrics
from usctbench.schema import AlgorithmConfig, ReconstructionResult, USCTCase


class StraightRaySARTAlgorithm:
    name = "straight_sart"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        def _run() -> ReconstructionResult:
            projector = StraightRayProjector.from_case(case)
            target, mask = target_delta_tof(case, projector)
            iterations = int(config.parameters.get("iterations", 10))
            relaxation = float(config.parameters.get("relaxation", 0.5))
            c0 = reference_sound_speed(case, config)
            delta_slowness, residual_norms = sart_solve(
                projector,
                target,
                mask,
                iterations=iterations,
                relaxation=relaxation,
            )
            sound_speed = slowness_to_sound_speed(delta_slowness, c0, speed_bounds(config))
            metrics = {
                "data_residual_norm": residual_norms[-1] if residual_norms else 0.0,
                "initial_data_residual_norm": residual_norms[0] if residual_norms else 0.0,
                "iterations": iterations,
            }
            if case.ground_truth.sound_speed_mps is not None:
                metrics.update(
                    compute_image_metrics(
                        sound_speed,
                        np.asarray(case.ground_truth.sound_speed_mps, dtype=float),
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

