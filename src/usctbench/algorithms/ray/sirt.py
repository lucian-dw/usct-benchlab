"""Straight-ray SIRT sound-speed reconstruction."""

from __future__ import annotations

import numpy as np

from usctbench.algorithms.ray._common import (
    masked_norm,
    reference_sound_speed,
    residual_metrics,
    run_with_failure_capture,
    sirt_solve,
    slowness_to_sound_speed,
    speed_bounds,
    target_delta_tof,
)
from usctbench.algorithms.ray.straight_projector import StraightRayProjector
from usctbench.metrics.image import compute_baseline_improvement_metrics, compute_image_metrics
from usctbench.schema import AlgorithmConfig, ReconstructionResult, USCTCase


class StraightRaySIRTAlgorithm:
    name = "straight_sirt"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        def _run() -> ReconstructionResult:
            projector = StraightRayProjector.from_case(case)
            target, mask = target_delta_tof(case, projector)
            iterations = int(config.parameters.get("iterations", 50))
            relaxation = float(config.parameters.get("relaxation", 0.8))
            c0 = reference_sound_speed(case, config)
            initial_norm = masked_norm(target, mask)
            delta_slowness, residual_norms = sirt_solve(
                projector,
                target,
                mask,
                iterations=iterations,
                relaxation=relaxation,
            )
            sound_speed = slowness_to_sound_speed(delta_slowness, c0, speed_bounds(config))
            final_norm = residual_norms[-1] if residual_norms else initial_norm
            metrics = {**residual_metrics(initial_norm, final_norm), "iterations": iterations}
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
