"""Straight-ray SART sound-speed reconstruction."""

from __future__ import annotations

import numpy as np

from usctbench.algorithms.ray._common import (
    configured_ray_weights,
    masked_norm,
    reference_sound_speed,
    residual_metrics,
    ray_weight_metrics,
    run_with_failure_capture,
    sart_solve,
    slowness_to_sound_speed,
    speed_bounds,
    target_delta_tof,
)
from usctbench.algorithms.ray.straight_projector import StraightRayProjector
from usctbench.metrics.image import compute_baseline_improvement_metrics, compute_image_metrics
from usctbench.schema import AlgorithmConfig, ReconstructionResult, USCTCase


class StraightRaySARTAlgorithm:
    name = "straight_sart"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        def _run() -> ReconstructionResult:
            projector = StraightRayProjector.from_case(case)
            target, mask = target_delta_tof(case, projector)
            weights = configured_ray_weights(case, projector, mask, config)
            iterations = int(config.parameters.get("iterations", 10))
            relaxation = float(config.parameters.get("relaxation", 0.2))
            subsets = int(config.parameters.get("subsets", 8))
            smooth_sigma = float(config.parameters.get("smooth_sigma", 0.0))
            smooth_every = int(config.parameters.get("smooth_every", 0))
            roi_update_only = bool(config.parameters.get("roi_update_only", False))
            c0 = reference_sound_speed(case, config)
            initial_norm = masked_norm(target, mask, weights)
            delta_slowness, residual_norms = sart_solve(
                projector,
                target,
                mask,
                iterations=iterations,
                relaxation=relaxation,
                subsets=subsets,
                smooth_sigma=smooth_sigma,
                smooth_every=smooth_every,
                roi_mask=case.grid.roi_mask if roi_update_only else None,
                weights=weights,
            )
            sound_speed = slowness_to_sound_speed(delta_slowness, c0, speed_bounds(config))
            final_norm = residual_norms[-1] if residual_norms else initial_norm
            metrics = {
                **residual_metrics(initial_norm, final_norm),
                "iterations": iterations,
                "relaxation": relaxation,
                "subsets": subsets,
                "smooth_sigma": smooth_sigma,
                "smooth_every": smooth_every,
                "roi_update_only": roi_update_only,
                "residual_curve": residual_norms,
                **ray_weight_metrics(weights, mask, config),
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
