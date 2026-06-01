"""Straight-ray CGLS sound-speed reconstruction."""

from __future__ import annotations

import numpy as np

from usctbench.algorithms.ray._common import (
    cgls_solve,
    masked_norm,
    reference_sound_speed,
    residual_metrics,
    ray_weights,
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
            weights = ray_weights(case, projector, mask)
            iterations = int(config.parameters.get("iterations", 30))
            regularization = str(config.parameters.get("regularization", "identity"))
            lambda_value = float(config.parameters.get("lambda", config.parameters.get("regularization_lambda", 0.0)))
            damping = float(config.parameters.get("damping", lambda_value**2))
            c0 = reference_sound_speed(case, config)
            initial_norm = masked_norm(target, mask, weights)
            delta_slowness, residual_norms = cgls_solve(
                projector,
                target,
                mask,
                iterations=iterations,
                damping=damping,
                regularization=regularization,
                weights=weights,
            )
            sound_speed = slowness_to_sound_speed(delta_slowness, c0, speed_bounds(config))
            final_norm = residual_norms[-1] if residual_norms else initial_norm
            metrics = {
                **residual_metrics(initial_norm, final_norm),
                "iterations": len(residual_norms) - 1,
                "regularization": regularization,
                "regularization_lambda": lambda_value,
                "regularization_lambda_squared": damping,
                "residual_curve": residual_norms,
                "ray_weight_mean": float(np.mean(weights[mask])) if np.any(mask) else 0.0,
                "ray_weight_nonzero_fraction": float(np.mean(weights[mask] > 0.0)) if np.any(mask) else 0.0,
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
