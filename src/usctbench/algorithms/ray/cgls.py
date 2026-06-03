"""Straight-ray CGLS sound-speed reconstruction."""

from __future__ import annotations

import numpy as np

from usctbench.algorithms.ray._common import (
    cgls_solve,
    coverage_preconditioner,
    configured_ray_weights,
    huber_irls_cgls_solve,
    image_diagnostic_metrics,
    masked_norm,
    reference_sound_speed,
    residual_metrics,
    ray_weight_metrics,
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
            weights = configured_ray_weights(case, projector, mask, config)
            iterations = int(config.parameters.get("iterations", 30))
            regularization = str(config.parameters.get("regularization", "identity"))
            lambda_value = float(config.parameters.get("lambda", config.parameters.get("regularization_lambda", 0.0)))
            damping = float(config.parameters.get("damping", lambda_value**2))
            roi_update_only = bool(config.parameters.get("roi_update_only", False))
            roi_laplacian = bool(config.parameters.get("roi_laplacian", config.parameters.get("roi_aware_laplacian", False)))
            use_coverage_preconditioning = bool(config.parameters.get("coverage_preconditioning", False))
            robust_loss = str(config.parameters.get("robust_loss", "none")).lower()
            c0 = reference_sound_speed(case, config)
            initial_norm = masked_norm(target, mask, weights)
            roi_mask = np.asarray(case.grid.roi_mask, dtype=bool) if case.grid.roi_mask is not None and (roi_update_only or roi_laplacian) else None
            preconditioner = None
            coverage = projector.adjoint(np.asarray(mask, dtype=float) * np.clip(weights, 0.0, 1.0))
            if use_coverage_preconditioning:
                preconditioner, coverage = coverage_preconditioner(
                    projector,
                    mask,
                    weights,
                    roi_mask=np.asarray(case.grid.roi_mask, dtype=bool) if case.grid.roi_mask is not None and roi_update_only else None,
                    eps=float(config.parameters.get("coverage_preconditioner_eps", 1.0e-12)),
                    max_scale=float(config.parameters.get("coverage_preconditioner_max_scale", 10.0)),
                    normalize=bool(config.parameters.get("coverage_preconditioner_normalize", True)),
                )
            if robust_loss in {"huber", "irls", "huber_irls"}:
                delta_slowness, residual_norms = huber_irls_cgls_solve(
                    projector,
                    target,
                    mask,
                    iterations=iterations,
                    damping=damping,
                    regularization=regularization,
                    weights=weights,
                    preconditioner=preconditioner,
                    roi_mask=roi_mask,
                    huber_delta=float(config.parameters.get("huber_delta_s", 5.0e-7)),
                    irls_iterations=int(config.parameters.get("irls_iterations", 3)),
                )
            else:
                delta_slowness, residual_norms = cgls_solve(
                    projector,
                    target,
                    mask,
                    iterations=iterations,
                    damping=damping,
                    regularization=regularization,
                    weights=weights,
                    preconditioner=preconditioner,
                    roi_mask=roi_mask,
                )
            if roi_update_only and case.grid.roi_mask is not None:
                delta_slowness = np.where(np.asarray(case.grid.roi_mask, dtype=bool), delta_slowness, 0.0)
            sound_speed = slowness_to_sound_speed(delta_slowness, c0, speed_bounds(config))
            final_norm = residual_norms[-1] if residual_norms else initial_norm
            metrics = {
                **residual_metrics(initial_norm, final_norm),
                "iterations": len(residual_norms) - 1,
                "regularization": regularization,
                "regularization_lambda": lambda_value,
                "regularization_lambda_squared": damping,
                "coverage_preconditioning": use_coverage_preconditioning,
                "roi_laplacian": roi_laplacian,
                "roi_update_only": roi_update_only,
                "robust_loss": robust_loss,
                "huber_delta_s": float(config.parameters.get("huber_delta_s", 5.0e-7)),
                "irls_iterations": int(config.parameters.get("irls_iterations", 0 if robust_loss not in {"huber", "irls", "huber_irls"} else 3)),
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
                metrics.update(
                    image_diagnostic_metrics(
                        sound_speed,
                        np.asarray(case.ground_truth.sound_speed_mps, dtype=float),
                        roi_mask=case.grid.roi_mask,
                        coverage=coverage,
                        boundary_band_pixels=int(config.parameters.get("boundary_band_pixels", 4)),
                    )
                )
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                sound_speed_mps=sound_speed,
                metrics=metrics,
            )

        return run_with_failure_capture(self.name, case, _run)
