"""Bent-ray travel-time adapter baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from usctbench.algorithms.ray import (
    StraightRayProjector,
    _gaussian_smooth,
    apply_mask,
    cgls_solve,
    configured_ray_weights,
    huber_irls_cgls_solve,
    image_diagnostic_metrics,
    masked_norm,
    ray_weight_metrics,
    reference_sound_speed,
    residual_metrics,
    run_with_failure_capture,
    slowness_to_sound_speed,
    speed_bounds,
    target_delta_tof,
)
from usctbench.core.registry import register_algorithm
from usctbench.core.schema import AlgorithmConfig, ReconstructionResult, USCTCase
from usctbench.metrics import (
    compute_baseline_improvement_metrics,
    compute_image_metrics,
)


def run_iterative_travel_time_solver(
    *,
    algorithm: str,
    case: USCTCase,
    config: AlgorithmConfig,
    method_family: str,
    default_outer_iterations: int,
    default_inner_iterations: int,
    default_regularization: str,
    default_regularization_lambda: float,
    default_smooth_sigma: float,
    extra_metrics: dict[str, Any] | None = None,
) -> ReconstructionResult:
    """Run a small regularized travel-time inversion using project-native I/O.

    This is the v0.1 smoke/quality path for algorithms whose full reference
    implementation lives in external MATLAB repositories. It keeps the same
    travel-time data contract and records method-family metadata so results are
    auditable without vendoring third-party code.
    """

    base_projector = StraightRayProjector.from_case(case)
    projector = base_projector
    target, mask = target_delta_tof(case, projector)
    weights = configured_ray_weights(case, projector, mask, config)
    c0 = reference_sound_speed(case, config)
    bounds = speed_bounds(config)
    outer_iterations = int(
        config.parameters.get("outer_iterations", default_outer_iterations)
    )
    inner_iterations = int(
        config.parameters.get("inner_iterations", default_inner_iterations)
    )
    step_length = float(config.parameters.get("step_length", 1.0))
    regularization = str(
        config.parameters.get("regularization", default_regularization)
    )
    lambda_value = float(
        config.parameters.get("regularization_lambda", default_regularization_lambda)
    )
    damping = float(config.parameters.get("damping", lambda_value**2))
    smooth_sigma = float(config.parameters.get("smooth_sigma", default_smooth_sigma))
    roi_update_only = bool(config.parameters.get("roi_update_only", False))
    line_search = bool(config.parameters.get("line_search", False))
    roi_laplacian = bool(config.parameters.get("roi_laplacian", False))
    robust_loss = str(config.parameters.get("robust_loss", "none")).lower()
    delta_slowness = np.zeros(case.grid.shape, dtype=float)
    initial_norm = masked_norm(target, mask, weights)
    residual_curve = [initial_norm]
    update_norms: list[float] = []
    roi_mask = (
        np.asarray(case.grid.roi_mask, dtype=bool)
        if case.grid.roi_mask is not None and (roi_update_only or roi_laplacian)
        else None
    )

    for _ in range(max(1, outer_iterations)):
        residual = target - projector.forward(delta_slowness)
        if robust_loss in {"huber", "irls", "huber_irls"}:
            update, _inner_curve = huber_irls_cgls_solve(
                projector,
                residual,
                mask,
                iterations=inner_iterations,
                damping=damping,
                regularization=regularization,
                weights=weights,
                roi_mask=roi_mask,
                huber_delta=float(config.parameters.get("huber_delta_s", 5.0e-7)),
                irls_iterations=int(config.parameters.get("irls_iterations", 2)),
            )
        else:
            update, _inner_curve = cgls_solve(
                projector,
                residual,
                mask,
                iterations=inner_iterations,
                damping=damping,
                regularization=regularization,
                weights=weights,
                roi_mask=roi_mask,
            )
        candidate, candidate_norm, accepted_step = _line_search_update(
            projector,
            target,
            mask,
            weights,
            delta_slowness,
            update,
            step_length=step_length,
            smooth_sigma=smooth_sigma,
            roi_mask=(
                np.asarray(case.grid.roi_mask, dtype=bool)
                if roi_update_only and case.grid.roi_mask is not None
                else None
            ),
            enabled=line_search,
        )
        delta_slowness = candidate
        update_norms.append(float(np.linalg.norm(update)))
        residual_curve.append(candidate_norm)
        if line_search and accepted_step < step_length:
            update_norms[-1] = float(np.linalg.norm(accepted_step * update))

    sound_speed = slowness_to_sound_speed(delta_slowness, c0, bounds)
    final_norm = residual_curve[-1] if residual_curve else initial_norm
    metrics: dict[str, Any] = {
        **residual_metrics(initial_norm, final_norm),
        "iterations": outer_iterations,
        "outer_iterations": outer_iterations,
        "inner_iterations": inner_iterations,
        "step_length": step_length,
        "regularization": regularization,
        "regularization_lambda": lambda_value,
        "regularization_lambda_squared": damping,
        "smooth_sigma": smooth_sigma,
        "roi_update_only": roi_update_only,
        "roi_laplacian": roi_laplacian,
        "line_search": line_search,
        "true_bent_ray": False,
        "uses_true_bent_rays": False,
        "robust_loss": robust_loss,
        "method_family": method_family,
        "residual_curve": residual_curve,
        "update_norm_curve": update_norms,
        **ray_weight_metrics(weights, mask, config),
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    if case.ground_truth.sound_speed_mps is not None:
        truth = np.asarray(case.ground_truth.sound_speed_mps, dtype=float)
        metrics.update(
            compute_image_metrics(sound_speed, truth, mask=case.grid.roi_mask)
        )
        metrics.update(
            compute_baseline_improvement_metrics(
                sound_speed, truth, c0, mask=case.grid.roi_mask
            )
        )
        coverage = projector.adjoint(
            np.asarray(mask, dtype=float) * np.clip(weights, 0.0, 1.0)
        )
        metrics.update(
            image_diagnostic_metrics(
                sound_speed,
                truth,
                roi_mask=case.grid.roi_mask,
                coverage=coverage,
                boundary_band_pixels=int(
                    config.parameters.get("boundary_band_pixels", 4)
                ),
            )
        )

    return ReconstructionResult(
        algorithm=algorithm,
        case_id=case.case_id,
        sound_speed_mps=sound_speed,
        metrics=metrics,
    )


def _line_search_update(
    projector: Any,
    target: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
    current: np.ndarray,
    update: np.ndarray,
    *,
    step_length: float,
    smooth_sigma: float,
    roi_mask: np.ndarray | None,
    enabled: bool,
) -> tuple[np.ndarray, float, float]:
    current_norm = float(
        np.linalg.norm(
            apply_mask(target - projector.forward(current), mask, weights)[mask]
        )
    )
    steps = [float(step_length)]
    if enabled:
        steps.extend(float(step_length) * (0.5**idx) for idx in range(1, 8))
    best = current
    best_norm = current_norm
    best_step = 0.0
    for step in steps:
        candidate = current + step * update
        if smooth_sigma > 0.0:
            candidate = _gaussian_smooth(candidate, smooth_sigma)
        if roi_mask is not None:
            candidate = np.where(roi_mask, candidate, 0.0)
        norm = float(
            np.linalg.norm(
                apply_mask(target - projector.forward(candidate), mask, weights)[mask]
            )
        )
        if (not enabled) or norm <= best_norm:
            return candidate, norm, step
        if norm < best_norm:
            best = candidate
            best_norm = norm
            best_step = step
    return best, best_norm, best_step


class BentRayGNAdapter:
    """Regularized bent-ray-style travel-time baseline."""

    name = "bent_ray_gn"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        def _run() -> ReconstructionResult:
            return run_iterative_travel_time_solver(
                algorithm=self.name,
                case=case,
                config=config,
                method_family="bent_ray_travel_time_baseline",
                default_outer_iterations=4,
                default_inner_iterations=16,
                default_regularization="laplacian",
                default_regularization_lambda=3.0e-5,
                default_smooth_sigma=0.6,
                extra_metrics={
                    "surrogate_travel_time_backend": True,
                    "full_external_eikonal_solver": False,
                    "v0_1_backend": "regularized_travel_time_baseline",
                    "external_reference": "refraction-corrected USCT literature",
                },
            )

        return run_with_failure_capture(self.name, case, _run)


def register_bent_ray_algorithm(*, replace: bool = False) -> None:
    register_algorithm(
        "bent_ray_gn",
        BentRayGNAdapter,
        description="Regularized bent-ray travel-time baseline.",
        tags=("travel-time", "refraction"),
        replace=replace,
    )


__all__ = [
    "BentRayGNAdapter",
    "register_bent_ray_algorithm",
    "run_iterative_travel_time_solver",
]
