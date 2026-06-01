"""Shared travel-time inversion helpers for optional ray-method adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from usctbench.algorithms.ray._common import (
    _gaussian_smooth,
    apply_mask,
    cgls_solve,
    masked_norm,
    ray_weights,
    reference_sound_speed,
    residual_metrics,
    slowness_to_sound_speed,
    speed_bounds,
    target_delta_tof,
)
from usctbench.algorithms.ray.straight_projector import StraightRayProjector
from usctbench.metrics.image import compute_baseline_improvement_metrics, compute_image_metrics
from usctbench.schema import AlgorithmConfig, ReconstructionResult, USCTCase


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

    projector = StraightRayProjector.from_case(case)
    target, mask = target_delta_tof(case, projector)
    weights = ray_weights(case, projector, mask)
    c0 = reference_sound_speed(case, config)
    bounds = speed_bounds(config)
    outer_iterations = int(config.parameters.get("outer_iterations", default_outer_iterations))
    inner_iterations = int(config.parameters.get("inner_iterations", default_inner_iterations))
    step_length = float(config.parameters.get("step_length", 1.0))
    regularization = str(config.parameters.get("regularization", default_regularization))
    lambda_value = float(config.parameters.get("regularization_lambda", default_regularization_lambda))
    damping = float(config.parameters.get("damping", lambda_value**2))
    smooth_sigma = float(config.parameters.get("smooth_sigma", default_smooth_sigma))
    roi_update_only = bool(config.parameters.get("roi_update_only", False))

    delta_slowness = np.zeros(case.grid.shape, dtype=float)
    initial_norm = masked_norm(target, mask, weights)
    residual_curve = [initial_norm]
    update_norms: list[float] = []

    for _ in range(max(1, outer_iterations)):
        residual = target - projector.forward(delta_slowness)
        update, _inner_curve = cgls_solve(
            projector,
            residual,
            mask,
            iterations=inner_iterations,
            damping=damping,
            regularization=regularization,
            weights=weights,
        )
        delta_slowness = delta_slowness + step_length * update
        if smooth_sigma > 0.0:
            delta_slowness = _gaussian_smooth(delta_slowness, smooth_sigma)
        if roi_update_only and case.grid.roi_mask is not None:
            delta_slowness = np.where(np.asarray(case.grid.roi_mask, dtype=bool), delta_slowness, 0.0)
        update_norms.append(float(np.linalg.norm(update)))
        residual_curve.append(float(np.linalg.norm(apply_mask(target - projector.forward(delta_slowness), mask, weights)[mask])))

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
        "method_family": method_family,
        "residual_curve": residual_curve,
        "update_norm_curve": update_norms,
        "ray_weight_mean": float(np.mean(weights[mask])) if np.any(mask) else 0.0,
        "ray_weight_nonzero_fraction": float(np.mean(weights[mask] > 0.0)) if np.any(mask) else 0.0,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    if case.ground_truth.sound_speed_mps is not None:
        truth = np.asarray(case.ground_truth.sound_speed_mps, dtype=float)
        metrics.update(compute_image_metrics(sound_speed, truth, mask=case.grid.roi_mask))
        metrics.update(compute_baseline_improvement_metrics(sound_speed, truth, c0, mask=case.grid.roi_mask))

    return ReconstructionResult(
        algorithm=algorithm,
        case_id=case.case_id,
        sound_speed_mps=sound_speed,
        metrics=metrics,
    )
