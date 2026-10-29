"""Ray-based USCT reconstruction algorithms and projector utilities."""

from __future__ import annotations


import numpy as np

from usctbench.core.schema import (
    AlgorithmConfig,
    USCTCase,
)

from usctbench.operators.straight_ray import StraightRayProjector


def reference_sound_speed(case: USCTCase, config: AlgorithmConfig) -> float:
    value = config.parameters.reference_sound_speed_mps
    if value is None:
        value = case.metadata.get("reference_sound_speed_mps", 1500.0)
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("reference_sound_speed_mps must be positive")
    return value


def speed_bounds(config: AlgorithmConfig) -> tuple[float, float]:
    bounds = config.parameters.sound_speed_bounds_mps
    low, high = float(bounds[0]), float(bounds[1])
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high <= low:
        raise ValueError("sound_speed_bounds_mps must be [positive_low, higher_high]")
    return low, high


def valid_ray_mask(case: USCTCase, projector: StraightRayProjector) -> np.ndarray:
    if case.measurement.valid_mask is None:
        return np.ones(projector.n_rays, dtype=bool)
    mask = np.asarray(case.measurement.valid_mask, dtype=bool).reshape(-1)
    if mask.size != projector.n_rays:
        raise ValueError(
            "measurement.valid_mask shape must match transmitter/receiver ray shape"
        )
    return mask


def configured_ray_weights(
    case: USCTCase,
    projector: StraightRayProjector,
    mask: np.ndarray | None,
    config: AlgorithmConfig,
) -> np.ndarray:
    """Return per-ray confidence weights after optional config shaping."""

    threshold = float(config.parameters.min_ray_weight)
    power = float(config.parameters.ray_weight_power)
    return ray_weights(case, projector, mask, min_weight=threshold, weight_power=power)


def ray_weights(
    case: USCTCase,
    projector: StraightRayProjector,
    mask: np.ndarray | None = None,
    *,
    min_weight: float = 0.0,
    weight_power: float = 1.0,
) -> np.ndarray:
    """Return per-ray confidence weights clipped to [0, 1].

    ``min_weight`` and ``weight_power`` let algorithms sharpen feature-derived
    confidence without editing the feature case. They default to the historical
    behavior: no threshold and no exponentiation.
    """

    if min_weight < 0.0 or min_weight > 1.0:
        raise ValueError("min_ray_weight/ray_weight_threshold must be in [0, 1]")
    if weight_power <= 0.0:
        raise ValueError("ray_weight_power must be positive")

    source = case.measurement.ray_weights
    if source is None:
        source = case.measurement.feature_quality
    if source is None:
        weights = np.ones(projector.n_rays, dtype=float)
    else:
        weights = np.asarray(source, dtype=float).reshape(-1)
        if weights.size != projector.n_rays:
            raise ValueError(
                "measurement.ray_weights/feature_quality shape must match transmitter/receiver ray shape"
            )
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 0.0, 1.0)
    if min_weight > 0.0:
        weights = np.where(weights >= float(min_weight), weights, 0.0)
    if weight_power != 1.0:
        weights = np.power(weights, float(weight_power))
    if mask is not None:
        weights = np.where(np.asarray(mask, dtype=bool), weights, 0.0)
    return weights


def ray_weight_metrics(
    weights: np.ndarray, mask: np.ndarray, config: AlgorithmConfig
) -> dict[str, float]:
    """Summarize transformed ray weights used by a reconstruction."""

    valid = np.asarray(mask, dtype=bool).reshape(-1)
    values = np.asarray(weights, dtype=float).reshape(-1)
    if not np.any(valid):
        return {
            "ray_weight_mean": 0.0,
            "ray_weight_nonzero_fraction": 0.0,
            "ray_weight_p10": 0.0,
            "ray_weight_p50": 0.0,
            "ray_weight_p90": 0.0,
            "min_ray_weight": float(config.parameters.min_ray_weight),
            "ray_weight_power": float(config.parameters.ray_weight_power),
        }
    used = values[valid]
    return {
        "ray_weight_mean": float(np.mean(used)),
        "ray_weight_nonzero_fraction": float(np.mean(used > 0.0)),
        "ray_weight_p10": float(np.percentile(used, 10.0)),
        "ray_weight_p50": float(np.percentile(used, 50.0)),
        "ray_weight_p90": float(np.percentile(used, 90.0)),
        "min_ray_weight": float(config.parameters.min_ray_weight),
        "ray_weight_power": float(config.parameters.ray_weight_power),
    }


def coverage_preconditioner(
    projector: StraightRayProjector,
    mask: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    roi_mask: np.ndarray | None = None,
    eps: float = 1.0e-12,
    max_scale: float = 10.0,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return D ~= 1/sqrt(A.T @ ray_weights + eps) and the coverage map."""

    ray_values = np.asarray(mask, dtype=float).reshape(-1)
    if weights is not None:
        ray_values = ray_values * np.clip(
            np.asarray(weights, dtype=float).reshape(-1), 0.0, 1.0
        )
    coverage = projector.adjoint(ray_values)
    preconditioner = 1.0 / np.sqrt(np.maximum(coverage, 0.0) + max(float(eps), 0.0))
    roi = (
        np.ones(projector.grid.shape, dtype=bool)
        if roi_mask is None
        else np.asarray(roi_mask, dtype=bool)
    )
    finite_roi = roi & np.isfinite(preconditioner) & (coverage > 0.0)
    if normalize and np.any(finite_roi):
        median = float(np.median(preconditioner[finite_roi]))
        if median > 0.0:
            preconditioner = preconditioner / median
    preconditioner = np.clip(
        np.nan_to_num(preconditioner, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        float(max_scale),
    )
    if roi_mask is not None:
        preconditioner = np.where(roi, preconditioner, 0.0)
    return preconditioner, coverage


def target_delta_tof(
    case: USCTCase, projector: StraightRayProjector
) -> tuple[np.ndarray, np.ndarray]:
    if case.measurement.delta_tof_s is None:
        raise ValueError(
            "straight-ray sound-speed reconstruction requires measurement.delta_tof_s"
        )
    target = np.asarray(case.measurement.delta_tof_s, dtype=float).reshape(-1)
    if target.size != projector.n_rays:
        raise ValueError(
            "measurement.delta_tof_s shape must match transmitter/receiver ray shape"
        )
    mask = valid_ray_mask(case, projector) & np.isfinite(target)
    target = np.where(mask, target, 0.0)
    return target, mask
