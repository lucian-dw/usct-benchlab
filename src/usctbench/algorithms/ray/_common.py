"""Shared helpers for straight-ray algorithms."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from usctbench.algorithms.ray.straight_projector import StraightRayProjector
from usctbench.schema import AlgorithmConfig, ReconstructionResult, ResultStatus, USCTCase


def parameter(config: AlgorithmConfig, key: str, default: Any) -> Any:
    return config.parameters.get(key, default)


def reference_sound_speed(case: USCTCase, config: AlgorithmConfig) -> float:
    value = parameter(config, "reference_sound_speed_mps", case.metadata.get("reference_sound_speed_mps", 1500.0))
    value = float(value)
    if value <= 0:
        raise ValueError("reference_sound_speed_mps must be positive")
    return value


def speed_bounds(config: AlgorithmConfig) -> tuple[float, float]:
    bounds = parameter(config, "sound_speed_bounds_mps", (1300.0, 1700.0))
    low, high = float(bounds[0]), float(bounds[1])
    if low <= 0 or high <= low:
        raise ValueError("sound_speed_bounds_mps must be [positive_low, higher_high]")
    return low, high


def valid_ray_mask(case: USCTCase, projector: StraightRayProjector) -> np.ndarray:
    if case.measurement.valid_mask is None:
        return np.ones(projector.n_rays, dtype=bool)
    mask = np.asarray(case.measurement.valid_mask, dtype=bool).reshape(-1)
    if mask.size != projector.n_rays:
        raise ValueError("measurement.valid_mask shape must match transmitter/receiver ray shape")
    return mask


def configured_ray_weights(
    case: USCTCase,
    projector: StraightRayProjector,
    mask: np.ndarray | None,
    config: AlgorithmConfig,
) -> np.ndarray:
    """Return per-ray confidence weights after optional config shaping."""

    threshold = float(
        config.parameters.get(
            "min_ray_weight",
            config.parameters.get("ray_weight_min", config.parameters.get("ray_weight_threshold", 0.0)),
        )
    )
    power = float(config.parameters.get("ray_weight_power", 1.0))
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
            raise ValueError("measurement.ray_weights/feature_quality shape must match transmitter/receiver ray shape")
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 0.0, 1.0)
    if min_weight > 0.0:
        weights = np.where(weights >= float(min_weight), weights, 0.0)
    if weight_power != 1.0:
        weights = np.power(weights, float(weight_power))
    if mask is not None:
        weights = np.where(np.asarray(mask, dtype=bool), weights, 0.0)
    return weights


def ray_weight_metrics(weights: np.ndarray, mask: np.ndarray, config: AlgorithmConfig) -> dict[str, float]:
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
            "min_ray_weight": float(
                config.parameters.get(
                    "min_ray_weight",
                    config.parameters.get("ray_weight_min", config.parameters.get("ray_weight_threshold", 0.0)),
                )
            ),
            "ray_weight_power": float(config.parameters.get("ray_weight_power", 1.0)),
        }
    used = values[valid]
    return {
        "ray_weight_mean": float(np.mean(used)),
        "ray_weight_nonzero_fraction": float(np.mean(used > 0.0)),
        "ray_weight_p10": float(np.percentile(used, 10.0)),
        "ray_weight_p50": float(np.percentile(used, 50.0)),
        "ray_weight_p90": float(np.percentile(used, 90.0)),
        "min_ray_weight": float(
            config.parameters.get(
                "min_ray_weight",
                config.parameters.get("ray_weight_min", config.parameters.get("ray_weight_threshold", 0.0)),
            )
        ),
        "ray_weight_power": float(config.parameters.get("ray_weight_power", 1.0)),
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
        ray_values = ray_values * np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, 1.0)
    coverage = projector.adjoint(ray_values)
    preconditioner = 1.0 / np.sqrt(np.maximum(coverage, 0.0) + max(float(eps), 0.0))
    roi = np.ones(projector.grid.shape, dtype=bool) if roi_mask is None else np.asarray(roi_mask, dtype=bool)
    finite_roi = roi & np.isfinite(preconditioner) & (coverage > 0.0)
    if normalize and np.any(finite_roi):
        median = float(np.median(preconditioner[finite_roi]))
        if median > 0.0:
            preconditioner = preconditioner / median
    preconditioner = np.clip(np.nan_to_num(preconditioner, nan=0.0, posinf=0.0, neginf=0.0), 0.0, float(max_scale))
    if roi_mask is not None:
        preconditioner = np.where(roi, preconditioner, 0.0)
    return preconditioner, coverage


def image_diagnostic_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    roi_mask: np.ndarray | None = None,
    coverage: np.ndarray | None = None,
    boundary_band_pixels: int = 4,
) -> dict[str, float]:
    """Return coverage/error and boundary-band image diagnostics."""

    pred = np.asarray(prediction, dtype=float)
    target = np.asarray(truth, dtype=float)
    roi = np.ones(pred.shape, dtype=bool) if roi_mask is None else np.asarray(roi_mask, dtype=bool)
    finite = roi & np.isfinite(pred) & np.isfinite(target)
    error = np.abs(pred - target)
    metrics: dict[str, float] = {}
    if coverage is not None:
        cov = np.asarray(coverage, dtype=float)
        metrics["coverage_abs_error_corr"] = _corr(cov[finite], error[finite])
        metrics["coverage_mean"] = _masked_mean(cov, finite)
        metrics["coverage_p10"] = _masked_percentile(cov, finite, 10.0)
    boundary = boundary_band_mask(roi, pixels=boundary_band_pixels)
    boundary_finite = boundary & np.isfinite(pred) & np.isfinite(target)
    metrics["boundary_band_pixels"] = float(boundary_band_pixels)
    metrics["boundary_band_fraction"] = float(np.sum(boundary_finite) / max(1, int(np.sum(finite))))
    if np.any(boundary_finite):
        metrics["boundary_band_rmse"] = float(np.sqrt(np.mean((pred[boundary_finite] - target[boundary_finite]) ** 2)))
        metrics["boundary_band_mae"] = float(np.mean(error[boundary_finite]))
    else:
        metrics["boundary_band_rmse"] = float("nan")
        metrics["boundary_band_mae"] = float("nan")
    return metrics


def boundary_band_mask(roi_mask: np.ndarray, *, pixels: int) -> np.ndarray:
    roi = np.asarray(roi_mask, dtype=bool)
    if pixels <= 0 or not np.any(roi):
        return np.zeros_like(roi, dtype=bool)
    eroded = roi.copy()
    for _ in range(int(pixels)):
        padded = np.pad(eroded, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return roi & ~eroded


def target_delta_tof(case: USCTCase, projector: StraightRayProjector) -> tuple[np.ndarray, np.ndarray]:
    if case.measurement.delta_tof_s is None:
        raise ValueError("straight-ray sound-speed reconstruction requires measurement.delta_tof_s")
    target = np.asarray(case.measurement.delta_tof_s, dtype=float).reshape(-1)
    if target.size != projector.n_rays:
        raise ValueError("measurement.delta_tof_s shape must match transmitter/receiver ray shape")
    mask = valid_ray_mask(case, projector) & np.isfinite(target)
    target = np.where(mask, target, 0.0)
    return target, mask


def target_attenuation_integral(case: USCTCase, projector: StraightRayProjector) -> tuple[np.ndarray, np.ndarray]:
    if case.measurement.log_amp is None:
        raise ValueError("attenuation reconstruction requires measurement.log_amp")
    log_amp = np.asarray(case.measurement.log_amp, dtype=float)
    if log_amp.ndim == 3:
        log_amp = np.nanmean(log_amp, axis=0)
    target = -log_amp.reshape(-1)
    if target.size != projector.n_rays:
        raise ValueError("measurement.log_amp shape must match transmitter/receiver ray shape")
    mask = valid_ray_mask(case, projector) & np.isfinite(target)
    target = np.where(mask, target, 0.0)
    return target, mask


def apply_mask(values: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None, *, sqrt_weights: bool = True) -> np.ndarray:
    out = np.asarray(values, dtype=float).reshape(-1).copy()
    out[~mask] = 0.0
    if weights is not None:
        weight_values = np.asarray(weights, dtype=float).reshape(-1)
        if weight_values.size != out.size:
            raise ValueError("weights shape must match values")
        multiplier = np.sqrt(np.clip(weight_values, 0.0, 1.0)) if sqrt_weights else np.clip(weight_values, 0.0, 1.0)
        out *= multiplier
    return out


def masked_norm(values: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite_mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if weights is None:
        return float(np.linalg.norm(values[finite_mask]))
    weight_values = np.asarray(weights, dtype=float).reshape(-1)
    weighted = values[finite_mask] * np.sqrt(np.clip(weight_values[finite_mask], 0.0, 1.0))
    return float(np.linalg.norm(weighted))


def residual_metrics(initial_norm: float, final_norm: float) -> dict[str, float]:
    if initial_norm > 0.0:
        relative = final_norm / initial_norm
        reduction = 1.0 - relative
    else:
        relative = 0.0 if final_norm == 0.0 else float("inf")
        reduction = 0.0
    return {
        "data_residual_norm": float(final_norm),
        "initial_data_residual_norm": float(initial_norm),
        "data_relative_residual": float(relative),
        "data_residual_reduction": float(reduction),
    }


def slowness_to_sound_speed(delta_slowness: np.ndarray, c0: float, bounds_mps: tuple[float, float]) -> np.ndarray:
    low, high = bounds_mps
    min_slowness = 1.0 / high
    max_slowness = 1.0 / low
    slowness = np.clip((1.0 / c0) + delta_slowness, min_slowness, max_slowness)
    return 1.0 / slowness


def cgls_solve(
    projector: StraightRayProjector,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    iterations: int,
    damping: float = 0.0,
    regularization: str = "identity",
    weights: np.ndarray | None = None,
    preconditioner: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    initial_image: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Solve a masked least-squares system with CGLS."""

    d = np.ones(projector.grid.shape, dtype=float) if preconditioner is None else np.asarray(preconditioner, dtype=float)
    if d.shape != projector.grid.shape:
        raise ValueError("preconditioner must match projector grid shape")
    if roi_mask is not None:
        d = np.where(np.asarray(roi_mask, dtype=bool), d, 0.0)
    if initial_image is None:
        z = np.zeros(projector.grid.shape, dtype=float)
    else:
        initial = np.asarray(initial_image, dtype=float)
        if initial.shape != projector.grid.shape:
            raise ValueError("initial_image must match projector grid shape")
        z = np.divide(initial, d, out=np.zeros_like(initial, dtype=float), where=d > 0.0)
    x = d * z
    residual = apply_mask(target - projector.forward(x), mask, weights)
    s = d * projector.adjoint(apply_mask(residual, mask, weights))
    if damping > 0:
        s = s - d * damping * _regularization_normal(x, regularization, roi_mask=roi_mask)
    p = s.copy()
    gamma = float(np.vdot(s, s))
    residual_norms = [float(np.linalg.norm(residual[mask]))]

    for _ in range(max(0, int(iterations))):
        if gamma <= 0:
            break
        q = apply_mask(projector.forward(d * p), mask, weights)
        denom = float(np.vdot(q, q))
        if damping > 0:
            reg_p = _regularization_forward(d * p, regularization, roi_mask=roi_mask)
            denom += damping * float(np.vdot(reg_p, reg_p))
        if denom <= 0:
            break
        alpha = gamma / denom
        z = z + alpha * p
        x = d * z
        residual = residual - alpha * q
        s_next = d * projector.adjoint(apply_mask(residual, mask, weights))
        if damping > 0:
            s_next = s_next - d * damping * _regularization_normal(x, regularization, roi_mask=roi_mask)
        gamma_next = float(np.vdot(s_next, s_next))
        residual_norms.append(float(np.linalg.norm(residual[mask])))
        if gamma_next <= 1.0e-30:
            break
        beta = gamma_next / gamma
        p = s_next + beta * p
        s = s_next
        gamma = gamma_next
    return x, residual_norms


def huber_irls_cgls_solve(
    projector: StraightRayProjector,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    iterations: int,
    damping: float = 0.0,
    regularization: str = "identity",
    weights: np.ndarray | None = None,
    preconditioner: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    huber_delta: float = 5.0e-7,
    irls_iterations: int = 3,
) -> tuple[np.ndarray, list[float]]:
    """Solve with Huber-style IRLS residual down-weighting."""

    base_weights = np.ones(projector.n_rays, dtype=float) if weights is None else np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, 1.0)
    robust_weights = np.ones(projector.n_rays, dtype=float)
    x = np.zeros(projector.grid.shape, dtype=float)
    residual_curve = [masked_norm(np.asarray(target, dtype=float).reshape(-1), mask, base_weights)]
    delta = max(float(huber_delta), 1.0e-15)
    for _ in range(max(1, int(irls_iterations))):
        effective_weights = base_weights * robust_weights
        x, inner_curve = cgls_solve(
            projector,
            target,
            mask,
            iterations=iterations,
            damping=damping,
            regularization=regularization,
            weights=effective_weights,
            preconditioner=preconditioner,
            roi_mask=roi_mask,
            initial_image=x,
        )
        raw_residual = np.asarray(target, dtype=float).reshape(-1) - projector.forward(x)
        abs_residual = np.abs(raw_residual)
        robust_weights = np.where(abs_residual <= delta, 1.0, delta / np.maximum(abs_residual, delta))
        robust_weights = np.where(np.asarray(mask, dtype=bool), robust_weights, 0.0)
        if inner_curve:
            residual_curve.extend(inner_curve[1:])
        residual_curve.append(masked_norm(raw_residual, mask, base_weights * robust_weights))
    return x, residual_curve


def _regularization_forward(image: np.ndarray, kind: str, *, roi_mask: np.ndarray | None = None) -> np.ndarray:
    kind_normalized = str(kind).lower()
    array = _roi_regularization_image(image, roi_mask)
    if kind_normalized in ("identity", "l2"):
        return array
    if kind_normalized in ("laplacian", "roughness"):
        return _laplacian(array)
    raise ValueError("regularization must be 'identity' or 'laplacian'")


def _regularization_normal(image: np.ndarray, kind: str, *, roi_mask: np.ndarray | None = None) -> np.ndarray:
    kind_normalized = str(kind).lower()
    roi = None if roi_mask is None else np.asarray(roi_mask, dtype=bool)
    array = _roi_regularization_image(image, roi)
    if kind_normalized in ("identity", "l2"):
        normal = array
        return normal if roi is None else np.where(roi, normal, 0.0)
    if kind_normalized in ("laplacian", "roughness"):
        normal = _laplacian(_laplacian(array))
        return normal if roi is None else np.where(roi, normal, 0.0)
    raise ValueError("regularization must be 'identity' or 'laplacian'")


def _roi_regularization_image(image: np.ndarray, roi_mask: np.ndarray | None) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    if roi_mask is None:
        return array
    return np.where(np.asarray(roi_mask, dtype=bool), array, 0.0)


def _laplacian(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    padded = np.pad(array, ((1, 1), (1, 1)), mode="edge")
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )


def sirt_solve(
    projector: StraightRayProjector,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    iterations: int,
    relaxation: float,
    nonnegative: bool = False,
    smooth_sigma: float = 0.0,
    smooth_every: int = 0,
    roi_mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Run a normalized simultaneous iterative reconstruction update."""

    x = np.zeros(projector.grid.shape, dtype=float)
    row_norm = projector.row_norms(power=1)
    row_norm[row_norm <= 0.0] = 1.0
    col_norm = projector.col_norms(power=1)
    if weights is not None:
        col_norm = projector.adjoint(np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, 1.0) * np.asarray(mask, dtype=float).reshape(-1))
    col_norm[col_norm <= 0.0] = 1.0
    residual_norms: list[float] = []
    for _ in range(max(0, int(iterations))):
        residual_raw = np.asarray(target, dtype=float).reshape(-1) - projector.forward(x)
        residual = apply_mask(residual_raw, mask, weights)
        residual_norms.append(float(np.linalg.norm(residual[mask])))
        update_values = apply_mask(residual_raw / row_norm, mask, weights, sqrt_weights=False)
        update = projector.adjoint(update_values) / col_norm
        x = x + float(relaxation) * update
        x = _post_update(x, iteration_index=_ + 1, nonnegative=nonnegative, smooth_sigma=smooth_sigma, smooth_every=smooth_every, roi_mask=roi_mask)
    residual = apply_mask(target - projector.forward(x), mask, weights)
    residual_norms.append(float(np.linalg.norm(residual[mask])))
    return x, residual_norms


def sart_solve(
    projector: StraightRayProjector,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    iterations: int,
    relaxation: float,
    subsets: int = 8,
    smooth_sigma: float = 0.0,
    smooth_every: int = 0,
    roi_mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Run subset-normalized SART updates over masked rays."""

    x = np.zeros(projector.grid.shape, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    weight_values = np.ones(projector.n_rays, dtype=float) if weights is None else np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, 1.0)
    row_norm = projector.row_norms(power=1)
    row_norm[row_norm <= 0.0] = 1.0
    ray_ids = np.flatnonzero(np.asarray(mask, dtype=bool))
    subset_count = max(1, min(int(subsets), int(ray_ids.size) if ray_ids.size else 1))
    residual_norms: list[float] = []

    for _ in range(max(0, int(iterations))):
        for subset in np.array_split(ray_ids, subset_count):
            if subset.size == 0:
                continue
            subset_mask = np.zeros(projector.n_rays, dtype=bool)
            subset_mask[subset] = True
            residual_raw = target - projector.forward(x)
            residual = apply_mask(residual_raw, subset_mask, weight_values)
            col_norm = projector.adjoint(subset_mask.astype(float) * weight_values)
            col_norm[col_norm <= 0.0] = 1.0
            update_values = apply_mask(residual_raw / row_norm, subset_mask, weight_values, sqrt_weights=False)
            update = projector.adjoint(update_values) / col_norm
            x = x + float(relaxation) * update
            x = _post_update(x, iteration_index=_ + 1, nonnegative=False, smooth_sigma=smooth_sigma, smooth_every=smooth_every, roi_mask=roi_mask)
        residual = apply_mask(target - projector.forward(x), mask, weight_values)
        residual_norms.append(float(np.linalg.norm(residual[mask])))
    return x, residual_norms


def _post_update(
    image: np.ndarray,
    *,
    iteration_index: int,
    nonnegative: bool,
    smooth_sigma: float,
    smooth_every: int,
    roi_mask: np.ndarray | None,
) -> np.ndarray:
    updated = np.asarray(image, dtype=float)
    if roi_mask is not None:
        updated = np.where(np.asarray(roi_mask, dtype=bool), updated, 0.0)
    if smooth_sigma > 0.0 and smooth_every > 0 and iteration_index % int(smooth_every) == 0:
        updated = _gaussian_smooth(updated, float(smooth_sigma))
        if roi_mask is not None:
            updated = np.where(np.asarray(roi_mask, dtype=bool), updated, 0.0)
    if nonnegative:
        updated = np.maximum(updated, 0.0)
    return updated


def _gaussian_smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
    except ModuleNotFoundError:
        return image
    return np.asarray(gaussian_filter(image, sigma=float(sigma), mode="nearest"), dtype=float)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 2:
        return float("nan")
    x = x[finite]
    y = y[finite]
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    finite = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)[finite]))


def _masked_percentile(values: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    finite = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float)[finite], float(percentile)))


def run_with_failure_capture(
    algorithm: str,
    case: USCTCase,
    func: Any,
) -> ReconstructionResult:
    started = time.perf_counter()
    try:
        result = func()
    except Exception as exc:  # pragma: no cover - exercised through CLI/integration failures.
        return ReconstructionResult(
            algorithm=algorithm,
            case_id=case.case_id,
            runtime_s=time.perf_counter() - started,
            status=ResultStatus.FAILED,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
    result.runtime_s = time.perf_counter() - started
    return result
