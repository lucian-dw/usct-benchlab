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


def target_delta_tof(case: USCTCase, projector: StraightRayProjector) -> tuple[np.ndarray, np.ndarray]:
    if case.measurement.delta_tof_s is None:
        raise ValueError("straight-ray sound-speed reconstruction requires measurement.delta_tof_s")
    target = np.asarray(case.measurement.delta_tof_s, dtype=float).reshape(-1)
    if target.size != projector.n_rays:
        raise ValueError("measurement.delta_tof_s shape must match transmitter/receiver ray shape")
    return target, valid_ray_mask(case, projector)


def target_attenuation_integral(case: USCTCase, projector: StraightRayProjector) -> tuple[np.ndarray, np.ndarray]:
    if case.measurement.log_amp is None:
        raise ValueError("attenuation reconstruction requires measurement.log_amp")
    log_amp = np.asarray(case.measurement.log_amp, dtype=float)
    if log_amp.ndim == 3:
        log_amp = np.nanmean(log_amp, axis=0)
    target = -log_amp.reshape(-1)
    if target.size != projector.n_rays:
        raise ValueError("measurement.log_amp shape must match transmitter/receiver ray shape")
    return target, valid_ray_mask(case, projector)


def apply_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).reshape(-1).copy()
    out[~mask] = 0.0
    return out


def masked_norm(values: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    return float(np.linalg.norm(values[mask]))


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
) -> tuple[np.ndarray, list[float]]:
    """Solve a masked least-squares system with CGLS."""

    x = np.zeros(projector.grid.shape, dtype=float)
    residual = apply_mask(target - projector.forward(x), mask)
    s = projector.adjoint(residual)
    if damping > 0:
        s = s - damping * _regularization_normal(x, regularization)
    p = s.copy()
    gamma = float(np.vdot(s, s))
    residual_norms = [float(np.linalg.norm(residual[mask]))]

    for _ in range(max(0, int(iterations))):
        if gamma <= 0:
            break
        q = apply_mask(projector.forward(p), mask)
        denom = float(np.vdot(q, q))
        if damping > 0:
            reg_p = _regularization_forward(p, regularization)
            denom += damping * float(np.vdot(reg_p, reg_p))
        if denom <= 0:
            break
        alpha = gamma / denom
        x = x + alpha * p
        residual = residual - alpha * q
        s_next = projector.adjoint(residual)
        if damping > 0:
            s_next = s_next - damping * _regularization_normal(x, regularization)
        gamma_next = float(np.vdot(s_next, s_next))
        residual_norms.append(float(np.linalg.norm(residual[mask])))
        if gamma_next <= 1.0e-30:
            break
        beta = gamma_next / gamma
        p = s_next + beta * p
        s = s_next
        gamma = gamma_next
    return x, residual_norms


def _regularization_forward(image: np.ndarray, kind: str) -> np.ndarray:
    kind_normalized = str(kind).lower()
    if kind_normalized in ("identity", "l2"):
        return np.asarray(image, dtype=float)
    if kind_normalized in ("laplacian", "roughness"):
        return _laplacian(image)
    raise ValueError("regularization must be 'identity' or 'laplacian'")


def _regularization_normal(image: np.ndarray, kind: str) -> np.ndarray:
    kind_normalized = str(kind).lower()
    if kind_normalized in ("identity", "l2"):
        return np.asarray(image, dtype=float)
    if kind_normalized in ("laplacian", "roughness"):
        return _laplacian(_laplacian(image))
    raise ValueError("regularization must be 'identity' or 'laplacian'")


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
) -> tuple[np.ndarray, list[float]]:
    """Run a normalized simultaneous iterative reconstruction update."""

    x = np.zeros(projector.grid.shape, dtype=float)
    row_norm = projector.row_norms(power=1)
    row_norm[row_norm <= 0.0] = 1.0
    col_norm = projector.col_norms(power=1)
    col_norm[col_norm <= 0.0] = 1.0
    residual_norms: list[float] = []
    for _ in range(max(0, int(iterations))):
        residual = apply_mask(target - projector.forward(x), mask)
        residual_norms.append(float(np.linalg.norm(residual[mask])))
        update = projector.adjoint(residual / row_norm) / col_norm
        x = x + float(relaxation) * update
        x = _post_update(x, iteration_index=_ + 1, nonnegative=nonnegative, smooth_sigma=smooth_sigma, smooth_every=smooth_every, roi_mask=roi_mask)
    residual = apply_mask(target - projector.forward(x), mask)
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
) -> tuple[np.ndarray, list[float]]:
    """Run subset-normalized SART updates over masked rays."""

    x = np.zeros(projector.grid.shape, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
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
            residual = apply_mask(target - projector.forward(x), subset_mask)
            col_norm = projector.adjoint(subset_mask.astype(float))
            col_norm[col_norm <= 0.0] = 1.0
            update = projector.adjoint(residual / row_norm) / col_norm
            x = x + float(relaxation) * update
            x = _post_update(x, iteration_index=_ + 1, nonnegative=False, smooth_sigma=smooth_sigma, smooth_every=smooth_every, roi_mask=roi_mask)
        residual = apply_mask(target - projector.forward(x), mask)
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
