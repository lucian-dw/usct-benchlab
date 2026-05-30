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
) -> tuple[np.ndarray, list[float]]:
    """Solve a masked least-squares system with CGLS."""

    x = np.zeros(projector.grid.shape, dtype=float)
    residual = apply_mask(target - projector.forward(x), mask)
    s = projector.adjoint(residual)
    if damping > 0:
        s = s - damping * x
    p = s.copy()
    gamma = float(np.vdot(s, s))
    residual_norms = [float(np.linalg.norm(residual[mask]))]

    for _ in range(max(0, int(iterations))):
        if gamma <= 0:
            break
        q = apply_mask(projector.forward(p), mask)
        denom = float(np.vdot(q, q))
        if damping > 0:
            denom += damping * float(np.vdot(p, p))
        if denom <= 0:
            break
        alpha = gamma / denom
        x = x + alpha * p
        residual = residual - alpha * q
        s_next = projector.adjoint(residual)
        if damping > 0:
            s_next = s_next - damping * x
        gamma_next = float(np.vdot(s_next, s_next))
        residual_norms.append(float(np.linalg.norm(residual[mask])))
        if gamma_next <= 1.0e-30:
            break
        beta = gamma_next / gamma
        p = s_next + beta * p
        s = s_next
        gamma = gamma_next
    return x, residual_norms


def sirt_solve(
    projector: StraightRayProjector,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    iterations: int,
    relaxation: float,
    nonnegative: bool = False,
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
        if nonnegative:
            x = np.maximum(x, 0.0)
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
) -> tuple[np.ndarray, list[float]]:
    """Run a simple ray-sequential ART/SART-style update."""

    x = np.zeros(projector.n_pixels, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    residual_norms: list[float] = []

    for _ in range(max(0, int(iterations))):
        for ray_id, (indices, lengths) in enumerate(zip(projector.indices_by_ray, projector.lengths_by_ray_m, strict=True)):
            if not mask[ray_id] or indices.size == 0:
                continue
            estimate = float(np.dot(lengths, x[indices]))
            denom = float(np.dot(lengths, lengths))
            if denom <= 0:
                continue
            correction = float(relaxation) * (target[ray_id] - estimate) / denom
            x[indices] += correction * lengths
        residual = apply_mask(target - projector.forward(x.reshape(projector.grid.shape)), mask)
        residual_norms.append(float(np.linalg.norm(residual[mask])))
    return x.reshape(projector.grid.shape), residual_norms


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
