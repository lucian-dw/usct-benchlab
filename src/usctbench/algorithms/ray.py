"""Ray-based USCT reconstruction algorithms and projector utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from usctbench.core.config import coerce_bool
from usctbench.core.registry import register_algorithm
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    ReconstructionResult,
    ResultStatus,
    USCTCase,
)
from usctbench.metrics import (
    compute_baseline_improvement_metrics,
    compute_image_metrics,
)


@dataclass(frozen=True)
class StraightRayProjector:
    """Line-integral projector over a Cartesian grid.

    Coordinates use the project-wide convention `[y, x]` in meters.
    """

    grid: GridSpec
    tx_pos_m: np.ndarray
    rx_pos_m: np.ndarray
    indices_by_ray: tuple[np.ndarray, ...]
    lengths_by_ray_m: tuple[np.ndarray, ...]

    @classmethod
    def from_case(cls, case: USCTCase) -> "StraightRayProjector":
        return cls.from_grid_geometry(case.grid, case.geometry)

    @classmethod
    def from_grid_geometry(
        cls, grid: GridSpec, geometry: GeometrySpec
    ) -> "StraightRayProjector":
        indices: list[np.ndarray] = []
        lengths: list[np.ndarray] = []
        for tx in geometry.tx_pos_m:
            for rx in geometry.rx_pos_m:
                ray_indices, ray_lengths = _trace_ray(grid, tx, rx)
                indices.append(ray_indices)
                lengths.append(ray_lengths)
        return cls(
            grid=grid,
            tx_pos_m=np.asarray(geometry.tx_pos_m, dtype=float),
            rx_pos_m=np.asarray(geometry.rx_pos_m, dtype=float),
            indices_by_ray=tuple(indices),
            lengths_by_ray_m=tuple(lengths),
        )

    @property
    def n_rays(self) -> int:
        return len(self.indices_by_ray)

    @property
    def n_pixels(self) -> int:
        ny, nx = self.grid.shape
        return ny * nx

    @property
    def ray_shape(self) -> tuple[int, int]:
        return (self.tx_pos_m.shape[0], self.rx_pos_m.shape[0])

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Compute line integrals for all source-receiver pairs."""

        flat = np.asarray(image, dtype=float).reshape(-1)
        if flat.size != self.n_pixels:
            raise ValueError(f"image has {flat.size} pixels, expected {self.n_pixels}")
        out = np.zeros(self.n_rays, dtype=float)
        for ray_id, (indices, lengths) in enumerate(
            zip(self.indices_by_ray, self.lengths_by_ray_m, strict=True)
        ):
            if indices.size:
                out[ray_id] = float(np.dot(lengths, flat[indices]))
        return out

    def adjoint(self, ray_values: np.ndarray) -> np.ndarray:
        """Backproject ray values using the exact transpose of `forward`."""

        values = np.asarray(ray_values, dtype=float).reshape(-1)
        if values.size != self.n_rays:
            raise ValueError(
                f"ray_values has {values.size} entries, expected {self.n_rays}"
            )
        flat = np.zeros(self.n_pixels, dtype=float)
        for value, indices, lengths in zip(
            values, self.indices_by_ray, self.lengths_by_ray_m, strict=True
        ):
            if indices.size:
                np.add.at(flat, indices, value * lengths)
        return flat.reshape(self.grid.shape)

    def row_norms(self, power: int = 2) -> np.ndarray:
        """Return per-ray sums of path lengths raised to `power`."""

        return np.array(
            [float(np.sum(lengths**power)) for lengths in self.lengths_by_ray_m]
        )

    def col_norms(self, power: int = 2) -> np.ndarray:
        """Return per-pixel sums of path lengths raised to `power`."""

        flat = np.zeros(self.n_pixels, dtype=float)
        for indices, lengths in zip(
            self.indices_by_ray, self.lengths_by_ray_m, strict=True
        ):
            if indices.size:
                np.add.at(flat, indices, lengths**power)
        return flat.reshape(self.grid.shape)


def _trace_ray(
    grid: GridSpec, tx_yx: np.ndarray, rx_yx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Trace one segment through the grid and return flat cell indices plus lengths."""

    y0, x0 = (float(tx_yx[0]), float(tx_yx[1]))
    y1, x1 = (float(rx_yx[0]), float(rx_yx[1]))
    dy_total = y1 - y0
    dx_total = x1 - x0
    segment_length = float(np.hypot(dy_total, dx_total))
    if segment_length == 0.0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    ny, nx = grid.shape
    dy, dx = grid.spacing_m
    origin_y, origin_x = grid.origin_m
    y_edges = origin_y + np.arange(ny + 1) * dy
    x_edges = origin_x + np.arange(nx + 1) * dx

    t_values = [0.0, 1.0]
    if dy_total != 0.0:
        t_values.extend(float((edge - y0) / dy_total) for edge in y_edges)
    if dx_total != 0.0:
        t_values.extend(float((edge - x0) / dx_total) for edge in x_edges)

    t = np.array([value for value in t_values if 0.0 <= value <= 1.0], dtype=float)
    if t.size < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    t = np.unique(np.round(t, decimals=15))

    indices: list[int] = []
    lengths: list[float] = []
    for start, end in zip(t[:-1], t[1:], strict=True):
        if end <= start:
            continue
        mid = 0.5 * (start + end)
        y_mid = y0 + mid * dy_total
        x_mid = x0 + mid * dx_total
        iy = int(np.floor((y_mid - origin_y) / dy))
        ix = int(np.floor((x_mid - origin_x) / dx))
        if 0 <= iy < ny and 0 <= ix < nx:
            indices.append(iy * nx + ix)
            lengths.append(segment_length * (end - start))

    if not indices:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    return np.asarray(indices, dtype=int), np.asarray(lengths, dtype=float)


def parameter(config: AlgorithmConfig, key: str, default: Any) -> Any:
    return config.parameters.get(key, default)


def reference_sound_speed(case: USCTCase, config: AlgorithmConfig) -> float:
    value = parameter(
        config,
        "reference_sound_speed_mps",
        case.metadata.get("reference_sound_speed_mps", 1500.0),
    )
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

    threshold = float(
        config.parameters.get(
            "min_ray_weight",
            config.parameters.get(
                "ray_weight_min", config.parameters.get("ray_weight_threshold", 0.0)
            ),
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
            "min_ray_weight": float(
                config.parameters.get(
                    "min_ray_weight",
                    config.parameters.get(
                        "ray_weight_min",
                        config.parameters.get("ray_weight_threshold", 0.0),
                    ),
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
                config.parameters.get(
                    "ray_weight_min", config.parameters.get("ray_weight_threshold", 0.0)
                ),
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
    roi = (
        np.ones(pred.shape, dtype=bool)
        if roi_mask is None
        else np.asarray(roi_mask, dtype=bool)
    )
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
    metrics["boundary_band_fraction"] = float(
        np.sum(boundary_finite) / max(1, int(np.sum(finite)))
    )
    if np.any(boundary_finite):
        metrics["boundary_band_rmse"] = float(
            np.sqrt(np.mean((pred[boundary_finite] - target[boundary_finite]) ** 2))
        )
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
        padded = np.pad(
            eroded, ((1, 1), (1, 1)), mode="constant", constant_values=False
        )
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return roi & ~eroded


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


def target_attenuation_integral(
    case: USCTCase, projector: StraightRayProjector
) -> tuple[np.ndarray, np.ndarray]:
    if case.measurement.log_amp is None:
        raise ValueError("attenuation reconstruction requires measurement.log_amp")
    log_amp = np.asarray(case.measurement.log_amp, dtype=float)
    if log_amp.ndim == 3:
        log_amp = np.nanmean(log_amp, axis=0)
    target = -log_amp.reshape(-1)
    if target.size != projector.n_rays:
        raise ValueError(
            "measurement.log_amp shape must match transmitter/receiver ray shape"
        )
    mask = valid_ray_mask(case, projector) & np.isfinite(target)
    target = np.where(mask, target, 0.0)
    return target, mask


def apply_mask(
    values: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    sqrt_weights: bool = True,
) -> np.ndarray:
    out = np.asarray(values, dtype=float).reshape(-1).copy()
    out[~mask] = 0.0
    if weights is not None:
        weight_values = np.asarray(weights, dtype=float).reshape(-1)
        if weight_values.size != out.size:
            raise ValueError("weights shape must match values")
        multiplier = (
            np.sqrt(np.clip(weight_values, 0.0, 1.0))
            if sqrt_weights
            else np.clip(weight_values, 0.0, 1.0)
        )
        out *= multiplier
    return out


def masked_norm(
    values: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None
) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite_mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if weights is None:
        return float(np.linalg.norm(values[finite_mask]))
    weight_values = np.asarray(weights, dtype=float).reshape(-1)
    weighted = values[finite_mask] * np.sqrt(
        np.clip(weight_values[finite_mask], 0.0, 1.0)
    )
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


def slowness_to_sound_speed(
    delta_slowness: np.ndarray, c0: float, bounds_mps: tuple[float, float]
) -> np.ndarray:
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

    d = (
        np.ones(projector.grid.shape, dtype=float)
        if preconditioner is None
        else np.asarray(preconditioner, dtype=float)
    )
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
        z = np.divide(
            initial, d, out=np.zeros_like(initial, dtype=float), where=d > 0.0
        )
    x = d * z
    residual = apply_mask(target - projector.forward(x), mask, weights)
    s = d * projector.adjoint(apply_mask(residual, mask, weights))
    if damping > 0:
        s = s - d * damping * _regularization_normal(
            x, regularization, roi_mask=roi_mask
        )
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
            s_next = s_next - d * damping * _regularization_normal(
                x, regularization, roi_mask=roi_mask
            )
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

    base_weights = (
        np.ones(projector.n_rays, dtype=float)
        if weights is None
        else np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, 1.0)
    )
    robust_weights = np.ones(projector.n_rays, dtype=float)
    x = np.zeros(projector.grid.shape, dtype=float)
    residual_curve = [
        masked_norm(np.asarray(target, dtype=float).reshape(-1), mask, base_weights)
    ]
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
        raw_residual = np.asarray(target, dtype=float).reshape(-1) - projector.forward(
            x
        )
        abs_residual = np.abs(raw_residual)
        robust_weights = np.where(
            abs_residual <= delta, 1.0, delta / np.maximum(abs_residual, delta)
        )
        robust_weights = np.where(np.asarray(mask, dtype=bool), robust_weights, 0.0)
        if inner_curve:
            residual_curve.extend(inner_curve[1:])
        residual_curve.append(
            masked_norm(raw_residual, mask, base_weights * robust_weights)
        )
    return x, residual_curve


def _regularization_forward(
    image: np.ndarray, kind: str, *, roi_mask: np.ndarray | None = None
) -> np.ndarray:
    kind_normalized = str(kind).lower()
    array = _roi_regularization_image(image, roi_mask)
    if kind_normalized in ("identity", "l2"):
        return array
    if kind_normalized in ("laplacian", "roughness"):
        return _laplacian(array)
    raise ValueError("regularization must be 'identity' or 'laplacian'")


def _regularization_normal(
    image: np.ndarray, kind: str, *, roi_mask: np.ndarray | None = None
) -> np.ndarray:
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


def _roi_regularization_image(
    image: np.ndarray, roi_mask: np.ndarray | None
) -> np.ndarray:
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
        col_norm = projector.adjoint(
            np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, 1.0)
            * np.asarray(mask, dtype=float).reshape(-1)
        )
    col_norm[col_norm <= 0.0] = 1.0
    residual_norms: list[float] = []
    for _ in range(max(0, int(iterations))):
        residual_raw = np.asarray(target, dtype=float).reshape(-1) - projector.forward(
            x
        )
        residual = apply_mask(residual_raw, mask, weights)
        residual_norms.append(float(np.linalg.norm(residual[mask])))
        update_values = apply_mask(
            residual_raw / row_norm, mask, weights, sqrt_weights=False
        )
        update = projector.adjoint(update_values) / col_norm
        x = x + float(relaxation) * update
        x = _post_update(
            x,
            iteration_index=_ + 1,
            nonnegative=nonnegative,
            smooth_sigma=smooth_sigma,
            smooth_every=smooth_every,
            roi_mask=roi_mask,
        )
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
    weight_values = (
        np.ones(projector.n_rays, dtype=float)
        if weights is None
        else np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, 1.0)
    )
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
            update_values = apply_mask(
                residual_raw / row_norm, subset_mask, weight_values, sqrt_weights=False
            )
            update = projector.adjoint(update_values) / col_norm
            x = x + float(relaxation) * update
            x = _post_update(
                x,
                iteration_index=_ + 1,
                nonnegative=False,
                smooth_sigma=smooth_sigma,
                smooth_every=smooth_every,
                roi_mask=roi_mask,
            )
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
    if (
        smooth_sigma > 0.0
        and smooth_every > 0
        and iteration_index % int(smooth_every) == 0
    ):
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
    return np.asarray(
        gaussian_filter(image, sigma=float(sigma), mode="nearest"), dtype=float
    )


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


def _masked_percentile(
    values: np.ndarray, mask: np.ndarray, percentile: float
) -> float:
    finite = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(
        np.percentile(np.asarray(values, dtype=float)[finite], float(percentile))
    )


def run_with_failure_capture(
    algorithm: str,
    case: USCTCase,
    func: Any,
) -> ReconstructionResult:
    started = time.perf_counter()
    try:
        result = func()
    except (
        Exception
    ) as exc:  # pragma: no cover - exercised through CLI/integration failures.
        return ReconstructionResult(
            algorithm=algorithm,
            case_id=case.case_id,
            runtime_s=time.perf_counter() - started,
            status=ResultStatus.FAILED,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
    result.runtime_s = time.perf_counter() - started
    return result


class StraightRayCGLSAlgorithm:
    name = "straight_cgls"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        def _run() -> ReconstructionResult:
            projector = StraightRayProjector.from_case(case)
            target, mask = target_delta_tof(case, projector)
            weights = configured_ray_weights(case, projector, mask, config)
            iterations = int(config.parameters.get("iterations", 30))
            regularization = str(config.parameters.get("regularization", "identity"))
            lambda_value = float(
                config.parameters.get(
                    "lambda", config.parameters.get("regularization_lambda", 0.0)
                )
            )
            damping = float(config.parameters.get("damping", lambda_value**2))
            roi_update_only = coerce_bool(
                config.parameters.get("roi_update_only", False)
            )
            roi_laplacian = coerce_bool(
                config.parameters.get(
                    "roi_laplacian", config.parameters.get("roi_aware_laplacian", False)
                )
            )
            use_coverage_preconditioning = coerce_bool(
                config.parameters.get("coverage_preconditioning", False)
            )
            robust_loss = str(config.parameters.get("robust_loss", "none")).lower()
            c0 = reference_sound_speed(case, config)
            initial_norm = masked_norm(target, mask, weights)
            roi_mask = (
                np.asarray(case.grid.roi_mask, dtype=bool)
                if case.grid.roi_mask is not None and (roi_update_only or roi_laplacian)
                else None
            )
            preconditioner = None
            coverage = projector.adjoint(
                np.asarray(mask, dtype=float) * np.clip(weights, 0.0, 1.0)
            )
            if use_coverage_preconditioning:
                preconditioner, coverage = coverage_preconditioner(
                    projector,
                    mask,
                    weights,
                    roi_mask=(
                        np.asarray(case.grid.roi_mask, dtype=bool)
                        if case.grid.roi_mask is not None and roi_update_only
                        else None
                    ),
                    eps=float(
                        config.parameters.get("coverage_preconditioner_eps", 1.0e-12)
                    ),
                    max_scale=float(
                        config.parameters.get("coverage_preconditioner_max_scale", 10.0)
                    ),
                    normalize=coerce_bool(
                        config.parameters.get("coverage_preconditioner_normalize", True)
                    ),
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
                delta_slowness = np.where(
                    np.asarray(case.grid.roi_mask, dtype=bool), delta_slowness, 0.0
                )
            sound_speed = slowness_to_sound_speed(
                delta_slowness, c0, speed_bounds(config)
            )
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
                "irls_iterations": int(
                    config.parameters.get(
                        "irls_iterations",
                        0 if robust_loss not in {"huber", "irls", "huber_irls"} else 3,
                    )
                ),
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
                        boundary_band_pixels=int(
                            config.parameters.get("boundary_band_pixels", 4)
                        ),
                    )
                )
            return ReconstructionResult(
                algorithm=self.name,
                case_id=case.case_id,
                sound_speed_mps=sound_speed,
                metrics=metrics,
            )

        return run_with_failure_capture(self.name, case, _run)


class StraightRaySIRTAlgorithm:
    name = "straight_sirt"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        def _run() -> ReconstructionResult:
            projector = StraightRayProjector.from_case(case)
            target, mask = target_delta_tof(case, projector)
            weights = configured_ray_weights(case, projector, mask, config)
            iterations = int(config.parameters.get("iterations", 50))
            relaxation = float(config.parameters.get("relaxation", 0.3))
            smooth_sigma = float(config.parameters.get("smooth_sigma", 0.0))
            smooth_every = int(config.parameters.get("smooth_every", 0))
            roi_update_only = coerce_bool(
                config.parameters.get("roi_update_only", False)
            )
            c0 = reference_sound_speed(case, config)
            initial_norm = masked_norm(target, mask, weights)
            delta_slowness, residual_norms = sirt_solve(
                projector,
                target,
                mask,
                iterations=iterations,
                relaxation=relaxation,
                smooth_sigma=smooth_sigma,
                smooth_every=smooth_every,
                roi_mask=case.grid.roi_mask if roi_update_only else None,
                weights=weights,
            )
            sound_speed = slowness_to_sound_speed(
                delta_slowness, c0, speed_bounds(config)
            )
            final_norm = residual_norms[-1] if residual_norms else initial_norm
            metrics = {
                **residual_metrics(initial_norm, final_norm),
                "iterations": iterations,
                "relaxation": relaxation,
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
            roi_update_only = coerce_bool(
                config.parameters.get("roi_update_only", False)
            )
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
            sound_speed = slowness_to_sound_speed(
                delta_slowness, c0, speed_bounds(config)
            )
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


def register_ray_algorithms(*, replace: bool = False) -> None:
    """Register built-in straight-ray sound-speed algorithms."""

    register_algorithm(
        "straight_sart",
        StraightRaySARTAlgorithm,
        description="Straight-ray SART sound-speed reconstruction.",
        tags=("ray", "sound-speed"),
        replace=replace,
    )
    register_algorithm(
        "straight_sirt",
        StraightRaySIRTAlgorithm,
        description="Straight-ray SIRT sound-speed reconstruction.",
        tags=("ray", "sound-speed"),
        replace=replace,
    )
    register_algorithm(
        "straight_cgls",
        StraightRayCGLSAlgorithm,
        description="Straight-ray CGLS sound-speed reconstruction.",
        tags=("ray", "sound-speed"),
        replace=replace,
    )


__all__ = [
    "StraightRayProjector",
    "StraightRayCGLSAlgorithm",
    "StraightRaySARTAlgorithm",
    "StraightRaySIRTAlgorithm",
    "register_ray_algorithms",
]
