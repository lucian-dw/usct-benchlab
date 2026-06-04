"""Metric helpers for USCT reconstructions."""

from __future__ import annotations

import math

import numpy as np


def _masked_values(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=float)
    truth = np.asarray(target, dtype=float)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have the same shape")
    finite = np.isfinite(pred) & np.isfinite(truth)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if not np.any(finite):
        raise ValueError("no finite pixels available for metric computation")
    return pred[finite], truth[finite]


def compute_image_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, float]:
    """Compute scalar image metrics over finite ROI pixels."""

    pred, truth = _masked_values(prediction, target, mask)
    error = pred - truth
    mse = float(np.mean(error**2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(error)))
    target_range = float(np.max(truth) - np.min(truth))
    nrmse = rmse / target_range if target_range > 0 else 0.0 if rmse == 0 else math.inf
    data_range = (
        target_range if target_range > 0 else max(float(np.max(np.abs(truth))), 1.0)
    )
    psnr = math.inf if mse == 0 else 20.0 * math.log10(data_range / math.sqrt(mse))
    return {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}nrmse": nrmse,
        f"{prefix}psnr": psnr,
        f"{prefix}ssim": _global_ssim(pred, truth, data_range=data_range),
    }


def compute_baseline_improvement_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    baseline: float | np.ndarray,
    *,
    mask: np.ndarray | None = None,
    prefix: str = "water_",
) -> dict[str, float | bool]:
    """Compare reconstruction RMSE against a baseline image or scalar."""

    pred, truth = _masked_values(prediction, target, mask)
    baseline_array = np.asarray(baseline, dtype=float)
    if baseline_array.ndim == 0:
        base = np.full_like(np.asarray(target, dtype=float), float(baseline_array))
    else:
        base = baseline_array
    base_values, _ = _masked_values(base, target, mask)
    reconstruction_rmse = math.sqrt(float(np.mean((pred - truth) ** 2)))
    baseline_rmse = math.sqrt(float(np.mean((base_values - truth) ** 2)))
    if baseline_rmse == 0.0:
        relative = math.inf if reconstruction_rmse < baseline_rmse else 0.0
    else:
        relative = (baseline_rmse - reconstruction_rmse) / baseline_rmse
    return {
        f"{prefix}baseline_rmse": baseline_rmse,
        f"{prefix}reconstruction_rmse": reconstruction_rmse,
        f"{prefix}absolute_rmse_improvement": baseline_rmse - reconstruction_rmse,
        f"{prefix}relative_rmse_improvement": relative,
        f"{prefix}improved": reconstruction_rmse < baseline_rmse,
    }


def _global_ssim(
    prediction: np.ndarray, target: np.ndarray, *, data_range: float
) -> float:
    """Compute a deterministic global SSIM approximation without SciPy."""

    pred = prediction.astype(float)
    truth = target.astype(float)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = float(np.mean(pred))
    mu_y = float(np.mean(truth))
    var_x = float(np.mean((pred - mu_x) ** 2))
    var_y = float(np.mean((truth - mu_y) ** 2))
    cov_xy = float(np.mean((pred - mu_x) * (truth - mu_y)))
    denom = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    if denom == 0:
        return 1.0
    return float(((2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)) / denom)


def residual_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    prefix: str = "data_",
) -> dict[str, float]:
    """Compute finite residual norms between predicted and observed data."""

    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if pred.shape != obs.shape:
        raise ValueError("predicted and observed data must have the same shape")
    finite = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if not np.any(finite):
        raise ValueError("no finite data samples available for residual metrics")
    residual = pred[finite] - obs[finite]
    obs_values = obs[finite]
    norm = float(np.linalg.norm(residual))
    observed_norm = float(np.linalg.norm(obs_values))
    return {
        f"{prefix}residual_norm": norm,
        f"{prefix}observed_norm": observed_norm,
        f"{prefix}relative_residual": (
            norm / observed_norm if observed_norm > 0 else math.inf if norm > 0 else 0.0
        ),
        f"{prefix}rmse": float(np.sqrt(np.mean(residual**2))),
        f"{prefix}mae": float(np.mean(np.abs(residual))),
        f"{prefix}num_samples": float(residual.size),
    }


def baseline_improvement(
    baseline_error: float,
    reconstruction_error: float,
    *,
    prefix: str = "baseline_",
) -> dict[str, float | bool]:
    """Report absolute and relative improvement over a baseline error."""

    baseline = float(baseline_error)
    reconstruction = float(reconstruction_error)
    if baseline < 0 or reconstruction < 0:
        raise ValueError("errors must be non-negative")
    if baseline == 0:
        relative = math.inf if reconstruction < baseline else 0.0
    else:
        relative = (baseline - reconstruction) / baseline
    return {
        f"{prefix}error": baseline,
        f"{prefix}reconstruction_error": reconstruction,
        f"{prefix}absolute_improvement": baseline - reconstruction,
        f"{prefix}relative_improvement": relative,
        f"{prefix}improved": reconstruction < baseline,
    }
