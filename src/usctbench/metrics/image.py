"""Image-domain metrics for reconstruction quality."""

from __future__ import annotations

import math

import numpy as np


def _masked_values(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
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
    data_range = target_range if target_range > 0 else max(float(np.max(np.abs(truth))), 1.0)
    psnr = math.inf if mse == 0 else 20.0 * math.log10(data_range / math.sqrt(mse))
    return {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}nrmse": nrmse,
        f"{prefix}psnr": psnr,
        f"{prefix}ssim": _global_ssim(pred, truth, data_range=data_range),
    }


def _global_ssim(prediction: np.ndarray, target: np.ndarray, *, data_range: float) -> float:
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

