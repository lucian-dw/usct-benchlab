"""Forward-model data-consistency metrics."""

from __future__ import annotations

import math

import numpy as np


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
        f"{prefix}relative_residual": norm / observed_norm if observed_norm > 0 else math.inf if norm > 0 else 0.0,
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

