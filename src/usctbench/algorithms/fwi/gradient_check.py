"""Finite-difference gradient checks for tiny FWI."""

from __future__ import annotations

import numpy as np

from .losses import loss_and_gradient


def finite_difference_directional_derivative(
    sound_speed_mps: np.ndarray,
    observed_waveform: np.ndarray,
    frequencies_hz: np.ndarray,
    spacing_m: float,
    direction: np.ndarray,
    *,
    epsilon: float = 1.0e-3,
) -> float:
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("direction must be non-zero")
    unit_direction = direction / norm
    plus = np.asarray(sound_speed_mps, dtype=float) + epsilon * unit_direction
    minus = np.asarray(sound_speed_mps, dtype=float) - epsilon * unit_direction
    loss_plus, _ = loss_and_gradient(plus, observed_waveform, frequencies_hz, spacing_m)
    loss_minus, _ = loss_and_gradient(minus, observed_waveform, frequencies_hz, spacing_m)
    return (loss_plus - loss_minus) / (2.0 * epsilon)


def check_tiny_fwi_gradient(
    sound_speed_mps: np.ndarray,
    observed_waveform: np.ndarray,
    frequencies_hz: np.ndarray,
    spacing_m: float,
    *,
    direction: np.ndarray | None = None,
    epsilon: float = 1.0e-3,
) -> dict[str, float]:
    speed = np.asarray(sound_speed_mps, dtype=float)
    if direction is None:
        direction = np.linspace(-1.0, 1.0, speed.size).reshape(speed.shape)
    direction = np.asarray(direction, dtype=float)
    unit_direction = direction / float(np.linalg.norm(direction))
    _, gradient = loss_and_gradient(speed, observed_waveform, frequencies_hz, spacing_m)
    analytic = float(np.dot(gradient, unit_direction))
    finite_difference = finite_difference_directional_derivative(
        speed,
        observed_waveform,
        frequencies_hz,
        spacing_m,
        unit_direction,
        epsilon=epsilon,
    )
    denom = max(abs(analytic), abs(finite_difference), 1.0e-12)
    return {
        "analytic": analytic,
        "finite_difference": finite_difference,
        "relative_error": abs(analytic - finite_difference) / denom,
    }

