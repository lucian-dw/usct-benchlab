"""Losses for the tiny synthetic waveform-inversion proof of life."""

from __future__ import annotations

import numpy as np


def path_travel_time_s(sound_speed_mps: np.ndarray, spacing_m: float) -> float:
    speed = np.asarray(sound_speed_mps, dtype=float)
    if speed.ndim != 1:
        raise ValueError("tiny FWI path model must be one-dimensional")
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    if np.any(speed <= 0):
        raise ValueError("sound speed must be positive")
    return float(np.sum(spacing_m / speed))


def waveform_from_speed(sound_speed_mps: np.ndarray, frequencies_hz: np.ndarray, spacing_m: float) -> np.ndarray:
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty 1-D array")
    if np.any(frequencies <= 0):
        raise ValueError("frequencies_hz must be positive")
    tau = path_travel_time_s(sound_speed_mps, spacing_m)
    phase = 2.0 * np.pi * frequencies * tau
    return np.concatenate([np.cos(phase), np.sin(phase)])


def waveform_mse_loss(predicted: np.ndarray, observed: np.ndarray) -> float:
    residual = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    if residual.shape != np.asarray(observed).shape:
        raise ValueError("predicted and observed waveforms must have the same shape")
    return 0.5 * float(np.mean(residual**2))


def loss_and_gradient(
    sound_speed_mps: np.ndarray,
    observed_waveform: np.ndarray,
    frequencies_hz: np.ndarray,
    spacing_m: float,
) -> tuple[float, np.ndarray]:
    speed = np.asarray(sound_speed_mps, dtype=float)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    observed = np.asarray(observed_waveform, dtype=float)
    tau = path_travel_time_s(speed, spacing_m)
    phase = 2.0 * np.pi * frequencies * tau
    predicted = np.concatenate([np.cos(phase), np.sin(phase)])
    if predicted.shape != observed.shape:
        raise ValueError("observed waveform shape does not match frequencies")

    residual = predicted - observed
    loss = 0.5 * float(np.mean(residual**2))
    d_loss_d_pred = residual / float(residual.size)
    d_pred_d_tau = np.concatenate(
        [
            -np.sin(phase) * (2.0 * np.pi * frequencies),
            np.cos(phase) * (2.0 * np.pi * frequencies),
        ]
    )
    d_loss_d_tau = float(np.dot(d_loss_d_pred, d_pred_d_tau))
    d_tau_d_speed = -spacing_m / (speed**2)
    return loss, d_loss_d_tau * d_tau_d_speed


def gradient_descent(
    initial_speed_mps: np.ndarray,
    observed_waveform: np.ndarray,
    frequencies_hz: np.ndarray,
    spacing_m: float,
    *,
    steps: int,
    learning_rate: float,
    bounds_mps: tuple[float, float] = (1300.0, 1700.0),
) -> tuple[np.ndarray, list[float]]:
    speed = np.asarray(initial_speed_mps, dtype=float).copy()
    low, high = bounds_mps
    if low <= 0 or high <= low:
        raise ValueError("bounds_mps must be positive and increasing")
    losses: list[float] = []
    for _ in range(max(0, int(steps))):
        loss, grad = loss_and_gradient(speed, observed_waveform, frequencies_hz, spacing_m)
        losses.append(loss)
        speed = np.clip(speed - float(learning_rate) * grad, low, high)
    loss, _ = loss_and_gradient(speed, observed_waveform, frequencies_hz, spacing_m)
    losses.append(loss)
    return speed, losses

