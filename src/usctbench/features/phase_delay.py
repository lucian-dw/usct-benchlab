"""Multi-frequency phase-slope delay extraction."""

from __future__ import annotations

import numpy as np

from usctbench.data.features import extract_frequency_features


def frequency_response_from_time(
    time_data: np.ndarray,
    time_axis_s: np.ndarray,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    """Sample complex frequency responses from time-domain traces."""

    signals = np.asarray(time_data, dtype=float)
    times = np.asarray(time_axis_s, dtype=float).reshape(-1)
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    if signals.ndim != 3:
        raise ValueError("time_data must have shape [n_tx, n_rx, n_time]")
    if times.size != signals.shape[-1]:
        raise ValueError("time_axis_s length must match time_data")
    if frequencies.ndim != 1 or frequencies.size == 0 or np.any(frequencies <= 0):
        raise ValueError("frequencies_hz must be positive")
    dt = float(np.median(np.diff(times))) if times.size > 1 else 1.0
    kernel = np.exp(-1j * 2.0 * np.pi * frequencies[:, None] * times[None, :])
    return np.tensordot(kernel, signals, axes=(1, 2)) * dt


def multi_frequency_phase_slope_delay(
    freq_data: np.ndarray,
    reference_freq_data: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    min_frequencies: int = 3,
    phase_residual_threshold_rad: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return phase-slope delay, phase RMS residual, and validity mask."""

    features = extract_frequency_features(
        freq_data,
        reference_freq_data,
        frequencies_hz,
        min_phase_frequencies=min_frequencies,
        allow_low_frequency_count=True,
        phase_residual_threshold_rad=phase_residual_threshold_rad,
    )
    delay = np.asarray(features["delta_tof_s"], dtype=float)
    phase_rms = np.asarray(features["phase_fit_rms_rad"], dtype=float)
    valid = np.asarray(features["valid_mask"], dtype=bool)
    if np.asarray(frequencies_hz).size < int(min_frequencies):
        valid &= False
    return delay, phase_rms, valid
