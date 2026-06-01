"""Amplitude feature extraction from wavefield/reference pairs."""

from __future__ import annotations

import numpy as np

from usctbench.data.features import log_amplitude_ratio


def log_amplitude_ratio_from_frequency(
    freq_data: np.ndarray,
    reference_freq_data: np.ndarray,
    *,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Compute per-frequency log amplitude ratios."""

    return log_amplitude_ratio(freq_data, reference_freq_data, eps=eps)


def log_amplitude_ratio_from_time(
    time_data: np.ndarray,
    reference: np.ndarray,
    *,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Compute log ratio of trace RMS amplitude relative to reference."""

    signal_rms = _rms(time_data)
    reference_rms = _rms(reference)
    return np.log(np.maximum(signal_rms, eps) / np.maximum(reference_rms, eps))


def _rms(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError("time-domain amplitude extraction expects shape [n_tx, n_rx, n_time]")
    return np.sqrt(np.nanmean(array**2, axis=-1))
