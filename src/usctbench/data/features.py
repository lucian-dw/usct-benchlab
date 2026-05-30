"""Feature extraction for classical ray-based USCT baselines."""

from __future__ import annotations

import numpy as np


def log_amplitude_ratio(
    signal: np.ndarray,
    reference: np.ndarray,
    *,
    eps: float = 1.0e-12,
) -> np.ndarray:
    """Compute `log(|signal| / |reference|)` with numerical clipping."""

    signal_amp = np.maximum(np.abs(signal), eps)
    reference_amp = np.maximum(np.abs(reference), eps)
    return np.log(signal_amp / reference_amp)


def phase_delay_seconds(
    signal: np.ndarray,
    reference: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    phase_convention: str = "-omega_t",
) -> np.ndarray:
    """Estimate delay from complex phase relative to a reference.

    For the default convention `exp(-i omega t)`, a positive extra travel time
    appears as a negative phase slope, so `delta_t = -slope / (2*pi)`.
    """

    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty 1-D array")
    if np.any(frequencies <= 0):
        raise ValueError("frequencies_hz must contain positive frequencies")

    ratio = np.asarray(signal) / np.asarray(reference)
    if ratio.shape[0] != frequencies.size:
        raise ValueError("signal/reference first dimension must match frequencies_hz")
    phase = np.unwrap(np.angle(ratio), axis=0)
    slope = _fit_phase_slope(frequencies, phase)
    sign = -1.0 if phase_convention == "-omega_t" else 1.0 if phase_convention == "+omega_t" else None
    if sign is None:
        raise ValueError("phase_convention must be '-omega_t' or '+omega_t'")
    return sign * slope / (2.0 * np.pi)


def valid_amplitude_mask(
    signal: np.ndarray,
    reference: np.ndarray,
    *,
    min_reference_amplitude: float = 1.0e-8,
    min_signal_amplitude: float = 1.0e-8,
) -> np.ndarray:
    """Return receivers with adequate signal and reference amplitudes."""

    signal_amp = np.abs(signal)
    reference_amp = np.abs(reference)
    if signal_amp.ndim >= 3:
        signal_amp = np.nanmax(signal_amp, axis=0)
        reference_amp = np.nanmax(reference_amp, axis=0)
    return (signal_amp >= min_signal_amplitude) & (reference_amp >= min_reference_amplitude)


def extract_frequency_features(
    signal: np.ndarray,
    reference: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    min_reference_amplitude: float = 1.0e-8,
    min_signal_amplitude: float = 1.0e-8,
) -> dict[str, np.ndarray]:
    """Extract ray features from frequency-domain data.

    Parameters are expected in shape `[n_freq, n_tx, n_rx]`.
    """

    signal_array = np.asarray(signal)
    reference_array = np.asarray(reference)
    if signal_array.shape != reference_array.shape:
        raise ValueError("signal and reference must have identical shapes")
    if signal_array.ndim != 3:
        raise ValueError("frequency-domain feature extraction expects shape [n_freq, n_tx, n_rx]")
    return {
        "delta_tof_s": phase_delay_seconds(signal_array, reference_array, frequencies_hz),
        "log_amp": log_amplitude_ratio(signal_array, reference_array),
        "valid_mask": valid_amplitude_mask(
            signal_array,
            reference_array,
            min_reference_amplitude=min_reference_amplitude,
            min_signal_amplitude=min_signal_amplitude,
        ),
    }


def _fit_phase_slope(frequencies_hz: np.ndarray, phase: np.ndarray) -> np.ndarray:
    centered_frequency = frequencies_hz - float(np.mean(frequencies_hz))
    denom = float(np.sum(centered_frequency**2))
    if denom == 0.0:
        # One frequency can only provide phase/frequency delay, not a slope fit.
        return phase[0] / frequencies_hz[0]
    centered_phase = phase - np.mean(phase, axis=0, keepdims=True)
    return np.tensordot(centered_frequency, centered_phase, axes=(0, 0)) / denom

