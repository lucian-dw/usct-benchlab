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
    min_frequencies: int = 3,
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
    if frequencies.size < int(min_frequencies):
        raise ValueError("phase-slope travel-time estimation requires at least three frequencies")

    ratio = np.asarray(signal) / np.asarray(reference)
    if ratio.shape[0] != frequencies.size:
        raise ValueError("signal/reference first dimension must match frequencies_hz")
    phase = np.unwrap(np.angle(ratio), axis=0)
    slope, _ = _fit_phase_slope(frequencies, phase)
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
        signal_amp = np.nanmin(signal_amp, axis=0)
        reference_amp = np.nanmin(reference_amp, axis=0)
    return (signal_amp >= min_signal_amplitude) & (reference_amp >= min_reference_amplitude)


def extract_frequency_features(
    signal: np.ndarray,
    reference: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    min_reference_amplitude: float = 1.0e-8,
    min_signal_amplitude: float = 1.0e-8,
    min_phase_frequencies: int = 3,
    allow_low_frequency_count: bool = False,
    phase_residual_threshold_rad: float = 0.5,
) -> dict[str, np.ndarray | str]:
    """Extract ray features from frequency-domain data.

    Parameters are expected in shape `[n_freq, n_tx, n_rx]`.
    """

    signal_array = np.asarray(signal)
    reference_array = np.asarray(reference)
    if signal_array.shape != reference_array.shape:
        raise ValueError("signal and reference must have identical shapes")
    if signal_array.ndim != 3:
        raise ValueError("frequency-domain feature extraction expects shape [n_freq, n_tx, n_rx]")
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.size < int(min_phase_frequencies) and not allow_low_frequency_count:
        raise ValueError("phase-slope travel-time features require at least three frequencies")
    ratio = signal_array / reference_array
    phase = np.unwrap(np.angle(ratio), axis=0)
    if frequencies.size >= 2:
        slope, fit_residual = _fit_phase_slope(frequencies, phase)
        delta_tof = -slope / (2.0 * np.pi)
        phase_fit_rms = np.sqrt(np.nanmean(fit_residual**2, axis=0))
    else:
        delta_tof = np.full(signal_array.shape[1:], np.nan, dtype=float)
        phase_fit_rms = np.full(signal_array.shape[1:], np.inf, dtype=float)
    amplitude_mask = valid_amplitude_mask(
        signal_array,
        reference_array,
        min_reference_amplitude=min_reference_amplitude,
        min_signal_amplitude=min_signal_amplitude,
    )
    phase_quality_mask = np.isfinite(delta_tof) & (phase_fit_rms <= float(phase_residual_threshold_rad))
    return {
        "delta_tof_s": delta_tof,
        "log_amp": log_amplitude_ratio(signal_array, reference_array),
        "valid_mask": amplitude_mask & phase_quality_mask,
        "amplitude_valid_mask": amplitude_mask,
        "phase_quality_mask": phase_quality_mask,
        "phase_fit_rms_rad": phase_fit_rms,
        "feature_quality": "low" if frequencies.size < int(min_phase_frequencies) else "ok",
    }


def _fit_phase_slope(frequencies_hz: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered_frequency = frequencies_hz - float(np.mean(frequencies_hz))
    denom = float(np.sum(centered_frequency**2))
    if denom == 0.0:
        slope = phase[0] / frequencies_hz[0]
        return slope, np.full_like(phase, np.nan, dtype=float)
    centered_phase = phase - np.mean(phase, axis=0, keepdims=True)
    slope = np.tensordot(centered_frequency, centered_phase, axes=(0, 0)) / denom
    intercept = np.mean(phase, axis=0) - slope * float(np.mean(frequencies_hz))
    fitted = frequencies_hz.reshape((-1,) + (1,) * (phase.ndim - 1)) * slope + intercept
    return slope, phase - fitted
