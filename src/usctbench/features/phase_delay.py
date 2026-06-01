"""Multi-frequency phase-slope delay extraction."""

from __future__ import annotations

import numpy as np

from usctbench.data.features import _fit_phase_slope, extract_frequency_features


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


def windowed_frequency_response_from_time(
    time_data: np.ndarray,
    time_axis_s: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    window_start_s: np.ndarray | float,
    window_end_s: np.ndarray | float,
) -> np.ndarray:
    """Sample complex responses from direct-arrival time windows."""

    signals = np.asarray(time_data, dtype=float)
    times = np.asarray(time_axis_s, dtype=float).reshape(-1)
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    if signals.ndim != 3:
        raise ValueError("time_data must have shape [n_tx, n_rx, n_time]")
    if times.size != signals.shape[-1]:
        raise ValueError("time_axis_s length must match time_data")
    if frequencies.ndim != 1 or frequencies.size == 0 or np.any(frequencies <= 0):
        raise ValueError("frequencies_hz must be positive")
    starts = _broadcast_window(window_start_s, signals.shape[:2])
    ends = _broadcast_window(window_end_s, signals.shape[:2])
    dt = float(np.median(np.diff(times))) if times.size > 1 else 1.0
    response = np.zeros((frequencies.size,) + signals.shape[:2], dtype=np.complex128)
    kernel = np.exp(-1j * 2.0 * np.pi * frequencies[:, None] * times[None, :])
    for tx in range(signals.shape[0]):
        for rx in range(signals.shape[1]):
            keep = (times >= starts[tx, rx]) & (times <= ends[tx, rx])
            if int(np.sum(keep)) < 3:
                response[:, tx, rx] = np.nan + 0.0j
                continue
            window = np.hanning(int(np.sum(keep)))
            if window.size <= 2:
                window = np.ones_like(window)
            trace = np.zeros(times.size, dtype=float)
            values = np.asarray(signals[tx, rx], dtype=float)
            trace[keep] = np.where(np.isfinite(values[keep]), values[keep], 0.0) * window
            response[:, tx, rx] = np.dot(kernel, trace) * dt
    return response


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


def gated_phase_slope_delay(
    time_data: np.ndarray,
    reference: np.ndarray,
    time_axis_s: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    water_tof_s: np.ndarray,
    delta_min_s: np.ndarray | float,
    delta_max_s: np.ndarray | float,
    gate_margin_s: float,
    min_frequencies: int = 3,
    min_relative_amplitude: float = 1.0e-4,
    phase_residual_threshold_rad: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return gated phase-slope delay, fit RMS, validity, and confidence."""

    water = np.asarray(water_tof_s, dtype=float)
    delta_min = np.asarray(delta_min_s, dtype=float)
    delta_max = np.asarray(delta_max_s, dtype=float)
    signal_freq = windowed_frequency_response_from_time(
        time_data,
        time_axis_s,
        frequencies_hz,
        window_start_s=water + delta_min - float(gate_margin_s),
        window_end_s=water + delta_max + float(gate_margin_s),
    )
    reference_freq = windowed_frequency_response_from_time(
        reference,
        time_axis_s,
        frequencies_hz,
        window_start_s=water - float(gate_margin_s),
        window_end_s=water + float(gate_margin_s),
    )
    delay, phase_rms, valid, used_count = _phase_slope_from_frequency(
        signal_freq,
        reference_freq,
        frequencies_hz,
        min_frequencies=min_frequencies,
        min_relative_amplitude=min_relative_amplitude,
        phase_residual_threshold_rad=phase_residual_threshold_rad,
    )
    confidence = np.zeros_like(delay, dtype=float)
    finite = valid & np.isfinite(phase_rms)
    confidence[finite] = np.clip(1.0 - phase_rms[finite] / float(phase_residual_threshold_rad), 0.0, 1.0)
    confidence *= np.clip(used_count / max(1, int(min_frequencies)), 0.0, 1.0)
    return delay, phase_rms, valid, confidence


def _phase_slope_from_frequency(
    signal: np.ndarray,
    reference: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    min_frequencies: int,
    min_relative_amplitude: float,
    phase_residual_threshold_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    signal_array = np.asarray(signal, dtype=np.complex128)
    reference_array = np.asarray(reference, dtype=np.complex128)
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    if signal_array.shape != reference_array.shape or signal_array.ndim != 3:
        raise ValueError("signal and reference must share shape [n_freq, n_tx, n_rx]")
    shape = signal_array.shape[1:]
    delay = np.full(shape, np.nan, dtype=float)
    rms = np.full(shape, np.inf, dtype=float)
    valid = np.zeros(shape, dtype=bool)
    used_count = np.zeros(shape, dtype=float)
    ref_amp = np.abs(reference_array)
    sig_amp = np.abs(signal_array)
    ref_scale = np.nanmax(ref_amp, axis=0)
    sig_scale = np.nanmax(sig_amp, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            signal_array,
            reference_array,
            out=np.full(signal_array.shape, np.nan + 0.0j, dtype=np.complex128),
            where=ref_amp > 0.0,
        )
    for tx in range(shape[0]):
        for rx in range(shape[1]):
            keep = (
                np.isfinite(ratio[:, tx, rx])
                & (ref_amp[:, tx, rx] >= float(min_relative_amplitude) * max(float(ref_scale[tx, rx]), 1.0e-30))
                & (sig_amp[:, tx, rx] >= float(min_relative_amplitude) * max(float(sig_scale[tx, rx]), 1.0e-30))
            )
            if int(np.sum(keep)) < int(min_frequencies):
                continue
            phase = np.unwrap(np.angle(ratio[keep, tx, rx]))
            freqs = frequencies[keep]
            slope, residual = _fit_phase_slope(freqs, phase[:, None, None])
            value = float(-slope[0, 0] / (2.0 * np.pi))
            fit_rms = float(np.sqrt(np.nanmean(residual[:, 0, 0] ** 2)))
            delay[tx, rx] = value
            rms[tx, rx] = fit_rms
            used_count[tx, rx] = float(np.sum(keep))
            valid[tx, rx] = np.isfinite(value) and fit_rms <= float(phase_residual_threshold_rad)
    return delay, rms, valid, used_count


def _broadcast_window(value: np.ndarray | float, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        return np.full(shape, float(array), dtype=float)
    return np.broadcast_to(array, shape).astype(float, copy=True)
