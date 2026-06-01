"""Time-of-flight feature estimators for raw wavefields."""

from __future__ import annotations

import numpy as np


def first_arrival_tof(
    time_data: np.ndarray,
    time_axis_s: np.ndarray,
    *,
    threshold_fraction: float = 0.08,
    min_energy: float = 1.0e-18,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate absolute first-arrival ToF by threshold crossing."""

    signals = np.asarray(time_data, dtype=float)
    times = np.asarray(time_axis_s, dtype=float).reshape(-1)
    if signals.ndim != 3:
        raise ValueError("time_data must have shape [n_tx, n_rx, n_time]")
    if signals.shape[-1] != times.size:
        raise ValueError("time_axis_s length must match time_data last dimension")
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError("threshold_fraction must be in (0, 1)")

    amplitude = np.abs(signals)
    peak = np.nanmax(amplitude, axis=-1)
    energy = np.nansum(signals**2, axis=-1)
    threshold = peak[..., None] * float(threshold_fraction)
    crossings = amplitude >= threshold
    has_crossing = np.any(crossings, axis=-1) & np.isfinite(peak) & (energy > float(min_energy))
    first_index = np.argmax(crossings, axis=-1)
    tof = np.full(signals.shape[:2], np.nan, dtype=float)
    tof[has_crossing] = times[first_index[has_crossing]]
    return tof, has_crossing


def peak_tof(time_data: np.ndarray, time_axis_s: np.ndarray, *, min_energy: float = 1.0e-18) -> tuple[np.ndarray, np.ndarray]:
    """Estimate pulse arrival by the absolute peak sample."""

    signals = np.asarray(time_data, dtype=float)
    times = np.asarray(time_axis_s, dtype=float).reshape(-1)
    if signals.ndim != 3:
        raise ValueError("time_data must have shape [n_tx, n_rx, n_time]")
    peak_index = np.nanargmax(np.abs(signals), axis=-1)
    energy = np.nansum(signals**2, axis=-1)
    valid = np.isfinite(energy) & (energy > float(min_energy))
    tof = np.full(signals.shape[:2], np.nan, dtype=float)
    tof[valid] = times[peak_index[valid]]
    return tof, valid


def cross_correlation_delay(
    time_data: np.ndarray,
    reference: np.ndarray,
    time_axis_s: np.ndarray,
    *,
    max_lag_s: float | None = None,
    min_energy: float = 1.0e-18,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate delay relative to a water/reference waveform by xcorr."""

    signals = np.asarray(time_data, dtype=float)
    refs = np.asarray(reference, dtype=float)
    times = np.asarray(time_axis_s, dtype=float).reshape(-1)
    if signals.shape != refs.shape or signals.ndim != 3:
        raise ValueError("time_data and reference must share shape [n_tx, n_rx, n_time]")
    if times.size != signals.shape[-1]:
        raise ValueError("time_axis_s length must match signal length")
    dt = _time_step(times)
    max_lag_samples = signals.shape[-1] - 1 if max_lag_s is None else max(0, int(round(float(max_lag_s) / dt)))

    delay = np.full(signals.shape[:2], np.nan, dtype=float)
    valid = np.zeros(signals.shape[:2], dtype=bool)
    for tx in range(signals.shape[0]):
        for rx in range(signals.shape[1]):
            signal = _demean(signals[tx, rx])
            ref = _demean(refs[tx, rx])
            signal_energy = float(np.dot(signal, signal))
            ref_energy = float(np.dot(ref, ref))
            if signal_energy <= min_energy or ref_energy <= min_energy:
                continue
            corr = np.correlate(signal, ref, mode="full")
            lags = np.arange(-(signals.shape[-1] - 1), signals.shape[-1])
            keep = np.abs(lags) <= max_lag_samples
            if not np.any(keep):
                continue
            lag = int(lags[keep][np.argmax(corr[keep])])
            delay[tx, rx] = lag * dt
            valid[tx, rx] = True
    return delay, valid


def _time_step(time_axis_s: np.ndarray) -> float:
    if time_axis_s.size < 2:
        raise ValueError("time_axis_s must contain at least two samples")
    diffs = np.diff(time_axis_s)
    dt = float(np.median(diffs))
    if dt <= 0.0:
        raise ValueError("time_axis_s must be increasing")
    return dt


def _demean(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    mean = float(np.mean(values[finite]))
    return np.where(finite, values - mean, 0.0)
