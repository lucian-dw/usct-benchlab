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


def aic_first_arrival_tof(
    time_data: np.ndarray,
    time_axis_s: np.ndarray,
    *,
    expected_tof_s: np.ndarray | None = None,
    window_start_s: np.ndarray | float | None = None,
    window_end_s: np.ndarray | float | None = None,
    min_energy: float = 1.0e-18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate absolute first-arrival ToF with an AIC picker.

    The picker is intentionally windowed around the expected direct arrival.
    Using it on a full trace tends to select late scattering minima in USCT
    wavefields.
    """

    signals, times = _validate_time_data(time_data, time_axis_s)
    starts, ends = _arrival_windows(signals.shape[:2], times, expected_tof_s, window_start_s, window_end_s)
    tof = np.full(signals.shape[:2], np.nan, dtype=float)
    valid = np.zeros(signals.shape[:2], dtype=bool)
    confidence = np.zeros(signals.shape[:2], dtype=float)
    for tx in range(signals.shape[0]):
        for rx in range(signals.shape[1]):
            indices = _window_indices(times, starts[tx, rx], ends[tx, rx])
            if indices.size < 8:
                continue
            trace = _demean(signals[tx, rx, indices])
            energy = float(np.dot(trace, trace))
            if energy <= float(min_energy):
                continue
            score = _aic_curve(trace)
            if score.size == 0 or not np.any(np.isfinite(score)):
                continue
            local_idx = int(np.nanargmin(score))
            idx = int(indices[min(max(local_idx, 0), indices.size - 1)])
            tof[tx, rx] = times[idx]
            valid[tx, rx] = True
            spread = float(np.nanpercentile(score, 90.0) - np.nanmin(score))
            confidence[tx, rx] = float(np.clip(spread / (abs(float(np.nanmedian(score))) + spread + 1.0e-12), 0.0, 1.0))
    return tof, valid, confidence


def envelope_first_arrival_tof(
    time_data: np.ndarray,
    time_axis_s: np.ndarray,
    *,
    expected_tof_s: np.ndarray | None = None,
    window_start_s: np.ndarray | float | None = None,
    window_end_s: np.ndarray | float | None = None,
    threshold_fraction: float = 0.12,
    min_energy: float = 1.0e-18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate absolute first-arrival ToF from the Hilbert envelope."""

    signals, times = _validate_time_data(time_data, time_axis_s)
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError("threshold_fraction must be in (0, 1)")
    starts, ends = _arrival_windows(signals.shape[:2], times, expected_tof_s, window_start_s, window_end_s)
    tof = np.full(signals.shape[:2], np.nan, dtype=float)
    valid = np.zeros(signals.shape[:2], dtype=bool)
    confidence = np.zeros(signals.shape[:2], dtype=float)
    for tx in range(signals.shape[0]):
        for rx in range(signals.shape[1]):
            indices = _window_indices(times, starts[tx, rx], ends[tx, rx])
            if indices.size < 4:
                continue
            trace = np.asarray(signals[tx, rx, indices], dtype=float)
            energy = float(np.nansum(trace**2))
            if energy <= float(min_energy):
                continue
            envelope = _envelope(trace)
            peak = float(np.nanmax(envelope))
            if not np.isfinite(peak) or peak <= 0.0:
                continue
            threshold = threshold_fraction * peak
            hits = np.flatnonzero(envelope >= threshold)
            if hits.size == 0:
                continue
            idx = int(indices[int(hits[0])])
            tof[tx, rx] = times[idx]
            valid[tx, rx] = True
            noise = float(np.nanmedian(envelope[: max(1, min(8, envelope.size // 4))]))
            confidence[tx, rx] = float(np.clip((peak - threshold) / (peak + noise + 1.0e-12), 0.0, 1.0))
    return tof, valid, confidence


def aic_first_arrival_delta(
    time_data: np.ndarray,
    reference: np.ndarray,
    time_axis_s: np.ndarray,
    *,
    water_tof_s: np.ndarray,
    delta_min_s: np.ndarray | float,
    delta_max_s: np.ndarray | float,
    gate_margin_s: float,
    min_energy: float = 1.0e-18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return object-water first-arrival delay using direct-arrival AIC windows."""

    object_tof, object_valid, object_conf = aic_first_arrival_tof(
        time_data,
        time_axis_s,
        expected_tof_s=np.asarray(water_tof_s) + 0.5 * (np.asarray(delta_min_s) + np.asarray(delta_max_s)),
        window_start_s=np.asarray(water_tof_s) + np.asarray(delta_min_s) - float(gate_margin_s),
        window_end_s=np.asarray(water_tof_s) + np.asarray(delta_max_s) + float(gate_margin_s),
        min_energy=min_energy,
    )
    water_tof, water_valid, water_conf = aic_first_arrival_tof(
        reference,
        time_axis_s,
        expected_tof_s=water_tof_s,
        window_start_s=np.asarray(water_tof_s) - float(gate_margin_s),
        window_end_s=np.asarray(water_tof_s) + float(gate_margin_s),
        min_energy=min_energy,
    )
    delta = object_tof - water_tof
    valid = object_valid & water_valid & np.isfinite(delta)
    confidence = np.sqrt(np.clip(object_conf, 0.0, 1.0) * np.clip(water_conf, 0.0, 1.0))
    return delta, valid, confidence


def envelope_first_arrival_delta(
    time_data: np.ndarray,
    reference: np.ndarray,
    time_axis_s: np.ndarray,
    *,
    water_tof_s: np.ndarray,
    delta_min_s: np.ndarray | float,
    delta_max_s: np.ndarray | float,
    gate_margin_s: float,
    threshold_fraction: float = 0.12,
    min_energy: float = 1.0e-18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return object-water first-arrival delay using envelope thresholding."""

    object_tof, object_valid, object_conf = envelope_first_arrival_tof(
        time_data,
        time_axis_s,
        expected_tof_s=np.asarray(water_tof_s) + 0.5 * (np.asarray(delta_min_s) + np.asarray(delta_max_s)),
        window_start_s=np.asarray(water_tof_s) + np.asarray(delta_min_s) - float(gate_margin_s),
        window_end_s=np.asarray(water_tof_s) + np.asarray(delta_max_s) + float(gate_margin_s),
        threshold_fraction=threshold_fraction,
        min_energy=min_energy,
    )
    water_tof, water_valid, water_conf = envelope_first_arrival_tof(
        reference,
        time_axis_s,
        expected_tof_s=water_tof_s,
        window_start_s=np.asarray(water_tof_s) - float(gate_margin_s),
        window_end_s=np.asarray(water_tof_s) + float(gate_margin_s),
        threshold_fraction=threshold_fraction,
        min_energy=min_energy,
    )
    delta = object_tof - water_tof
    valid = object_valid & water_valid & np.isfinite(delta)
    confidence = np.sqrt(np.clip(object_conf, 0.0, 1.0) * np.clip(water_conf, 0.0, 1.0))
    return delta, valid, confidence


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


def normalized_cross_correlation_delay(
    time_data: np.ndarray,
    reference: np.ndarray,
    time_axis_s: np.ndarray,
    *,
    min_lag_s: np.ndarray | float | None = None,
    max_lag_s: np.ndarray | float | None = None,
    signal_gate_start_s: np.ndarray | float | None = None,
    signal_gate_end_s: np.ndarray | float | None = None,
    reference_gate_start_s: np.ndarray | float | None = None,
    reference_gate_end_s: np.ndarray | float | None = None,
    min_energy: float = 1.0e-18,
) -> dict[str, np.ndarray]:
    """Estimate delay using per-lag normalized cross-correlation.

    Unlike :func:`cross_correlation_delay`, this function is intended for
    formal wavefield-derived features and therefore supports bounded lag
    search plus direct-arrival time gates.
    """

    signals = np.asarray(time_data, dtype=float)
    refs = np.asarray(reference, dtype=float)
    times = np.asarray(time_axis_s, dtype=float).reshape(-1)
    if signals.shape != refs.shape or signals.ndim != 3:
        raise ValueError("time_data and reference must share shape [n_tx, n_rx, n_time]")
    if times.size != signals.shape[-1]:
        raise ValueError("time_axis_s length must match signal length")

    dt = _time_step(times)
    shape = signals.shape[:2]
    min_lag = _broadcast_or_default(min_lag_s, shape, -(signals.shape[-1] - 1) * dt)
    max_lag = _broadcast_or_default(max_lag_s, shape, (signals.shape[-1] - 1) * dt)
    sig_start = _broadcast_or_default(signal_gate_start_s, shape, times[0])
    sig_end = _broadcast_or_default(signal_gate_end_s, shape, times[-1])
    ref_start = _broadcast_or_default(reference_gate_start_s, shape, times[0])
    ref_end = _broadcast_or_default(reference_gate_end_s, shape, times[-1])

    delay = np.full(shape, np.nan, dtype=float)
    valid = np.zeros(shape, dtype=bool)
    peak_corr = np.zeros(shape, dtype=float)
    peak_ratio = np.ones(shape, dtype=float)
    peak_lag = np.full(shape, np.nan, dtype=float)
    confidence = np.zeros(shape, dtype=float)

    lag_values = np.arange(-(signals.shape[-1] - 1), signals.shape[-1], dtype=int)
    lag_seconds = lag_values.astype(float) * dt
    for tx in range(shape[0]):
        for rx in range(shape[1]):
            keep_lag = (lag_seconds >= min_lag[tx, rx]) & (lag_seconds <= max_lag[tx, rx])
            if not np.any(keep_lag):
                continue
            sig0, signal = _gated_demeaned_segment(signals[tx, rx], times, sig_start[tx, rx], sig_end[tx, rx])
            ref0, ref = _gated_demeaned_segment(refs[tx, rx], times, ref_start[tx, rx], ref_end[tx, rx])
            if signal.size < 3 or ref.size < 3:
                continue
            corr_values = []
            lags = lag_values[keep_lag]
            for lag in lags:
                a, b = _overlap_for_segment_lag(signal, sig0, ref, ref0, int(lag))
                if a.size < 3:
                    corr_values.append(np.nan)
                    continue
                a_energy = float(np.dot(a, a))
                b_energy = float(np.dot(b, b))
                if a_energy <= min_energy or b_energy <= min_energy:
                    corr_values.append(np.nan)
                    continue
                corr_values.append(float(np.dot(a, b) / np.sqrt(a_energy * b_energy)))
            corr_array = np.asarray(corr_values, dtype=float)
            if not np.any(np.isfinite(corr_array)):
                continue
            best_local = int(np.nanargmax(corr_array))
            best_corr = float(corr_array[best_local])
            lag = int(lags[best_local])
            delay[tx, rx] = lag * dt
            peak_lag[tx, rx] = lag * dt
            peak_corr[tx, rx] = best_corr
            ratio = _peak_ratio(corr_array, best_local)
            peak_ratio[tx, rx] = ratio
            valid[tx, rx] = np.isfinite(best_corr)
            confidence[tx, rx] = xcorr_peak_quality(best_corr, ratio)
    return {
        "delay_s": delay,
        "valid_mask": valid,
        "peak_correlation": peak_corr,
        "peak_ratio": peak_ratio,
        "peak_lag_s": peak_lag,
        "confidence": confidence,
    }


def bounded_cross_correlation_delay(
    time_data: np.ndarray,
    reference: np.ndarray,
    time_axis_s: np.ndarray,
    *,
    water_tof_s: np.ndarray,
    delta_min_s: np.ndarray | float,
    delta_max_s: np.ndarray | float,
    gate_margin_s: float,
    min_energy: float = 1.0e-18,
) -> dict[str, np.ndarray]:
    """Bounded normalized xcorr around the direct water arrival."""

    water = np.asarray(water_tof_s, dtype=float)
    delta_min = np.asarray(delta_min_s, dtype=float)
    delta_max = np.asarray(delta_max_s, dtype=float)
    return normalized_cross_correlation_delay(
        time_data,
        reference,
        time_axis_s,
        min_lag_s=delta_min - float(gate_margin_s),
        max_lag_s=delta_max + float(gate_margin_s),
        signal_gate_start_s=water + delta_min - float(gate_margin_s),
        signal_gate_end_s=water + delta_max + float(gate_margin_s),
        reference_gate_start_s=water - float(gate_margin_s),
        reference_gate_end_s=water + float(gate_margin_s),
        min_energy=min_energy,
    )


def xcorr_peak_quality(
    peak_correlation: np.ndarray | float,
    peak_ratio: np.ndarray | float,
    *,
    min_peak_correlation: float = 0.2,
    good_peak_correlation: float = 0.65,
    min_peak_ratio: float = 1.02,
    good_peak_ratio: float = 1.35,
) -> np.ndarray | float:
    """Map xcorr peak sharpness metrics to [0, 1] confidence."""

    corr = np.asarray(peak_correlation, dtype=float)
    ratio = np.asarray(peak_ratio, dtype=float)
    corr_score = np.clip((corr - min_peak_correlation) / max(good_peak_correlation - min_peak_correlation, 1.0e-12), 0.0, 1.0)
    ratio_score = np.clip((ratio - min_peak_ratio) / max(good_peak_ratio - min_peak_ratio, 1.0e-12), 0.0, 1.0)
    score = np.sqrt(corr_score * ratio_score)
    if np.isscalar(peak_correlation) and np.isscalar(peak_ratio):
        return float(score)
    return score


def _time_step(time_axis_s: np.ndarray) -> float:
    if time_axis_s.size < 2:
        raise ValueError("time_axis_s must contain at least two samples")
    diffs = np.diff(time_axis_s)
    dt = float(np.median(diffs))
    if dt <= 0.0:
        raise ValueError("time_axis_s must be increasing")
    return dt


def _validate_time_data(time_data: np.ndarray, time_axis_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signals = np.asarray(time_data, dtype=float)
    times = np.asarray(time_axis_s, dtype=float).reshape(-1)
    if signals.ndim != 3:
        raise ValueError("time_data must have shape [n_tx, n_rx, n_time]")
    if signals.shape[-1] != times.size:
        raise ValueError("time_axis_s length must match time_data last dimension")
    return signals, times


def _demean(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    mean = float(np.mean(values[finite]))
    return np.where(finite, values - mean, 0.0)


def _arrival_windows(
    shape: tuple[int, int],
    times: np.ndarray,
    expected_tof_s: np.ndarray | None,
    window_start_s: np.ndarray | float | None,
    window_end_s: np.ndarray | float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if expected_tof_s is None:
        default_center = 0.5 * (float(times[0]) + float(times[-1]))
    else:
        expected = np.asarray(expected_tof_s, dtype=float)
        default_center = expected
    default_width = max(8.0 * _time_step(times), 0.05 * (float(times[-1]) - float(times[0])))
    starts = _broadcast_or_default(window_start_s, shape, default_center - default_width)
    ends = _broadcast_or_default(window_end_s, shape, default_center + default_width)
    starts = np.maximum(starts, float(times[0]))
    ends = np.minimum(ends, float(times[-1]))
    bad = ends <= starts
    ends[bad] = starts[bad] + _time_step(times)
    return starts, ends


def _broadcast_or_default(value: np.ndarray | float | None, shape: tuple[int, int], default: np.ndarray | float) -> np.ndarray:
    if value is None:
        array = np.asarray(default, dtype=float)
    else:
        array = np.asarray(value, dtype=float)
    if array.shape == ():
        return np.full(shape, float(array), dtype=float)
    return np.broadcast_to(array, shape).astype(float, copy=True)


def _window_indices(times: np.ndarray, start: float, end: float) -> np.ndarray:
    return np.flatnonzero((times >= float(start)) & (times <= float(end)))


def _apply_time_gate(signal: np.ndarray, times: np.ndarray, start: float, end: float) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    out = np.zeros_like(values)
    keep = (times >= float(start)) & (times <= float(end)) & np.isfinite(values)
    if not np.any(keep):
        return out
    window = np.hanning(int(np.sum(keep)))
    if window.size <= 2:
        window = np.ones_like(window)
    out[keep] = values[keep] * window
    return out


def _gated_demeaned_segment(signal: np.ndarray, times: np.ndarray, start: float, end: float) -> tuple[int, np.ndarray]:
    values = np.asarray(signal, dtype=float)
    keep = np.flatnonzero((times >= float(start)) & (times <= float(end)) & np.isfinite(values))
    if keep.size == 0:
        return 0, np.asarray([], dtype=float)
    segment = values[keep].astype(float, copy=True)
    window = np.hanning(segment.size)
    if window.size <= 2:
        window = np.ones_like(window)
    segment *= window
    segment = _demean(segment)
    return int(keep[0]), segment


def _aic_curve(trace: np.ndarray) -> np.ndarray:
    values = np.asarray(trace, dtype=float)
    n = values.size
    if n < 8:
        return np.asarray([], dtype=float)
    eps = 1.0e-18
    score = np.full(n, np.nan, dtype=float)
    for idx in range(2, n - 2):
        left = values[:idx]
        right = values[idx:]
        left_var = max(float(np.var(left)), eps)
        right_var = max(float(np.var(right)), eps)
        score[idx] = idx * np.log(left_var) + (n - idx - 1) * np.log(right_var)
    return score


def _envelope(trace: np.ndarray) -> np.ndarray:
    values = np.asarray(trace, dtype=float)
    try:
        from scipy.signal import hilbert
    except ModuleNotFoundError:
        return np.abs(values)
    return np.abs(hilbert(values))


def _overlap_for_lag(signal: np.ndarray, ref: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        return signal[lag:], ref[: signal.size - lag]
    return signal[: signal.size + lag], ref[-lag:]


def _overlap_for_segment_lag(
    signal: np.ndarray,
    signal_start: int,
    ref: np.ndarray,
    ref_start: int,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return overlapping samples for lag using original trace indices.

    `lag` follows the project convention used by full-trace xcorr:
    positive lag means the object signal is delayed relative to the reference.
    """

    sig0 = int(signal_start)
    sig1 = sig0 + int(signal.size)
    ref0 = int(ref_start)
    ref1 = ref0 + int(ref.size)
    start_ref = max(ref0, sig0 - int(lag))
    end_ref = min(ref1, sig1 - int(lag))
    if end_ref <= start_ref:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    start_sig = start_ref + int(lag)
    end_sig = end_ref + int(lag)
    return signal[start_sig - sig0 : end_sig - sig0], ref[start_ref - ref0 : end_ref - ref0]


def _peak_ratio(corr: np.ndarray, best_index: int, *, exclusion: int = 2) -> float:
    finite = np.asarray(corr, dtype=float)
    if finite.size == 0 or not np.isfinite(finite[best_index]):
        return 1.0
    masked = finite.copy()
    lo = max(0, best_index - exclusion)
    hi = min(masked.size, best_index + exclusion + 1)
    masked[lo:hi] = np.nan
    second = float(np.nanmax(masked)) if np.any(np.isfinite(masked)) else 0.0
    peak = float(finite[best_index])
    return float(peak / max(abs(second), 1.0e-12))
