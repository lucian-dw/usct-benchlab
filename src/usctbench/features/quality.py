"""Feature extraction orchestration and quality summaries."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from usctbench.features.amplitude import log_amplitude_ratio_from_frequency, log_amplitude_ratio_from_time
from usctbench.features.phase_delay import frequency_response_from_time, gated_phase_slope_delay
from usctbench.features.tof import (
    aic_first_arrival_delta,
    bounded_cross_correlation_delay,
    cross_correlation_delay,
    envelope_first_arrival_delta,
    first_arrival_tof,
    peak_tof,
)
from usctbench.io.hdf5 import read_case_hdf5, write_case_hdf5
from usctbench.provenance import MeasurementProvenance, stamp_measurement_metadata
from usctbench.schema import MeasurementSpec, USCTCase
from usctbench.viz.preview import write_preview_png


LOCAL_SINOGRAM_SMOOTH_WEIGHT_FLOOR = 0.65


def extract_wavefield_features(
    case: USCTCase | str | Path,
    *,
    out: str | Path | None = None,
    method: str = "robust_fusion",
    min_phase_frequencies: int = 3,
    speed_bounds_mps: tuple[float, float] = (1300.0, 1700.0),
) -> tuple[USCTCase, dict[str, Any]]:
    """Extract ToF, phase-delay, amplitude, masks, and QC from a wavefield case."""

    wave_case = read_case_hdf5(case) if isinstance(case, (str, Path)) else case
    if wave_case.measurement.time_data is None:
        raise ValueError("wavefield feature extraction requires measurement.time_data")
    if wave_case.measurement.water_reference is None:
        raise ValueError("wavefield feature extraction requires measurement.water_reference")
    if wave_case.measurement.time_axis_s is None:
        raise ValueError("wavefield feature extraction requires measurement.time_axis_s")

    method = _normalize_method(method)
    time_data = np.asarray(wave_case.measurement.time_data, dtype=float)
    reference = np.asarray(wave_case.measurement.water_reference, dtype=float)
    time_axis = np.asarray(wave_case.measurement.time_axis_s, dtype=float)
    dt = _time_step(time_axis)
    c_min, c_max = _speed_bounds(speed_bounds_mps)
    c0 = float(wave_case.metadata.get("reference_sound_speed_mps", wave_case.metadata.get("simulation_metadata", {}).get("reference_sound_speed_mps", 1500.0)))
    distances = _pairwise_distance(wave_case)
    geometry_water_tof = distances / c0
    measured_water_tof, measured_water_valid = peak_tof(reference, time_axis)
    water_tof = _aligned_water_tof(measured_water_tof, measured_water_valid, geometry_water_tof)
    delta_min, delta_max = _physical_delay_bounds(distances, c0, c_min, c_max)
    source_peak = float(wave_case.metadata.get("simulation_metadata", {}).get("source_peak_frequency_hz", 250000.0))
    gate_margin_s = max(12.0 * dt, 2.0 / source_peak if source_peak > 0.0 else 1.0e-6, 5.0e-7)
    agreement_tolerance_s = max(8.0 * dt, 5.0e-7)
    physical_margin_s = max(5.0 * dt, 2.5e-7)
    reciprocity_tolerance_s = max(12.0 * dt, 7.5e-7)
    outlier_tolerance_s = max(20.0 * dt, 1.5e-6)

    # Keep unbounded xcorr only as a diagnostic. It is no longer a formal
    # solver feature because it can lock on late scattering/reflections.
    unbounded_xcorr, unbounded_valid = cross_correlation_delay(time_data, reference, time_axis)
    bounded = bounded_cross_correlation_delay(
        time_data,
        reference,
        time_axis,
        water_tof_s=water_tof,
        delta_min_s=delta_min,
        delta_max_s=delta_max,
        gate_margin_s=gate_margin_s,
    )
    xcorr_delay = np.asarray(bounded["delay_s"], dtype=float)
    xcorr_valid = np.asarray(bounded["valid_mask"], dtype=bool)
    peak_correlation = np.asarray(bounded["peak_correlation"], dtype=float)
    peak_corr_quality = np.clip((peak_correlation - 0.65) / max(0.95 - 0.65, 1.0e-12), 0.0, 1.0)
    xcorr_confidence = np.maximum(np.asarray(bounded["confidence"], dtype=float), 0.75 * peak_corr_quality)

    aic_delta, aic_valid, aic_confidence = aic_first_arrival_delta(
        time_data,
        reference,
        time_axis,
        water_tof_s=water_tof,
        delta_min_s=delta_min,
        delta_max_s=delta_max,
        gate_margin_s=gate_margin_s,
    )
    envelope_delta, envelope_valid, envelope_confidence = envelope_first_arrival_delta(
        time_data,
        reference,
        time_axis,
        water_tof_s=water_tof,
        delta_min_s=delta_min,
        delta_max_s=delta_max,
        gate_margin_s=gate_margin_s,
    )
    frequencies = wave_case.measurement.frequencies_hz
    if frequencies is None:
        frequencies = np.asarray(wave_case.metadata.get("simulation_metadata", {}).get("frequencies_hz", []), dtype=float)
    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
    freq_data = wave_case.measurement.freq_data
    if frequencies.size:
        if freq_data is None:
            freq_data = frequency_response_from_time(time_data, time_axis, frequencies)
        reference_freq = frequency_response_from_time(reference, time_axis, frequencies)
        phase_delay, phase_rms, phase_valid, phase_confidence = gated_phase_slope_delay(
            time_data,
            reference,
            time_axis,
            frequencies,
            water_tof_s=water_tof,
            delta_min_s=delta_min,
            delta_max_s=delta_max,
            gate_margin_s=gate_margin_s,
            min_frequencies=min_phase_frequencies,
        )
        log_amp = log_amplitude_ratio_from_frequency(freq_data, reference_freq)
        log_amp_for_solver = np.nanmean(log_amp, axis=0)
    else:
        phase_delay = np.full(time_data.shape[:2], np.nan, dtype=float)
        phase_rms = np.full(time_data.shape[:2], np.inf, dtype=float)
        phase_valid = np.zeros(time_data.shape[:2], dtype=bool)
        phase_confidence = np.zeros(time_data.shape[:2], dtype=float)
        log_amp_for_solver = log_amplitude_ratio_from_time(time_data, reference)

    non_self = _non_self_mask(wave_case)
    amplitude_valid, amplitude_quality = _amplitude_quality(time_data, reference, non_self)
    physical_xcorr = _within_bounds(xcorr_delay, delta_min, delta_max, physical_margin_s)
    physical_aic = _within_bounds(aic_delta, delta_min, delta_max, physical_margin_s)
    physical_envelope = _within_bounds(envelope_delta, delta_min, delta_max, physical_margin_s)
    aic_envelope_agree = (
        np.isfinite(aic_delta)
        & np.isfinite(envelope_delta)
        & (np.abs(aic_delta - envelope_delta) <= agreement_tolerance_s)
    )
    use_aic_first = aic_valid & physical_aic & (aic_envelope_agree | ~envelope_valid)
    use_envelope_first = ~use_aic_first & envelope_valid & physical_envelope
    first_delta = np.where(use_aic_first, aic_delta, np.where(use_envelope_first, envelope_delta, np.nan))
    first_valid = use_aic_first | use_envelope_first
    first_confidence = np.where(use_aic_first, aic_confidence, np.where(use_envelope_first, envelope_confidence, 0.0))
    physical_first = _within_bounds(first_delta, delta_min, delta_max, physical_margin_s)
    physical_phase = _within_bounds(phase_delay, delta_min, delta_max, physical_margin_s)
    xcorr_quality_valid = xcorr_valid & (xcorr_confidence >= 0.15) & (peak_correlation >= 0.2)

    fusion = _robust_fusion(
        candidates=[xcorr_delay, first_delta, phase_delay],
        valids=[xcorr_valid & physical_xcorr, first_valid & physical_first, phase_valid & physical_phase],
        agreement_tolerance_s=agreement_tolerance_s,
    )
    reciprocity_good, reciprocity_rmse, reciprocity_bad_fraction = _reciprocity_delay_quality(
        fusion["selected_delta"],
        fusion["candidate_agreement"],
        tolerance_s=reciprocity_tolerance_s,
    )
    local_good, local_outlier_fraction = _sinogram_outlier_mask(
        fusion["selected_delta"],
        fusion["candidate_agreement"] & non_self,
        tolerance_s=outlier_tolerance_s,
    )
    local_smooth_quality, local_smooth_mad = _sinogram_local_smooth_quality(
        fusion["selected_delta"],
        fusion["candidate_agreement"] & non_self,
    )

    robust_valid = (
        non_self
        & amplitude_valid
        & xcorr_quality_valid
        & fusion["candidate_agreement"]
        & _within_bounds(fusion["selected_delta"], delta_min, delta_max, physical_margin_s)
        & reciprocity_good
        & local_good
    )
    robust_delta = np.where(robust_valid, fusion["selected_delta"], np.nan)
    feature_quality = _feature_quality_map(
        valid_mask=robust_valid,
        xcorr_confidence=xcorr_confidence,
        first_confidence=first_confidence,
        phase_confidence=phase_confidence,
        amplitude_quality=amplitude_quality,
        candidate_agreement=fusion["candidate_agreement"],
        reciprocity_good=reciprocity_good,
        local_good=local_good,
        local_smooth_quality=local_smooth_quality,
    )
    ray_weights = np.where(robust_valid, feature_quality, 0.0)

    if method == "xcorr_bounded":
        selected_delta = np.where(non_self & amplitude_valid & xcorr_quality_valid & physical_xcorr, xcorr_delay, np.nan)
        selected_valid = np.isfinite(selected_delta)
        selected_source = "bounded_normalized_xcorr_tof_delay"
        selected_weights = np.where(selected_valid, np.clip(xcorr_confidence * amplitude_quality, 0.0, 1.0), 0.0)
    elif method == "first_arrival_aic":
        selected_delta = np.where(non_self & amplitude_valid & first_valid & physical_first, first_delta, np.nan)
        selected_valid = np.isfinite(selected_delta)
        selected_source = "aic_first_arrival_delay"
        selected_weights = np.where(selected_valid, np.clip(first_confidence * amplitude_quality, 0.0, 1.0), 0.0)
    elif method == "phase_slope_gated":
        selected_delta = np.where(non_self & amplitude_valid & phase_valid & physical_phase, phase_delay, np.nan)
        selected_valid = np.isfinite(selected_delta)
        selected_source = "gated_phase_slope_delay"
        selected_weights = np.where(selected_valid, np.clip(phase_confidence * amplitude_quality, 0.0, 1.0), 0.0)
    else:
        selected_delta = robust_delta
        selected_valid = robust_valid
        selected_source = "robust_fusion"
        selected_weights = ray_weights

    selected_delta = np.where(selected_valid, selected_delta, np.nan)
    qc = _feature_qc(
        selected_delta=selected_delta,
        selected_valid=selected_valid,
        non_self=non_self,
        delta_min=delta_min,
        delta_max=delta_max,
        xcorr_delay=xcorr_delay,
        xcorr_valid=xcorr_valid,
        xcorr_confidence=xcorr_confidence,
        first_delta=first_delta,
        first_valid=first_valid,
        phase_delay=phase_delay,
        phase_valid=phase_valid,
        phase_rms=phase_rms,
        candidate_agreement=fusion["candidate_agreement"],
        reciprocity_rmse=reciprocity_rmse,
        reciprocity_bad_fraction=reciprocity_bad_fraction,
        local_outlier_fraction=local_outlier_fraction,
        local_smooth_quality=local_smooth_quality,
        local_smooth_mad_s=local_smooth_mad,
        log_amp=log_amp_for_solver,
        unbounded_xcorr=unbounded_xcorr,
        unbounded_valid=unbounded_valid,
    )
    provenance = wave_case.metadata.get("measurement_provenance", MeasurementProvenance.SELF_SIMULATED_KWAVE_WAVEFIELD.value)
    metadata = stamp_measurement_metadata(
        wave_case.metadata,
        measurement_provenance=provenance,
        benchmark_type=str(wave_case.metadata.get("benchmark_type", provenance)),
        forward_model=str(wave_case.metadata.get("forward_model", "wavefield_derived_features")),
        feature_source=selected_source,
        uses_complex_wavefield=freq_data is not None,
        simulation_qc_passed=wave_case.metadata.get("simulation_qc_passed"),
        extra={
            "measurement_domain": "features",
            "feature_qc_passed": bool(qc["passed"]),
            "feature_failed_qc": not bool(qc["passed"]),
            "feature_extraction": {
                "method": method,
                "selected_delta_tof": selected_source,
                "single_frequency_phase_allowed": False,
                "water_reference_handling": "required",
                "bounded_xcorr_required": True,
                "unbounded_xcorr_formal_feature_allowed": False,
                "local_sinogram_smooth_quality_used": True,
                "local_sinogram_smooth_weight_floor": LOCAL_SINOGRAM_SMOOTH_WEIGHT_FLOOR,
                "speed_bounds_mps": [c_min, c_max],
                "gate_margin_s": gate_margin_s,
                "agreement_tolerance_s": agreement_tolerance_s,
            },
            "feature_qc": qc,
        },
    )
    feature_case = wave_case.model_copy(
        update={
            "case_id": f"{wave_case.case_id}_features",
            "measurement": MeasurementSpec(
                domain="features",
                frequencies_hz=frequencies if frequencies.size else None,
                freq_data=freq_data,
                time_data=time_data,
                water_reference=reference,
                source_wavelet=wave_case.measurement.source_wavelet,
                time_axis_s=time_axis,
                tof_s=water_tof + selected_delta,
                delta_tof_s=selected_delta,
                tof_first_arrival_s=first_delta,
                tof_xcorr_s=xcorr_delay,
                phase_slope_delay_s=phase_delay,
                log_amp=log_amp_for_solver,
                valid_mask=selected_valid,
                feature_quality=np.where(selected_valid, feature_quality, 0.0),
                ray_weights=selected_weights,
            ),
            "metadata": metadata,
        }
    )
    if out is not None:
        out_path = Path(out)
        write_case_hdf5(feature_case, out_path)
        _write_feature_artifacts(
            feature_case,
            qc,
            out_path.parent,
            xcorr_delta=xcorr_delay,
            phase_delta=phase_delay,
            first_delta=first_delta,
            selected_delta=selected_delta,
            outlier_mask=~local_good & non_self,
        )
    return feature_case, qc


def _normalize_method(method: str) -> str:
    normalized = str(method).replace("-", "_").lower()
    aliases = {
        "all": "robust_fusion",
        "xcorr": "xcorr_bounded",
        "phase_slope": "phase_slope_gated",
        "phase": "phase_slope_gated",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"xcorr_bounded", "first_arrival_aic", "phase_slope_gated", "robust_fusion"}
    if normalized not in allowed:
        raise ValueError(f"unknown feature extraction method {method!r}; expected one of {sorted(allowed)}")
    return normalized


def _non_self_mask(case: USCTCase) -> np.ndarray:
    shape = (case.geometry.tx_pos_m.shape[0], case.geometry.rx_pos_m.shape[0])
    mask = np.ones(shape, dtype=bool)
    if shape[0] == shape[1]:
        mask &= ~np.eye(shape[0], dtype=bool)
    return mask


def _feature_quality_map(
    *,
    valid_mask: np.ndarray,
    xcorr_confidence: np.ndarray,
    first_confidence: np.ndarray,
    phase_confidence: np.ndarray,
    amplitude_quality: np.ndarray,
    candidate_agreement: np.ndarray,
    reciprocity_good: np.ndarray,
    local_good: np.ndarray,
    local_smooth_quality: np.ndarray,
) -> np.ndarray:
    method_quality = np.nanmax(np.stack([xcorr_confidence, first_confidence, phase_confidence]), axis=0)
    quality = np.clip(0.45 * xcorr_confidence + 0.25 * method_quality + 0.2 * amplitude_quality + 0.1 * candidate_agreement.astype(float), 0.0, 1.0)
    # Keep coverage intact but softly down-weight rays whose selected delay is
    # locally inconsistent with neighboring tx/rx entries in the sinogram.
    local_factor = LOCAL_SINOGRAM_SMOOTH_WEIGHT_FLOOR + (1.0 - LOCAL_SINOGRAM_SMOOTH_WEIGHT_FLOOR) * np.clip(local_smooth_quality, 0.0, 1.0)
    quality = quality * local_factor
    quality = np.where(reciprocity_good & local_good & valid_mask, quality, 0.0)
    return np.clip(quality, 0.0, 1.0)


def _feature_qc(
    *,
    selected_delta: np.ndarray,
    selected_valid: np.ndarray,
    non_self: np.ndarray,
    delta_min: np.ndarray,
    delta_max: np.ndarray,
    xcorr_delay: np.ndarray,
    xcorr_valid: np.ndarray,
    xcorr_confidence: np.ndarray,
    first_delta: np.ndarray,
    first_valid: np.ndarray,
    phase_delay: np.ndarray,
    phase_valid: np.ndarray,
    phase_rms: np.ndarray,
    candidate_agreement: np.ndarray,
    reciprocity_rmse: float,
    reciprocity_bad_fraction: float,
    local_outlier_fraction: float,
    local_smooth_quality: np.ndarray,
    local_smooth_mad_s: float,
    log_amp: np.ndarray,
    unbounded_xcorr: np.ndarray,
    unbounded_valid: np.ndarray,
) -> dict[str, Any]:
    denominator = max(1, int(np.sum(non_self)))
    selected_valid_fraction = float(np.sum(selected_valid & non_self) / denominator)
    physical_violation = non_self & np.isfinite(selected_delta) & ~((selected_delta >= delta_min) & (selected_delta <= delta_max))
    xcorr_low_conf = non_self & xcorr_valid & (xcorr_confidence < 0.15)
    first_xcorr_disagreement = _disagreement_mask(first_delta, xcorr_delay, first_valid & xcorr_valid & non_self)
    phase_xcorr_disagreement = _disagreement_mask(phase_delay, xcorr_delay, phase_valid & xcorr_valid & non_self)
    finite_log = np.asarray(log_amp, dtype=float)
    finite_log = finite_log[np.isfinite(finite_log)]
    dynamic_range = float(np.nanmax(finite_log) - np.nanmin(finite_log)) if finite_log.size else 0.0
    phase_valid_fraction = float(np.sum(phase_valid & non_self) / denominator)
    candidate_agreement_fraction = float(np.sum(candidate_agreement & non_self) / denominator)
    unbounded_outlier = _rmse(unbounded_xcorr, xcorr_delay, unbounded_valid & xcorr_valid & non_self)
    checks = {
        "selected_valid_fraction >= 0.45": selected_valid_fraction >= 0.45,
        "physical_bound_violation_fraction <= 0.02": float(np.sum(physical_violation) / denominator) <= 0.02,
        "candidate_agreement_fraction >= 0.35": candidate_agreement_fraction >= 0.35,
        "reciprocity_bad_fraction <= 0.25": float(reciprocity_bad_fraction) <= 0.25,
        "sinogram_median_filter_outlier_fraction <= 0.35": float(local_outlier_fraction) <= 0.35,
    }
    fail_reasons = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not fail_reasons,
        "fail_reasons": fail_reasons,
        "tof_valid_fraction": selected_valid_fraction,
        "selected_valid_fraction": selected_valid_fraction,
        "tof_outlier_fraction": float(local_outlier_fraction),
        "physical_bound_violation_fraction": float(np.sum(physical_violation) / denominator),
        "xcorr_low_confidence_fraction": float(np.sum(xcorr_low_conf) / denominator),
        "first_xcorr_disagreement_fraction": float(np.sum(first_xcorr_disagreement) / denominator),
        "phase_xcorr_disagreement_fraction": float(np.sum(phase_xcorr_disagreement) / denominator),
        "first_arrival_vs_xcorr_rmse_s": _rmse(first_delta, xcorr_delay, first_valid & xcorr_valid & non_self),
        "phase_slope_vs_xcorr_rmse_s": _rmse(phase_delay, xcorr_delay, phase_valid & xcorr_valid & non_self),
        "unbounded_vs_bounded_xcorr_rmse_s": unbounded_outlier,
        "reciprocity_delay_rmse": float(reciprocity_rmse),
        "reciprocity_bad_fraction": float(reciprocity_bad_fraction),
        "sinogram_median_filter_outlier_fraction": float(local_outlier_fraction),
        "local_sinogram_smooth_quality_mean": _masked_mean(local_smooth_quality, selected_valid & non_self),
        "local_sinogram_smooth_quality_p10": _masked_percentile(local_smooth_quality, selected_valid & non_self, 10.0),
        "local_sinogram_smooth_mad_s": float(local_smooth_mad_s),
        "candidate_agreement_fraction": candidate_agreement_fraction,
        "log_amp_dynamic_range": dynamic_range,
        "phase_unwrap_failure_fraction": float(1.0 - phase_valid_fraction),
        "phase_slope_valid_fraction": phase_valid_fraction,
        "bad_tx_rx_fraction": float(1.0 - selected_valid_fraction),
    }


def _robust_fusion(
    *,
    candidates: list[np.ndarray],
    valids: list[np.ndarray],
    agreement_tolerance_s: float,
) -> dict[str, np.ndarray]:
    stack = np.stack([np.asarray(candidate, dtype=float) for candidate in candidates], axis=0)
    valid_stack = np.stack([np.asarray(valid, dtype=bool) & np.isfinite(candidate) for candidate, valid in zip(candidates, valids, strict=True)], axis=0)
    values = np.where(valid_stack, stack, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        selected = np.nanmedian(values, axis=0)
    count = np.sum(valid_stack, axis=0)
    agreement = np.zeros(stack.shape[1:], dtype=bool)
    for i in range(stack.shape[0]):
        for j in range(i + 1, stack.shape[0]):
            pair = valid_stack[i] & valid_stack[j] & (np.abs(stack[i] - stack[j]) <= float(agreement_tolerance_s))
            agreement |= pair
    agreement &= count >= 2
    selected = np.where(agreement, selected, np.nan)
    return {"selected_delta": selected, "candidate_count": count, "candidate_agreement": agreement}


def _amplitude_quality(time_data: np.ndarray, reference: np.ndarray, non_self: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal_energy = np.nansum(np.asarray(time_data, dtype=float) ** 2, axis=-1)
    ref_energy = np.nansum(np.asarray(reference, dtype=float) ** 2, axis=-1)
    signal_median = _positive_median(signal_energy[non_self])
    ref_median = _positive_median(ref_energy[non_self])
    valid = (signal_energy >= max(1.0e-18, signal_median * 1.0e-8)) & (ref_energy >= max(1.0e-18, ref_median * 1.0e-8))
    quality = np.sqrt(
        np.clip(signal_energy / max(signal_median, 1.0e-30), 0.0, 1.0)
        * np.clip(ref_energy / max(ref_median, 1.0e-30), 0.0, 1.0)
    )
    return valid & np.isfinite(signal_energy) & np.isfinite(ref_energy), np.clip(quality, 0.0, 1.0)


def _positive_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite) & (finite > 0.0)]
    return float(np.median(finite)) if finite.size else 0.0


def _within_bounds(values: np.ndarray, lower: np.ndarray, upper: np.ndarray, margin: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.isfinite(array) & (array >= np.asarray(lower) - float(margin)) & (array <= np.asarray(upper) + float(margin))


def _disagreement_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray, *, tolerance_s: float = 5.0e-7) -> np.ndarray:
    finite = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    out = np.zeros_like(finite, dtype=bool)
    out[finite] = np.abs(np.asarray(a, dtype=float)[finite] - np.asarray(b, dtype=float)[finite]) > float(tolerance_s)
    return out


def _reciprocity_delay_quality(delta: np.ndarray, valid: np.ndarray, *, tolerance_s: float) -> tuple[np.ndarray, float, float]:
    good = np.ones_like(valid, dtype=bool)
    if delta.shape[0] != delta.shape[1]:
        return good, 0.0, 0.0
    mutual = valid & valid.T & np.isfinite(delta) & np.isfinite(delta.T)
    if not np.any(mutual):
        return good, float("nan"), 0.0
    diff = np.asarray(delta, dtype=float) - np.asarray(delta, dtype=float).T
    bad = mutual & (np.abs(diff) > float(tolerance_s))
    good[bad] = False
    return good, float(np.sqrt(np.mean(diff[mutual] ** 2))), float(np.sum(bad) / max(1, int(np.sum(mutual))))


def _sinogram_outlier_mask(delta: np.ndarray, valid: np.ndarray, *, tolerance_s: float) -> tuple[np.ndarray, float]:
    good = np.ones_like(valid, dtype=bool)
    finite = np.asarray(valid, dtype=bool) & np.isfinite(delta)
    if int(np.sum(finite)) < 16:
        return good, 0.0
    values = np.asarray(delta, dtype=float)
    try:
        from scipy.ndimage import median_filter
    except ModuleNotFoundError:
        return good, 0.0
    fill = float(np.nanmedian(values[finite]))
    filled = np.where(finite, values, fill)
    median = median_filter(filled, size=(3, 5), mode="wrap")
    residual = np.abs(values - median)
    mad = float(np.nanmedian(residual[finite]))
    threshold = max(float(tolerance_s), 8.0 * mad)
    bad = finite & (residual > threshold)
    good[bad] = False
    return good, float(np.sum(bad) / max(1, int(np.sum(finite))))


def _sinogram_local_smooth_quality(delta: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float]:
    """Return continuous local sinogram consistency in [0, 1].

    This is deliberately softer than `_sinogram_outlier_mask`: it preserves the
    ray in `valid_mask` but lowers solver weights when a delay disagrees with a
    wrapped local median neighborhood.
    """

    quality = np.ones_like(valid, dtype=float)
    finite = np.asarray(valid, dtype=bool) & np.isfinite(delta)
    if int(np.sum(finite)) < 16:
        return quality, 0.0
    values = np.asarray(delta, dtype=float)
    try:
        from scipy.ndimage import median_filter
    except ModuleNotFoundError:
        return quality, 0.0
    fill = float(np.nanmedian(values[finite]))
    filled = np.where(finite, values, fill)
    median = median_filter(filled, size=(3, 5), mode="wrap")
    residual = np.abs(values - median)
    mad = float(np.nanmedian(residual[finite]))
    scale = max(3.0 * mad, 2.5e-7)
    local = 1.0 / (1.0 + (residual / scale) ** 2)
    quality[finite] = np.clip(local[finite], 0.0, 1.0)
    quality[~finite] = 0.0
    return quality, mad


def _feature_qc_failed(case: USCTCase) -> bool:
    return bool(case.metadata.get("feature_failed_qc", False))


def _rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    finite = np.asarray(mask, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return float("nan")
    diff = np.asarray(a, dtype=float)[finite] - np.asarray(b, dtype=float)[finite]
    return float(np.sqrt(np.mean(diff**2)))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    finite = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)[finite]))


def _masked_percentile(values: np.ndarray, mask: np.ndarray, percentile: float) -> float:
    finite = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float)[finite], float(percentile)))


def _write_feature_artifacts(
    case: USCTCase,
    qc: dict[str, Any],
    out_dir: Path,
    *,
    xcorr_delta: np.ndarray,
    phase_delta: np.ndarray,
    first_delta: np.ndarray,
    selected_delta: np.ndarray,
    outlier_mask: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_preview_png(xcorr_delta, out_dir / "xcorr_delta_tof_sinogram.png")
    write_preview_png(phase_delta, out_dir / "phase_delta_tof_sinogram.png")
    write_preview_png(first_delta, out_dir / "first_arrival_delta_tof_sinogram.png")
    write_preview_png(selected_delta, out_dir / "selected_delta_tof_sinogram.png")
    if case.measurement.log_amp is not None:
        write_preview_png(case.measurement.log_amp, out_dir / "log_amplitude_preview.png")
    if case.measurement.feature_quality is not None:
        write_preview_png(case.measurement.feature_quality, out_dir / "feature_quality_sinogram.png")
        write_preview_png(case.measurement.feature_quality, out_dir / "feature_preview.png")
    write_preview_png(np.asarray(outlier_mask, dtype=float), out_dir / "feature_outlier_mask.png")
    (out_dir / "feature_qc.json").write_text(json.dumps(qc, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "feature_quality.json").write_text(json.dumps(qc, indent=2, sort_keys=True), encoding="utf-8")


def _pairwise_distance(case: USCTCase) -> np.ndarray:
    tx = np.asarray(case.geometry.tx_pos_m, dtype=float)
    rx = np.asarray(case.geometry.rx_pos_m, dtype=float)
    return np.linalg.norm(tx[:, None, :] - rx[None, :, :], axis=-1)


def _physical_delay_bounds(distance_m: np.ndarray, c0: float, c_min: float, c_max: float) -> tuple[np.ndarray, np.ndarray]:
    distance = np.asarray(distance_m, dtype=float)
    lower = distance / float(c_max) - distance / float(c0)
    upper = distance / float(c_min) - distance / float(c0)
    return np.minimum(lower, upper), np.maximum(lower, upper)


def _aligned_water_tof(measured: np.ndarray, valid: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    finite = np.asarray(valid, dtype=bool) & np.isfinite(measured) & np.isfinite(geometry)
    if not np.any(finite):
        return np.asarray(geometry, dtype=float)
    bias = float(np.nanmedian(np.asarray(measured, dtype=float)[finite] - np.asarray(geometry, dtype=float)[finite]))
    aligned = np.asarray(geometry, dtype=float) + bias
    return np.where(finite, measured, aligned)


def _speed_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    c_min, c_max = float(bounds[0]), float(bounds[1])
    if c_min <= 0.0 or c_max <= c_min:
        raise ValueError("speed_bounds_mps must be (positive_min, larger_max)")
    return c_min, c_max


def _time_step(time_axis_s: np.ndarray) -> float:
    times = np.asarray(time_axis_s, dtype=float).reshape(-1)
    if times.size < 2:
        raise ValueError("time_axis_s must contain at least two samples")
    dt = float(np.median(np.diff(times)))
    if dt <= 0.0:
        raise ValueError("time_axis_s must be increasing")
    return dt
