"""Instrument source calibration from an independent water-only acquisition."""

from __future__ import annotations

import numpy as np

from usctbench.evaluation.data import residual_statistics


def prepare_source_spectrum(source, n_frequency: int, n_tx: int) -> np.ndarray:
    """Normalize a finite scalar, frequency vector or (frequency, tx) spectrum."""
    spectrum = np.asarray(source, dtype=complex)
    if spectrum.ndim == 1:
        if spectrum.shape != (n_frequency,):
            raise ValueError(
                "source_spectrum frequency count does not match observations"
            )
        spectrum = spectrum[:, None]
    elif spectrum.ndim != 0 and spectrum.shape != (n_frequency, n_tx):
        raise ValueError(
            "source_spectrum must be scalar, (frequency,), or (frequency, tx)"
        )
    if not np.isfinite(spectrum).all():
        raise ValueError("source_spectrum must be finite")
    return np.broadcast_to(spectrum, (n_frequency, n_tx)).copy()


def fit_water_source(
    unit_water_prediction, water_reference, *, train_mask, weights=None
):
    """Fit one complex source coefficient per (frequency, transmitter).

    Minimize sum_rx |q[f,tx] G_water[f,tx,rx] - p_water[f,tx,rx]|^2.
    This is a *multiplicative* calibration of both the forward field and its
    Jacobian, not an additive replacement of the background prediction.
    train_mask explicitly declares usable calibration channels; held-out specimen
    measurements never enter this fit. Only independently acquired reference data may be passed here. No specimen
    measurements or ground-truth labels are accepted. A source-only model does
    not identify arbitrary receiver responses or finite-aperture directivity.
    """
    unit = np.asarray(unit_water_prediction, dtype=complex)
    water = np.asarray(water_reference, dtype=complex)
    if unit.ndim != 3 or water.shape != unit.shape:
        raise ValueError(
            "water and unit predictions must have matching (frequency, tx, rx) shape"
        )
    mask = np.asarray(train_mask)
    if mask.dtype.kind != "b" or mask.shape != water.shape:
        raise ValueError("train_mask must be boolean and match (frequency, tx, rx)")
    precision = (
        np.ones(water.shape)
        if weights is None
        else np.broadcast_to(np.asarray(weights, dtype=float), water.shape)
    )
    if np.any(precision < 0) or not np.isfinite(precision).all():
        raise ValueError("weights must be finite nonnegative precisions")
    active = mask & np.isfinite(water) & (precision > 0)
    precision = np.where(active, precision, 0)
    if not np.all(np.isfinite(unit[active])):
        raise ValueError("nonfinite unit-source prediction on calibration channels")
    u = np.where(active, unit, 0)
    w = np.where(active, water, 0)
    denominator = np.sum(precision * np.abs(u) ** 2, axis=-1)
    if np.any(denominator <= np.finfo(float).tiny) or not np.all(
        np.isfinite(denominator)
    ):
        raise ValueError(
            "each frequency/transmitter needs a nonzero water calibration channel"
        )
    source = np.sum(precision * u.conj() * w, axis=-1) / denominator
    if not np.all(np.isfinite(source)) or np.any(
        np.abs(source) <= np.finfo(float).tiny
    ):
        raise ValueError("water source calibration overflow or no excitation")
    statistics = residual_statistics(
        unit * source[..., None], water, mask=active, weights=precision
    )
    return source, {
        "method": "independent_water_complex_least_squares_per_frequency_transmitter",
        "measurement_scope": "independent_water_acquisition_not_specimen_holdout",
        "calibration_samples": int(active.sum()),
        "minimum_channels_per_source": int(active.sum(axis=-1).min()),
        "relative_residual": statistics["relative_residual"],
    }
