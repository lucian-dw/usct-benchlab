from __future__ import annotations

import numpy as np

import pytest

from usctbench.data.features import extract_frequency_features, log_amplitude_ratio, phase_delay_seconds, valid_amplitude_mask
from usctbench.features.quality import _sinogram_local_smooth_quality


def test_log_amplitude_ratio_uses_reference_convention():
    signal = np.array([2.0 + 0.0j, 0.5 + 0.0j])
    reference = np.array([1.0 + 0.0j, 1.0 + 0.0j])

    values = log_amplitude_ratio(signal, reference)

    np.testing.assert_allclose(values, np.log([2.0, 0.5]))


def test_phase_delay_positive_for_slow_signal_default_convention():
    frequencies = np.array([0.5e6, 0.75e6, 1.0e6])
    delay_s = 2.0e-7
    reference = np.ones((3, 1, 1), dtype=complex)
    signal = np.exp(-1j * 2.0 * np.pi * frequencies[:, None, None] * delay_s)

    estimated = phase_delay_seconds(signal, reference, frequencies)

    np.testing.assert_allclose(estimated, np.array([[delay_s]]), rtol=1.0e-10, atol=1.0e-12)


def test_extract_frequency_features_returns_ray_arrays():
    frequencies = np.array([0.5e6, 0.75e6, 1.0e6])
    reference = np.ones((3, 2, 3), dtype=complex)
    signal = reference * (0.5 * np.exp(-1j * 2.0 * np.pi * frequencies[:, None, None] * 1.0e-7))

    features = extract_frequency_features(signal, reference, frequencies)

    assert features["delta_tof_s"].shape == (2, 3)
    assert features["log_amp"].shape == (3, 2, 3)
    assert features["valid_mask"].shape == (2, 3)
    assert features["feature_quality"] == "ok"
    np.testing.assert_allclose(features["delta_tof_s"], 1.0e-7)
    np.testing.assert_allclose(features["log_amp"], np.log(0.5))


def test_extract_frequency_features_rejects_single_frequency_tof_by_default():
    frequencies = np.array([0.5e6])
    reference = np.ones((1, 2, 3), dtype=complex)
    signal = reference.copy()

    with pytest.raises(ValueError, match="at least three frequencies"):
        extract_frequency_features(signal, reference, frequencies)

    features = extract_frequency_features(signal, reference, frequencies, allow_low_frequency_count=True)
    assert features["feature_quality"] == "low"
    assert not np.any(features["valid_mask"])


def test_valid_amplitude_mask_requires_all_frequency_bins():
    reference = np.ones((3, 1, 1), dtype=complex)
    signal = np.ones((3, 1, 1), dtype=complex)
    signal[1, 0, 0] = 0.0

    mask = valid_amplitude_mask(signal, reference, min_signal_amplitude=0.5)

    assert mask.shape == (1, 1)
    assert mask[0, 0] is np.False_


def test_local_sinogram_smooth_quality_downweights_spike():
    pytest.importorskip("scipy.ndimage")
    delta = np.zeros((9, 9), dtype=float)
    valid = np.ones_like(delta, dtype=bool)
    delta[4, 4] = 2.0e-6

    quality, mad = _sinogram_local_smooth_quality(delta, valid)

    assert quality.shape == delta.shape
    assert mad >= 0.0
    assert quality[4, 4] < 0.2
    assert quality[4, 5] > quality[4, 4]
