from __future__ import annotations

import numpy as np

from usctbench.algorithms.fwi.gradient_check import check_tiny_fwi_gradient
from usctbench.algorithms.fwi.losses import waveform_from_speed


def test_tiny_fwi_gradient_check_passes():
    frequencies = np.array([1.0e5, 1.5e5, 2.0e5])
    truth = np.linspace(1450.0, 1520.0, 8)
    observed = waveform_from_speed(truth, frequencies, spacing_m=1.0e-3)
    model = np.full_like(truth, 1500.0)

    check = check_tiny_fwi_gradient(model, observed, frequencies, spacing_m=1.0e-3)

    assert check["relative_error"] < 1.0e-5

