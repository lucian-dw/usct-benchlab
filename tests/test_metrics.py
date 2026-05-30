from __future__ import annotations

import math

import numpy as np
import pytest

from usctbench.metrics.image import compute_image_metrics


def test_image_metrics_are_zero_or_perfect_for_identical_arrays():
    target = np.array([[1.0, 2.0], [3.0, 4.0]])
    metrics = compute_image_metrics(target, target)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["nrmse"] == 0.0
    assert math.isinf(metrics["psnr"])
    assert metrics["ssim"] == pytest.approx(1.0)


def test_image_metrics_respect_mask():
    prediction = np.array([[1.0, 10.0], [3.0, 4.0]])
    target = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[True, False], [True, True]])
    metrics = compute_image_metrics(prediction, target, mask=mask)

    assert metrics["rmse"] == 0.0

