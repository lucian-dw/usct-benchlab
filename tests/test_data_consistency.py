from __future__ import annotations

import pytest

from usctbench.metrics.data_consistency import baseline_improvement, residual_metrics


def test_residual_metrics_with_mask():
    metrics = residual_metrics(
        predicted=[1.0, 2.0, 30.0],
        observed=[1.0, 4.0, 0.0],
        mask=[True, True, False],
    )

    assert metrics["data_residual_norm"] == pytest.approx(2.0)
    assert metrics["data_num_samples"] == 2.0
    assert metrics["data_relative_residual"] == pytest.approx(2.0 / (17.0**0.5))


def test_baseline_improvement_reports_relative_gain():
    metrics = baseline_improvement(10.0, 6.0)

    assert metrics["baseline_absolute_improvement"] == 4.0
    assert metrics["baseline_relative_improvement"] == pytest.approx(0.4)
    assert metrics["baseline_improved"] is True

