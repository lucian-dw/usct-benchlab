from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location("render_class_comparison_panels", Path("scripts/render_class_comparison_panels.py"))
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_filter_cases_by_id = _MODULE._filter_cases_by_id
_format_label = _MODULE._format_label
_metric_subset = _MODULE._metric_subset
_transpose_panels = _MODULE._transpose_panels


class _Case:
    def __init__(self, case_id: str):
        self.case_id = case_id


def test_filter_cases_by_id_preserves_requested_order():
    cases = [_Case("b"), _Case("a"), _Case("c")]

    filtered = _filter_cases_by_id(cases, ["c", "a"])

    assert [case.case_id for case in filtered] == ["c", "a"]


def test_filter_cases_by_id_errors_on_missing_case():
    with pytest.raises(SystemExit, match="requested case ids not found: missing"):
        _filter_cases_by_id([_Case("present")], ["missing"])


def test_fwi_comparison_label_uses_kwave_ground_truth_metrics():
    label = _format_label(
        "fwi_kwave_adapter",
        {
            "rmse": 99.0,
            "ssim": 0.01,
            "kwave_gt_rmse": 17.732,
            "kwave_gt_ssim": 0.70485,
        },
    )

    assert "kWave RMSE=17.7" in label
    assert "kWave SSIM=0.705" in label
    assert "RMSE=99.0" not in label


def test_transpose_panels_renders_algorithms_as_columns():
    panels = [
        [("GT case_a", None, {}), ("alg1", None, {}), ("alg2", None, {})],
        [("GT case_b", None, {}), ("alg1", None, {}), ("alg2", None, {})],
    ]

    transposed = _transpose_panels(panels)

    assert [[cell[0] for cell in col] for col in transposed] == [
        ["GT case_a", "GT case_b"],
        ["alg1", "alg1"],
        ["alg2", "alg2"],
    ]


def test_metric_subset_keeps_fwi_acceptance_metrics():
    subset = _metric_subset(
        {
            "rmse": 32.0,
            "kwave_gt_rmse": 17.7,
            "kwave_gt_ssim": 0.7,
            "kwave_native_psnr": 22.8,
            "kwave_gt_final_relative_rmse_improvement": 0.29,
            "unrelated": 1,
        }
    )

    assert subset["kwave_gt_rmse"] == 17.7
    assert subset["kwave_gt_ssim"] == 0.7
    assert subset["kwave_native_psnr"] == 22.8
    assert subset["kwave_gt_final_relative_rmse_improvement"] == 0.29
    assert "unrelated" not in subset
