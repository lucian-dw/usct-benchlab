import numpy as np
from skimage.metrics import structural_similarity

from usctbench.metrics import compute_image_metrics
import pytest

from usctbench.metrics import compute_regional_image_metrics, non_water_tissue_mask


def test_zero_baseline_has_no_relative_improvement():
    from usctbench.metrics import compute_baseline_improvement_metrics

    truth = np.full((3, 3), 1500.0)
    result = compute_baseline_improvement_metrics(truth + 2, truth, 1500.0)
    assert result["water_relative_rmse_improvement"] is None
    assert result["water_relative_rmse_improvement_status"] == "zero_baseline"
    assert result["water_absolute_rmse_improvement"] == -2.0


def phantom():
    truth = np.full((40, 40), 1500.0)
    truth[8:32, 8:32] = 1450
    truth[15:25, 15:25] = 1530
    truth[18:21, 18:21] = 1500
    return truth


def test_metrics_reject_nonfinite_recon_instead_of_hiding_pixels():
    truth = np.full((12, 12), 1500.0)
    prediction = truth.copy()
    prediction[5, 5] = np.nan
    with pytest.raises(FloatingPointError):
        compute_image_metrics(prediction, truth)
    mask = np.ones(truth.shape, bool)
    mask[5, 5] = False
    assert compute_image_metrics(prediction, truth, mask=mask)["rmse"] == 0


def test_water_artifact_cannot_change_tissue_metrics():
    truth = phantom()
    tissue = non_water_tissue_mask(truth)
    assert tissue[19, 19] and tissue.sum() == 24**2
    clean = truth + np.where(tissue, 2.0, 0.0)
    corrupted = clean + np.where(tissue, 0.0, 100.0)
    a, b = [compute_regional_image_metrics(x, truth) for x in (clean, corrupted)]
    for key in ("rmse", "psnr", "ssim", "tissue_rmse", "tissue_psnr", "tissue_ssim"):
        assert a[key] == b[key]
    assert b["full_image_rmse"] > a["full_image_rmse"]
    assert b["water_background_rmse"] == 100
    assert (
        a["image_evaluation"]["tissue_pixels"]
        == b["image_evaluation"]["tissue_pixels"]
        == tissue.sum()
    )


def test_small_regions_do_not_silently_replace_local_ssim():
    import json
    from usctbench.metrics import compute_image_metrics

    truth = np.full((2, 5), 1500.0)
    scores = compute_image_metrics(truth, truth)
    assert scores["ssim"] is None
    assert scores["ssim_status"] == "unavailable_no_complete_window"
    assert scores["psnr"] is None
    assert scores["psnr_status"] == "perfect_reconstruction"
    json.dumps(scores, allow_nan=False)


def test_range_is_shared_and_constant_tissue_does_not_inflate_psnr():
    truth = np.full((30, 30), 1500.0)
    truth[5:25, 5:25] = 1450
    result = compute_regional_image_metrics(truth + 5, truth)
    assert result["psnr"] == pytest.approx(20)
    assert result["image_evaluation"]["data_range_mps"] == 50
    assert result["psnr"] == result["full_image_psnr"]


def test_missing_or_empty_tissue_has_no_fabricated_primary_score():
    truth = np.full((20, 20), 1500.0)
    for target, status in ((truth, "empty_tissue"), (None, "ground_truth_unavailable")):
        result = compute_regional_image_metrics(truth, target)
        assert result["rmse"] is result["psnr"] is result["ssim"] is None
        assert result["water_reconstruction_rmse"] is None
        assert result["image_evaluation"]["status"] == status
    full = compute_regional_image_metrics(truth, truth, primary_region="full_image")
    assert full["rmse"] == 0


def test_nonfinite_output_in_water_is_not_hidden_by_primary_mask():
    truth = phantom()
    pred = truth.copy()
    pred[0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        compute_regional_image_metrics(pred, truth)


def test_compute_image_metrics_uses_standard_ssim():
    truth = np.arange(25, dtype=float).reshape(5, 5)
    prediction = truth.copy()
    prediction[2, 2] += 1.0

    metrics = compute_image_metrics(prediction, truth)

    assert np.isclose(
        metrics["ssim"],
        structural_similarity(truth, prediction, data_range=24.0, win_size=5),
    )
    assert "global_ssim" in metrics
