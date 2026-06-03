from __future__ import annotations

import numpy as np
import pytest

from usctbench.algorithms.ray.cgls import StraightRayCGLSAlgorithm
from usctbench.algorithms.ray.sart import StraightRaySARTAlgorithm
from usctbench.algorithms.ray.sirt import StraightRaySIRTAlgorithm
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.schema import AlgorithmConfig, MeasurementSpec, USCTCase


def test_homogeneous_case_returns_reference_speed():
    case = make_sound_speed_case(shape=(16, 16), n_transducers=20, inclusion_mps=1500.0)
    algorithms = [
        (StraightRayCGLSAlgorithm(), AlgorithmConfig(parameters={"iterations": 5, "reference_sound_speed_mps": 1500.0})),
        (StraightRaySIRTAlgorithm(), AlgorithmConfig(parameters={"iterations": 5, "relaxation": 0.3, "reference_sound_speed_mps": 1500.0})),
        (StraightRaySARTAlgorithm(), AlgorithmConfig(parameters={"iterations": 3, "relaxation": 0.2, "reference_sound_speed_mps": 1500.0})),
    ]

    for algorithm, config in algorithms:
        result = algorithm.run(case, config)
        assert result.failure_reason is None
        np.testing.assert_allclose(result.sound_speed_mps, 1500.0, atol=1.0e-9)
        assert result.metrics["data_residual_norm"] == 0.0
        assert max(result.metrics["residual_curve"]) == 0.0


def test_positive_delay_reconstructs_slower_center():
    case = make_sound_speed_case(shape=(18, 18), n_transducers=28, inclusion_mps=1450.0)
    result = StraightRayCGLSAlgorithm().run(
        case,
        AlgorithmConfig(
            parameters={
                "iterations": 35,
                "reference_sound_speed_mps": 1500.0,
                "sound_speed_bounds_mps": [1400.0, 1600.0],
            }
        ),
    )

    assert result.failure_reason is None
    center = result.sound_speed_mps[8:10, 8:10]
    corner = result.sound_speed_mps[:3, :3]
    assert float(np.mean(center)) < 1495.0
    assert float(np.mean(center)) < float(np.mean(corner))
    assert result.metrics["water_improved"] is True
    assert 0.0 <= result.metrics["data_relative_residual"] < 1.0
    assert result.metrics["data_residual_reduction"] > 0.0


def test_sirt_and_sart_reconstruct_slower_center():
    case = make_sound_speed_case(shape=(16, 16), n_transducers=24, inclusion_mps=1450.0)
    configs = [
        (
            StraightRaySIRTAlgorithm(),
            AlgorithmConfig(
                parameters={
                    "iterations": 40,
                    "relaxation": 0.8,
                    "reference_sound_speed_mps": 1500.0,
                    "sound_speed_bounds_mps": [1400.0, 1600.0],
                }
            ),
        ),
        (
            StraightRaySARTAlgorithm(),
            AlgorithmConfig(
                parameters={
                    "iterations": 8,
                    "relaxation": 0.2,
                    "reference_sound_speed_mps": 1500.0,
                    "sound_speed_bounds_mps": [1400.0, 1600.0],
                }
            ),
        ),
    ]

    for algorithm, config in configs:
        result = algorithm.run(case, config)
        center = result.sound_speed_mps[7:9, 7:9]
        corner = result.sound_speed_mps[:3, :3]
        assert result.failure_reason is None
        assert np.isfinite(result.metrics["data_residual_norm"])
        assert np.isfinite(result.metrics["data_relative_residual"])
        assert np.isfinite(result.metrics["data_residual_reduction"])
        assert result.metrics["residual_curve"]
        assert float(np.mean(center)) < 1495.0
        assert float(np.mean(center)) < float(np.mean(corner))


def test_missing_delta_tof_reports_failure():
    case = make_sound_speed_case(shape=(8, 8), n_transducers=10)
    bad_case = USCTCase(
        case_id=case.case_id,
        grid=case.grid,
        geometry=case.geometry,
        measurement=MeasurementSpec(domain="features", log_amp=np.zeros((10, 10))),
        ground_truth=case.ground_truth,
        metadata=case.metadata,
    )
    result = StraightRayCGLSAlgorithm().run(bad_case, AlgorithmConfig(parameters={"iterations": 1}))

    assert result.status == "failed"
    assert "delta_tof_s" in result.failure_reason


def test_cgls_laplacian_regularization_runs_on_synthetic_case():
    case = make_sound_speed_case(shape=(16, 16), n_transducers=16, inclusion_mps=1450.0)
    result = StraightRayCGLSAlgorithm().run(
        case,
        AlgorithmConfig(
            parameters={
                "iterations": 6,
                "reference_sound_speed_mps": 1500.0,
                "regularization": "laplacian",
                "regularization_lambda": 1.0e-4,
            }
        ),
    )

    assert result.status == "success"
    assert result.sound_speed_mps is not None
    assert np.isfinite(result.sound_speed_mps).all()
    assert result.metrics["regularization"] == "laplacian"
    assert result.metrics["regularization_lambda_squared"] == pytest.approx(1.0e-8)


def test_cgls_coverage_preconditioning_and_roi_laplacian_report_diagnostics():
    case = make_sound_speed_case(shape=(14, 14), n_transducers=18, inclusion_mps=1450.0)
    result = StraightRayCGLSAlgorithm().run(
        case,
        AlgorithmConfig(
            parameters={
                "iterations": 8,
                "reference_sound_speed_mps": 1500.0,
                "regularization": "laplacian",
                "regularization_lambda": 1.0e-3,
                "coverage_preconditioning": True,
                "roi_laplacian": True,
                "roi_update_only": True,
                "boundary_band_pixels": 2,
            }
        ),
    )

    assert result.status == "success"
    assert result.metrics["coverage_preconditioning"] is True
    assert result.metrics["roi_laplacian"] is True
    assert result.metrics["roi_update_only"] is True
    assert np.isfinite(result.metrics["boundary_band_rmse"])
    assert "coverage_abs_error_corr" in result.metrics


def test_cgls_huber_irls_runs_on_synthetic_case():
    case = make_sound_speed_case(shape=(12, 12), n_transducers=16, inclusion_mps=1450.0)
    result = StraightRayCGLSAlgorithm().run(
        case,
        AlgorithmConfig(
            parameters={
                "iterations": 4,
                "robust_loss": "huber",
                "irls_iterations": 2,
                "huber_delta_s": 1.0e-6,
                "regularization": "laplacian",
                "regularization_lambda": 1.0e-3,
            }
        ),
    )

    assert result.status == "success"
    assert result.metrics["robust_loss"] == "huber"
    assert result.metrics["irls_iterations"] == 2
    assert result.metrics["data_residual_reduction"] >= 0.0
