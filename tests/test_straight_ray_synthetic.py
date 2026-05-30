from __future__ import annotations

import numpy as np

from usctbench.algorithms.ray.cgls import StraightRayCGLSAlgorithm
from usctbench.algorithms.ray.sart import StraightRaySARTAlgorithm
from usctbench.algorithms.ray.sirt import StraightRaySIRTAlgorithm
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.schema import AlgorithmConfig, MeasurementSpec, USCTCase


def test_homogeneous_case_returns_reference_speed():
    case = make_sound_speed_case(shape=(16, 16), n_transducers=20, inclusion_mps=1500.0)
    result = StraightRayCGLSAlgorithm().run(
        case,
        AlgorithmConfig(parameters={"iterations": 5, "reference_sound_speed_mps": 1500.0}),
    )

    assert result.failure_reason is None
    np.testing.assert_allclose(result.sound_speed_mps, 1500.0, atol=1.0e-9)


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
