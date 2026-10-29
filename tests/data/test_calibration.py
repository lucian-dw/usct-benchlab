import numpy as np
import pytest

from usctbench.data.calibration import fit_water_source
from usctbench.algorithms.rwave import RayBornAlgorithm
from usctbench.core.schema import AlgorithmConfig, MeasurementSpec, GroundTruthSpec
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.operators.ray_born import RayBornOperator


def test_complex_calibration_known_sources_and_missing_channels():
    rng = np.random.default_rng(127)
    unit = rng.normal(size=(3, 4, 6)) + 1j * rng.normal(size=(3, 4, 6))
    source = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
    water = unit * source[..., None]
    water[..., 0] = np.nan
    train = np.ones(unit.shape, dtype=bool)
    train[..., 2] = False
    water[..., 2] *= 8
    fitted, record = fit_water_source(
        unit, water, train_mask=train, weights=np.ones(unit.shape)
    )
    np.testing.assert_allclose(fitted, source, rtol=1e-14, atol=1e-14)
    assert record["relative_residual"] < 1e-14
    with pytest.raises(ValueError, match="nonzero"):
        fit_water_source(unit * 0, water, train_mask=train)
    with pytest.raises(ValueError, match="matching"):
        fit_water_source(unit, water[0], train_mask=train)


def test_water_scales_jacobian_not_just_background_and_needs_no_truth():
    case = make_sound_speed_case(shape=(8, 8), n_transducers=8, inclusion_mps=1490)
    frequencies = np.array([120e3, 160e3, 200e3])
    source = np.array([2 + 3j, -4 + 2j, 3 - 1j])
    op = RayBornOperator(
        case.grid,
        case.geometry,
        frequencies,
        source_spectrum=np.broadcast_to(source[:, None], (3, 8)),
    )
    case.measurement = MeasurementSpec(
        domain="frequency",
        freq_data=op.predict(case.ground_truth.sound_speed_mps),
        frequencies_hz=frequencies,
        valid_mask=op.valid_pair_mask,
        water_reference=op.background_data(),
    )
    case.ground_truth = GroundTruthSpec()
    cfg = AlgorithmConfig(
        parameters={"variant": "wkb_fixed"},
        run_controls={
            "max_iterations": 4,
            "restore_best_validation": False,
            "objective_rtol": None,
            "update_rtol": None,
        },
        evaluation={"receiver_indices": [1], "frequency_indices": [2]},
    )
    fitted = RayBornAlgorithm().run(case, cfg)
    configured_case = case.model_copy(deep=True)
    configured_case.measurement.source_spectrum = source
    configured = RayBornAlgorithm().run(configured_case, cfg)
    assert fitted.failure_reason is None, fitted.failure_reason
    assert configured.failure_reason is None, configured.failure_reason
    np.testing.assert_allclose(
        fitted.sound_speed_mps, configured.sound_speed_mps, atol=1e-8
    )
    assert fitted.metrics["source_spectrum_assumed_unit"] is False
    assert fitted.metrics["source_calibration"]["relative_residual"] < 1e-14
    changed = case.model_copy(deep=True)
    changed.measurement.freq_data[..., 1] *= 8
    changed.measurement.freq_data[2] *= 11
    isolated = RayBornAlgorithm().run(changed, cfg)
    np.testing.assert_array_equal(fitted.sound_speed_mps, isolated.sound_speed_mps)
