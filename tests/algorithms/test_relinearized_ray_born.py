"""Nonlinear model acceptance, not just a frozen-Jacobian residual test."""

from usctbench.core.config import prepare_config

import numpy as np
import pytest

from usctbench.algorithms.born import BornAlgorithm
from usctbench.core.schema import AlgorithmConfig, GroundTruthSpec, MeasurementSpec
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.data.pressure import dtft_pressure_contract
from usctbench.operators.ray_born import RayBornForward


def nonlinear_case():
    case = make_sound_speed_case(
        shape=(8, 8), n_transducers=8, inclusion_radius_m=0.002
    )
    case.geometry.tx_pos_m *= 0.25
    case.geometry.rx_pos_m *= 0.25
    yy, xx = np.indices(case.grid.shape)
    speed = 1500 + 8 * np.exp(-((yy - 3) ** 2 + (xx - 4) ** 2) / 5)
    forward = RayBornForward(case.grid, case.geometry, [250e3, 300e3, 350e3])
    linearization = forward.linearize(1 / speed**2)
    case.measurement = MeasurementSpec(
        domain="frequency",
        frequencies_hz=forward.frequencies_hz,
        freq_data=linearization.value,
        source_spectrum=np.ones((3, 8), complex),
        valid_mask=linearization.jacobian.valid_pair_mask,
    )
    case.ground_truth = GroundTruthSpec()
    case.metadata["pressure_contract"] = dtft_pressure_contract()
    return case, forward, linearization


def config(*, max_iterations=3, run_controls=None, evaluation=None, **parameters):
    return AlgorithmConfig(
        diagnostics=True,
        parameters={
            "variant": "wkb_nonlinear",
            "inner_iterations": 4,
            "smooth_sigma": 0.6,
            **parameters,
        },
        run_controls={"max_iterations": max_iterations, **(run_controls or {})},
        evaluation={"receiver_indices": [1]} if evaluation is None else evaluation,
    )


def test_rebuilds_background_and_accepts_only_true_nonlinear_decrease():
    case, _, _ = nonlinear_case()
    result = BornAlgorithm().run(
        case, prepare_config(BornAlgorithm().name, config(), case=case)
    )
    assert result.status in {"completed", "budget_exhausted"}, result.failure_reason
    metrics = result.metrics
    assert metrics["nonlinear_background_updates"] >= 1, metrics
    curve = [row["objective"] for row in metrics["iteration_history"]]
    assert np.all(np.diff(curve) < 0), curve
    assert metrics["data_residual_reduction"] > 0.1
    assert metrics["stopping"]["work"]["background_builds"] >= len(curve)
    assert metrics["variant"] == "wkb_nonlinear"
    assert "rmse" not in metrics


def test_born_caller_transfers_initial_cache_to_gn(monkeypatch):
    import gc
    import weakref
    from usctbench.algorithms import born

    case, _, _ = nonlinear_case()

    def solver(problem, control, *, initial_linearization, **kwargs):
        owner = weakref.ref(initial_linearization.jacobian)
        assert owner() is not None
        del initial_linearization
        gc.collect()
        assert owner() is None
        control.monitor.finish("max_iterations")
        return control.output(problem.initial)

    monkeypatch.setattr(born, "gauss_newton", solver)
    result = BornAlgorithm().run(
        case,
        prepare_config(
            "born",
            config(variant="full_green_nonlinear", background_sound_speed_mps=1505),
            case=case,
        ),
    )
    assert np.isfinite(result.sound_speed_mps).all()


def test_nonlinear_holdout_is_not_used_for_updates_or_line_search():
    case, _, _ = nonlinear_case()
    changed = case.model_copy(deep=True)
    changed.measurement.freq_data[..., 1] *= 15j
    original = BornAlgorithm().run(
        case,
        prepare_config(BornAlgorithm().name, config(max_iterations=1), case=case),
    )
    altered = BornAlgorithm().run(
        changed,
        prepare_config(BornAlgorithm().name, config(max_iterations=1), case=changed),
    )
    assert original.status == altered.status in {"completed", "budget_exhausted"}
    np.testing.assert_array_equal(original.sound_speed_mps, altered.sound_speed_mps)
    assert (
        original.metrics["line_search_history"]
        == altered.metrics["line_search_history"]
    )


@pytest.mark.parametrize(
    "source", [1 + 0j, np.ones(3, complex), np.ones((3, 8), complex)]
)
def test_source_broadcast_matches_explicit_frequency_tx_calibration(source):
    case, _, _ = nonlinear_case()
    expected = BornAlgorithm().run(
        case,
        prepare_config(BornAlgorithm().name, config(max_iterations=1), case=case),
    )
    case.measurement.source_spectrum = np.asarray(source)
    result = BornAlgorithm().run(
        case,
        prepare_config(BornAlgorithm().name, config(max_iterations=1), case=case),
    )
    assert result.status == expected.status in {"completed", "budget_exhausted"}
    np.testing.assert_array_equal(result.sound_speed_mps, expected.sound_speed_mps)


def test_pressure_requires_source_calibration_or_independent_water():
    case, _, _ = nonlinear_case()
    case.measurement.source_spectrum = None
    with pytest.raises(ValueError, match="calibrated source_spectrum"):
        BornAlgorithm().run(
            case, prepare_config(BornAlgorithm().name, config(), case=case)
        )


def test_relative_regularization_is_invariant_to_pressure_units():
    case, _, _ = nonlinear_case()
    other = case.model_copy(deep=True)
    other.measurement.freq_data *= 1e-6
    other.measurement.source_spectrum *= 1e-6
    cfg = config(
        max_iterations=1,
        regularization_scaling="relative_jacobian_diagonal",
        regularization_lambda=0.2,
        regularization="laplacian",
    )
    a, b = (
        BornAlgorithm().run(c, prepare_config(BornAlgorithm().name, cfg, case=c))
        for c in (case, other)
    )
    assert a.status == b.status in {"completed", "budget_exhausted"}
    np.testing.assert_allclose(a.sound_speed_mps, b.sound_speed_mps, rtol=1e-10)
    np.testing.assert_allclose(
        b.metrics["effective_damping"],
        a.metrics["effective_damping"] * 1e-12,
        rtol=1e-12,
    )


def test_underresolved_complex_inverse_grid_is_rejected():
    case, _, _ = nonlinear_case()
    case.grid.spacing_m = (0.005, 0.005)
    with pytest.raises(ValueError, match="underresolved"):
        BornAlgorithm().run(
            case, prepare_config(BornAlgorithm().name, config(), case=case)
        )


def test_phase_seed_and_pressure_updates_exclude_heldout_frequency():
    from usctbench.operators.ray_born import RayBornOperator

    case, _, _ = nonlinear_case()
    f = np.array([100e3, 150e3, 200e3, 250e3])
    water = RayBornOperator(case.grid, case.geometry, f)
    speed = np.full(case.grid.shape, 1503.0)
    forward = RayBornForward(
        case.grid, case.geometry, f, green_backend="volume_integral"
    )
    case.measurement = MeasurementSpec(
        domain="frequency",
        frequencies_hz=f,
        freq_data=forward.forward(1 / speed**2),
        water_reference=water.background_data(),
        source_spectrum=water.source_spectrum,
        valid_mask=water.valid_pair_mask,
    )
    changed = case.model_copy(deep=True)
    changed.measurement.freq_data[-1] *= 30j
    cfg = config(
        max_iterations=1,
        initialization="phase_cgls",
        initialization_iterations=8,
        initialization_smooth_mm=0.3,
        variant="full_green_nonlinear",
        background_sound_speed_mps=1505.0,
        regularization_scaling="relative_jacobian_diagonal",
        evaluation={"frequency_indices": [3], "receiver_indices": [1]},
        run_controls={
            "restore_best_validation": False,
            "update_rtol": None,
            "objective_rtol": None,
        },
    )
    a, b = [
        BornAlgorithm().run(c, prepare_config(BornAlgorithm().name, cfg, case=c))
        for c in (case, changed)
    ]
    assert a.status == b.status in {"completed", "budget_exhausted"}, a.failure_reason
    np.testing.assert_array_equal(a.sound_speed_mps, b.sound_speed_mps)
    assert a.metrics["initialization_qc"]["frequencies_hz"] == [100e3, 150e3, 200e3]
    from usctbench.evaluation.data import make_data_split

    split = make_data_split(
        case.measurement.freq_data,
        valid_mask=case.measurement.valid_mask,
        frequency_indices=[3],
        receiver_indices=[1],
        tx_positions=case.geometry.tx_pos_m,
        rx_positions=case.geometry.rx_pos_m,
    )
    configured = RayBornOperator(
        case.grid,
        case.geometry,
        f,
        background_sound_speed_mps=1505.0,
        green_backend="volume_integral",
        source_spectrum=water.source_spectrum,
    )
    diagonal = configured.normal_diagonal(split.train_weights)
    np.testing.assert_allclose(
        a.metrics["penalty_scale"], np.median(diagonal[diagonal > 0])
    )
    expected_contrast_norm = np.linalg.norm(
        (case.measurement.freq_data - configured.background_data())[split.train]
    )
    np.testing.assert_allclose(
        a.metrics["contrast_evaluation"]["train"]["weighted_observed_norm"],
        expected_contrast_norm,
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"initialization_iterations": 0},
        {"initialization_iterations": 1.5},
        {"initialization_iterations": True},
        {"initialization_lambda": float("nan")},
        {"initialization_smooth_mm": -1},
        {"variant": "wkb_fixed", "initialization": "phase_cgls"},
    ],
)
def test_invalid_initialization_settings_fail_explicitly(settings):
    case, _, _ = nonlinear_case()
    candidate = config(**settings)
    if settings.get("variant") == "wkb_fixed":
        # Keep this an initialization rejection test, not an unrelated budget
        # rejection now that variant-specific unused fields fail closed.
        candidate = AlgorithmConfig(parameters=settings)
    with pytest.raises(ValueError, match="initialization"):
        BornAlgorithm().run(
            case, prepare_config(BornAlgorithm().name, candidate, case=case)
        )


@pytest.mark.parametrize(
    "variant,scaling,expected",
    [
        ("full_green_nonlinear", "absolute", 12),
        ("full_green_nonlinear", "relative_jacobian_diagonal", 12),
        ("wkb_nonlinear", "absolute", 8),
    ],
)
def test_configured_initial_background_is_computed_once(
    monkeypatch, variant, scaling, expected
):
    from usctbench.operators import volume_integral, eikonal

    case = make_sound_speed_case(shape=(4, 5), n_transducers=4)
    case.geometry.tx_pos_m *= 0.12
    case.geometry.rx_pos_m *= 0.12
    frequencies = np.array([100e3, 150e3, 200e3])
    case.measurement = MeasurementSpec(
        domain="frequency",
        frequencies_hz=frequencies,
        freq_data=np.ones((3, 4, 4), complex),
        source_spectrum=np.ones((3, 4), complex),
        valid_mask=~np.eye(4, dtype=bool),
    )
    case.metadata["pressure_contract"] = dtft_pressure_contract()
    counter = []
    if variant.startswith("full_green"):
        real_solve = volume_integral.gmres

        def counted(*args, **kwargs):
            counter.append(1)
            return real_solve(*args, **kwargs)

        monkeypatch.setattr(volume_integral, "gmres", counted)
        key = "green_solves"
    else:
        real_solve = eikonal.PreparedMarching.solve

        def counted(*args, **kwargs):
            counter.append(1)
            return real_solve(*args, **kwargs)

        monkeypatch.setattr(eikonal.PreparedMarching, "solve", counted)
        key = "eikonal_solves"
    cfg = prepare_config(
        "born",
        {
            "parameters": {
                "variant": variant,
                "background_sound_speed_mps": 1505.0,
                "max_cache_bytes": 0,
                "regularization_scaling": scaling,
            },
            "run_controls": {"max_iterations": 0, "max_forward_calls": 2},
        },
        case=case,
    )
    result = BornAlgorithm().run(case, cfg)
    assert len(counter) == expected
    assert result.metrics["stopping"]["work"][key] == len(counter)
    assert result.metrics["stopping"]["selected_iteration"] == 0
    assert result.metrics["stopping"]["work"]["forward_calls"] == 2
    np.testing.assert_allclose(result.sound_speed_mps, 1505.0, atol=1e-12)
