"""Admission and stopping semantics, independent of image quality."""

import json
from usctbench.core.stopping import BudgetExhausted

import numpy as np
import pytest
from pydantic import ValidationError

from usctbench.core.run_controls import RunControls
from usctbench.core.schema import AlgorithmConfig
from usctbench.core.stopping import StopMonitor, WorkLedger


def test_minimal_policy_disables_optional_quality_rules():
    policy = RunControls().capped(default_iterations=30)
    assert policy.max_iterations == 30
    assert policy.update_rtol is None
    assert policy.min_iterations == 3
    assert policy.update_patience == 2
    assert policy.objective_rtol is None
    assert policy.validation_patience is None
    assert not policy.restore_best_validation


@pytest.mark.parametrize(
    "reason",
    [
        "stationary_gradient",
        "stationary_projected_gradient",
        "stationary_surrogate_gradient",
        "stationary_surrogate_projected_gradient",
    ],
)
def test_stationarity_is_not_update_stagnation(reason):
    policy = RunControls().capped(default_iterations=3)
    monitor = StopMonitor(policy, WorkLedger(policy))
    monitor.reason = reason
    assert monitor.record()["termination_category"] == "stationarity"
    monitor.reason = "small_model_update"
    assert monitor.record()["termination_category"] == "stagnation"


def test_caps_intersect_and_zero_is_not_missing():
    requested = RunControls(max_iterations=30, max_elapsed_s=2, max_forward_calls=10)
    actual = requested.capped(
        RunControls(
            max_iterations=0, max_elapsed_s=4, max_forward_calls=3, max_adjoint_calls=0
        )
    )
    assert actual.max_iterations == 0
    assert actual.max_elapsed_s == 2
    assert actual.max_forward_calls == 3
    assert actual.max_adjoint_calls == 0
    assert requested.max_iterations == 30


@pytest.mark.parametrize(
    "values",
    [
        {"max_iterations": True},
        {"max_iterations": 1.5},
        {"max_elapsed_s": float("nan")},
        {"max_elapsed_s": float("inf")},
        {"update_rtol": -1},
        {"update_patience": 0},
        {"unexpected": 2},
    ],
)
def test_invalid_controls_fail_closed(values):
    with pytest.raises(ValidationError):
        RunControls(**values)


def test_one_small_step_or_pre_minimum_steps_do_not_stop():
    policy = RunControls(min_iterations=2, update_patience=2, update_rtol=0.01).capped(
        default_iterations=10
    )
    monitor = StopMonitor(policy, WorkLedger(policy))
    updates = [None, 0.001, 0.001, 0.5, 0.001, 0.001]
    for k, update in enumerate(updates):
        reason = monitor.observe(
            k,
            residual_norm=1,
            observed_norm=2,
            objective=0.5,
            update_relative=update,
        )
        assert reason == ("small_model_update" if k == 5 else None)
    assert monitor.record()["completed_iterations"] == 5


@pytest.mark.parametrize("cap, expected", [(100, 30), (10, 10), (0, 0)])
def test_cap_cannot_increase_inherited_default(cap, expected):
    policy = RunControls().capped(
        default_iterations=30, host_limits={"max_iterations": cap}
    )
    assert policy.max_iterations == expected
    assert (
        RunControls()
        .capped({"max_iterations": cap}, default_iterations=30)
        .max_iterations
        == expected
    )


def test_explicit_request_overrides_default_before_cap():
    assert (
        RunControls(max_iterations=60)
        .capped(default_iterations=30, host_limits={"max_iterations": 40})
        .max_iterations
        == 40
    )
    with pytest.raises(ValueError, match="default_iterations"):
        RunControls().capped({"max_iterations": 100})


def test_public_boundary_resolves_partial_controls_and_host_caps():
    from usctbench.core.config import prepare_config

    request = AlgorithmConfig(run_controls=RunControls(max_elapsed_s=60))
    resolved = prepare_config(
        "straight_cgls", request, host_limits={"max_iterations": 100}
    )
    assert resolved.run_controls.max_iterations == 30
    assert request.run_controls.max_iterations is None
    resolved = prepare_config(
        "straight_cgls", request, host_limits={"max_iterations": 0}
    )
    assert resolved.run_controls.max_iterations == 0


def test_stage_change_resets_update_patience_not_global_iteration_budget():
    policy = RunControls(max_iterations=4, min_iterations=1, update_rtol=0.01).capped()
    monitor = StopMonitor(policy, WorkLedger(policy))
    for k, stage in enumerate(("a", "a", "b", "c", "d")):
        monitor.set_stage(stage)
        reason = monitor.observe(
            k,
            residual_norm=1,
            observed_norm=2,
            objective=0.5,
            update_relative=None if k == 0 else 0.001,
        )
        assert reason == ("max_iterations" if k == 4 else None)
    record = monitor.record()
    assert record["stage_id"] == record["selected_stage_id"] == "d"
    assert record["resolved_policy"]["max_iterations"] == 4
    assert [x["stage_id"] for x in monitor.history] == ["a", "a", "b", "c", "d"]


def monitor(**kwargs):
    policy = RunControls(
        **{
            "max_iterations": 50,
            "min_iterations": 1,
            "update_patience": 1,
            "restore_best_validation": True,
            **kwargs,
        }
    )
    return StopMonitor(policy, WorkLedger(policy, clock=lambda: 0.0))


def observe(m, i, **kwargs):
    args = dict(residual_norm=1.0, observed_norm=2.0, objective=0.5)
    args.update(kwargs)
    return m.observe(i, **args)


def test_or_rules_record_all_triggers_with_priority():
    m = monitor(max_iterations=1, target_relative_residual=0.6, noise_norm=1.0)
    assert observe(m, 0) is None
    assert observe(m, 1) == "target_residual"
    assert m.record()["triggered_rules"] == [
        "target_residual",
        "noise_discrepancy",
        "max_iterations",
    ]
    assert m.record()["quality_target_met"] is True
    assert m.history[0]["quality_triggered_rules"] == []
    assert m.history[1]["quality_triggered_rules"] == [
        "target_residual",
        "noise_discrepancy",
    ]
    json.dumps(m.record(), allow_nan=False)


@pytest.mark.parametrize(
    "kwargs,second,reason",
    [
        ({"max_iterations": 0}, {}, "max_iterations"),
        ({"update_rtol": 0.01}, {"update_relative": 0.001}, "small_model_update"),
        ({"objective_patience": 1, "objective_rtol": 1e-6}, {}, "objective_plateau"),
        ({"max_iterations": 5}, {"residual_norm": 0, "objective": 0}, "exact_data_fit"),
    ],
)
def test_stop_reasons(kwargs, second, reason):
    m = monitor(**kwargs)
    if kwargs.get("max_iterations") == 0:
        assert observe(m, 0) == reason
    else:
        observe(m, 0)
        assert observe(m, 1, **second) == reason
    record = m.record()
    assert record["quality_target_met"] is (reason == "exact_data_fit")


def test_validation_selects_best_not_last_iteration():
    m = monitor(validation_patience=2, objective_rtol=None)
    observe(m, 0, validation_relative=0.9)
    observe(m, 1, validation_relative=0.5)
    observe(m, 2, validation_relative=0.7)
    assert observe(m, 3, validation_relative=0.8) == "validation_plateau"
    index = m.selected_iteration
    assert index == 1
    assert m.record()["completed_iterations"] == 3


@pytest.mark.parametrize("restore_best_validation", [True, False])
def test_quality_target_tracks_selected_checkpoint_not_termination(
    restore_best_validation,
):
    m = monitor(
        max_iterations=1,
        target_relative_residual=0.3,
        restore_best_validation=restore_best_validation,
    )
    assert observe(m, 0, validation_relative=0.1, residual_norm=1) is None
    assert (
        observe(m, 1, validation_relative=0.2, residual_norm=0.5) == "target_residual"
    )
    index = m.selected_iteration
    assert index == (0 if restore_best_validation else 1)
    record = m.record()
    assert record["quality_target_met"] is (not restore_best_validation)
    assert record["selected_iteration"] == index
    assert record["reason"] == "target_residual"
    assert record["triggered_rules"] == ["target_residual", "max_iterations"]
    assert m.history[0]["quality_triggered_rules"] == []
    assert m.history[1]["quality_triggered_rules"] == ["target_residual"]
    json.dumps({"record": record, "history": m.history}, allow_nan=False)


def test_budgets_checked_before_operator_call():
    policy = RunControls(max_forward_calls=1)
    ledger = WorkLedger(policy)
    assert ledger.call("linear_forward", lambda: 7) == 7
    with pytest.raises(BudgetExhausted, match="forward_calls_budget"):
        ledger.call("linear_forward", lambda: pytest.fail("budget overrun"))
    assert ledger.counts["forward_calls"] == 1
    with pytest.raises(BudgetExhausted, match="time_budget"):
        WorkLedger(RunControls(max_elapsed_s=0)).call("adjoint", lambda: 0)


def test_numerical_failure_preserves_last_complete_iteration():
    m = monitor()
    observe(m, 0)
    assert observe(m, 1, objective=np.nan) == "numerical_failure"
    index = m.selected_iteration
    assert index == 0


def test_termination_is_sticky_and_cannot_replace_checkpoint():
    m = monitor(max_iterations=1)
    observe(m, 0)
    observe(m, 1)
    before = m.record()
    assert m.finish("numerical_failure") == "max_iterations"
    assert observe(m, 2, objective=np.nan) == "max_iterations"
    assert m.record() == before
    assert m.record()["termination_category"] == "budget"
    assert m.record()["quality_target_met"] is False


def test_control_rejected_objective_cannot_mismatch_image_and_prediction():
    from usctbench.algorithms._control import InversionControl
    from usctbench.core.schema import AlgorithmConfig
    from usctbench.data.synthetic import make_sound_speed_case

    case = make_sound_speed_case(shape=(4, 4), n_transducers=4)
    obs = np.ones((4, 4))
    from usctbench.core.config import prepare_config

    cfg = prepare_config(
        "straight_cgls", AlgorithmConfig(run_controls={"max_iterations": 4})
    )
    control = InversionControl(case, cfg, obs)
    initial = np.zeros((4, 4))
    control.observe(0, initial, initial, objective=1)
    control.observe(1, np.ones((4, 4)), obs, objective=np.nan)
    selected, metrics = control.output(initial)
    np.testing.assert_array_equal(selected, initial)
    assert metrics["data_residual_norm"] == 4
    assert metrics["stop_reason"] == "numerical_failure"
    assert metrics["stopping"]["termination_category"] == "failure"


def test_control_owns_selected_snapshot_without_retaining_best_linearization():
    from usctbench.algorithms._control import InversionControl
    from usctbench.core.config import prepare_config
    from usctbench.data.synthetic import make_sound_speed_case

    case = make_sound_speed_case(shape=(4, 4), n_transducers=4)
    config = prepare_config(
        "straight_cgls",
        AlgorithmConfig(
            evaluation={"receiver_indices": [1], "exclude_reciprocal": False},
            run_controls={"max_iterations": 3, "restore_best_validation": True},
        ),
    )
    control = InversionControl(case, config, np.ones((4, 4)))
    first = np.full((4, 4), 0.2)
    prediction = np.full((4, 4), 0.9)
    lin0, lin1 = object(), object()
    control.observe(0, first, prediction, objective=1, linearization=lin0)
    control.observe(
        1, np.full((4, 4), 0.5), np.full((4, 4), 0.1), objective=0.5, linearization=lin1
    )
    selected, metrics = control.output(np.zeros((4, 4)))
    np.testing.assert_array_equal(selected, first)
    np.testing.assert_array_equal(control.selected_state.prediction, prediction)
    assert control.current_state.linearization is lin1
    assert control.best_state.linearization is None
    assert metrics["stopping"]["selected_iteration"] == 0
    assert metrics["stopping"]["completed_iterations"] == 1
    selected[:] = 99
    np.testing.assert_array_equal(control.selected_state.model, first)
