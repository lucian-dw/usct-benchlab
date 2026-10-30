"""Nonlinear outer-loop scale, feasible-step and acceptance regressions."""

from tests.helpers import DenseJacobian
import weakref

import numpy as np
import pytest
from scipy.linalg import norm

import usctbench.solvers.nonlinear as nonlinear
from usctbench.algorithms.born import prepare_born_problem
from usctbench.algorithms.bent_ray import prepare_bent_problem
from usctbench.algorithms._control import InversionControl
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    MeasurementSpec,
    USCTCase,
)
from usctbench.operators.base import Linearization, LinearSolveInfo


class _DenseForward(DenseJacobian):
    def __init__(self, matrix, reference):
        super().__init__(matrix, reference.shape)
        self.reference = reference
        self.states = []
        self.trials = []
        self.trial_failure = None
        self.background_builds = self.eikonal_solves = 0

    def forward(self, image):
        return super().forward(image).reshape(1, -1)

    def adjoint(self, data):
        return super().adjoint(data.ravel())

    def linearize(self, model):
        self.states.append(model.copy())
        self.background_builds += 1
        value = self.forward(model - self.reference)
        if self.trial_failure is not None and len(self.states) > 1:
            value = self.trial_failure(value, len(self.states))
        return Linearization(value, self)


@pytest.fixture
def solve(monkeypatch):
    original_step = nonlinear.solve_linearized
    directions = []

    def capture_step(*args, **kwargs):
        direction, info = original_step(*args, **kwargs)
        directions.append(direction.copy())
        return direction, info

    monkeypatch.setattr(nonlinear, "solve_linearized", capture_step)

    def run(
        matrix,
        observed,
        *,
        initial=None,
        problem_kind="born",
        max_iterations=1,
        run_controls=None,
        diagnostics=True,
        trial_failure=None,
        reuse_initial=False,
        **options,
    ):
        if initial is None:
            initial = np.full(
                (1, matrix.shape[1]), 1 / 1500 ** (2 if problem_kind == "born" else 1)
            )
        observed = np.asarray(observed).reshape(1, -1)
        case = USCTCase(
            case_id="feasible_displacement_backtracking",
            grid=GridSpec(shape=initial.shape, spacing_m=(0.001, 0.001)),
            geometry=GeometrySpec(
                tx_pos_m=[[-0.01, 0]],
                rx_pos_m=[[0.01, i * 0.001] for i in range(observed.size)],
            ),
            measurement=MeasurementSpec(domain="features", tof_s=observed),
        )
        config = AlgorithmConfig(
            diagnostics=diagnostics,
            parameters={},
            run_controls={
                "max_iterations": max_iterations,
                "objective_rtol": None,
                "update_rtol": None,
                "restore_best_validation": False,
                **(run_controls or {}),
            },
        )
        from usctbench.core.config import prepare_config

        control = InversionControl(
            case, prepare_config("bent_ray_gn", config), observed
        )
        forward = _DenseForward(matrix, initial)
        forward.trial_failure = trial_failure
        settings = dict(
            bounds=(1400, 1600),
            regularization="identity",
            inner_solver="lsmr",
            inner_iterations=64,
            max_backtracks=8,
        )
        settings.update(options)
        problem_settings = {
            key: settings.pop(key)
            for key in (
                "bounds",
                "regularization",
                "roi",
                "prior_reference",
                "max_update_mps",
            )
            if key in settings
        }
        if problem_kind == "born":
            problem = prepare_born_problem(forward, initial, **problem_settings)
        else:
            problem = prepare_bent_problem(
                forward,
                initial,
                initial,
                offset=np.zeros_like(observed),
                **problem_settings,
            )
        trials = problem.trials

        def capture_trials(*args):
            for candidate, fraction in trials(*args):
                forward.trials.append(candidate.copy())
                yield candidate, fraction

        problem.trials = capture_trials
        if reuse_initial:
            settings["initial_linearization"] = control.call(
                "forward", problem.linearize, initial
            )
        state, metrics = nonlinear.gauss_newton(problem, control, **settings)
        return state, metrics, forward, directions, control

    return run


def test_accepted_linearization_is_current_work_state(solve):
    state, metrics, forward, _, control = solve(np.eye(2), np.array([1e-9, -1e-9]))
    current = control.current_state
    np.testing.assert_array_equal(current.model, state)
    np.testing.assert_array_equal(current.prediction, current.linearization.value)
    assert current.objective == metrics["iteration_history"][-1]["objective"]
    assert current.iteration == metrics["iterations"]
    # One initial build and one accepted trial, no new build at iteration start.
    assert forward.background_builds == 2
    previous = current
    control.monitor.reason = None
    with pytest.raises(FloatingPointError, match="non-finite prediction") as error:
        control.observe(
            current.iteration + 1, state, np.full_like(current.prediction, np.nan)
        )
    control.fail(error.value)
    assert control.current_state is previous
    assert control.monitor.reason == "numerical_failure"


def _original_first_trial(initial, proposal, step_length, bounds, max_update, roi=None):
    candidate = np.clip(
        initial + step_length * proposal, 1 / bounds[1] ** 2, 1 / bounds[0] ** 2
    )
    speed = 1 / np.sqrt(initial)
    candidate = (
        1 / np.clip(1 / np.sqrt(candidate), speed - max_update, speed + max_update) ** 2
    )
    return candidate if roi is None else np.where(roi, candidate, initial)


def test_separate_prior_matches_dense_solution_and_keeps_roi_exterior(solve):
    initial = np.full((1, 4), 1 / 1500**2)
    prior = initial + np.array([[-1.0, 2.0, -1.0, 1.0]]) * 1e-9
    matrix = np.diag([300.0, 200, 100, 50])
    observed = matrix @ np.array([2.0, -1.0, 1.0, 2.0]) * 1e-9
    damping = 5000.0
    roi = np.array([[True, True, False, False]])
    state, metrics, _, _, _ = solve(
        matrix,
        observed,
        initial=initial,
        prior_reference=prior,
        damping=damping,
        roi=roi,
    )
    selected = matrix[:, roi.ravel()]
    delta = np.linalg.solve(
        selected.T @ selected + damping * np.eye(roi.sum()),
        selected.T @ observed + damping * (prior - initial)[roi],
    )
    expected = initial.copy()
    expected[roi] += delta
    np.testing.assert_allclose(state, expected, rtol=1e-12)
    np.testing.assert_array_equal(state[~roi], initial[~roi])
    residual = matrix @ (state - initial).ravel() - observed
    objective = 0.5 * (residual @ residual + damping * np.sum((state - prior) ** 2))
    assert metrics["iteration_history"][-1]["objective"] == pytest.approx(
        objective, rel=1e-12
    )


def _assert_displacement_trials(initial, states, rows, first, step_length):
    assert len(states) == len(rows) > 0
    np.testing.assert_array_equal(states[0], first)
    full_displacement = first - initial
    for index, (state, row) in enumerate(zip(states, rows)):
        fraction = 0.5**index
        assert row["step_length"] == step_length
        assert row["backtracking_fraction"] == fraction
        expected = first if index == 0 else initial + fraction * full_displacement
        np.testing.assert_array_equal(state, expected)
        displacement = state - initial
        assert row["displacement_norm"] == pytest.approx(norm(displacement.ravel()))
        assert row["changed_pixel_count"] == np.count_nonzero(displacement)
        if index:
            assert row["displacement_norm"] == pytest.approx(
                0.5 * rows[index - 1]["displacement_norm"], rel=2e-12, abs=0
            )
            assert row["displacement_norm"] < rows[index - 1]["displacement_norm"]


def _assert_armijo(metrics):
    costs = {row["iteration"]: row["objective"] for row in metrics["iteration_history"]}
    for row in metrics["line_search_history"]:
        if row["accepted"]:
            assert row["projected_directional_derivative"] < 0
            assert row["objective"] <= (
                costs[row["iteration"] - 1]
                + 1e-4 * row["projected_directional_derivative"]
            )


def test_certified_quadratic_step_recovers_from_saturated_first_trial(
    solve,
):
    matrix = 1e8 * np.array([[1.0, 1.0], [0.0, 1e-6]])
    state, metrics, forward, directions, _ = solve(
        matrix, [1e-3, 1e-3], inner_iterations=64
    )
    assert metrics["inner_solver_history"][0]["converged"]
    assert metrics["iterations"] == 1
    assert metrics["iteration_history"][-1]["objective"] < 0.51e-6
    rows = metrics["line_search_history"]
    assert len(rows) == 5 and all(row["proposal"] == "newton_proposal" for row in rows)
    first = _original_first_trial(forward.reference, directions[0], 1, (1400, 1600), 12)
    _assert_displacement_trials(forward.reference, forward.states[1:], rows, first, 1)
    np.testing.assert_array_equal(state, forward.states[-1])
    assert rows[-1]["accepted"]
    assert np.max(np.abs(1 / np.sqrt(first) - 1500)) == pytest.approx(12)
    for candidate in forward.states[1:]:
        assert np.all(candidate >= 1 / 1600**2)
        assert np.all(candidate <= 1 / 1400**2)
        assert np.max(np.abs(1 / np.sqrt(candidate) - 1500)) <= 12 + 1e-12
    _assert_armijo(metrics)


@pytest.mark.parametrize("step_length", [0.4, 3.0])
def test_linear_unconstrained_step_length_is_applied_once(solve, step_length):
    matrix = 1e8 * np.eye(4)
    truth_step = np.array([[1, -2, 3, -4]]) * 1e-10
    state, metrics, forward, directions, _ = solve(
        matrix, matrix @ truth_step.ravel(), step_length=step_length
    )
    rows = metrics["line_search_history"]
    assert len(rows) == (1 if step_length < 1 else 2)
    first = _original_first_trial(
        forward.reference, directions[0], step_length, (1400, 1600), 12
    )
    assert np.max(np.abs(1 / np.sqrt(first) - 1500)) < 12
    _assert_displacement_trials(
        forward.reference, forward.states[1:], rows, first, step_length
    )
    np.testing.assert_allclose(
        state - forward.reference,
        step_length * rows[-1]["backtracking_fraction"] * truth_step,
        rtol=2e-12,
        atol=1e-22,
    )
    assert metrics["iteration_history"][-1]["objective"] < (
        metrics["iteration_history"][0]["objective"]
    )
    _assert_armijo(metrics)


def test_roi_is_fixed_for_the_initial_projection_and_every_halving(solve):
    matrix = np.eye(4)
    matrix[:2, :2] = 1e8 * np.array([[1.0, 1.0], [0.0, 1e-6]])
    initial = 1 / np.array([[1500.0, 1500], [1490, 1510]]) ** 2
    roi = np.array([[True, True], [False, False]])
    state, metrics, forward, directions, _ = solve(
        matrix, [1e-3, 1e-3, 1, 1], initial=initial, roi=roi, step_length=3
    )
    first = _original_first_trial(initial, directions[0], 3, (1400, 1600), 12, roi)
    _assert_displacement_trials(
        initial, forward.states[1:], metrics["line_search_history"], first, 3
    )
    for candidate in forward.states:
        np.testing.assert_array_equal(candidate[~roi], initial[~roi])
        assert np.all(1 / np.sqrt(candidate[roi]) >= 1400)
        assert np.all(1 / np.sqrt(candidate[roi]) <= 1600)
    assert all(
        row["changed_pixel_count"] == 2 for row in metrics["line_search_history"]
    )
    assert np.any(state[roi] != initial[roi])
    _assert_armijo(metrics)


@pytest.mark.parametrize("problem_kind", ["born", "bent"])
def test_projected_uphill_newton_uses_projected_gradient_without_evaluation(
    solve, problem_kind
):
    hessian = np.array([[3.0, -8], [-8, 22]])
    matrix = np.linalg.cholesky(hessian).T
    power = 2 if problem_kind == "born" else 1
    initial = np.array([[1 / 1600**power, 1 / 1500**power]])
    gradient = np.array([[1.0, -2]]) * 1e-6
    observed = -np.linalg.solve(matrix.T, gradient.ravel())
    state, metrics, forward, directions, _ = solve(
        matrix, observed, initial=initial, problem_kind=problem_kind
    )
    rows = metrics["line_search_history"]
    assert metrics["direction_checks"][0]["raw_directional_derivative"] < 0
    newton_rows = [row for row in rows if row["proposal"] == "newton_proposal"]
    assert len(newton_rows) == 9  # Initial trial plus eight backtracks.
    assert all(row["projected_directional_derivative"] > 0 for row in newton_rows)
    assert all(row["rejection"] == "non_descent_projected_step" for row in newton_rows)
    assert all(row["objective"] is None for row in newton_rows)
    for index, row in enumerate(newton_rows):
        assert row["changed_pixel_count"] == 1
        assert row["backtracking_fraction"] == 0.5**index
        assert row["displacement_norm"] == pytest.approx(
            0.5**index * newton_rows[0]["displacement_norm"], rel=2e-12, abs=0
        )
        if problem_kind == "born":
            first_newton = _original_first_trial(
                initial, directions[0], 1, (1400, 1600), 12
            )
            expected = initial + 0.5**index * (first_newton - initial)
        else:
            expected = np.clip(initial + 0.5**index * directions[0], 1 / 1600, 1 / 1400)
        np.testing.assert_allclose(forward.trials[index], expected, rtol=0, atol=1e-20)
        assert row["projected_directional_derivative"] == pytest.approx(
            np.vdot(gradient, forward.trials[index] - initial).real,
            rel=1e-10,
            abs=0,
        )
    assert rows[-1]["proposal"] == "projected_gradient" and rows[-1]["accepted"]
    feasible_gradient = gradient.copy()
    feasible_gradient[0, 0] = 0
    fallback = -feasible_gradient * (
        norm(directions[0].ravel()) / norm(feasible_gradient.ravel())
    )
    first = (
        _original_first_trial(initial, fallback, 1, (1400, 1600), 12)
        if problem_kind == "born"
        else np.clip(initial + fallback, 1 / 1600, 1 / 1400)
    )
    np.testing.assert_array_equal(forward.states[1], first)
    assert len(forward.states) == 1 + sum(row["objective"] is not None for row in rows)
    assert state[0, 0] == initial[0, 0]
    _assert_armijo(metrics)


def test_nonlinear_solver_rejects_uphill_smoothing_and_recovers_dense_solution(solve):
    matrix = np.diag(np.sqrt([1000.0, 1, 1, 1]))
    observed = -1e-5 * np.array([-3.0, 0.6, 0.6, -0.5]) / np.diag(matrix)
    initial = np.full((2, 2), 1 / 1500)
    state, metrics, _, _, _ = solve(
        matrix,
        observed,
        initial=initial,
        problem_kind="bent",
        damping=0.2,
        smooth_sigma=1,
        max_iterations=4,
    )
    expected = np.linalg.solve(matrix.T @ matrix + 0.2 * np.eye(4), matrix.T @ observed)
    np.testing.assert_allclose((state - initial).ravel(), expected, atol=2e-14)
    check = metrics["direction_checks"][0]
    assert (
        check["raw_directional_derivative"]
        < 0
        < check["smoothed_directional_derivative"]
    )
    assert check["smoothing_rejected"]
    assert metrics["stop_reason"] == "stationary_gradient"
    assert metrics["gradient_checks"][-1]["relative_to_initial"] <= 1e-8
    assert (
        metrics["iteration_history"][-1]["objective"]
        < metrics["iteration_history"][0]["objective"]
    )
    _assert_armijo(metrics)


def test_zero_inner_update_does_not_claim_stationarity(monkeypatch, solve):
    monkeypatch.setattr(
        nonlinear,
        "solve_linearized",
        lambda jac, residual, current, *a, **kw: (
            np.zeros_like(current),
            LinearSolveInfo("fixture", 0, "iteration_limit", 1, 1, False),
        ),
    )
    state, metrics, forward, _, _ = solve(
        np.eye(4),
        np.full(4, 1e-6),
        problem_kind="bent",
    )
    assert metrics["stop_reason"] == "max_iterations"
    assert metrics["gradient_checks"][0]["norm"] > 0
    assert metrics["line_search_history"][-1]["proposal"] == "projected_gradient"
    assert metrics["data_residual_reduction"] > 0
    assert np.any(state != forward.reference)


def test_bound_kkt_stationarity_requires_no_newton_or_trial(solve):
    initial = np.full((2, 2), 1 / 1600)
    state, metrics, forward, directions, control = solve(
        np.eye(4),
        np.full(4, -1e-6),
        initial=initial,
        problem_kind="bent",
        max_iterations=4,
    )
    assert metrics["stop_reason"] == "stationary_projected_gradient"
    assert len(forward.states) == 1 and not directions
    assert control.work.counts["adjoint_calls"] == 1
    assert control.work.counts.get("line_search", 0) == 0
    np.testing.assert_array_equal(state, initial)


@pytest.mark.parametrize("recover", [False, True])
def test_nonfinite_trials_are_rejected_without_changing_checkpoint(solve, recover):
    state, metrics, forward, _, control = solve(
        np.eye(4),
        np.full(4, 1e-6),
        problem_kind="bent",
        smooth_sigma=0.5,
        trial_failure=lambda value, count: (
            value if recover and count > 2 else np.full_like(value, np.nan)
        ),
    )
    assert (
        metrics["line_search_history"][0]["rejection"]
        == "nonfinite_trial_prediction_or_objective"
    )
    current = control.current_state
    np.testing.assert_array_equal(current.model, state)
    np.testing.assert_array_equal(current.prediction, current.linearization.value)
    np.testing.assert_allclose(
        current.prediction, forward.forward(state - forward.reference)
    )
    assert current.objective == metrics["iteration_history"][-1]["objective"]
    assert current.iteration == metrics["iterations"]
    if recover:
        assert metrics["stop_reason"] == "max_iterations"
        assert metrics["data_residual_reduction"] > 0
        _assert_armijo(metrics)
    else:
        assert metrics["stop_reason"] == "numerical_failure"
        assert len(metrics["iteration_history"]) == 1
        np.testing.assert_array_equal(state, forward.reference)
        assert metrics["data_residual_norm"] == pytest.approx(2e-6)


@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_relative_gradient_stopping_is_invariant_to_objective_scale(solve, scale):
    matrix = np.diag(np.sqrt([1000.0, 1, 1, 1]))
    observed = -1e-5 * np.array([-3.0, 0.6, 0.6, -0.5]) / np.diag(matrix)
    state, metrics, forward, _, _ = solve(
        matrix * scale,
        observed * scale,
        problem_kind="bent",
        damping=0.2 * scale**2,
        max_iterations=4,
    )
    expected = np.linalg.solve(matrix.T @ matrix + 0.2 * np.eye(4), matrix.T @ observed)
    np.testing.assert_allclose(
        (state - forward.reference).ravel(), expected, atol=2e-14
    )
    assert metrics["stop_reason"] == "stationary_gradient"
    assert metrics["iterations"] == 1
    assert metrics["gradient_checks"][0]["relative_to_initial"] == 1
    assert metrics["gradient_checks"][-1]["relative_to_initial"] <= 1e-8


def test_precomputed_initial_linearization_commits_at_forward_budget(solve):
    state, metrics, forward, _, control = solve(
        np.eye(2),
        [1e-9, -1e-9],
        reuse_initial=True,
        run_controls={"max_forward_calls": 1},
    )
    assert forward.background_builds == 1
    assert control.work.counts["forward_calls"] == 1
    assert metrics["stop_reason"] == "forward_calls_budget"
    assert control.current_state.linearization is not None
    np.testing.assert_array_equal(state, forward.reference)
    np.testing.assert_array_equal(control.current_state.prediction, 0)
    assert metrics["iteration_history"][0]["objective"] == pytest.approx(1e-18)


def test_precomputed_linearization_and_diagnostics_preserve_numerical_path(solve):
    args = (1e8 * np.eye(2), [0.01, -0.01])
    first, detailed, forward, _, control = solve(
        *args,
        reuse_initial=True,
        max_iterations=3,
        step_length=0.5,
    )
    second, compact, other_forward, _, other = solve(
        *args,
        diagnostics=False,
        max_iterations=3,
        step_length=0.5,
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(
        first - forward.reference,
        (1 - 0.5**3) * np.array([[1e-10, -1e-10]]),
        rtol=1e-12,
        atol=1e-22,
    )
    assert forward.background_builds == other_forward.background_builds == 4
    assert control.work.counts == other.work.counts
    assert (
        compact["iteration_history"][-1]["objective"]
        == detailed["iteration_history"][-1]["objective"]
    )
    assert "line_search_history" not in compact
    assert "direction_checks" not in compact
    assert "gradient_checks" not in compact


@pytest.mark.parametrize("step_length", [0.5, 3.0])
def test_advancing_state_releases_obsolete_linearizations_and_model_caches(step_length):
    from usctbench.core.config import prepare_config

    initial = np.full((1, 2), 1 / 1500**2)
    observed = np.array([[0.01, -0.01]])
    matrix = 1e8 * np.eye(2)
    references = []

    class CachedJacobian(DenseJacobian):
        def adjoint(self, values):
            return super().adjoint(values.ravel())

    def linearize(model):
        # During a trial, only the current accepted background may remain alive.
        for lin, cache in references:
            if (
                control.current_state is not None
                and lin() is control.current_state.linearization
            ):
                assert cache() is control.current_state.linearization.jacobian
            else:
                assert lin() is None and cache() is None
        jacobian = CachedJacobian(matrix, initial.shape)
        jacobian.model_cache = model.copy()
        result = Linearization(
            jacobian.forward(model - initial).reshape(1, 2), jacobian
        )
        references.append((weakref.ref(result), weakref.ref(jacobian)))
        return result

    case = USCTCase(
        case_id="accepted_background_lifetime",
        grid=GridSpec(shape=initial.shape, spacing_m=(0.001, 0.001)),
        geometry=GeometrySpec(tx_pos_m=[[0, 0]], rx_pos_m=[[0.01, 0], [0, 0.01]]),
        measurement=MeasurementSpec(domain="features", tof_s=observed),
    )
    control = InversionControl(
        case,
        prepare_config(
            "bent_ray_gn",
            AlgorithmConfig(
                run_controls={
                    "max_iterations": 3,
                    "objective_rtol": None,
                    "update_rtol": None,
                    "restore_best_validation": False,
                }
            ),
        ),
        observed,
    )
    problem = nonlinear.GaussNewtonProblem(
        initial,
        initial,
        1 / 1600**2,
        1 / 1400**2,
        linearize,
        lambda candidate, current: np.clip(candidate, 1 / 1600**2, 1 / 1400**2),
        np.copy,
        nonlinear.backtrack_feasible_displacement,
        "identity",
        None,
        1e-10,
    )
    state, metrics = nonlinear.gauss_newton(
        problem,
        control,
        step_length=step_length,
        initial_linearization=control.call("forward", linearize, initial),
    )
    assert len(references) == (4 if step_length < 1 else 7)
    assert all(lin() is None and cache() is None for lin, cache in references[:-1])
    assert references[-1][0]() is control.current_state.linearization
    accepted_step = step_length if step_length < 1 else step_length / 2
    np.testing.assert_allclose(
        state - initial,
        (1 - (1 - accepted_step) ** 3) * np.array([[1e-10, -1e-10]]),
        rtol=1e-12,
        atol=1e-22,
    )
    assert metrics["iterations"] == 3
