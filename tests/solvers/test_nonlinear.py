"""Nonlinear outer-loop scale, feasible-step and acceptance regressions."""

from types import SimpleNamespace
from tests.helpers import DenseJacobian

import numpy as np
import pytest
from scipy.linalg import norm

import usctbench.solvers.nonlinear as nonlinear
from usctbench.algorithms.rwave import prepare_born_problem
from usctbench.algorithms._control import InversionControl
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    MeasurementSpec,
    USCTCase,
)
from usctbench.operators.base import Linearization


class _DenseForward:
    def __init__(self, matrix, reference):
        self.matrix, self.reference = matrix, reference
        self.data_dtype = matrix.dtype
        self.states = []
        self.background_builds = self.eikonal_solves = 0

    def forward(self, image):
        return (self.matrix @ image.ravel()).reshape(1, -1)

    def adjoint(self, data):
        return (self.matrix.T @ data.ravel()).reshape(self.reference.shape)

    def linearize(self, model):
        self.states.append(model.copy())
        self.background_builds += 1
        return Linearization(self.forward(model - self.reference), self)


@pytest.fixture
def solve(monkeypatch):
    original_step = nonlinear.solve_linearized
    directions = []

    def capture_step(*args, **kwargs):
        direction, info = original_step(*args, **kwargs)
        directions.append(direction.copy())
        return direction, info

    monkeypatch.setattr(nonlinear, "solve_linearized", capture_step)

    def run(matrix, observed, *, initial=None, **options):
        if initial is None:
            initial = np.full((1, matrix.shape[1]), 1 / 1500**2)
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
            diagnostics=True,
            parameters={},
            run_controls={
                "max_iterations": 1,
                "objective_rtol": None,
                "update_rtol": None,
                "restore_best_validation": False,
            },
        )
        from usctbench.core.config import prepare_config

        control = InversionControl(
            case, prepare_config("bent_ray_gn", config), observed
        )
        forward = _DenseForward(matrix, initial)
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
        problem = prepare_born_problem(forward, initial, **problem_settings)
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
    assert metrics["prior_reference_source"] == "explicit"


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
    initial_cost = metrics["iteration_history"][0]["objective"]
    for row in metrics["line_search_history"]:
        if row["accepted"]:
            assert row["projected_directional_derivative"] < 0
            assert row["objective"] <= (
                initial_cost + 1e-4 * row["projected_directional_derivative"]
            )


@pytest.mark.parametrize("inner_iterations", [64, 256])
def test_certified_quadratic_step_recovers_from_saturated_first_trial(
    solve, inner_iterations
):
    matrix = 1e8 * np.array([[1.0, 1.0], [0.0, 1e-6]])
    state, metrics, forward, directions, _ = solve(
        matrix, [1e-3, 1e-3], inner_iterations=inner_iterations
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
    constraints = metrics["line_search_constraints"]
    assert (
        constraints["strategy"] == "project_once_then_backtrack_feasible_displacement"
    )
    assert constraints["step_length_applies_to"] == "raw_proposal_before_projection"
    assert (
        constraints["backtracking_fraction_applies_to"]
        == "initial_feasible_displacement"
    )
    assert constraints["sound_speed_bounds_mps"] == [1400, 1600]
    assert constraints["max_update_mps"] == 12
    assert constraints["displacement_parameter"] == "squared_slowness"
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
    assert metrics["line_search_constraints"]["roi_fixed"]
    assert all(
        row["changed_pixel_count"] == 2 for row in metrics["line_search_history"]
    )
    assert np.any(state[roi] != initial[roi])
    _assert_armijo(metrics)


def test_projected_uphill_newton_uses_projected_gradient_without_evaluation(solve):
    hessian = np.array([[3.0, -8], [-8, 22]])
    matrix = np.linalg.cholesky(hessian).T
    initial = np.array([[1 / 1600**2, 1 / 1500**2]])
    gradient = np.array([[1.0, -2]]) * 1e-6
    observed = -np.linalg.solve(matrix.T, gradient.ravel())
    state, metrics, forward, directions, _ = solve(matrix, observed, initial=initial)
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
    assert rows[-1]["proposal"] == "projected_gradient" and rows[-1]["accepted"]
    feasible_gradient = gradient.copy()
    feasible_gradient[0, 0] = 0
    fallback = -feasible_gradient * (
        norm(directions[0].ravel()) / norm(feasible_gradient.ravel())
    )
    first = _original_first_trial(initial, fallback, 1, (1400, 1600), 12)
    np.testing.assert_array_equal(forward.states[1], first)
    assert len(forward.states) == 1 + sum(row["objective"] is not None for row in rows)
    assert state[0, 0] == initial[0, 0]
    _assert_armijo(metrics)


@pytest.mark.parametrize("max_backtracks", [0, 10])
def test_nonlinear_solver_rejects_uphill_smoothing_and_recovers_dense_solution(
    max_backtracks,
):
    from usctbench.algorithms._control import InversionControl
    from usctbench.core.schema import (
        AlgorithmConfig,
        GridSpec,
        GeometrySpec,
        MeasurementSpec,
        USCTCase,
    )
    from usctbench.operators.base import Linearization
    from usctbench.solvers.nonlinear import gauss_newton

    shape = (2, 2)
    initial = np.full(shape, 1 / 1500**2)
    matrix = np.diag(np.sqrt([1000.0, 1, 1, 1])) * 1e-6 / initial[0, 0]
    normalized_gradient = np.array([-3.0, 0.6, 0.6, -0.5])
    observed = (-0.02e-6 * normalized_gradient / np.sqrt([1000.0, 1, 1, 1])).reshape(
        shape
    )

    class Forward:
        background_builds = eikonal_solves = 0

        def linearize(self, model):
            self.background_builds += 1
            jacobian = DenseJacobian(matrix, shape)
            return Linearization(
                jacobian.forward(model - initial).reshape(shape),
                SimpleNamespace(
                    forward=jacobian.forward,
                    data_dtype=jacobian.data_dtype,
                    adjoint=lambda y: jacobian.adjoint(y.ravel()),
                ),
            )

    positions = [[-0.01, 0], [0.01, 0]]
    case = USCTCase(
        case_id="convex_newton_smoothing_counterexample",
        grid=GridSpec(shape=shape, spacing_m=(0.001, 0.001)),
        geometry=GeometrySpec(tx_pos_m=positions, rx_pos_m=positions),
        measurement=MeasurementSpec(domain="features", delta_tof_s=observed),
    )
    config = AlgorithmConfig(
        diagnostics=True,
        parameters={},
        run_controls={
            "max_iterations": 3,
            "target_relative_residual": 1e-08,
            "restore_best_validation": False,
        },
    )
    from usctbench.core.config import prepare_config

    control = InversionControl(case, prepare_config("straight_cgls", config), observed)
    problem = prepare_born_problem(Forward(), initial, (1400, 1600))
    result, metrics = gauss_newton(
        problem,
        control,
        inner_iterations=8,
        damping=0,
        smooth_sigma=1,
        max_backtracks=max_backtracks,
    )
    expected = initial + np.linalg.solve(matrix, observed.ravel()).reshape(shape)
    np.testing.assert_allclose(result, expected, rtol=1e-10)
    assert metrics["data_relative_residual"] < 1e-8
    assert metrics["direction_checks"][0]["smoothing_rejected"]
    assert metrics["direction_checks"][0]["raw_directional_derivative"] < 0
    assert all(
        row["projected_directional_derivative"] < 0
        for row in metrics["line_search_history"]
        if row["accepted"]
    )
