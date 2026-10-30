"""Native Bent optimizer contracts; dense fixtures isolate optimization from PDEs."""

from usctbench.core.config import prepare_config

from tests.helpers import DenseJacobian
import numpy as np
import pytest

import usctbench.algorithms.bent_ray as bent
import usctbench.solvers.nonlinear as nonlinear
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    MeasurementSpec,
    USCTCase,
)
from usctbench.operators.base import Linearization
from usctbench.operators.eikonal import EikonalForward
from usctbench.operators.model_space import BilinearBasis
from usctbench.operators.regularization import regularizer


def _config(*, max_iterations=4, run_controls=None, evaluation=None, **parameters):
    return AlgorithmConfig(
        diagnostics=True,
        parameters={
            "inner_iterations": 80,
            "regularization_lambda": 0.0,
            "regularization": "identity",
            **parameters,
        },
        run_controls={"max_iterations": max_iterations, **(run_controls or {})},
        evaluation=evaluation or {},
    )


def _case(shape, data):
    nt, nr = data.shape
    return USCTCase(
        case_id="bent_solver_contract",
        grid=GridSpec(shape=shape, spacing_m=(0.001, 0.0008)),
        geometry=GeometrySpec(
            tx_pos_m=[[-0.001, 0.0012 + i * 0.001] for i in range(nt)],
            rx_pos_m=[[0.0092, 0.0052 + i * 0.001] for i in range(nr)],
        ),
        measurement=MeasurementSpec(domain="features", delta_tof_s=data),
        metadata={"reference_sound_speed_mps": 1500.0},
    )


def _offset(case):
    return (
        np.linalg.norm(
            case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None, :], axis=-1
        )
        / 1500
    )


class _DenseForward(DenseJacobian):
    def __init__(self, case, matrix):
        super().__init__(matrix, case.grid.shape)
        self.grid = case.grid
        self.ray_shape = case.measurement.delta_tof_s.shape
        self.n_rays = int(np.prod(self.ray_shape))
        self.spatial_order = 1
        self.offset = _offset(case)
        self.states = []

    def adjoint(self, values):
        return super().adjoint(values.ravel())

    def linearize(self, state):
        self.states.append(state.copy())
        value = self.offset.ravel() + self.forward(state - 1 / 1500)
        return Linearization(value, self)


def _dense(monkeypatch, matrix, data, shape=(2, 2)):
    case = _case(shape, np.asarray(data).reshape(2, -1))
    forward = _DenseForward(case, matrix)
    monkeypatch.setattr(bent, "EikonalForward", lambda *a, **kw: forward)
    return case, forward


def _assert_armijo(result):
    costs = {
        row["iteration"]: row["objective"]
        for row in result.metrics["iteration_history"]
    }
    for row in result.metrics["line_search_history"]:
        if row["accepted"]:
            assert row["projected_directional_derivative"] < 0
            assert row["objective"] <= (
                costs[row["iteration"] - 1]
                + 1e-4 * row["projected_directional_derivative"]
            )


@pytest.mark.parametrize("kind", ["identity", "laplacian"])
@pytest.mark.parametrize("reduced", [False, True])
def test_weighted_regularized_solution_matches_fine_grid_dense_reference(
    monkeypatch, kind, reduced
):
    rng = np.random.default_rng(241)
    shape = (4, 5)
    matrix = rng.normal(size=(24, 20))
    data = matrix @ rng.normal(scale=2e-6, size=20) + rng.normal(scale=1e-6, size=24)
    case, _ = _dense(monkeypatch, matrix, data, shape)
    weights = rng.uniform(0.2, 0.9, (2, 12))
    case.measurement.ray_weights = weights
    config = _config(
        regularization_lambda=np.sqrt(0.3),
        regularization=kind,
        gradient_rtol=1e-6,
        evaluation={"receiver_indices": [1]},
        **({"model_grid_shape": [2, 3]} if reduced else {}),
    )
    basis = BilinearBasis(case.grid, (2, 3)).matrix.toarray() if reduced else np.eye(20)
    penalty = np.column_stack(
        [regularizer(column.reshape(shape), kind).ravel() for column in basis.T]
    )
    train_weights = weights.copy()
    train_weights[:, 1] = 0
    design = matrix @ basis
    expected = np.linalg.solve(
        design.T @ (train_weights.ravel()[:, None] * design)
        + 0.3 * penalty.T @ penalty,
        design.T @ (train_weights.ravel() * data),
    )
    result = bent.BentRayAlgorithm().run(
        case, prepare_config(bent.BentRayAlgorithm().name, config, case=case)
    )
    assert result.status in {"completed", "budget_exhausted"}, result.failure_reason
    np.testing.assert_allclose(
        (1 / result.sound_speed_mps - 1 / 1500).ravel(), basis @ expected, atol=2e-11
    )
    altered = case.model_copy(deep=True)
    altered.measurement.delta_tof_s[:, 1] += 1.0
    other = bent.BentRayAlgorithm().run(
        altered, prepare_config(bent.BentRayAlgorithm().name, config, case=altered)
    )
    np.testing.assert_array_equal(other.sound_speed_mps, result.sound_speed_mps)
    assert other.metrics["gradient_checks"] == result.metrics["gradient_checks"]
    _assert_armijo(result)


def test_disabling_backtracking_does_not_allow_ascent(monkeypatch):
    case, _ = _dense(monkeypatch, np.eye(4), np.full(4, 1e-6))
    result = bent.BentRayAlgorithm().run(
        case,
        prepare_config(
            bent.BentRayAlgorithm().name,
            _config(line_search=False, step_length=10),
            case=case,
        ),
    )
    assert result.status == "failed"
    assert result.metrics["stop_reason"] == "line_search_failed"
    assert len(result.metrics["line_search_history"]) == 2
    assert not any(row["accepted"] for row in result.metrics["line_search_history"])
    np.testing.assert_allclose(result.sound_speed_mps, 1500)


@pytest.mark.parametrize("reference", [None, 1500.0, 1490.0])
def test_differential_reference_override_must_match_metadata(reference):
    case = _case((3, 3), np.ones((2, 2)) * 1e-8)
    if reference is None:
        case.metadata.clear()
    else:
        case.metadata["reference_sound_speed_mps"] = reference
    with pytest.raises(ValueError, match="measurement reference"):
        bent.BentRayAlgorithm().run(
            case,
            prepare_config(
                bent.BentRayAlgorithm().name,
                _config(reference_sound_speed_mps=1600),
                case=case,
            ),
        )
    matching = 1500 if reference is None else reference
    result = bent.BentRayAlgorithm().run(
        case,
        prepare_config(
            bent.BentRayAlgorithm().name,
            _config(
                reference_sound_speed_mps=matching,
                run_controls={"max_forward_calls": 0},
            ),
            case=case,
        ),
    )
    assert result.status in {"completed", "budget_exhausted"}, result.failure_reason
    assert result.metrics["measurement_reference_sound_speed_mps"] == matching


def test_absolute_tof_allows_model_reference_override():
    case = _case((3, 3), np.ones((2, 2)) * 1e-8)
    case.measurement.tof_s = _offset(case)
    case.measurement.delta_tof_s = None
    result = bent.BentRayAlgorithm().run(
        case,
        prepare_config(
            bent.BentRayAlgorithm().name,
            _config(
                reference_sound_speed_mps=1600, run_controls={"max_forward_calls": 0}
            ),
            case=case,
        ),
    )
    assert result.status in {"completed", "budget_exhausted"}, result.failure_reason
    assert result.metrics["measurement_reference_sound_speed_mps"] is None
    np.testing.assert_allclose(result.sound_speed_mps, 1600)


@pytest.mark.parametrize("spatial_order", [1, 2])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("kind", ["identity", "laplacian"])
def test_native_regularized_objective_directional_derivative_and_roi(
    monkeypatch, spatial_order, reduced, kind
):
    rng = np.random.default_rng(83)
    case = _case((7, 8), np.zeros((2, 4)))
    forward = EikonalForward(case.grid, case.geometry, spatial_order=spatial_order)
    truth = (1 + rng.uniform(-0.02, 0.02, case.grid.shape)) / 1500
    case.measurement.delta_tof_s = forward.forward(truth).reshape(2, 4) - _offset(case)
    case.measurement.ray_weights = rng.uniform(0.2, 0.9, (2, 4))
    roi = np.ones(case.grid.shape, bool)
    roi[[0, -1], :] = False
    case.grid.roi_mask = roi
    basis = BilinearBasis(case.grid, (3, 4)) if reduced else None
    parameters = (
        {"model_grid_shape": [3, 4]}
        if reduced
        else {
            "initial_sound_speed_mps": 1500 + rng.uniform(-20, 20, case.grid.shape),
            "roi_update_only": True,
        }
    )
    records = []
    original_step = nonlinear.solve_linearized

    def record_step(jacobian, residual, current, control, **kwargs):
        update, info = original_step(jacobian, residual, current, control, **kwargs)
        records.append((current.copy(), update.copy(), control))
        return update, info

    monkeypatch.setattr(nonlinear, "solve_linearized", record_step)
    config = _config(
        max_iterations=2,
        inner_iterations=12,
        regularization_lambda=np.sqrt(1e-05),
        regularization=kind,
        step_length=0.3,
        smooth_sigma=0.5,
        eikonal_order=spatial_order,
        evaluation={"receiver_indices": [1]},
        **parameters,
    )
    result = bent.BentRayAlgorithm().run(
        case, prepare_config(bent.BentRayAlgorithm().name, config, case=case)
    )
    assert result.status in {"completed", "budget_exhausted"}, result.failure_reason
    assert len(records) == 2
    current, direction, control = records[-1]
    fine = current if basis is None else basis.forward(current)
    tangent = direction if basis is None else basis.forward(direction)
    assert np.linalg.norm(regularizer(fine, kind)) > 0

    def objective(alpha):
        delta = fine + alpha * tangent
        prediction = forward.forward(1 / 1500 + delta).reshape(2, 4) - _offset(case)
        residual = np.where(control.split.train, control.safe_observed - prediction, 0)
        return 0.5 * np.sum(control.precision * residual**2) + 0.5e-5 * np.sum(
            regularizer(delta, kind) ** 2
        )

    epsilon = 1e-3
    finite_difference = (objective(epsilon) - objective(-epsilon)) / (2 * epsilon)
    slope = result.metrics["direction_checks"][-1]["raw_directional_derivative"]
    np.testing.assert_allclose(slope, finite_difference, rtol=2e-4, atol=1e-23)
    if not reduced:
        np.testing.assert_array_equal(
            result.sound_speed_mps[~roi], np.full(np.sum(~roi), 1500.0)
        )
    altered = case.model_copy(deep=True)
    altered.measurement.delta_tof_s[:, 1] += 1.0
    other = bent.BentRayAlgorithm().run(
        altered, prepare_config(bent.BentRayAlgorithm().name, config, case=altered)
    )
    np.testing.assert_array_equal(result.sound_speed_mps, other.sound_speed_mps)
    assert result.metrics["direction_checks"] == other.metrics["direction_checks"]
    _assert_armijo(result)
