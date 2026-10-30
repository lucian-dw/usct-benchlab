"""Independent optimization references for the registered straight CGLS path."""

from types import SimpleNamespace

from usctbench.data.synthetic import make_sound_speed_case

import numpy as np
import pytest
from scipy.optimize import lsq_linear, minimize, minimize_scalar

from usctbench.algorithms._control import InversionControl
from usctbench.algorithms.straight import StraightRayCGLSAlgorithm
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    MeasurementSpec,
    USCTCase,
)
from usctbench.core.config import prepare_config
from usctbench.operators.straight_ray import StraightRayProjector
from usctbench.solvers.projected_cg import projected_cg


def huber_case():
    # The two active Siddon rows are exactly [0.001] and [0.002] meters.
    return USCTCase(
        case_id="one_pixel_huber_reference",
        grid=GridSpec(shape=(1, 1), spacing_m=(0.001, 0.002)),
        geometry=GeometrySpec(
            type="custom",
            tx_pos_m=[[-0.001, 0.0005], [0.0005, -0.001]],
            rx_pos_m=[[0.002, 0.0005], [0.0005, 0.003]],
        ),
        measurement=MeasurementSpec(
            domain="features",
            delta_tof_s=[[1e-8, 0], [0, 4e-8]],
            valid_mask=np.eye(2, dtype=bool),
        ),
        metadata={"reference_sound_speed_mps": 1500.0},
    )


def bounded_case():
    # A grazing/short first ray is bound-active; the second pixel remains free.
    return USCTCase(
        case_id="two_pixel_bound_reference",
        grid=GridSpec(shape=(1, 2), spacing_m=(0.001, 0.001)),
        geometry=GeometrySpec(
            type="custom",
            tx_pos_m=[[0.0001, 0.0001], [-0.001, 0.0015]],
            rx_pos_m=[[0.00011, 0.0001], [0.002, 0.0015]],
        ),
        measurement=MeasurementSpec(
            domain="features",
            delta_tof_s=[[1e-6, 0], [0, 2.5e-9]],
            valid_mask=np.eye(2, dtype=bool),
        ),
        metadata={"reference_sound_speed_mps": 1500.0},
    )


def parameters(robust):
    if robust:
        return {
            "robust_loss": "huber",
            "huber_delta_s": 1e-8,
            "irls_max_inner_steps": 10,
        }
    return {
        "sound_speed_bounds_mps": [1 / (1 / 1500 + 1e-5), 1 / (1 / 1500 - 1e-5)],
    }


def run(case, params, *, run_controls=None, evaluation=None):
    result = StraightRayCGLSAlgorithm().run(
        case,
        prepare_config(
            StraightRayCGLSAlgorithm().name,
            AlgorithmConfig(
                parameters=params,
                run_controls={"max_iterations": 30, **(run_controls or {})},
                evaluation=evaluation or {},
            ),
            case=case,
        ),
    )
    assert result.status in {"completed", "budget_exhausted"}, result.failure_reason
    assert case.ground_truth.sound_speed_mps is None
    return result


def perturbation(result):
    return 1 / result.sound_speed_mps - 1 / 1500


@pytest.mark.parametrize("inner_steps", [30, 10, 1])
def test_huber_inner_convergence_refreshes_before_global_stopping(inner_steps):
    result = run(
        huber_case(), {**parameters(True), "irls_max_inner_steps": inner_steps}
    )
    # In units x=delta_s/1e-5, min rho(x-1)+rho(2*x-4) has x=1.8.
    np.testing.assert_allclose(perturbation(result), [[1.8e-5]], rtol=1e-10)
    assert result.metrics["iteration_history"][-1]["objective"] == pytest.approx(
        4e-17, rel=1e-10, abs=1e-28
    )
    assert result.metrics["stopping"]["work"]["irls_reweights"] >= 2
    assert result.metrics["iterations"] <= 30
    assert result.metrics["solver_optimality"]["verified"]
    assert result.metrics["solver_optimality"]["converged"]


@pytest.mark.parametrize("weight", [0.25, 1.0])
def test_weighted_huber_matches_independent_scalar_minimization(weight):
    case = huber_case()
    case.measurement.ray_weights = np.diag([weight, 1.0])

    def objective(x):
        residual = np.abs(np.array([x - 1, 2 * x - 4]))
        rho = np.where(residual <= 1, 0.5 * residual**2, residual - 0.5)
        return np.array([weight, 1.0]) @ rho

    expected = minimize_scalar(
        objective, method="bounded", bounds=(-5, 5), options={"xatol": 1e-13}
    )
    assert expected.success
    result = run(case, parameters(True))
    actual = perturbation(result).item() / 1e-5
    assert actual == pytest.approx(expected.x, rel=1e-7)
    assert objective(actual) == pytest.approx(expected.fun, rel=1e-12)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("preconditioned", [False, True])
def test_bound_active_cgls_matches_bvls_and_projected_kkt(sign, preconditioned):
    case = bounded_case()
    case.measurement.delta_tof_s *= sign
    # Rescale both image and time units before asking SciPy for a reference.
    matrix = np.diag([0.01, 1.0])
    data = sign * np.array([100.0, 0.25])
    expected = lsq_linear(matrix, data, bounds=(-1, 1), method="bvls", tol=1e-13)
    assert expected.success
    result = run(
        case, {**parameters(False), "coverage_preconditioning": preconditioned}
    )
    x = perturbation(result).ravel() / 1e-5
    np.testing.assert_allclose(x, expected.x, rtol=1e-9, atol=1e-10)
    gradient = matrix.T @ (matrix @ x - data)
    kkt = x - np.clip(x - gradient, -1, 1)
    assert np.linalg.norm(kkt, np.inf) < 1e-10
    assert result.metrics["stop_reason"] != "line_search_failed"
    assert result.metrics["solver_optimality"]["converged"]


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("unit_scale", [1e-6, 1.0, 1e6])
def test_stationarity_is_relative_not_an_absolute_physical_gradient_cutoff(
    robust, unit_scale
):
    case = huber_case() if robust else bounded_case()
    case.grid.spacing_m = tuple(np.array(case.grid.spacing_m) * unit_scale)
    case.geometry.tx_pos_m *= unit_scale
    case.geometry.rx_pos_m *= unit_scale
    case.measurement.delta_tof_s *= unit_scale
    params = parameters(robust)
    if robust:
        params["huber_delta_s"] *= unit_scale
    result = run(case, params)
    expected = [[1.8e-5]] if robust else [[1e-5, 2.5e-6]]
    np.testing.assert_allclose(perturbation(result), expected, rtol=1e-9, atol=1e-15)
    certificate = result.metrics["solver_optimality"]
    assert result.metrics["iterations"] >= 2
    assert certificate["verified"] and certificate["converged"]
    initial_norm = certificate["initial_projected_gradient_norm"]
    assert initial_norm > 0
    assert certificate["projected_gradient_tolerance"] == pytest.approx(
        initial_norm * 1e-10, rel=1e-12, abs=0
    )
    assert certificate["projected_gradient_norm"] / initial_norm <= 1e-10


@pytest.mark.parametrize("robust", [False, True])
def test_coupled_weighted_bounded_penalty_matches_independent_lbfgsb(robust):
    case = bounded_case()
    case.geometry.tx_pos_m = np.array([[0.0005, -0.001], [-0.001, 0.0005]])
    case.geometry.rx_pos_m = np.array([[0.0005, 0.003], [0.002, 0.0005]])
    case.measurement.delta_tof_s = np.diag([3e-8, -0.25e-8])
    case.measurement.ray_weights = np.diag([0.4, 1.0])
    matrix = np.array([[1.0, 1.0], [1.0, 0.0]])
    weights = np.array([0.4, 1.0])

    def objective(x):
        residual = matrix @ x - np.array([3.0, -0.25])
        if robust:
            loss = np.where(abs(residual) <= 1, 0.5 * residual**2, abs(residual) - 0.5)
            derivative = np.clip(residual, -1, 1)
        else:
            loss, derivative = 0.5 * residual**2, residual
        return (
            weights @ loss + 0.1 * (x @ x),
            matrix.T @ (weights * derivative) + 0.2 * x,
        )

    expected = minimize(
        objective,
        np.zeros(2),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-1, 1)] * 2,
        options={"gtol": 1e-12, "ftol": 1e-15},
    )
    assert expected.success
    params = {
        **parameters(False),
        "regularization_lambda": np.sqrt(0.2e-6),
    }
    if robust:
        params.update(robust_loss="huber", huber_delta_s=1e-8, irls_max_inner_steps=34)
    result = run(case, params, run_controls={"max_iterations": 100})
    x = perturbation(result).ravel() / 1e-5
    np.testing.assert_allclose(x, expected.x, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(x, [0.125 if robust else 0.34375, 1.0], rtol=1e-8)
    gradient = objective(x)[1]
    assert np.linalg.norm(x - np.clip(x - gradient, -1, 1), np.inf) < 1e-9
    assert result.metrics["solver_optimality"]["converged"]
    costs = np.array([row["objective"] for row in result.metrics["iteration_history"]])
    assert np.all(np.diff(costs) <= 1e-12 * costs[:-1])


@pytest.mark.parametrize("iterations", [0, 1, 2])
def test_irls_restarts_do_not_extend_iteration_budget(iterations):
    result = run(
        huber_case(), parameters(True), run_controls={"max_iterations": iterations}
    )
    assert result.metrics["iterations"] <= iterations
    assert result.metrics["stop_reason"] == "max_iterations"
    assert not result.metrics["solver_optimality"]["converged"]


def heldout_case():
    case = huber_case()
    case.geometry.rx_pos_m = np.vstack(
        [case.geometry.rx_pos_m, case.geometry.rx_pos_m[0]]
    )
    case.measurement.delta_tof_s = np.array([[1e-8, 0, 1.5e-8], [0, 4e-8, 0]])
    case.measurement.valid_mask = np.array([[True, False, True], [False, True, False]])
    case.measurement.ray_weights = np.ones((2, 3))
    return case


def test_huber_reweighting_and_optimality_exclude_heldout_values_and_weights():
    case = heldout_case()
    params = {
        **parameters(True),
        "coverage_preconditioning": True,
    }
    first = run(case, params, evaluation={"receiver_indices": [2]})
    changed = case.model_copy(deep=True)
    changed.measurement.delta_tof_s[:, 2] += 1e-4
    changed.measurement.ray_weights[:, 2] = 0.125
    second = run(changed, params, evaluation={"receiver_indices": [2]})
    np.testing.assert_array_equal(first.sound_speed_mps, second.sound_speed_mps)
    np.testing.assert_array_equal(
        first.metrics["residual_curve"], second.metrics["residual_curve"]
    )
    assert first.metrics["solver_optimality"] == second.metrics["solver_optimality"]
    assert first.metrics["stopping"]["work"] == second.metrics["stopping"]["work"]


@pytest.mark.parametrize("kind", ["identity", "laplacian"])
@pytest.mark.parametrize("restricted", [False, True])
@pytest.mark.parametrize("preconditioned", [False, True])
def test_quadratic_weights_roi_and_penalty_match_dense_augmented_reference(
    kind, restricted, preconditioned
):
    rng = np.random.default_rng(904)
    shape = (2, 3)
    matrix = rng.uniform(0.01, 1, (12, 6))
    observed = rng.normal(size=(12, 1))
    observed[2] = np.nan
    weights = rng.uniform(0.1, 1, (12, 1))
    weights[4] = 0
    roi = (
        np.array([[True, False, True], [True, True, False]])
        if restricted
        else np.ones(shape, bool)
    )
    penalty = np.eye(6)
    if kind == "laplacian":
        penalty = np.zeros((6, 6))
        for y, x in np.ndindex(shape):
            i = y * shape[1] + x
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < shape[0] and 0 <= xx < shape[1]:
                    penalty[i, yy * shape[1] + xx] = 1
                    penalty[i, i] -= 1

    class DenseOperator:
        def forward(self, x):
            return matrix @ x.ravel()

        def adjoint(self, data):
            return (matrix.T @ data.ravel()).reshape(shape)

    case = SimpleNamespace(
        geometry=SimpleNamespace(tx_pos_m=None, rx_pos_m=None),
        grid=SimpleNamespace(roi_mask=roi),
        ground_truth=SimpleNamespace(sound_speed_mps=None),
    )
    config = AlgorithmConfig(
        parameters={},
        run_controls={
            "max_iterations": 30,
            "update_rtol": None,
            "objective_rtol": None,
        },
    )
    control = InversionControl(
        case,
        prepare_config("straight_cgls", config),
        observed,
        weights=weights,
    )
    actual, metrics = projected_cg(
        DenseOperator(),
        control,
        initial=np.zeros(shape),
        reference=np.ones(shape),
        lower=-np.inf,
        upper=np.inf,
        damping=0.4,
        regularization=kind,
        roi=roi,
        regularization_roi=roi,
        preconditioner=rng.uniform(0.2, 3, shape) if preconditioned else None,
    )
    rows = np.isfinite(observed.ravel()) & (weights.ravel() > 0)
    columns = roi.ravel()
    design = np.vstack(
        [
            np.sqrt(weights[rows]) * matrix[rows][:, columns],
            np.sqrt(0.4) * penalty[:, columns],
        ]
    )
    target = np.r_[np.sqrt(weights[rows].ravel()) * observed[rows].ravel(), np.zeros(6)]
    expected = np.linalg.lstsq(design, target, rcond=None)[0]
    np.testing.assert_allclose(actual.ravel()[columns], expected, rtol=1e-9, atol=1e-11)
    assert metrics["stop_reason"] == "stationary_gradient"
    assert metrics["solver_optimality"]["verified"]
    assert metrics["solver_optimality"]["converged"]
    np.testing.assert_array_equal(actual[~roi], 0)
    assert metrics["stop_reason"] in {"stationary_gradient", "max_iterations"}


class Identity:
    def forward(self, x):
        return x.copy()

    def adjoint(self, y):
        return y.copy()


def test_bounded_complex_fixed_born_offset_has_analytic_kkt_solution():
    matrix = (0.6 + 0.8j) * np.array([[1.0, 8.0], [0.0, 1.0]])
    offset = np.array([[2e-8j], [3e-8]])
    observed = offset + (0.6 + 0.8j) * 1e-9 * np.array([[-8.0], [65.0]])

    class ComplexOperator:
        def forward(self, x):
            return matrix @ x.ravel()

        def adjoint(self, y):
            return (matrix.conj().T @ y.ravel()).real.reshape(1, 2)

    case = SimpleNamespace(
        geometry=SimpleNamespace(tx_pos_m=None, rx_pos_m=None),
        grid=SimpleNamespace(roi_mask=None),
    )
    config = prepare_config(
        "straight_cgls", AlgorithmConfig(run_controls={"max_iterations": 10})
    )
    control = InversionControl(case, config, observed)
    actual, metrics = projected_cg(
        ComplexOperator(),
        control,
        initial=np.zeros((1, 2)),
        offset=offset,
        reference=np.full((1, 2), 1700.0**-2),
        lower=0,
        upper=1300.0**-2 - 1700.0**-2,
    )
    np.testing.assert_allclose(actual, [[0, 1e-9 / 65]], rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(
        control.last_prediction, offset + (matrix @ actual.ravel())[:, None]
    )
    costs = np.array([row["objective"] for row in metrics["iteration_history"]])
    assert np.all(np.diff(costs) <= 0)
    assert metrics["solver_optimality"]["converged"]


def test_exact_data_fit_does_not_bypass_regularizer():
    from usctbench.algorithms.parameters import CGLSParameters

    # min 1/2 ||x-1||^2 + 1/2 ||x||^2 has x*=1/2, not x=1.
    case = USCTCase(
        case_id="regularized_exact_fit",
        measurement=MeasurementSpec(domain="features", delta_tof_s=np.ones((2, 2))),
        grid=GridSpec(shape=(2, 2), spacing_m=(1, 1)),
        geometry=GeometrySpec(tx_pos_m=[[0, 0], [1, 0]], rx_pos_m=[[0, 1], [1, 1]]),
    )
    control = InversionControl(
        case,
        AlgorithmConfig(
            parameters=CGLSParameters(),
            run_controls={
                "max_iterations": 4,
                "objective_rtol": None,
                "update_rtol": None,
                "restore_best_validation": False,
            },
        ),
        np.ones((2, 2)),
    )
    result, metrics = projected_cg(
        Identity(),
        control,
        initial=np.ones((2, 2)),
        reference=np.zeros((2, 2)),
        lower=-np.inf,
        upper=np.inf,
        damping=1,
    )
    np.testing.assert_allclose(result, 0.5, atol=1e-15)
    assert metrics["iteration_history"][-1]["objective"] == 1
    assert metrics["stopping"]["reason"] != "exact_data_fit"


def small_case():
    return make_sound_speed_case(shape=(8, 8), n_transducers=8, inclusion_mps=1470)


def config(
    algorithm=None,
    *,
    max_iterations=4,
    run_controls=None,
    evaluation=None,
    **parameters,
):
    return AlgorithmConfig(
        parameters={
            **parameters,
        },
        run_controls={"max_iterations": max_iterations, **(run_controls or {})},
        evaluation=evaluation or {},
    )


def test_huber_preconditioning_uses_only_training_data_and_global_budget():
    case = small_case()
    cfg = config(
        evaluation={"receiver_indices": [2]},
        robust_loss="huber",
        huber_delta_s=1e-8,
        irls_max_inner_steps=2,
        coverage_preconditioning=True,
    )
    first = StraightRayCGLSAlgorithm().run(
        case, prepare_config(StraightRayCGLSAlgorithm().name, cfg, case=case)
    )
    changed_data = case.measurement.delta_tof_s.copy()
    changed_data[:, 2] *= 30
    second = StraightRayCGLSAlgorithm().run(
        case.model_copy(
            update={
                "measurement": case.measurement.model_copy(
                    update={"delta_tof_s": changed_data}
                )
            }
        ),
        prepare_config(
            StraightRayCGLSAlgorithm().name,
            cfg,
            case=case.model_copy(
                update={
                    "measurement": case.measurement.model_copy(
                        update={"delta_tof_s": changed_data}
                    )
                }
            ),
        ),
    )
    assert first.failure_reason is None, first.failure_reason
    np.testing.assert_array_equal(first.sound_speed_mps, second.sound_speed_mps)
    assert first.metrics["iterations"] <= 4
    costs = [row["objective"] for row in first.metrics["iteration_history"]]
    assert np.all(np.diff(costs) <= 1e-25)
    assert first.metrics["objective_name"] == "weighted_huber"


def test_bound_projected_metrics_correspond_to_returned_image():
    algorithm = StraightRayCGLSAlgorithm
    case = small_case()
    cfg = config(algorithm, sound_speed_bounds_mps=[1499, 1501])
    result = algorithm().run(case, prepare_config(algorithm().name, cfg, case=case))
    assert result.failure_reason is None, result.failure_reason
    assert np.min(result.sound_speed_mps) >= 1499 - 1e-10
    assert np.max(result.sound_speed_mps) <= 1501 + 1e-10
    op = StraightRayProjector.from_case(case)
    residual = (
        op.forward(1 / result.sound_speed_mps - 1 / 1500).reshape(op.ray_shape)
        - case.measurement.delta_tof_s
    )
    np.testing.assert_allclose(
        np.linalg.norm(residual[case.measurement.valid_mask]),
        result.metrics["data_residual_norm"],
        rtol=1e-9,
        atol=1e-16,
    )
