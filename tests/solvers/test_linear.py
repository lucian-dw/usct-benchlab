"""Independent dense references for the shared matrix-free inverse solvers."""

from types import SimpleNamespace

import numpy as np
import pytest

from usctbench.core.stopping import WorkLedger
from usctbench.core.run_controls import RunControls
from usctbench.solvers.linear import solve_linearized
from usctbench.operators.regularization import regularizer
from tests.helpers import DenseJacobian


def controls(precision):
    work = WorkLedger(RunControls())
    return SimpleNamespace(
        observed=np.zeros(len(precision)),
        precision=precision,
        work=work,
        call=work.call,
    )


@pytest.mark.parametrize("method", ["normal_cg", "lsmr", "lsqr"])
@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("penalty", ["identity", "laplacian", (0.2, 1.3)])
def test_linearized_step_matches_dense_augmented_lstsq(method, complex_data, penalty):
    rng = np.random.default_rng(641)
    shape = (3, 4)
    matrix = rng.normal(size=(40, 12))
    b = rng.normal(size=40)
    if complex_data:
        matrix = matrix + 1j * rng.normal(size=matrix.shape)
        b = b + 1j * rng.normal(size=b.shape)
    weights = rng.uniform(0.2, 2, len(b))
    weights[::5] = 0
    roi = np.ones(shape, bool)
    roi[0, 0] = roi[-1, -1] = False
    current = rng.normal(size=shape)
    damping = 0.23
    reg = np.column_stack(
        [regularizer(e.reshape(shape), penalty).ravel() for e in np.eye(12)]
    )
    design = np.vstack(
        [
            np.sqrt(weights)[:, None] * matrix.real,
            np.sqrt(weights)[:, None] * matrix.imag,
            np.sqrt(damping) * reg,
        ]
    )[:, roi.ravel()]
    target = np.concatenate(
        [
            np.sqrt(weights) * b.real,
            np.sqrt(weights) * b.imag,
            -np.sqrt(damping) * reg @ current.ravel(),
        ]
    )
    expected = np.linalg.lstsq(design, target, rcond=None)[0]
    control = controls(weights)
    actual, info = solve_linearized(
        DenseJacobian(matrix, shape),
        weights * b,
        current,
        control,
        weights=weights,
        method=method,
        iterations=40,
        damping=damping,
        regularization=penalty,
        roi=roi,
    )
    np.testing.assert_allclose(actual[roi], expected, rtol=1e-9, atol=1e-10)
    np.testing.assert_array_equal(actual[~roi], 0)
    gradient = design.T @ (design @ actual[roi] - target)
    assert np.linalg.norm(gradient) < 1e-8
    assert info.converged
    assert info.relative_residual < 1e-7


@pytest.mark.parametrize("method", ["lsmr", "lsqr"])
def test_rank_deficient_masked_complex_problem_matches_minimum_norm(method):
    rng = np.random.default_rng(134)
    matrix = (rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3))) @ rng.normal(
        size=(3, 6)
    )
    matrix[-1] = np.nan
    target = rng.normal(size=8) + 1j * rng.normal(size=8)
    target[-1] = np.nan
    weights = np.ones(8)
    weights[-1] = 0

    class Masked(DenseJacobian):
        def adjoint(self, values):
            return (self.matrix[:-1].conj().T @ values[:-1]).real.reshape(self.shape)

    step, info = solve_linearized(
        Masked(matrix, (2, 3)),
        target,
        np.zeros((2, 3)),
        controls(weights),
        weights=weights,
        iterations=40,
        method=method,
    )
    design = np.vstack([matrix[:-1].real, matrix[:-1].imag])
    expected = np.linalg.lstsq(
        design, np.r_[target[:-1].real, target[:-1].imag], rcond=None
    )[0]
    np.testing.assert_allclose(step.ravel(), expected, rtol=1e-8, atol=1e-10)
    assert info.converged


@pytest.mark.parametrize("method", ["normal_cg", "lsmr", "lsqr"])
@pytest.mark.parametrize("preconditioner", ["none", "column_rms"])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_scaled_rectangular_penalty_column_preconditioning(
    method, preconditioner, scale
):
    rng = np.random.default_rng(189)
    a = (rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3))) * scale
    penalty_matrix = rng.normal(size=(5, 3)) * scale
    current = rng.normal(size=(1, 3)) * 1e-9
    residual = (rng.normal(size=8) + 1j * rng.normal(size=8)) * scale * 1e-9
    roi = np.ones((1, 3), dtype=bool)
    weight = rng.uniform(0.3, 2, 8)
    damping = 0.2
    penalty = DenseJacobian(penalty_matrix, current.shape)
    step, info = solve_linearized(
        DenseJacobian(a, current.shape),
        weight * residual,
        current,
        controls(weight),
        weights=weight,
        iterations=50,
        damping=damping,
        regularization=penalty,
        roi=roi,
        method=method,
        preconditioner=preconditioner,
    )
    design = np.vstack(
        [
            np.sqrt(weight)[:, None] * a.real,
            np.sqrt(weight)[:, None] * a.imag,
            np.sqrt(damping) * penalty_matrix,
        ]
    )[:, roi.ravel()]
    target = np.r_[
        np.sqrt(weight) * residual.real,
        np.sqrt(weight) * residual.imag,
        -np.sqrt(damping) * penalty_matrix @ current.ravel(),
    ]
    expected = np.linalg.lstsq(design, target, rcond=None)[0]
    np.testing.assert_allclose(step[roi], expected, rtol=1e-9, atol=1e-23)
    np.testing.assert_array_equal(step[~roi], 0)
    assert info.converged


def test_normal_cg_scale_preparation_counts_work_without_exit_recomputation():
    control = controls(np.ones(1))
    step, info = solve_linearized(
        DenseJacobian(np.array([[1e-6]]), (1, 1)),
        np.array([1e-15]),
        np.zeros((1, 1)),
        control,
        weights=np.ones(1),
        iterations=8,
        method="normal_cg",
    )
    assert info.converged
    assert info.stop_reason == "normal_residual_target"
    np.testing.assert_allclose(step, 1e-9, rtol=1e-14)
    # One normalization pair and one Krylov iteration. No exit J/J* pair.
    assert control.work.counts["forward_calls"] == 2
    assert control.work.counts["adjoint_calls"] == 2


def test_iteration_limit_is_not_convergence():
    rng = np.random.default_rng(87)
    a = rng.normal(size=(20, 6))
    _, info = solve_linearized(
        DenseJacobian(a, (2, 3)),
        rng.normal(size=20),
        np.zeros((2, 3)),
        controls(np.ones(20)),
        weights=np.ones(20),
        iterations=1,
    )
    assert info.iterations == 1
    assert not info.converged
    assert info.stop_reason == "iteration_limit"


@pytest.mark.parametrize("method", ["normal_cg", "lsmr", "lsqr"])
@pytest.mark.parametrize("reduced", [False, True])
def test_complex_jacobian_real_observation_has_real_half_solution(method, reduced):
    from usctbench.operators.model_space import ReducedLinearOperator

    jacobian = DenseJacobian(np.array([[1 + 1j]]), (1, 1))
    if reduced:
        jacobian.n_rays, jacobian.ray_shape = 1, (1,)
        basis = SimpleNamespace(
            grid=None, shape=(1, 1), forward=lambda x: x, adjoint=lambda x: x
        )
        jacobian = ReducedLinearOperator(jacobian, basis)
        assert jacobian.data_dtype == np.dtype(complex)
    control = controls(np.ones(1))
    result, info = solve_linearized(
        jacobian,
        np.ones(1),
        np.zeros((1, 1)),
        control,
        weights=np.ones(1),
        iterations=4,
        method=method,
    )
    np.testing.assert_allclose(result, 0.5, rtol=1e-14)
    assert info.converged
