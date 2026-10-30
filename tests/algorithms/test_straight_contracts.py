"""Measurement reference, objective and failure-status contracts."""

import numpy as np
import pytest

from usctbench.algorithms._control import InversionControl
from usctbench.core.schema import (
    AlgorithmConfig,
    GridSpec,
    GeometrySpec,
    MeasurementSpec,
    USCTCase,
)
from usctbench.solvers.row_action import row_action


@pytest.mark.parametrize("subsets,expected", [(1, 2 / 3), (2, 0.5)])
def test_inconsistent_row_action_matches_its_documented_model(subsets, expected):
    matrix = np.array([[1.0], [2.0]])

    class Operator:
        def forward(self, x):
            return matrix @ x.ravel()

        def adjoint(self, y):
            return (matrix.T @ y.ravel()).reshape(1, 1)

        def row_norms(self, power):
            return (matrix**power).sum(axis=1)

    observed = np.ones((1, 2))
    case = USCTCase(
        case_id="inconsistent_ray_reference",
        grid=GridSpec(shape=(1, 1), spacing_m=(0.001, 0.001)),
        geometry=GeometrySpec(tx_pos_m=[[-1.0, 0]], rx_pos_m=[[1.0, 0], [2.0, 0]]),
        measurement=MeasurementSpec(domain="features", delta_tof_s=observed),
    )
    config = AlgorithmConfig(
        parameters={},
        run_controls={
            "max_iterations": 40,
            "update_rtol": None,
            "objective_rtol": None,
            "restore_best_validation": False,
        },
    )
    from usctbench.core.config import prepare_config

    control = InversionControl(case, prepare_config("straight_cgls", config), observed)
    state, metrics = row_action(
        Operator(),
        control,
        initial=np.zeros((1, 1)),
        reference=np.ones((1, 1)),
        project=lambda x: x,
        relaxation=1,
        subsets=subsets,
    )
    np.testing.assert_allclose(state, expected, atol=1e-12)
    assert (
        abs(expected - np.linalg.lstsq(matrix, observed.ravel(), rcond=None)[0][0])
        > 0.05
    )
    residual = matrix.ravel() * expected - observed.ravel()
    objective = 0.5 * np.sum(residual**2 / matrix.ravel())
    assert metrics["iteration_history"][-1]["objective"] == pytest.approx(objective)
    assert metrics["objective_name"] == "row_normalized_weighted_least_squares"
    assert metrics["data_residual_norm"] == pytest.approx(np.linalg.norm(residual))
