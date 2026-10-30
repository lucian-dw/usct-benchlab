import numpy as np

from usctbench.core.config import prepare_config
import pytest

from usctbench.algorithms.straight import (
    StraightRayCGLSAlgorithm,
    StraightRaySIRTAlgorithm,
    StraightRaySARTAlgorithm,
)
from usctbench.algorithms.bent_ray import BentRayAlgorithm
from usctbench.core.schema import AlgorithmConfig, GroundTruthSpec
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.operators.eikonal import EikonalForward


def test_completed_bent_initializer_survives_first_gn_budget_exit(monkeypatch):
    from usctbench.algorithms import bent_ray

    case = make_sound_speed_case(shape=(4, 5), n_transducers=4)
    case.measurement.delta_tof_s += 1e-7
    captured = []
    real_solve = bent_ray.solve_linearized

    def solve(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        captured.append(result[0].copy())
        return result

    monkeypatch.setattr(bent_ray, "solve_linearized", solve)
    cfg = prepare_config(
        "bent_ray_gn",
        {
            "parameters": {
                "initialization": "cgls",
                "initialization_iterations": 2,
                "inner_solver": "lsmr",
                "roi_update_only": False,
            },
            "run_controls": {"max_iterations": 0, "max_forward_calls": 4},
        },
        case=case,
    )
    result = BentRayAlgorithm().run(case, cfg)
    expected = 1 / np.clip(1 / 1500 + captured[0], 1 / 1700, 1 / 1300)
    np.testing.assert_allclose(result.sound_speed_mps, expected)
    assert np.max(np.abs(expected - 1500)) > 1e-3
    assert result.metrics["stopping"]["selected_iteration"] is None


@pytest.mark.parametrize("name", ["straight_cgls", "bent_ray_gn"])
def test_model_prior_uses_no_gt_or_validation_for_updates_and_obeys_budget(name):
    case = make_sound_speed_case(
        shape=(8, 8), n_transducers=8, inclusion_radius_m=0.002, inclusion_mps=1490
    )
    if name == "bent_ray_gn":
        case.geometry.tx_pos_m *= 0.25
        case.geometry.rx_pos_m *= 0.25
        op = EikonalForward(case.grid, case.geometry)
        distance = np.linalg.norm(
            case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None], axis=-1
        )
        case.measurement.delta_tof_s = (
            op.forward(1 / case.ground_truth.sound_speed_mps).reshape(8, 8)
            - distance / 1500
        )
    case.grid.roi_mask = None
    case.ground_truth = GroundTruthSpec()
    altered = case.model_copy(deep=True)
    altered.measurement.delta_tof_s[:, 1] += 7e-6
    config = AlgorithmConfig(
        parameters={
            "model_grid_shape": [4, 4],
            **({"inner_iterations": 4} if name == "bent_ray_gn" else {}),
            **(
                {"regularization": "laplacian", "regularization_lambda": 0.0001}
                if name in {"straight_cgls", "bent_ray_gn"}
                else {}
            ),
        },
        run_controls={
            "max_iterations": 2,
            "restore_best_validation": False,
            "validation_patience": None,
            "objective_rtol": None,
            "update_rtol": None,
        },
        evaluation={"receiver_indices": [1]},
    )
    algo = {
        "straight_cgls": StraightRayCGLSAlgorithm,
        "straight_sirt": StraightRaySIRTAlgorithm,
        "straight_sart": StraightRaySARTAlgorithm,
        "bent_ray_gn": BentRayAlgorithm,
    }[name]()
    a, b = algo.run(case, prepare_config(algo.name, config, case=case)), algo.run(
        altered, prepare_config(algo.name, config, case=altered)
    )
    assert a.status == b.status in {"completed", "budget_exhausted"}, (
        a.failure_reason,
        b.failure_reason,
    )
    assert a.sound_speed_mps.shape == case.grid.shape
    assert "rmse" not in a.metrics
    np.testing.assert_array_equal(a.sound_speed_mps, b.sound_speed_mps)
    assert a.metrics["model_parameterization"]["propagation_shape"] == [8, 8]
    assert a.metrics["data_residual_reduction"] > 0


@pytest.mark.parametrize("failure", ["budget", "nonfinite"])
def test_bent_coefficient_exit_maps_selected_variable_to_physical_grid(
    monkeypatch, failure
):
    case = make_sound_speed_case(shape=(4, 5), n_transducers=4)
    case.grid.roi_mask = None
    case.ground_truth = GroundTruthSpec()
    if failure == "nonfinite":

        def fail(*args):
            raise FloatingPointError("independent injected field failure")

        monkeypatch.setattr(EikonalForward, "linearize", fail)
    config = prepare_config(
        "bent_ray_gn",
        {
            "parameters": {"model_grid_shape": [2, 3]},
            "run_controls": {
                "max_iterations": 2,
                "max_forward_calls": 0 if failure == "budget" else 2,
            },
        },
        case=case,
    )
    result = BentRayAlgorithm().run(case, config)
    np.testing.assert_array_equal(result.sound_speed_mps, np.full((4, 5), 1500.0))
    assert result.metrics["stopping"]["selected_iteration"] is None
    assert result.metrics["stop_reason"] == (
        "forward_calls_budget" if failure == "budget" else "numerical_failure"
    )
