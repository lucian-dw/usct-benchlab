from __future__ import annotations

from usctbench.algorithms.attenuation import AttenuationSIRTAlgorithm
from usctbench.algorithms.bent_ray import BentRayGNAdapter
from usctbench.algorithms.ray import (
    StraightRayCGLSAlgorithm,
    StraightRayProjector,
    StraightRaySARTAlgorithm,
    StraightRaySIRTAlgorithm,
)
from usctbench.algorithms.rwave import RWaveAdapter
from usctbench.core.schema import AlgorithmConfig, ResultStatus
from usctbench.data.synthetic import make_attenuation_case


def test_projector_adjoint_identity(synthetic_case):
    projector = StraightRayProjector.from_case(synthetic_case)
    image = synthetic_case.ground_truth.sound_speed_mps
    ray_values = projector.forward(image)

    lhs = float((projector.forward(image) * ray_values).sum())
    rhs = float((image * projector.adjoint(ray_values)).sum())

    assert abs(lhs - rhs) / max(abs(lhs), 1.0) < 1.0e-10


def test_ray_sound_speed_algorithms_run(synthetic_case):
    config = AlgorithmConfig(
        parameters={
            "iterations": 3,
            "subsets": 4,
            "inner_iterations": 4,
            "outer_iterations": 1,
        }
    )

    for algorithm in (
        StraightRayCGLSAlgorithm(),
        StraightRaySIRTAlgorithm(),
        StraightRaySARTAlgorithm(),
        BentRayGNAdapter(),
        RWaveAdapter(),
    ):
        result = algorithm.run(synthetic_case, config)
        assert result.status == ResultStatus.SUCCESS
        assert result.sound_speed_mps is not None
        assert "data_relative_residual" in result.metrics


def test_adapter_baseline_metadata_is_explicit(synthetic_case):
    config = AlgorithmConfig(parameters={"outer_iterations": 1, "inner_iterations": 3})

    bent = BentRayGNAdapter().run(synthetic_case, config)
    rwave = RWaveAdapter().run(synthetic_case, config)

    assert bent.metrics["full_external_eikonal_solver"] is False
    assert bent.metrics["v0_1_backend"] == "regularized_travel_time_baseline"
    assert rwave.metrics["full_ray_born_solver"] is False
    assert rwave.metrics["v0_1_backend"] == "adapter_style_travel_time_baseline"
    assert rwave.metrics["ray_born_inspired"] is True
    assert "ray_born_linearization" not in rwave.metrics


def test_attenuation_algorithm_runs_on_log_amplitude_case():
    case = make_attenuation_case(shape=(10, 10), n_transducers=8)
    result = AttenuationSIRTAlgorithm().run(
        case, AlgorithmConfig(parameters={"iterations": 3})
    )

    assert result.status == ResultStatus.SUCCESS
    assert result.attenuation_np_per_m is not None
    assert result.metrics["attenuation_input_has_signal"] is True
