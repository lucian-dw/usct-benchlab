from __future__ import annotations

from usctbench.algorithms.fwi.tiny_fwi import TinyFWIAlgorithm
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.schema import AlgorithmConfig


def test_tiny_fwi_loss_decreases_on_synthetic_case():
    case = make_sound_speed_case(shape=(12, 12), n_transducers=16, inclusion_mps=1450.0)

    result = TinyFWIAlgorithm().run(
        case,
        AlgorithmConfig(
            parameters={
                "frequencies_hz": [1.0e5, 1.5e5, 2.0e5],
                "initial_sound_speed_mps": 1500.0,
                "steps": 20,
                "learning_rate": 1.0e6,
            }
        ),
    )

    assert result.failure_reason is None
    assert result.metrics["loss_decreased"] is True
    assert result.metrics["final_loss"] < result.metrics["initial_loss"]
    assert "water_relative_rmse_improvement" in result.metrics
