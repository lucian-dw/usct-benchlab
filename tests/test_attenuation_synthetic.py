from __future__ import annotations

from usctbench.algorithms.ray._common import target_attenuation_integral, valid_ray_mask
from usctbench.algorithms.ray.attenuation import AttenuationSIRTAlgorithm
from usctbench.algorithms.ray.straight_projector import StraightRayProjector
from usctbench.data.synthetic import make_attenuation_case
from usctbench.schema import AlgorithmConfig


def test_attenuation_sirt_reduces_log_amplitude_residual():
    case = make_attenuation_case(shape=(18, 18), n_transducers=24)
    projector = StraightRayProjector.from_case(case)
    target, mask = target_attenuation_integral(case, projector)
    initial_norm = float(((target[mask]) ** 2).sum() ** 0.5)

    result = AttenuationSIRTAlgorithm().run(
        case,
        AlgorithmConfig(parameters={"iterations": 40, "relaxation": 0.8}),
    )
    final = projector.forward(result.attenuation_np_per_m)
    final_norm = float((((target - final)[valid_ray_mask(case, projector)]) ** 2).sum() ** 0.5)

    assert result.failure_reason is None
    assert final_norm < 0.8 * initial_norm

