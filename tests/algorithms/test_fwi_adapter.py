from __future__ import annotations

import h5py
import numpy as np

from usctbench.algorithms.fwi.adapter import (
    KWaveFWIAdapterAlgorithm,
    read_kwave_fwi_result,
)
from usctbench.algorithms.fwi.tiny import TinyFWIAlgorithm
from usctbench.core.schema import AlgorithmConfig, ResultStatus


def test_tiny_fwi_loss_decreases(synthetic_case):
    result = TinyFWIAlgorithm().run(
        synthetic_case,
        AlgorithmConfig(parameters={"steps": 5, "learning_rate": 1.0e6}),
    )

    assert result.status == ResultStatus.SUCCESS
    assert result.metrics["loss_decreased"] is True


def test_kwave_adapter_ingests_result_file(synthetic_case, tmp_path):
    result_path = tmp_path / "fwi_result.mat"
    with h5py.File(result_path, "w") as handle:
        handle.create_dataset("VEL_ESTIM", data=np.full((12, 12), 1490.0))
        handle.create_dataset(
            "C_INTERP", data=np.asarray(synthetic_case.ground_truth.sound_speed_mps)
        )
        handle.create_dataset("LOSS_ITER", data=np.array([3.0, 2.0, 1.0]))

    external = read_kwave_fwi_result(result_path)
    result = KWaveFWIAdapterAlgorithm().run(
        synthetic_case, AlgorithmConfig(parameters={"result_path": str(result_path)})
    )

    assert external["sound_speed_mps"].shape == (12, 12)
    assert result.status == ResultStatus.SUCCESS
    assert result.metrics["external_result_loaded"] is True
    assert result.sound_speed_mps is not None


def test_kwave_adapter_skips_missing_result(synthetic_case, tmp_path):
    result = KWaveFWIAdapterAlgorithm().run(
        synthetic_case,
        AlgorithmConfig(parameters={"result_path": str(tmp_path / "missing.mat")}),
    )

    assert result.status == ResultStatus.SKIPPED
