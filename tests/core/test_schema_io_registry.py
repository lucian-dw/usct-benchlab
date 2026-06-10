from __future__ import annotations

import numpy as np
import pytest
from skimage.metrics import structural_similarity

from usctbench.core.io import read_case_hdf5, read_result_hdf5, write_result_hdf5
from usctbench.core.registry import (
    clear_registry,
    get_algorithm,
    list_algorithms,
    register_algorithm,
)
from usctbench.core.schema import (
    AlgorithmConfig,
    MeasurementDomain,
    MeasurementSpec,
    ReconstructionResult,
    USCTCase,
)
from usctbench.metrics import compute_image_metrics


class DummyAlgorithm:
    name = "dummy"

    def run(self, case, config):
        return ReconstructionResult(
            algorithm=self.name, case_id=case.case_id, metrics={"ok": True}
        )


def test_case_and_result_hdf5_roundtrip(written_case, tmp_path):
    case = read_case_hdf5(written_case)
    assert case.case_id == "synthetic_circular_sos"
    assert case.measurement.delta_tof_s is not None

    result = DummyAlgorithm().run(case, AlgorithmConfig(parameters={}))
    result_path = write_result_hdf5(result, tmp_path / "result.h5")
    loaded = read_result_hdf5(result_path)

    assert loaded.algorithm == "dummy"
    assert loaded.case_id == case.case_id
    assert loaded.metrics["ok"] is True


def test_registry_registers_and_instantiates_algorithm():
    clear_registry()
    register_algorithm("dummy", DummyAlgorithm)

    assert [entry.name for entry in list_algorithms()] == ["dummy"]
    assert get_algorithm("dummy").name == "dummy"

    clear_registry()


def test_case_schema_rejects_feature_shape_mismatch(synthetic_case):
    measurement = MeasurementSpec(
        domain=MeasurementDomain.FEATURES, delta_tof_s=np.zeros((3, 3))
    )

    with pytest.raises(ValueError, match="measurement.delta_tof_s"):
        USCTCase(
            case_id="bad_feature_shape",
            grid=synthetic_case.grid,
            geometry=synthetic_case.geometry,
            measurement=measurement,
            ground_truth=synthetic_case.ground_truth,
        )


def test_compute_image_metrics_uses_standard_ssim():
    truth = np.arange(25, dtype=float).reshape(5, 5)
    prediction = truth.copy()
    prediction[2, 2] += 1.0

    metrics = compute_image_metrics(prediction, truth)

    assert np.isclose(
        metrics["ssim"],
        structural_similarity(truth, prediction, data_range=24.0, win_size=5),
    )
    assert "global_ssim" in metrics
