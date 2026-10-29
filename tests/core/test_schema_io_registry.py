from __future__ import annotations

import numpy as np
import h5py
import pytest
from skimage.metrics import structural_similarity

from usctbench.core.io import (
    read_case_hdf5,
    read_result_hdf5,
    write_result_hdf5,
    write_case_hdf5,
)
from usctbench.core.registry import (
    clear_registry,
    get_algorithm,
    list_algorithms,
    register_algorithm,
)
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    GroundTruthSpec,
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


def test_complex_layout_units_and_valid_zero_roundtrip(tmp_path):
    pressure = np.arange(24).reshape(2, 3, 4).astype(complex) * (1 + 2j)
    valid = np.ones(pressure.shape, dtype=bool)
    valid[1, 2, 3] = False
    pressure[1, 2, 3] = np.nan
    case = USCTCase(
        case_id="rectangular",
        grid=GridSpec(shape=(7, 9), spacing_m=(0.001, 0.002), origin_m=(0.02, -0.01)),
        geometry=GeometrySpec(tx_pos_m=np.zeros((3, 2)), rx_pos_m=np.ones((4, 2))),
        measurement=MeasurementSpec(
            domain="frequency",
            frequencies_hz=[1e5, 2e5],
            freq_data=pressure,
            valid_mask=valid,
        ),
        metadata={"frequency_data_units": "Pa*s"},
    )
    path = write_case_hdf5(case, tmp_path / "complex.h5")
    restored = read_case_hdf5(path)
    np.testing.assert_array_equal(restored.measurement.freq_data, pressure)
    np.testing.assert_array_equal(restored.measurement.valid_mask, valid)
    assert restored.grid == case.grid
    assert (
        restored.measurement.freq_data[0, 0, 0] == 0
        and restored.measurement.valid_mask[0, 0, 0]
    )
    with h5py.File(path, "a") as handle:
        assert handle["measurement/freq_data"].attrs["units"] == "Pa*s"
        assert handle["measurement/freq_data"].attrs["axes"] == "frequency,tx,rx"
        handle["measurement/freq_data"].attrs["axes"] = "tx,rx,frequency"
    with pytest.raises(ValueError, match="axes"):
        read_case_hdf5(path)


def test_provenance_is_not_inferred_from_descriptive_text():
    from usctbench.core.provenance import case_measurement_metadata

    facts = case_measurement_metadata(
        {"source_dataset": "kwave", "case_type": "oracle"}
    )
    assert facts["measurement_provenance"] == "unknown"


def test_sound_speed_contract_has_no_material_absorption_output(written_case, tmp_path):
    assert "log_amp" not in MeasurementSpec.model_fields
    assert "attenuation_np_per_m" not in GroundTruthSpec.model_fields
    assert "attenuation_np_per_m" not in ReconstructionResult.model_fields
    with pytest.raises(ValueError, match="feature-domain"):
        MeasurementSpec(domain="features", log_amp=np.ones((8, 8)))
    case = read_case_hdf5(written_case)
    truth = case.ground_truth.sound_speed_mps
    result = ReconstructionResult(
        algorithm="straight_cgls", case_id=case.case_id, sound_speed_mps=truth
    )
    path = write_result_hdf5(result, tmp_path / "sos_result.h5")
    np.testing.assert_array_equal(read_result_hdf5(path).sound_speed_mps, truth)
    for file in (written_case, path):
        with h5py.File(file) as handle:
            names = []
            handle.visit(names.append)
            assert not any("attenuation" in name or "log_amp" in name for name in names)
    # No old attenuation-only feature reader remains.
    with h5py.File(written_case, "a") as handle:
        del handle["measurement/delta_tof_s"]
        handle["measurement/log_amp"] = np.ones((8, 8))
    with pytest.raises(ValueError, match="feature-domain"):
        read_case_hdf5(written_case)


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
