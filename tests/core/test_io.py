from __future__ import annotations

import numpy as np
import h5py
import pytest

from usctbench.core.io import (
    read_case_hdf5,
    read_result_hdf5,
    write_result_hdf5,
    write_case_hdf5,
)
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    MeasurementDomain,
    MeasurementSpec,
    ReconstructionResult,
    USCTCase,
)
from usctbench.data.pressure import dtft_pressure_contract


class DummyAlgorithm:
    name = "dummy"

    def run(self, case, config):
        return ReconstructionResult(
            algorithm=self.name, case_id=case.case_id, metrics={"ok": True}
        )


@pytest.mark.parametrize(
    "experiment_id,variant", [(None, None), ("reference_run", "wkb_fixed")]
)
def test_case_and_result_hdf5_roundtrip(written_case, tmp_path, experiment_id, variant):
    case = read_case_hdf5(written_case)
    assert case.case_id == "synthetic_circular_sos"
    assert case.measurement.delta_tof_s is not None

    result = DummyAlgorithm().run(case, AlgorithmConfig(parameters={}))
    result.experiment_id = experiment_id
    result.variant = variant
    result_path = write_result_hdf5(result, tmp_path / "result.h5")
    with h5py.File(result_path, "r") as handle:
        assert (handle.attrs.get("experiment_id"), handle.attrs.get("variant")) == (
            experiment_id,
            variant,
        )
    loaded = read_result_hdf5(result_path)

    assert loaded.algorithm == "dummy"
    assert loaded.case_id == case.case_id
    assert (loaded.experiment_id, loaded.variant) == (experiment_id, variant)
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
            source_spectrum=np.full((2, 3), 2 + 3j),
            valid_mask=valid,
        ),
        metadata={"pressure_contract": dtft_pressure_contract("Pa*s")},
    )
    path = write_case_hdf5(case, tmp_path / "complex.h5")
    restored = read_case_hdf5(path)
    np.testing.assert_array_equal(restored.measurement.freq_data, pressure)
    np.testing.assert_array_equal(restored.measurement.valid_mask, valid)
    assert restored.grid == case.grid
    assert restored.measurement.domain.value == "frequency"
    np.testing.assert_array_equal(
        restored.measurement.source_spectrum, case.measurement.source_spectrum
    )
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
