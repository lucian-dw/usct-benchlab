from __future__ import annotations

import numpy as np
import pytest

from usctbench.io.hdf5 import read_case_hdf5, read_result_hdf5, write_case_hdf5, write_result_hdf5
from usctbench.schema import (
    GeometrySpec,
    GridSpec,
    GroundTruthSpec,
    MeasurementSpec,
    ReconstructionResult,
    USCTCase,
)

pytest.importorskip("h5py")


def _tiny_case() -> USCTCase:
    shape = (4, 5)
    tx_pos = np.array([[0.0, -0.03], [0.0, 0.03]])
    rx_pos = np.array([[-0.02, 0.0], [0.02, 0.0], [0.0, 0.02]])
    delta_tof_s = np.array([[1.0e-7, 2.0e-7, 3.0e-7], [4.0e-7, 5.0e-7, 6.0e-7]])
    return USCTCase(
        case_id="synthetic_tiny",
        grid=GridSpec(
            shape=shape,
            spacing_m=(5.0e-4, 5.0e-4),
            origin_m=(-1.0e-3, -1.25e-3),
            roi_mask=np.ones(shape, dtype=bool),
        ),
        geometry=GeometrySpec(type="custom", tx_pos_m=tx_pos, rx_pos_m=rx_pos),
        measurement=MeasurementSpec(
            domain="features",
            delta_tof_s=delta_tof_s,
            valid_mask=np.ones_like(delta_tof_s, dtype=bool),
        ),
        ground_truth=GroundTruthSpec(sound_speed_mps=np.full(shape, 1500.0)),
        metadata={"source": "unit-test", "units": {"sound_speed": "m/s"}},
    )


def test_usct_case_hdf5_roundtrip(tmp_path):
    case = _tiny_case()
    path = write_case_hdf5(case, tmp_path / "case.h5")

    loaded = read_case_hdf5(path)

    assert loaded.case_id == case.case_id
    assert loaded.grid.shape == case.grid.shape
    assert loaded.grid.spacing_m == case.grid.spacing_m
    assert loaded.metadata == case.metadata
    np.testing.assert_array_equal(loaded.grid.roi_mask, case.grid.roi_mask)
    np.testing.assert_allclose(loaded.geometry.tx_pos_m, case.geometry.tx_pos_m)
    np.testing.assert_allclose(loaded.geometry.rx_pos_m, case.geometry.rx_pos_m)
    np.testing.assert_allclose(loaded.measurement.delta_tof_s, case.measurement.delta_tof_s)
    np.testing.assert_array_equal(loaded.measurement.valid_mask, case.measurement.valid_mask)
    np.testing.assert_allclose(loaded.ground_truth.sound_speed_mps, case.ground_truth.sound_speed_mps)


def test_reconstruction_result_hdf5_roundtrip(tmp_path):
    result = ReconstructionResult(
        algorithm="unit_baseline",
        case_id="synthetic_tiny",
        sound_speed_mps=np.full((4, 5), 1501.0),
        metrics={"rmse_mps": 1.0},
        runtime_s=0.25,
        artifacts={"preview": "preview.png"},
    )

    path = write_result_hdf5(result, tmp_path / "result.h5")
    loaded = read_result_hdf5(path)

    assert loaded.algorithm == result.algorithm
    assert loaded.case_id == result.case_id
    assert loaded.metrics == result.metrics
    assert loaded.artifacts == result.artifacts
    assert loaded.runtime_s == result.runtime_s
    np.testing.assert_allclose(loaded.sound_speed_mps, result.sound_speed_mps)

