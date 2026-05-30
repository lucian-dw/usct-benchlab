from __future__ import annotations

import zipfile

import h5py
import numpy as np

from usctbench.data.conversion import convert_nbp_slice2d_mat
from usctbench.data.nbpslice2d import inspect_nbp_slice2d_zip, make_nbp_slice2d_smoke_subset
from usctbench.io.hdf5 import read_case_hdf5


def test_convert_nbp_slice2d_mat_to_standard_case(tmp_path):
    mat_path = tmp_path / "A000001.mat"
    _write_nbp_mat(mat_path, density_label="A")

    records = convert_nbp_slice2d_mat(mat_path, tmp_path / "cases", output_shape=(4, 4), n_transducers=6)

    assert len(records) == 1
    record = records[0]
    assert record["conversion"] == "nbpslice2d_to_feature_case"
    assert record["density_class"] == "almost_entirely_fatty"
    assert record["has_simulated_attenuation"] is True
    case = read_case_hdf5(record["path"])
    assert case.case_id == "A000001"
    assert case.grid.shape == (4, 4)
    assert case.grid.spacing_m == (2.0e-4, 2.0e-4)
    assert case.grid.roi_mask is not None
    assert case.measurement.delta_tof_s.shape == (6, 6)
    assert case.measurement.log_amp.shape == (6, 6)
    assert case.ground_truth.sound_speed_mps is not None
    assert float(np.nanmin(case.ground_truth.sound_speed_mps)) > 1400.0
    assert case.ground_truth.attenuation_np_per_m is not None
    assert float(np.nanmax(case.ground_truth.attenuation_np_per_m)) > 0.0
    assert case.metadata["source_dataset"] == "NBPslices2D"
    assert case.metadata["attenuation_source_units"] == "dB/(MHz^y mm)"


def test_make_nbp_slice2d_smoke_subset_from_zip(tmp_path):
    zip_path = tmp_path / "NBPslices2D.zip"
    source_a = tmp_path / "A000001.mat"
    source_b = tmp_path / "B000001.mat"
    _write_nbp_mat(source_a, density_label="A")
    _write_nbp_mat(source_b, density_label="B")
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(source_b, "NBPslices2D/B000001.mat")
        archive.write(source_a, "NBPslices2D/A000001.mat")
        archive.writestr("__MACOSX/._ignored.mat", b"ignored")

    index = inspect_nbp_slice2d_zip(zip_path)
    manifest = make_nbp_slice2d_smoke_subset(zip_path, tmp_path / "sample", converted_shape=(4, 4), n_transducers=6)

    assert index["summary"]["num_cases"] == 2
    assert index["summary"]["density_label_counts"] == {"A": 1, "B": 1}
    assert len(manifest["converted_cases"]) == 2
    assert (tmp_path / "sample" / "nbpslice2d_index.json").exists()
    assert (tmp_path / "sample" / "nbpslice2d_smoke_manifest.json").exists()
    assert sorted(case["case_id"] for case in manifest["converted_cases"]) == ["A000001", "B000001"]


def _write_nbp_mat(path, *, density_label: str) -> None:
    sound_speed = np.full((8, 8), 1.500, dtype=np.float32)
    sound_speed[2:6, 2:6] = 1.455
    attenuation = np.full((8, 8), 0.00022, dtype=np.float32)
    attenuation[2:6, 2:6] = 0.020
    density = np.full((8, 8), 1.0e-6, dtype=np.float32)
    label = np.zeros((8, 8), dtype=np.uint8)
    label[2:6, 2:6] = 100
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sos", data=sound_speed)
        handle.create_dataset("att", data=attenuation)
        handle.create_dataset("den", data=density)
        handle.create_dataset("label", data=label)
        handle.create_dataset("type", data=np.asarray([[ord(density_label)]], dtype=np.uint16))
        handle.create_dataset("y", data=np.asarray([[1.1]], dtype=np.float64))
