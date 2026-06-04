from __future__ import annotations

import zipfile

import h5py
import numpy as np

from usctbench.core.io import read_case_hdf5
from usctbench.data.nbpslice2d import inspect_nbp_slice2d_zip, make_nbp_slice2d_smoke_subset
from usctbench.data.synthetic import make_attenuation_case, make_sound_speed_case, make_synthetic_smoke_subset


def test_synthetic_cases_have_algorithm_ready_features():
    sos_case = make_sound_speed_case(shape=(10, 10), n_transducers=8)
    attenuation_case = make_attenuation_case(shape=(10, 10), n_transducers=8)

    assert sos_case.measurement.delta_tof_s.shape == (8, 8)
    assert attenuation_case.measurement.log_amp.shape == (8, 8)
    assert sos_case.ground_truth.sound_speed_mps.shape == (10, 10)


def test_make_synthetic_smoke_subset_writes_hdf5_cases(tmp_path):
    records = make_synthetic_smoke_subset(tmp_path / "synthetic", shape=(8, 8), n_transducers=8)

    assert len(records) == 2
    case = read_case_hdf5(tmp_path / "synthetic" / "cases" / "synthetic_circular_sos.h5")
    assert case.grid.shape == (8, 8)


def test_nbpslice2d_zip_inspection_and_smoke_conversion(tmp_path):
    zip_path = tmp_path / "NBPslices2D.zip"
    mat_path = tmp_path / "A_sample.mat"
    with h5py.File(mat_path, "w") as handle:
        handle.create_dataset("sos", data=np.full((4, 4), 1.5))
        handle.create_dataset("den", data=np.ones((4, 4)))
        handle.create_dataset("att", data=np.zeros((4, 4)))
        handle.create_dataset("label", data=np.ones((4, 4), dtype=np.uint8))
        handle.create_dataset("type", data=np.asarray([ord("A")], dtype=np.uint8))
        handle.create_dataset("y", data=np.asarray([1.0]))
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(mat_path, "A/A_sample.mat")

    index = inspect_nbp_slice2d_zip(zip_path, tmp_path / "index.json")
    manifest = make_nbp_slice2d_smoke_subset(zip_path, tmp_path / "nbp", converted_shape=(4, 4), n_transducers=8)

    assert index["summary"]["num_cases"] == 1
    assert manifest["converted_cases"]
    assert (tmp_path / "nbp" / "cases").exists()
