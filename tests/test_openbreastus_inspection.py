from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from usctbench.data.openbreastus import inspect_openbreastus
from usctbench.data.conversion import convert_speed_mat_volume, speed_mat_metadata
from usctbench.data.smoke_subset import make_smoke_subset
from usctbench.io.hdf5 import read_case_hdf5


def test_inspect_openbreastus_indexes_local_tree(tmp_path):
    root = tmp_path / "openbreastus"
    case_dir = root / "dense" / "case001"
    case_dir.mkdir(parents=True)
    np.save(case_dir / "sound_speed_500kHz.npy", np.zeros((3, 4)))
    (case_dir / "rf_data_500kHz.mat").write_text("placeholder", encoding="utf-8")

    out = tmp_path / "index.json"
    index = inspect_openbreastus(root, out)

    assert out.exists()
    assert index["summary"]["num_cases"] == 1
    assert index["summary"]["density_counts"]["dense"] == 1
    case = index["cases"][0]
    assert case["case_id"] == "dense_case001"
    assert case["density_class"] == "dense"
    assert "sound_speed" in case["roles"]
    assert 500000.0 in case["available_frequencies_hz"]
    assert case["capabilities"]["has_sound_speed"] is True
    assert case["capabilities"]["convertible_to_usct_case"] is False
    assert "no supported automatic USCTCase conversion mode was identified" in case["limitations"]
    assert json.loads(out.read_text(encoding="utf-8"))["cases"][0]["case_id"] == "dense_case001"


def test_inspect_openbreastus_records_hdf5_schema_and_capabilities(tmp_path):
    root = tmp_path / "openbreastus"
    case_dir = root / "train" / "dense" / "case001"
    case_dir.mkdir(parents=True)
    with h5py.File(case_dir / "sound_speed_500kHz.mat", "w") as handle:
        handle.create_dataset("breast_train", data=np.ones((4, 5, 2), dtype=np.float32) * 1500.0)
    with h5py.File(case_dir / "rf_reference_500kHz.h5", "w") as handle:
        handle.create_dataset("reference_wavefield", data=np.ones((2, 3, 4), dtype=np.complex64))
    (case_dir / "geometry.json").write_text("{}", encoding="utf-8")

    index = inspect_openbreastus(root)

    case = index["cases"][0]
    assert case["split"] == "train"
    assert case["capabilities"]["has_speed_mat_volume"] is True
    assert case["capabilities"]["has_reference"] is True
    assert case["capabilities"]["has_geometry"] is True
    assert "speed_map_to_straight_ray_surrogate" in case["capabilities"]["conversion_modes"]
    assert "frequency_reference_features" in case["capabilities"]["conversion_modes"]
    speed_file = [file for file in case["files"] if file["path"].endswith("sound_speed_500kHz.mat")][0]
    assert speed_file["schema"]["format"] == "mat-v7.3-hdf5"
    assert speed_file["schema"]["datasets"]["breast_train"]["shape"] == [4, 5, 2]
    assert index["summary"]["capability_counts"]["convertible_to_usct_case"] == 1


def test_make_smoke_subset_selects_one_case_per_density(tmp_path):
    root = tmp_path / "openbreastus"
    for density in ("dense", "fatty"):
        for idx in range(2):
            case_dir = root / density / f"case{idx:03d}"
            case_dir.mkdir(parents=True)
            np.save(case_dir / "sound_speed.npy", np.zeros((2, 2)))

    out = tmp_path / "smoke"
    manifest = make_smoke_subset(root, out, cases_per_density=1)

    assert len(manifest["cases"]) == 2
    assert {case["density_class"] for case in manifest["cases"]} == {"dense", "fatty"}
    assert (out / "openbreastus_index.json").exists()
    assert (out / "openbreastus_smoke_manifest.json").exists()
    assert (out / "schema_inspection_report.md").exists()
    assert "case_limitations" in manifest["case_capability_summary"]


def test_make_smoke_subset_converts_speed_mat_volume(tmp_path):
    root = tmp_path / "openbreastus"
    root.mkdir()
    mat_path = root / "breast_train_speed.mat"
    with h5py.File(mat_path, "w") as handle:
        data = np.zeros((8, 8, 2), dtype=np.float32)
        data[:, :, 0] = 1500.0
        data[2:6, 2:6, 0] = 1450.0
        data[:, :, 1] = 1510.0
        handle.create_dataset("breast_train", data=data)

    out = tmp_path / "smoke"
    manifest = make_smoke_subset(root, out, cases_per_density=1, converted_shape=(4, 4), n_transducers=8)

    assert len(manifest["converted_cases"]) == 1
    case_path = out / "cases" / "breast_train_speed_000000.h5"
    loaded = read_case_hdf5(case_path)
    assert loaded.case_id == "breast_train_speed_000000"
    assert loaded.grid.shape == (4, 4)
    assert loaded.measurement.delta_tof_s.shape == (8, 8)
    assert loaded.measurement.log_amp.shape == (8, 8)
    assert loaded.ground_truth.sound_speed_mps.shape == (4, 4)
    assert loaded.metadata["conversion"] == "speed_map_to_straight_ray_surrogate"
    assert loaded.metadata["feature_provenance"] == "surrogate_delta_tof_from_ground_truth_sound_speed"
    assert "log_amp is a zero surrogate" in loaded.metadata["measurement_limitations"][2]
    assert manifest["converted_cases"][0]["has_measured_attenuation"] is False
    assert manifest["converted_cases"][0]["attenuation_evidence"] == "surrogate_zero_log_amp"
    assert manifest["case_capability_summary"]["conversion_mode_counts"]["speed_map_to_straight_ray_surrogate"] == 1


def test_convert_speed_mat_volume_accepts_matlab_v5_case_first_volume(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    mat_path = tmp_path / "breast_test_speed.mat"
    data = np.zeros((2, 8, 8), dtype=np.float32)
    data[0] = 1500.0
    data[1] = 1500.0
    data[1, 2:6, 2:6] = 1460.0
    scipy_io.savemat(mat_path, {"breast_test": data})

    metadata = speed_mat_metadata(mat_path)
    records = convert_speed_mat_volume(
        mat_path,
        tmp_path / "cases",
        indices=[1],
        dataset_name="breast_test",
        case_id_prefix="openbreast_test",
        output_shape=(4, 4),
        n_transducers=8,
    )

    assert metadata["mat_format"] == "matlab-v5"
    assert metadata["sample_axis"] == 0
    assert records[0]["case_id"] == "openbreast_test_000001"
    loaded = read_case_hdf5(records[0]["path"])
    assert loaded.grid.shape == (4, 4)
    assert loaded.metadata["source_index"] == 1
    assert float(np.min(loaded.ground_truth.sound_speed_mps)) < 1500.0


def test_make_smoke_subset_prefers_and_converts_kwave_channel_mat(tmp_path):
    root = tmp_path / "openbreastus"
    root.mkdir()
    with h5py.File(root / "z_speed_only.mat", "w") as handle:
        handle.create_dataset("breast_train", data=np.ones((8, 8, 1), dtype=np.float32) * 1500.0)
    _write_kwave_channel_mat(root / "kwave_train_0001.mat")

    out = tmp_path / "smoke"
    manifest = make_smoke_subset(root, out, cases_per_density=1, converted_shape=(4, 4), n_transducers=6)

    assert [case["case_id"] for case in manifest["cases"]] == ["kwave_train_0001"]
    assert manifest["case_capability_summary"]["conversion_mode_counts"]["kwave_channel_mat_to_feature_case"] == 1
    converted = manifest["converted_cases"][0]
    assert converted["has_simulated_attenuation"] is True
    assert converted["attenuation_evidence"] == "simulated_ground_truth_line_integral"
    loaded = read_case_hdf5(converted["path"])
    assert loaded.grid.shape == (4, 4)
    assert loaded.measurement.delta_tof_s.shape == (6, 6)
    assert loaded.measurement.log_amp.shape == (6, 6)
    assert loaded.ground_truth.sound_speed_mps.shape == (4, 4)
    assert loaded.ground_truth.attenuation_np_per_m.shape == (4, 4)
    assert float(np.linalg.norm(loaded.measurement.log_amp)) > 0.0
    assert loaded.metadata["conversion"] == "kwave_channel_mat_to_feature_case"


def test_make_smoke_subset_removes_stale_converted_cases(tmp_path):
    root = tmp_path / "openbreastus"
    root.mkdir()
    _write_kwave_channel_mat(root / "kwave_train_0001.mat")
    out = tmp_path / "smoke"
    stale = out / "cases" / "stale_speed_only.h5"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")

    manifest = make_smoke_subset(root, out, cases_per_density=1, converted_shape=(4, 4), n_transducers=6)

    assert not stale.exists()
    assert len(list((out / "cases").glob("*.h5"))) == len(manifest["converted_cases"]) == 1


def _write_kwave_channel_mat(path):
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    positions_xy = np.column_stack((0.11 * np.cos(angles), 0.11 * np.sin(angles)))
    sound_speed = np.full((8, 8), 1500.0, dtype=np.float64)
    sound_speed[3:5, 3:5] = 1450.0
    attenuation = np.full((8, 8), 0.05, dtype=np.float64)
    attenuation[2:6, 2:6] = 2.0
    channel = np.ones((8, 8, 16), dtype=np.float32)
    time = np.linspace(0.0, 1.0e-4, 16)[:, None]
    coords = np.linspace(-0.12, 0.12, 8)[:, None]
    with h5py.File(path, "w") as handle:
        handle.create_dataset("C", data=sound_speed)
        handle.create_dataset("atten", data=attenuation)
        handle.create_dataset("full_dataset", data=channel)
        handle.create_dataset("time", data=time)
        handle.create_dataset("transducerPositionsXY", data=positions_xy)
        handle.create_dataset("xi_orig", data=coords)
        handle.create_dataset("yi_orig", data=coords)
        handle.create_dataset("sim_metadata/f_tx", data=np.asarray([[1.0e6]]))
        handle.create_dataset("sim_metadata/sim_label", data=np.asarray([ord(c) for c in "kwave_train_0001"], dtype=np.uint16)[:, None])
        handle.create_dataset(
            "sim_metadata/source_npy_path",
            data=np.asarray([ord(c) for c in "/tmp/source.npy"], dtype=np.uint16)[:, None],
        )
