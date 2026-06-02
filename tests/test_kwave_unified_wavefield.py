from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from usctbench.features import extract_wavefield_features
from usctbench.io.hdf5 import read_case_hdf5, write_case_hdf5
from usctbench.provenance import MeasurementProvenance, case_measurement_metadata
from usctbench.sim.kwave_forward import _read_external_kwave_dataset, run_kwave_simulation_from_config
from usctbench.sim.qc import run_simulation_qc


def test_kwave_smoke_sim_qc_and_features(tmp_path):
    config = {
        "name": "unit_kwave_smoke",
        "case": {"case_id": "unit_property", "shape": [24, 24], "n_transducers": 8, "inclusion_radius_m": 0.01},
        "simulation": {
            "backend": "native_smoke",
            "reference_sound_speed_mps": 1500.0,
            "source_peak_frequency_hz": 250000.0,
            "dt_s": 8.0e-8,
            "n_time": 768,
            "pml_thickness_pixels": 8,
            "frequencies_hz": [150000.0, 250000.0, 350000.0, 450000.0],
        },
        "outputs": {"wavefield_case": str(tmp_path / "wavefield_cases" / "unit_wave.h5")},
    }
    config_path = tmp_path / "kwave_smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    wave_path = run_kwave_simulation_from_config(config_path)
    wave_case = read_case_hdf5(wave_path)

    assert wave_case.metadata["measurement_provenance"] == MeasurementProvenance.SELF_SIMULATED_KWAVE_WAVEFIELD.value
    assert wave_case.measurement.time_data.shape == (8, 8, 768)
    assert wave_case.measurement.freq_data.shape == (4, 8, 8)
    assert wave_case.measurement.water_reference is not None
    assert wave_case.ground_truth.density_kg_per_m3.shape == wave_case.grid.shape
    assert wave_case.metadata["synthetic_inclusion_radius_m"] == 0.01

    qc = run_simulation_qc(wave_path)
    assert qc["passed"], qc["fail_reasons"]
    assert (wave_path.parent / "simulation_qc.json").exists()
    assert (wave_path.parent / "feature_preview.png").exists()

    feature_path = tmp_path / "feature_cases" / "unit_features.h5"
    feature_case, feature_qc = extract_wavefield_features(wave_path, out=feature_path)
    loaded = read_case_hdf5(feature_path)

    assert loaded.metadata["measurement_provenance"] == MeasurementProvenance.SELF_SIMULATED_KWAVE_WAVEFIELD.value
    assert loaded.measurement.delta_tof_s.shape == (8, 8)
    assert loaded.measurement.time_data is not None
    assert loaded.measurement.freq_data is not None
    assert feature_qc["tof_valid_fraction"] > 0.5
    assert feature_case.measurement.feature_quality.shape == (8, 8)


def test_external_kwave_reader_transposes_xy_images_to_yx(tmp_path):
    path = tmp_path / "external_dataset.mat"
    c_xy = np.array([[1500.0, 1510.0], [1520.0, 1530.0], [1540.0, 1550.0]], dtype=np.float32)
    attenuation_xy = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], dtype=np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("time", data=np.linspace(0.0, 4.0e-7, 5))
        handle.create_dataset("transducerPositionsXY", data=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=float))
        handle.create_dataset("full_dataset", data=np.ones((4, 4, 5), dtype=np.float32))
        handle.create_dataset("C", data=c_xy)
        handle.create_dataset("atten", data=attenuation_xy)
        handle.create_dataset("xi_orig", data=np.array([-0.02, 0.0, 0.02], dtype=float))
        handle.create_dataset("yi_orig", data=np.array([-0.01, 0.01], dtype=float))

    data = _read_external_kwave_dataset(path)

    assert data["grid"].shape == (2, 3)
    np.testing.assert_allclose(data["sound_speed_mps"], c_xy.T)
    np.testing.assert_allclose(data["attenuation_np_per_m"], attenuation_xy.T)
    assert data["metadata"]["array_axis_convention_raw"] == "[x,y]"
    assert data["metadata"]["array_axis_convention_internal"] == "[row=y,col=x]"
    assert data["metadata"]["array_axis_conversion"] == "transpose_external_xy_to_internal_yx"


def test_simulation_qc_excludes_self_pairs_from_boundary_energy(tmp_path):
    config = {
        "name": "unit_self_pair_qc",
        "case": {"case_id": "unit_self_pair_property", "shape": [24, 24], "n_transducers": 8},
        "simulation": {
            "backend": "native_smoke",
            "reference_sound_speed_mps": 1500.0,
            "source_peak_frequency_hz": 250000.0,
            "dt_s": 8.0e-8,
            "n_time": 768,
            "pml_thickness_pixels": 8,
            "frequencies_hz": [150000.0, 250000.0, 350000.0, 450000.0],
        },
        "outputs": {"wavefield_case": str(tmp_path / "wavefield_cases" / "unit_self_pair_wave.h5")},
    }
    config_path = tmp_path / "kwave_self_pair.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    wave_path = run_kwave_simulation_from_config(config_path)
    wave_case = read_case_hdf5(wave_path)

    time_data = np.array(wave_case.measurement.time_data, copy=True)
    water = np.array(wave_case.measurement.water_reference, copy=True)
    for idx in range(time_data.shape[0]):
        time_data[idx, idx, :75] = 1.0e6
        water[idx, idx, :75] = 1.0e6
    stamped = wave_case.model_copy(
        update={
            "measurement": wave_case.measurement.model_copy(
                update={
                    "time_data": time_data,
                    "water_reference": water,
                }
            )
        }
    )
    write_case_hdf5(stamped, wave_path)

    qc = run_simulation_qc(wave_path)
    assert qc["passed"], qc["fail_reasons"]
    assert qc["metrics"]["excluded_receiver_fraction"] == 0.125
    assert qc["metrics"]["boundary_energy_fraction"] < 0.4


def test_kwave_cache_reuses_matching_output(tmp_path):
    config = {
        "name": "unit_cache",
        "case": {"case_id": "unit_cache_property", "shape": [16, 16], "n_transducers": 6},
        "simulation": {"backend": "native_smoke", "dt_s": 8.0e-8, "n_time": 512, "frequencies_hz": [150000.0, 250000.0, 350000.0]},
        "outputs": {"wavefield_case": str(tmp_path / "wave.h5")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    first = run_kwave_simulation_from_config(config_path)
    case = read_case_hdf5(first)
    metadata = {**case.metadata, "unit_cache_marker": "kept"}
    write_case_hdf5(case.model_copy(update={"metadata": metadata}), first)

    second = run_kwave_simulation_from_config(config_path)
    assert first == second
    assert read_case_hdf5(second).metadata["unit_cache_marker"] == "kept"


def test_simulation_failed_qc_metadata_is_reportable(tmp_path):
    config = {
        "name": "unit_failed_qc",
        "case": {"case_id": "unit_failed_property", "shape": [16, 16], "n_transducers": 6},
        "simulation": {"backend": "native_smoke", "dt_s": 8.0e-8, "n_time": 512, "frequencies_hz": [150000.0, 250000.0, 350000.0]},
        "outputs": {"wavefield_case": str(tmp_path / "wave.h5")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    wave_path = run_kwave_simulation_from_config(config_path)
    case = read_case_hdf5(wave_path)
    case = case.model_copy(update={"metadata": {**case.metadata, "simulation_failed_qc": True, "simulation_qc_passed": False}})

    metadata = case_measurement_metadata(case.metadata)
    assert metadata["simulation_failed_qc"] is True
    assert metadata["measurement_provenance"] == MeasurementProvenance.SELF_SIMULATED_KWAVE_WAVEFIELD.value
