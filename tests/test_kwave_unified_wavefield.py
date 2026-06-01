from __future__ import annotations

import json
from pathlib import Path

import yaml

from usctbench.features import extract_wavefield_features
from usctbench.io.hdf5 import read_case_hdf5, write_case_hdf5
from usctbench.provenance import MeasurementProvenance, case_measurement_metadata
from usctbench.sim.kwave_forward import run_kwave_simulation_from_config
from usctbench.sim.qc import run_simulation_qc


def test_kwave_smoke_sim_qc_and_features(tmp_path):
    config = {
        "name": "unit_kwave_smoke",
        "case": {"case_id": "unit_property", "shape": [24, 24], "n_transducers": 8},
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
