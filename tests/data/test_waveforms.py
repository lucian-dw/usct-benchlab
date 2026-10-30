import h5py
import importlib.util
import json
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from usctbench.core.io import read_case_hdf5, write_case_hdf5
from usctbench.data.waveforms import (
    _read_channels,
    read_kwave_pressure_mat,
    pressure_spectrum,
)


@pytest.mark.parametrize("axes", list(permutations(("time", "tx", "rx"))))
def test_chunked_channel_slabs_preserve_axes_and_selection(tmp_path, axes):
    canonical = np.arange(17 * 5 * 6, dtype=np.float32).reshape(17, 5, 6)
    permutation = tuple(("time", "tx", "rx").index(a) for a in axes)
    raw = canonical.transpose(permutation)
    chunks = tuple(1 if a == "time" else raw.shape[i] for i, a in enumerate(axes))
    with h5py.File(tmp_path / "chunked.h5", "w") as handle:
        dataset = handle.create_dataset(
            "p", data=raw, chunks=chunks, compression="gzip"
        )
        actual = _read_channels(dataset, axes, [4, 0], [5, 2], 17, 300)
        expected = canonical[:, [4, 0]][:, :, [5, 2]]
        np.testing.assert_array_equal(actual, expected)


def test_dtft_sign_scale_nonzero_origin_and_nonfft_frequency():
    t = 2.3e-6 + np.arange(32) * 1e-7
    p = np.zeros((32, 2, 3))
    p[7] = 3.0
    f = np.array([123456.0, 310123.0])
    result = pressure_spectrum(p, t, f)
    expected = 3e-7 * np.exp(2j * np.pi * f * t[7])
    np.testing.assert_allclose(
        result, np.broadcast_to(expected[:, None, None], result.shape)
    )


@pytest.mark.parametrize(
    "time,freq",
    [([0, 1, 3], [0.1]), ([0, 0, 0], [1]), ([0, 1, 2], [0.6]), ([0, 1, 2], [0.1, 0.1])],
)
def test_reject_invalid_sampling(time, freq):
    with pytest.raises(ValueError):
        pressure_spectrum(np.ones((3, 2, 2)), time, freq)


def write_fixture(path, speed=1510, truth=True):
    # Asymmetric image and acquisition distinguish all axes, not just shapes.
    p = np.arange(3 * 3 * 16, dtype=np.float32).reshape(3, 3, 16)
    with h5py.File(path, "w") as f:
        f["full_dataset"] = p
        f["water"] = p * 0.5
        f["time"] = np.arange(16) * 1e-7 + 1e-6
        f["transducerPositionsXY"] = [[0.01, 0], [0, 0.02], [-0.03, 0]]
        f["xi_orig"] = [-0.001, 0, 0.001]
        f["yi_orig"] = [-0.002, 0.002]
        if truth:
            f["C"] = np.full((2, 3), speed) + np.arange(6).reshape(2, 3)
    return p


def test_preserve_channels_grid_reference_roundtrip_and_no_oracle(tmp_path):
    source = tmp_path / "channels.mat"
    p = write_fixture(source)
    kw = dict(
        frequencies_hz=[200e3, 333e3],
        tx_indices=[2, 0],
        rx_indices=[1, 2],
        water_dataset="water",
    )
    case = read_kwave_pressure_mat(source, **kw)
    assert sorted(tmp_path.iterdir()) == [source]
    write_case_hdf5(case, tmp_path / "case.h5")
    restored = read_case_hdf5(tmp_path / "case.h5")
    expected = p[[2, 0]][:, [1, 2]].transpose(2, 0, 1)
    np.testing.assert_array_equal(restored.measurement.time_data, expected)
    np.testing.assert_array_equal(
        restored.measurement.freq_data, case.measurement.freq_data
    )
    np.testing.assert_allclose(
        case.measurement.water_reference, 0.5 * case.measurement.freq_data
    )
    np.testing.assert_allclose(case.geometry.tx_pos_m, [[0, -0.03], [0, 0.01]])
    np.testing.assert_allclose(case.grid.origin_m, [-0.004, -0.0015])
    np.testing.assert_allclose(case.grid.spacing_m, [0.004, 0.001])
    np.testing.assert_array_equal(
        case.ground_truth.sound_speed_mps, 1510 + np.arange(6).reshape(2, 3)
    )
    assert case.measurement.delta_tof_s is None
    assert case.metadata["ground_truth_used_for_measurement"] is False
    write_fixture(source, speed=1690)
    changed = read_kwave_pressure_mat(source, **kw)
    np.testing.assert_array_equal(
        case.measurement.freq_data, changed.measurement.freq_data
    )
    assert (
        case.metadata["reference_sound_speed_mps"]
        == changed.metadata["reference_sound_speed_mps"]
    )


def test_no_truth_and_byte_guard(tmp_path):
    source = tmp_path / "channels.mat"
    write_fixture(source, truth=False)
    case = read_kwave_pressure_mat(source)
    assert case.ground_truth.sound_speed_mps is None
    assert case.measurement.time_data.shape == (16, 3, 3)
    with pytest.raises(ValueError, match="max_output_bytes"):
        read_kwave_pressure_mat(source, max_output_bytes=16)
    with pytest.raises(ValueError, match="indices"):
        read_kwave_pressure_mat(source, tx_indices=[1, 1])


def test_pressure_without_label_coordinates_and_explicit_truth_grid(tmp_path):
    from usctbench.core.schema import GridSpec

    source = tmp_path / "channels.mat"
    pressure = write_fixture(source)
    with h5py.File(source, "a") as handle:
        del handle["xi_orig"]
        del handle["yi_orig"]
    grid = GridSpec(shape=(2, 3), spacing_m=(0.004, 0.001), origin_m=(-0.004, -0.0015))
    case = read_kwave_pressure_mat(source, grid=grid)
    np.testing.assert_array_equal(
        case.measurement.time_data, pressure.transpose(2, 0, 1)
    )
    assert case.ground_truth.sound_speed_mps is None
    assert (
        case.metadata["ground_truth_attachment"] == "not_attached_missing_coordinates"
    )
    attached = read_kwave_pressure_mat(source, grid=grid, truth_grid=grid)
    np.testing.assert_array_equal(
        attached.measurement.time_data, case.measurement.time_data
    )
    np.testing.assert_allclose(
        attached.ground_truth.sound_speed_mps, 1510 + np.arange(6).reshape(2, 3)
    )
    assert attached.metadata["ground_truth_attachment"] == "attached_explicit_grid"


def test_time_only_import_avoids_dtft_and_resamples_gt_at_physical_coordinates(
    tmp_path, monkeypatch
):
    from usctbench.data import waveforms
    from usctbench.core.schema import GridSpec

    source = tmp_path / "channels.mat"
    write_fixture(source)

    def forbidden(*args):
        raise AssertionError("time-only import must not compute a spectrum")

    monkeypatch.setattr(waveforms, "pressure_spectrum", forbidden)
    grid = GridSpec(shape=(2, 2), spacing_m=(0.002, 0.0005), origin_m=(-0.002, -0.0005))
    case = read_kwave_pressure_mat(source, grid=grid, water_dataset="water")
    # Source image is 1510 + 3*row + col; sample at actual y/x, not resize indices.
    expected = (
        1510 + 3 * np.array([0.25, 0.75])[:, None] + np.array([0.75, 1.25])[None, :]
    )
    np.testing.assert_allclose(case.ground_truth.sound_speed_mps, expected)
    assert case.measurement.freq_data is None
    assert case.measurement.water_reference is None
    assert case.measurement.frequencies_hz is None
    np.testing.assert_array_equal(
        case.measurement.water_reference_time, 0.5 * case.measurement.time_data
    )
    write_case_hdf5(case, tmp_path / "case.h5")
    restored = read_case_hdf5(tmp_path / "case.h5")
    assert restored.measurement.water_reference is None
    np.testing.assert_array_equal(
        restored.measurement.water_reference_time, case.measurement.water_reference_time
    )
    outside = grid.model_copy(update={"origin_m": (0.02, 0.02)})
    with pytest.raises(ValueError, match="exceeds source physical FOV"):
        read_kwave_pressure_mat(source, grid=outside)


def test_pressure_reader_resizes_grid_preserving_coordinate_edges(tmp_path):
    source = tmp_path / "channels.mat"
    write_fixture(source)
    original = read_kwave_pressure_mat(source)
    resized = read_kwave_pressure_mat(source, output_shape=(7, 5))
    np.testing.assert_allclose(
        np.array(resized.grid.spacing_m) * resized.grid.shape,
        np.array(original.grid.spacing_m) * original.grid.shape,
    )
    np.testing.assert_array_equal(resized.grid.origin_m, original.grid.origin_m)
    assert resized.grid.roi_mask is None
    assert resized.metadata["roi_source"] == "none"


@pytest.mark.parametrize("coordinates", [None, [0], [0, 1, 3], [2, 1, 0], [0, np.nan]])
def test_pressure_reader_requires_physical_coordinates_without_explicit_grid(
    tmp_path, coordinates
):
    source = tmp_path / "channels.mat"
    write_fixture(source, truth=False)
    with h5py.File(source, "a") as handle:
        del handle["xi_orig"]
        if coordinates is not None:
            handle["xi_orig"] = coordinates
    with pytest.raises(ValueError, match="coordinates"):
        read_kwave_pressure_mat(source)


def test_simulation_extract_saves_completed_case_once(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts" / "simulate_kwave.py"
    spec = importlib.util.spec_from_file_location("simulate_kwave", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    positions = np.array([[0.01, 0], [0, 0.01], [-0.01, 0], [0, -0.01]])
    distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    time = np.arange(800) * 1e-7
    phase_time = time[None, None, :] - distances[..., None] / 1500 - 5e-6
    pressure = np.exp(-((phase_time / 2e-6) ** 2)) * np.cos(
        2 * np.pi * 200e3 * phase_time
    )
    with h5py.File(tmp_path / "pressure.mat", "w") as handle:
        handle["full_dataset"] = pressure
        handle["water_dataset"] = pressure
        handle["time"] = time
        handle["transducerPositionsXY"] = positions
        handle["xi_orig"] = [-0.001, 0, 0.001]
        handle["yi_orig"] = [-0.002, 0.002]
        handle["simulation_cfl"] = 0.2
        handle["simulation_ppw"] = 10
        handle["simulation_shape"] = [32, 32]
        handle["pml_pixels"] = 4
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": "fixture_pressure",
                "source_frequency_hz": 200e3,
                "maximum_frequency_hz": 300e3,
            }
        )
    )
    writes = []

    def write(case, path):
        assert case.measurement.delta_tof_s is not None
        assert "simulation_qc" in case.metadata
        writes.append(path)
        return write_case_hdf5(case, path)

    monkeypatch.setattr(module, "write_case_hdf5", write)
    module.extract(
        SimpleNamespace(
            out=str(tmp_path),
            image_size=8,
            frequencies=[100e3, 150e3, 200e3],
            tof_method="xcorr",
            envelope_fraction=0.1,
            aic_envelope_fraction=0.25,
            aic_window_s=None,
        )
    )
    assert writes == [tmp_path / "pressure_case.h5"]
    case = read_case_hdf5(writes[0])
    assert case.case_id == "fixture_pressure"
    assert case.metadata["simulation_qc"]["passed"] is True
    np.testing.assert_allclose(
        case.measurement.delta_tof_s[case.measurement.valid_mask], 0, atol=1e-18
    )
