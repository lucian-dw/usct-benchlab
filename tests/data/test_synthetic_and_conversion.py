from __future__ import annotations

import zipfile

import h5py
import numpy as np
import pytest

from usctbench.core.io import read_case_hdf5
from usctbench.data.conversion import _grid_from_coordinates
from usctbench.data.conversion import convert_nbp_slice2d_mat, convert_kwave_channel_mat
from usctbench.operators.straight_ray import StraightRayProjector
from usctbench.data.nbpslice2d import (
    inspect_nbp_slice2d_zip,
    prepare_nbpslice2d as prepare_nbp_cases,
)
from usctbench.data.synthetic import (
    make_sound_speed_case,
    prepare_synthetic_demo,
)


def test_synthetic_cases_have_algorithm_ready_features():
    sos_case = make_sound_speed_case(shape=(10, 10), n_transducers=8)

    assert sos_case.measurement.delta_tof_s.shape == (8, 8)
    assert sos_case.ground_truth.sound_speed_mps.shape == (10, 10)


def test_prepare_synthetic_demo_writes_hdf5_cases(tmp_path):
    records = prepare_synthetic_demo(
        tmp_path / "synthetic", shape=(8, 8), n_transducers=8
    )

    assert len(records) == 1
    case = read_case_hdf5(
        tmp_path / "synthetic" / "cases" / "synthetic_circular_sos.h5"
    )
    assert case.grid.shape == (8, 8)


def test_volume_axes_units_and_missing_coordinates_are_explicit(tmp_path):
    from usctbench.data.conversion import convert_speed_mat_volume

    source = tmp_path / "volume.mat"
    image = np.arange(24).reshape(4, 6) + 1480.0
    with h5py.File(source, "w") as handle:
        handle["sos"] = (image.T / 1000.0)[None, ...]
    records = convert_speed_mat_volume(
        source,
        tmp_path / "converted",
        dataset_name="sos",
        sample_axis=0,
        image_axes="xy",
        sound_speed_units="km/s",
        output_shape=image.shape,
        spacing_m=(0.001, 0.002),
        n_transducers=8,
    )
    case = read_case_hdf5(records[0]["path"])
    np.testing.assert_allclose(case.ground_truth.sound_speed_mps, image)
    assert case.grid.spacing_m == (0.001, 0.002)
    assert case.metadata["source_image_axes"] == "xy"
    assert case.metadata["category_source"] == "unknown"
    with pytest.raises(ValueError, match="coordinates are required"):
        _grid_from_coordinates((4, 6), xi=None, yi=None)


def test_grid_from_coordinates_uses_point_spacing():
    xi = np.linspace(-0.01, 0.01, 5)
    yi = np.linspace(-0.02, 0.02, 9)

    grid = _grid_from_coordinates((9, 5), xi=xi, yi=yi)

    assert np.isclose(grid.spacing_m[0], 0.005)
    assert np.isclose(grid.spacing_m[1], 0.005)
    np.testing.assert_allclose(grid.origin_m, [-0.0225, -0.0125])
    resized = _grid_from_coordinates((7, 3), xi=xi, yi=yi)
    np.testing.assert_allclose(
        np.array(resized.spacing_m) * resized.shape,
        np.array(grid.spacing_m) * grid.shape,
    )
    np.testing.assert_array_equal(resized.origin_m, grid.origin_m)


def test_rectangular_maps_and_labels_share_physical_cell_centers():
    from usctbench.data.conversion import _downsample_mean, _downsample_label

    shape, target = (11, 17), (7, 9)
    spacing = np.array([0.0002, 0.0007])
    yy, xx = np.indices(shape)
    speed = 1450 + 1000 * ((yy + 0.5) * spacing[0] + (xx + 0.5) * spacing[1])
    labels = (yy >= 4).astype(np.uint8) + 2 * (xx >= 10).astype(np.uint8)
    target_spacing = spacing * shape / target
    ty, tx = np.indices(target)
    expected_speed = 1450 + 1000 * (
        (ty + 0.5) * target_spacing[0] + (tx + 0.5) * target_spacing[1]
    )
    iy = np.rint((ty + 0.5) * target_spacing[0] / spacing[0] - 0.5).astype(int)
    ix = np.rint((tx + 0.5) * target_spacing[1] / spacing[1] - 0.5).astype(int)
    np.testing.assert_allclose(_downsample_mean(speed, target), expected_speed)
    np.testing.assert_array_equal(_downsample_label(labels, target), labels[iy, ix])


def test_reader_does_not_intercept_caller_oserror(tmp_path):
    from usctbench.data.conversion import _open_speed_volume

    path = tmp_path / "speed.mat"
    with h5py.File(path, "w") as handle:
        handle["speed"] = np.ones((3, 4, 5))
    error = OSError("consumer write failed")
    with pytest.raises(OSError) as caught:
        with _open_speed_volume(path):
            raise error
    assert caught.value is error


def test_openbreastus_conversion_preserves_unrelated_case_files(tmp_path, monkeypatch):
    from usctbench.data import openbreastus

    root, out = tmp_path / "input", tmp_path / "output"
    root.mkdir()
    (out / "cases").mkdir(parents=True)
    saved = out / "cases" / "unrelated.h5"
    saved.write_bytes(b"existing user result")
    monkeypatch.setattr(
        openbreastus, "inspect_openbreastus", lambda *args: {"cases": []}
    )
    openbreastus.prepare_openbreastus(root, out, symlink_sources=False)
    assert saved.read_bytes() == b"existing user result"


@pytest.mark.parametrize("dataset", ["nbp", "kwave"])
def test_raw_extra_material_maps_are_not_required_or_used(tmp_path, dataset):
    source = tmp_path / "A_case.mat"
    speed = np.arange(64, dtype=float).reshape(8, 8) + 1450
    with h5py.File(source, "w") as handle:
        if dataset == "nbp":
            handle["sos"] = speed / 1000
            handle["den"] = np.ones((8, 8))
            handle["label"] = np.ones((8, 8), dtype=np.uint8)
            handle["type"] = [ord("A")]
        else:
            handle["C"] = speed.T
            handle["xi_orig"] = (np.arange(8) + 0.5) * 0.001 - 0.004
            handle["yi_orig"] = (np.arange(8) + 0.5) * 0.001 - 0.004
            angle = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            handle["transducerPositionsXY"] = 0.02 * np.column_stack(
                [np.cos(angle), np.sin(angle)]
            )
            handle["time"] = np.arange(5) * 1e-7
            handle["full_dataset"] = np.zeros((8, 8, 5))
    converter = (
        convert_nbp_slice2d_mat if dataset == "nbp" else convert_kwave_channel_mat
    )

    def convert(folder):
        record = converter(
            source, tmp_path / folder, output_shape=(8, 8), n_transducers=8
        )[0]
        return read_case_hdf5(record["path"])

    without = convert("without")
    with h5py.File(source, "a") as handle:
        # Malformed, unit-unknown raw maps are deliberately irrelevant.
        handle["att" if dataset == "nbp" else "atten"] = np.full((2, 3), np.nan)
        handle["y"] = [-999.0]
    with_extra = convert("with_extra")
    np.testing.assert_array_equal(
        with_extra.ground_truth.sound_speed_mps, without.ground_truth.sound_speed_mps
    )
    np.testing.assert_array_equal(
        with_extra.measurement.delta_tof_s, without.measurement.delta_tof_s
    )
    np.testing.assert_allclose(
        with_extra.ground_truth.sound_speed_mps, speed, rtol=1e-15
    )
    op = StraightRayProjector.from_case(with_extra)
    expected = op.forward(
        1 / with_extra.ground_truth.sound_speed_mps
        - 1 / with_extra.metadata["reference_sound_speed_mps"]
    )
    np.testing.assert_array_equal(with_extra.measurement.delta_tof_s.ravel(), expected)
    assert not any("atten" in key for key in with_extra.metadata)
    if dataset == "nbp":
        assert with_extra.measurement.frequencies_hz is None


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
    manifest = prepare_nbp_cases(
        zip_path, tmp_path / "nbp", converted_shape=(4, 4), n_transducers=8
    )

    assert index["summary"]["num_cases"] == 1
    assert manifest["cases"]
    assert (tmp_path / "nbp" / "cases").exists()
