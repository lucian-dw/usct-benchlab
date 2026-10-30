from __future__ import annotations

import zipfile
import json

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from usctbench.core.io import read_case_hdf5
from usctbench.data.nbpslice2d import convert_nbp_slice2d_mat
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
    assert sos_case.metadata["roi_source"] == "full_grid"
    assert sos_case.metadata["roi_use"] == "reconstruction_domain"


def test_prepare_synthetic_demo_writes_hdf5_cases(tmp_path):
    records = prepare_synthetic_demo(
        tmp_path / "synthetic", shape=(8, 8), n_transducers=8
    )

    assert len(records) == 1
    case = read_case_hdf5(
        tmp_path / "synthetic" / "cases" / "synthetic_circular_sos.h5"
    )
    assert case.grid.shape == (8, 8)
    manifest = json.loads((tmp_path / "synthetic" / "manifest.json").read_text())
    assert manifest["generation"]["inclusion_radius_m"] == 0.008


@pytest.mark.parametrize("mat_format", ["hdf5", "v5"])
def test_volume_axes_units_are_explicit(tmp_path, mat_format):
    from usctbench.data.openbreastus import convert_speed_mat_volume

    source = tmp_path / "volume.mat"
    image = np.arange(24).reshape(4, 6) + 1480.0
    arrays = {
        "sos": (image.T / 1000.0)[None, ...],
        "unrelated_larger_array": np.ones((8, 10, 12)),
    }
    if mat_format == "hdf5":
        with h5py.File(source, "w") as handle:
            for name, value in arrays.items():
                handle[name] = value
    else:
        savemat(source, arrays)
    with pytest.raises(TypeError, match="dataset_name"):
        convert_speed_mat_volume(source, tmp_path / "missing_name")
    with pytest.raises(ValueError, match="dataset_name"):
        convert_speed_mat_volume(source, tmp_path / "empty_name", dataset_name="")
    with pytest.raises(KeyError):
        convert_speed_mat_volume(
            source, tmp_path / "unknown_name", dataset_name="unknown"
        )
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
    assert case.metadata["roi_source"] == "full_grid"


def test_rectangular_maps_and_labels_share_physical_cell_centers():
    from usctbench.data.conversion import resample_bilinear, resample_labels_nearest

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
    np.testing.assert_allclose(resample_bilinear(speed, target), expected_speed)
    np.testing.assert_array_equal(
        resample_labels_nearest(labels, target), labels[iy, ix]
    )


def test_reader_does_not_intercept_caller_oserror(tmp_path):
    from usctbench.data.openbreastus import _open_speed_volume

    path = tmp_path / "speed.mat"
    with h5py.File(path, "w") as handle:
        handle["speed"] = np.ones((3, 4, 5))
    error = OSError("consumer write failed")
    with pytest.raises(OSError) as caught:
        with _open_speed_volume(path, dataset_name="speed"):
            raise error
    assert caught.value is error


def test_openbreastus_conversion_preserves_unrelated_case_files(tmp_path):
    from usctbench.data import openbreastus

    root, out = tmp_path / "input", tmp_path / "output"
    root.mkdir()
    (out / "cases").mkdir(parents=True)
    saved = out / "cases" / "unrelated.h5"
    saved.write_bytes(b"existing user result")
    with h5py.File(root / "speed.mat", "w") as handle:
        handle["speed"] = np.full((8, 9, 2), 1490.0)
    manifest = openbreastus.prepare_openbreastus(
        root,
        out,
        files=["speed.mat"],
        dataset_name="speed",
        sample_indices=[1],
        spacing_m=(0.001, 0.002),
        converted_shape=(8, 9),
        n_transducers=8,
    )
    assert len(manifest["cases"]) == 1
    case = read_case_hdf5(manifest["cases"][0]["path"])
    np.testing.assert_allclose(case.ground_truth.sound_speed_mps, 1490)
    assert case.grid.shape == (8, 9)
    assert saved.read_bytes() == b"existing user result"


def test_raw_extra_material_maps_are_not_required_or_used(tmp_path):
    source = tmp_path / "A_case.mat"
    speed = np.arange(64, dtype=float).reshape(8, 8) + 1450
    with h5py.File(source, "w") as handle:
        handle["sos"] = speed / 1000
        handle["den"] = np.ones((8, 8))
        handle["label"] = np.ones((8, 8), dtype=np.uint8)
        handle["type"] = [ord("A")]

    def convert(folder):
        record = convert_nbp_slice2d_mat(
            source, tmp_path / folder, output_shape=(8, 8), n_transducers=8
        )[0]
        return read_case_hdf5(record["path"])

    without = convert("without")
    with h5py.File(source, "a") as handle:
        # Malformed, unit-unknown raw maps are deliberately irrelevant.
        handle["att"] = np.full((2, 3), np.nan)
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
    assert with_extra.measurement.frequencies_hz is None
    assert with_extra.metadata["roi_source"] == "ground_truth_label"
    assert with_extra.metadata["roi_use"] == "reconstruction_prior"


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


def test_nbp_reads_each_required_array_once_and_records_missing_roi(
    tmp_path, monkeypatch
):
    source = tmp_path / "unlabelled.mat"
    with h5py.File(source, "w") as handle:
        handle["sos"] = np.full((4, 5), 1.5)
        handle["den"] = np.ones((4, 5))
        handle["label"] = np.zeros((4, 5), dtype=np.uint8)
        handle["type"] = [ord("B")]
    reads = {"/sos": 0, "/den": 0, "/label": 0, "/type": 0}
    original = h5py.Dataset.__getitem__

    def read(dataset, *args, **kwargs):
        if dataset.name in reads:
            reads[dataset.name] += 1
        return original(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    record = convert_nbp_slice2d_mat(
        source, tmp_path / "cases", output_shape=(4, 5), n_transducers=4
    )[0]
    assert all(count == 1 for count in reads.values())
    case = read_case_hdf5(record["path"])
    assert case.grid.roi_mask is None
    assert case.metadata["roi_source"] == "none"
    assert case.metadata["roi_use"] == "unrestricted"
    assert case.metadata["density_label"] == "B"


@pytest.mark.parametrize(
    "label,type_code,match",
    [
        (np.zeros((2, 3)), [65], "matching 2-D"),
        (np.zeros((4, 5)), [65, 66], "one finite integer"),
    ],
)
def test_nbp_rejects_inconsistent_required_metadata(tmp_path, label, type_code, match):
    source = tmp_path / "invalid.mat"
    with h5py.File(source, "w") as handle:
        handle["sos"] = np.full((4, 5), 1.5)
        handle["den"] = np.ones((4, 5))
        handle["label"] = label
        handle["type"] = type_code
    with pytest.raises(ValueError, match=match):
        convert_nbp_slice2d_mat(source, tmp_path / "cases", n_transducers=4)
