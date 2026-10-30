"""Explicit OpenBreastUS file selection and separate format inspection."""

import json
from pathlib import Path

import h5py
from scipy.io import whosmat

from .conversion import convert_speed_mat_volume


def inspect_openbreastus(root):
    """Report stored array names/shapes without inferring physics or classes."""
    root = Path(root).expanduser().resolve(strict=True)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".mat", ".h5", ".hdf5"}:
            continue
        record = {"path": path.relative_to(root).as_posix(), "arrays": []}
        try:
            if h5py.is_hdf5(path):
                with h5py.File(path, "r") as handle:

                    def visit(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            record["arrays"].append(
                                {
                                    "name": name,
                                    "shape": list(obj.shape),
                                    "dtype": str(obj.dtype),
                                }
                            )

                    handle.visititems(visit)
            else:
                record["arrays"] = [
                    {"name": name, "shape": list(shape), "dtype": dtype}
                    for name, shape, dtype in whosmat(path)
                ]
        except (OSError, ValueError, NotImplementedError) as exc:
            record["read_error"] = f"{type(exc).__name__}: {exc}"
        files.append(record)
    return {"root": str(root), "files": files}


def prepare_openbreastus(
    root,
    out,
    *,
    files,
    sample_indices,
    spacing_m,
    dataset_name,
    sample_axis=2,
    image_axes="yx",
    sound_speed_units="m/s",
    converted_shape=(64, 64),
    n_transducers=32,
    reference_sound_speed_mps=1500.0,
):
    """Convert declared speed volumes/slices to model-generated ToF cases.

    Pressure import is a separate operation. No density labels, sample groups,
    source links or physical units are inferred from file names.
    """
    root = Path(root).expanduser().resolve(strict=True)
    out = Path(out).expanduser().resolve()
    selected = [(root / name).resolve(strict=True) for name in files]
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate OpenBreastUS input files")
    # Distinct relative names retain their path components in generated IDs.
    prefixes = [
        p.relative_to(root).with_suffix("").as_posix().replace("/", "__")
        for p in selected
    ]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError("Input files would produce duplicate case IDs")
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for source, prefix in zip(selected, prefixes):
        cases.extend(
            convert_speed_mat_volume(
                source,
                out / "cases",
                dataset_name=dataset_name,
                indices=sample_indices,
                case_id_prefix=prefix,
                output_shape=converted_shape,
                spacing_m=spacing_m,
                n_transducers=n_transducers,
                reference_sound_speed_mps=reference_sound_speed_mps,
                sample_axis=sample_axis,
                image_axes=image_axes,
                sound_speed_units=sound_speed_units,
            )
        )
    manifest = {
        "source_root": str(root),
        "cases": cases,
        "settings": {
            "files": list(files),
            "dataset_name": dataset_name,
            "sample_indices": list(sample_indices),
            "shape": list(converted_shape),
            "spacing_m": list(spacing_m),
            "sample_axis": sample_axis,
            "image_axes": image_axes,
            "sound_speed_units": sound_speed_units,
            "n_transducers": n_transducers,
            "reference_sound_speed_mps": reference_sound_speed_mps,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
