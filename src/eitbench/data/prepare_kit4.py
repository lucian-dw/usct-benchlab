"""Prepare the ten fixed KIT4 test cases from the authors' Zenodo archives."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat

from ..operators.fem import disk_mesh

ROOT = Path.cwd()
BASE = "https://zenodo.org/records/1203914/files/"
CASES = [
    ("1_1", "one plastic circle"),
    ("1_2", "one hollow metal circle"),
    ("2_1", "two plastic objects: triangle and circle"),
    ("2_3", "two hollow metal circles"),
    ("3_1", "three plastic objects: rectangle, triangle, circle"),
    ("3_4", "three hollow metal circles"),
    ("4_1", "metal circle and plastic triangle"),
    ("4_3", "metal circle and plastic circle"),
    ("5_1", "one metal and two plastic objects"),
    ("5_2", "two metal and one plastic objects"),
]
MD5 = {
    "data_mat_files.zip": "ad9f03325cafc9aae8c0149927de8e06",
    "target_photos.zip": "dbebe6df5c3c8e8f6bcaa0cb0e703915",
}


def prepare(archive, photos, output):
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source"
    source.mkdir(exist_ok=True)
    photo_dir = output / "photos"
    photo_dir.mkdir(exist_ok=True)
    checksums = {}
    for name, path in [("data_mat_files.zip", archive), ("target_photos.zip", photos)]:
        payload = path.read_bytes()
        if hashlib.md5(payload).hexdigest() != MD5[name]:
            raise ValueError("official archive MD5 mismatch: " + name)
        checksums[name] = {
            "url": BASE + name,
            "md5": MD5[name],
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    mesh = disk_mesh()

    def read_mat(case):
        name = "datamat_" + case + ".mat"
        raw = z.read("data_mat_files/" + name)
        (source / name).write_bytes(raw)
        return loadmat(io.BytesIO(raw)), {
            "file": "source/" + name,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    records = []
    with ZipFile(archive) as z, ZipFile(photos) as zp:
        reference, ref_record = read_mat("1_0")
        for case, description in CASES:
            target, original = read_mat(case)
            if not np.array_equal(
                target["CurrentPattern"], reference["CurrentPattern"]
            ) or not np.array_equal(target["MeasPattern"], reference["MeasPattern"]):
                raise ValueError("reference and target protocols differ")
            # The stored nonzero CurrentPattern is sqrt(2). Adopt mA -> A;
            # preserve the stored amplitude, never replace it with 2 mA.
            currents = np.asarray(target["CurrentPattern"], float) * 1e-3
            measurement = np.asarray(target["MeasPattern"], float).T
            # Discard pairs touching either drive electrode, for each injection.
            mask = (np.abs(measurement) @ (np.abs(currents) > 0).astype(int)) == 0
            values = dict(
                nodes=mesh.nodes,
                triangles=mesh.triangles,
                boundary_edges=mesh.boundary_edges,
                electrode_ids=mesh.electrode_ids,
                currents_A=currents,
                measurement_matrix=measurement,
                voltages_V=np.asarray(target["Uel"], float),
                reference_voltages_V=np.asarray(reference["Uel"], float),
                measurement_mask=mask,
                thickness_m=np.array(0.07),
                contact_impedance_ohm_m2=np.full(16, 0.001),
                background_conductivity_S_m=np.array(0.03),
                case_id=np.array(case),
                data_kind=np.array("experimental"),
            )
            np.savez_compressed(output / ("phantom_" + case + ".npz"), **values)
            filename = "fantom_" + case + ".jpg"
            members = [
                p
                for p in zp.namelist()
                if p.split("/")[-1] == filename and "__MACOSX" not in p
            ]
            if len(members) != 1:
                raise ValueError("missing or ambiguous photo " + filename)
            raw = zp.read(members[0])
            (photo_dir / filename).write_bytes(raw)
            records.append(
                dict(
                    case_id=case,
                    description=description,
                    input="phantom_" + case + ".npz",
                    source=original,
                    photo="photos/" + filename,
                    photo_sha256=hashlib.sha256(raw).hexdigest(),
                    selected_measurements=int(mask.sum()),
                )
            )
    # example.npz is byte-identical to the first selected phantom.
    (output / "example.npz").write_bytes((output / "phantom_1_1.npz").read_bytes())
    manifest = dict(
        dataset="2D electrical impedance tomography dataset",
        version="1.0.0",
        doi="10.5281/zenodo.1203914",
        license="CC-BY-4.0",
        citation="Hauptmann, Kolehmainen, Mach, Savolainen, Seppänen, Siltanen (2017).",
        selection="Fixed diverse cases selected before inversion; no target-specific tuning.",
        reference=ref_record,
        archives=checksums,
        cases=records,
        preprocessing={
            "CurrentPattern": "stored values times 1e-3, mA -> A convention; sqrt(2) preserved",
            "MeasPattern": "transpose: original columns become measurement rows",
            "Uel": "stored signed values interpreted as V; no abs, sign flip, normalization or resimulation",
            "mask": "per injection discard voltage pairs touching either current electrode",
            "geometry": "new deterministic 2D mesh; dimensions from dataset report",
            "scale_caveat": "The report gives 2 mA amplitude whereas MAT stores sqrt(2). RMS interpretation is an inference. Calibrated absolute S/m depends on this current-unit convention.",
        },
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "cases": len(records),
                "nodes": len(mesh.nodes),
                "triangles": len(mesh.triangles),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--photos", type=Path)
    parser.add_argument("--download-dir", type=Path, default=ROOT / "downloads")
    parser.add_argument("--output", type=Path, default=ROOT / "data/eit/kit4")
    args = parser.parse_args()
    paths = [args.archive, args.photos]
    for i, name in enumerate(MD5):
        if paths[i] is None:
            args.download_dir.mkdir(parents=True, exist_ok=True)
            paths[i] = args.download_dir / name
            if not paths[i].exists():
                urllib.request.urlretrieve(BASE + name, paths[i])
    prepare(paths[0], paths[1], args.output)


if __name__ == "__main__":
    main()
