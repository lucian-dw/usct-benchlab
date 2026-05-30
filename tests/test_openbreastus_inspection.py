from __future__ import annotations

import json

import numpy as np

from usctbench.data.openbreastus import inspect_openbreastus
from usctbench.data.smoke_subset import make_smoke_subset


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
    assert json.loads(out.read_text(encoding="utf-8"))["cases"][0]["case_id"] == "dense_case001"


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

