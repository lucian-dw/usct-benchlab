import json

import h5py
import numpy as np

from scripts.render_examples import read_images


def test_experiment_identity_is_separate_from_algorithm(tmp_path):
    cases = tmp_path / "cases"
    cases.mkdir()
    with h5py.File(cases / "case.h5", "w") as h:
        h.attrs["case_id"] = "case"
        grid = h.create_group("grid")
        grid.attrs.update(shape=(2, 3), spacing_m=(1, 1), origin_m=(0, 0))
    for experiment in ("default", "tuned"):
        folder = tmp_path / experiment / "case"
        folder.mkdir(parents=True)
        with h5py.File(folder / "result.h5", "w") as h:
            h["sound_speed_mps"] = np.full((2, 3), 1500)
            h.attrs.update(algorithm="straight_cgls", variant="cgls")
        if experiment == "default":
            (folder / "run.json").write_text(
                json.dumps({"algorithm": "straight_cgls", "variant": "cgls"})
            )
    _, records = read_images(cases, tmp_path, ["case"], ["default", "tuned"])
    assert [r["experiment_id"] for r in records] == ["default", "tuned"]
    assert all(
        r["algorithm"] == "straight_cgls"
        and r["variant"] == "cgls"
        and not r["missing"]
        for r in records
    )
