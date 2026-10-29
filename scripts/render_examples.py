#!/usr/bin/env python3
"""Render selected saved sound-speed images without loading measurement arrays."""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


def read_images(cases_dir, run_dir, case_ids, algorithms):
    """Read grid/GT and selected result images only; preserve imported [y,x] axes."""
    cases = {}
    for path in sorted(cases_dir.glob("*.h5")):
        with h5py.File(path) as handle:
            case_id = str(handle.attrs["case_id"])
            if case_id not in case_ids:
                continue
            grid = handle["grid"].attrs
            shape = tuple(grid["shape"])
            dy, dx = grid["spacing_m"]
            y0, x0 = grid["origin_m"]
            extent = (x0, x0 + shape[1] * dx, y0 + shape[0] * dy, y0)
            truth = handle.get("ground_truth/sound_speed_mps")
            cases[case_id] = (None if truth is None else truth[()], shape, extent)
    missing = set(case_ids) - cases.keys()
    if missing:
        raise ValueError(f"missing cases: {sorted(missing)}")
    rows, summary = [], []
    for case_id in case_ids:
        truth, shape, extent = cases[case_id]
        row = [(truth, {}, extent)]
        for algorithm in algorithms:
            folder = run_dir / algorithm / case_id
            path = folder / "result.h5"
            image, metrics = None, {}
            if path.exists():
                with h5py.File(path) as handle:
                    if "sound_speed_mps" in handle:
                        image = handle["sound_speed_mps"][()]
                if (folder / "metrics.json").exists():
                    metrics = json.loads((folder / "metrics.json").read_text())
            if image is not None and image.shape != shape:
                raise ValueError(f"{path}: result shape differs from the case grid")
            row.append((image, metrics, extent))
            summary.append(
                {
                    "case_id": case_id,
                    "algorithm": algorithm,
                    "missing": image is None,
                    **{key: metrics.get(key) for key in ("rmse", "psnr", "ssim")},
                }
            )
        rows.append(row)
    return rows, summary


def render(rows, path, labels, row_labels, title, limits):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = [image for row in rows for image, _, _ in row if image is not None]
    if not images:
        raise ValueError("no sound-speed images to display")
    if limits is None:
        finite = [image[np.isfinite(image)] for image in images]
        finite = [a for a in finite if a.size]
        if not finite:
            raise ValueError("no finite sound speeds to display")
        limits = (min(a.min() for a in finite), max(a.max() for a in finite))
    low, high = limits
    if high <= low:
        high = low + 1
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(
        len(rows),
        len(labels),
        squeeze=False,
        figsize=(2.5 * len(labels), 2.7 * len(rows)),
        layout="constrained",
    )
    artist = None
    for i, row in enumerate(rows):
        for j, (image, metrics, extent) in enumerate(row):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if image is None:
                ax.text(0.5, 0.5, "unavailable", ha="center", va="center")
            else:
                artist = ax.imshow(
                    image,
                    cmap="gray",
                    vmin=low,
                    vmax=high,
                    origin="upper",
                    extent=extent,
                    interpolation="nearest",
                )
            if i == 0:
                ax.set_title(labels[j], fontweight="bold")
            if j == 0:
                ax.set_ylabel(row_labels[i], fontweight="bold")
            text = []
            for name in ("psnr", "ssim"):
                value = metrics.get(name)
                if isinstance(value, (float, int)) and np.isfinite(value):
                    text.append(f"{name.upper()} {value:.3g}")
            ax.set_xlabel("  ".join(text), fontweight="bold", fontsize=9)
    if title:
        fig.suptitle(title, fontweight="bold")
    if artist is not None:
        fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=0.7, label="m/s")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    parser.add_argument("--algorithms", nargs="+", required=True)
    parser.add_argument("--row-labels", nargs="+")
    parser.add_argument("--algorithm-labels", nargs="+")
    parser.add_argument("--title", default="")
    parser.add_argument("--limits-mps", type=float, nargs=2)
    args = parser.parse_args()
    labels = args.algorithm_labels or args.algorithms
    row_labels = args.row_labels or args.case_ids
    if len(labels) != len(args.algorithms) or len(row_labels) != len(args.case_ids):
        parser.error("labels must match the number of selected cases/algorithms")
    rows, summary = read_images(
        args.cases_dir, args.run_dir, args.case_ids, args.algorithms
    )
    render(rows, args.out, ["GT", *labels], row_labels, args.title, args.limits_mps)
    with args.out.with_suffix(".summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(args.out)


if __name__ == "__main__":
    main()
