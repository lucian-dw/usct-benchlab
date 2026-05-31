#!/usr/bin/env python3
"""Render compact class-comparison panels from benchmark result folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from usctbench.io.hdf5 import read_case_hdf5, read_result_hdf5


DEFAULT_SOUND_SPEED_ALGORITHMS = ["straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn", "rwave_adapter"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-dir", required=True, help="Directory containing standard USCTCase HDF5 files.")
    parser.add_argument("--run-dir", required=True, help="Benchmark run directory containing algorithm/case_id/result.h5.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--field", choices=["sound_speed", "attenuation"], default="sound_speed")
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_SOUND_SPEED_ALGORITHMS)
    parser.add_argument("--title", default="")
    parser.add_argument("--max-cases", type=int, default=0, help="Optional cap on displayed cases.")
    parser.add_argument("--cmap", default="gray", help="Matplotlib colormap. Use gray for sound-speed review.")
    parser.add_argument("--unit", default=None)
    args = parser.parse_args()

    cases = [read_case_hdf5(path) for path in sorted(Path(args.cases_dir).glob("*.h5"))]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit(f"no cases found under {args.cases_dir}")

    panels: list[list[tuple[str, np.ndarray | None, dict[str, Any]]]] = []
    summary_rows: list[dict[str, Any]] = []
    for case in cases:
        gt = _ground_truth(case, args.field)
        col: list[tuple[str, np.ndarray | None, dict[str, Any]]] = [(f"GT {case.case_id}", gt, {})]
        for algorithm in args.algorithms:
            result_path = Path(args.run_dir) / algorithm / case.case_id / "result.h5"
            if not result_path.exists():
                col.append((algorithm, None, {"missing": True}))
                continue
            result = read_result_hdf5(result_path)
            image = _result_image(result, args.field)
            metrics = dict(result.metrics)
            col.append((_format_label(algorithm, metrics), image, metrics))
            summary_rows.append({"case_id": case.case_id, "algorithm": algorithm, **_metric_subset(metrics)})
        panels.append(col)

    _write_panel(
        panels,
        Path(args.out),
        title=args.title,
        cmap=args.cmap,
        unit=args.unit or ("m/s" if args.field == "sound_speed" else "Np/m"),
    )
    summary_path = Path(args.out).with_suffix(".summary.csv")
    _write_summary_csv(summary_rows, summary_path)
    manifest_path = Path(args.out).with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "cases_dir": str(Path(args.cases_dir)),
                "run_dir": str(Path(args.run_dir)),
                "out": str(Path(args.out)),
                "field": args.field,
                "algorithms": args.algorithms,
                "cmap": args.cmap,
                "cases": [case.case_id for case in cases],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(args.out)
    return 0


def _ground_truth(case: Any, field: str) -> np.ndarray:
    if field == "sound_speed":
        image = case.ground_truth.sound_speed_mps
    else:
        image = case.ground_truth.attenuation_np_per_m
    if image is None:
        raise ValueError(f"case {case.case_id} has no {field} ground truth")
    return np.asarray(image, dtype=float)


def _result_image(result: Any, field: str) -> np.ndarray | None:
    image = result.sound_speed_mps if field == "sound_speed" else result.attenuation_np_per_m
    if image is None:
        return None
    return np.asarray(image, dtype=float)


def _format_label(algorithm: str, metrics: dict[str, Any]) -> str:
    pieces = [algorithm]
    if _is_finite_number(metrics.get("rmse")):
        pieces.append(f"RMSE={float(metrics['rmse']):.1f}")
    if _is_finite_number(metrics.get("ssim")):
        pieces.append(f"SSIM={float(metrics['ssim']):.3f}")
    return "\n".join(pieces)


def _metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "rmse",
        "mae",
        "nrmse",
        "ssim",
        "psnr",
        "data_relative_residual",
        "data_residual_reduction",
        "water_relative_rmse_improvement",
        "ring_artifact_index",
        "coverage_abs_error_corr",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def _write_panel(
    panels: list[list[tuple[str, np.ndarray | None, dict[str, Any]]]],
    out: Path,
    *,
    title: str,
    cmap: str,
    unit: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.parent.mkdir(parents=True, exist_ok=True)
    ncols = len(panels)
    nrows = max(len(col) for col in panels)
    values = [image for col in panels for _, image, _ in col if image is not None]
    vmin, vmax = _shared_limits(values)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(6.0, 2.6 * ncols), max(5.0, 2.35 * nrows)),
        dpi=180,
        squeeze=False,
    )
    image_artist = None
    for col_idx, col in enumerate(panels):
        for row_idx in range(nrows):
            ax = axes[row_idx][col_idx]
            ax.set_axis_off()
            if row_idx >= len(col):
                continue
            label, image, _metrics = col[row_idx]
            if image is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
            else:
                image_artist = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(label, fontsize=7, pad=2)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.subplots_adjust(
        left=0.02,
        right=0.92 if image_artist is not None else 0.98,
        bottom=0.02,
        top=0.95 if title else 0.98,
        wspace=0.08,
        hspace=0.18,
    )
    if image_artist is not None:
        colorbar = fig.colorbar(image_artist, ax=axes.ravel().tolist(), fraction=0.018, pad=0.012)
        colorbar.set_label(unit, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _shared_limits(images: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([np.asarray(image, dtype=float)[np.isfinite(image)].reshape(-1) for image in images])
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({key for row in rows for key in row})
    preferred = ["case_id", "algorithm", "rmse", "ssim", "data_relative_residual"]
    fieldnames = preferred + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
