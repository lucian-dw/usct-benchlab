#!/usr/bin/env python3
"""Render required visual artifacts for a k-Wave/FWI smoke run."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from usctbench.algorithms.fwi.kwave_adapter import read_kwave_fwi_result
from usctbench.io.hdf5 import read_case_hdf5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path, help="Standard USCTCase HDF5 used for the run.")
    parser.add_argument("--result", required=True, type=Path, help="External k-Wave FWI result MAT/HDF5 file.")
    parser.add_argument("--out", required=True, type=Path, help="Directory for rendered smoke artifacts.")
    parser.add_argument("--log", type=Path, default=None, help="External pipeline log to copy to run.log.")
    parser.add_argument("--iteration", default="final", help="1-based VEL_ESTIM_ITER checkpoint to render, or final.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    external = read_kwave_fwi_result(args.result)
    case = read_case_hdf5(args.case)
    selected_iteration = _parse_iteration(args.iteration, int(external.get("iterations", 0)))
    reconstruction = _select_reconstruction(args.result, selected_iteration)
    if reconstruction is None:
        reconstruction = np.asarray(external["sound_speed_mps"], dtype=float)
    ground_truth = _read_result_dataset(args.result, "C_INTERP")
    if ground_truth is None and case.ground_truth.sound_speed_mps is not None:
        ground_truth = _resize_to_shape(np.asarray(case.ground_truth.sound_speed_mps, dtype=float), reconstruction.shape)
    if ground_truth is None:
        ground_truth = np.full_like(reconstruction, np.nan)
    else:
        ground_truth = _resize_to_shape(np.asarray(ground_truth, dtype=float), reconstruction.shape)

    losses = np.asarray(external.get("losses", []), dtype=float).reshape(-1)
    gradients = _read_result_dataset(args.result, "GRAD_IMG_ITER")

    _write_image(args.out / "reconstruction.png", reconstruction, title="Reconstruction", cmap="gray", unit="m/s")
    _write_image(args.out / "ground_truth.png", ground_truth, title="Ground truth", cmap="gray", unit="m/s")
    _write_image(args.out / "error.png", reconstruction - ground_truth, title="Reconstruction error", cmap="coolwarm", unit="m/s", symmetric=True)
    _write_loss(args.out / "loss_curve.png", losses)

    gradient_metadata: dict[str, Any] = {}
    if gradients is not None and gradients.size:
        gradients = np.asarray(gradients, dtype=float)
        if gradients.ndim == 3:
            idx_001 = 0
            idx_020 = min(19, gradients.shape[0] - 1)
            _write_image(args.out / "gradient_step001.png", gradients[idx_001], title="Gradient step 001", cmap="coolwarm", symmetric=True)
            _write_image(
                args.out / "gradient_step020.png",
                gradients[idx_020],
                title=f"Gradient step {idx_020 + 1:03d}",
                cmap="coolwarm",
                symmetric=True,
            )
            gradient_metadata = {
                "gradient_dataset": "GRAD_IMG_ITER",
                "gradient_shape": list(gradients.shape),
                "gradient_step001_source_index": int(idx_001),
                "gradient_step020_source_index": int(idx_020),
            }

    if args.log is not None and args.log.exists():
        shutil.copyfile(args.log, args.out / "run.log")
    else:
        (args.out / "run.log").write_text("No external log was provided.\n", encoding="utf-8")

    metadata = {
        "case_path": str(args.case),
        "result_path": str(args.result),
        "external_dataset_path": external.get("dataset_path") or "",
        "iterations": int(external.get("iterations", 0)),
        "selected_iteration": selected_iteration or int(external.get("iterations", 0)),
        "initial_loss": external.get("initial_loss"),
        "final_loss": external.get("final_loss"),
        "loss_decreased": external.get("loss_decreased"),
        "psnr_value": external.get("psnr_value"),
        "ssim_value": external.get("ssim_value"),
        "reconstruction_shape": list(reconstruction.shape),
        "ground_truth_shape": list(ground_truth.shape),
        **gradient_metadata,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(args.out)
    return 0


def _read_result_dataset(path: Path, name: str) -> np.ndarray | None:
    try:
        import h5py
    except ModuleNotFoundError:
        return None
    with h5py.File(path, "r") as handle:
        if name not in handle:
            return None
        return np.asarray(handle[name][()])


def _select_reconstruction(path: Path, iteration: int | None) -> np.ndarray | None:
    if iteration is None:
        return None
    stack = _read_result_dataset(path, "VEL_ESTIM_ITER")
    if stack is None:
        return None
    array = np.asarray(stack, dtype=float)
    if array.ndim < 3 or array.shape[0] == 0:
        return None
    index = max(0, min(int(iteration) - 1, array.shape[0] - 1))
    return array[index]


def _parse_iteration(value: str, total: int) -> int | None:
    text = str(value).strip().lower()
    if text in {"", "final", "last"}:
        return None
    if text in {"first", "initial"}:
        return 1
    iteration = int(text)
    if iteration <= 0:
        return None
    if total > 0:
        return min(iteration, total)
    return iteration


def _write_image(
    path: Path,
    image: np.ndarray,
    *,
    title: str,
    cmap: str,
    unit: str | None = None,
    symmetric: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    array = np.asarray(image, dtype=float)
    if symmetric:
        vmax = float(np.nanmax(np.abs(array))) if np.isfinite(array).any() else 1.0
        vmin = -vmax
    else:
        vmin = float(np.nanpercentile(array, 1)) if np.isfinite(array).any() else None
        vmax = float(np.nanpercentile(array, 99)) if np.isfinite(array).any() else None
    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if unit:
        colorbar.set_label(unit)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_loss(path: Path, losses: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3), dpi=160)
    if losses.size:
        ax.plot(np.arange(1, losses.size + 1), losses, marker="o", linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "No LOSS_ITER found", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("FWI loss")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _resize_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    if array.shape == shape:
        return array
    y_idx = np.linspace(0, array.shape[0] - 1, shape[0])
    x_idx = np.linspace(0, array.shape[1] - 1, shape[1])
    y0 = np.floor(y_idx).astype(int)
    x0 = np.floor(x_idx).astype(int)
    y1 = np.clip(y0 + 1, 0, array.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, array.shape[1] - 1)
    wy = (y_idx - y0)[:, None]
    wx = (x_idx - x0)[None, :]
    top = (1.0 - wx) * array[np.ix_(y0, x0)] + wx * array[np.ix_(y0, x1)]
    bottom = (1.0 - wx) * array[np.ix_(y1, x0)] + wx * array[np.ix_(y1, x1)]
    return (1.0 - wy) * top + wy * bottom


if __name__ == "__main__":
    raise SystemExit(main())
