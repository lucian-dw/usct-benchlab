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
    parser.add_argument("--iteration", default="final", help="1-based VEL_ESTIM_ITER checkpoint to render, best, or final.")
    parser.add_argument("--render-best-and-final", action="store_true", help="Also render reconstruction_best.png and reconstruction_final.png.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    external = read_kwave_fwi_result(args.result)
    case = read_case_hdf5(args.case)
    total_iterations = int(external.get("iterations", 0))
    final_reconstruction = np.asarray(external["sound_speed_mps"], dtype=float)
    ground_truth = _read_result_dataset(args.result, "C_INTERP")
    if ground_truth is None and case.ground_truth.sound_speed_mps is not None:
        ground_truth = _resize_to_shape(np.asarray(case.ground_truth.sound_speed_mps, dtype=float), final_reconstruction.shape)
    if ground_truth is None:
        ground_truth = np.full_like(final_reconstruction, np.nan)
    else:
        ground_truth = _resize_to_shape(np.asarray(ground_truth, dtype=float), final_reconstruction.shape)

    best_iteration, best_metrics = _best_iteration(args.result, ground_truth)
    selected_iteration = _parse_iteration(args.iteration, total_iterations, best_iteration)
    reconstruction = _select_reconstruction(args.result, selected_iteration)
    if reconstruction is None:
        reconstruction = final_reconstruction
    best_reconstruction = _select_reconstruction(args.result, best_iteration) if best_iteration is not None else None

    losses = np.asarray(external.get("losses", []), dtype=float).reshape(-1)
    gradients = _read_result_dataset(args.result, "GRAD_IMG_ITER")

    _write_image(args.out / "reconstruction.png", reconstruction, title="Reconstruction", cmap="gray", unit="m/s")
    _write_image(args.out / "ground_truth.png", ground_truth, title="Ground truth", cmap="gray", unit="m/s")
    _write_image(args.out / "error.png", reconstruction - ground_truth, title="Reconstruction error", cmap="coolwarm", unit="m/s", symmetric=True)
    if args.render_best_and_final:
        _write_image(args.out / "reconstruction_final.png", final_reconstruction, title="Final reconstruction", cmap="gray", unit="m/s")
        _write_image(
            args.out / "error_final.png",
            final_reconstruction - ground_truth,
            title="Final reconstruction error",
            cmap="coolwarm",
            unit="m/s",
            symmetric=True,
        )
        if best_iteration is not None:
            if best_reconstruction is not None:
                _write_image(
                    args.out / "reconstruction_best.png",
                    best_reconstruction,
                    title=f"Best reconstruction iter {best_iteration:03d}",
                    cmap="gray",
                    unit="m/s",
                )
                _write_image(
                    args.out / "error_best.png",
                    best_reconstruction - ground_truth,
                    title=f"Best reconstruction error iter {best_iteration:03d}",
                    cmap="coolwarm",
                    unit="m/s",
                    symmetric=True,
                )
    _write_loss(args.out / "loss_curve.png", losses)
    _write_contact_sheet(
        args.out / "contact_sheet.png",
        _contact_sheet_panels(
            ground_truth=ground_truth,
            reconstruction=reconstruction,
            final_reconstruction=final_reconstruction,
            best_reconstruction=best_reconstruction,
            selected_iteration=selected_iteration,
            best_iteration=best_iteration,
            total_iterations=total_iterations,
        ),
    )

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
        "requested_iteration": str(args.iteration),
        "best_iteration": best_iteration,
        **best_metrics,
        "initial_loss": external.get("initial_loss"),
        "final_loss": external.get("final_loss"),
        "loss_decreased": external.get("loss_decreased"),
        "psnr_value": external.get("psnr_value"),
        "ssim_value": external.get("ssim_value"),
        "reconstruction_shape": list(reconstruction.shape),
        "ground_truth_shape": list(ground_truth.shape),
        "contact_sheet": "contact_sheet.png",
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
    array = _iteration_stack(stack)
    if array is None or array.shape[0] == 0:
        return None
    index = max(0, min(int(iteration) - 1, array.shape[0] - 1))
    return array[index]


def _best_iteration(path: Path, ground_truth: np.ndarray) -> tuple[int | None, dict[str, float | str]]:
    stack = _read_result_dataset(path, "VEL_ESTIM_ITER")
    array = _iteration_stack(stack) if stack is not None else None
    if array is None or array.shape[0] == 0 or not np.isfinite(ground_truth).any():
        return None, {"best_iteration_reason": "unavailable"}
    truth = _resize_to_shape(ground_truth, array.shape[1:])
    finite = np.isfinite(truth)
    rmses = []
    ssims = []
    for image in array:
        error = np.asarray(image, dtype=float)[finite] - truth[finite]
        rmse = float(np.sqrt(np.mean(error**2)))
        rmses.append(rmse)
        ssims.append(_global_ssim(np.asarray(image, dtype=float)[finite], truth[finite]))
    best_index = int(np.argmin(np.asarray(rmses, dtype=float)))
    return best_index + 1, {
        "best_iteration_metric": "rmse",
        "best_iteration_rmse": rmses[best_index],
        "best_iteration_ssim": ssims[best_index],
        "final_iteration_rmse": rmses[-1],
        "final_iteration_ssim": ssims[-1],
    }


def _parse_iteration(value: str, total: int, best_iteration: int | None = None) -> int | None:
    text = str(value).strip().lower()
    if text in {"", "final", "last"}:
        return None
    if text in {"best", "best_rmse", "auto"}:
        return best_iteration
    if text in {"first", "initial"}:
        return 1
    iteration = int(text)
    if iteration <= 0:
        return None
    if total > 0:
        return min(iteration, total)
    return iteration


def _iteration_stack(stack: np.ndarray) -> np.ndarray | None:
    array = np.asarray(stack, dtype=float)
    if array.ndim < 3:
        return None
    if array.shape[0] <= array.shape[-1]:
        return array
    return np.moveaxis(array, -1, 0)


def _global_ssim(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    data_range = float(np.nanmax(truth) - np.nanmin(truth))
    if not np.isfinite(data_range) or data_range <= 0:
        data_range = max(float(np.nanmax(np.abs(truth))), 1.0)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = float(np.mean(pred))
    mu_y = float(np.mean(truth))
    var_x = float(np.mean((pred - mu_x) ** 2))
    var_y = float(np.mean((truth - mu_y) ** 2))
    cov_xy = float(np.mean((pred - mu_x) * (truth - mu_y)))
    denom = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    if denom == 0:
        return 1.0
    return float(((2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)) / denom)


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


def _contact_sheet_panels(
    *,
    ground_truth: np.ndarray,
    reconstruction: np.ndarray,
    final_reconstruction: np.ndarray,
    best_reconstruction: np.ndarray | None,
    selected_iteration: int | None,
    best_iteration: int | None,
    total_iterations: int,
) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    selected_title = "Final reconstruction" if selected_iteration is None else f"Selected reconstruction iter {selected_iteration:03d}"
    panels: list[tuple[str, np.ndarray, np.ndarray | None]] = [
        ("Ground truth", ground_truth, None),
        (selected_title, reconstruction, reconstruction - ground_truth),
    ]
    if selected_iteration is not None:
        final_title = f"Final reconstruction iter {total_iterations:03d}" if total_iterations > 0 else "Final reconstruction"
        panels.append((final_title, final_reconstruction, final_reconstruction - ground_truth))
    if best_reconstruction is not None and best_iteration is not None and best_iteration != selected_iteration:
        panels.append((f"Best reconstruction iter {best_iteration:03d}", best_reconstruction, best_reconstruction - ground_truth))
    return panels


def _write_contact_sheet(path: Path, panels: list[tuple[str, np.ndarray, np.ndarray | None]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    speed_images = [image for _label, image, _error in panels]
    error_images = [error for _label, _image, error in panels if error is not None]
    speed_vmin, speed_vmax = _shared_limits(speed_images)
    error_limit = _shared_symmetric_limit(error_images)

    ncols = max(1, len(panels))
    fig, axes = plt.subplots(2, ncols, figsize=(max(7.0, 3.0 * ncols), 6.0), dpi=180, squeeze=False)
    speed_artist = None
    error_artist = None
    for col, (label, image, error) in enumerate(panels):
        top = axes[0][col]
        top.set_axis_off()
        speed_artist = top.imshow(image, cmap="gray", vmin=speed_vmin, vmax=speed_vmax, interpolation="nearest")
        top.set_title(label, fontsize=9, pad=3)

        bottom = axes[1][col]
        bottom.set_axis_off()
        if error is None:
            bottom.text(0.5, 0.5, "reference", ha="center", va="center", fontsize=9, transform=bottom.transAxes)
            bottom.set_title("Error", fontsize=9, pad=3)
        else:
            error_artist = bottom.imshow(
                error,
                cmap="coolwarm",
                vmin=-error_limit,
                vmax=error_limit,
                interpolation="nearest",
            )
            bottom.set_title("Error vs GT", fontsize=9, pad=3)

    fig.suptitle("k-Wave FWI comparison", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.9, bottom=0.03, top=0.9, wspace=0.06, hspace=0.16)
    if speed_artist is not None:
        speed_bar = fig.colorbar(speed_artist, ax=axes[0].ravel().tolist(), fraction=0.018, pad=0.012)
        speed_bar.set_label("m/s", fontsize=8)
        speed_bar.ax.tick_params(labelsize=7)
    if error_artist is not None:
        error_bar = fig.colorbar(error_artist, ax=axes[1].ravel().tolist(), fraction=0.018, pad=0.012)
        error_bar.set_label("m/s", fontsize=8)
        error_bar.ax.tick_params(labelsize=7)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _shared_limits(images: list[np.ndarray]) -> tuple[float, float]:
    finite_values = [np.asarray(image, dtype=float)[np.isfinite(image)].reshape(-1) for image in images]
    nonempty = [values for values in finite_values if values.size]
    if not nonempty:
        return 0.0, 1.0
    finite = np.concatenate(nonempty)
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _shared_symmetric_limit(images: list[np.ndarray]) -> float:
    finite_values = [np.abs(np.asarray(image, dtype=float)[np.isfinite(image)]).reshape(-1) for image in images]
    nonempty = [values for values in finite_values if values.size]
    if not nonempty:
        return 1.0
    finite = np.concatenate(nonempty)
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(finite, 99.0))
    if not np.isfinite(limit) or limit <= 0:
        return 1.0
    return limit


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
