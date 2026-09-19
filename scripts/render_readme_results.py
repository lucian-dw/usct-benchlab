"""Render the preserved eight-case study; never run reconstruction or select iterates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

from usctbench.core.io import read_case_hdf5, read_result_hdf5
from usctbench.metrics import compute_regional_image_metrics

METHODS = ["straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn", "ray_born"]
DATASETS = {"openbreastus": ["HET", "FIB", "FAT", "EXD"], "nbpslice2d": list("ABCD")}


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def align(image, grid, reference):
    """Interpolate in physical coordinates, permitting endpoint roundoff only."""
    axes = [
        origin + (np.arange(size) + 0.5) * step
        for size, step, origin in zip(grid.shape, grid.spacing_m, grid.origin_m)
    ]
    targets = [reference["y_m"], reference["x_m"]]
    for source, target in zip(axes, targets):
        if target.min() < source[0] - 1e-10 or target.max() > source[-1] + 1e-10:
            raise ValueError("Evaluation grid extends beyond the reconstruction")
    yy, xx = np.meshgrid(
        *(np.clip(t, a[0], a[-1]) for a, t in zip(axes, targets)), indexing="ij"
    )
    return RegularGridInterpolator(axes, image)(np.stack([yy, xx], axis=-1))


def render(root, out):
    out.mkdir(parents=True, exist_ok=True)
    try:
        findfont(FontProperties(family="Times New Roman"), fallback_to_default=False)
        font = "Times New Roman"
    except ValueError:
        font = "DejaVu Serif"
    plt.rcParams.update({"font.family": font, "font.size": 11})
    inventory = json.loads((root / "legacy_fwi_inventory.json").read_text())
    records, sources = [], []
    for dataset, labels in DATASETS.items():
        refs = []
        for label in labels:
            path = root / dataset / label / "reference.npz"
            with np.load(path, allow_pickle=False) as archive:
                refs.append({key: archive[key] for key in archive.files})
            sources.append(
                {"file": path.relative_to(root).as_posix(), "sha256": checksum(path)}
            )
        lo = min(float(ref["ground_truth_mps"].min()) for ref in refs)
        hi = max(float(ref["ground_truth_mps"].max()) for ref in refs)
        for tier, suffix in (("matched", "matched"), ("kwave_fullrate", "kwave")):
            fig, axes = plt.subplots(4, 7, figsize=(21, 13))
            fig.subplots_adjust(
                left=0.05, right=0.945, top=0.88, bottom=0.07, wspace=0.13, hspace=0.34
            )
            for row, (label, ref) in enumerate(zip(labels, refs)):
                folder = root / dataset / label
                gt = ref["ground_truth_mps"]
                legacy = next(
                    e
                    for e in inventory
                    if e["dataset"] == dataset and e["label"] == label
                )
                for col in range(7):
                    ax = axes[row, col]
                    ax.set_axis_off()
                    if col == 0:
                        image = gt
                    elif col == 6:
                        image = ref["sound_speed_mps"]
                        record = {
                            "method": "fwi_reference",
                            "status": "stored_reference",
                            "stop_reason": "historical_schedule_completed",
                            "iterations": legacy["iterations"],
                            "original_result_sha256": legacy["result_sha256"],
                            "frequencies_hz": legacy["frequencies_hz"],
                            "measurement_source": "historical_kwave_not_regenerated_acquisition",
                            "native_shape": list(image.shape),
                        }
                    else:
                        method = METHODS[col - 1]
                        info = json.loads(
                            (folder / f"{tier}_{method}_complete.json").read_text()
                        )
                        case_path = folder / Path(info["case_path"]).name
                        # Relocate the saved run by its dataset-relative suffix.
                        parts = Path(info["result_path"]).parts
                        result_path = folder.joinpath(*parts[parts.index("recon") :])
                        if checksum(case_path) != info["case_sha256"]:
                            raise ValueError(f"Case hash mismatch: {case_path.name}")
                        case = read_case_hdf5(case_path)
                        result = read_result_hdf5(result_path)
                        if (
                            info["status"] != "success"
                            or result.sound_speed_mps is None
                        ):
                            raise ValueError(
                                f"Missing successful final result: {dataset}/{label}/{method}"
                            )
                        image = align(result.sound_speed_mps, case.grid, ref)
                        record = {
                            key: value
                            for key, value in info.items()
                            if key
                            not in {"case_path", "result_path", "rmse", "psnr", "ssim"}
                        }
                        record.update(
                            case_file=case_path.relative_to(root).as_posix(),
                            result_file=result_path.relative_to(root).as_posix(),
                            result_sha256=checksum(result_path),
                            native_shape=list(case.grid.shape),
                            measurement_source=case.metadata.get(
                                "measurement_provenance"
                            ),
                        )
                    if image.shape != gt.shape or not np.isfinite(image).all():
                        raise ValueError("Invalid aligned image")
                    im = ax.imshow(
                        image, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest"
                    )
                    if col:
                        metrics = compute_regional_image_metrics(image, gt)
                        records.append(
                            {
                                "dataset": dataset,
                                "label": label,
                                "tier": tier,
                                **record,
                                **metrics,
                            }
                        )
                        ax.text(
                            0.5,
                            -0.055,
                            f'PSNR {metrics["psnr"]:.2f}   SSIM {metrics["ssim"]:.3f}',
                            transform=ax.transAxes,
                            ha="center",
                            va="top",
                            fontsize=10.5,
                            fontweight="bold",
                        )
                    if row == 0:
                        headers = [
                            "GT",
                            "CGLS",
                            "SIRT",
                            "SART",
                            "Bent",
                            "Ray-Born",
                            "FWI (reference)",
                        ]
                        ax.set_title(
                            headers[col], fontsize=16, fontweight="bold", pad=13
                        )
                    if col == 0:
                        ax.text(
                            -0.12,
                            0.5,
                            label,
                            transform=ax.transAxes,
                            ha="right",
                            va="center",
                            fontsize=16,
                            fontweight="bold",
                        )
            title = (
                "OpenBreastUS"
                if dataset == "openbreastus"
                else "NBPslice2D: 2D Acoustic Numerical Breast Phantoms for USCT"
            )
            fig.suptitle(title, fontsize=22, fontweight="bold", y=0.976)
            subtitle = (
                "Model-matched: straight ToF / Eikonal ToF / fixed Born pressure"
                if tier == "matched"
                else "k-Wave-derived ToF / Eikonal inversion / nonlinear Full-Green pressure inversion"
            )
            fig.text(0.5, 0.936, subtitle, ha="center", fontsize=13)
            fig.text(
                0.5,
                0.034,
                "128 TX / 128 RX | Tissue PSNR (dB) / SSIM | FWI: stored result, different acquisition history and budget",
                ha="center",
                fontsize=10.5,
            )
            cax = fig.add_axes([0.958, 0.27, 0.009, 0.43])
            fig.colorbar(im, cax=cax)
            cax.set_title("m/s", fontsize=11)
            fig.savefig(
                out / f"{dataset}_{suffix}.png",
                dpi=130,
                facecolor="white",
                pil_kwargs={"optimize": True},
            )
            plt.close(fig)
            path = out / f"{dataset}_{suffix}.png"
            # All plotted channels are grayscale; remove redundant RGB/alpha bytes.
            with Image.open(path) as rendered:
                gray = rendered.convert("L")
            gray.save(path, optimize=True)
    manifest = {
        "study": root.name,
        "render_only": True,
        "renderer_sha256": checksum(Path(__file__)),
        "font": font,
        "evaluation": "801x801 reference physical coordinates; linear interpolation; shared GT range per dataset; no rotations or sharpening",
        "fwi_is_current_wust_validation": False,
        "reference_files": sources,
        "records": records,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    fields = [
        "dataset",
        "label",
        "tier",
        "method",
        "status",
        "stop_reason",
        "iterations",
        "rmse",
        "psnr",
        "ssim",
        "full_image_rmse",
        "full_image_psnr",
        "full_image_ssim",
        "water_background_rmse",
    ]
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Preserved readme_high_frequency_reference study directory",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/assets/reconstruction"))
    args = parser.parse_args()
    render(args.root, args.out)
