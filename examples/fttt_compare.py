"""Reproduce the small Eikonal reference comparison; no reconstruction GT enters solvers.

Run from the repository root with the package installed or PYTHONPATH=src:
python examples/fttt_compare.py --checkpoint checkpoints/fttt_reference_16.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np

from usctbench import run
from usctbench.algorithms.fttt.train_prior import synthetic_speed_images
from usctbench.core.io import write_case_hdf5
from usctbench.core.schema import USCTCase, MeasurementSpec, GroundTruthSpec
from usctbench.data.synthetic import make_grid, make_ring_geometry
from usctbench.operators.eikonal import EikonalForward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="runs/fttt_reference")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"use a new output directory: {out}")
    out.mkdir(parents=True)
    shape = (16, 16)
    grid, geometry = make_grid(shape), make_ring_geometry(12, 0.012)
    # Independent test seed, distinct from prior training and validation.
    speed = synthetic_speed_images(1, shape, seed=907)[0].astype(float)
    native = EikonalForward(grid, geometry)
    distance = np.linalg.norm(
        geometry.tx_pos_m[:, None] - geometry.rx_pos_m[None], axis=-1
    )
    delta = native.forward(1 / speed).reshape(distance.shape) - distance / 1500
    valid = distance > 0
    case = USCTCase(
        case_id="fttt_independent_ellipse_907",
        grid=grid,
        geometry=geometry,
        measurement=MeasurementSpec(
            domain="features", delta_tof_s=delta, valid_mask=valid
        ),
        ground_truth=GroundTruthSpec(sound_speed_mps=speed),
        metadata={
            "reference_sound_speed_mps": 1500.0,
            "synthetic": True,
            "forward_model": "eikonal_oracle",
            "test_image_seed": 907,
            "limitation": "Same-model synthetic integration evidence, not experimental imaging validation.",
        },
    )
    write_case_hdf5(case, out / "case.h5")
    records, images = [], {"truth": speed}
    requests = [
        (
            "bent_ray_gn",
            "bent_ray_gn",
            {
                "parameters": {"regularization_lambda": 0.02},
                "run_controls": {"max_iterations": 4},
            },
        ),
    ]
    for seed in args.seeds:
        for variant in ("dip", "inr", "dps", "dps_unguided"):
            method = "dps" if variant.startswith("dps") else variant
            parameters = {"variant": method, "seed": seed}
            if method == "dps":
                parameters.update(
                    checkpoint=str(Path(args.checkpoint).resolve()),
                    sampling_steps=args.sampling_steps,
                    guidance_scale=0.0 if variant == "dps_unguided" else 0.1,
                )
            requests.append(
                (
                    f"{variant}_seed{seed}",
                    "fttt",
                    {
                        "parameters": parameters,
                        "run_controls": {
                            "max_iterations": (
                                args.sampling_steps
                                if method == "dps"
                                else args.iterations
                            )
                        },
                    },
                )
            )
    for label, algorithm, request in requests:
        result = run(algorithm, case, request, out / label, preview=False)
        if result.status == "failed":
            raise RuntimeError(f"{label}: {result.failure_reason}")
        metrics = result.metrics
        record = {
            "label": label,
            "status": str(result.status),
            "stop_reason": metrics.get("stop_reason"),
            "relative_data_residual": metrics.get("data_relative_residual"),
            "data_residual_reduction": metrics.get("data_residual_reduction"),
            "speed_rmse_mps": float(
                np.sqrt(np.mean((result.sound_speed_mps - speed) ** 2))
            ),
            "runtime_s": result.runtime_s,
            "work": metrics["stopping"]["work"],
            "pretrained_prior": metrics.get("pretrained_prior"),
        }
        records.append(record)
        images[label] = result.sound_speed_mps
        print(
            json.dumps({k: v for k, v in record.items() if k != "pretrained_prior"}),
            flush=True,
        )
    (out / "summary.json").write_text(
        json.dumps(
            {
                "scope": "Independent synthetic test; fixed per-method defaults, no hyperparameter search.",
                "budget_note": "Unequal native iteration units; actual wall time and operator calls are reported.",
                "records": records,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = [
        "truth",
        "bent_ray_gn",
        *[f"{v}_seed{args.seeds[0]}" for v in ("dip", "inr", "dps", "dps_unguided")],
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for ax, label in zip(axes.flat, labels):
        im = ax.imshow(
            images[label], vmin=1400, vmax=1600, cmap="viridis", origin="upper"
        )
        ax.set_title(label)
        ax.set_axis_off()
    fig.colorbar(im, ax=axes.ravel().tolist(), label="sound speed [m/s]")
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
