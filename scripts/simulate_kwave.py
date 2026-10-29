#!/usr/bin/env python3
"""Prepare, simulate and import a lossless k-Wave object/water pressure pair.

Run --help. All generated arrays/logs live below --out, never in source control.
External MATLAB/k-Wave run only through the explicit `simulate` command.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
from scipy.io import savemat
from scipy.ndimage import map_coordinates

from usctbench.core.io import read_case_hdf5, write_case_hdf5
from usctbench.data.arrival import arrival_observation_metadata, water_relative_delays
from usctbench.data.waveforms import convert_kwave_pressure_mat


def prepare(args):
    case = read_case_hdf5(args.case)
    speed = case.ground_truth.sound_speed_mps
    if speed is None:
        raise ValueError("simulation requires an explicitly labelled property map")
    out = Path(args.out)
    if (out / "manifest.json").exists() or (out / "pressure.mat").exists():
        raise FileExistsError("prepare requires a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    n = args.transducers
    if n < 4 or n > len(case.geometry.tx_pos_m):
        raise ValueError("transducers must be between 4 and the available source count")
    if (
        not np.isfinite(args.source_frequency)
        or args.source_frequency <= 0
        or not np.isfinite(args.max_frequency)
        or args.max_frequency < args.source_frequency
    ):
        raise ValueError("frequencies must be finite, positive, and maximum >= source")
    if not np.all(np.isfinite(speed)) or np.min(speed) <= 0:
        raise ValueError("property speed must be finite and positive")
    indices = np.linspace(0, len(case.geometry.tx_pos_m), n, endpoint=False, dtype=int)
    positions = case.geometry.tx_pos_m[indices]
    extent = np.max(np.abs(positions)) * 2.6
    # Enough space for the recorded ring and an internal PML. The pixel size is
    # chosen independently of image labels except for the conservative PPW bound.
    size = int(
        2
        ** np.ceil(np.log2(extent * args.max_frequency * 8 / min(1300.0, speed.min())))
    )
    # With the 2.6-radius extent, at least 256 cells leave 24 pixels between
    # the outermost element and the edge (20-cell PML plus a source margin).
    size = max(256, size)
    dx = extent / size
    y, x = np.mgrid[:size, :size]
    points = np.array([y - size // 2, x - size // 2]) * dx
    coordinates = [
        (points[a] - case.grid.origin_m[a]) / case.grid.spacing_m[a] - 0.5
        for a in range(2)
    ]
    sim = map_coordinates(speed, coordinates, order=1, mode="constant", cval=1500.0)
    elements = np.rint(positions / dx).astype(int) + size // 2
    positions = (elements - size // 2) * dx
    if (
        len(np.unique(elements, axis=0)) != n
        or np.any(elements < 24)
        or np.any(elements >= size - 24)
    ):
        raise ValueError("duplicate elements or geometry overlapping the PML")
    input_path = out / "simulation_input.mat"
    savemat(
        input_path,
        {
            "sim_speed_yx": sim,
            "image_speed_yx": speed,
            "image_y": case.grid.origin_m[0]
            + (np.arange(speed.shape[0]) + 0.5) * case.grid.spacing_m[0],
            "image_x": case.grid.origin_m[1]
            + (np.arange(speed.shape[1]) + 0.5) * case.grid.spacing_m[1],
            "positions_yx": positions,
            "elements_yx": elements,
            "dx": dx,
            "pml": 20.0,
            "source_frequency_hz": args.source_frequency,
            "maximum_frequency_hz": args.max_frequency,
            "end_time_s": 2.4
            * np.max(np.linalg.norm(positions, axis=1))
            / min(1300.0, speed.min())
            + 6 / args.source_frequency,
        },
    )
    record = {
        "case_id": case.case_id,
        "property_case": str(Path(args.case).resolve()),
        "dataset": case.metadata.get("source_dataset"),
        "density_label": case.metadata.get("density_label"),
        "simulation_shape": [size, size],
        "spacing_m": dx,
        "transducers": n,
        "source_frequency_hz": args.source_frequency,
        "maximum_frequency_hz": args.max_frequency,
        "density_kg_per_m3": 1000,
        "attenuation": 0,
        "physics_scope": "2D lossless constant-density SOS validation, not clinical measurements",
    }
    (out / "manifest.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


def simulate(args):
    out = Path(args.out).resolve()
    target = out / "pressure.mat"
    if target.exists():
        raise FileExistsError(
            "pressure.mat already exists; use a fresh output directory"
        )
    pending = out / "pressure.partial.mat"
    if pending.exists():
        raise FileExistsError(
            "incomplete pressure.partial.mat exists; inspect it or use a fresh output directory"
        )
    script_dir = Path(__file__).resolve().parent

    def quote(s):
        return str(s).replace("'", "''")

    expression = (
        f"s=settings; addpath('{quote(script_dir)}'); kwave_validation_pair('{quote(out/'simulation_input.mat')}',"
        f"'{quote(pending)}','{quote(args.kwave_path)}',{args.device});"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"LD_PRELOAD", "LD_LIBRARY_PATH"}
    }
    with (out / "simulation.log").open("w") as log:
        subprocess.run(
            [args.matlab, "-batch", expression],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    pending.rename(target)
    print(target)


def extract(args):
    out = Path(args.out)
    record = json.loads((out / "manifest.json").read_text())
    if (
        args.image_size < 8
        or not args.frequencies
        or min(args.frequencies) <= 0
        or max(args.frequencies) > record["maximum_frequency_hz"]
    ):
        raise ValueError(
            "image_size must be >= 8 and frequencies inside the simulated PPW band"
        )
    case_file = getattr(args, "case_file", "pressure_case.h5")
    if Path(case_file).name != case_file or not case_file.endswith(".h5"):
        raise ValueError(
            "case-file must be an HDF5 filename within the output directory"
        )
    path = out / case_file
    case = convert_kwave_pressure_mat(
        out / "pressure.mat",
        path,
        frequencies_hz=np.array(args.frequencies),
        output_shape=(args.image_size, args.image_size),
        water_dataset="water_dataset",
    )
    case.case_id = record["case_id"]
    case.metadata.update(record)
    distances = np.linalg.norm(
        case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None], axis=-1
    )
    delta, valid, weights, qc = water_relative_delays(
        case.measurement.time_data,
        case.measurement.water_reference_time,
        case.measurement.time_axis_s,
        distances,
        pulse_duration_s=3 / record["source_frequency_hz"],
        picker=args.tof_method,
        envelope_fraction=args.envelope_fraction,
        aic_envelope_fraction=args.aic_envelope_fraction,
        aic_window_s=args.aic_window_s,
    )
    # Direct-adjacent pairs do not cross the object and are most sensitive to
    # finite source support. The rule depends on geometry only, never on labels.
    geometry_valid = distances > 0.5 * np.max(distances)
    valid &= geometry_valid
    # A ToF rejection is represented by delta_tof_s=NaN, not by hiding that
    # pressure channel at every frequency in a later pressure-domain holdout.
    case.measurement.valid_mask &= geometry_valid
    case.measurement.delta_tof_s = np.where(valid, delta, np.nan)
    case.measurement.ray_weights = np.where(valid, weights, 0)
    case.metadata.update(arrival_observation_metadata(qc))
    case.metadata["arrival_qc"] = dict(qc)
    with h5py.File(out / "pressure.mat") as f:
        qc.update(
            {
                k: np.asarray(f[k]).squeeze().tolist()
                for k in (
                    "simulation_cfl",
                    "simulation_ppw",
                    "simulation_shape",
                    "pml_pixels",
                )
            }
        )
    qc["nan_inf_count"] = int((~np.isfinite(case.measurement.time_data)).sum())
    qc["nan_inf_count"] += int(
        (~np.isfinite(case.measurement.water_reference_time)).sum()
    )
    qc["used_pair_fraction"] = float(valid.sum() / max(np.sum(distances > 0), 1))
    qc["gate_scope"] = (
        "CFL/PPW/finiteness/channel coverage, not complete physical validation"
    )
    qc["passed"] = bool(
        qc["nan_inf_count"] == 0
        and qc["simulation_cfl"] <= 0.3
        and qc["simulation_ppw"] >= 8
        and qc["used_pair_fraction"] >= 0.5
    )
    case.metadata["simulation_qc"] = dict(qc)
    case.metadata["simulation_qc_passed"] = qc["passed"]
    case.metadata["simulation_failed_qc"] = not qc["passed"]
    write_case_hdf5(case, path)
    qc_path = out / (
        "simulation_qc.json"
        if case_file == "pressure_case.h5"
        else Path(case_file).stem + "_qc.json"
    )
    qc_path.write_text(json.dumps(qc, indent=2))
    print(json.dumps(qc, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--case", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--transducers", type=int, default=16)
    p.add_argument("--source-frequency", type=float, default=200e3)
    p.add_argument("--max-frequency", type=float, default=250e3)
    p = sub.add_parser("simulate")
    p.add_argument("--out", required=True)
    p.add_argument("--matlab", default="matlab")
    p.add_argument("--kwave-path", required=True)
    p.add_argument("--device", type=int, default=0)
    p = sub.add_parser("extract")
    p.add_argument(
        "--tof-method", choices=["xcorr", "envelope", "aic"], default="xcorr"
    )
    p.add_argument("--envelope-fraction", type=float, default=0.1)
    p.add_argument("--aic-envelope-fraction", type=float, default=0.25)
    p.add_argument(
        "--aic-window-s",
        type=float,
        help="AIC window duration in seconds; default is source pulse duration",
    )
    p.add_argument("--case-file", default="pressure_case.h5")
    p.add_argument("--out", required=True)
    p.add_argument("--image-size", type=int, default=48)
    p.add_argument(
        "--frequencies", type=float, nargs="+", default=[150e3, 200e3, 250e3]
    )
    args = parser.parse_args()
    {"prepare": prepare, "simulate": simulate, "extract": extract}[args.action](args)


if __name__ == "__main__":
    main()
