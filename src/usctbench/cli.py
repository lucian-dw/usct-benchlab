"""Thin command handlers over the public reconstruction and data APIs."""

import argparse
import json
import os
from pathlib import Path

from usctbench import describe, evaluate, list_algorithms, run, suite
from usctbench.core.config import load_yaml
from usctbench.core.io import read_case_hdf5, write_case_hdf5
from usctbench.core.schema import GridSpec
from usctbench.data import openbreastus, nbpslice2d
from usctbench.data.synthetic import make_sound_speed_case
from usctbench.data.waveforms import convert_kwave_pressure_mat


def _print(value):
    print(json.dumps(value, indent=2, allow_nan=False))


def _list(args):
    entries = list_algorithms()
    if args.json:
        _print([entry.describe() for entry in entries])
    else:
        for entry in entries:
            print(f"{entry.name}: {entry.description}")
    return 0


def _describe(args):
    spec = describe(args.algorithm_id, args.variant)
    if args.case:
        from usctbench.core.algorithm_specs import case_capabilities

        spec["case_capabilities"] = case_capabilities(
            args.algorithm_id, read_case_hdf5(args.case), variant=args.variant
        )
    _print(spec)
    return 0


def _prepare(args):
    if args.dataset == "synthetic":
        case = make_sound_speed_case(
            shape=tuple(args.shape),
            n_transducers=args.n_transducers,
            background_mps=args.reference_sound_speed_mps,
            inclusion_mps=args.inclusion_mps,
            inclusion_radius_m=args.inclusion_radius_m,
        )
        write_case_hdf5(case, Path(args.out) / "cases" / f"{case.case_id}.h5")
    elif args.dataset == "openbreastus":
        if not args.root or args.spacing_m is None:
            raise ValueError("OpenBreastUS requires --root and explicit --spacing-m")
        openbreastus.prepare_cases(
            args.root,
            args.out,
            cases_per_density=args.cases_per_class,
            converted_shape=tuple(args.shape),
            spacing_m=tuple(args.spacing_m),
            n_transducers=args.n_transducers,
            sample_axis=args.sample_axis,
            image_axes=args.image_axes,
            sound_speed_units=args.sound_speed_units,
        )
    else:
        if not args.root:
            raise ValueError("NBPslice2D requires --root pointing to NBPslices2D.zip")
        nbpslice2d.prepare_cases(
            args.root,
            args.out,
            cases_per_type=args.cases_per_class,
            converted_shape=tuple(args.shape),
            n_transducers=args.n_transducers,
            reference_sound_speed_mps=args.reference_sound_speed_mps,
        )
    print(args.out)
    return 0


def _import_pressure(args):
    grid = GridSpec.model_validate(load_yaml(args.grid)) if args.grid else None
    case = convert_kwave_pressure_mat(
        args.input,
        args.out,
        grid=grid,
        frequencies_hz=args.frequencies_hz,
        tx_indices=args.tx_indices,
        rx_indices=args.rx_indices,
        channel_axes=args.channel_axes.split(","),
        reference_sound_speed_mps=args.reference_sound_speed_mps,
        water_dataset=args.water_dataset,
    )
    _print(
        {
            "case_id": case.case_id,
            "out": args.out,
            "time_shape": list(case.measurement.time_data.shape),
        }
    )
    return 0


def _inspect(args):
    if args.dataset == "openbreastus":
        value = openbreastus.inspect_openbreastus(args.input)
    elif args.dataset == "nbpslice2d":
        value = nbpslice2d.inspect_nbp_slice2d_zip(args.input)
    else:
        case = read_case_hdf5(args.input)
        value = {
            "case_id": case.case_id,
            "grid": {
                "shape": case.grid.shape,
                "spacing_m": case.grid.spacing_m,
                "origin_m": case.grid.origin_m,
            },
            "domain": str(case.measurement.domain),
            "tx": len(case.geometry.tx_pos_m),
            "rx": len(case.geometry.rx_pos_m),
        }
    _print(value)
    return 0


def _run(args):
    deployment = {
        key: value
        for key, value in {
            "runtime_root": args.runtime_root,
            "matlab_executable": args.matlab_executable,
            "backend": args.backend,
            "cuda_visible_devices": args.cuda_visible_devices,
        }.items()
        if value is not None
    }
    result = run(
        args.algorithm,
        args.case,
        args.config,
        args.out,
        deployment=deployment or None,
        preview=not args.no_preview,
    )
    _print(
        {
            "status": str(result.status),
            "stop_reason": result.metrics.get("stop_reason"),
            "failure_reason": result.failure_reason,
            "artifacts": result.artifacts,
        }
    )
    return int(result.status == "failed")


def _evaluate(args):
    result = evaluate(args.run)
    _print(result)
    return int(not result["passed"])


def _suite(args):
    result = suite(args.suite)
    _print(result)
    return int(not result["passed"])


def _demo(args):
    case = make_sound_speed_case(shape=(16, 16), n_transducers=16)
    result = run(
        "straight_cgls", case, {"run_controls": {"max_iterations": 10}}, args.out
    )
    _print({"status": str(result.status), "out": args.out})
    return int(result.status == "failed")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="usct", description="USCT benchmark command-line interface."
    )
    commands = parser.add_subparsers()
    p = commands.add_parser("list-algorithms")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_list)
    p = commands.add_parser("describe-algorithm")
    p.add_argument("algorithm_id")
    p.add_argument("--variant")
    p.add_argument("--json", action="store_true")
    p.add_argument("--case")
    p.set_defaults(handler=_describe)
    data = commands.add_parser("data").add_subparsers(required=True)
    p = data.add_parser("prepare")
    p.add_argument(
        "--dataset", choices=("synthetic", "openbreastus", "nbpslice2d"), required=True
    )
    p.add_argument("--root")
    p.add_argument("--out", required=True)
    p.add_argument("--shape", type=int, nargs=2, default=(32, 32), metavar=("NY", "NX"))
    p.add_argument("--n-transducers", type=int, default=32)
    p.add_argument("--cases-per-class", type=int, default=1)
    p.add_argument("--spacing-m", type=float, nargs=2, metavar=("DY", "DX"))
    p.add_argument("--sample-axis", type=int, choices=(0, 1, 2), default=2)
    p.add_argument("--image-axes", choices=("yx", "xy"), default="yx")
    p.add_argument("--sound-speed-units", choices=("m/s", "km/s"), default="m/s")
    p.add_argument("--reference-sound-speed-mps", type=float, default=1500)
    p.add_argument("--inclusion-mps", type=float, default=1450)
    p.add_argument("--inclusion-radius-m", type=float, default=0.006)
    p.set_defaults(handler=_prepare)
    p = data.add_parser("import-pressure")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--grid", help="GridSpec YAML; required if acquisition lacks grid coordinates"
    )
    p.add_argument("--frequencies-hz", type=float, nargs="+")
    p.add_argument("--tx-indices", type=int, nargs="+")
    p.add_argument("--rx-indices", type=int, nargs="+")
    p.add_argument("--channel-axes", required=True, help="Stored axes, e.g. tx,rx,time")
    p.add_argument("--reference-sound-speed-mps", required=True, type=float)
    p.add_argument("--water-dataset")
    p.set_defaults(handler=_import_pressure)
    p = commands.add_parser("inspect")
    p.add_argument("input")
    p.add_argument(
        "--dataset", choices=("case", "openbreastus", "nbpslice2d"), default="case"
    )
    p.set_defaults(handler=_inspect)
    p = commands.add_parser("run")
    p.add_argument("algorithm")
    p.add_argument("--case", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True, help="Exact output directory for this case")
    p.add_argument("--no-preview", action="store_true")
    for flag, environment in (
        ("runtime-root", "WUST_RUNTIME_ROOT"),
        ("matlab-executable", "WUST_MATLAB_EXECUTABLE"),
        ("cuda-visible-devices", "CUDA_VISIBLE_DEVICES"),
    ):
        p.add_argument("--" + flag, default=os.environ.get(environment))
    p.add_argument(
        "--backend", choices=("cpu", "gpu"), default=os.environ.get("WUST_BACKEND")
    )
    p.set_defaults(handler=_run)
    p = commands.add_parser("evaluate")
    p.add_argument("--run", required=True)
    p.set_defaults(handler=_evaluate)
    p = commands.add_parser("bench")
    p.add_argument("--suite", required=True)
    p.set_defaults(handler=_suite)
    p = commands.add_parser("demo")
    p.add_argument("--out", default="runs/demo")
    p.set_defaults(handler=_demo)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    try:
        return args.handler(args)
    except (KeyError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
