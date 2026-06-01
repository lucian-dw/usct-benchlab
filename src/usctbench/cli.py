"""Command-line entry point for the USCT benchmark harness."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from usctbench.benchmark.runner import evaluate_run, run_algorithm_case, run_benchmark_suite
from usctbench.data.nbpslice2d import inspect_nbp_slice2d_zip, make_nbp_slice2d_quality_subset, make_nbp_slice2d_smoke_subset
from usctbench.data.openbreastus import inspect_openbreastus, write_schema_report
from usctbench.data.smoke_subset import make_quality_subset, make_smoke_subset
from usctbench.data.synthetic import make_synthetic_smoke_subset
from usctbench.features import extract_wavefield_features
from usctbench.registry import list_algorithms
from usctbench.sim import run_kwave_simulation_from_config, run_simulation_qc


def register_builtin_algorithms() -> None:
    """Register built-in algorithms exactly once for CLI use."""

    from usctbench.algorithms.adapters import register_adapter_algorithms
    from usctbench.algorithms.fwi import register_fwi_algorithms
    from usctbench.algorithms.ray import register_ray_algorithms

    register_adapter_algorithms(replace=True)
    register_fwi_algorithms(replace=True)
    register_ray_algorithms(replace=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="usct", description="USCT benchmark command-line interface.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list-algorithms", help="List registered reconstruction algorithms.")

    data_parser = subparsers.add_parser("data", help="Data inspection and subset commands.")
    data_subparsers = data_parser.add_subparsers(dest="data_command")

    inspect_parser = data_subparsers.add_parser("inspect-openbreastus", help="Inspect a local OpenBreastUS tree.")
    inspect_parser.add_argument("--root", required=True, help="OpenBreastUS data root.")
    inspect_parser.add_argument("--out", default="openbreastus_index.json", help="Output index JSON path.")

    smoke_parser = data_subparsers.add_parser("make-smoke", help="Create a smoke benchmark subset.")
    smoke_parser.add_argument("--root", required=True, help="OpenBreastUS data root.")
    smoke_parser.add_argument("--out", required=True, help="Output smoke subset root.")
    smoke_parser.add_argument("--cases-per-density", type=int, default=1)
    smoke_parser.add_argument("--converted-shape", type=int, default=64, help="Square image size for converted speed-map cases.")
    smoke_parser.add_argument("--n-transducers", type=int, default=32, help="Synthetic ring transducers for converted speed-map cases.")
    smoke_parser.add_argument("--spacing-m", type=float, default=1.0e-3, help="Assumed pixel spacing for converted speed-map cases.")
    smoke_parser.add_argument("--no-convert-speed-mat", action="store_true", help="Only write the manifest; do not create HDF5 cases.")

    quality_parser = data_subparsers.add_parser("make-quality", help="Create 256x256 OpenBreastUS quality-comparison cases.")
    quality_parser.add_argument("--root", required=True, help="OpenBreastUS data root.")
    quality_parser.add_argument("--out", required=True, help="Output quality subset root.")
    quality_parser.add_argument("--cases-per-density", type=int, default=1)
    quality_parser.add_argument("--converted-shape", type=int, default=256, help="Square image size for quality comparison.")
    quality_parser.add_argument("--n-transducers", type=int, default=128, help="Synthetic ring transducers for quality comparison.")
    quality_parser.add_argument("--spacing-m", type=float, default=1.0e-3, help="Assumed pixel spacing for converted speed-map cases.")
    quality_parser.add_argument("--no-convert-speed-mat", action="store_true", help="Only write the manifest; do not create HDF5 cases.")

    nbp_inspect_parser = data_subparsers.add_parser("inspect-nbpslice2d", help="Inspect an NBPslices2D ZIP archive.")
    nbp_inspect_parser.add_argument("--zip", required=True, help="NBPslices2D ZIP path.")
    nbp_inspect_parser.add_argument("--out", default="nbpslice2d_index.json", help="Output index JSON path.")

    nbp_smoke_parser = data_subparsers.add_parser("make-nbp-smoke", help="Create an NBPslices2D smoke benchmark subset.")
    nbp_smoke_parser.add_argument("--zip", required=True, help="NBPslices2D ZIP path.")
    nbp_smoke_parser.add_argument("--out", required=True, help="Output smoke subset root.")
    nbp_smoke_parser.add_argument("--cases-per-type", type=int, default=1, help="Cases to convert for each A/B/C/D density label.")
    nbp_smoke_parser.add_argument("--converted-shape", type=int, default=64, help="Square image size for converted cases.")
    nbp_smoke_parser.add_argument("--n-transducers", type=int, default=32, help="Synthetic ring transducers for converted cases.")
    nbp_smoke_parser.add_argument("--reference-sound-speed-mps", type=float, default=1500.0)
    nbp_smoke_parser.add_argument("--attenuation-frequency-mhz", type=float, default=1.0)

    nbp_quality_parser = data_subparsers.add_parser("make-nbp-quality", help="Create 256x256 NBPslice2D quality-comparison cases.")
    nbp_quality_parser.add_argument("--zip", required=True, help="NBPslices2D ZIP path.")
    nbp_quality_parser.add_argument("--out", required=True, help="Output quality subset root.")
    nbp_quality_parser.add_argument("--cases-per-type", type=int, default=1, help="Cases to convert for each A/B/C/D density label.")
    nbp_quality_parser.add_argument("--converted-shape", type=int, default=256, help="Square image size for quality comparison.")
    nbp_quality_parser.add_argument("--n-transducers", type=int, default=128, help="Synthetic ring transducers for quality comparison.")
    nbp_quality_parser.add_argument("--reference-sound-speed-mps", type=float, default=1500.0)
    nbp_quality_parser.add_argument("--attenuation-frequency-mhz", type=float, default=1.0)

    synthetic_smoke_parser = data_subparsers.add_parser("make-synthetic-smoke", help="Create deterministic local synthetic smoke cases.")
    synthetic_smoke_parser.add_argument("--out", default="data/synthetic_smoke", help="Output synthetic smoke subset root.")
    synthetic_smoke_parser.add_argument("--shape", type=int, default=24, help="Square image size.")
    synthetic_smoke_parser.add_argument("--n-transducers", type=int, default=24)

    run_parser = subparsers.add_parser("run", help="Run one algorithm on one case.")
    run_parser.add_argument("algorithm", help="Registered algorithm name.")
    run_parser.add_argument("--case", required=True, help="Input case HDF5 path.")
    run_parser.add_argument("--config", required=True, help="Algorithm YAML config path.")
    run_parser.add_argument("--out", required=True, help="Output run directory.")

    eval_parser = subparsers.add_parser("eval", help="Evaluate one run directory.")
    eval_parser.add_argument("--run", required=True, help="Run directory.")
    eval_parser.add_argument("--protocol", required=True, help="Benchmark protocol YAML path.")

    sim_parser = subparsers.add_parser("sim", help="Simulation commands.")
    sim_subparsers = sim_parser.add_subparsers(dest="sim_command")
    kwave_parser = sim_subparsers.add_parser("kwave", help="Generate or reuse a k-Wave-compatible wavefield case.")
    kwave_parser.add_argument("--config", required=True, help="Simulation YAML config path.")
    qc_parser = sim_subparsers.add_parser("qc", help="Run simulation QC for a wavefield case.")
    qc_parser.add_argument("--case", required=True, help="Wavefield case HDF5 path.")
    qc_parser.add_argument("--out", default=None, help="QC artifact directory; defaults to the case directory.")

    features_parser = subparsers.add_parser("features", help="Wavefield feature extraction commands.")
    features_subparsers = features_parser.add_subparsers(dest="features_command")
    extract_parser = features_subparsers.add_parser("extract", help="Extract algorithm-ready features from a wavefield case.")
    extract_parser.add_argument("--case", required=True, help="Wavefield case HDF5 path.")
    extract_parser.add_argument("--method", default="all", choices=["all", "xcorr", "phase-slope"], help="Selected ToF feature for solver input.")
    extract_parser.add_argument("--out", default=None, help="Feature case HDF5 output path.")

    bench_parser = subparsers.add_parser("bench", help="Run a benchmark suite.")
    bench_parser.add_argument("--suite", required=True, help="Benchmark suite YAML path.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    register_builtin_algorithms()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "list-algorithms":
        entries = list_algorithms()
        if not entries:
            print("No algorithms registered.")
            return 0
        for entry in entries:
            tags = f" [{' '.join(entry.tags)}]" if entry.tags else ""
            suffix = f" - {entry.description}" if entry.description else ""
            print(f"{entry.name}{tags}{suffix}")
        return 0

    if args.command == "data":
        if args.data_command is None:
            parser.parse_args(["data", "--help"])
            return 0
        if args.data_command == "inspect-openbreastus":
            index = inspect_openbreastus(args.root, args.out)
            write_schema_report(index, Path(args.out).with_suffix(".schema_report.md"))
            print(args.out)
            return 0
        if args.data_command == "make-smoke":
            make_smoke_subset(
                args.root,
                args.out,
                cases_per_density=args.cases_per_density,
                convert_speed_mat=not args.no_convert_speed_mat,
                converted_shape=(args.converted_shape, args.converted_shape),
                spacing_m=(args.spacing_m, args.spacing_m),
                n_transducers=args.n_transducers,
            )
            print(args.out)
            return 0
        if args.data_command == "make-quality":
            make_quality_subset(
                args.root,
                args.out,
                cases_per_density=args.cases_per_density,
                convert_speed_mat=not args.no_convert_speed_mat,
                converted_shape=(args.converted_shape, args.converted_shape),
                spacing_m=(args.spacing_m, args.spacing_m),
                n_transducers=args.n_transducers,
            )
            print(args.out)
            return 0
        if args.data_command == "inspect-nbpslice2d":
            inspect_nbp_slice2d_zip(args.zip, args.out)
            print(args.out)
            return 0
        if args.data_command == "make-nbp-smoke":
            make_nbp_slice2d_smoke_subset(
                args.zip,
                args.out,
                cases_per_type=args.cases_per_type,
                converted_shape=(args.converted_shape, args.converted_shape),
                n_transducers=args.n_transducers,
                reference_sound_speed_mps=args.reference_sound_speed_mps,
                attenuation_frequency_mhz=args.attenuation_frequency_mhz,
            )
            print(args.out)
            return 0
        if args.data_command == "make-nbp-quality":
            make_nbp_slice2d_quality_subset(
                args.zip,
                args.out,
                cases_per_type=args.cases_per_type,
                converted_shape=(args.converted_shape, args.converted_shape),
                n_transducers=args.n_transducers,
                reference_sound_speed_mps=args.reference_sound_speed_mps,
                attenuation_frequency_mhz=args.attenuation_frequency_mhz,
            )
            print(args.out)
            return 0
        if args.data_command == "make-synthetic-smoke":
            make_synthetic_smoke_subset(
                args.out,
                shape=(args.shape, args.shape),
                n_transducers=args.n_transducers,
            )
            print(args.out)
            return 0

    if args.command == "run":
        out_dir = run_algorithm_case(args.algorithm, args.case, args.config, args.out)
        print(out_dir / "result.h5")
        return 0 if not (out_dir / "failure_report.md").exists() else 1

    if args.command == "eval":
        result = evaluate_run(args.run, args.protocol)
        print(result["report_md"])
        return 0 if _benchmark_passed(result) else 1

    if args.command == "sim":
        if args.sim_command is None:
            parser.parse_args(["sim", "--help"])
            return 0
        if args.sim_command == "kwave":
            wave_case = run_kwave_simulation_from_config(args.config)
            run_simulation_qc(wave_case)
            print(wave_case)
            return 0
        if args.sim_command == "qc":
            result = run_simulation_qc(args.case, args.out)
            print(Path(args.out or args.case).parent / "simulation_qc.json" if args.out is None else Path(args.out) / "simulation_qc.json")
            return 0 if result.get("passed") else 1

    if args.command == "features":
        if args.features_command is None:
            parser.parse_args(["features", "--help"])
            return 0
        if args.features_command == "extract":
            out = Path(args.out) if args.out else _default_feature_case_path(Path(args.case))
            extract_wavefield_features(args.case, out=out, method=args.method)
            print(out)
            return 0

    if args.command == "bench":
        result = run_benchmark_suite(args.suite)
        print(result["run_root"])
        return 0 if _benchmark_passed(result) else 1

    parser.error(f"unknown command: {args.command}")
    return 2


def _benchmark_passed(result: dict) -> bool:
    return bool(result["records"]) and all(record.get("pass") for record in result["records"]) and bool(result.get("run_checks", {}).get("passed"))


def _default_feature_case_path(case_path: Path) -> Path:
    if case_path.parent.name == "wavefield_cases":
        return case_path.parent.parent / "feature_cases" / f"{case_path.stem}_features.h5"
    return case_path.with_name(f"{case_path.stem}_features.h5")


if __name__ == "__main__":
    raise SystemExit(main())
