#!/usr/bin/env python3
"""Manual BenchLab/WUST acceptance: one fixture generation, one reconstruction."""

import argparse
import importlib.util
import json
from pathlib import Path
import time

from usctbench.algorithms.fwi.algorithm import WUSTFWIAlgorithm
from usctbench.algorithms.fwi.io import read_artifact
from usctbench.core.io import write_case_hdf5, write_result_hdf5
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    MeasurementSpec,
    USCTCase,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="cpu")
    parser.add_argument("--timeout-s", type=float, default=180)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    fixture_module = args.runtime / "Runtime/tests/integration_runtime.py"
    spec = importlib.util.spec_from_file_location("wust_integration", fixture_module)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    started = time.monotonic()
    path = fixture.create_fixture(args.out / "fixture", args.matlab)
    fixture_seconds = time.monotonic() - started
    doc, arrays = read_artifact(path, "wust.frequency_input")
    grid = doc["metadata"]["grid"]
    pressure = arrays["pressure"].transpose(2, 0, 1)
    mask = arrays["mask"].transpose(2, 0, 1)
    # Positive-sign Fourier data deliberately exercise WUST-owned conjugation.
    case = USCTCase(
        case_id="wust_cross_repository",
        grid=GridSpec(
            shape=grid["shape_yx"],
            spacing_m=grid["spacing_yx_m"],
            origin_m=grid["origin_yx_m"],
        ),
        geometry=GeometrySpec(
            type="custom",
            tx_pos_m=arrays["tx_xy_m"][:, ::-1],
            rx_pos_m=arrays["rx_xy_m"][:, ::-1],
        ),
        measurement=MeasurementSpec(
            domain="frequency",
            frequencies_hz=arrays["frequencies_hz"],
            freq_data=pressure.conj(),
            valid_mask=mask,
        ),
        metadata={
            "reference_sound_speed_mps": 1500.0,
            "measurement_provenance": "coherent_Helmholtz_fixture",
            "pressure_contract": {
                "fourier_sign": 1,
                "real_pressure": True,
                "pressure_type": "total_pressure",
                "data_units": "unknown",
                "spectrum_normalization": "unknown",
            },
        },
    )
    write_case_hdf5(case, args.out / "case.h5")
    _, initial = read_artifact(args.out / "fixture/initial.json", "wust.initial_model")
    config = AlgorithmConfig(
        name="fwi_wust",
        run_controls={"max_elapsed_s": args.timeout_s, "max_iterations": 2},
        parameters={
            "sound_speed_bounds_mps": [1300.0, 1800.0],
            "stencil_bounds_mps": [1400.0, 1700.0],
            "pml_m": 0.003,
            "frequency_schedule_hz": [100000.0, 100000.0, 120000.0],
            "source_batch_size": 2,
            "update_mask": initial["update_mask"].tolist(),
        },
        deployment={
            "runtime_root": str(args.runtime.resolve()),
            "matlab_executable": args.matlab,
            "backend": args.backend,
            "scratch_root": str(args.out),
        },
    )
    (args.out / "config.json").write_text(config.model_dump_json(indent=2))
    result = WUSTFWIAlgorithm().run(case, config)
    write_result_hdf5(result, args.out / "result.h5")
    (args.out / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
    if result.status not in {"completed", "budget_exhausted"}:
        raise RuntimeError(result.failure_reason)
    fixture.check_reconstruction(
        result.metrics["wust"], result.sound_speed_mps, backend=args.backend
    )
    preparation = result.metrics["wust"]["observation_preparation"]
    assert preparation["conversion"] == "conjugate"
    assert preparation["canonical_to_input_permutation_1based"] == [2, 1]
    report = {
        "passed": True,
        "backend": args.backend,
        "shape": list(case.grid.shape),
        "fixture_seconds": fixture_seconds,
        "reconstruction_seconds": result.runtime_s,
        "fixture_matlab_invocations": 1,
        "reconstruction_matlab_invocations": 1,
        "stop_reason": result.metrics["stop_reason"],
        "environment": result.metrics["wust"]["environment"],
    }
    (args.out / "gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
