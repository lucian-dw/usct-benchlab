"""Standalone external k-Wave forward runner.

This module intentionally avoids importing pydantic/usctbench schema objects so
it can run inside the existing A100 `usct-kwave` environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external k-Wave forward-only simulation.")
    parser.add_argument("--usct-kwave-root", required=True)
    parser.add_argument("--mat-path", required=True)
    parser.add_argument("--mat-key", required=True)
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--array-mode", default="partial64")
    parser.add_argument("--background-speed", type=float, default=1500.0)
    parser.add_argument("--ncalc", type=int, default=552)
    parser.add_argument("--xmax-mm", type=float, default=120.0)
    parser.add_argument("--circle-radius-mm", type=float, default=110.0)
    parser.add_argument("--atten-bkgnd", type=float, default=0.0)
    parser.add_argument("--sos2atten", type=float, default=0.0)
    parser.add_argument("--y-atten", type=float, default=1.01)
    parser.add_argument("--f-tx-mhz", type=float, default=0.25)
    parser.add_argument("--frac-bw", type=float, default=0.75)
    parser.add_argument("--cfl", type=float, default=0.3)
    parser.add_argument("--pml-size", type=int, default=20)
    parser.add_argument("--downsample-factor", type=int, default=1)
    parser.add_argument("--backend", default="cuda-binary")
    parser.add_argument("--binary-path", default="")
    parser.add_argument("--generation-mode", choices=["direct"], default="direct")
    parser.add_argument("--direct-num-workers", type=int, default=1)
    parser.add_argument("--kwave-data-path", default="auto")
    parser.add_argument("--kwave-data-name-prefix", default="")
    parser.add_argument("--cuda-device", action="append", type=int, default=[])
    parser.add_argument("--siminfo-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-matlab", action="store_true")
    parser.add_argument("--no-connect-existing", action="store_true")
    parser.add_argument("--shared-engine-name", default="usct_forward")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.usct_kwave_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    from openbreastus_diffusion.kwave_dps.matlab_bridge import connect_matlab
    from openbreastus_diffusion.kwave_dps.run_full_pipeline import matlab_vector
    from openbreastus_diffusion.kwave_dps.utils import matlab_struct_to_python

    eng = connect_matlab(
        shared_engine_name=args.shared_engine_name,
        start_matlab=bool(args.start_matlab),
        connect_existing=not bool(args.no_connect_existing),
    )
    siminfo_path = Path(args.siminfo_path)
    dataset_path = Path(args.dataset_path)
    siminfo_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    siminfo = eng.GenOpenBreastKWaveSimInfo(
        str(args.mat_path),
        float(args.sample_index),
        str(siminfo_path),
        "mat_key",
        str(args.mat_key),
        "array_mode",
        str(args.array_mode),
        "background_speed",
        float(args.background_speed),
        "Ncalc",
        float(args.ncalc),
        "xmax",
        float(args.xmax_mm) * 1.0e-3,
        "circle_radius",
        float(args.circle_radius_mm) * 1.0e-3,
        "PMLSize",
        float(args.pml_size),
        "PMLInside",
        False,
        "cfl",
        float(args.cfl),
        "frac_bw",
        float(args.frac_bw),
        "f_tx",
        float(args.f_tx_mhz) * 1.0e6,
        "atten_bkgnd",
        float(args.atten_bkgnd),
        "sos2atten",
        float(args.sos2atten),
        "y_atten",
        float(args.y_atten),
        "backend",
        str(args.backend),
        "binary_path",
        str(args.binary_path),
        nargout=1,
    )
    generated = eng.GenOpenBreastDatasetDirectFromSimInfo(
        str(siminfo_path),
        str(dataset_path),
        "backend",
        str(args.backend),
        "binary_path",
        str(args.binary_path),
        "downsample_factor",
        float(args.downsample_factor),
        "kwave_data_path",
        str(args.kwave_data_path),
        "kwave_data_name_prefix",
        str(args.kwave_data_name_prefix or dataset_path.stem),
        "num_workers",
        float(args.direct_num_workers),
        "device_nums",
        matlab_vector(args.cuda_device),
        "delete_kwave_data",
        True,
        "overwrite",
        bool(args.overwrite),
        "verbose",
        True,
        nargout=1,
    )
    summary: dict[str, Any] = {
        "siminfo": matlab_struct_to_python(siminfo),
        "generation": matlab_struct_to_python(generated),
        "request": vars(args),
    }
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_json_default, sort_keys=True))
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
