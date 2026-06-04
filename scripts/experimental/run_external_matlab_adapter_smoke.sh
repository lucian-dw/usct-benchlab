#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f "$REPO_DIR/.env" ]; then
  # shellcheck disable=SC1091
  . "$REPO_DIR/.env"
fi

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"
WORKSPACE="${USCT_WORKSPACE:-$(cd "$REPO_DIR/.." && pwd)}"
USCT_DATA_ROOT="${USCT_DATA_ROOT:-$WORKSPACE/data/openbreastus}"
export USCT_EXTERNAL_MATLAB_SAMPLE_ROOT="${USCT_EXTERNAL_MATLAB_SAMPLE_ROOT:-$WORKSPACE/data/openbreastus_external_matlab_smoke_64}"
export USCT_RUN_ROOT="${USCT_RUN_ROOT:-$WORKSPACE/runs/usctbench_runs}"
USCT_SMOKE_CASES_PER_DENSITY="${USCT_SMOKE_CASES_PER_DENSITY:-1}"
USCT_SMOKE_CONVERTED_SHAPE="${USCT_SMOKE_CONVERTED_SHAPE:-64}"
USCT_SMOKE_N_TRANSDUCERS="${USCT_SMOKE_N_TRANSDUCERS:-32}"
export MATLAB_BIN="${MATLAB_BIN:-/usr/local/MATLAB/R2021b/bin/matlab}"
export USCT_REFRACTION_CORRECTED_ROOT="${USCT_REFRACTION_CORRECTED_ROOT:-$WORKSPACE/external/refractionCorrectedUSCT.github.io}"
export USCT_RWAVE_ROOT="${USCT_RWAVE_ROOT:-$WORKSPACE/external/ray-based-quantitative-ultrasound-tomography}"
export USCT_BENT_RAY_ENTRYPOINT="${USCT_BENT_RAY_ENTRYPOINT:-$REPO_DIR/scripts/matlab_adapters/refraction_corrected_usct_entrypoint.m}"
export USCT_RWAVE_ENTRYPOINT="${USCT_RWAVE_ENTRYPOINT:-$REPO_DIR/scripts/matlab_adapters/rwave_tof_greens_entrypoint.m}"

cd "$REPO_DIR"

if [[ ! -x "$MATLAB_BIN" ]]; then
  echo "Missing executable MATLAB_BIN: $MATLAB_BIN" >&2
  exit 2
fi
if [[ ! -d "$USCT_REFRACTION_CORRECTED_ROOT" ]]; then
  echo "Missing USCT_REFRACTION_CORRECTED_ROOT: $USCT_REFRACTION_CORRECTED_ROOT" >&2
  exit 2
fi
if [[ ! -d "$USCT_RWAVE_ROOT" ]]; then
  echo "Missing USCT_RWAVE_ROOT: $USCT_RWAVE_ROOT" >&2
  exit 2
fi

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli data make-smoke \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_EXTERNAL_MATLAB_SAMPLE_ROOT" \
  --cases-per-density "$USCT_SMOKE_CASES_PER_DENSITY" \
  --converted-shape "$USCT_SMOKE_CONVERTED_SHAPE" \
  --n-transducers "$USCT_SMOKE_N_TRANSDUCERS"

run_root="$(PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli bench \
  --suite configs/benchmarks/archive/external_matlab_adapter_smoke.yaml)"
echo "$run_root"

comparison_dir="$run_root/comparison_artifacts"
mkdir -p "$comparison_dir"
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" scripts/render_class_comparison_panels.py \
  --cases-dir "$USCT_EXTERNAL_MATLAB_SAMPLE_ROOT/cases" \
  --run-dir "$run_root" \
  --out "$comparison_dir/external_matlab_smoke_${USCT_SMOKE_CONVERTED_SHAPE}_gray.png" \
  --field sound_speed \
  --algorithms bent_ray_gn rwave_adapter \
  --title "External MATLAB adapters ${USCT_SMOKE_CONVERTED_SHAPE} smoke comparison" \
  --cmap gray \
  --transpose
