#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_DIR/.env" ]; then
  # shellcheck disable=SC1091
  . "$REPO_DIR/.env"
fi

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"
WORKSPACE="${USCT_WORKSPACE:-$(cd "$REPO_DIR/.." && pwd)}"
USCT_DATA_ROOT="${USCT_DATA_ROOT:-$WORKSPACE/data/openbreastus}"
USCT_SAMPLE_ROOT="${USCT_SAMPLE_ROOT:-$WORKSPACE/data/openbreastus_smoke_64}"
USCT_RUN_ROOT="${USCT_RUN_ROOT:-$WORKSPACE/runs/usctbench_runs}"
USCT_SMOKE_CASES_PER_DENSITY="${USCT_SMOKE_CASES_PER_DENSITY:-1}"
USCT_SMOKE_CONVERTED_SHAPE="${USCT_SMOKE_CONVERTED_SHAPE:-64}"
USCT_SMOKE_N_TRANSDUCERS="${USCT_SMOKE_N_TRANSDUCERS:-32}"

export USCT_SAMPLE_ROOT
export USCT_RUN_ROOT

cd "$REPO_DIR"

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli data inspect-openbreastus \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_RUN_ROOT/openbreastus_index.json"

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli data make-smoke \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_SAMPLE_ROOT" \
  --cases-per-density "$USCT_SMOKE_CASES_PER_DENSITY" \
  --converted-shape "$USCT_SMOKE_CONVERTED_SHAPE" \
  --n-transducers "$USCT_SMOKE_N_TRANSDUCERS"

run_root="$(PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli bench \
  --suite configs/benchmarks/openbreastus_smoke.yaml)"
echo "$run_root"

comparison_dir="$run_root/comparison_artifacts"
mkdir -p "$comparison_dir"
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" scripts/render_class_comparison_panels.py \
  --cases-dir "$USCT_SAMPLE_ROOT/cases" \
  --run-dir "$run_root" \
  --out "$comparison_dir/openbreastus_smoke_${USCT_SMOKE_CONVERTED_SHAPE}_sound_speed_gray.png" \
  --field sound_speed \
  --algorithms straight_sart bent_ray_gn rwave_adapter \
  --title "OpenBreastUS ${USCT_SMOKE_CONVERTED_SHAPE} sound-speed smoke comparison" \
  --cmap gray \
  --transpose
