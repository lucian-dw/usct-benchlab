#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_DIR/.env" ]; then
  # shellcheck disable=SC1091
  . "$REPO_DIR/.env"
fi

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"
WORKSPACE="${USCT_WORKSPACE:-$(cd "$REPO_DIR/.." && pwd)}"
USCT_DATA_ROOT="${USCT_DATA_ROOT:-$WORKSPACE/data/openbreastus}"
USCT_QUALITY_SAMPLE_ROOT="${USCT_QUALITY_SAMPLE_ROOT:-$WORKSPACE/data/openbreastus_quality_256}"
USCT_QUALITY_RUN_ROOT="${USCT_QUALITY_RUN_ROOT:-$WORKSPACE/runs/usctbench_runs}"
USCT_QUALITY_CONVERTED_SHAPE="${USCT_QUALITY_CONVERTED_SHAPE:-256}"
USCT_QUALITY_N_TRANSDUCERS="${USCT_QUALITY_N_TRANSDUCERS:-128}"
USCT_QUALITY_CASES_PER_DENSITY="${USCT_QUALITY_CASES_PER_DENSITY:-1}"

export USCT_QUALITY_SAMPLE_ROOT
export USCT_QUALITY_RUN_ROOT

cd "$REPO_DIR"

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli data make-quality \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_QUALITY_SAMPLE_ROOT" \
  --cases-per-density "$USCT_QUALITY_CASES_PER_DENSITY" \
  --converted-shape "$USCT_QUALITY_CONVERTED_SHAPE" \
  --n-transducers "$USCT_QUALITY_N_TRANSDUCERS"

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli bench \
  --suite configs/benchmarks/openbreastus_quality.yaml
