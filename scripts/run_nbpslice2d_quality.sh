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
USCT_NBP_ZIP_PATH="${USCT_NBP_ZIP_PATH:-$HOME/USCT_kwave/openbreastus_diffusion/data/openbreastus_speed_crop400/NBPslices2D.zip}"
USCT_NBP_QUALITY_SAMPLE_ROOT="${USCT_NBP_QUALITY_SAMPLE_ROOT:-$WORKSPACE/data/nbpslice2d_quality_256}"
USCT_NBP_QUALITY_RUN_ROOT="${USCT_NBP_QUALITY_RUN_ROOT:-$WORKSPACE/runs/usctbench_runs}"
USCT_NBP_QUALITY_CASES_PER_TYPE="${USCT_NBP_QUALITY_CASES_PER_TYPE:-1}"
USCT_NBP_QUALITY_CONVERTED_SHAPE="${USCT_NBP_QUALITY_CONVERTED_SHAPE:-256}"
USCT_NBP_QUALITY_N_TRANSDUCERS="${USCT_NBP_QUALITY_N_TRANSDUCERS:-128}"

export USCT_NBP_QUALITY_SAMPLE_ROOT
export USCT_NBP_QUALITY_RUN_ROOT

cd "$REPO_DIR"

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli data make-nbp-quality \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_NBP_QUALITY_SAMPLE_ROOT" \
  --cases-per-type "$USCT_NBP_QUALITY_CASES_PER_TYPE" \
  --converted-shape "$USCT_NBP_QUALITY_CONVERTED_SHAPE" \
  --n-transducers "$USCT_NBP_QUALITY_N_TRANSDUCERS"

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli bench \
  --suite configs/benchmarks/nbpslice2d_quality.yaml
