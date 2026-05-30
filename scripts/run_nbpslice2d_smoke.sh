#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_DIR/.env" ]; then
  # shellcheck disable=SC1091
  . "$REPO_DIR/.env"
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "$HOME/miniconda3/bin/python" ]; then
    PYTHON_BIN="$HOME/miniconda3/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

WORKSPACE="${USCT_WORKSPACE:-$(cd "$REPO_DIR/.." && pwd)}"
USCT_NBP_ZIP_PATH="${USCT_NBP_ZIP_PATH:-$HOME/USCT_kwave/openbreastus_diffusion/data/openbreastus_speed_crop400/NBPslices2D.zip}"
USCT_NBP_SAMPLE_ROOT="${USCT_NBP_SAMPLE_ROOT:-$WORKSPACE/data/nbpslice2d_sample}"
USCT_NBP_RUN_ROOT="${USCT_NBP_RUN_ROOT:-$WORKSPACE/runs/usctbench_runs}"
USCT_NBP_CASES_PER_TYPE="${USCT_NBP_CASES_PER_TYPE:-1}"
USCT_NBP_CONVERTED_SHAPE="${USCT_NBP_CONVERTED_SHAPE:-64}"
USCT_NBP_N_TRANSDUCERS="${USCT_NBP_N_TRANSDUCERS:-32}"

export USCT_NBP_SAMPLE_ROOT
export USCT_NBP_RUN_ROOT

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >=3.10 is required, got {sys.version.split()[0]}")
PY

cd "$REPO_DIR"

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli data inspect-nbpslice2d \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_NBP_SAMPLE_ROOT/nbpslice2d_index.json"

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli data make-nbp-smoke \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_NBP_SAMPLE_ROOT" \
  --cases-per-type "$USCT_NBP_CASES_PER_TYPE" \
  --converted-shape "$USCT_NBP_CONVERTED_SHAPE" \
  --n-transducers "$USCT_NBP_N_TRANSDUCERS"

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli bench \
  --suite configs/benchmarks/nbpslice2d_smoke.yaml
