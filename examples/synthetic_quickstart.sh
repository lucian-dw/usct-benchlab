#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
ROOT="${USCT_EXAMPLE_ROOT:-/tmp/usctbench_examples}"
DATA_ROOT="$ROOT/synthetic_demo"
RUN_ROOT="$ROOT/runs"

cd "$REPO_DIR"

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli data make-synthetic-smoke \
  --out "$DATA_ROOT" \
  --shape 32 \
  --n-transducers 32

USCT_SYNTHETIC_CASE_GLOB="$DATA_ROOT/cases/*.h5" \
USCT_RUN_ROOT="$RUN_ROOT" \
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli bench \
  --suite configs/benchmarks/synthetic_demo.yaml
