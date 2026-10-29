#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
ROOT="${1:-${USCT_EXAMPLE_ROOT:-/tmp/usctbench_examples}}"
DATA_ROOT="$ROOT/synthetic_demo"
RUN_ROOT="$ROOT/runs"

cd "$REPO_DIR"

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli data prepare --dataset synthetic \
  --out "$DATA_ROOT" \
  --shape 32 32 \
  --n-transducers 32

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli run straight_cgls \
  --case "$DATA_ROOT/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/cgls.yaml \
  --out "$RUN_ROOT/cgls"
