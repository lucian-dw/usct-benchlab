#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="${1:-${USCT_EXAMPLE_ROOT:-/tmp/usctbench_examples}}"
DATA_ROOT="$ROOT/synthetic_demo"
RUN_ROOT="$ROOT/runs"

cd "$REPO_DIR"

usct data prepare --dataset synthetic \
  --out "$DATA_ROOT" \
  --shape 32 32 \
  --n-transducers 32 --reference-sound-speed-mps 1500

usct run straight_cgls \
  --case "$DATA_ROOT/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/cgls.yaml \
  --out "$RUN_ROOT/cgls"
