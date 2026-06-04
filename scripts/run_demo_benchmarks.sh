#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_demo_benchmarks.sh

Runs available demo benchmark suites. By default, a small synthetic demo subset
is created under /tmp when no synthetic cases are present. NBPslice2D and
OpenBreastUS demos run only when their case globs match existing .h5 files.

Environment:
  USCT_SYNTHETIC_ROOT       Synthetic demo root, default /tmp/usctbench_synthetic_demo
  USCT_SYNTHETIC_CASE_GLOB Synthetic case glob
  USCT_NBP_CASE_GLOB       NBPslice2D converted case glob
  USCT_OPENBREASTUS_CASE_GLOB OpenBreastUS converted case glob
  USCT_RUN_ROOT            Benchmark output root
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"

cd "$REPO_DIR"

if [ "${RUN_SYNTHETIC_DEMO:-1}" = "1" ]; then
  SYNTHETIC_ROOT="${USCT_SYNTHETIC_ROOT:-/tmp/usctbench_synthetic_demo}"
  SYNTHETIC_GLOB="${USCT_SYNTHETIC_CASE_GLOB:-$SYNTHETIC_ROOT/cases/*.h5}"
  if ! compgen -G "$SYNTHETIC_GLOB" >/dev/null; then
    PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli data make-synthetic-smoke \
      --out "$SYNTHETIC_ROOT" \
      --shape 32 \
      --n-transducers 32
  fi
  USCT_SYNTHETIC_CASE_GLOB="$SYNTHETIC_GLOB" PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli bench \
    --suite configs/benchmarks/synthetic_demo.yaml
fi

if compgen -G "${USCT_NBP_CASE_GLOB:-data/nbpslice2d_demo/cases/*.h5}" >/dev/null; then
  PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli bench \
    --suite configs/benchmarks/nbpslice2d_demo.yaml
else
  echo "Skipping NBPslice2D demo: no cases matched ${USCT_NBP_CASE_GLOB:-data/nbpslice2d_demo/cases/*.h5}"
fi

if compgen -G "${USCT_OPENBREASTUS_CASE_GLOB:-data/openbreastus_demo/cases/*.h5}" >/dev/null; then
  PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli bench \
    --suite configs/benchmarks/openbreastus_demo.yaml
else
  echo "Skipping OpenBreastUS demo: no cases matched ${USCT_OPENBREASTUS_CASE_GLOB:-data/openbreastus_demo/cases/*.h5}"
fi
