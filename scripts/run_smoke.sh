#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_DIR/.env"
  set +a
fi

cd "$REPO_DIR"

if [ "$(basename "$REPO_DIR")" = "code" ]; then
  WORKSPACE="${USCT_WORKSPACE:-$(cd "$REPO_DIR/.." && pwd)}"
else
  WORKSPACE="${USCT_WORKSPACE:-$REPO_DIR}"
fi

USCT_DATA_ROOT="${USCT_DATA_ROOT:-$WORKSPACE/data/openbreastus}"
USCT_SAMPLE_ROOT="${USCT_SAMPLE_ROOT:-$WORKSPACE/data/openbreastus_sample}"
USCT_RUN_ROOT="${USCT_RUN_ROOT:-$WORKSPACE/runs/usctbench_runs}"
export USCT_WORKSPACE="$WORKSPACE"
export USCT_DATA_ROOT
export USCT_SAMPLE_ROOT
export USCT_RUN_ROOT

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-}}"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python || command -v python3 || true)"
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "No Python executable found." >&2
  exit 1
fi

"$PYTHON_BIN" -m pytest -q

SMOKE_SUITE="${USCT_SMOKE_BENCHMARK_SUITE:-configs/benchmarks/openbreastus_smoke.yaml}"
if [ -d "$USCT_SAMPLE_ROOT/cases" ] && find "$USCT_SAMPLE_ROOT/cases" -maxdepth 1 -name '*.h5' -type f | grep -q .; then
  echo "USCT_SAMPLE_ROOT=$USCT_SAMPLE_ROOT"
  echo "USCT_RUN_ROOT=$USCT_RUN_ROOT"
  echo "Running OpenBreastUS smoke benchmark: $SMOKE_SUITE"
  run_root="$(PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli bench --suite "$SMOKE_SUITE" | tail -n 1)"
  echo "SMOKE_RUN_ROOT=$run_root"
  PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" scripts/audit_v01_readiness.py --root "$REPO_DIR" --run-dir "$run_root"
elif [ "${USCT_REQUIRE_SMOKE_CASES:-0}" = "1" ]; then
  echo "No HDF5 smoke cases found under $USCT_SAMPLE_ROOT/cases." >&2
  echo "Set USCT_DATA_ROOT and run: usct data make-smoke --root \"\$USCT_DATA_ROOT\" --out \"\$USCT_SAMPLE_ROOT\"" >&2
  exit 1
else
  echo "No HDF5 smoke cases found under $USCT_SAMPLE_ROOT/cases; OpenBreastUS smoke benchmark skipped."
fi
