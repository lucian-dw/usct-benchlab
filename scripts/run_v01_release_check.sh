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

INDEX_PATH="${USCT_OPENBREASTUS_INDEX:-$USCT_RUN_ROOT/openbreastus_index.json}"
SMOKE_SUITE="${USCT_SMOKE_BENCHMARK_SUITE:-configs/benchmarks/openbreastus_smoke.yaml}"
SMOKE_MANIFEST="$USCT_SAMPLE_ROOT/openbreastus_smoke_manifest.json"

"$PYTHON_BIN" -m pytest -q
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli data inspect-openbreastus --root "$USCT_DATA_ROOT" --out "$INDEX_PATH"
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli data make-smoke --root "$USCT_DATA_ROOT" --out "$USCT_SAMPLE_ROOT" --cases-per-density 1

run_root="$(PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli bench --suite "$SMOKE_SUITE" | tail -n 1)"
echo "V01_RUN_ROOT=$run_root"
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" scripts/audit_v01_readiness.py \
  --root "$REPO_DIR" \
  --run-dir "$run_root" \
  --openbreastus-index "$INDEX_PATH" \
  --smoke-manifest "$SMOKE_MANIFEST" \
  --require-v01-dod
