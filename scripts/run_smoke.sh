#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-}}"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python || command -v python3 || true)"
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "No Python executable found." >&2
  exit 1
fi

"$PYTHON_BIN" -m pytest -q

if [ -d "${USCT_SAMPLE_ROOT:-}" ]; then
  echo "USCT_SAMPLE_ROOT=${USCT_SAMPLE_ROOT}"
else
  echo "USCT_SAMPLE_ROOT is not set or does not exist; OpenBreastUS smoke benchmark skipped."
fi
