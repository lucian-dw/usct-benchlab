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

bash scripts/setup_workspace.sh

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-}}"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python || command -v python3 || true)"
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "No Python executable found. Activate a Python 3.10+ environment first." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required. Activate a newer conda/env or set PYTHON_BIN.")
PY

"$PYTHON_BIN" -m pip install -U pip wheel setuptools
"$PYTHON_BIN" -m pip install -e ".[dev]"

echo "A100 bootstrap complete."
echo "No sudo was used. OpenBreastUS must already exist under:"
echo "  ${USCT_DATA_ROOT:-$HOME/usct-benchlab/data/openbreastus}"
