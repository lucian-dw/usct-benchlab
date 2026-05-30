#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR"

bash scripts/setup_workspace.sh

python -m pip install -U pip wheel setuptools
python -m pip install -e ".[dev]"

echo "A100 bootstrap complete."
echo "No sudo was used. OpenBreastUS must already exist under:"
echo "  ${USCT_DATA_ROOT:-$HOME/usct-benchlab/data/openbreastus}"

