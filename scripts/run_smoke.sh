#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR"

python -m pytest -q

if [ -d "${USCT_SAMPLE_ROOT:-}" ]; then
  echo "USCT_SAMPLE_ROOT=${USCT_SAMPLE_ROOT}"
else
  echo "USCT_SAMPLE_ROOT is not set or does not exist; OpenBreastUS smoke benchmark skipped."
fi
