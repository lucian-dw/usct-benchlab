#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
  echo "Usage: bash scripts/setup_workspace.sh WORKSPACE" >&2
  exit 2
fi
mkdir -p "$1/data" "$1/runs"
ROOT="$(cd "$1" && pwd)"
printf 'Workspace: %s\nData: %s/data\nRuns: %s/runs\n' "$ROOT" "$ROOT" "$ROOT"
printf 'Next: bash examples/synthetic_quickstart.sh %q\n' "$ROOT"
