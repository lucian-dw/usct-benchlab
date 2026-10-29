#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 4 ]]; then
  echo "Usage: bash examples/fwi_quickstart.sh CASE CONFIG WUST_ROOT OUT" >&2
  exit 2
fi
export WUST_RUNTIME_ROOT="$3"
usct run fwi_wust --case "$1" --config "$2" --out "$4"
