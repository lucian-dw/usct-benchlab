#!/usr/bin/env bash
set -euo pipefail

# Expected layout:
# <workspace>/
#   code/ or repository root
#   data/
#   runs/
#   external/
#   checkpoints/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$REPO_DIR/.env"
  set +a
fi

# If repo is <workspace>/code, workspace is parent.
# If repo itself is named usct-benchlab, workspace is repo.
if [ "$(basename "$REPO_DIR")" = "code" ]; then
  WORKSPACE="${USCT_WORKSPACE:-$(cd "$REPO_DIR/.." && pwd)}"
else
  WORKSPACE="${USCT_WORKSPACE:-$REPO_DIR}"
fi

mkdir -p "$WORKSPACE/data/openbreastus"
mkdir -p "$WORKSPACE/data/openbreastus_sample"
mkdir -p "$WORKSPACE/runs/usctbench_runs"
mkdir -p "$WORKSPACE/external"
mkdir -p "$WORKSPACE/checkpoints"

cd "$REPO_DIR"

mkdir -p data/raw data/processed data/interim runs external checkpoints

# Symlinks are created relative to the repository directory.
if [ "$(basename "$REPO_DIR")" = "code" ]; then
  ln -sfn ../../../data/openbreastus data/raw/openbreastus
  ln -sfn ../../../data/openbreastus_sample data/processed/openbreastus_sample
  ln -sfn ../../runs/usctbench_runs runs/current
  ln -sfn ../../external external/local_external
  ln -sfn ../../checkpoints checkpoints/local_checkpoints
else
  ln -sfn ../openbreastus data/raw/openbreastus
  ln -sfn ../openbreastus_sample data/processed/openbreastus_sample
  ln -sfn usctbench_runs runs/current
  ln -sfn . external/local_external
  ln -sfn . checkpoints/local_checkpoints
fi

echo "Workspace prepared."
echo "REPO_DIR=$REPO_DIR"
echo "WORKSPACE=$WORKSPACE"
echo "USCT_DATA_ROOT=$WORKSPACE/data/openbreastus"
echo "USCT_SAMPLE_ROOT=$WORKSPACE/data/openbreastus_sample"
echo "USCT_RUN_ROOT=$WORKSPACE/runs/usctbench_runs"
