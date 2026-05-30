#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${USCT_WORKSPACE:-$HOME/usct-benchlab}"
REPO_DIR="$WORKSPACE/code"

mkdir -p "$WORKSPACE/data/openbreastus"
mkdir -p "$WORKSPACE/data/openbreastus_sample"
mkdir -p "$WORKSPACE/runs/usctbench_runs"
mkdir -p "$WORKSPACE/external"
mkdir -p "$WORKSPACE/checkpoints"

cd "$REPO_DIR"

mkdir -p data/raw data/processed data/interim runs external checkpoints

ln -sfn ../../../data/openbreastus data/raw/openbreastus
ln -sfn ../../../data/openbreastus_sample data/processed/openbreastus_sample
ln -sfn ../../runs/usctbench_runs runs/current
ln -sfn ../../external external/local_external
ln -sfn ../../checkpoints checkpoints/local_checkpoints

echo "Workspace prepared at: $WORKSPACE"
