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

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-}}"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python || command -v python3 || true)"
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "No Python executable found." >&2
  exit 1
fi

USCT_KWAVE_ROOT="${USCT_KWAVE_ROOT:-$HOME/USCT_kwave}"
USCT_KWAVE_DATASET_PATH="${USCT_KWAVE_DATASET_PATH:-$USCT_KWAVE_ROOT/openbreastus_diffusion/kwave_dps/outputs/full_pipeline/datasets/kWave_openbreast_test001_full128_scale1p000_cuda_binary.mat}"
USCT_KWAVE_FWI_RESULT_PATH="${USCT_KWAVE_FWI_RESULT_PATH:-$USCT_KWAVE_ROOT/openbreastus_diffusion/kwave_dps/outputs/full_pipeline/results/openbreast_test001_full128_scale1p000_cuda_binary_f0300_dxi0p6_iter3.mat}"
USCT_KWAVE_FWI_SAMPLE_ROOT="${USCT_KWAVE_FWI_SAMPLE_ROOT:-$WORKSPACE/data/fwi_kwave_adapter_sample}"
USCT_KWAVE_FWI_RUN_ROOT="${USCT_KWAVE_FWI_RUN_ROOT:-$WORKSPACE/runs/usctbench_runs/fwi_kwave_adapter_smoke}"
export USCT_KWAVE_ROOT
export USCT_KWAVE_FWI_RESULT_PATH

mkdir -p "$USCT_KWAVE_FWI_SAMPLE_ROOT" "$USCT_KWAVE_FWI_RUN_ROOT"

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c "from usctbench.data.conversion import convert_kwave_channel_mat; convert_kwave_channel_mat('$USCT_KWAVE_DATASET_PATH', '$USCT_KWAVE_FWI_SAMPLE_ROOT', case_id_prefix='openbreast_test001_fwi', output_shape=(64, 64), n_transducers=32)"

case_path="$USCT_KWAVE_FWI_SAMPLE_ROOT/openbreast_test001_fwi.h5"
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli run \
  fwi_kwave_adapter \
  --case "$case_path" \
  --config configs/algorithms/fwi_kwave_adapter.yaml \
  --out "$USCT_KWAVE_FWI_RUN_ROOT"

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli eval \
  --run "$USCT_KWAVE_FWI_RUN_ROOT" \
  --protocol configs/benchmarks/fwi_kwave_adapter_smoke.yaml

echo "FWI_KWAVE_ADAPTER_RUN_ROOT=$USCT_KWAVE_FWI_RUN_ROOT"
