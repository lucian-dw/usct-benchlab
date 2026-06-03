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
  USCT_WORKSPACE="${USCT_WORKSPACE:-$(cd "$REPO_DIR/.." && pwd)}"
else
  USCT_WORKSPACE="${USCT_WORKSPACE:-$REPO_DIR}"
fi

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-}}"
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python || command -v python3 || true)"
fi
if [ -z "$PYTHON_BIN" ]; then
  echo "No Python executable found." >&2
  exit 1
fi

export USCT_RUN_ROOT="${USCT_RUN_ROOT:-$USCT_WORKSPACE/runs/usctbench_runs}"
export USCT_KWAVE_ROOT="${USCT_KWAVE_ROOT:-$HOME/USCT_kwave}"
export USCT_KWAVE_PYTHON_BIN="${USCT_KWAVE_PYTHON_BIN:-$HOME/miniconda3/envs/usct-kwave/bin/python}"
export USCT_KWAVE_SOURCE_MAT="${USCT_KWAVE_SOURCE_MAT:-$USCT_KWAVE_ROOT/openbreastus_diffusion/data/openbreastus_speed_crop400/breast_test_speed.mat}"
export USCT_KWAVE_MAT_KEY="${USCT_KWAVE_MAT_KEY:-breast_test}"
export USCT_KWAVE_SAMPLE_INDEX="${USCT_KWAVE_SAMPLE_INDEX:-201}"
export USCT_KWAVE_FWI_SAMPLE_ROOT="${USCT_KWAVE_FWI_SAMPLE_ROOT:-$USCT_WORKSPACE/data/openbreastus_kwave_fwi_smoke}"
export USCT_KWAVE_RUN_ID="${USCT_KWAVE_RUN_ID:-fwi_kwave_bulk_support_pure201}"
export USCT_KWAVE_WRAPPER_CONVERTED_SHAPE="${USCT_KWAVE_WRAPPER_CONVERTED_SHAPE:-256}"
export USCT_KWAVE_EXECUTION_MODE="${USCT_KWAVE_EXECUTION_MODE:-invert_existing_dataset}"

sample_zero_index=$((USCT_KWAVE_SAMPLE_INDEX - 1))
case_prefix="$(printf 'openbreast_test%03d_kwave_bulk_support' "$USCT_KWAVE_SAMPLE_INDEX")"
run_root="$USCT_RUN_ROOT/$USCT_KWAVE_RUN_ID"
external_root="$run_root/external_kwave"
export USCT_KWAVE_OUTPUT_DIR="${USCT_KWAVE_OUTPUT_DIR:-$external_root/full_pipeline}"
export USCT_KWAVE_SIMINFO_PATH="${USCT_KWAVE_SIMINFO_PATH:-$external_root/sim_info/SimInfo_${case_prefix}.mat}"
export USCT_KWAVE_SCRATCH_DIR="${USCT_KWAVE_SCRATCH_DIR:-$external_root/scratch/${case_prefix}}"
default_preproc_dataset="$USCT_KWAVE_ROOT/openbreastus_diffusion/kwave_dps/outputs/dps_preproc_test$(printf '%03d' "$USCT_KWAVE_SAMPLE_INDEX")_latestckpt/datasets/kWave_openbreast_test$(printf '%03d' "$USCT_KWAVE_SAMPLE_INDEX")_full128_scale1p000_preproc_crop300_tla250_ab0_s2a0_cuda_binary.mat"
case "$USCT_KWAVE_SAMPLE_INDEX" in
  1|001)
    default_preproc_dataset="$USCT_KWAVE_ROOT/openbreastus_diffusion/kwave_dps/outputs/dps_preproc_test001_step127650_notebook30/datasets/kWave_openbreast_test001_full128_scale1p000_preproc_crop300_tla250_ab0_s2a0_cuda_binary.mat"
    ;;
  201)
    default_preproc_dataset="$USCT_KWAVE_ROOT/openbreastus_diffusion/kwave_dps/outputs/dps_preproc_test201_latestckpt/datasets/kWave_openbreast_test201_full128_scale1p000_preproc_crop300_tla250_ab0_s2a0_cuda_binary.mat"
    ;;
  501)
    default_preproc_dataset="$USCT_KWAVE_ROOT/openbreastus_diffusion/kwave_dps/outputs/dps_preproc_test501_step127650_notebook30/datasets/kWave_openbreast_test501_full128_scale1p000_preproc_crop300_tla250_ab0_s2a0_cuda_binary.mat"
    ;;
  701)
    default_preproc_dataset="$USCT_KWAVE_ROOT/openbreastus_diffusion/kwave_dps/outputs/dps_preproc_test701_step127650_notebook30/datasets/kWave_openbreast_test701_full128_scale1p000_preproc_crop300_tla250_ab0_s2a0_cuda_binary.mat"
    ;;
esac
export USCT_KWAVE_DATASET_PATH="${USCT_KWAVE_DATASET_PATH:-$default_preproc_dataset}"
export USCT_KWAVE_FWI_RESULT_PATH="${USCT_KWAVE_FWI_RESULT_PATH:-$external_root/results/${case_prefix}_WaveformInversionResults.mat}"
export USCT_KWAVE_EXTERNAL_LOG_PATH="${USCT_KWAVE_EXTERNAL_LOG_PATH:-$external_root/run.log}"
export USCT_KWAVE_WARM_START_PATH="${USCT_KWAVE_WARM_START_PATH:-$external_root/warm_start/${case_prefix}_bulk_support_init.mat}"
export USCT_KWAVE_WARM_START_SUMMARY_PATH="${USCT_KWAVE_WARM_START_SUMMARY_PATH:-$external_root/warm_start/${case_prefix}_bulk_support_init.json}"
export USCT_KWAVE_WARM_START_DIAGNOSTIC_PREFIX="${USCT_KWAVE_WARM_START_DIAGNOSTIC_PREFIX:-$external_root/warm_start/${case_prefix}_bulk_support}"
export USCT_KWAVE_RECONSTRUCTION_ITERATION="${USCT_KWAVE_RECONSTRUCTION_ITERATION:-best}"

if [[ "$USCT_KWAVE_EXECUTION_MODE" != "invert_existing_dataset" && ! -f "$USCT_KWAVE_SOURCE_MAT" ]]; then
  echo "Missing USCT_KWAVE_SOURCE_MAT: $USCT_KWAVE_SOURCE_MAT" >&2
  exit 2
fi
if [[ ! -f "$USCT_KWAVE_DATASET_PATH" ]]; then
  echo "Missing USCT_KWAVE_DATASET_PATH: $USCT_KWAVE_DATASET_PATH" >&2
  echo "Set USCT_KWAVE_DATASET_PATH to an existing preproc_crop300_tla250 k-Wave MAT, or set USCT_KWAVE_EXECUTION_MODE=full_pipeline_from_speed_map to generate one." >&2
  exit 2
fi
if [[ ! -x "$USCT_KWAVE_PYTHON_BIN" ]]; then
  echo "Missing executable USCT_KWAVE_PYTHON_BIN: $USCT_KWAVE_PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$USCT_KWAVE_FWI_SAMPLE_ROOT" "$external_root"

# This HDF5 case is only the usct-benchlab benchmark wrapper used for metadata,
# metrics, and artifact layout. The authoritative RF data used by FWI comes
# from the external preproc_crop300_tla250 k-Wave MAT at USCT_KWAVE_DATASET_PATH.
case_path="$(
  PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -c "from usctbench.data.conversion import convert_kwave_channel_mat; records = convert_kwave_channel_mat('$USCT_KWAVE_DATASET_PATH', '$USCT_KWAVE_FWI_SAMPLE_ROOT', case_id_prefix='$case_prefix', output_shape=($USCT_KWAVE_WRAPPER_CONVERTED_SHAPE, $USCT_KWAVE_WRAPPER_CONVERTED_SHAPE), n_transducers=128); print(records[0]['path'])"
)"

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli run \
  fwi_kwave_adapter \
  --case "$case_path" \
  --config "$REPO_DIR/configs/algorithms/fwi_kwave_full_pipeline.yaml" \
  --out "$run_root/fwi_kwave_adapter"

case_id="$(basename "$case_path" .h5)"
case_run_dir="$run_root/fwi_kwave_adapter/$case_id"
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$REPO_DIR/scripts/render_kwave_fwi_smoke_outputs.py" \
  --case "$case_path" \
  --result "$USCT_KWAVE_FWI_RESULT_PATH" \
  --out "$case_run_dir/kwave_smoke_outputs" \
  --log "$USCT_KWAVE_EXTERNAL_LOG_PATH" \
  --iteration "$USCT_KWAVE_RECONSTRUCTION_ITERATION" \
  --render-best-and-final

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli eval \
  --run "$run_root" \
  --protocol "$REPO_DIR/configs/benchmarks/fwi_kwave_full_pipeline_smoke.yaml"

echo "$case_run_dir/kwave_smoke_outputs"
