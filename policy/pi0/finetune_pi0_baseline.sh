#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GPU_ID="${1:-${GPU_ID:-0}}"
CONFIG_NAME="${CONFIG_NAME:-pi0_base_aloha_robotwin_lora}"
EXP_NAME="${EXP_NAME:-pi0_baseline_run}"
DATA_REPO_ID="${DATA_REPO_ID:-}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
BATCH_SIZE="${BATCH_SIZE:-}"
FSDP_DEVICES="${FSDP_DEVICES:-}"
OVERWRITE="${OVERWRITE:-True}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"

cmd=(uv run scripts/train.py "${CONFIG_NAME}" "--exp-name=${EXP_NAME}")

if [[ "${OVERWRITE}" == "True" ]]; then
  cmd+=("--overwrite")
fi
if [[ -n "${DATA_REPO_ID}" ]]; then
  cmd+=("--data.repo-id=${DATA_REPO_ID}")
fi
if [[ -n "${NUM_TRAIN_STEPS}" ]]; then
  cmd+=("--num-train-steps=${NUM_TRAIN_STEPS}")
fi
if [[ -n "${SAVE_INTERVAL}" ]]; then
  cmd+=("--save-interval=${SAVE_INTERVAL}")
fi
if [[ -n "${KEEP_PERIOD}" ]]; then
  cmd+=("--keep-period=${KEEP_PERIOD}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  cmd+=("--batch-size=${BATCH_SIZE}")
fi
if [[ -n "${FSDP_DEVICES}" ]]; then
  cmd+=("--fsdp-devices=${FSDP_DEVICES}")
fi

printf 'Launching pi0 baseline training\n'
printf '  GPU_ID=%s\n' "${GPU_ID}"
printf '  CONFIG_NAME=%s\n' "${CONFIG_NAME}"
printf '  EXP_NAME=%s\n' "${EXP_NAME}"
if [[ -n "${DATA_REPO_ID}" ]]; then
  printf '  DATA_REPO_ID=%s\n' "${DATA_REPO_ID}"
fi

"${cmd[@]}"
