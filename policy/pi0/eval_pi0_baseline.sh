#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

STEP_OR_LATEST="${1:-${STEP_OR_LATEST:-latest}}"
SEED="${2:-${SEED:-0}}"
GPU_ID="${3:-${GPU_ID:-0}}"

TRAIN_CONFIG_NAME="${TRAIN_CONFIG_NAME:-pi0_base_aloha_robotwin_lora}"
MODEL_NAME="${MODEL_NAME:-pi0_baseline_run}"
TASK_NAME="${TASK_NAME:-beat_block_hammer}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
PI0_STEP="${PI0_STEP:-50}"

CHECKPOINT_ROOT="${SCRIPT_DIR}/checkpoints/${TRAIN_CONFIG_NAME}/${MODEL_NAME}"

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "checkpoint root not found: ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

if [[ "${STEP_OR_LATEST}" == "latest" ]]; then
  CHECKPOINT_ID="$(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | rg '^[0-9]+$' | sort -n | tail -1)"
else
  CHECKPOINT_ID="${STEP_OR_LATEST}"
fi

if [[ -z "${CHECKPOINT_ID}" ]]; then
  echo "no checkpoint step found under: ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

printf 'Evaluating pi0 baseline\n'
printf '  GPU_ID=%s\n' "${GPU_ID}"
printf '  TRAIN_CONFIG_NAME=%s\n' "${TRAIN_CONFIG_NAME}"
printf '  MODEL_NAME=%s\n' "${MODEL_NAME}"
printf '  CHECKPOINT_ID=%s\n' "${CHECKPOINT_ID}"

bash "${SCRIPT_DIR}/eval.sh" \
  "${TASK_NAME}" \
  "${TASK_CONFIG}" \
  "${TRAIN_CONFIG_NAME}" \
  "${MODEL_NAME}" \
  "${SEED}" \
  "${GPU_ID}" \
  "${CHECKPOINT_ID}" \
  "${PI0_STEP}"
