#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${1:-}"
SEED="${2:-0}"
GPU_ID="${3:-3}"
TASK_CONFIG="${4:-demo_clean}"
UNNORM_KEY="${UNNORM_KEY:-aloha_beat_block_hammer_builder}"
TASK_NAME="beat_block_hammer"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/home/zaijia001/ssd/RoboTwin/data/beat_block_hammer/runs_openvla_v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_MERGE_LORA_CHECKPOINT="${AUTO_MERGE_LORA_CHECKPOINT:-1}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$(find "${RUN_ROOT_DIR}" -maxdepth 1 -mindepth 1 -type d -name '*chkpt' | sort -V | tail -n 1)"
fi

if [[ -z "${CHECKPOINT_PATH}" || ! -d "${CHECKPOINT_PATH}" ]]; then
  echo "checkpoint path not found"
  exit 1
fi

if [[ "${AUTO_MERGE_LORA_CHECKPOINT}" == "1" && ! -f "${CHECKPOINT_PATH}/config.json" && -d "${CHECKPOINT_PATH}/lora_adapter" ]]; then
  echo "detected unmerged LoRA checkpoint: ${CHECKPOINT_PATH}"
  GPU_ID="${GPU_ID}" bash "${SCRIPT_DIR}/merge_lora_beat_block_hammer_v1.sh" "${CHECKPOINT_PATH}"
fi

if [[ ! -f "${CHECKPOINT_PATH}/config.json" ]]; then
  echo "config.json still missing after checkpoint preparation: ${CHECKPOINT_PATH}"
  echo "expected either a merged checkpoint directory or a LoRA checkpoint with lora_adapter/"
  exit 1
fi

unset LD_LIBRARY_PATH
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
set +u
conda activate RoboTwin_openvla
set -u

cd "${SCRIPT_DIR}"
bash eval.sh "${TASK_NAME}" "${TASK_CONFIG}" "${CHECKPOINT_PATH}" "${SEED}" "${GPU_ID}" "${UNNORM_KEY}"
