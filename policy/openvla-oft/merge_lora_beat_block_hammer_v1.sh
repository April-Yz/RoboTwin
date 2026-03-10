#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${1:-}"
GPU_ID="${GPU_ID:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${SCRIPT_DIR}/runs/beat_block_hammer_v1}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-openvla/openvla-7b}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  CHECKPOINT_DIR="$(find "${RUN_ROOT_DIR}" -maxdepth 1 -mindepth 1 -type d -name '*chkpt' | sort -V | tail -n 1)"
fi

if [[ -z "${CHECKPOINT_DIR}" || ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "checkpoint directory not found"
  exit 1
fi

unset LD_LIBRARY_PATH
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
set +u
conda activate RoboTwin_openvla
set -u

cd "${SCRIPT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python vla-scripts/merge_lora_weights_and_save.py \
  --base_checkpoint "${BASE_CHECKPOINT}" \
  --lora_finetuned_checkpoint_dir "${CHECKPOINT_DIR}"

echo "merged checkpoint: ${CHECKPOINT_DIR}"
