#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-${1:-1}}"
SESSION_NAME="${SESSION_NAME:-openvla_beat_block_hammer_v1_gpu${GPU_ID}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/finetune_beat_block_hammer_v1.sh"
LOG_DIR="${LOG_DIR:-/home/zaijia001/ssd/RoboTwin/data/beat_block_hammer/tmux_logs}"

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${SESSION_NAME}.log"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" "bash ${TRAIN_SCRIPT} ${GPU_ID} |& tee ${LOG_PATH}"

echo "started tmux session: ${SESSION_NAME}"
echo "log file: ${LOG_PATH}"
echo "attach with: tmux attach -t ${SESSION_NAME}"
