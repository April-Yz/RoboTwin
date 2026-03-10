#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-${1:-1}}"
SESSION_NAME="${SESSION_NAME:-openvla_beat_block_hammer_v1_gpu${GPU_ID}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/finetune_beat_block_hammer_v1.sh"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/tmux_logs}"
TMUX_BIN="${TMUX_BIN:-/usr/bin/tmux}"

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${SESSION_NAME}.log"

if ! command -v "${TMUX_BIN}" >/dev/null 2>&1 && [[ ! -x "${TMUX_BIN}" ]]; then
  echo "tmux binary not found: ${TMUX_BIN}"
  exit 1
fi

if env -u LD_LIBRARY_PATH "${TMUX_BIN}" has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
fi

env -u LD_LIBRARY_PATH "${TMUX_BIN}" new-session -d -s "${SESSION_NAME}" "bash ${TRAIN_SCRIPT} ${GPU_ID} |& tee ${LOG_PATH}"

echo "started tmux session: ${SESSION_NAME}"
echo "log file: ${LOG_PATH}"
echo "attach with: tmux attach -t ${SESSION_NAME}"
