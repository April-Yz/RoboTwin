#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

unset CUDA_VISIBLE_DEVICES
export SAPIEN_RT_DENOISER="${SAPIEN_RT_DENOISER:-none}"
if [[ -f /etc/vulkan/icd.d/nvidia_icd.json && -z "${VK_ICD_FILENAMES:-}" ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

python view_pick_diverse_bottles_piper_scene.py "$@"
