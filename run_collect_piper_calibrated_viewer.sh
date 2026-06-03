#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

unset CUDA_VISIBLE_DEVICES
export SAPIEN_RT_DENOISER="${SAPIEN_RT_DENOISER:-none}"
if [[ -f /etc/vulkan/icd.d/nvidia_icd.json && -z "${VK_ICD_FILENAMES:-}" ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

python script/collect_data.py pick_diverse_bottles_piper demo_clean_piper_calibrated_viewer
