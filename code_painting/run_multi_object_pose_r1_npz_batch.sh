#!/usr/bin/env bash
set -euo pipefail

cd /home/zaijia001/ssd/RoboTwin

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
if [[ "${CONDA_DEFAULT_ENV:-}" != "RoboTwin_bw" ]]; then
  set +u
  conda activate RoboTwin_bw
  set -u
fi

if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

input_root=${1:-"/home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue_multi/obj_vis"}
output_root=${2:-"/home/zaijia001/ssd/RoboTwin/code_painting/output_multi_object_pose_d_pour_blue"}
fps=${3:-"5"}
shift 3 || true

echo "[run-multi-batch] input_root=${input_root}"
echo "[run-multi-batch] output_root=${output_root}"
echo "[run-multi-batch] fps=${fps}"

python /home/zaijia001/ssd/RoboTwin/code_painting/render_multi_object_pose_r1_npz_batch.py \
  --input_root "${input_root}" \
  --output_root "${output_root}" \
  --fps "${fps}" \
  --head_only 1 \
  --overlay_text 0 \
  --third_person_view 0 \
  "$@"
