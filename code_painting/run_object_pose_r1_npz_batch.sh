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

input_root=${1:-"/home/zaijia001/ssd/data/R1/hand/object_pose_bottle"}
mesh_file=${2:-"/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle.obj"}
output_root=${3:-"/home/zaijia001/ssd/RoboTwin/code_painting/output_object_pose_bottle_batch"}
fps=${4:-"5"}
shift 4 || true

echo "[run-batch] input_root=${input_root}"
echo "[run-batch] mesh_file=${mesh_file}"
echo "[run-batch] output_root=${output_root}"
echo "[run-batch] fps=${fps}"

python /home/zaijia001/ssd/RoboTwin/code_painting/render_object_pose_r1_npz_batch.py \
  --input_root "${input_root}" \
  --mesh_file "${mesh_file}" \
  --output_root "${output_root}" \
  --fps "${fps}" \
  --head_only 1 \
  --overlay_text 0 \
  --third_person_view 0 \
  "$@"
