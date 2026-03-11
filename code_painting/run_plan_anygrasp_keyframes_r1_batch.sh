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

anygrasp_root=${1:-"/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results"}
replay_root=${2:-"/home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue"}
hand_dir=${3:-"/home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis"}
output_root=${4:-"/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes"}
shift 4 || true

echo "[run-anygrasp-plan-batch] anygrasp_root=${anygrasp_root}"
echo "[run-anygrasp-plan-batch] replay_root=${replay_root}"
echo "[run-anygrasp-plan-batch] hand_dir=${hand_dir}"
echo "[run-anygrasp-plan-batch] output_root=${output_root}"

python /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py \
  --anygrasp_root "${anygrasp_root}" \
  --replay_root "${replay_root}" \
  --hand_dir "${hand_dir}" \
  --output_root "${output_root}" \
  "$@"
