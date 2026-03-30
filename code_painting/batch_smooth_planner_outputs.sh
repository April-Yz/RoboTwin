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

TASK_NAME="${TASK_NAME:-d_pour_blue}"
INPUT_ROOT="${INPUT_ROOT:-/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth}"
INTERP_FACTOR="${INTERP_FACTOR:-2}"
FPS="${FPS:-10}"
KEEP_HOVER_FRAMES_EVERY="${KEEP_HOVER_FRAMES_EVERY:-3}"
DEDUP_POS_THRESH_M="${DEDUP_POS_THRESH_M:-0.002}"
DEDUP_ROT_THRESH_DEG="${DEDUP_ROT_THRESH_DEG:-1.5}"
DEDUP_JOINT_THRESH_RAD="${DEDUP_JOINT_THRESH_RAD:-0.01}"
DEDUP_GRIPPER_THRESH="${DEDUP_GRIPPER_THRESH:-0.01}"
OVERLAY_TEXT="${OVERLAY_TEXT:-0}"
DISABLE_TABLE="${DISABLE_TABLE:-1}"
BASE_OCCLUDER_ENABLE="${BASE_OCCLUDER_ENABLE:-0}"
LIGHTING_MODE="${LIGHTING_MODE:-front_no_shadow}"

if [ "$#" -gt 0 ]; then
  IDS=("$@")
else
  mapfile -t IDS < <(seq 0 60)
fi

mkdir -p "${OUTPUT_ROOT}"

for idx in "${IDS[@]}"; do
  input_dir="${INPUT_ROOT}/${TASK_NAME}_${idx}"
  output_dir="${OUTPUT_ROOT}/${TASK_NAME}_${idx}"
  if [ ! -f "${input_dir}/pose_debug.jsonl" ]; then
    echo "[skip] missing pose_debug.jsonl: ${input_dir}"
    continue
  fi

  python /home/zaijia001/ssd/RoboTwin/code_painting/smooth_planner_outputs_from_pose_debug.py \
    --input_dir "${input_dir}" \
    --output_dir "${output_dir}" \
    --interp_factor "${INTERP_FACTOR}" \
    --fps "${FPS}" \
    --keep_hover_frames_every "${KEEP_HOVER_FRAMES_EVERY}" \
    --dedup_pos_thresh_m "${DEDUP_POS_THRESH_M}" \
    --dedup_rot_thresh_deg "${DEDUP_ROT_THRESH_DEG}" \
    --dedup_joint_thresh_rad "${DEDUP_JOINT_THRESH_RAD}" \
    --dedup_gripper_thresh "${DEDUP_GRIPPER_THRESH}" \
    --overlay_text "${OVERLAY_TEXT}" \
    --disable_table "${DISABLE_TABLE}" \
    --base_occluder_enable "${BASE_OCCLUDER_ENABLE}" \
    --lighting_mode "${LIGHTING_MODE}"
done
