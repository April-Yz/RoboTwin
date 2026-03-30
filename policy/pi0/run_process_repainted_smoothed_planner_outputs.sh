#!/usr/bin/env bash
set -euo pipefail

cd /home/zaijia001/ssd/RoboTwin/policy/pi0

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
if [[ "${CONDA_DEFAULT_ENV:-}" != "RoboTwin_bw" ]]; then
  set +u
  conda activate RoboTwin_bw
  set -u
fi

TASK_NAME="${TASK_NAME:-d_pour_blue}"
INSTRUCTION="${INSTRUCTION:-pour water}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-27}"
HEAD_ROOT="${HEAD_ROOT:-/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_smooth}"
HEAD_DIR_TEMPLATE="${HEAD_DIR_TEMPLATE:-id_{id}_head_cam_arm_gripper_cup_bottle_pad_target}"
HEAD_VIDEO_NAME="${HEAD_VIDEO_NAME:-target_with_original_head_cam_plan.mp4}"
PLANNER_ROOT="${PLANNER_ROOT:-/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth}"
PLANNER_DIR_TEMPLATE="${PLANNER_DIR_TEMPLATE:-d_pour_blue_{id}}"
LEFT_WRIST_VIDEO_NAME="${LEFT_WRIST_VIDEO_NAME:-left_wrist_cam_plan.mp4}"
RIGHT_WRIST_VIDEO_NAME="${RIGHT_WRIST_VIDEO_NAME:-right_wrist_cam_plan.mp4}"
POSE_DEBUG_NAME="${POSE_DEBUG_NAME:-pose_debug.jsonl}"
REVIEW_JSON="${REVIEW_JSON:-/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json}"
REVIEW_MODE="${REVIEW_MODE:-strict}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-27-planner-smooth}"

python /home/zaijia001/ssd/RoboTwin/policy/pi0/scripts/process_repainted_planner_outputs.py \
  "${TASK_NAME}" \
  "${INSTRUCTION}" \
  "${EXPERT_DATA_NUM}" \
  --head-root "${HEAD_ROOT}" \
  --head-dir-template "${HEAD_DIR_TEMPLATE}" \
  --head-video-name "${HEAD_VIDEO_NAME}" \
  --planner-root "${PLANNER_ROOT}" \
  --planner-dir-template "${PLANNER_DIR_TEMPLATE}" \
  --left-wrist-video-name "${LEFT_WRIST_VIDEO_NAME}" \
  --right-wrist-video-name "${RIGHT_WRIST_VIDEO_NAME}" \
  --pose-debug-name "${POSE_DEBUG_NAME}" \
  --review-json "${REVIEW_JSON}" \
  --review-mode "${REVIEW_MODE}" \
  --ignore-ids \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
