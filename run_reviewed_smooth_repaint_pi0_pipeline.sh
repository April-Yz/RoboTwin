#!/usr/bin/env bash
set -euo pipefail

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh

TASK_NAME="${TASK_NAME:-d_pour_blue}"
REVIEW_JSON="${REVIEW_JSON:-/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json}"
REVIEW_MODE="${REVIEW_MODE:-strict}"

ROBOTWIN_ENV="${ROBOTWIN_ENV:-RoboTwin_bw}"
INPAINT_ENV="${INPAINT_ENV:-inpainting-sam2-r1}"

RUN_SMOOTH="${RUN_SMOOTH:-1}"
RUN_REPAINT="${RUN_REPAINT:-1}"
RUN_PI0="${RUN_PI0:-1}"
DRY_RUN="${DRY_RUN:-0}"

SMOOTH_INPUT_ROOT="${SMOOTH_INPUT_ROOT:-/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3}"
SMOOTH_OUTPUT_ROOT="${SMOOTH_OUTPUT_ROOT:-/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth}"

REPAINT_STAGE1_ROOT="${REPAINT_STAGE1_ROOT:-/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint}"
REPAINT_OUTPUT_ROOT="${REPAINT_OUTPUT_ROOT:-/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_smooth}"

INSTRUCTION="${INSTRUCTION:-pour water}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-auto}"
PI0_OUTPUT_DIR="${PI0_OUTPUT_DIR:-}"
HEAD_DIR_TEMPLATE="${HEAD_DIR_TEMPLATE:-id_{id}_head_cam_arm_gripper_cup_bottle_pad_target}"
HEAD_VIDEO_NAME="${HEAD_VIDEO_NAME:-target_with_original_head_cam_plan.mp4}"
PLANNER_DIR_TEMPLATE="${PLANNER_DIR_TEMPLATE:-d_pour_blue_{id}}"
LEFT_WRIST_VIDEO_NAME="${LEFT_WRIST_VIDEO_NAME:-left_wrist_cam_plan.mp4}"
RIGHT_WRIST_VIDEO_NAME="${RIGHT_WRIST_VIDEO_NAME:-right_wrist_cam_plan.mp4}"
POSE_DEBUG_NAME="${POSE_DEBUG_NAME:-pose_debug.jsonl}"

mapfile -t IDS < <(
  python3 - <<'PY' "${REVIEW_JSON}" "${REVIEW_MODE}"
import json
import sys

json_path = sys.argv[1]
mode = sys.argv[2]

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

videos = data.get('videos', {})
selected = []
for key, item in videos.items():
    label = item.get('label')
    usable = item.get('usable')
    vid = item.get('id', key)
    keep = False
    if mode == 'strict':
        keep = (label == 'y') or (usable is True)
    elif mode == 'include_ambiguous':
        keep = (label in {'y', 'm'}) or (usable is True) or (usable == 'ambiguous')
    else:
        raise SystemExit(f'Unsupported REVIEW_MODE: {mode}')
    if keep:
        selected.append(int(vid))

for vid in sorted(set(selected)):
    print(vid)
PY
)

if [ "${#IDS[@]}" -eq 0 ]; then
  echo "[error] no ids selected from review json: ${REVIEW_JSON}" >&2
  exit 1
fi

SELECTED_COUNT="${#IDS[@]}"
if [ "${EXPERT_DATA_NUM}" = "auto" ]; then
  EXPERT_DATA_NUM="${SELECTED_COUNT}"
fi
if [ -z "${PI0_OUTPUT_DIR}" ]; then
  PI0_OUTPUT_DIR="/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/${TASK_NAME}-${EXPERT_DATA_NUM}-planner-smooth-reviewed-${REVIEW_MODE}"
fi

IDS_STR="${IDS[*]}"
echo "[info] REVIEW_JSON=${REVIEW_JSON}"
echo "[info] REVIEW_MODE=${REVIEW_MODE}"
echo "[info] selected ${SELECTED_COUNT} ids: ${IDS_STR}"
echo "[info] RUN_SMOOTH=${RUN_SMOOTH} RUN_REPAINT=${RUN_REPAINT} RUN_PI0=${RUN_PI0}"
echo "[info] PI0_OUTPUT_DIR=${PI0_OUTPUT_DIR}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[info] dry run only; no step executed"
  exit 0
fi

if [ "${RUN_SMOOTH}" = "1" ]; then
  set +u
  conda activate "${ROBOTWIN_ENV}"
  set -u
  cd /home/zaijia001/ssd/RoboTwin
  TASK_NAME="${TASK_NAME}" \
  INPUT_ROOT="${SMOOTH_INPUT_ROOT}" \
  OUTPUT_ROOT="${SMOOTH_OUTPUT_ROOT}" \
  bash /home/zaijia001/ssd/RoboTwin/code_painting/batch_smooth_planner_outputs.sh "${IDS[@]}"
fi

if [ "${RUN_REPAINT}" = "1" ]; then
  set +u
  conda activate "${INPAINT_ENV}"
  set -u
  cd /home/zaijia001/ssd/inpainting_sam2_robot
  TASK_NAME="${TASK_NAME}" \
  ROBOT_ROOT="${SMOOTH_OUTPUT_ROOT}" \
  STAGE1_ROOT="${REPAINT_STAGE1_ROOT}" \
  OUTPUT_ROOT="${REPAINT_OUTPUT_ROOT}" \
  bash /home/zaijia001/ssd/inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad_from_smooth.sh "${IDS[@]}"
fi

if [ "${RUN_PI0}" = "1" ]; then
  set +u
  conda activate "${ROBOTWIN_ENV}"
  set -u
  cd /home/zaijia001/ssd/RoboTwin/policy/pi0
  TASK_NAME="${TASK_NAME}" \
  INSTRUCTION="${INSTRUCTION}" \
  EXPERT_DATA_NUM="${EXPERT_DATA_NUM}" \
  HEAD_ROOT="${REPAINT_OUTPUT_ROOT}" \
  HEAD_DIR_TEMPLATE="${HEAD_DIR_TEMPLATE}" \
  HEAD_VIDEO_NAME="${HEAD_VIDEO_NAME}" \
  PLANNER_ROOT="${SMOOTH_OUTPUT_ROOT}" \
  PLANNER_DIR_TEMPLATE="${PLANNER_DIR_TEMPLATE}" \
  LEFT_WRIST_VIDEO_NAME="${LEFT_WRIST_VIDEO_NAME}" \
  RIGHT_WRIST_VIDEO_NAME="${RIGHT_WRIST_VIDEO_NAME}" \
  POSE_DEBUG_NAME="${POSE_DEBUG_NAME}" \
  REVIEW_JSON="${REVIEW_JSON}" \
  REVIEW_MODE="${REVIEW_MODE}" \
  OUTPUT_DIR="${PI0_OUTPUT_DIR}" \
  bash /home/zaijia001/ssd/RoboTwin/policy/pi0/run_process_repainted_smoothed_planner_outputs.sh
fi

echo "[done] reviewed smooth->repaint->pi0 pipeline finished"
