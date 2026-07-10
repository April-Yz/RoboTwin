#!/usr/bin/env bash
set -euo pipefail

RUN_TAG=graspnet
TASK_GROUP=6task
SUBSET_N=25
N=120

REPO=/home/zaijia001/ssd/RoboTwin
WRAPPER=$REPO/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh
REVIEW_ROOT=$REPO/code_painting/l16_ours_review_first25
PLANNER_ROOT=$REPO/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/S_graspnet_topscore_rightcam_m003_selected25
STAGE2_ROOT=/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/stage2_color_graspnet_selected25/e0_robot_object
DEFAULT_STAGE1_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object
STACK_STAGE1_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/B_points_negative
LOG_ROOT=/home/zaijia001/tmp/graspnet_selected25_logs_20260710
LEROBOT_LOCAL=/home/zaijia001/.cache/huggingface/lerobot/local
ZIP_NAME=robot_graspnet_piper0515_6task_25ep.zip
RCLONE_DST=gdrive:piper/multi/6task/robot_graspnet_piper0515
HEAD_DIR_TEMPLATE="id_{id}_l16_white_color_human_object"
DRY_RUN=${DRY_RUN:-0}
SKIP_UPLOAD=${SKIP_UPLOAD:-0}

TASKS=(pick_diverse_bottles place_bread_basket handover_bottle pnp_bread pnp_tray stack_cups)
declare -A IDS=(
  [pick_diverse_bottles]="0 1 2 3 4 5 6 8 9 10 11 13 15 20 23 25 27 28 30 31 33 36 37 38 39"
  [place_bread_basket]="0 1 2 4 23 24 25 26 27 28 29 30 32 33 34 39 42 43 47 48 50 52 54 55 56"
  [handover_bottle]="1 3 5 8 10 14 15 18 19 20 21 22 23 26 27 28 30 31 32 34 35 36 39 40 41"
  [pnp_bread]="7 8 12 14 17 19 20 36 37 38 40 42 43 44 45 47 49 50 51 53 54 55 56 57 59"
  [pnp_tray]="2 3 4 5 6 8 9 10 12 13 15 16 17 18 19 20 21 23 24 25 26 27 28 29 30"
  [stack_cups]="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 23 25 26 29"
)
declare -A GPUS=(
  [pick_diverse_bottles]=0
  [place_bread_basket]=1
  [handover_bottle]=2
  [pnp_bread]=3
  [pnp_tray]=0
  [stack_cups]=1
)

mkdir -p "$PLANNER_ROOT" "$STAGE2_ROOT" "$LOG_ROOT"
cd "$REPO"

run_stage1_task() {
  local task=$1
  local gpu=${GPUS[$task]}
  local ids=${IDS[$task]}
  echo "[stage1/start] task=$task gpu=$gpu ids=$ids"
  bash "$WRAPPER" \
    --skip_preview_generation \
    --gpu "$gpu" --ids $ids --continue_on_error --tasks "$task" \
    --candidate_selection_mode top_score_auto \
    --candidate_max_rotation_distance_deg -1 \
    --candidate_keep_camera_up 0 \
    --trajectory_mode joint_interp \
    --ik_max_rotation_threshold_rad 3.14 \
    --replan_attempts 5 \
    --disable_execution_collisions \
    --pure_scene_output 1 \
    --debug_candidate_top_k 0 \
    --debug_common_candidate_top_k 0 \
    --debug_visualize_selected_keyframe_axes 0 \
    --debug_visualize_ik_waypoints 0 \
    --piper_calibration_bundle "$REPO/calibration_bundle_piper_new_table_0515.json" \
    --wrist_left_forward_offset_m -0.04 \
    --wrist_right_forward_offset_m -0.03 \
    --wrist_left_roll_deg 14.635 \
    --wrist_right_roll_deg -44.649 \
    --wrist_left_yaw_deg 0.182 \
    --wrist_right_yaw_deg 0.840 \
    --wrist_left_pitch_deg -90 \
    --wrist_right_pitch_deg -90 \
    --wrist_left_lateral_offset_m -0.0207 \
    --wrist_right_lateral_offset_m 0.0274 \
    --output_root "$PLANNER_ROOT"
  echo "[stage1/done] task=$task"
}

run_stage2_task() {
  local task=$1
  local gpu=${GPUS[$task]}
  local ids=${IDS[$task]}
  local stage1_root=$DEFAULT_STAGE1_ROOT
  if [[ "$task" == "stack_cups" ]]; then
    stage1_root=$STACK_STAGE1_ROOT
  fi
  set +u
  source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
  conda activate RoboTwin_bw
  set -u
  echo "[stage2/start] task=$task gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu python "$REPO/code_painting/repaint_l16_white_color_debug.py" \
    --task "$task" \
    --ids "$ids" \
    --l16-root "$PLANNER_ROOT" \
    --stage1-root "$stage1_root" \
    --out-root "$STAGE2_ROOT" \
    --fps 5 \
    --overwrite
  echo "[stage2/done] task=$task"
}

wait_all() {
  local label=$1
  shift
  local failed=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      echo "[error] $label pid=$pid failed" >&2
      failed=1
    fi
  done
  [[ "$failed" == 0 ]] || { echo "[fatal] $label failed" >&2; exit 1; }
}

if [[ "${SKIP_STAGE1:-0}" != "1" ]]; then
  pids=()
  for task in "${TASKS[@]}"; do
    (run_stage1_task "$task") >"$LOG_ROOT/stage1_${task}.log" 2>&1 &
    pids+=("$!")
  done
  wait_all stage1 "${pids[@]}"
fi

if [[ "${SKIP_STAGE2:-0}" != "1" ]]; then
  pids=()
  for task in "${TASKS[@]}"; do
    (run_stage2_task "$task") >"$LOG_ROOT/stage2_${task}.log" 2>&1 &
    pids+=("$!")
  done
  wait_all stage2 "${pids[@]}"
fi

set +u
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
set -u
TASKS="${TASKS[*]}" \
TASK_GROUP="$TASK_GROUP" \
REVIEW_ROOT="$REVIEW_ROOT" \
HEAD_ROOT="$STAGE2_ROOT" \
HEAD_DIR_TEMPLATE="$HEAD_DIR_TEMPLATE" \
STACK_HEAD_ROOT="$STAGE2_ROOT" \
PLANNER_ROOT="$PLANNER_ROOT" \
DATASET_SUFFIX="$RUN_TAG" \
N="$N" \
SUBSET_N="$SUBSET_N" \
STEPS="process lerobot subset piper0515" \
DRY_RUN=1 \
bash "$REPO/code_painting/run_l16_ours_selected_pipeline.sh"

cd "$LEROBOT_LOCAL"
rm -f "$ZIP_NAME"
dirs=()
for task in "${TASKS[@]}"; do
  dir="h2o_${task}_${RUN_TAG}_piper0515_${SUBSET_N}ep"
  count=$(find "$dir/data/chunk-000" -maxdepth 1 -type f -name 'episode_*.parquet' | wc -l)
  echo "[validate] task=$task parquet=$count"
  [[ "$count" -eq "$SUBSET_N" ]] || { echo "[fatal] $task expected 25 episodes" >&2; exit 1; }
  [[ -f "$dir/meta/piper0515_world_to_base_conversion.json" ]] || { echo "[fatal] missing piper0515 marker for $task" >&2; exit 1; }
  dirs+=("$dir")
done
zip -r "$ZIP_NAME" "${dirs[@]}"

if [[ "$SKIP_UPLOAD" == "1" ]]; then
  echo "[upload/skip] local zip is ready; rclone was not called"
elif [[ "$DRY_RUN" == "1" ]]; then
  rclone copy "$LEROBOT_LOCAL/$ZIP_NAME" "$RCLONE_DST" -P --drive-chunk-size 64M --transfers 4 --dry-run
else
  rclone copy "$LEROBOT_LOCAL/$ZIP_NAME" "$RCLONE_DST" -P --drive-chunk-size 64M --transfers 4
fi

echo "[done] zip=$LEROBOT_LOCAL/$ZIP_NAME"
echo "[done] rclone=$RCLONE_DST"
