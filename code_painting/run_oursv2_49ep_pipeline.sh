#!/usr/bin/env bash
set -euo pipefail

REPO=/home/zaijia001/ssd/RoboTwin
PI0_ROOT=$REPO/policy/pi0
LEROBOT_LOCAL=/home/zaijia001/.cache/huggingface/lerobot/local
TASKS=(pick_diverse_bottles place_bread_basket handover_bottle pnp_bread pnp_tray stack_cups)
TASKS_STRING="${TASKS[*]}"

SELECTION_SCRIPT=$REPO/code_painting/prepare_oursv2_49ep_selection.py
SELECTION_ROOT=$REPO/code_painting/l16_oursv2_review_49ep
HEAD_ROOT=/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/stage2_color_rightcam_m003_full_0_120/e0_robot_object
PLANNER_ROOT=$REPO/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_de_human_replay_clean_right_cam
HEAD_DIR_TEMPLATE="id_{id}_l16_white_color_human_object"

POOL_SUFFIX=oursv2_49pool
FINAL_SUFFIX=oursv2
N=120
SUBSET_N=49
ZIP_NAME=robot_oursv2_piper0515_6task_49ep.zip
RCLONE_DST=gdrive:piper/multi/6task/robot_oursv2_piper0515_49ep
DRY_RUN=${DRY_RUN:-0}
SKIP_BUILD=${SKIP_BUILD:-0}

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
cd "$REPO"

python3 "$SELECTION_SCRIPT" --allow-repeat
echo "[selection] $SELECTION_ROOT/oursv2_49ep_selection_summary.md"

if [[ "$SKIP_BUILD" != "1" ]]; then
  TASKS="$TASKS_STRING" \
  REVIEW_ROOT="$SELECTION_ROOT" \
  HEAD_ROOT="$HEAD_ROOT" \
  STACK_HEAD_ROOT="$HEAD_ROOT" \
  HEAD_DIR_TEMPLATE="$HEAD_DIR_TEMPLATE" \
  PLANNER_ROOT="$PLANNER_ROOT" \
  DATASET_SUFFIX="$POOL_SUFFIX" \
  N="$N" \
  SUBSET_N="$SUBSET_N" \
  STEPS="process lerobot" \
  DRY_RUN=1 \
  bash "$REPO/code_painting/run_l16_ours_selected_pipeline.sh"
fi

set +u
conda activate RoboTwin_openvla
set -u
cd "$PI0_ROOT"
for task in "${TASKS[@]}"; do
  processed="$PI0_ROOT/processed_data/h2o_${task}_${POOL_SUFFIX}-${N}"
  source_repo="local/h2o_${task}_${POOL_SUFFIX}"
  subset_repo="local/h2o_${task}_${FINAL_SUFFIX}_${SUBSET_N}ep"
  spec=$(python3 "$SELECTION_SCRIPT" \
    --output-root "$SELECTION_ROOT" \
    --print-processed-spec \
    --task "$task" \
    --processed-root "$processed")
  echo "[subset] task=$task episodes=$spec"
  uv run python scripts/subset_lerobot_episodes.py \
    --source "$source_repo" \
    --output-repo-id "$subset_repo" \
    --episodes "$spec" \
    --allow-duplicates \
    --overwrite
done

set +u
conda activate simplevla-rl
set -u
cd "$REPO"
for task in "${TASKS[@]}"; do
  python code_painting/convert_lerobot_piper0515_world_to_base.py \
    --source "local/h2o_${task}_${FINAL_SUFFIX}_${SUBSET_N}ep" \
    --output-repo-id "local/h2o_${task}_${FINAL_SUFFIX}_piper0515_${SUBSET_N}ep" \
    --robot-config "$REPO/robot_config_PiperPika_agx_dual_table_0515.json" \
    --gripper-scale 0.0967 \
    --overwrite
done

cd "$LEROBOT_LOCAL"
zip_path="$LEROBOT_LOCAL/$ZIP_NAME"
rm -f "$zip_path"
zip_dirs=()
for task in "${TASKS[@]}"; do
  dir="h2o_${task}_${FINAL_SUFFIX}_piper0515_${SUBSET_N}ep"
  count=$(find "$dir/data/chunk-000" -maxdepth 1 -type f -name 'episode_*.parquet' | wc -l)
  marker="$dir/meta/piper0515_world_to_base_conversion.json"
  echo "[validate] task=$task parquet=$count marker=$marker"
  [[ "$count" -eq "$SUBSET_N" ]] || { echo "[fatal] $task expected $SUBSET_N parquet, got $count" >&2; exit 1; }
  [[ -f "$marker" ]] || { echo "[fatal] missing $marker" >&2; exit 1; }
  zip_dirs+=("$dir")
done
zip -r "$ZIP_NAME" "${zip_dirs[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  rclone copy "$zip_path" "$RCLONE_DST" -P --drive-chunk-size 64M --transfers 4 --dry-run
else
  rclone copy "$zip_path" "$RCLONE_DST" -P --drive-chunk-size 64M --transfers 4
fi

echo "[done] zip=$zip_path"
echo "[done] rclone=$RCLONE_DST"
