#!/usr/bin/env bash
set -eo pipefail

# SKEYP ablation: keyframe-based human-to-robot replay plus robot-only repaint.
# The script is intentionally isolated from the existing ours/reinit commands:
# it writes to skeyp-specific roots and only reuses stable public entrypoints.

ROOT=${ROOT:-/home/zaijia001/ssd/RoboTwin}
SAM2_ROOT=${SAM2_ROOT:-/home/zaijia001/ssd/inpainting_sam2_robot}
SAM3_ROOT=${SAM3_ROOT:-/home/zaijia001/ssd/inpainting_sam3_robot}
CONDA_SH=${CONDA_SH:-/home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh}

TASKS=${TASKS:-"pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray"}
REVIEW_ROOT=${REVIEW_ROOT:-$ROOT/code_painting/l16_ours_review_first25}

RUN_TAG=${RUN_TAG:-skeyp_selected25_rightcam_m003_20260708}
LOG_DIR=${LOG_DIR:-/home/zaijia001/tmp/${RUN_TAG}_logs}
STAGE1_SOURCE_ROOT=${STAGE1_SOURCE_ROOT:-/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1}
STAGE1_ROOT=${STAGE1_ROOT:-/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_skeyp/stage1}
PLANNER_ROOT=${PLANNER_ROOT:-$ROOT/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/${RUN_TAG}}
REPAINT_ROOT=${REPAINT_ROOT:-/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_skeyp_visible_reinit/e0_robot}

DATASET_SUFFIX=${DATASET_SUFFIX:-skeyp}
TASK_GROUP=${TASK_GROUP:-6task}
N=${N:-120}
SUBSET_N=${SUBSET_N:-25}
ZIP_NAME=${ZIP_NAME:-robot_skeyp_piper0515_6task_25ep.zip}
LEROBOT_LOCAL=${LEROBOT_LOCAL:-/home/zaijia001/.cache/huggingface/lerobot/local}
DRY_RUN=${DRY_RUN:-0}

STAGE1_FPS=${STAGE1_FPS:-5}
REPAINT_FPS=${REPAINT_FPS:-5}
OVERWRITE_REPAINT=${OVERWRITE_REPAINT:-0}
OVERWRITE_PLAN=${OVERWRITE_PLAN:-1}

GPU_PICK_DIVERSE_BOTTLES=${GPU_PICK_DIVERSE_BOTTLES:-0}
GPU_PLACE_BREAD_BASKET=${GPU_PLACE_BREAD_BASKET:-1}
GPU_STACK_CUPS=${GPU_STACK_CUPS:-2}
GPU_HANDOVER_BOTTLE=${GPU_HANDOVER_BOTTLE:-3}
GPU_PNP_BREAD=${GPU_PNP_BREAD:-0}
GPU_PNP_TRAY=${GPU_PNP_TRAY:-1}

mkdir -p "$LOG_DIR" "$STAGE1_ROOT" "$PLANNER_ROOT" "$REPAINT_ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

gpu_for_task() {
  case "$1" in
    pick_diverse_bottles) echo "$GPU_PICK_DIVERSE_BOTTLES" ;;
    place_bread_basket) echo "$GPU_PLACE_BREAD_BASKET" ;;
    stack_cups) echo "$GPU_STACK_CUPS" ;;
    handover_bottle) echo "$GPU_HANDOVER_BOTTLE" ;;
    pnp_bread) echo "$GPU_PNP_BREAD" ;;
    pnp_tray) echo "$GPU_PNP_TRAY" ;;
    *) echo 0 ;;
  esac
}

selected_ids() {
  local task=$1
  local json="$REVIEW_ROOT/selections/$task/ours_review_selection.json"
  if [[ ! -f "$json" ]]; then
    echo "[error] missing selection json: $json" >&2
    return 1
  fi
  python3 - "$json" "$SUBSET_N" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
limit = int(sys.argv[2])
data = json.loads(path.read_text())

def parse_id(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r'(?:id_|foundation_input_|rgb_)?(\d+)', value)
        if m:
            return int(m.group(1))
    return None

records = []
if isinstance(data, dict):
    for key in ("selected_ids", "ids", "selected", "episodes"):
        if isinstance(data.get(key), list):
            for item in data[key]:
                if isinstance(item, dict):
                    idx = parse_id(item.get("id") or item.get("episode_id") or item.get("video") or item.get("path"))
                    status = str(item.get("status") or item.get("label") or item.get("decision") or "").lower()
                    if idx is not None and status not in {"n", "no", "bad", "reject", "rejected", "skip"}:
                        records.append(idx)
                else:
                    idx = parse_id(item)
                    if idx is not None:
                        records.append(idx)
    if isinstance(data.get("videos"), dict):
        for key, item in data["videos"].items():
            idx = parse_id(key)
            if isinstance(item, dict):
                idx = parse_id(item.get("id") or item.get("episode_id") or item.get("video") or key)
                status = str(item.get("status") or item.get("label") or item.get("decision") or "").lower()
                if status in {"n", "no", "bad", "reject", "rejected", "skip"}:
                    continue
            if idx is not None:
                records.append(idx)

seen = set()
ordered = []
for idx in records:
    if idx not in seen:
        seen.add(idx)
        ordered.append(idx)

if not ordered:
    raise SystemExit(f"no ids found in {path}")
print(" ".join(str(x) for x in ordered[:limit]))
PY
}

stage1_bg_path() {
  local root=$1
  local task=$2
  local id=$3
  echo "$root/$task/id_$id/stage1_human_inpaint/removed_w_mask_rgb_$id.mp4"
}

run_stage1_hands_only() {
  local task=$1
  local id=$2
  local gpu=$3
  local human="$HOME/ssd/data/piper/hand/$task/harmer_input/rgb_$id.mp4"
  local out="$STAGE1_ROOT/$task/id_$id"
  local dummy_robot
  dummy_robot=$(find "$HOME/ssd/human_replay/h2_pure_d435/$task" -maxdepth 2 -type f -name zed_replay_d435.mp4 -print -quit 2>/dev/null)

  if [[ ! -f "$human" ]]; then
    echo "[error] missing human video: $human" >&2
    return 1
  fi
  if [[ -z "$dummy_robot" || ! -f "$dummy_robot" ]]; then
    echo "[error] missing dummy robot video under $HOME/ssd/human_replay/h2_pure_d435/$task" >&2
    return 1
  fi

  log "stage1 fresh hands-only task=$task id=$id gpu=$gpu"
  source "$CONDA_SH"
  conda activate inpainting-sam2-r1
  cd "$SAM2_ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" python run_human_robot_inpaint_repaint.py \
    --human_video "$human" \
    --robot_video "$dummy_robot" \
    --output_root "$out" \
    --coords_type key_in --point_coords 10 80 --point_labels 1 \
    --human_text_prompt "arms, hands, wrists, watch." \
    --robot_text_prompt "left robot arm, right robot arm, forearm, wrist, gripper, end effector." \
    --human_box_threshold 0.35 --human_text_threshold 0.25 \
    --robot_box_threshold 0.20 --robot_text_threshold 0.20 \
    --human_dilate_kernel_size 100 \
    --robot_dilate_kernel_size 0 \
    --robot_max_mask_area_ratio 1.0 \
    --robot_erode_kernel_size 3 \
    --robot_composite_erode_kernel_size 1 \
    --robot_blend_alpha_sigma 1.0 \
    --robot_exclude_bottom_ratio 0.14 \
    --fps "$STAGE1_FPS" --device cuda --mask_idx 2 \
    --human_save_debug_artifacts 0 \
    --robot_save_removed_video 0 \
    --robot_save_mask_artifacts 0 \
    --robot_save_debug_videos 0 \
    --robot_save_composite_video 0
  cd "$ROOT"
}

ensure_stage1_bg() {
  local task=$1
  local id=$2
  local gpu=$3
  local dst
  local src
  dst=$(stage1_bg_path "$STAGE1_ROOT" "$task" "$id")
  src=$(stage1_bg_path "$STAGE1_SOURCE_ROOT" "$task" "$id")

  if [[ -e "$dst" || -L "$dst" ]]; then
    log "stage1 bg exists task=$task id=$id -> $dst"
    return 0
  fi

  mkdir -p "$(dirname "$dst")"
  if [[ -f "$src" ]]; then
    ln -sfn "$src" "$dst"
    log "stage1 bg linked task=$task id=$id -> $src"
  else
    run_stage1_hands_only "$task" "$id" "$gpu"
  fi

  if [[ ! -f "$dst" ]]; then
    echo "[error] stage1 bg not produced: $dst" >&2
    return 1
  fi
}

run_planner() {
  local task=$1
  local gpu=$2
  shift 2
  local ids=("$@")

  log "planner task=$task gpu=$gpu ids=${ids[*]}"
  cd "$ROOT"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] bash code_painting/run_plan_keyframes_human_replay_piper_d435.sh --gpu $gpu --ids ${ids[*]} --tasks $task --output_root $PLANNER_ROOT"
    return 0
  fi

  bash "$ROOT/code_painting/run_plan_keyframes_human_replay_piper_d435.sh" \
    --gpu "$gpu" --ids "${ids[@]}" --continue_on_error --tasks "$task" \
    --target_retreat_m 0.14 \
    --wrist_left_forward_offset_m -0.04 --wrist_right_forward_offset_m -0.03 \
    --wrist_left_roll_deg 14.635 --wrist_right_roll_deg -44.649 \
    --wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840 \
    --wrist_left_pitch_deg -90 --wrist_right_pitch_deg -90 \
    --wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274 \
    --output_root "$PLANNER_ROOT"
}

run_robot_only_repaint() {
  local task=$1
  local id=$2
  local gpu=$3
  local robot="$PLANNER_ROOT/$task/foundation_input_$id/head_cam_plan.mp4"
  local bg
  local out="$REPAINT_ROOT/$task/id_${id}_skeyp"
  local final="$out/final_repainted.mp4"
  bg=$(stage1_bg_path "$STAGE1_ROOT" "$task" "$id")

  if [[ "$OVERWRITE_REPAINT" != "1" && -f "$final" ]]; then
    log "repaint final exists task=$task id=$id -> $final"
    return 0
  fi
  if [[ ! -f "$robot" ]]; then
    echo "[error] missing planner head video: $robot" >&2
    return 1
  fi
  if [[ ! -f "$bg" ]]; then
    echo "[error] missing stage1 bg: $bg" >&2
    return 1
  fi

  mkdir -p "$out"
  log "robot-only repaint task=$task id=$id gpu=$gpu"
  cd "$SAM3_ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" python remove_anything_video_sam3_robot_visible_reinit.py \
    --input_video "$robot" \
    --target_video "$bg" \
    --output_dir "$out" \
    --coords_type key_in --point_coords 10 80 --point_labels 1 \
    --text_prompt "robot arm, robotic gripper, robot wrist, robot forearm." \
    --box_threshold 0.35 --text_threshold 0.30 \
    --dilate_kernel_size 0 \
    --max_mask_area_ratio 0.35 \
    --min_mask_area_ratio 0.002 \
    --max_white_pixel_ratio_in_mask 0.60 \
    --init_policy first_visible \
    --reinit_policy on_lost \
    --detector_stride 1 \
    --min_visible_consecutive 1 \
    --lost_patience 2 \
    --empty_mask_when_lost 1 \
    --erode_kernel_size 3 \
    --composite_erode_kernel_size 1 \
    --blend_alpha_sigma 1.0 \
    --exclude_bottom_ratio 0.14 \
    --fps "$REPAINT_FPS" --device cuda \
    --save_mask_frames 0 \
    --save_mask_video 1 \
    --save_vis_mask_video 1 \
    --save_vis_box_video 1 \
    --save_removed_video 0 \
    --save_target_composite_video 1
  cd "$ROOT"

  if [[ ! -f "$final" ]]; then
    echo "[error] repaint final missing after run: $final" >&2
    return 1
  fi
}

run_task() {
  local task=$1
  local gpu=$2
  local ids
  ids=$(selected_ids "$task")
  read -r -a id_array <<< "$ids"

  if [[ "${#id_array[@]}" -ne "$SUBSET_N" ]]; then
    echo "[error] task=$task expected $SUBSET_N ids, got ${#id_array[@]}: ${id_array[*]}" >&2
    return 1
  fi

  log "task=$task gpu=$gpu selected ids=${id_array[*]}"
  for id in "${id_array[@]}"; do
    ensure_stage1_bg "$task" "$id" "$gpu"
  done

  run_planner "$task" "$gpu" "${id_array[@]}"

  source "$CONDA_SH"
  conda activate inpainting-sam3-dino3
  for id in "${id_array[@]}"; do
    run_robot_only_repaint "$task" "$id" "$gpu"
  done
}

run_task_parallel_stage() {
  local pids=()
  local names=()
  for task in $TASKS; do
    local gpu
    gpu=$(gpu_for_task "$task")
    names+=("$task")
    (
      run_task "$task" "$gpu"
    ) > "$LOG_DIR/task_${task}.log" 2>&1 &
    pids+=("$!")
    log "submitted task worker task=$task gpu=$gpu log=$LOG_DIR/task_${task}.log pid=${pids[-1]}"
  done

  local fail=0
  local i
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "task worker finished task=${names[$i]}"
    else
      log "task worker FAILED task=${names[$i]} log=$LOG_DIR/task_${names[$i]}.log"
      fail=1
    fi
  done

  if [[ "$fail" != "0" ]]; then
    return 1
  fi
}

run_conversion() {
  log "running selected pipeline conversion with DATASET_SUFFIX=$DATASET_SUFFIX"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] STEPS='process lerobot subset piper0515' bash code_painting/run_l16_ours_selected_pipeline.sh"
    return 0
  fi
  cd "$ROOT"
  STEPS="process lerobot subset piper0515" \
  TASKS="$TASKS" \
  TASK_GROUP="$TASK_GROUP" \
  REVIEW_ROOT="$REVIEW_ROOT" \
  HEAD_ROOT="$REPAINT_ROOT" \
  STACK_HEAD_ROOT="$REPAINT_ROOT" \
  HEAD_DIR_TEMPLATE='id_{id}_skeyp' \
  STACK_HEAD_DIR_TEMPLATE='id_{id}_skeyp' \
  PLANNER_ROOT="$PLANNER_ROOT" \
  DATASET_SUFFIX="$DATASET_SUFFIX" \
  N="$N" \
  SUBSET_N="$SUBSET_N" \
  DRY_RUN=1 \
  bash "$ROOT/code_painting/run_l16_ours_selected_pipeline.sh"
}

make_zip() {
  local repos=()
  local task
  for task in $TASKS; do
    repos+=("h2o_${task}_${DATASET_SUFFIX}_piper0515_${SUBSET_N}ep")
  done

  log "creating local zip $LEROBOT_LOCAL/$ZIP_NAME"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] cd $LEROBOT_LOCAL && zip -r $ZIP_NAME ${repos[*]}"
    return 0
  fi
  cd "$LEROBOT_LOCAL"
  rm -f "$ZIP_NAME"
  zip -r "$ZIP_NAME" "${repos[@]}"
}

validate_outputs() {
  log "validating key output counts"
  local task
  local failed=0
  for task in $TASKS; do
    local ids
    local final_count
    local repo
    ids=$(selected_ids "$task")
    final_count=0
    for id in $ids; do
      if [[ -f "$REPAINT_ROOT/$task/id_${id}_skeyp/final_repainted.mp4" ]]; then
        final_count=$((final_count + 1))
      fi
    done
    repo="$LEROBOT_LOCAL/h2o_${task}_${DATASET_SUFFIX}_piper0515_${SUBSET_N}ep"
    printf '%s final_repainted=%s/%s repo=%s\n' "$task" "$final_count" "$SUBSET_N" "$repo"
    if [[ "$final_count" -ne "$SUBSET_N" || ! -d "$repo" ]]; then
      failed=1
    fi
  done

  if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -f "$LEROBOT_LOCAL/$ZIP_NAME" ]]; then
      echo "[error] missing zip: $LEROBOT_LOCAL/$ZIP_NAME" >&2
      failed=1
    else
      ls -lh "$LEROBOT_LOCAL/$ZIP_NAME"
      if command -v zipinfo >/dev/null 2>&1; then
        printf 'zip parquet count: '
        zipinfo -1 "$LEROBOT_LOCAL/$ZIP_NAME" | grep -c '/data/.*\.parquet$' || true
        printf 'zip piper0515 marker count: '
        zipinfo -1 "$LEROBOT_LOCAL/$ZIP_NAME" | grep -c 'piper0515_world_to_base_conversion.json$' || true
      fi
    fi
  fi

  if [[ "$failed" != "0" ]]; then
    return 1
  fi
}

main() {
  log "SKEYP selected25 pipeline start"
  log "logs: $LOG_DIR"
  log "stage1 root: $STAGE1_ROOT"
  log "planner root: $PLANNER_ROOT"
  log "repaint root: $REPAINT_ROOT"

  run_task_parallel_stage
  run_conversion
  make_zip
  validate_outputs

  log "SKEYP selected25 pipeline done"
  cat <<EOF

Local outputs:
- Stage1 hands-only backgrounds: $STAGE1_ROOT
- Keyframe planner outputs: $PLANNER_ROOT
- Robot-only repainted videos: $REPAINT_ROOT
- Piper0515 LeRobot repos: $LEROBOT_LOCAL/h2o_<TASK>_${DATASET_SUFFIX}_piper0515_${SUBSET_N}ep
- Local zip: $LEROBOT_LOCAL/$ZIP_NAME

Manual rclone upload command:
rclone copy $LEROBOT_LOCAL/$ZIP_NAME gdrive:piper/multi/6task/robot_skeyp_piper0515 -P --drive-chunk-size 64M --transfers 4
EOF
}

main "$@"
