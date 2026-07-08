#!/usr/bin/env bash
set -eo pipefail

# SKEYP v2: reinit-compatible gripper-only ablation.
# This pipeline deliberately avoids the v1 planner-output path:
# - motion/state/wrist come from reinit-style h2_pure_d435 world_targets_and_status.npz
# - Stage-1 keeps real objects by removing only hands/arms/wrists/watch
# - Stage-2 repaints only the gripper/end-effector onto the hands-only background

ROOT=${ROOT:-/home/zaijia001/ssd/RoboTwin}
SAM2_ROOT=${SAM2_ROOT:-/home/zaijia001/ssd/inpainting_sam2_robot}
SAM3_ROOT=${SAM3_ROOT:-/home/zaijia001/ssd/inpainting_sam3_robot}
CONDA_SH=${CONDA_SH:-/home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh}

TASKS=${TASKS:-"pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray"}
REVIEW_ROOT=${REVIEW_ROOT:-$ROOT/code_painting/l16_ours_review_first25}
RUN_TAG=${RUN_TAG:-skeyp_v2_reinit_gripperonly_20260708}
LOG_DIR=${LOG_DIR:-/home/zaijia001/tmp/${RUN_TAG}_logs}

# Keep v2 beside v1 roots, but under separate subdirectories.
STAGE1_ROOT=${STAGE1_ROOT:-/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_skeyp/v2_reinit_gripperonly/stage1}
STAGE1_SOURCE_ROOTS=${STAGE1_SOURCE_ROOTS:-"/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1 /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_skeyp/stage1"}
RETARGET_SOURCE_ROOT=${RETARGET_SOURCE_ROOT:-$ROOT/code_painting/human_replay/h2_pure_d435}
RETARGET_ROOT=${RETARGET_ROOT:-$ROOT/code_painting/human_replay/skeyp_v2_reinit_gripperonly/h2_pure_d435_selected25}
REPAINT_ROOT=${REPAINT_ROOT:-/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_skeyp_visible_reinit/v2_reinit_gripperonly/e0_gripper}

DATASET_SUFFIX=${DATASET_SUFFIX:-skeyp_reinit_gripperonly}
TASK_GROUP=${TASK_GROUP:-6task}
N=${N:-120}
SUBSET_N=${SUBSET_N:-25}
ZIP_NAME=${ZIP_NAME:-robot_skeyp_reinit_gripperonly_piper0515_6task_25ep.zip}
PI0_ROOT=${PI0_ROOT:-$ROOT/policy/pi0}
LEROBOT_LOCAL=${LEROBOT_LOCAL:-/home/zaijia001/.cache/huggingface/lerobot/local}
PIPER0515_SUFFIX=${PIPER0515_SUFFIX:-piper0515}
PIPER0515_ROBOT_CONFIG=${PIPER0515_ROBOT_CONFIG:-$ROOT/robot_config_PiperPika_agx_dual_table_0515.json}
PIPER0515_GRIPPER_SCALE=${PIPER0515_GRIPPER_SCALE:-0.0967}
DRY_RUN=${DRY_RUN:-0}

STAGE1_FPS=${STAGE1_FPS:-5}
REPAINT_FPS=${REPAINT_FPS:-5}
OVERWRITE_REPAINT=${OVERWRITE_REPAINT:-0}
GRIPPER_PROMPT=${GRIPPER_PROMPT:-"robotic gripper, gripper fingers, end effector, robot hand."}
GRIPPER_BOX_THRESHOLD=${GRIPPER_BOX_THRESHOLD:-0.30}
GRIPPER_TEXT_THRESHOLD=${GRIPPER_TEXT_THRESHOLD:-0.25}
GRIPPER_MAX_MASK_AREA_RATIO=${GRIPPER_MAX_MASK_AREA_RATIO:-0.12}
GRIPPER_MIN_MASK_AREA_RATIO=${GRIPPER_MIN_MASK_AREA_RATIO:-0.0002}

GPU_PICK_DIVERSE_BOTTLES=${GPU_PICK_DIVERSE_BOTTLES:-0}
GPU_PLACE_BREAD_BASKET=${GPU_PLACE_BREAD_BASKET:-1}
GPU_STACK_CUPS=${GPU_STACK_CUPS:-2}
GPU_HANDOVER_BOTTLE=${GPU_HANDOVER_BOTTLE:-3}
GPU_PNP_BREAD=${GPU_PNP_BREAD:-0}
GPU_PNP_TRAY=${GPU_PNP_TRAY:-1}

mkdir -p "$LOG_DIR" "$STAGE1_ROOT" "$RETARGET_ROOT" "$REPAINT_ROOT"

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

instruction_for_task() {
  case "$1" in
    pick_diverse_bottles) echo "pick up one bottle with one arm, and pick up another bottle with the other arm." ;;
    place_bread_basket) echo "Use one arm to pick up the bread, put it into the basket, and use another arm to lift the basket." ;;
    stack_cups) echo "Stack the dark red and light red cups onto the green cup." ;;
    handover_bottle) echo "Use the right arm to grasp the bottle on the table, handover it to the left arm." ;;
    pnp_bread) echo "Pick up two breads, then place them onto the blue plate." ;;
    pnp_tray) echo "Use the left arm to grasp the red cup, and use the right arm to grasp the bottle, then place them onto the blue tray." ;;
    *) echo "unknown task: $1" >&2; return 1 ;;
  esac
}

selected_ids() {
  local task=$1
  local json="$REVIEW_ROOT/selections/$task/ours_review_selection.json"
  if [[ ! -f "$json" ]]; then
    echo "[error] missing selection json: $json" >&2
    return 1
  fi
  python3 - "$json" "$SUBSET_N" <<'PYIDS'
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
PYIDS
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
  local out="$STAGE1_ROOT/$task/id_$id/stage1_human_inpaint"

  if [[ ! -f "$human" ]]; then
    echo "[error] missing human video: $human" >&2
    return 1
  fi

  log "stage1 fresh hands-only task=$task id=$id gpu=$gpu"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] remove_anything_video_sam2.py --input_video $human --output_dir $out"
    return 0
  fi
  source "$CONDA_SH"
  conda activate inpainting-sam2-r1
  cd "$SAM2_ROOT"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" python remove_anything_video_sam2.py \
    --input_video "$human" \
    --coords_type key_in --point_coords 10 80 --point_labels 1 \
    --dilate_kernel_size 100 \
    --text_prompt "arms, hands, wrists, watch." \
    --box_threshold 0.35 --text_threshold 0.25 \
    --output_dir "$out" \
    --sam_ckpt "$SAM2_ROOT/pretrained_models/sam_vit_h_4b8939.pth" \
    --lama_config "$SAM2_ROOT/lama/configs/prediction/default.yaml" \
    --lama_ckpt "$SAM2_ROOT/pretrained_models/big-lama" \
    --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
    --vi_ckpt "$SAM2_ROOT/pretrained_models/sttn.pth" \
    --mask_idx 2 --fps "$STAGE1_FPS" --device cuda \
    --save_mask_frames 0 --save_mask_video 0 --save_vis_mask_video 0 --save_vis_box_video 0
  cd "$ROOT"
}

ensure_stage1_bg() {
  local task=$1
  local id=$2
  local gpu=$3
  local dst
  dst=$(stage1_bg_path "$STAGE1_ROOT" "$task" "$id")
  if [[ -e "$dst" || -L "$dst" ]]; then
    log "stage1 bg exists task=$task id=$id -> $dst"
    return 0
  fi

  mkdir -p "$(dirname "$dst")"
  local src_root src src_dir dst_dir debug_name src_debug dst_debug
  dst_dir=$(dirname "$dst")
  for src_root in $STAGE1_SOURCE_ROOTS; do
    src=$(stage1_bg_path "$src_root" "$task" "$id")
    if [[ -f "$src" ]]; then
      src_dir=$(dirname "$src")
      ln -sfn "$src" "$dst"
      for debug_name in         "w_box_rgb_${id}.mp4"         "w_mask_rgb_${id}.mp4"         "mask_rgb_${id}.mp4"         "removed_rgb_${id}.mp4"; do
        src_debug="$src_dir/$debug_name"
        dst_debug="$dst_dir/$debug_name"
        if [[ -f "$src_debug" && ! -e "$dst_debug" ]]; then
          ln -sfn "$src_debug" "$dst_debug"
        fi
      done
      log "stage1 bg linked task=$task id=$id -> $src"
      break
    fi
  done

  if [[ ! -e "$dst" && ! -L "$dst" ]]; then
    run_stage1_hands_only "$task" "$id" "$gpu"
  fi

  if [[ ! -f "$dst" ]]; then
    echo "[error] stage1 bg not produced: $dst" >&2
    return 1
  fi
}

ensure_retarget_link() {
  local task=$1
  local id=$2
  local src="$RETARGET_SOURCE_ROOT/$task/id${id}_d435_z005"
  local dst="$RETARGET_ROOT/$task/id${id}_d435_z005"
  local name
  for name in world_targets_and_status.npz zed_replay_d435.mp4 left_wrist_replay.mp4 right_wrist_replay.mp4; do
    if [[ ! -f "$src/$name" ]]; then
      echo "[error] missing retarget input: $src/$name" >&2
      return 1
    fi
  done
  mkdir -p "$RETARGET_ROOT/$task"
  if [[ ! -e "$dst" && ! -L "$dst" ]]; then
    ln -sfn "$src" "$dst"
    log "retarget linked task=$task id=$id -> $src"
  fi
}

run_gripper_only_repaint() {
  local task=$1
  local id=$2
  local gpu=$3
  local robot="$RETARGET_ROOT/$task/id${id}_d435_z005/zed_replay_d435.mp4"
  local bg
  local out="$REPAINT_ROOT/$task/id_${id}_skeyp_gripper"
  local final="$out/final_repainted.mp4"
  bg=$(stage1_bg_path "$STAGE1_ROOT" "$task" "$id")

  if [[ "$OVERWRITE_REPAINT" != "1" && -f "$final" ]]; then
    log "gripper repaint final exists task=$task id=$id -> $final"
    return 0
  fi
  if [[ ! -f "$robot" ]]; then
    echo "[error] missing gripper replay video: $robot" >&2
    return 1
  fi
  if [[ ! -f "$bg" ]]; then
    echo "[error] missing stage1 bg: $bg" >&2
    return 1
  fi

  mkdir -p "$out"
  log "gripper-only repaint task=$task id=$id gpu=$gpu prompt=$GRIPPER_PROMPT"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] visible-reinit gripper repaint input=$robot bg=$bg out=$out"
    return 0
  fi
  cd "$SAM3_ROOT"
  CUDA_VISIBLE_DEVICES="$gpu" python remove_anything_video_sam3_robot_visible_reinit.py \
    --input_video "$robot" \
    --target_video "$bg" \
    --output_dir "$out" \
    --coords_type key_in --point_coords 10 80 --point_labels 1 \
    --text_prompt "$GRIPPER_PROMPT" \
    --box_threshold "$GRIPPER_BOX_THRESHOLD" --text_threshold "$GRIPPER_TEXT_THRESHOLD" \
    --dilate_kernel_size 0 \
    --max_mask_area_ratio "$GRIPPER_MAX_MASK_AREA_RATIO" \
    --min_mask_area_ratio "$GRIPPER_MIN_MASK_AREA_RATIO" \
    --max_white_pixel_ratio_in_mask 0.80 \
    --init_policy first_visible \
    --reinit_policy on_lost \
    --detector_stride 1 \
    --min_visible_consecutive 1 \
    --lost_patience 2 \
    --empty_mask_when_lost 1 \
    --erode_kernel_size 1 \
    --composite_erode_kernel_size 0 \
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
    echo "[error] gripper repaint final missing after run: $final" >&2
    return 1
  fi
}

run_task_repaint() {
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
    ensure_retarget_link "$task" "$id"
  done

  source "$CONDA_SH"
  conda activate inpainting-sam3-dino3
  for id in "${id_array[@]}"; do
    run_gripper_only_repaint "$task" "$id" "$gpu"
  done
}

run_repaint_parallel_stage() {
  local pids=()
  local names=()
  for task in $TASKS; do
    local gpu
    gpu=$(gpu_for_task "$task")
    names+=("$task")
    ( run_task_repaint "$task" "$gpu" ) > "$LOG_DIR/task_${task}.log" 2>&1 &
    pids+=("$!")
    log "submitted v2 repaint worker task=$task gpu=$gpu log=$LOG_DIR/task_${task}.log pid=${pids[-1]}"
  done

  local fail=0
  local i
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "v2 repaint worker finished task=${names[$i]}"
    else
      log "v2 repaint worker FAILED task=${names[$i]} log=$LOG_DIR/task_${names[$i]}.log"
      fail=1
    fi
  done
  if [[ "$fail" != "0" ]]; then
    return 1
  fi
}

process_task() {
  local task=$1
  local ids
  local instruction
  local dataset
  ids=$(selected_ids "$task")
  instruction=$(instruction_for_task "$task")
  dataset="$PI0_ROOT/processed_data/h2o_${task}_${DATASET_SUFFIX}-${N}"

  log "process task=$task dataset=$dataset ids=$ids"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] process_repainted_headcam_with_wrist.py task=$task ids=$ids"
    return 0
  fi
  source "$CONDA_SH"
  conda activate RoboTwin_bw
  cd "$PI0_ROOT"
  python scripts/process_repainted_headcam_with_wrist.py "h2o_${task}_${DATASET_SUFFIX}" "$instruction" "$N" \
    --head-root "$REPAINT_ROOT/$task" \
    --head-dir-template 'id_{id}_skeyp_gripper' \
    --head-video-name final_repainted.mp4 \
    --retarget-root "$RETARGET_ROOT/$task" \
    --retarget-dir-template 'id{id}_d435_z005' \
    --world-targets-name world_targets_and_status.npz \
    --left-wrist-video-name left_wrist_replay.mp4 \
    --right-wrist-video-name right_wrist_replay.mp4 \
    --ids $ids \
    --output-dir "$dataset"
}

lerobot_task() {
  local task=$1
  local instruction
  local dataset
  local source_repo
  instruction=$(instruction_for_task "$task")
  dataset="$PI0_ROOT/processed_data/h2o_${task}_${DATASET_SUFFIX}-${N}"
  source_repo="local/h2o_${task}_${DATASET_SUFFIX}"

  log "lerobot task=$task source=$source_repo"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] convert_aloha_data_to_lerobot_R1.py raw=$dataset repo=$source_repo"
    return 0
  fi
  source "$CONDA_SH"
  conda activate RoboTwin_openvla
  cd "$PI0_ROOT"
  uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py \
    --raw-dir "$dataset" \
    --repo-id "$source_repo" \
    --task "$instruction" \
    --use-wrist \
    --mode video
}

subset_task() {
  local task=$1
  local dataset="$PI0_ROOT/processed_data/h2o_${task}_${DATASET_SUFFIX}-${N}"
  local source_repo="local/h2o_${task}_${DATASET_SUFFIX}"
  local subset_repo="local/h2o_${task}_${DATASET_SUFFIX}_${SUBSET_N}ep"
  local episodes

  log "subset task=$task output=$subset_repo"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] subset first $SUBSET_N episodes from $source_repo"
    return 0
  fi
  episodes=$(python3 - "$dataset" "$SUBSET_N" <<'PYEPS'
import sys
from pathlib import Path
root = Path(sys.argv[1])
limit = int(sys.argv[2])
eps = []
for p in sorted(root.glob("episode_*/instructions.json"), key=lambda p: int(p.parent.name.split("_")[-1])):
    eps.append(int(p.parent.name.split("_")[-1]))
    if len(eps) >= limit:
        break
if len(eps) < limit:
    raise SystemExit(f"expected {limit} processed episodes under {root}, got {len(eps)}")
print(",".join(map(str, eps)))
PYEPS
)
  source "$CONDA_SH"
  conda activate RoboTwin_openvla
  cd "$PI0_ROOT"
  uv run python scripts/subset_lerobot_episodes.py \
    --source "$source_repo" \
    --output-repo-id "$subset_repo" \
    --episodes "$episodes" \
    --overwrite
}

piper0515_task() {
  local task=$1
  local subset_repo="local/h2o_${task}_${DATASET_SUFFIX}_${SUBSET_N}ep"
  local output_repo="local/h2o_${task}_${DATASET_SUFFIX}_${PIPER0515_SUFFIX}_${SUBSET_N}ep"

  log "piper0515 task=$task output=$output_repo"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] convert_lerobot_piper0515_world_to_base.py source=$subset_repo output=$output_repo"
    return 0
  fi
  source "$CONDA_SH"
  conda activate simplevla-rl
  cd "$ROOT"
  python code_painting/convert_lerobot_piper0515_world_to_base.py \
    --source "$subset_repo" \
    --output-repo-id "$output_repo" \
    --robot-config "$PIPER0515_ROBOT_CONFIG" \
    --gripper-scale "$PIPER0515_GRIPPER_SCALE" \
    --overwrite
}

run_conversion_stage() {
  local task
  for task in $TASKS; do
    process_task "$task"
    lerobot_task "$task"
    subset_task "$task"
    piper0515_task "$task"
  done
}

make_zip() {
  local repos=()
  local task
  for task in $TASKS; do
    repos+=("h2o_${task}_${DATASET_SUFFIX}_${PIPER0515_SUFFIX}_${SUBSET_N}ep")
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
  log "validating skeyp v2 output counts"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1: skipping strict output existence checks"
    return 0
  fi
  local failed=0
  local task ids final_count processed_count repo
  for task in $TASKS; do
    ids=$(selected_ids "$task")
    final_count=0
    for id in $ids; do
      if [[ -f "$REPAINT_ROOT/$task/id_${id}_skeyp_gripper/final_repainted.mp4" ]]; then
        final_count=$((final_count + 1))
      fi
    done
    processed_count=$(find "$PI0_ROOT/processed_data/h2o_${task}_${DATASET_SUFFIX}-${N}" -mindepth 2 -maxdepth 2 -type f -name 'episode_*.hdf5' 2>/dev/null | wc -l)
    repo="$LEROBOT_LOCAL/h2o_${task}_${DATASET_SUFFIX}_${PIPER0515_SUFFIX}_${SUBSET_N}ep"
    printf '%s final_repainted=%s/%s processed_hdf5=%s repo=%s\n' "$task" "$final_count" "$SUBSET_N" "$processed_count" "$repo"
    if [[ "$final_count" -ne "$SUBSET_N" || "$processed_count" -ne "$SUBSET_N" || ! -f "$repo/meta/piper0515_world_to_base_conversion.json" ]]; then
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
  log "SKEYP v2 reinit gripper-only pipeline start"
  log "logs: $LOG_DIR"
  log "stage1 root: $STAGE1_ROOT"
  log "retarget root: $RETARGET_ROOT"
  log "repaint root: $REPAINT_ROOT"
  log "dataset suffix: $DATASET_SUFFIX"

  run_repaint_parallel_stage
  run_conversion_stage
  make_zip
  validate_outputs

  log "SKEYP v2 reinit gripper-only pipeline done"
  cat <<EOF

Local outputs:
- Stage1 hands-only backgrounds: $STAGE1_ROOT
- Reinit-style retarget links: $RETARGET_ROOT
- Gripper-only repainted videos: $REPAINT_ROOT
- Processed HDF5: $PI0_ROOT/processed_data/h2o_<TASK>_${DATASET_SUFFIX}-${N}
- Piper0515 LeRobot repos: $LEROBOT_LOCAL/h2o_<TASK>_${DATASET_SUFFIX}_${PIPER0515_SUFFIX}_${SUBSET_N}ep
- Local zip: $LEROBOT_LOCAL/$ZIP_NAME

Manual rclone upload command:
rclone copy $LEROBOT_LOCAL/$ZIP_NAME gdrive:piper/multi/6task/robot_skeyp_reinit_gripperonly_piper0515 -P --drive-chunk-size 64M --transfers 4
EOF
}

main "$@"
