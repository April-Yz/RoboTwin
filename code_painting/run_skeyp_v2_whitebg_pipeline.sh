#!/usr/bin/env bash
set -eo pipefail

# SKEYP v2 white-background route.
# Stage-1 is the same hands-only background as skeyp v2.
# Stage-2 bypasses DINO/SAM and composites non-white pixels from the
# reinit-style zed_replay_d435.mp4 onto that background.

ROOT=${ROOT:-/home/zaijia001/ssd/RoboTwin}
CONDA_SH=${CONDA_SH:-/home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh}
TASKS=${TASKS:-"pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray"}
REVIEW_ROOT=${REVIEW_ROOT:-$ROOT/code_painting/l16_ours_review_first25}
RUN_TAG=${RUN_TAG:-skeyp_v2_reinit_whitebg_20260708}
LOG_DIR=${LOG_DIR:-/home/zaijia001/tmp/${RUN_TAG}_logs}

STAGE1_ROOT=${STAGE1_ROOT:-/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_skeyp/v2_reinit_gripperonly/stage1}
STAGE1_SOURCE_ROOTS=${STAGE1_SOURCE_ROOTS:-"/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/stage1 /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_skeyp/stage1"}
RETARGET_SOURCE_ROOT=${RETARGET_SOURCE_ROOT:-$ROOT/code_painting/human_replay/h2_pure_d435}
RETARGET_ROOT=${RETARGET_ROOT:-$ROOT/code_painting/human_replay/skeyp_v2_reinit_gripperonly/h2_pure_d435_selected25}
REPAINT_ROOT=${REPAINT_ROOT:-/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_skeyp_visible_reinit/v2_reinit_whitebg/e0_robot_color}

DATASET_SUFFIX=${DATASET_SUFFIX:-skeyp_reinit_whitebg}
TASK_GROUP=${TASK_GROUP:-6task}
N=${N:-120}
SUBSET_N=${SUBSET_N:-25}
ZIP_NAME=${ZIP_NAME:-robot_skeyp_reinit_whitebg_piper0515_6task_25ep.zip}
PI0_ROOT=${PI0_ROOT:-$ROOT/policy/pi0}
LEROBOT_LOCAL=${LEROBOT_LOCAL:-/home/zaijia001/.cache/huggingface/lerobot/local}
PIPER0515_SUFFIX=${PIPER0515_SUFFIX:-piper0515}
PIPER0515_ROBOT_CONFIG=${PIPER0515_ROBOT_CONFIG:-$ROOT/robot_config_PiperPika_agx_dual_table_0515.json}
PIPER0515_GRIPPER_SCALE=${PIPER0515_GRIPPER_SCALE:-0.0967}
DRY_RUN=${DRY_RUN:-0}
OVERWRITE_REPAINT=${OVERWRITE_REPAINT:-0}
FPS=${FPS:-5}

WHITE_VALUE_MIN=${WHITE_VALUE_MIN:-210}
WHITE_SAT_MAX=${WHITE_SAT_MAX:-45}
WHITE_RGB_MIN=${WHITE_RGB_MIN:-200}
WHITE_RGB_DELTA_MAX=${WHITE_RGB_DELTA_MAX:-55}
BORDER_ONLY=${BORDER_ONLY:-1}
WHITE_DILATE_KERNEL=${WHITE_DILATE_KERNEL:-2}
BLEND_ALPHA_SIGMA=${BLEND_ALPHA_SIGMA:-1.0}
SAVE_MASK_FRAMES=${SAVE_MASK_FRAMES:-0}

mkdir -p "$LOG_DIR" "$STAGE1_ROOT" "$RETARGET_ROOT" "$REPAINT_ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
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

ensure_stage1_bg() {
  local task=$1
  local id=$2
  local dst src_root src src_dir dst_dir debug_name src_debug dst_debug
  dst=$(stage1_bg_path "$STAGE1_ROOT" "$task" "$id")
  if [[ -e "$dst" || -L "$dst" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  dst_dir=$(dirname "$dst")
  for src_root in $STAGE1_SOURCE_ROOTS; do
    src=$(stage1_bg_path "$src_root" "$task" "$id")
    if [[ -f "$src" ]]; then
      src_dir=$(dirname "$src")
      ln -sfn "$src" "$dst"
      for debug_name in "w_box_rgb_${id}.mp4" "w_mask_rgb_${id}.mp4" "mask_rgb_${id}.mp4" "removed_rgb_${id}.mp4"; do
        src_debug="$src_dir/$debug_name"
        dst_debug="$dst_dir/$debug_name"
        if [[ -f "$src_debug" && ! -e "$dst_debug" ]]; then
          ln -sfn "$src_debug" "$dst_debug"
        fi
      done
      break
    fi
  done
  if [[ ! -f "$dst" ]]; then
    echo "[error] missing Stage-1 BG and no reusable source: $dst" >&2
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
    if [[ ! -f "$src/$name" && ! -f "$dst/$name" ]]; then
      echo "[error] missing retarget input: $src/$name" >&2
      return 1
    fi
  done
  mkdir -p "$RETARGET_ROOT/$task"
  if [[ ! -e "$dst" && ! -L "$dst" ]]; then
    ln -sfn "$src" "$dst"
  fi
}

run_color_repaint() {
  local task=$1
  local id=$2
  local final="$REPAINT_ROOT/$task/id_${id}_skeyp_whitebg/final_repainted.mp4"
  if [[ "$OVERWRITE_REPAINT" != "1" && -f "$final" ]]; then
    log "whitebg final exists task=$task id=$id -> $final"
    return 0
  fi
  log "whitebg repaint task=$task id=$id"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] repaint_skeyp_reinit_white_color.py --task $task --ids $id"
    return 0
  fi
  source "$CONDA_SH"
  conda activate RoboTwin_bw
  cd "$ROOT"
  local overwrite_arg=()
  if [[ "$OVERWRITE_REPAINT" == "1" ]]; then
    overwrite_arg=(--overwrite)
  fi
  python code_painting/repaint_skeyp_reinit_white_color.py \
    --task "$task" \
    --ids "$id" \
    --retarget-root "$RETARGET_ROOT" \
    --stage1-root "$STAGE1_ROOT" \
    --out-root "$REPAINT_ROOT" \
    --fps "$FPS" \
    --white-value-min "$WHITE_VALUE_MIN" \
    --white-sat-max "$WHITE_SAT_MAX" \
    --white-rgb-min "$WHITE_RGB_MIN" \
    --white-rgb-delta-max "$WHITE_RGB_DELTA_MAX" \
    --border-only "$BORDER_ONLY" \
    --white-dilate-kernel "$WHITE_DILATE_KERNEL" \
    --blend-alpha-sigma "$BLEND_ALPHA_SIGMA" \
    --save-mask-frames "$SAVE_MASK_FRAMES" \
    "${overwrite_arg[@]}"
}

run_task_repaint() {
  local task=$1
  local ids
  ids=$(selected_ids "$task")
  read -r -a id_array <<< "$ids"
  if [[ "${#id_array[@]}" -ne "$SUBSET_N" ]]; then
    echo "[error] task=$task expected $SUBSET_N ids, got ${#id_array[@]}: ${id_array[*]}" >&2
    return 1
  fi
  log "task=$task selected ids=${id_array[*]}"
  for id in "${id_array[@]}"; do
    ensure_stage1_bg "$task" "$id"
    ensure_retarget_link "$task" "$id"
    run_color_repaint "$task" "$id"
  done
}

run_repaint_parallel_stage() {
  local pids=()
  local names=()
  for task in $TASKS; do
    names+=("$task")
    ( run_task_repaint "$task" ) > "$LOG_DIR/task_${task}.log" 2>&1 &
    pids+=("$!")
    log "submitted whitebg worker task=$task log=$LOG_DIR/task_${task}.log pid=${pids[-1]}"
  done
  local fail=0 i
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "whitebg worker finished task=${names[$i]}"
    else
      log "whitebg worker FAILED task=${names[$i]} log=$LOG_DIR/task_${names[$i]}.log"
      fail=1
    fi
  done
  [[ "$fail" == "0" ]]
}

process_task() {
  local task=$1
  local ids instruction dataset
  ids=$(selected_ids "$task")
  instruction=$(instruction_for_task "$task")
  dataset="$PI0_ROOT/processed_data/h2o_${task}_${DATASET_SUFFIX}-${N}"
  log "process task=$task dataset=$dataset ids=$ids"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] process task=$task ids=$ids"
    return 0
  fi
  source "$CONDA_SH"
  conda activate RoboTwin_bw
  cd "$PI0_ROOT"
  python scripts/process_repainted_headcam_with_wrist.py "h2o_${task}_${DATASET_SUFFIX}" "$instruction" "$N" \
    --head-root "$REPAINT_ROOT/$task" \
    --head-dir-template 'id_{id}_skeyp_whitebg' \
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
  local instruction dataset source_repo
  instruction=$(instruction_for_task "$task")
  dataset="$PI0_ROOT/processed_data/h2o_${task}_${DATASET_SUFFIX}-${N}"
  source_repo="local/h2o_${task}_${DATASET_SUFFIX}"
  log "lerobot task=$task source=$source_repo"
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
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
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
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
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
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
  log "validating skeyp whitebg output counts"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  local failed=0 task ids final_count processed_count repo
  for task in $TASKS; do
    ids=$(selected_ids "$task")
    final_count=0
    for id in $ids; do
      if [[ -f "$REPAINT_ROOT/$task/id_${id}_skeyp_whitebg/final_repainted.mp4" ]]; then
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
  [[ "$failed" == "0" ]]
}

main() {
  log "SKEYP v2 whitebg pipeline start"
  log "logs: $LOG_DIR"
  log "stage1 root: $STAGE1_ROOT"
  log "retarget root: $RETARGET_ROOT"
  log "repaint root: $REPAINT_ROOT"
  log "dataset suffix: $DATASET_SUFFIX"
  run_repaint_parallel_stage
  run_conversion_stage
  make_zip
  validate_outputs
  log "SKEYP v2 whitebg pipeline done"
  cat <<EOF

Local outputs:
- Stage1 hands-only backgrounds: $STAGE1_ROOT
- Reinit-style retarget links: $RETARGET_ROOT
- Whitebg repainted videos: $REPAINT_ROOT
- Processed HDF5: $PI0_ROOT/processed_data/h2o_<TASK>_${DATASET_SUFFIX}-${N}
- Piper0515 LeRobot repos: $LEROBOT_LOCAL/h2o_<TASK>_${DATASET_SUFFIX}_${PIPER0515_SUFFIX}_${SUBSET_N}ep
- Local zip: $LEROBOT_LOCAL/$ZIP_NAME

Manual rclone upload command:
rclone copy $LEROBOT_LOCAL/$ZIP_NAME gdrive:piper/multi/6task/robot_skeyp_reinit_whitebg_piper0515 -P --drive-chunk-size 64M --transfers 4
EOF
}

main "$@"
