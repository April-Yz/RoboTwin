#!/usr/bin/env bash
set -eo pipefail

# Build L16 "ours" training datasets from review JSON files.
# Default flow: process planner outputs -> LeRobot -> selected subset ->
# Piper0515 world-to-base conversion -> zip/rclone dry-run.

TASKS=${TASKS:-"pick_diverse_bottles place_bread_basket handover_bottle pnp_bread pnp_tray stack_cups"}
STEPS=${STEPS:-"process lerobot subset piper0515 zip"}
DATASET_SUFFIX=${DATASET_SUFFIX:-ours}
N=${N:-120}
SUBSET_N=${SUBSET_N:-25}
TASK_GROUP=${TASK_GROUP:-6task}
DRY_RUN=${DRY_RUN:-1}
REVIEW_ROOT=${REVIEW_ROOT:-/home/zaijia001/ssd/RoboTwin/code_painting/l16_ours_review}
HEAD_ROOT=${HEAD_ROOT:-/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object}
STACK_HEAD_ROOT=${STACK_HEAD_ROOT:-/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative}
PLANNER_ROOT=${PLANNER_ROOT:-/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean}
PI0_ROOT=${PI0_ROOT:-/home/zaijia001/ssd/RoboTwin/policy/pi0}
LEROBOT_LOCAL=${LEROBOT_LOCAL:-/home/zaijia001/.cache/huggingface/lerobot/local}
PIPER0515_SUFFIX=${PIPER0515_SUFFIX:-piper0515}
PIPER0515_ROBOT_CONFIG=${PIPER0515_ROBOT_CONFIG:-/home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json}
PIPER0515_GRIPPER_SCALE=${PIPER0515_GRIPPER_SCALE:-0.0967}
RCLONE_DST_ENV=${RCLONE_DST:-}
ZIP_NAME_ENV=${ZIP_NAME:-}

has_step() {
  case " ${STEPS} " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -z "$ZIP_NAME_ENV" ]]; then
  if has_step piper0515; then
    ZIP_NAME="robot_ours_${PIPER0515_SUFFIX}_${TASK_GROUP}_${SUBSET_N}ep.zip"
  else
    ZIP_NAME="robot_ours_${TASK_GROUP}_${SUBSET_N}ep.zip"
  fi
else
  ZIP_NAME="$ZIP_NAME_ENV"
fi

if [[ -z "$RCLONE_DST_ENV" ]]; then
  if has_step piper0515; then
    RCLONE_DST="gdrive:piper/multi/${TASK_GROUP}/robot_ours_${PIPER0515_SUFFIX}"
  else
    RCLONE_DST="gdrive:piper/multi/${TASK_GROUP}/robot_ours"
  fi
else
  RCLONE_DST="$RCLONE_DST_ENV"
fi

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

head_root_for_task() {
  if [[ "$1" == "stack_cups" && -d "${STACK_HEAD_ROOT}/stack_cups" ]]; then
    echo "$STACK_HEAD_ROOT"
  else
    echo "$HEAD_ROOT"
  fi
}

selected_count() {
  python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print(0)
    raise SystemExit
payload = json.loads(p.read_text())
count = 0
for item in payload.get("videos", {}).values():
    status = str(item.get("status", "")).lower()
    label = item.get("label")
    usable = item.get("usable")
    if status in {"reject", "discard", "bad"} or label == "n" or usable is False:
        continue
    if status in {"usable", "good", "accept", "accepted"} or label == "y" or usable is True:
        count += 1
print(count)
PY
}

subset_episodes_from_processed() {
  python3 - "$1" "$2" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
limit = int(sys.argv[2])
episodes = []
for p in sorted(root.glob("episode_*/instructions.json"), key=lambda p: int(p.parent.name.split("_")[-1])):
    episodes.append(int(p.parent.name.split("_")[-1]))
    if len(episodes) >= limit:
        break
if not episodes:
    raise SystemExit(f"no processed episodes under {root}")
print(",".join(map(str, episodes)))
PY
}

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh

for TASK in $TASKS; do
  REVIEW_JSON="${REVIEW_ROOT}/selections/${TASK}/ours_review_selection.json"
  COUNT=$(selected_count "$REVIEW_JSON")
  if [[ "$COUNT" -le 0 ]]; then
    echo "[skip] ${TASK}: no accepted rows in ${REVIEW_JSON}"
    continue
  fi

  INSTRUCTION=$(instruction_for_task "$TASK")
  TASK_HEAD_ROOT=$(head_root_for_task "$TASK")
  DATASET="${PI0_ROOT}/processed_data/h2o_${TASK}_${DATASET_SUFFIX}-${N}"
  SOURCE="local/h2o_${TASK}_${DATASET_SUFFIX}"
  SUBSET_REPO="local/h2o_${TASK}_${DATASET_SUFFIX}_${SUBSET_N}ep"
  PIPER0515_REPO="local/h2o_${TASK}_${DATASET_SUFFIX}_${PIPER0515_SUFFIX}_${SUBSET_N}ep"

  if has_step process; then
    echo "===== process ${TASK}: selected=${COUNT} dataset=${DATASET} ====="
    conda activate RoboTwin_bw
    cd "$PI0_ROOT"
    python scripts/process_repainted_planner_outputs.py "h2o_${TASK}_${DATASET_SUFFIX}" "$INSTRUCTION" "$N" \
      --head-root "${TASK_HEAD_ROOT}/${TASK}" \
      --head-dir-template 'id_{id}_l16_whitebg_human_object' \
      --head-video-name final_repainted.mp4 \
      --planner-root "${PLANNER_ROOT}/${TASK}" \
      --planner-dir-template 'foundation_input_{id}' \
      --left-wrist-video-name left_wrist_cam_plan.mp4 \
      --right-wrist-video-name right_wrist_cam_plan.mp4 \
      --pose-debug-name pose_debug.jsonl \
      --review-json "$REVIEW_JSON" \
      --review-mode strict \
      --output-dir "$DATASET"
  fi

  if has_step lerobot; then
    echo "===== lerobot ${TASK}: source=${SOURCE} ====="
    conda activate RoboTwin_openvla
    cd "$PI0_ROOT"
    if [[ ! -d "$DATASET" ]]; then
      echo "[skip] missing processed dataset ${DATASET}"
      continue
    fi
    uv run examples/aloha_real/convert_aloha_data_to_lerobot_R1.py \
      --raw-dir "$DATASET" \
      --repo-id "$SOURCE" \
      --task "$INSTRUCTION" \
      --use-wrist \
      --mode video
  fi

  if has_step subset; then
    echo "===== subset ${TASK}: output=${SUBSET_REPO} ====="
    conda activate RoboTwin_openvla
    cd "$PI0_ROOT"
    if [[ ! -d "$DATASET" ]]; then
      echo "[skip] missing processed dataset ${DATASET}"
      continue
    fi
    TAKE_N=$SUBSET_N
    if [[ "$COUNT" -lt "$SUBSET_N" ]]; then
      TAKE_N=$COUNT
      echo "[warn] ${TASK}: only ${COUNT} accepted rows; subset will contain ${TAKE_N} episodes"
    fi
    EPISODES=$(subset_episodes_from_processed "$DATASET" "$TAKE_N")
    uv run python scripts/subset_lerobot_episodes.py \
      --source "$SOURCE" \
      --output-repo-id "$SUBSET_REPO" \
      --episodes "$EPISODES" \
      --overwrite
  fi

  if has_step piper0515; then
    echo "===== piper0515 ${TASK}: output=${PIPER0515_REPO} ====="
    conda activate simplevla-rl
    cd /home/zaijia001/ssd/RoboTwin
    python code_painting/convert_lerobot_piper0515_world_to_base.py \
      --source "$SUBSET_REPO" \
      --output-repo-id "$PIPER0515_REPO" \
      --robot-config "$PIPER0515_ROBOT_CONFIG" \
      --gripper-scale "$PIPER0515_GRIPPER_SCALE" \
      --overwrite
  fi
done

if has_step zip; then
  echo "===== zip/rclone ${ZIP_NAME} ====="
  cd "$LEROBOT_LOCAL"
  ZIP_DIRS=()
  for TASK in $TASKS; do
    if has_step piper0515; then
      D="h2o_${TASK}_${DATASET_SUFFIX}_${PIPER0515_SUFFIX}_${SUBSET_N}ep"
    else
      D="h2o_${TASK}_${DATASET_SUFFIX}_${SUBSET_N}ep"
    fi
    if [[ -d "$D" ]]; then
      ZIP_DIRS+=("$D")
    else
      echo "[zip skip] missing $D"
    fi
  done
  if [[ "${#ZIP_DIRS[@]}" -eq 0 ]]; then
    echo "[zip skip] no subset dirs found"
    exit 0
  fi
  zip -r "$ZIP_NAME" "${ZIP_DIRS[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    rclone copy "${LEROBOT_LOCAL}/${ZIP_NAME}" "$RCLONE_DST" -P --drive-chunk-size 64M --transfers 4 --dry-run
  else
    rclone copy "${LEROBOT_LOCAL}/${ZIP_NAME}" "$RCLONE_DST" -P --drive-chunk-size 64M --transfers 4
  fi
fi
