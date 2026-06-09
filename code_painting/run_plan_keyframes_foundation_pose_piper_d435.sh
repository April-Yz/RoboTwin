#!/usr/bin/env bash
set -euo pipefail
# Mode K: Foundation Pose Target — object position + hand orientation as IK target.
#
# Usage:
#   bash run_plan_keyframes_foundation_pose_piper_d435.sh --gpu 2 --ids 0 --viewer --tasks stack_cups
#   bash run_plan_keyframes_foundation_pose_piper_d435.sh --gpu 2 --ids 0 --viewer --tasks pick_diverse_bottles --foundation_pose_retreat_m 0.05

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
HAND_KEYFRAMES_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review

GPU=2
VIEWER=0
VIEWER_WAIT_AT_END=0
CONTINUE_ON_ERROR=0
DRY_RUN=0
OUTPUT_ROOT=""
TASK=""
IDS=()
ID_START=""
ID_END=""
FOUNDATION_POSE_RETREAT_M=0.03
TRAJECTORY_MODE=cartesian_interp_ik
CARTESIAN_AUTO_STEP_M=0.03
IK_MAX_ROTATION_THRESHOLD_RAD=3.14
APPROACH_OFFSET_M=0.12
REPLAN_MAX_ATTEMPTS=3
VISUALIZE_TARGETS=1
DISABLE_EXECUTION_COLLISIONS=1
PURE_SCENE_OUTPUT=1
DEBUG_CANDIDATE_TOP_K=1
DEBUG_VISUALIZE_CAMERAS=0
VIEWER_SHOW_CAMERA_FRUSTUMS=0

usage() {
  cat <<'EOF'
Usage: run_plan_keyframes_foundation_pose_piper_d435.sh [OPTIONS]

Mode K (Ablation): Foundation Pose Target — object position + hand orientation.
Foundation object localization provides the WHERE, human hand provides the HOW.

Required:
  --tasks <TASK>        Task name
  --ids <ID> [<ID>...]  Episode ID(s), or range like \"0-10\"

Key parameter:
  --foundation_pose_retreat_m <M>  Retreat along gripper +Z from object center
                                   to grasp surface (default: 0.03 = 3cm)

Options:
  --gpu <N>             GPU index (default: 2)
  --viewer              Enable viewer mode
  --viewer_wait_at_end <0|1>
  --continue_on_error   Skip failed IDs
  --dry_run             Print commands only
  --output_root <PATH>  Custom output root
  --id_start <N>        Start of ID range (inclusive, requires --id_end)
  --id_end <N>          End of ID range (inclusive, requires --id_start)
  --approach_offset_m <M>  Pregrasp approach distance (default: 0.12)
  --trajectory_mode <M>    Trajectory mode (default: cartesian_interp_ik)
  --cartesian_auto_step_m <M>
  --ik_max_rotation_threshold_rad <RAD>
  --debug_viewer_overlay  Show Mode N target axes and top-1 C-gripper actors in viewer/videos
  --debug_visualize_cameras <0|1>
  --viewer_show_camera_frustums <0|1>
  --pure_scene_output <0|1>
  --debug_candidate_top_k <N>
EOF
  exit 0
}

while (($# > 0)); do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --viewer) VIEWER=1; shift ;;
    --viewer_wait_at_end) VIEWER_WAIT_AT_END="$2"; shift 2 ;;
    --continue_on_error) CONTINUE_ON_ERROR=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --foundation_pose_retreat_m) FOUNDATION_POSE_RETREAT_M="$2"; shift 2 ;;
    --approach_offset_m) APPROACH_OFFSET_M="$2"; shift 2 ;;
    --replan_until_reached_max_attempts) REPLAN_MAX_ATTEMPTS="$2"; shift 2 ;;
    --trajectory_mode) TRAJECTORY_MODE="$2"; shift 2 ;;
    --cartesian_auto_step_m) CARTESIAN_AUTO_STEP_M="$2"; shift 2 ;;
    --ik_max_rotation_threshold_rad) IK_MAX_ROTATION_THRESHOLD_RAD="$2"; shift 2 ;;
    --visualize_targets) VISUALIZE_TARGETS=1; shift ;;
    --disable_execution_collisions) DISABLE_EXECUTION_COLLISIONS=1; shift ;;
    --debug_viewer_overlay) PURE_SCENE_OUTPUT=0; VISUALIZE_TARGETS=1; DEBUG_CANDIDATE_TOP_K=1; DEBUG_VISUALIZE_CAMERAS=1; VIEWER_SHOW_CAMERA_FRUSTUMS=1; shift ;;
    --debug_visualize_cameras) DEBUG_VISUALIZE_CAMERAS="$2"; shift 2 ;;
    --viewer_show_camera_frustums) VIEWER_SHOW_CAMERA_FRUSTUMS="$2"; shift 2 ;;
    --pure_scene_output) PURE_SCENE_OUTPUT="$2"; shift 2 ;;
    --debug_candidate_top_k) DEBUG_CANDIDATE_TOP_K="$2"; shift 2 ;;
    --tasks) shift; TASK="$1"; shift ;;
    --id_start) ID_START="$2"; shift 2 ;;
    --id_end) ID_END="$2"; shift 2 ;;
    --ids)
      shift
      IDS=()
      while (($# > 0)); do
        if [[ "$1" == --* ]]; then break; fi
        # Support range format like "0-10"
        if [[ "$1" =~ ^([0-9]+)-([0-9]+)$ ]]; then
          for ((_id=${BASH_REMATCH[1]}; _id<=${BASH_REMATCH[2]}; _id++)); do
            IDS+=("$_id")
          done
        else
          IDS+=("$1")
        fi
        shift
      done
      ;;
    --help|-h) usage ;;
    *) echo "[error] unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$TASK" ]] && { echo "[error] --tasks required"; exit 2; }

# Expand --id_start/--id_end into IDS array
if [[ -n "$ID_START" && -n "$ID_END" ]]; then
  for ((_id=ID_START; _id<=ID_END; _id++)); do
    IDS+=("$_id")
  done
fi

((${#IDS[@]} == 0)) && { echo "[error] --ids or --id_start/--id_end required"; exit 2; }

resolve_task_config() {
  local task="$1"
  MESH_ARGS=()
  case "$task" in
    pick_diverse_bottles)
      MESH_ARGS=(--object_mesh_override left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj) ;;
    place_bread_basket)
      MESH_ARGS=(--object_mesh_override basket=/home/zaijia001/ssd/data/R1/hand/obj_mesh/basket/basket.obj --object_mesh_override bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_y/bread_y.obj) ;;
    stack_cups)
      MESH_ARGS=(--object_mesh_override left_light_pink_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/light_pink_cup/light_pink_cup.obj --object_mesh_override right_dark_red_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj) ;;
    handover_bottle)
      MESH_ARGS=(--object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj) ;;
    pnp_bread)
      MESH_ARGS=(--object_mesh_override left_bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_nj/bread_niujiao.obj --object_mesh_override right_bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_yr/bread_yerong.obj) ;;
    pnp_tray)
      MESH_ARGS=(--object_mesh_override left_dark_red_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj) ;;
    *) echo "[error] unknown task: $task" >&2; exit 2 ;;
  esac
}

resolve_task_config "$TASK"

ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output
[[ -d "$ANY_ROOT" ]] || ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output_old_cam
HAND_KEYFRAMES_JSON=${HAND_KEYFRAMES_ROOT}/${TASK}/hand_keyframes_all.json

if [[ -n "$OUTPUT_ROOT" ]]; then
  OUT_ROOT=${OUTPUT_ROOT}/${TASK}
elif ((VIEWER)); then
  OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/foundation_pose_viewer/${TASK}
else
  OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/foundation_pose/${TASK}
fi

echo "===== Mode K: Foundation Pose Target task=${TASK} ids=${IDS[*]} viewer=${VIEWER} retreat=${FOUNDATION_POSE_RETREAT_M}m ====="

for ID in "${IDS[@]}"; do
  ANY=${ANY_ROOT}/foundation_input_${ID}
  REPLAY=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435/foundation_input_${ID}
  HAND=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz
  OUT=${OUT_ROOT}/foundation_input_${ID}

  [[ -d "$ANY" ]] || { echo "[skip] missing anygrasp $ANY"; continue; }
  [[ -d "$REPLAY" ]] || { echo "[skip] missing D435 replay $REPLAY"; continue; }
  [[ -f "$HAND" ]] || { echo "[skip] missing hand $HAND"; continue; }
  [[ -f "$HAND_KEYFRAMES_JSON" ]] || { echo "[skip] missing keyframes json"; continue; }

  echo "[run] task=${TASK} id=${ID}"
  if ((DRY_RUN)); then echo "  [dry-run]"; continue; fi

  RUN_ENV=(env CUDA_VISIBLE_DEVICES=${GPU})
  if ((VIEWER)); then
    [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
    RUN_ENV=(env -u CUDA_VISIBLE_DEVICES)
  fi

  K_ARGS=(
    --anygrasp_dir "$ANY" --replay_dir "$REPLAY" --hand_npz "$HAND"
    --output_dir "$OUT" --hand_keyframes_json "$HAND_KEYFRAMES_JSON"
    --video_id "$ID" --task "$TASK" --gpu "$GPU"
    --foundation_pose_retreat_m "$FOUNDATION_POSE_RETREAT_M"
    --urdfik_trajectory_mode "$TRAJECTORY_MODE"
    --urdfik_cartesian_interp_auto_step_m "$CARTESIAN_AUTO_STEP_M"
    --urdfik_max_rotation_threshold_rad "$IK_MAX_ROTATION_THRESHOLD_RAD"
    --approach_offset_m "$APPROACH_OFFSET_M"
    --replan_until_reached_max_attempts "$REPLAN_MAX_ATTEMPTS"
    --pure_scene_output "$PURE_SCENE_OUTPUT"
    --debug_candidate_top_k "$DEBUG_CANDIDATE_TOP_K"
    --debug_visualize_cameras "$DEBUG_VISUALIZE_CAMERAS"
    --viewer_show_camera_frustums "$VIEWER_SHOW_CAMERA_FRUSTUMS"
  )

  if ((VIEWER)); then
    K_ARGS+=(--enable_viewer 1 --viewer_wait_at_end "$VIEWER_WAIT_AT_END")
  fi
  if ((VISUALIZE_TARGETS)); then K_ARGS+=(--debug_visualize_targets 1); fi
  if ((DISABLE_EXECUTION_COLLISIONS)); then K_ARGS+=(--enable_grasp_action_object_collision 0); fi
  K_ARGS+=("${MESH_ARGS[@]}")

  echo "[start] task=${TASK} id=${ID} output=${OUT}"
  mkdir -p "$OUT"
  PYTHON_BIN=/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python
  if ! "${RUN_ENV[@]}" "$PYTHON_BIN" -u \
    /home/zaijia001/ssd/RoboTwin/code_painting/plan_keyframes_foundation_pose.py \
    "${K_ARGS[@]}" 2>"${OUT}/stderr.log"; then
    echo "[stderr-tail] $(tail -5 "${OUT}/stderr.log" 2>/dev/null)"
    if ((CONTINUE_ON_ERROR)); then echo "[error-continue] id=${ID}"; continue; fi
    exit 1
  fi
  echo "[done] task=${TASK} id=${ID}"
done
