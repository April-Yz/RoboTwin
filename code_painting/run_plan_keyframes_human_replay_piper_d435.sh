#!/usr/bin/env bash
set -euo pipefail
# Mode M: Human Replay Target - use hand gripper pose directly as IK target (no AnyGrasp).
#
# Usage:
#   bash run_plan_keyframes_human_replay_piper_d435.sh --gpu 2 --ids 0 --viewer --tasks stack_cups
#   bash run_plan_keyframes_human_replay_piper_d435.sh --gpu 2 --ids 0 --viewer --tasks pick_diverse_bottles --output_root /path/to/output

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
TRAJECTORY_MODE=joint_interp
CARTESIAN_AUTO_STEP_M=0.03
IK_MAX_ROTATION_THRESHOLD_RAD=3.14
IK_NUM_SEEDS=1
IK_SOLUTION_SELECTION=joint_continuity
IK_SEED_PERTURBATIONS=6
IK_SEED_PERTURBATION_SCALE=0.05
IK_MAX_JOINT_STEP_RAD=0
APPLY_GLOBAL_TRANS_TO_IK=0
EXECUTE_PARTIAL_CARTESIAN_PLAN=0
JOINT_TRAJECTORY_INTERPOLATION=cubic
ACTION_ORIENTATION_SOURCE=grasp
DUAL_STAGE_FREEZE_REACHED_ARMS_ON_REPLAN=1
REACH_ROT_TOL_DEG=180
REACH_POS_TOL_M=0.04
FAIL_ON_EXECUTION_FAILURE=1
APPROACH_OFFSET_M=0.12
TARGET_RETREAT_M=0.0
PIPER_CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json
# Wrist camera tuning (same convention as O.1)
WRIST_LEFT_FORWARD_OFFSET_M=0.0
WRIST_RIGHT_FORWARD_OFFSET_M=0.0
WRIST_LEFT_LATERAL_OFFSET_M=0.0
WRIST_RIGHT_LATERAL_OFFSET_M=0.0
WRIST_LEFT_ROLL_DEG=0.0
WRIST_RIGHT_ROLL_DEG=0.0
WRIST_LEFT_YAW_DEG=0.0
WRIST_RIGHT_YAW_DEG=0.0
WRIST_LEFT_PITCH_DEG=0.0
WRIST_RIGHT_PITCH_DEG=0.0
REPLAN_MAX_ATTEMPTS=5
VISUALIZE_TARGETS=1
DISABLE_EXECUTION_COLLISIONS=1

usage() {
  cat <<'EOF'
Usage: run_plan_keyframes_human_replay_piper_d435.sh [OPTIONS]

Mode M (Ablation): Human Replay Target - use hand gripper pose as IK target.
No AnyGrasp candidates. Same planner pipeline otherwise.

Required:
  --tasks <TASK>        Task name (e.g. stack_cups, pick_diverse_bottles)
  --ids <ID> [<ID>...]  Episode ID(s), or range like \"0-10\"

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
  --target_retreat_m <M>  Offset grasp target backward along approach axis to convert hand TCP to link6 target (default: 0.0; set to gripper_bias e.g. 0.12 for Piper)
  --trajectory_mode <M>    Trajectory mode (default: joint_interp)
  --cartesian_auto_step_m <M>
  --ik_max_rotation_threshold_rad <RAD>
  --ik_num_seeds <N>
  --ik_solution_selection <pose_error|joint_continuity>
  --ik_seed_perturbations <N>
  --ik_seed_perturbation_scale <RAD>
  --ik_max_joint_step_rad <RAD>
  --apply_global_trans_to_ik <0|1>
  --execute_partial_cartesian_plan <0|1>
  --joint_trajectory_interpolation <linear|cubic>
  --action_orientation_source <keyframe|grasp>
  --dual_stage_freeze_reached_arms_on_replan <0|1>
  --reach_rot_tol_deg <DEG>
  --reach_pos_tol_m <M>
  --fail_on_execution_failure <0|1>
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
    --approach_offset_m) APPROACH_OFFSET_M="$2"; shift 2 ;;
    --target_retreat_m) TARGET_RETREAT_M="$2"; shift 2 ;;
    --piper_calibration_bundle) PIPER_CALIBRATION_BUNDLE="$2"; shift 2 ;;
    --wrist_left_forward_offset_m) WRIST_LEFT_FORWARD_OFFSET_M="$2"; shift 2 ;;
    --wrist_right_forward_offset_m) WRIST_RIGHT_FORWARD_OFFSET_M="$2"; shift 2 ;;
    --wrist_left_lateral_offset_m) WRIST_LEFT_LATERAL_OFFSET_M="$2"; shift 2 ;;
    --wrist_right_lateral_offset_m) WRIST_RIGHT_LATERAL_OFFSET_M="$2"; shift 2 ;;
    --wrist_left_roll_deg) WRIST_LEFT_ROLL_DEG="$2"; shift 2 ;;
    --wrist_right_roll_deg) WRIST_RIGHT_ROLL_DEG="$2"; shift 2 ;;
    --wrist_left_yaw_deg) WRIST_LEFT_YAW_DEG="$2"; shift 2 ;;
    --wrist_right_yaw_deg) WRIST_RIGHT_YAW_DEG="$2"; shift 2 ;;
    --wrist_left_pitch_deg) WRIST_LEFT_PITCH_DEG="$2"; shift 2 ;;
    --wrist_right_pitch_deg) WRIST_RIGHT_PITCH_DEG="$2"; shift 2 ;;
    --replan_until_reached_max_attempts) REPLAN_MAX_ATTEMPTS="$2"; shift 2 ;;
    --trajectory_mode) TRAJECTORY_MODE="$2"; shift 2 ;;
    --cartesian_auto_step_m) CARTESIAN_AUTO_STEP_M="$2"; shift 2 ;;
    --ik_max_rotation_threshold_rad) IK_MAX_ROTATION_THRESHOLD_RAD="$2"; shift 2 ;;
    --ik_num_seeds) IK_NUM_SEEDS="$2"; shift 2 ;;
    --ik_solution_selection) IK_SOLUTION_SELECTION="$2"; shift 2 ;;
    --ik_seed_perturbations) IK_SEED_PERTURBATIONS="$2"; shift 2 ;;
    --ik_seed_perturbation_scale) IK_SEED_PERTURBATION_SCALE="$2"; shift 2 ;;
    --ik_max_joint_step_rad) IK_MAX_JOINT_STEP_RAD="$2"; shift 2 ;;
    --apply_global_trans_to_ik) APPLY_GLOBAL_TRANS_TO_IK="$2"; shift 2 ;;
    --execute_partial_cartesian_plan) EXECUTE_PARTIAL_CARTESIAN_PLAN="$2"; shift 2 ;;
    --joint_trajectory_interpolation) JOINT_TRAJECTORY_INTERPOLATION="$2"; shift 2 ;;
    --action_orientation_source) ACTION_ORIENTATION_SOURCE="$2"; shift 2 ;;
    --dual_stage_freeze_reached_arms_on_replan) DUAL_STAGE_FREEZE_REACHED_ARMS_ON_REPLAN="$2"; shift 2 ;;
    --reach_rot_tol_deg) REACH_ROT_TOL_DEG="$2"; shift 2 ;;
    --reach_pos_tol_m) REACH_POS_TOL_M="$2"; shift 2 ;;
    --fail_on_execution_failure) FAIL_ON_EXECUTION_FAILURE="$2"; shift 2 ;;
    --visualize_targets) VISUALIZE_TARGETS=1; shift ;;
    --disable_execution_collisions) DISABLE_EXECUTION_COLLISIONS=1; shift ;;
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

# Resolve task config
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
  OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/human_replay_viewer/${TASK}
else
  OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/human_replay/${TASK}
fi

echo "===== Mode M: Human Replay Target task=${TASK} ids=${IDS[*]} viewer=${VIEWER} ====="

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

  M_ARGS=(
    --anygrasp_dir "$ANY" --replay_dir "$REPLAY" --hand_npz "$HAND"
    --output_dir "$OUT" --hand_keyframes_json "$HAND_KEYFRAMES_JSON"
    --video_id "$ID" --task "$TASK" --gpu "$GPU"
    --urdfik_trajectory_mode "$TRAJECTORY_MODE"
    --urdfik_cartesian_interp_auto_step_m "$CARTESIAN_AUTO_STEP_M"
    --urdfik_max_rotation_threshold_rad "$IK_MAX_ROTATION_THRESHOLD_RAD"
    --urdfik_num_seeds "$IK_NUM_SEEDS"
    --urdfik_solution_selection "$IK_SOLUTION_SELECTION"
    --urdfik_seed_perturbations "$IK_SEED_PERTURBATIONS"
    --urdfik_seed_perturbation_scale "$IK_SEED_PERTURBATION_SCALE"
    --urdfik_max_joint_step_rad "$IK_MAX_JOINT_STEP_RAD"
    --piper_urdfik_apply_global_trans_to_ik "$APPLY_GLOBAL_TRANS_TO_IK"
    --execute_partial_cartesian_plan "$EXECUTE_PARTIAL_CARTESIAN_PLAN"
    --joint_trajectory_interpolation "$JOINT_TRAJECTORY_INTERPOLATION"
    --action_orientation_source "$ACTION_ORIENTATION_SOURCE"
    --dual_stage_freeze_reached_arms_on_replan "$DUAL_STAGE_FREEZE_REACHED_ARMS_ON_REPLAN"
    --reach_rot_tol_deg "$REACH_ROT_TOL_DEG"
    --reach_pos_tol_m "$REACH_POS_TOL_M"
    --fail_on_execution_failure "$FAIL_ON_EXECUTION_FAILURE"
    --wrist_preview 1
    --viewer_show_camera_frustums 1
    --debug_visualize_cameras 1 --debug_camera_axis_length 0.10
    --approach_offset_m "$APPROACH_OFFSET_M"
    --piper_calibration_bundle "$PIPER_CALIBRATION_BUNDLE"
    --target_retreat_m "$TARGET_RETREAT_M"
    --wrist_left_forward_offset_m "$WRIST_LEFT_FORWARD_OFFSET_M"
    --wrist_right_forward_offset_m "$WRIST_RIGHT_FORWARD_OFFSET_M"
    --wrist_left_lateral_offset_m "$WRIST_LEFT_LATERAL_OFFSET_M"
    --wrist_right_lateral_offset_m "$WRIST_RIGHT_LATERAL_OFFSET_M"
    --wrist_left_roll_deg "$WRIST_LEFT_ROLL_DEG"
    --wrist_right_roll_deg "$WRIST_RIGHT_ROLL_DEG"
    --wrist_left_yaw_deg "$WRIST_LEFT_YAW_DEG"
    --wrist_right_yaw_deg "$WRIST_RIGHT_YAW_DEG"
    --wrist_left_pitch_deg "$WRIST_LEFT_PITCH_DEG"
    --wrist_right_pitch_deg "$WRIST_RIGHT_PITCH_DEG"
    --replan_until_reached_max_attempts "$REPLAN_MAX_ATTEMPTS"
  )

  if ((VIEWER)); then
    M_ARGS+=(--enable_viewer 1 --viewer_wait_at_end "$VIEWER_WAIT_AT_END")
  fi
  if ((VISUALIZE_TARGETS)); then M_ARGS+=(--debug_visualize_targets 1); fi
  if ((DISABLE_EXECUTION_COLLISIONS)); then M_ARGS+=(--enable_grasp_action_object_collision 0); fi
  M_ARGS+=("${MESH_ARGS[@]}")

  echo "[start] task=${TASK} id=${ID} output=${OUT}"
  mkdir -p "$OUT"
  PYTHON_BIN=/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python
  if ! "${RUN_ENV[@]}" "$PYTHON_BIN" -u \
    /home/zaijia001/ssd/RoboTwin/code_painting/plan_keyframes_human_replay.py \
    "${M_ARGS[@]}" 2>"${OUT}/stderr.log"; then
    echo "[stderr-tail] $(tail -5 "${OUT}/stderr.log" 2>/dev/null)"
    if ((CONTINUE_ON_ERROR)); then echo "[error-continue] id=${ID}"; continue; fi
    exit 1
  fi
  echo "[done] task=${TASK} id=${ID}"
done
