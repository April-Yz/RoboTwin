#!/usr/bin/env bash
set -euo pipefail

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
CONDA_BIN=/home/zaijia001/ssd/miniconda3/bin/conda
cd /home/zaijia001/ssd/RoboTwin

GPU=2
MAX_PER_TASK=0
DRY_RUN=0
CONTINUE_ON_ERROR=0
VIEWER=0
VIEWER_WAIT_AT_END=0
DEBUG_STOP_AFTER_KEYFRAME1=0
OUTPUT_ROOT=""
PREVIEW_ROOT_BASE=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435
TRAJECTORY_MODE=cartesian_interp_ik
CARTESIAN_AUTO_STEP_M=0.01
JOINT_INTERP_WAYPOINTS=40
REPLAN_ATTEMPTS=1
DUAL_STAGE_REQUIRE_ALL_PLANS=1
EXECUTE_INTERP_STEPS=24
JOINT_COMMAND_SCENE_STEPS=10
SETTLE_STEPS=30
JOINT_TARGET_WAIT_STEPS=25
PRINT_POSE_EVERY=0
REACH_ERROR_POSE_SOURCE=ee
VISUALIZE_TARGETS=0
TARGET_AXES_ONLY=0
PURE_SCENE_OUTPUT=1
EXECUTE_PARTIAL_CARTESIAN_PLAN=0
IK_MAX_POSITION_THRESHOLD_M=0.02
IK_MAX_ROTATION_THRESHOLD_RAD=0.12
PIPER_APPLY_GLOBAL_TRANS_TO_IK=0
CANDIDATE_ORIENTATION_REMAP_LABEL=identity
CANDIDATE_SELECTION_MODE=planner
CANDIDATE_MAX_ROTATION_DISTANCE_DEG=-1.0
CANDIDATE_KEEP_CAMERA_UP=0
ENFORCE_CANDIDATE_DISTANCE_CONSTRAINT=1
CANDIDATE_TARGET_LOCAL_X_OFFSET_M=-0.05
CANDIDATE_TARGET_LOCAL_Z_OFFSET_M=0.0
APPROACH_AXIS=local_x
APPROACH_OFFSET_M=0.12
DEBUG_GRIPPER_ACTOR_FORWARD_AXIS=local_x
ENABLE_EXECUTION_COLLISIONS=1
DEBUG_CANDIDATE_TOP_K=5
DEBUG_COMMON_CANDIDATE_TOP_K=0
DEBUG_VISUALIZE_SELECTED_KEYFRAME_AXES=1
DEBUG_VISUALIZE_IK_WAYPOINTS=1
PIPER_CALIBRATION_BUNDLE=""
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
IDS_FILTER=()
ID_START=""
ID_END=""
TASKS=(pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray)

while (($# > 0)); do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --max_per_task)
      MAX_PER_TASK="$2"
      shift 2
      ;;
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    --continue_on_error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --viewer)
      VIEWER=1
      shift
      ;;
    --viewer_wait_at_end)
      VIEWER_WAIT_AT_END="$2"
      shift 2
      ;;
    --debug_stop_after_keyframe1)
      DEBUG_STOP_AFTER_KEYFRAME1=1
      shift
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --preview_root)
      PREVIEW_ROOT_BASE="$2"
      shift 2
      ;;
    --trajectory_mode)
      TRAJECTORY_MODE="$2"
      shift 2
      ;;
    --cartesian_auto_step_m)
      CARTESIAN_AUTO_STEP_M="$2"
      shift 2
      ;;
    --joint_interp_waypoints)
      JOINT_INTERP_WAYPOINTS="$2"
      shift 2
      ;;
    --replan_attempts)
      REPLAN_ATTEMPTS="$2"
      shift 2
      ;;
    --allow_partial_dual_stage)
      DUAL_STAGE_REQUIRE_ALL_PLANS=0
      shift
      ;;
    --execute_interp_steps)
      EXECUTE_INTERP_STEPS="$2"
      shift 2
      ;;
    --joint_command_scene_steps)
      JOINT_COMMAND_SCENE_STEPS="$2"
      shift 2
      ;;
    --settle_steps)
      SETTLE_STEPS="$2"
      shift 2
      ;;
    --joint_target_wait_steps)
      JOINT_TARGET_WAIT_STEPS="$2"
      shift 2
      ;;
    --print_pose_every)
      PRINT_POSE_EVERY="$2"
      shift 2
      ;;
    --reach_error_pose_source)
      REACH_ERROR_POSE_SOURCE="$2"
      shift 2
      ;;
    --visualize_targets)
      VISUALIZE_TARGETS=1
      PURE_SCENE_OUTPUT=0
      shift
      ;;
    --target_axes_only)
      TARGET_AXES_ONLY=1
      VISUALIZE_TARGETS=1
      PURE_SCENE_OUTPUT=0
      DEBUG_CANDIDATE_TOP_K=0
      DEBUG_COMMON_CANDIDATE_TOP_K=0
      DEBUG_VISUALIZE_SELECTED_KEYFRAME_AXES=0
      DEBUG_VISUALIZE_IK_WAYPOINTS=0
      shift
      ;;
    --execute_partial_cartesian_plan)
      EXECUTE_PARTIAL_CARTESIAN_PLAN=1
      shift
      ;;
    --disable_execution_collisions)
      ENABLE_EXECUTION_COLLISIONS=0
      shift
      ;;
    --debug_candidate_top_k)
      DEBUG_CANDIDATE_TOP_K="$2"
      shift 2
      ;;
    --debug_common_candidate_top_k)
      DEBUG_COMMON_CANDIDATE_TOP_K="$2"
      shift 2
      ;;
    --debug_visualize_selected_keyframe_axes)
      DEBUG_VISUALIZE_SELECTED_KEYFRAME_AXES="$2"
      shift 2
      ;;
    --debug_visualize_ik_waypoints)
      DEBUG_VISUALIZE_IK_WAYPOINTS="$2"
      shift 2
      ;;
    --ik_max_position_threshold_m)
      IK_MAX_POSITION_THRESHOLD_M="$2"
      shift 2
      ;;
    --ik_max_rotation_threshold_rad)
      IK_MAX_ROTATION_THRESHOLD_RAD="$2"
      shift 2
      ;;
    --piper_apply_global_trans_to_ik)
      PIPER_APPLY_GLOBAL_TRANS_TO_IK="$2"
      shift 2
      ;;
    --candidate_orientation_remap_label)
      CANDIDATE_ORIENTATION_REMAP_LABEL="$2"
      shift 2
      ;;
    --candidate_selection_mode)
      CANDIDATE_SELECTION_MODE="$2"
      shift 2
      ;;
    --candidate_max_rotation_distance_deg)
      CANDIDATE_MAX_ROTATION_DISTANCE_DEG="$2"
      shift 2
      ;;
    --candidate_keep_camera_up)
      CANDIDATE_KEEP_CAMERA_UP="$2"
      shift 2
      ;;
    --enforce_candidate_distance_constraint)
      ENFORCE_CANDIDATE_DISTANCE_CONSTRAINT="$2"
      shift 2
      ;;
    --candidate_target_local_x_offset_m)
      CANDIDATE_TARGET_LOCAL_X_OFFSET_M="$2"
      shift 2
      ;;
    --candidate_target_local_z_offset_m)
      CANDIDATE_TARGET_LOCAL_Z_OFFSET_M="$2"
      shift 2
      ;;
    --approach_axis)
      APPROACH_AXIS="$2"
      shift 2
      ;;
    --approach_offset_m)
      APPROACH_OFFSET_M="$2"
      shift 2
      ;;
    --debug_gripper_actor_forward_axis)
      DEBUG_GRIPPER_ACTOR_FORWARD_AXIS="$2"
      shift 2
      ;;
    --piper_calibration_bundle)
      PIPER_CALIBRATION_BUNDLE="$2"
      shift 2
      ;;
    --wrist_left_forward_offset_m)
      WRIST_LEFT_FORWARD_OFFSET_M="$2"
      shift 2
      ;;
    --wrist_right_forward_offset_m)
      WRIST_RIGHT_FORWARD_OFFSET_M="$2"
      shift 2
      ;;
    --wrist_left_lateral_offset_m)
      WRIST_LEFT_LATERAL_OFFSET_M="$2"
      shift 2
      ;;
    --wrist_right_lateral_offset_m)
      WRIST_RIGHT_LATERAL_OFFSET_M="$2"
      shift 2
      ;;
    --wrist_left_roll_deg)
      WRIST_LEFT_ROLL_DEG="$2"
      shift 2
      ;;
    --wrist_right_roll_deg)
      WRIST_RIGHT_ROLL_DEG="$2"
      shift 2
      ;;
    --wrist_left_yaw_deg)
      WRIST_LEFT_YAW_DEG="$2"
      shift 2
      ;;
    --wrist_right_yaw_deg)
      WRIST_RIGHT_YAW_DEG="$2"
      shift 2
      ;;
    --wrist_left_pitch_deg)
      WRIST_LEFT_PITCH_DEG="$2"
      shift 2
      ;;
    --wrist_right_pitch_deg)
      WRIST_RIGHT_PITCH_DEG="$2"
      shift 2
      ;;
    --ids)
      shift
      IDS_FILTER=()
      while (($# > 0)); do
        if [[ "$1" == --* ]]; then
          break
        fi
        IDS_FILTER+=("$1")
        shift
      done
      ;;
    --id_start)
      ID_START="$2"
      shift 2
      ;;
    --id_end)
      ID_END="$2"
      shift 2
      ;;
    --pure_scene_output)
      PURE_SCENE_OUTPUT="$2"
      shift 2
      ;;
    --tasks)
      shift
      TASKS=()
      while (($# > 0)); do
        if [[ "$1" == --* ]]; then
          break
        fi
        TASKS+=("$1")
        shift
      done
      ;;
    *)
      echo "[error] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

resolve_task_config() {
  local task="$1"
  LEFT_OBJ=""
  RIGHT_OBJ=""
  MESH_ARGS=()
  case "$task" in
    pick_diverse_bottles)
      LEFT_OBJ=left_bottle
      RIGHT_OBJ=right_bottle
      MESH_ARGS=(--object_mesh_override left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj)
      ;;
    place_bread_basket)
      LEFT_OBJ=basket
      RIGHT_OBJ=bread
      MESH_ARGS=(--object_mesh_override basket=/home/zaijia001/ssd/data/R1/hand/obj_mesh/basket/basket.obj --object_mesh_override bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_y/bread_y.obj)
      ;;
    stack_cups)
      LEFT_OBJ=left_light_pink_cup
      RIGHT_OBJ=right_dark_red_cup
      MESH_ARGS=(--object_mesh_override left_light_pink_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/light_pink_cup/light_pink_cup.obj --object_mesh_override right_dark_red_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj)
      ;;
    handover_bottle)
      LEFT_OBJ=right_bottle
      RIGHT_OBJ=right_bottle
      MESH_ARGS=(--object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj)
      ;;
    pnp_bread)
      LEFT_OBJ=left_bread
      RIGHT_OBJ=right_bread
      MESH_ARGS=(--object_mesh_override left_bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_nj/bread_niujiao.obj --object_mesh_override right_bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_yr/bread_yerong.obj)
      ;;
    pnp_tray)
      LEFT_OBJ=left_dark_red_cup
      RIGHT_OBJ=right_bottle
      MESH_ARGS=(--object_mesh_override left_dark_red_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj)
      ;;
    *)
      echo "[error] unknown task: $task" >&2
      exit 2
      ;;
  esac
}

for TASK in "${TASKS[@]}"; do
  resolve_task_config "$TASK"
  ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output
  [[ -d "$ANY_ROOT" ]] || ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output_old_cam
  if [[ -n "$OUTPUT_ROOT" ]]; then
    OUT_ROOT=${OUTPUT_ROOT}/${TASK}
  elif ((VIEWER)); then
    OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_viewer/${TASK}
  elif ((DEBUG_STOP_AFTER_KEYFRAME1)); then
    OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_keyframe1_debug/${TASK}
  else
    OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v1/${TASK}
  fi
  PREVIEW_ROOT=${PREVIEW_ROOT_BASE}/${TASK}
  mapfile -t IDS < <(
    find "$PREVIEW_ROOT" -maxdepth 2 -name summary.json 2>/dev/null \
      | sed -E 's#.*/foundation_input_([0-9]+)/summary.json#\1#' \
      | sort -n
  )
  if ((MAX_PER_TASK > 0 && ${#IDS[@]} > MAX_PER_TASK)); then
    IDS=("${IDS[@]:0:MAX_PER_TASK}")
  fi
  if ((${#IDS_FILTER[@]} > 0)); then
    FILTERED_IDS=()
    for want_id in "${IDS_FILTER[@]}"; do
      for have_id in "${IDS[@]}"; do
        if [[ "$have_id" == "$want_id" ]]; then
          FILTERED_IDS+=("$have_id")
          break
        fi
      done
    done
    IDS=("${FILTERED_IDS[@]}")
  fi
  if [[ -n "$ID_START" || -n "$ID_END" ]]; then
    start=${ID_START:-0}
    end=${ID_END:-999999}
    FILTERED_IDS=()
    for have_id in "${IDS[@]}"; do
      if ((have_id >= start && have_id <= end)); then
        FILTERED_IDS+=("$have_id")
      fi
    done
    IDS=("${FILTERED_IDS[@]}")
  fi
  echo "===== run D435 planner task=${TASK} summaries=${#IDS[@]} max_per_task=${MAX_PER_TASK} dry_run=${DRY_RUN} viewer=${VIEWER} debug_stop_after_keyframe1=${DEBUG_STOP_AFTER_KEYFRAME1} trajectory_mode=${TRAJECTORY_MODE} dual_require_all=${DUAL_STAGE_REQUIRE_ALL_PLANS} reach_pose=${REACH_ERROR_POSE_SOURCE} visualize_targets=${VISUALIZE_TARGETS} target_axes_only=${TARGET_AXES_ONLY} collisions=${ENABLE_EXECUTION_COLLISIONS} pure_scene=${PURE_SCENE_OUTPUT} partial_cartesian=${EXECUTE_PARTIAL_CARTESIAN_PLAN} ik_max_pos=${IK_MAX_POSITION_THRESHOLD_M} ik_max_rot=${IK_MAX_ROTATION_THRESHOLD_RAD} piper_global_trans_ik=${PIPER_APPLY_GLOBAL_TRANS_TO_IK} preview_root=${PREVIEW_ROOT_BASE} remap=${CANDIDATE_ORIENTATION_REMAP_LABEL} local_x_offset=${CANDIDATE_TARGET_LOCAL_X_OFFSET_M} local_z_offset=${CANDIDATE_TARGET_LOCAL_Z_OFFSET_M} approach_axis=${APPROACH_AXIS} approach_offset=${APPROACH_OFFSET_M} gripper_actor_forward=${DEBUG_GRIPPER_ACTOR_FORWARD_AXIS} exec_steps=${EXECUTE_INTERP_STEPS} scene_steps=${JOINT_COMMAND_SCENE_STEPS} ====="
  for ID in "${IDS[@]}"; do
    ANY=${ANY_ROOT}/foundation_input_${ID}
    REPLAY=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435/foundation_input_${ID}
    HAND=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz
    PREVIEW=${PREVIEW_ROOT}/foundation_input_${ID}/summary.json
    OUT=${OUT_ROOT}/foundation_input_${ID}
    [[ -d "$ANY" ]] || { echo "[skip] task=${TASK} id=${ID} missing anygrasp $ANY"; continue; }
    [[ -d "$REPLAY" ]] || { echo "[skip] task=${TASK} id=${ID} missing D435 replay $REPLAY"; continue; }
    [[ -f "$HAND" ]] || { echo "[skip] task=${TASK} id=${ID} missing hand $HAND"; continue; }
    [[ -f "$PREVIEW" ]] || { echo "[skip] task=${TASK} id=${ID} missing preview $PREVIEW"; continue; }
    echo "[run] task=${TASK} id=${ID} preview=${PREVIEW}"
    if ((DRY_RUN)); then
      continue
    fi
    RUN_ENV=(env CUDA_VISIBLE_DEVICES=${GPU})
    VIEWER_ARGS=(--enable_viewer 0 --viewer_wait_at_end 0 --viewer_show_camera_frustums 0)
    if ((VIEWER)); then
      [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
      RUN_ENV=(env -u CUDA_VISIBLE_DEVICES)
      VIEWER_ARGS=(--enable_viewer 1 --viewer_wait_at_end ${VIEWER_WAIT_AT_END} --viewer_frame_delay 0.02 --viewer_show_camera_frustums 0)
    fi
    DEBUG_STOP_ARGS=()
    if ((DEBUG_STOP_AFTER_KEYFRAME1)); then
      DEBUG_STOP_ARGS=(--debug_stop_after_keyframe1 1)
    fi
    CALIBRATION_ARGS=()
    if [[ -n "$PIPER_CALIBRATION_BUNDLE" ]]; then
      CALIBRATION_ARGS=(--piper_calibration_bundle "$PIPER_CALIBRATION_BUNDLE")
    fi
    if ! "${RUN_ENV[@]}" "$CONDA_BIN" run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_piper.py \
      --anygrasp_dir "$ANY" \
      --replay_dir "$REPLAY" \
      --hand_npz "$HAND" \
      --output_dir "$OUT" \
      --reuse_preview_summary_json "$PREVIEW" \
      --reuse_preview_frame_mode annotated_json_keyframes \
      --reuse_preview_candidate_group orientation \
      --reuse_preview_top_rank 1 \
      --image_width 640 \
      --image_height 480 \
      --fovy_deg 42.499880046655484 \
      --arm auto \
      --execute_both_arms 1 \
      --dual_stage_require_all_plans ${DUAL_STAGE_REQUIRE_ALL_PLANS} \
      --require_keyframe1_reached_before_close 1 \
      --require_keyframe1_reached_before_action 1 \
      "${DEBUG_STOP_ARGS[@]}" \
      --planner_backend urdfik \
      --urdfik_trajectory_mode ${TRAJECTORY_MODE} \
      --urdfik_joint_interp_waypoints ${JOINT_INTERP_WAYPOINTS} \
      --urdfik_cartesian_interp_steps -1 \
      --urdfik_cartesian_interp_auto_step_m ${CARTESIAN_AUTO_STEP_M} \
      --execute_partial_cartesian_plan ${EXECUTE_PARTIAL_CARTESIAN_PLAN} \
      --urdfik_max_position_threshold_m ${IK_MAX_POSITION_THRESHOLD_M} \
      --urdfik_max_rotation_threshold_rad ${IK_MAX_ROTATION_THRESHOLD_RAD} \
      --piper_urdfik_apply_global_trans_to_ik ${PIPER_APPLY_GLOBAL_TRANS_TO_IK} \
      --candidate_selection_mode ${CANDIDATE_SELECTION_MODE} \
      --candidate_max_rotation_distance_deg ${CANDIDATE_MAX_ROTATION_DISTANCE_DEG} \
      --candidate_keep_camera_up ${CANDIDATE_KEEP_CAMERA_UP} \
      --enforce_candidate_distance_constraint ${ENFORCE_CANDIDATE_DISTANCE_CONSTRAINT} \
      --candidate_orientation_remap_label ${CANDIDATE_ORIENTATION_REMAP_LABEL} \
      --left_target_object "$LEFT_OBJ" \
      --right_target_object "$RIGHT_OBJ" \
      --candidate_target_local_x_offset_m ${CANDIDATE_TARGET_LOCAL_X_OFFSET_M} \
      --candidate_target_local_z_offset_m ${CANDIDATE_TARGET_LOCAL_Z_OFFSET_M} \
      --approach_axis ${APPROACH_AXIS} \
      --approach_offset_m ${APPROACH_OFFSET_M} \
      --reach_error_pose_source ${REACH_ERROR_POSE_SOURCE} \
      --replan_until_reached 1 \
      --replan_until_reached_max_attempts ${REPLAN_ATTEMPTS} \
      --save_debug_preview 1 \
      --save_debug_execution_preview 0 \
      --save_pose_debug 1 \
      --debug_visualize_targets ${VISUALIZE_TARGETS} \
      --debug_candidate_top_k ${DEBUG_CANDIDATE_TOP_K} \
      --debug_common_candidate_top_k ${DEBUG_COMMON_CANDIDATE_TOP_K} \
      --debug_visualize_selected_keyframe_axes ${DEBUG_VISUALIZE_SELECTED_KEYFRAME_AXES} \
      --debug_visualize_ik_waypoints ${DEBUG_VISUALIZE_IK_WAYPOINTS} \
      --debug_gripper_actor_forward_axis ${DEBUG_GRIPPER_ACTOR_FORWARD_AXIS} \
      --reach_pos_tol_m 0.03 \
      --reach_rot_tol_deg 180 \
      --enable_grasp_action_object_collision ${ENABLE_EXECUTION_COLLISIONS} \
      --grasp_action_object_collision_start_stage pregrasp \
      --execution_object_collision_mode convex \
      --execution_object_visual_scale_override ${LEFT_OBJ}=0.8 \
      --execution_object_collision_scale_override ${LEFT_OBJ}=0.8 \
      --execution_object_visual_scale_override ${RIGHT_OBJ}=0.8 \
      --execution_object_collision_scale_override ${RIGHT_OBJ}=0.8 \
      --gripper_contact_monitor_mode all_robot_links \
      --execute_interp_steps ${EXECUTE_INTERP_STEPS} \
      --joint_command_scene_steps ${JOINT_COMMAND_SCENE_STEPS} \
      --settle_steps ${SETTLE_STEPS} \
      --joint_target_wait_steps ${JOINT_TARGET_WAIT_STEPS} \
      --joint_target_wait_tol_rad 0.01 \
      --print_execution_pose_every ${PRINT_POSE_EVERY} \
      --hold_frames_after_stage 8 \
      --pure_scene_output ${PURE_SCENE_OUTPUT} \
      --overlay_text 0 \
      --head_only 0 \
      --third_person_view 1 \
      --vscode_compatible_video 1 \
      --lighting_mode front_no_shadow \
      --robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json \
      --camera_cv_axis_mode legacy_r1 \
      "${CALIBRATION_ARGS[@]}" \
      --wrist_left_forward_offset_m ${WRIST_LEFT_FORWARD_OFFSET_M} \
      --wrist_right_forward_offset_m ${WRIST_RIGHT_FORWARD_OFFSET_M} \
      --wrist_left_lateral_offset_m ${WRIST_LEFT_LATERAL_OFFSET_M} \
      --wrist_right_lateral_offset_m ${WRIST_RIGHT_LATERAL_OFFSET_M} \
      --wrist_left_roll_deg ${WRIST_LEFT_ROLL_DEG} \
      --wrist_right_roll_deg ${WRIST_RIGHT_ROLL_DEG} \
      --wrist_left_yaw_deg ${WRIST_LEFT_YAW_DEG} \
      --wrist_right_yaw_deg ${WRIST_RIGHT_YAW_DEG} \
      --wrist_left_pitch_deg ${WRIST_LEFT_PITCH_DEG} \
      --wrist_right_pitch_deg ${WRIST_RIGHT_PITCH_DEG} \
      --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325 \
      --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399 \
      "${VIEWER_ARGS[@]}" \
      "${MESH_ARGS[@]}"; then
      if ((CONTINUE_ON_ERROR)); then
        echo "[error-continue] task=${TASK} id=${ID} planner command failed"
        continue
      fi
      exit 1
    fi
  done
done
