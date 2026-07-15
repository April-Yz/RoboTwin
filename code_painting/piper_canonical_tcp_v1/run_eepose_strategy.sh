#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY=/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10
DATA=/home/zaijia001/ssd/data/piper/hand
PREVIEW_ROOT="$REPO/code_painting/anygrasp_h2o_preview_d435_robot_frame"
CALIBRATION="$REPO/calibration_bundle_piper_new_table_0515.json"

TASK=""
ID=""
ARM=auto
STRATEGY=""
GPU=0
OUTPUT_ROOT="$SCRIPT_DIR/outputs"
DRY_RUN=0

usage() {
  echo "Usage: $0 --task TASK --id ID --arm auto|left|right --strategy orientation|fused|top_score [--gpu N] [--output-root PATH] [--dry-run]"
}

while (($#)); do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --id) ID="$2"; shift 2 ;;
    --arm) ARM="$2"; shift 2 ;;
    --strategy) STRATEGY="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$TASK" && -n "$ID" && -n "$STRATEGY" ]] || { usage >&2; exit 2; }
[[ "$ARM" =~ ^(auto|left|right)$ ]] || { echo "[error] invalid arm=$ARM" >&2; exit 2; }
[[ "$STRATEGY" =~ ^(orientation|fused|top_score)$ ]] || { echo "[error] invalid strategy=$STRATEGY" >&2; exit 2; }

ANY="$DATA/$TASK/${TASK}_output/foundation_input_${ID}"
[[ -d "$ANY" ]] || ANY="$DATA/$TASK/${TASK}_output_old_cam/foundation_input_${ID}"
REPLAY="$DATA/$TASK/foundation_replay_d435/foundation_input_${ID}"
HAND="$DATA/$TASK/harmer_output/hand_detections_${ID}.npz"
PREVIEW="$PREVIEW_ROOT/$TASK/foundation_input_${ID}/summary.json"
OUT="$OUTPUT_ROOT/$TASK/foundation_input_${ID}/eepose/$STRATEGY"

for path in "$ANY" "$REPLAY"; do [[ -d "$path" ]] || { echo "[error] missing directory: $path" >&2; exit 3; }; done
for path in "$HAND" "$PREVIEW" "$CALIBRATION"; do [[ -f "$path" ]] || { echo "[error] missing file: $path" >&2; exit 3; }; done

if [[ -e "$OUT/SUCCESS" ]]; then
  echo "[skip-existing-success] $OUT"
  exit 0
fi
if [[ -d "$OUT" ]] && find "$OUT" -mindepth 1 -print -quit | grep -q .; then
  echo "[error] refusing non-empty output without SUCCESS: $OUT" >&2
  exit 4
fi

LEFT_OBJ=""
RIGHT_OBJ=""
MESH_ARGS=()
case "$TASK" in
  pick_diverse_bottles)
    LEFT_OBJ=left_bottle; RIGHT_OBJ=right_bottle
    MESH_ARGS=(--object_mesh_override left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj)
    ;;
  place_bread_basket)
    LEFT_OBJ=basket; RIGHT_OBJ=bread
    MESH_ARGS=(--object_mesh_override basket=/home/zaijia001/ssd/data/R1/hand/obj_mesh/basket/basket.obj --object_mesh_override bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_y/bread_y.obj)
    ;;
  stack_cups)
    LEFT_OBJ=left_light_pink_cup; RIGHT_OBJ=right_dark_red_cup
    MESH_ARGS=(--object_mesh_override left_light_pink_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/light_pink_cup/light_pink_cup.obj --object_mesh_override right_dark_red_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj)
    ;;
  handover_bottle)
    LEFT_OBJ=right_bottle; RIGHT_OBJ=right_bottle
    MESH_ARGS=(--object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj)
    ;;
  pnp_bread)
    LEFT_OBJ=left_bread; RIGHT_OBJ=right_bread
    MESH_ARGS=(--object_mesh_override left_bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_nj/bread_niujiao.obj --object_mesh_override right_bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_yr/bread_yerong.obj)
    ;;
  pnp_tray)
    LEFT_OBJ=left_dark_red_cup; RIGHT_OBJ=right_bottle
    MESH_ARGS=(--object_mesh_override left_dark_red_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj)
    ;;
  *) echo "[error] unsupported task=$TASK" >&2; exit 2 ;;
esac

if [[ "$ARM" == auto ]]; then EXECUTE_BOTH=1; else EXECUTE_BOTH=0; fi
GROUP="$STRATEGY"
SELECTION_MODE=planner
ORIENTATION_REMAP=swap_red_blue_keep_green
SOURCE_SEMANTICS="robot-frame preview T_W_CGRASP; apply R_CGRASP_RTCP before IK"
if [[ "$STRATEGY" == top_score ]]; then
  GROUP=orientation
  SELECTION_MODE=top_score_auto
  ORIENTATION_REMAP=identity
  SOURCE_SEMANTICS="raw AnyGrasp T_W_RTCP; no preview-axis remap"
fi

mkdir -p "$OUT"
"$PY" "$SCRIPT_DIR/frame_contract.py" \
  --write "$OUT/frame_contract.json" --task "$TASK" --episode-id "$ID" \
  --strategy "$STRATEGY" --source-semantics "$SOURCE_SEMANTICS" >/dev/null

CMD=(
  env CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$SCRIPT_DIR/planner.py"
  --anygrasp_dir "$ANY"
  --replay_dir "$REPLAY"
  --hand_npz "$HAND"
  --output_dir "$OUT"
  --reuse_preview_summary_json "$PREVIEW"
  --reuse_preview_frame_mode annotated_json_keyframes
  --reuse_preview_candidate_group "$GROUP"
  --reuse_preview_top_rank 1
  --candidate_selection_mode "$SELECTION_MODE"
  --candidate_orientation_remap_label "$ORIENTATION_REMAP"
  --candidate_post_rot_xyz_deg 0 0 0
  --candidate_target_local_x_offset_m 0
  --candidate_target_local_z_offset_m 0
  --candidate_keep_camera_up 0
  --candidate_max_rotation_distance_deg -1
  --enforce_target_object_constraint 1
  --enforce_candidate_distance_constraint 0
  --left_target_object "$LEFT_OBJ"
  --right_target_object "$RIGHT_OBJ"
  --arm "$ARM"
  --execute_both_arms "$EXECUTE_BOTH"
  --dual_stage_require_all_plans 1
  --dual_stage_freeze_reached_arms_on_replan 1
  --require_keyframe1_reached_before_close 1
  --require_keyframe1_reached_before_action 1
  --planner_backend urdfik
  --urdfik_trajectory_mode joint_interp
  --urdfik_joint_interp_waypoints 40
  --urdfik_position_threshold_m 0.001
  --urdfik_rotation_threshold_rad 0.02
  --urdfik_max_position_threshold_m 0.02
  --urdfik_max_rotation_threshold_rad 0.12
  --urdfik_num_seeds 20
  --urdfik_solution_selection joint_continuity
  --urdfik_seed_perturbations 6
  --urdfik_seed_perturbation_scale 0.05
  --piper_urdfik_apply_global_trans_to_ik 0
  --reach_error_pose_source tcp
  --reach_pos_tol_m 0.03
  --reach_rot_tol_deg 10
  --replan_until_reached 1
  --replan_until_reached_max_attempts 3
  --fail_on_execution_failure 1
  --approach_axis local_x
  --approach_offset_m 0.12
  --execute_interp_steps 24
  --joint_command_scene_steps 10
  --settle_steps 30
  --joint_target_wait_steps 25
  --joint_target_wait_tol_rad 0.01
  --hold_frames_after_stage 8
  --save_debug_preview 1
  --save_debug_execution_preview 1
  --save_pose_debug 1
  --debug_visualize_targets 1
  --debug_candidate_top_k 0
  --debug_common_candidate_top_k 0
  --debug_visualize_selected_keyframe_axes 1
  --debug_visualize_ik_waypoints 1
  --debug_gripper_actor_forward_axis local_x
  --debug_target_axis_length 0.06
  --pure_scene_output 0
  --overlay_text 1
  --head_only 0
  --third_person_view 1
  --vscode_compatible_video 1
  --enable_viewer 0
  --viewer_wait_at_end 0
  --enable_grasp_action_object_collision 0
  --disable_table 1
  --lighting_mode front_no_shadow
  --robot_config "$REPO/robot_config_PiperPika_agx_dual_table_0515.json"
  --piper_calibration_bundle "$CALIBRATION"
  --camera_cv_axis_mode legacy_r1
  --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325
  --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399
  "${MESH_ARGS[@]}"
)

printf "[run] task=%s id=%s arm=%s strategy=%s source_remap=%s gpu=%s output=%s\n" "$TASK" "$ID" "$ARM" "$STRATEGY" "$ORIENTATION_REMAP" "$GPU" "$OUT"
if ((DRY_RUN)); then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

printf '%q ' "${CMD[@]}" >"$OUT/command.sh.txt"
printf '\n' >>"$OUT/command.sh.txt"
if "${CMD[@]}" > >(tee "$OUT/stdout.log") 2> >(tee "$OUT/stderr.log" >&2); then
  touch "$OUT/SUCCESS"
  echo "[success] $OUT"
else
  status=$?
  printf '%s\n' "$status" >"$OUT/EXIT_CODE"
  echo "[failed] status=$status output=$OUT" >&2
  exit "$status"
fi
