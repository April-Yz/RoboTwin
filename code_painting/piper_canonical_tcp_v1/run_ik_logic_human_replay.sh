#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY=/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10
DATA=/home/zaijia001/ssd/data/piper/hand
KEYFRAMES_ROOT="$REPO/code_painting/h2o_manual_review"
CALIBRATION="$REPO/calibration_bundle_piper_new_table_0515.json"

TASK=""; ID=""; IK_LOGIC=""; GPU=0; OUTPUT_ROOT=""; DRY_RUN=0
while (($#)); do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --id) ID="$2"; shift 2 ;;
    --ik-logic) IK_LOGIC="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) echo "Usage: $0 --task TASK --id ID --ik-logic legacy|canonical --output-root PATH [--gpu N] [--dry-run]"; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TASK" && -n "$ID" && -n "$IK_LOGIC" && -n "$OUTPUT_ROOT" ]] || { echo "[error] required: --task --id --ik-logic --output-root" >&2; exit 2; }
[[ "$IK_LOGIC" =~ ^(legacy|canonical)$ ]] || { echo "[error] invalid ik-logic=$IK_LOGIC" >&2; exit 2; }

ANY="$DATA/$TASK/${TASK}_output/foundation_input_${ID}"
[[ -d "$ANY" ]] || ANY="$DATA/$TASK/${TASK}_output_old_cam/foundation_input_${ID}"
REPLAY="$DATA/$TASK/foundation_replay_d435/foundation_input_${ID}"
HAND="$DATA/$TASK/harmer_output/hand_detections_${ID}.npz"
KEYFRAMES="$KEYFRAMES_ROOT/$TASK/hand_keyframes_all.json"
OUT="$OUTPUT_ROOT/$TASK/foundation_input_${ID}/eepose/human_replay"
for path in "$ANY" "$REPLAY"; do [[ -d "$path" ]] || { echo "[error] missing directory: $path" >&2; exit 3; }; done
for path in "$HAND" "$KEYFRAMES" "$CALIBRATION"; do [[ -f "$path" ]] || { echo "[error] missing file: $path" >&2; exit 3; }; done
if [[ -f "$OUT/head_cam_plan.mp4" ]]; then echo "[reuse-existing-video] $OUT/head_cam_plan.mp4"; exit 0; fi
if [[ -d "$OUT" ]] && find "$OUT" -mindepth 1 -print -quit | grep -q .; then
  echo "[error] refusing non-empty incomplete output: $OUT" >&2; exit 4
fi

MESH_ARGS=()
case "$TASK" in
  pick_diverse_bottles) MESH_ARGS=(--object_mesh_override left_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cola/cola.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj) ;;
  place_bread_basket) MESH_ARGS=(--object_mesh_override basket=/home/zaijia001/ssd/data/R1/hand/obj_mesh/basket/basket.obj --object_mesh_override bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_y/bread_y.obj) ;;
  stack_cups) MESH_ARGS=(--object_mesh_override left_light_pink_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/light_pink_cup/light_pink_cup.obj --object_mesh_override right_dark_red_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj) ;;
  handover_bottle) MESH_ARGS=(--object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj) ;;
  pnp_bread) MESH_ARGS=(--object_mesh_override left_bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_nj/bread_niujiao.obj --object_mesh_override right_bread=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bread_yr/bread_yerong.obj) ;;
  pnp_tray) MESH_ARGS=(--object_mesh_override left_dark_red_cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/dark_red_cup/dark_red_cup.obj --object_mesh_override right_bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj) ;;
  *) echo "[error] unsupported task=$TASK" >&2; exit 2 ;;
esac

if [[ "$IK_LOGIC" == canonical ]]; then
  PLANNER="$SCRIPT_DIR/planner.py"
  MAX_ROTATION_THRESHOLD=0.12
  NUM_SEEDS=20
  SEED_PERTURBATIONS=6
  REACH_POS_TOL=0.03
  REACH_ROT_TOL=10
  RETREAT_M=0
  APPROACH_AXIS=local_x
  ORIENTATION_ADAPTER="materialized CGRASP_HUMAN to RTCP"
else
  PLANNER="$REPO/code_painting/plan_anygrasp_keyframes_piper.py"
  MAX_ROTATION_THRESHOLD=3.14
  NUM_SEEDS=1
  SEED_PERTURBATIONS=0
  REACH_POS_TOL=0.04
  REACH_ROT_TOL=180
  RETREAT_M=0.14
  APPROACH_AXIS=local_z
  ORIENTATION_ADAPTER=identity
fi

CMD=(
  env CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$SCRIPT_DIR/plan_human_replay_ik_logic.py"
  --ik-logic "$IK_LOGIC" --anygrasp_dir "$ANY" --replay_dir "$REPLAY" --hand_npz "$HAND"
  --output_dir "$OUT" --hand_keyframes_json "$KEYFRAMES" --video_id "$ID" --task "$TASK" --gpu "$GPU"
  --planner_backend urdfik --urdfik_trajectory_mode joint_interp --urdfik_joint_interp_waypoints 40
  --urdfik_num_seeds "$NUM_SEEDS" --urdfik_solution_selection joint_continuity
  --urdfik_seed_perturbations "$SEED_PERTURBATIONS" --urdfik_seed_perturbation_scale 0.05
  --urdfik_max_position_threshold_m 0.02 --urdfik_max_rotation_threshold_rad "$MAX_ROTATION_THRESHOLD"
  --approach_offset_m 0.12 --reach_pos_tol_m "$REACH_POS_TOL" --reach_rot_tol_deg "$REACH_ROT_TOL"
  --replan_until_reached_max_attempts 3 --fail_on_execution_failure 1
  --action_orientation_source grasp --execute_interp_steps 24 --joint_command_scene_steps 10 --settle_steps 30
  --joint_target_wait_steps 25 --joint_target_wait_tol_rad 0.01 --hold_frames_after_stage 8
  --debug_visualize_targets 1 --debug_visualize_selected_keyframe_axes 1 --debug_visualize_ik_waypoints 1
  --pure_scene_output 0 --third_person_view 1 --head_only 0 --lighting_mode front_no_shadow
  --piper_calibration_bundle "$CALIBRATION" --camera_cv_axis_mode legacy_r1
  --head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325
  --head_camera_local_quat_wxyz 0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399
  "${MESH_ARGS[@]}"
)

printf '[run] ik_logic=%s strategy=human_replay task=%s id=%s output=%s\n' "$IK_LOGIC" "$TASK" "$ID" "$OUT"
if ((DRY_RUN)); then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
mkdir -p "$OUT"
"$PY" "$SCRIPT_DIR/write_ik_logic_contract.py" \
  --output "$OUT/input_target_contract.json" --task "$TASK" --episode-id "$ID" \
  --ik-logic "$IK_LOGIC" --strategy human_replay \
  --source-semantics "human hand/gripper center T_W_CGRASP_HUMAN" \
  --orientation-remap "$ORIENTATION_ADAPTER" --planner-entry "$PLANNER" \
  --target-retreat-m "$RETREAT_M" --approach-axis "$APPROACH_AXIS"
printf '%q ' "${CMD[@]}" >"$OUT/command.sh.txt"; printf '\n' >>"$OUT/command.sh.txt"
if "${CMD[@]}" > >(tee "$OUT/stdout.log") 2> >(tee "$OUT/stderr.log" >&2); then
  touch "$OUT/SUCCESS"; echo "[success] $OUT"
else
  status=$?; printf '%s\n' "$status" >"$OUT/EXIT_CODE"
  echo "[failed] ik_logic=$IK_LOGIC strategy=human_replay status=$status output=$OUT" >&2
  exit "$status"
fi
