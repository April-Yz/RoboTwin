#!/usr/bin/env bash
set -euo pipefail

# Mode O: first-frame FoundationPose pick_diverse_bottles baseline.
# No manual keyframes, no human hand orientation, no AnyGrasp candidate ranking.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU=2
VIEWER=0
VIEWER_WAIT_AT_END=0
CONTINUE_ON_ERROR=0
DRY_RUN=0
OUTPUT_ROOT=""
IDS=()
ID_START=""
ID_END=""
FOUNDATION_FRAME=0
GRASP_SURFACE_RETREAT_M=0.03
APPROACH_OFFSET_M=0.08
LIFT_M=0.10
PLACE_Z_MODE=env_target
TRAJECTORY_MODE=cartesian_interp_ik
CARTESIAN_AUTO_STEP_M=0.03
IK_MAX_ROTATION_THRESHOLD_RAD=3.14
DISABLE_EXECUTION_COLLISIONS=1

usage() {
  cat <<'EOF'
Usage: run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh [OPTIONS]

Mode O: first-frame FoundationPose baseline for pick_diverse_bottles.

Required:
  --ids <ID> [<ID>...]       Episode ID(s), or range like "0-10"
  --id_start <N> --id_end <N>

Options:
  --gpu <N>                         GPU index (default: 2)
  --viewer                          Enable SAPIEN viewer
  --viewer_wait_at_end <0|1>        Keep viewer open at the end
  --continue_on_error               Continue after a failed id
  --dry_run                         Print only
  --output_root <PATH>              Custom output root
  --foundation_frame <N>            FoundationPose frame used for target design (default: 0)
  --grasp_surface_retreat_m <M>     Offset from object center back to side grasp surface (default: 0.03)
  --approach_offset_m <M>           Planner pregrasp retreat (default: 0.08, matching env pre_grasp_dis)
  --lift_m <M>                      Used when --place_z_mode object_plus_lift (default: 0.10)
  --place_z_mode <env_target|object_plus_lift>
  --trajectory_mode <MODE>          Planner trajectory mode (default: cartesian_interp_ik)
  --cartesian_auto_step_m <M>       Cartesian auto step (default: 0.03)
  --ik_max_rotation_threshold_rad <R>
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
    --foundation_frame) FOUNDATION_FRAME="$2"; shift 2 ;;
    --grasp_surface_retreat_m) GRASP_SURFACE_RETREAT_M="$2"; shift 2 ;;
    --approach_offset_m) APPROACH_OFFSET_M="$2"; shift 2 ;;
    --lift_m) LIFT_M="$2"; shift 2 ;;
    --place_z_mode) PLACE_Z_MODE="$2"; shift 2 ;;
    --trajectory_mode) TRAJECTORY_MODE="$2"; shift 2 ;;
    --cartesian_auto_step_m) CARTESIAN_AUTO_STEP_M="$2"; shift 2 ;;
    --ik_max_rotation_threshold_rad) IK_MAX_ROTATION_THRESHOLD_RAD="$2"; shift 2 ;;
    --disable_execution_collisions) DISABLE_EXECUTION_COLLISIONS=1; shift ;;
    --id_start) ID_START="$2"; shift 2 ;;
    --id_end) ID_END="$2"; shift 2 ;;
    --ids)
      shift
      IDS=()
      while (($# > 0)); do
        if [[ "$1" == --* ]]; then break; fi
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

if [[ -n "$ID_START" && -n "$ID_END" ]]; then
  for ((_id=ID_START; _id<=ID_END; _id++)); do
    IDS+=("$_id")
  done
fi

((${#IDS[@]} == 0)) && { echo "[error] --ids or --id_start/--id_end required"; exit 2; }

TASK=pick_diverse_bottles
ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output
[[ -d "$ANY_ROOT" ]] || ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output_old_cam

if [[ -n "$OUTPUT_ROOT" ]]; then
  OUT_ROOT=${OUTPUT_ROOT}/${TASK}
elif ((VIEWER)); then
  OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/first_frame_foundation_viewer/${TASK}
else
  OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/first_frame_foundation/${TASK}
fi

echo "===== Mode O: first-frame Foundation task=${TASK} ids=${IDS[*]} viewer=${VIEWER} frame=${FOUNDATION_FRAME} ====="

for ID in "${IDS[@]}"; do
  ANY=${ANY_ROOT}/foundation_input_${ID}
  REPLAY=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435/foundation_input_${ID}
  HAND=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz
  OUT=${OUT_ROOT}/foundation_input_${ID}

  [[ -d "$ANY" ]] || { echo "[skip] missing anygrasp scene dir $ANY"; continue; }
  [[ -d "$REPLAY" ]] || { echo "[skip] missing D435 replay $REPLAY"; continue; }
  [[ -f "$HAND" ]] || { echo "[skip] missing hand npz placeholder $HAND"; continue; }

  echo "[run] task=${TASK} id=${ID} output=${OUT}"
  if ((DRY_RUN)); then
    echo "  ANY=${ANY}"
    echo "  REPLAY=${REPLAY}"
    echo "  HAND=${HAND}"
    continue
  fi

  RUN_ENV=(env CUDA_VISIBLE_DEVICES=${GPU})
  if ((VIEWER)); then
    [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
    RUN_ENV=(env -u CUDA_VISIBLE_DEVICES)
  fi

  ARGS=(
    --anygrasp_dir "$ANY"
    --replay_dir "$REPLAY"
    --hand_npz "$HAND"
    --output_dir "$OUT"
    --video_id "$ID"
    --gpu "$GPU"
    --foundation_frame "$FOUNDATION_FRAME"
    --grasp_surface_retreat_m "$GRASP_SURFACE_RETREAT_M"
    --approach_offset_m "$APPROACH_OFFSET_M"
    --lift_m "$LIFT_M"
    --place_z_mode "$PLACE_Z_MODE"
    --urdfik_trajectory_mode "$TRAJECTORY_MODE"
    --urdfik_cartesian_interp_auto_step_m "$CARTESIAN_AUTO_STEP_M"
    --urdfik_max_rotation_threshold_rad "$IK_MAX_ROTATION_THRESHOLD_RAD"
  )
  if ((VIEWER)); then
    ARGS+=(--enable_viewer 1 --viewer_wait_at_end "$VIEWER_WAIT_AT_END")
  fi
  if ((DISABLE_EXECUTION_COLLISIONS)); then
    ARGS+=(--enable_grasp_action_object_collision 0)
  fi

  mkdir -p "$OUT"
  PYTHON_BIN=/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python
  if ! "${RUN_ENV[@]}" "$PYTHON_BIN" -u \
    "${SCRIPT_DIR}/plan_first_frame_foundation_pick_diverse_bottles.py" \
    "${ARGS[@]}" 2>"${OUT}/stderr.log"; then
    echo "[stderr-tail] $(tail -5 "${OUT}/stderr.log" 2>/dev/null)"
    if ((CONTINUE_ON_ERROR)); then echo "[error-continue] id=${ID}"; continue; fi
    exit 1
  fi
  echo "[done] task=${TASK} id=${ID}"
done
