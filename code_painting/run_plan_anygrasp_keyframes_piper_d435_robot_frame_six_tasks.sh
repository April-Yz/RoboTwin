#!/usr/bin/env bash
set -euo pipefail

# Planner runner for D435 preview summaries generated in canonical robot/replay
# frame mode. In those summaries the target blue local +Z axis is the gripper
# approach/fingertip direction, so compensation and pregrasp retreat use local Z.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREVIEW_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435_robot_frame
GPU=2
MAX_PER_TASK=0
ID_START=""
ID_END=""
SKIP_PREVIEW_GENERATION=0
TASKS=()
IDS=()
FORWARD_ARGS=()

while (($# > 0)); do
  case "$1" in
    --gpu)
      GPU="$2"
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    --max_per_task)
      MAX_PER_TASK="$2"
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    --id_start)
      ID_START="$2"
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    --id_end)
      ID_END="$2"
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    --tasks)
      FORWARD_ARGS+=("$1")
      shift
      TASKS=()
      while (($# > 0)); do
        if [[ "$1" == --* ]]; then
          break
        fi
        TASKS+=("$1")
        FORWARD_ARGS+=("$1")
        shift
      done
      ;;
    --ids)
      FORWARD_ARGS+=("$1")
      shift
      IDS=()
      while (($# > 0)); do
        if [[ "$1" == --* ]]; then
          break
        fi
        IDS+=("$1")
        FORWARD_ARGS+=("$1")
        shift
      done
      ;;
    --preview_root)
      PREVIEW_ROOT="$2"
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    --skip_preview_generation)
      SKIP_PREVIEW_GENERATION=1
      shift
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

PREVIEW_ARGS=(--gpu "$GPU" --output_root "$PREVIEW_ROOT" --max_per_task "$MAX_PER_TASK")
if ((${#TASKS[@]} > 0)); then
  PREVIEW_ARGS+=(--tasks "${TASKS[@]}")
fi
if ((${#IDS[@]} > 0)); then
  PREVIEW_ARGS+=(--ids "${IDS[@]}")
fi
if [[ -n "$ID_START" ]]; then
  PREVIEW_ARGS+=(--id_start "$ID_START")
fi
if [[ -n "$ID_END" ]]; then
  PREVIEW_ARGS+=(--id_end "$ID_END")
fi

if ((SKIP_PREVIEW_GENERATION == 0)); then
  bash "${SCRIPT_DIR}/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh" "${PREVIEW_ARGS[@]}"
fi

exec bash "${SCRIPT_DIR}/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh" \
  --preview_root "$PREVIEW_ROOT" \
  --candidate_orientation_remap_label identity \
  --candidate_target_local_x_offset_m 0.0 \
  --candidate_target_local_z_offset_m -0.05 \
  --approach_axis local_z \
  --approach_offset_m 0.12 \
  --debug_gripper_actor_forward_axis local_z \
  "${FORWARD_ARGS[@]}"
