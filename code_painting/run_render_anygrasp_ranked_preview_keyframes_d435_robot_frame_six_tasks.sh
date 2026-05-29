#!/usr/bin/env bash
set -euo pipefail

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh

GPU=2
OUT_ROOT_BASE=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435_robot_frame
SOURCE_PREVIEW_ROOT_BASE=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435
MAX_PER_TASK=0
ID_START=""
ID_END=""
SKIP_EXISTING=1
IDS_FILTER=()
TASKS=(pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray)

while (($# > 0)); do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --output_root)
      OUT_ROOT_BASE="$2"
      shift 2
      ;;
    --source_preview_root)
      SOURCE_PREVIEW_ROOT_BASE="$2"
      shift 2
      ;;
    --max_per_task)
      MAX_PER_TASK="$2"
      shift 2
      ;;
    --id_start)
      ID_START="$2"
      shift 2
      ;;
    --id_end)
      ID_END="$2"
      shift 2
      ;;
    --skip_existing)
      SKIP_EXISTING="$2"
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

for TASK in "${TASKS[@]}"; do
  case "$TASK" in
    pick_diverse_bottles) LEFT_OBJ=left_bottle; RIGHT_OBJ=right_bottle ;;
    place_bread_basket) LEFT_OBJ=basket; RIGHT_OBJ=bread ;;
    stack_cups) LEFT_OBJ=left_light_pink_cup; RIGHT_OBJ=right_dark_red_cup ;;
    handover_bottle) LEFT_OBJ=right_bottle; RIGHT_OBJ=right_bottle ;;
    pnp_bread) LEFT_OBJ=left_bread; RIGHT_OBJ=right_bread ;;
    pnp_tray) LEFT_OBJ=left_dark_red_cup; RIGHT_OBJ=right_bottle ;;
    *)
      echo "[error] unknown task: $TASK" >&2
      exit 2
      ;;
  esac

  ANN=/home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json
  ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output
  [[ -d "$ANY_ROOT" ]] || ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output_old_cam
  REPLAY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435
  HAND_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output
  OUT_ROOT=${OUT_ROOT_BASE}/${TASK}
  SOURCE_PREVIEW_ROOT=${SOURCE_PREVIEW_ROOT_BASE}/${TASK}

  [[ -f "$ANN" ]] || { echo "[skip] task=${TASK} missing annotation $ANN"; continue; }
  [[ -d "$ANY_ROOT" ]] || { echo "[skip] task=${TASK} missing ANY_ROOT=$ANY_ROOT"; continue; }
  [[ -d "$REPLAY_ROOT" ]] || { echo "[skip] task=${TASK} missing REPLAY_ROOT=$REPLAY_ROOT"; continue; }

  TASK_IDS=()
  if ((${#IDS_FILTER[@]} > 0)); then
    TASK_IDS=("${IDS_FILTER[@]}")
  elif [[ -d "$SOURCE_PREVIEW_ROOT" ]]; then
    mapfile -t TASK_IDS < <(
      find "$SOURCE_PREVIEW_ROOT" -maxdepth 2 -name summary.json 2>/dev/null \
        | sed -E 's#.*/foundation_input_([0-9]+)/summary.json#\1#' \
        | sort -n
    )
  else
    mapfile -t TASK_IDS < <(
      find "$ANY_ROOT" -maxdepth 1 -type d -name 'foundation_input_*' 2>/dev/null \
        | sed -E 's#.*/foundation_input_([0-9]+)$#\1#' \
        | sort -n
    )
  fi

  if [[ -n "$ID_START" || -n "$ID_END" ]]; then
    start=${ID_START:-0}
    end=${ID_END:-999999}
    FILTERED_IDS=()
    for have_id in "${TASK_IDS[@]}"; do
      if ((have_id >= start && have_id <= end)); then
        FILTERED_IDS+=("$have_id")
      fi
    done
    TASK_IDS=("${FILTERED_IDS[@]}")
  fi
  if ((MAX_PER_TASK > 0 && ${#TASK_IDS[@]} > MAX_PER_TASK)); then
    TASK_IDS=("${TASK_IDS[@]:0:MAX_PER_TASK}")
  fi

  if ((SKIP_EXISTING)); then
    FILTERED_IDS=()
    for have_id in "${TASK_IDS[@]}"; do
      if [[ ! -f "${OUT_ROOT}/foundation_input_${have_id}/summary.json" ]]; then
        FILTERED_IDS+=("$have_id")
      fi
    done
    TASK_IDS=("${FILTERED_IDS[@]}")
  fi

  if ((${#TASK_IDS[@]} == 0)); then
    echo "===== render robot-frame D435 preview task=${TASK} out=${OUT_ROOT} ids=none skip_existing=${SKIP_EXISTING} ====="
    continue
  fi

  echo "===== render robot-frame D435 preview task=${TASK} out=${OUT_ROOT} ids=${TASK_IDS[*]} skip_existing=${SKIP_EXISTING} ====="
  VIDEO_PREFIX=foundation_input CUDA_VISIBLE_DEVICES=${GPU} bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh \
    "$ANY_ROOT" \
    "$REPLAY_ROOT" \
    "$HAND_ROOT" \
    "$OUT_ROOT" \
    --ids "${TASK_IDS[@]}" \
    --hand_keyframes_json "$ANN" \
    --left_target_object "$LEFT_OBJ" \
    --right_target_object "$RIGHT_OBJ" \
    --anygrasp_score_weight 0.25 \
    --orientation_score_weight 0.75 \
    --max_rotation_distance_deg 90 \
    --candidate_frame_mode robot_replay \
    --candidate_target_local_x_offset_m 0.0 \
    --candidate_target_local_z_offset_m -0.05 \
    --draw_object_overlay 1 \
    --draw_hand_reference 1 \
    --debug_dump_object_distances 1 \
    --top_k 20 \
    --camera_cv_axis_mode legacy_r1
done
