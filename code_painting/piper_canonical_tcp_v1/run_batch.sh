#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY=/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10
MANIFEST="$SCRIPT_DIR/batch_manifest.tsv"
OUTPUT_ROOT="$SCRIPT_DIR/outputs"
MODE=all
GPU=0

while (($#)); do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--mode all|joint|eepose] [--gpu N] [--output-root PATH]"; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$MODE" =~ ^(all|joint|eepose)$ ]] || { echo "[error] invalid mode=$MODE" >&2; exit 2; }

OURS_ROOT="$REPO/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_de_human_replay_clean_right_cam"
mkdir -p "$OUTPUT_ROOT/_batch_logs"
FAILURES="$OUTPUT_ROOT/_batch_logs/${MODE}_failures.tsv"
: >"$FAILURES"

while IFS=$'\t' read -r TASK ID ARM; do
  [[ -z "$TASK" || "$TASK" == \#* ]] && continue
  EP_ROOT="$OUTPUT_ROOT/$TASK/foundation_input_${ID}"
  echo "[episode] mode=$MODE task=$TASK id=$ID arm=$ARM gpu=$GPU"
  if [[ "$MODE" == all || "$MODE" == joint ]]; then
    SOURCE="$OURS_ROOT/$TASK/foundation_input_${ID}"
    if [[ -f "$SOURCE/pose_debug.jsonl" ]]; then
      if ! "$PY" "$SCRIPT_DIR/compare_joint_control.py" \
        --pose-debug "$SOURCE/pose_debug.jsonl" \
        --source-video "$SOURCE/head_cam_plan.mp4" \
        --robot-config "$REPO/robot_config_PiperPika_agx_dual_table_0515.json" \
        --output-dir "$EP_ROOT/joint_control" --task "$TASK" --episode-id "$ID"; then
        printf 'joint\t%s\t%s\t%s\n' "$TASK" "$ID" "$ARM" >>"$FAILURES"
      fi
    else
      echo "[missing-ours-q-trace] $SOURCE/pose_debug.jsonl"
      printf 'joint_missing_input\t%s\t%s\t%s\n' "$TASK" "$ID" "$ARM" >>"$FAILURES"
    fi
  fi
  if [[ "$MODE" == all || "$MODE" == eepose ]]; then
    for STRATEGY in orientation fused top_score; do
      if ! bash "$SCRIPT_DIR/run_eepose_strategy.sh" \
        --task "$TASK" --id "$ID" --arm "$ARM" --strategy "$STRATEGY" \
        --gpu "$GPU" --output-root "$OUTPUT_ROOT"; then
        printf '%s\t%s\t%s\t%s\n' "$STRATEGY" "$TASK" "$ID" "$ARM" >>"$FAILURES"
      fi
    done
    if [[ -f "$EP_ROOT/eepose/orientation/head_cam_plan.mp4" && \
          -f "$EP_ROOT/eepose/fused/head_cam_plan.mp4" && \
          -f "$EP_ROOT/eepose/top_score/head_cam_plan.mp4" ]]; then
      "$PY" "$SCRIPT_DIR/compose_strategy_video.py" \
        --episode-root "$EP_ROOT" --output "$EP_ROOT/eepose/strategy_comparison.mp4" || \
        printf 'compose\t%s\t%s\t%s\n' "$TASK" "$ID" "$ARM" >>"$FAILURES"
    fi
  fi
  if ! "$PY" "$SCRIPT_DIR/vscode_video.py" \
    --root "$EP_ROOT" --apply --workers 2 \
    --manifest "$EP_ROOT/vscode_transcode_manifest.json"; then
    printf 'video_transcode\t%s\t%s\t%s\n' "$TASK" "$ID" "$ARM" >>"$FAILURES"
  fi
done <"$MANIFEST"

if [[ -s "$FAILURES" ]]; then
  echo "[batch-finished-with-failures] $FAILURES"
else
  touch "$OUTPUT_ROOT/_batch_logs/${MODE}_SUCCESS"
  echo "[batch-success] mode=$MODE"
fi
