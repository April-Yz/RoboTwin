#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY=/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10

TASK=""
ID=""
ARM=auto
GPU=0
CANONICAL_ROOT="$SCRIPT_DIR/outputs_canonical_20260715"
OUTPUT_ROOT="$SCRIPT_DIR/outputs_replay_method_compare_20260716"
LEGACY_TARGET_RETREAT_M=""
CANONICAL_APPROACH_OFFSET_M=0.12
LEGACY_APPROACH_OFFSET_M=0.12
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: run_replay_method_compare.sh --task TASK --id ID --arm auto|left|right \
  --legacy-target-retreat-m METERS [--gpu N] [--canonical-root PATH] [--output-root PATH] [--dry-run]

The retreat argument is intentionally required:
  0.00 = current legacy wrapper default; final human pose is not shifted.
  0.12 = historical Piper link6-compensation ablation; final pose retreats 12 cm on local human +Z.
EOF
}

while (($#)); do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --id) ID="$2"; shift 2 ;;
    --arm) ARM="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --canonical-root) CANONICAL_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --legacy-target-retreat-m) LEGACY_TARGET_RETREAT_M="$2"; shift 2 ;;
    --canonical-approach-offset-m) CANONICAL_APPROACH_OFFSET_M="$2"; shift 2 ;;
    --legacy-approach-offset-m) LEGACY_APPROACH_OFFSET_M="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$TASK" && -n "$ID" && -n "$LEGACY_TARGET_RETREAT_M" ]] || { usage >&2; exit 2; }
[[ "$ARM" =~ ^(auto|left|right)$ ]] || { echo "[error] invalid arm=$ARM" >&2; exit 2; }
[[ "$LEGACY_TARGET_RETREAT_M" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || { echo "[error] invalid retreat=$LEGACY_TARGET_RETREAT_M" >&2; exit 2; }

CANONICAL_EP="$CANONICAL_ROOT/$TASK/foundation_input_${ID}"
RETREAT_TAG="$(printf '%s' "$LEGACY_TARGET_RETREAT_M" | tr -- '-.' 'mp')"
CANONICAL_HUMAN_SOURCE_ROOT="$OUTPUT_ROOT/_sources/canonical_human_replay"
CANONICAL_HUMAN_EP="$CANONICAL_HUMAN_SOURCE_ROOT/$TASK/foundation_input_${ID}/eepose/human_replay"
LEGACY_ROOT="$OUTPUT_ROOT/_sources/legacy_human_replay_retreat_${RETREAT_TAG}"
LEGACY_EP="$LEGACY_ROOT/$TASK/foundation_input_${ID}"
EP_OUT="$OUTPUT_ROOT/$TASK/foundation_input_${ID}"
CANONICAL_VIDEO_OUT="$EP_OUT/canonical_four_method_d435.mp4"
VIDEO_OUT="$EP_OUT/canonical_vs_legacy_five_method_d435.mp4"

for STRATEGY in orientation fused top_score; do
  VIDEO="$CANONICAL_EP/eepose/$STRATEGY/head_cam_plan.mp4"
  if [[ -f "$VIDEO" ]]; then
    echo "[reuse-canonical-video] $VIDEO"
    continue
  fi
  CMD=(bash "$SCRIPT_DIR/run_eepose_strategy.sh" --task "$TASK" --id "$ID" --arm "$ARM" --strategy "$STRATEGY" --gpu "$GPU" --output-root "$CANONICAL_ROOT")
  if ((DRY_RUN)); then printf '[dry-run] '; printf '%q ' "${CMD[@]}"; printf '\n'; continue; fi
  if ! "${CMD[@]}"; then
    [[ -f "$VIDEO" ]] || { echo "[error] canonical $STRATEGY produced no head video" >&2; exit 5; }
    echo "[warning] canonical $STRATEGY reported failure; preserving its diagnostic video"
  fi
done

if [[ ! -f "$CANONICAL_HUMAN_EP/head_cam_plan.mp4" ]]; then
  CANONICAL_HUMAN_CMD=(bash "$SCRIPT_DIR/run_canonical_human_replay.sh" --task "$TASK" --id "$ID" --gpu "$GPU" --output-root "$CANONICAL_HUMAN_SOURCE_ROOT")
  if ((DRY_RUN)); then printf '[dry-run] '; printf '%q ' "${CANONICAL_HUMAN_CMD[@]}"; printf '\n'; else
    if ! "${CANONICAL_HUMAN_CMD[@]}"; then
      [[ -f "$CANONICAL_HUMAN_EP/head_cam_plan.mp4" ]] || { echo "[error] canonical Human Replay produced no head video" >&2; exit 5; }
      echo "[warning] canonical Human Replay reported failure; preserving its diagnostic video"
    fi
  fi
else
  echo "[reuse-canonical-human-video] $CANONICAL_HUMAN_EP/head_cam_plan.mp4"
fi

if [[ ! -f "$LEGACY_EP/head_cam_plan.mp4" ]]; then
  LEGACY_CMD=(
    bash "$REPO/code_painting/run_plan_keyframes_human_replay_piper_d435.sh"
    --gpu "$GPU" --tasks "$TASK" --ids "$ID"
    --output_root "$LEGACY_ROOT"
    --approach_offset_m "$LEGACY_APPROACH_OFFSET_M"
    --target_retreat_m "$LEGACY_TARGET_RETREAT_M"
  )
  if ((DRY_RUN)); then printf '[dry-run] '; printf '%q ' "${LEGACY_CMD[@]}"; printf '\n'; exit 0; fi
  if [[ -d "$LEGACY_EP" ]] && find "$LEGACY_EP" -mindepth 1 -print -quit | grep -q .; then
    echo "[error] refusing non-empty incomplete legacy output: $LEGACY_EP" >&2
    exit 4
  fi
  mkdir -p "$LEGACY_EP"
  printf '%q ' "${LEGACY_CMD[@]}" >"$LEGACY_EP/command.sh.txt"
  printf '\n' >>"$LEGACY_EP/command.sh.txt"
  if "${LEGACY_CMD[@]}" > >(tee "$LEGACY_EP/wrapper.stdout.log") 2> >(tee "$LEGACY_EP/wrapper.stderr.log" >&2); then
    touch "$LEGACY_EP/SUCCESS"
  else
    status=$?
    printf '%s\n' "$status" >"$LEGACY_EP/EXIT_CODE"
    [[ -f "$LEGACY_EP/head_cam_plan.mp4" ]] || exit "$status"
    echo "[warning] legacy run reported failure; preserving its diagnostic video"
  fi
else
  echo "[reuse-legacy-video] $LEGACY_EP/head_cam_plan.mp4"
fi

if ((DRY_RUN)); then exit 0; fi
mkdir -p "$EP_OUT"
"$PY" "$SCRIPT_DIR/compose_replay_method_compare.py" \
  --canonical-episode-root "$CANONICAL_EP" \
  --canonical-human-root "$CANONICAL_HUMAN_EP" \
  --legacy-human-root "$LEGACY_EP" \
  --output "$VIDEO_OUT" \
  --canonical-output "$CANONICAL_VIDEO_OUT" \
  --canonical-approach-offset-m "$CANONICAL_APPROACH_OFFSET_M" \
  --legacy-approach-offset-m "$LEGACY_APPROACH_OFFSET_M" \
  --legacy-target-retreat-m "$LEGACY_TARGET_RETREAT_M"
"$PY" "$SCRIPT_DIR/vscode_video.py" \
  --root "$EP_OUT" --apply --workers 2 --manifest "$EP_OUT/vscode_transcode_manifest.json"
touch "$EP_OUT/SUCCESS"
echo "[success] $EP_OUT"
