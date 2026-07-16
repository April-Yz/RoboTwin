#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10
TASK=""; ID=""; ARM=auto; GPU=0; CAMERA_PROFILE=d435
OUTPUT_ROOT="$SCRIPT_DIR/outputs_ik_semantic_grid_v2_20260716"
DRY_RUN=0

while (($#)); do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --id) ID="$2"; shift 2 ;;
    --arm) ARM="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --camera-profile) CAMERA_PROFILE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) echo "Usage: $0 --task TASK --id ID [--camera-profile d435|wide] [--arm auto|left|right] [--gpu N] [--output-root PATH] [--dry-run]"; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TASK" && -n "$ID" ]] || { echo "[error] --task and --id are required" >&2; exit 2; }
[[ "$CAMERA_PROFILE" =~ ^(d435|wide)$ ]] || { echo "[error] invalid camera-profile=$CAMERA_PROFILE" >&2; exit 2; }

LEGACY_ROOT="$OUTPUT_ROOT/_sources/$CAMERA_PROFILE/legacy_original"
CANONICAL_ROOT="$OUTPUT_ROOT/_sources/$CAMERA_PROFILE/canonical_rtcp"
EP_OUT="$OUTPUT_ROOT/_grid_meta/$CAMERA_PROFILE/$TASK/foundation_input_${ID}"
VIDEO_OUT="$OUTPUT_ROOT/vis/${TASK}_id${ID}_v${CAMERA_PROFILE}.mp4"
AUDIT_OUT="$EP_OUT/semantic_source_audit.json"
MANIFEST_OUT="$EP_OUT/grid_manifest.json"

run_cell() {
  local logic="$1" strategy="$2" root="$3"
  local runner="$SCRIPT_DIR/run_ik_logic_strategy.sh"
  [[ "$strategy" == human_replay ]] && runner="$SCRIPT_DIR/run_ik_logic_human_replay.sh"
  local cmd=(bash "$runner" --task "$TASK" --id "$ID" --ik-logic "$logic" --gpu "$GPU" --output-root "$root" --camera-profile "$CAMERA_PROFILE")
  [[ "$strategy" != human_replay ]] && cmd+=(--arm "$ARM" --strategy "$strategy")
  ((DRY_RUN)) && cmd+=(--dry-run)
  if ! "${cmd[@]}"; then
    local video="$root/$TASK/foundation_input_${ID}/eepose/$strategy/head_cam_plan.mp4"
    [[ -f "$video" ]] || { echo "[error] $logic/$strategy failed before producing diagnostic video" >&2; return 5; }
    echo "[warning] $logic/$strategy execution failed; preserving diagnostic video"
  fi
}

for logic in legacy canonical; do
  if [[ "$logic" == legacy ]]; then root="$LEGACY_ROOT"; else root="$CANONICAL_ROOT"; fi
  for strategy in orientation fused top_score human_replay; do
    run_cell "$logic" "$strategy" "$root"
  done
done
((DRY_RUN)) && exit 0

mkdir -p "$EP_OUT"
"$PY" "$SCRIPT_DIR/audit_ik_logic_inputs.py" \
  --legacy-root "$LEGACY_ROOT" --canonical-root "$CANONICAL_ROOT" \
  --task "$TASK" --id "$ID" --camera-profile "$CAMERA_PROFILE" --output "$AUDIT_OUT"
"$PY" "$SCRIPT_DIR/compose_ik_logic_grid.py" \
  --legacy-root "$LEGACY_ROOT" --canonical-root "$CANONICAL_ROOT" \
  --task "$TASK" --id "$ID" --camera-profile "$CAMERA_PROFILE" --input-audit "$AUDIT_OUT" \
  --output "$VIDEO_OUT" --manifest "$MANIFEST_OUT"
touch "$EP_OUT/SUCCESS"
echo "[success] camera=$CAMERA_PROFILE video=$VIDEO_OUT"
