#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$SCRIPT_DIR/outputs_ik_semantic_grid_v2_20260716"
GPU=0
PROFILE=both
DRY_RUN=0

while (($#)); do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) echo "Usage: $0 [--output-root PATH] [--gpu N] [--profile d435|wide|both] [--dry-run]"; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$PROFILE" =~ ^(d435|wide|both)$ ]] || { echo "[error] invalid profile=$PROFILE" >&2; exit 2; }

TASK_SPECS=(
  pick_diverse_bottles:0
  place_bread_basket:0
  stack_cups:0
  handover_bottle:1
  pnp_bread:7
  pnp_tray:0
)
if [[ "$PROFILE" == both ]]; then PROFILES=(d435 wide); else PROFILES=("$PROFILE"); fi

STATUS_DIR="$OUTPUT_ROOT/_batch"
STATUS_FILE="$STATUS_DIR/camera_profiles_status.tsv"
if ((!DRY_RUN)); then
  mkdir -p "$STATUS_DIR" "$OUTPUT_ROOT/vis"
  printf 'profile\ttask\tid\tstatus\tvideo\n' >"$STATUS_FILE"
fi

failures=0
for profile in "${PROFILES[@]}"; do
  for spec in "${TASK_SPECS[@]}"; do
    task="${spec%%:*}"; id="${spec##*:}"
    video="$OUTPUT_ROOT/vis/${task}_id${id}_v${profile}.mp4"
    cmd=(bash "$SCRIPT_DIR/run_ik_logic_grid.sh" --task "$task" --id "$id" --arm auto --gpu "$GPU" --camera-profile "$profile" --output-root "$OUTPUT_ROOT")
    ((DRY_RUN)) && cmd+=(--dry-run)
    printf '[batch] profile=%s task=%s id=%s video=%s\n' "$profile" "$task" "$id" "$video"
    if "${cmd[@]}"; then
      ((!DRY_RUN)) && printf '%s\t%s\t%s\tcomplete\t%s\n' "$profile" "$task" "$id" "$video" >>"$STATUS_FILE"
    else
      status=$?; failures=$((failures + 1))
      ((!DRY_RUN)) && printf '%s\t%s\t%s\tfailed:%s\t%s\n' "$profile" "$task" "$id" "$status" "$video" >>"$STATUS_FILE"
    fi
  done
done

((DRY_RUN)) && exit 0
if ((failures)); then
  echo "[batch-failed] failures=$failures status=$STATUS_FILE" >&2
  exit 1
fi
touch "$STATUS_DIR/SUCCESS"
echo "[batch-success] videos=$OUTPUT_ROOT/vis status=$STATUS_FILE"
