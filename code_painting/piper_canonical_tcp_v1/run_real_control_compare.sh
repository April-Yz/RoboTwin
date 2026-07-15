#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY=/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python
DATA=/home/zaijia001/ssd/data/piper/hand
VIS_ROOT="$DATA/vis"
VIS_SCRIPT="$VIS_ROOT/render_piper_pos_compare.py"
SIM_SCRIPT="$VIS_ROOT/render_piper_sim_frames.py"
URDF="$REPO/assets/embodiments/piper_pika_agx/piper_pika_agx.urdf"
CALIBRATION="$REPO/calibration_bundle_piper_new_table_0515.json"

TASK=""
EPISODE=""
GPU=0
MAX_FRAMES=0
OUTPUT_ROOT="$SCRIPT_DIR/outputs_real_control_compare_20260716"
DRY_RUN=0

usage() {
  echo "Usage: $0 --task TASK --episode episodeN [--gpu N] [--max-frames N] [--output-root PATH] [--dry-run]"
}

while (($#)); do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --episode) EPISODE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TASK" && -n "$EPISODE" ]] || { usage >&2; exit 2; }

SOURCE="$VIS_ROOT/.pos_source/$TASK/$EPISODE"
DIRECT_SIM_SOURCE="$VIS_ROOT/.sim_source/$TASK/$EPISODE"
OUT="$OUTPUT_ROOT/$TASK/$EPISODE"
PLAN="$OUT/control_plan.npz"
OURS_SIM="$OUT/sim_oursv2"
CANONICAL_SIM="$OUT/sim_canonical"

for path in "$SOURCE"; do
  [[ -d "$path" ]] || { echo "[error] missing directory: $path" >&2; exit 3; }
done
for path in "$URDF" "$CALIBRATION" "$VIS_SCRIPT" "$SIM_SCRIPT" \
  "$SCRIPT_DIR/tag_control_sim_manifest.py"; do
  [[ -f "$path" ]] || { echo "[error] missing file: $path" >&2; exit 3; }
done
if [[ -e "$OUT/SUCCESS" ]]; then
  echo "[skip-existing-success] $OUT"
  exit 0
fi
if [[ -d "$OUT" ]] && find "$OUT" -mindepth 1 -print -quit | grep -q .; then
  echo "[error] refusing non-empty output without SUCCESS: $OUT" >&2
  exit 4
fi

PLAN_CMD=(env CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$SCRIPT_DIR/plan_real_control_compare.py"
  --task "$TASK" --episode "$EPISODE" --episode-dir "$SOURCE"
  --output-dir "$OUT" --urdf "$URDF" --calibration "$CALIBRATION"
  --vis-script "$VIS_SCRIPT" --max-frames "$MAX_FRAMES")
if ((DRY_RUN)); then
  printf '%q ' "${PLAN_CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$OUT"
record_failure() {
  status=$?
  if ((status != 0)); then
    printf '%s\n' "$status" >"$OUT/EXIT_CODE"
    echo "[failed] status=$status output=$OUT" >&2
  fi
}
trap record_failure EXIT
printf '%q ' "${PLAN_CMD[@]}" >"$OUT/command.sh.txt"; printf '\n' >>"$OUT/command.sh.txt"
echo "[plan] $TASK/$EPISODE GPU=$GPU max_frames=$MAX_FRAMES"
"${PLAN_CMD[@]}" > >(tee "$OUT/plan.stdout.log") 2> >(tee "$OUT/plan.stderr.log" >&2)

DIRECT_SIM="$DIRECT_SIM_SOURCE"
if [[ ! -d "$DIRECT_SIM_SOURCE" ]] || ! find "$DIRECT_SIM_SOURCE" -maxdepth 1 -type f -name '*.jpg' -print -quit | grep -q .; then
  DIRECT_SIM="$OUT/sim_direct"
  echo "[render-direct] $TASK/$EPISODE"
  "$PY" "$SIM_SCRIPT" --task "$TASK" --episode "$EPISODE" \
    --episode-dir "$SOURCE" --output-dir "$DIRECT_SIM" --urdf "$URDF" \
    --calibration "$CALIBRATION" --vis-script "$VIS_SCRIPT" \
    --max-frames "$MAX_FRAMES"
fi

for branch in oursv2 canonical; do
  echo "[render-$branch] $TASK/$EPISODE"
  "$PY" "$SIM_SCRIPT" --task "$TASK" --episode "$EPISODE" \
    --episode-dir "$SOURCE" --output-dir "$OUT/sim_$branch" --urdf "$URDF" \
    --calibration "$CALIBRATION" --vis-script "$VIS_SCRIPT" \
    --planned-npz "$OUT/${branch}_renderer.npz" --max-frames "$MAX_FRAMES"
  "$PY" "$SCRIPT_DIR/tag_control_sim_manifest.py" \
    --manifest "$OUT/sim_$branch/manifest.json" --branch "$branch"
done

for mode in joint eepose; do
  echo "[compose-$mode] $TASK/$EPISODE"
  "$PY" "$SCRIPT_DIR/compose_real_control_compare.py" \
    --task "$TASK" --episode "$EPISODE" --mode "$mode" --episode-dir "$SOURCE" \
    --plan "$PLAN" --output "$OUT/${mode}_control.mp4" \
    --manifest "$OUT/${mode}_control.manifest.json" --urdf "$URDF" \
    --calibration "$CALIBRATION" --vis-script "$VIS_SCRIPT" \
    --direct-sim-dir "$DIRECT_SIM" --ours-sim-dir "$OURS_SIM" \
    --canonical-sim-dir "$CANONICAL_SIM"
done
"$PY" "$SCRIPT_DIR/vscode_video.py" --root "$OUT" \
  --manifest "$OUT/vscode_video_check.json"
touch "$OUT/SUCCESS"
trap - EXIT
echo "[success] $OUT"
