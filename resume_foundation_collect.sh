#!/bin/bash
# 恢复 Foundation 采集：只跑缺失 ID（跳过已有 video+hdf5 的）
# 用法: bash resume_foundation_collect.sh <pick_diverse_bottles|pnp_tray> <v1|v2|v3|v4> <gpu_id> [run_tag]
set -euo pipefail

TASK=${1:-}
VERSION=${2:-}
GPU=${3:-}
RUN_TAG=${4:-verified_v2}

if [[ ! "$TASK" =~ ^(pick_diverse_bottles|pnp_tray)$ ]] || \
   [[ ! "$VERSION" =~ ^v[1-4]$ ]] || [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 <pick_diverse_bottles|pnp_tray> <v1|v2|v3|v4> <gpu_id> [run_tag]"
  exit 2
fi

cd /home/zaijia001/ssd/RoboTwin
DATA_DIR="data/${TASK}_piper_ik_foundation"

# Derive the directory prefix from the task name
if [[ "$TASK" == "pnp_tray" ]]; then
  DIR_PREFIX="demo_pnp_tray_piper_ik_foundation"
  # pnp_tray has 51 IDs (0-50)
  MAX_ID=50
else
  DIR_PREFIX="demo_piper_ik_foundation"
  # pick_diverse_bottles has 102 IDs (0-101)
  MAX_ID=101
fi

missing=""
for id in $(seq 0 $MAX_ID); do
  d=$(ls -d "$DATA_DIR"/${DIR_PREFIX}_${VERSION}_o1_2_id${id}_${RUN_TAG} 2>/dev/null || true)
  if [ -z "$d" ]; then
    missing="$missing $id"
  elif [ ! -d "$d/video" ] || [ ! -f "$d/data/episode0_succ.hdf5" ]; then
    rm -rf "$d"  # 清理不完整的
    missing="$missing $id"
  fi
done

if [ -z "$missing" ]; then
  echo "All IDs complete for ${VERSION}!"
  exit 0
fi

count=$(echo "$missing" | wc -w)
echo "=== ${VERSION} missing ${count} IDs (max ${MAX_ID}):${missing} ==="
echo ""

# Allow individual failures without aborting the whole loop
for id in $missing; do
  echo "[$(date +%H:%M:%S)] ${VERSION} id=${id}"
  bash collect_foundation_piper_ik_verified.sh "$TASK" "$VERSION" "$id" "$GPU" "$RUN_TAG" && echo "  -> OK" || echo "  -> FAILED"
done

echo "===== ${VERSION} DONE: $(date) ====="
