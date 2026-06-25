#!/usr/bin/env bash
set -eo pipefail

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw
cd /home/zaijia001/ssd/RoboTwin

TARGET_COUNT=${TARGET_COUNT:-25}
INITIAL_SPEED=${INITIAL_SPEED:-1.0}
OVERWRITE_MONTAGE=${OVERWRITE_MONTAGE:-1}

ARGS=(
  --tasks place_bread_basket
  --target_count "$TARGET_COUNT"
  --initial_speed "$INITIAL_SPEED"
)

if [[ "$OVERWRITE_MONTAGE" == "1" ]]; then
  ARGS+=(--overwrite_montage)
fi

python /home/zaijia001/ssd/RoboTwin/code_painting/review_l16_ours_montages.py "${ARGS[@]}" "$@"
