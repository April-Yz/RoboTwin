#!/usr/bin/env bash
set -euo pipefail

cd /home/zaijia001/ssd/RoboTwin

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
if [[ "${CONDA_DEFAULT_ENV:-}" != "RoboTwin_bw" ]]; then
  set +u
  conda activate RoboTwin_bw
  set -u
fi

input_npz=${1:-"/home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz"}
output_dir=${2:-"/home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget"}
fps=${3:-"5"}
shift 3 || true

python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_r1_npz.py \
  --input_npz "${input_npz}" \
  --output_dir "${output_dir}" \
  --fps "${fps}" \
  "$@"
