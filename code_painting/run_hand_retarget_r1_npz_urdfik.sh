#!/usr/bin/env bash
set -euo pipefail

cd /home/zaijia001/ssd/RoboTwin

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
if [[ "${CONDA_DEFAULT_ENV:-}" != "RoboTwin_bw" ]]; then
  set +u
  conda activate RoboTwin_bw
  set -u
fi

if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

input_npz=${1:-"/home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz"}
output_dir=${2:-"/home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_urdfik"}
fps=${3:-"5"}
shift 3 || true

run_single() {
  local npz_path="$1"
  local npz_output_dir="$2"
  shift 2

  echo "[run] input=${npz_path}"
  echo "[run] output=${npz_output_dir}"

  python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_r1_npz_urdfik.py \
    --input_npz "${npz_path}" \
    --output_dir "${npz_output_dir}" \
    --fps "${fps}" \
    "$@"
}

if [[ -d "${input_npz}" ]]; then
  mapfile -t npz_files < <(find "${input_npz}" -maxdepth 1 -type f -name '*.npz' | sort)
  if [[ ${#npz_files[@]} -eq 0 ]]; then
    echo "[error] no .npz files found in directory: ${input_npz}" >&2
    exit 1
  fi

  mkdir -p "${output_dir}"
  for npz_path in "${npz_files[@]}"; do
    npz_name=$(basename "${npz_path}")
    npz_stem="${npz_name%.npz}"
    run_single "${npz_path}" "${output_dir}/${npz_stem}" "$@"
  done
else
  run_single "${input_npz}" "${output_dir}" "$@"
fi
