#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/home/zaijia001/ssd/RoboTwin
RUNNER=${ROOT_DIR}/code_painting/run_hand_retarget_r1_npz_urdfik.sh

# Output root for all datasets.
OUT_ROOT=${ROOT_DIR}/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool
LOG_ROOT=${OUT_ROOT}/logs
FPS=5

# Worker pool (GPU IDs). 4 workers: GPU2 x2, GPU3 x2.
WORKERS=(2 2 3 3)

# Common args forwarded to render_hand_retarget_r1_npz_urdfik.py.
COMMON_ARGS=(
  --require_stored_gripper_pose 1
  --pose_source gripper
  --orientation_remap_label swap_red_blue_keep_green
  --stored_orientation_post_rot_xyz_deg 0 0 0
  --debug_force_orientation none
  --enable_viewer 0
  --viewer_wait_at_end 0
)

# Default dataset list. If script has positional args, they replace this list.
DATASETS=(
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pnp_banana_low/hand_vis
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pnp_pear_apple/hand_vis
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_brown/hand_vis
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_low/hand_vis
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_stack_cup/hand_vis
)

if [[ $# -gt 0 ]]; then
  DATASETS=("$@")
fi

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

run_one() {
  local gpu_id="$1"
  local in_dir="$2"

  if [[ ! -d "${in_dir}" ]]; then
    echo "[skip] missing dataset dir: ${in_dir}" >&2
    return 0
  fi

  local dataset_name
  dataset_name="$(basename "$(dirname "${in_dir}")")"
  local out_dir="${OUT_ROOT}/${dataset_name}"
  local log_file="${LOG_ROOT}/${dataset_name}.log"

  echo "[start] gpu=${gpu_id} dataset=${dataset_name}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    bash "${RUNNER}" \
      "${in_dir}" \
      "${out_dir}" \
      "${FPS}" \
      "${COMMON_ARGS[@]}" \
      > "${log_file}" 2>&1
  echo "[done ] gpu=${gpu_id} dataset=${dataset_name} log=${log_file}"
}

num_workers=${#WORKERS[@]}
num_datasets=${#DATASETS[@]}

if [[ ${num_datasets} -eq 0 ]]; then
  echo "[error] no dataset provided" >&2
  exit 1
fi

for worker_idx in "${!WORKERS[@]}"; do
  gpu_id="${WORKERS[worker_idx]}"
  (
    for dataset_idx in "${!DATASETS[@]}"; do
      (( dataset_idx % num_workers == worker_idx )) || continue
      run_one "${gpu_id}" "${DATASETS[dataset_idx]}"
    done
  ) &
done

wait
echo "[all done] outputs=${OUT_ROOT}"
