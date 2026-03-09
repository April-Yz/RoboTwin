#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/home/zaijia001/ssd/RoboTwin
RUNNER=${ROOT_DIR}/code_painting/run_hand_retarget_r1_npz_urdfik.sh

# Output root for all datasets (actual out dir suffix is controlled by VARIANT).
OUT_ROOT_BASE=${ROOT_DIR}/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool
FPS=5

# Worker pool (GPU IDs). Default: GPU2 x2, GPU3 x2.
# Can be overridden by: --workers 2,2,3,3  or --workers 2,2,2,2,2,2
WORKERS=(2 2 3 3)
VARIANT=clean
OUT_ROOT=""
LIGHTING_MODE=front_no_shadow
SMOOTH_MODE=0
SMOOTH_INTERP_FRAMES=1

# Common args forwarded to render_hand_retarget_r1_npz_urdfik.py.
COMMON_ARGS=(
  --require_stored_gripper_pose 1
  --pose_source gripper
  --orientation_remap_label swap_red_blue_keep_green
  --stored_orientation_post_rot_xyz_deg 0 0 0
  --lighting_mode "${LIGHTING_MODE}"
  --debug_force_orientation none
)

# Default dataset list. If script has positional args, they replace this list.
DEFAULT_DATASETS=(
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pnp_banana_low/hand_vis
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pnp_pear_apple/hand_vis
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_brown/hand_vis
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_low/hand_vis
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_stack_cup/hand_vis
)

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [--workers W] [--variant clean|debug] [--lighting_mode MODE] [--smooth_mode 0|1] [--smooth_interp_frames N] [--out_root DIR] [hand_vis_dir ...]

Options:
  --workers   Comma-separated GPU worker list (default: 2,2,3,3)
              Example: --workers 2,2,2,2,2,2
  --variant   clean (default) or debug
  --lighting_mode  default|front|front_no_shadow (default: front_no_shadow)
  --smooth_mode  Enable slow/smooth replay (default: 0)
  --smooth_interp_frames  Interpolated frames between NPZ frames when smooth mode is on (default: 1)
  --out_root  Custom output root; defaults to \${OUT_ROOT_BASE}_<variant>
EOF
}

DATASETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers)
      if [[ $# -lt 2 ]]; then
        echo "[error] --workers requires a value" >&2
        exit 1
      fi
      IFS=',' read -r -a WORKERS <<< "$2"
      shift 2
      ;;
    --variant)
      if [[ $# -lt 2 ]]; then
        echo "[error] --variant requires a value" >&2
        exit 1
      fi
      VARIANT="$2"
      shift 2
      ;;
    --out_root)
      if [[ $# -lt 2 ]]; then
        echo "[error] --out_root requires a value" >&2
        exit 1
      fi
      OUT_ROOT="$2"
      shift 2
      ;;
    --lighting_mode)
      if [[ $# -lt 2 ]]; then
        echo "[error] --lighting_mode requires a value" >&2
        exit 1
      fi
      LIGHTING_MODE="$2"
      shift 2
      ;;
    --smooth_mode)
      if [[ $# -lt 2 ]]; then
        echo "[error] --smooth_mode requires a value" >&2
        exit 1
      fi
      SMOOTH_MODE="$2"
      shift 2
      ;;
    --smooth_interp_frames)
      if [[ $# -lt 2 ]]; then
        echo "[error] --smooth_interp_frames requires a value" >&2
        exit 1
      fi
      SMOOTH_INTERP_FRAMES="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      DATASETS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#WORKERS[@]} -eq 0 ]]; then
  echo "[error] workers list is empty" >&2
  exit 1
fi

if [[ "${VARIANT}" != "clean" && "${VARIANT}" != "debug" ]]; then
  echo "[error] --variant must be clean or debug" >&2
  exit 1
fi

if [[ "${LIGHTING_MODE}" != "default" && "${LIGHTING_MODE}" != "front" && "${LIGHTING_MODE}" != "front_no_shadow" ]]; then
  echo "[error] --lighting_mode must be one of: default, front, front_no_shadow" >&2
  exit 1
fi

if ! [[ "${SMOOTH_MODE}" =~ ^[0-9]+$ ]]; then
  echo "[error] --smooth_mode must be integer 0/1" >&2
  exit 1
fi
if ! [[ "${SMOOTH_INTERP_FRAMES}" =~ ^[0-9]+$ ]]; then
  echo "[error] --smooth_interp_frames must be integer >=0" >&2
  exit 1
fi

# Ensure COMMON_ARGS uses the final CLI-selected lighting mode.
for i in "${!COMMON_ARGS[@]}"; do
  if [[ "${COMMON_ARGS[$i]}" == "--lighting_mode" ]]; then
    COMMON_ARGS[$((i + 1))]="${LIGHTING_MODE}"
    break
  fi
done

if [[ "${SMOOTH_MODE}" -gt 0 ]]; then
  COMMON_ARGS+=(
    --smooth_mode 1
    --smooth_interp_frames "${SMOOTH_INTERP_FRAMES}"
  )
fi

if [[ -z "${OUT_ROOT}" ]]; then
  OUT_ROOT="${OUT_ROOT_BASE}_${VARIANT}"
fi
LOG_ROOT=${OUT_ROOT}/logs

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi

if [[ "${VARIANT}" == "clean" ]]; then
  COMMON_ARGS+=(
    --clean_output 1
    --debug_visualize_targets 0
    --enable_viewer 0
    --viewer_wait_at_end 0
  )
else
  COMMON_ARGS+=(
    --clean_output 0
    --enable_viewer 0
    --viewer_wait_at_end 0
  )
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
echo "[all done] variant=${VARIANT} outputs=${OUT_ROOT}"
