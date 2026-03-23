#!/usr/bin/env bash
set -euo pipefail

cd /home/zaijia001/ssd/RoboTwin

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
if [[ "${CONDA_DEFAULT_ENV:-}" != "RoboTwin_bw" ]]; then
  set +u
  conda activate RoboTwin_bw
  set -u
fi

anygrasp_root=${1:-"/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results"}
replay_root=${2:-"/home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot"}
hand_dir=${3:-"/home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis"}
output_root=${4:-"/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_batch"}
shift 4 || true

ids=()
forward_args=()
while (($# > 0)); do
  case "$1" in
    --ids)
      shift
      while (($# > 0)); do
        if [[ "$1" == --* ]]; then
          break
        fi
        ids+=("$1")
        shift
      done
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

collect_ids_from_root() {
  local root="$1"
  local path=""
  for path in "${root}"/d_pour_blue_*; do
    [[ -d "${path}" ]] || continue
    basename "${path}" | sed -E 's/^d_pour_blue_([0-9]+)$/\1/'
  done | sort -n
}

if ((${#ids[@]} == 0)); then
  mapfile -t ids < <(collect_ids_from_root "${anygrasp_root}")
fi

echo "[run-anygrasp-preview-batch] anygrasp_root=${anygrasp_root}"
echo "[run-anygrasp-preview-batch] replay_root=${replay_root}"
echo "[run-anygrasp-preview-batch] hand_dir=${hand_dir}"
echo "[run-anygrasp-preview-batch] output_root=${output_root}"
echo "[run-anygrasp-preview-batch] ids=${ids[*]:-}"

if ((${#ids[@]} == 0)); then
  echo "[run-anygrasp-preview-batch] no ids found under ${anygrasp_root}" >&2
  exit 1
fi

mkdir -p "${output_root}"

for id in "${ids[@]}"; do
  video_name="d_pour_blue_${id}"
  anygrasp_dir="${anygrasp_root}/${video_name}"
  replay_dir="${replay_root}/${video_name}"
  hand_npz="${hand_dir}/hand_detections_${id}.npz"
  base_image_dir="${replay_dir}/head_anygrasp_frames"
  output_dir="${output_root}/${video_name}"

  if [[ ! -d "${anygrasp_dir}" ]]; then
    echo "[run-anygrasp-preview-batch] skip id=${id} missing anygrasp_dir=${anygrasp_dir}" >&2
    continue
  fi
  if [[ ! -d "${replay_dir}" ]]; then
    echo "[run-anygrasp-preview-batch] skip id=${id} missing replay_dir=${replay_dir}" >&2
    continue
  fi
  if [[ ! -d "${base_image_dir}" ]]; then
    echo "[run-anygrasp-preview-batch] skip id=${id} missing base_image_dir=${base_image_dir}" >&2
    continue
  fi
  if [[ ! -f "${hand_npz}" ]]; then
    echo "[run-anygrasp-preview-batch] skip id=${id} missing hand_npz=${hand_npz}" >&2
    continue
  fi

  echo "[run-anygrasp-preview-batch] processing id=${id} video=${video_name}"
  /home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
    /home/zaijia001/ssd/RoboTwin/code_painting/render_anygrasp_ranked_preview.py \
    --anygrasp_dir "${anygrasp_dir}" \
    --replay_dir "${replay_dir}" \
    --hand_npz "${hand_npz}" \
    --base_image_dir "${base_image_dir}" \
    --base_image_mode raw \
    --output_dir "${output_dir}" \
    "${forward_args[@]}"
done
