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
output_root=${4:-"/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch"}
shift 4 || true

hand_keyframes_json="${hand_dir}/hand_keyframes_all.json"
video_prefix="${VIDEO_PREFIX:-d_pour_blue}"
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
    --hand_keyframes_json)
      shift
      hand_keyframes_json="$1"
      shift
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
  for path in "${root}/${video_prefix}"_*; do
    [[ -d "${path}" ]] || continue
    basename "${path}" | sed -E "s/^${video_prefix}_([0-9]+)$/\\1/"
  done | sort -n
}

annotation_check() {
  local json_path="$1"
  local id="$2"
  python - "${json_path}" "${id}" <<'PY'
import json
import sys

json_path, sid = sys.argv[1], int(sys.argv[2])
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
videos = data.get("videos", data)
info = videos.get(f"hand_vis_{sid}.mp4")
if not isinstance(info, dict):
    print("missing 0")
    raise SystemExit(0)
status = str(info.get("status", "done")).lower()
keyframes = [int(v) for v in info.get("keyframes", [])]
print(f"{status} {len(keyframes)}")
PY
}

if ((${#ids[@]} == 0)); then
  mapfile -t ids < <(collect_ids_from_root "${anygrasp_root}")
fi

echo "[run-anygrasp-preview-keyframes-batch] anygrasp_root=${anygrasp_root}"
echo "[run-anygrasp-preview-keyframes-batch] replay_root=${replay_root}"
echo "[run-anygrasp-preview-keyframes-batch] hand_dir=${hand_dir}"
echo "[run-anygrasp-preview-keyframes-batch] hand_keyframes_json=${hand_keyframes_json}"
echo "[run-anygrasp-preview-keyframes-batch] video_prefix=${video_prefix}"
echo "[run-anygrasp-preview-keyframes-batch] output_root=${output_root}"
echo "[run-anygrasp-preview-keyframes-batch] ids=${ids[*]:-}"

if ((${#ids[@]} == 0)); then
  echo "[run-anygrasp-preview-keyframes-batch] no ids found under ${anygrasp_root}" >&2
  exit 1
fi
if [[ ! -f "${hand_keyframes_json}" ]]; then
  echo "[run-anygrasp-preview-keyframes-batch] missing hand_keyframes_json=${hand_keyframes_json}" >&2
  exit 1
fi

mkdir -p "${output_root}"

for id in "${ids[@]}"; do
  video_name="${video_prefix}_${id}"
  anygrasp_dir="${anygrasp_root}/${video_name}"
  replay_dir="${replay_root}/${video_name}"
  hand_npz="${hand_dir}/hand_detections_${id}.npz"
  base_image_dir="${replay_dir}/head_anygrasp_frames"
  output_dir="${output_root}/${video_name}"
  read -r annotation_status annotation_keyframe_count < <(annotation_check "${hand_keyframes_json}" "${id}")

  if [[ "${annotation_status}" == "missing" ]]; then
    echo "[run-anygrasp-preview-keyframes-batch] skip id=${id} missing annotation key hand_vis_${id}.mp4 in ${hand_keyframes_json}" >&2
    continue
  fi
  if [[ "${annotation_status}" == "reject" || "${annotation_status}" == "discard" || "${annotation_status}" == "bad" ]]; then
    echo "[run-anygrasp-preview-keyframes-batch] skip id=${id} annotation_status=${annotation_status}" >&2
    continue
  fi
  if ((annotation_keyframe_count < 2)); then
    echo "[run-anygrasp-preview-keyframes-batch] skip id=${id} annotation has ${annotation_keyframe_count} keyframes; need at least 2" >&2
    continue
  fi

  if [[ ! -d "${anygrasp_dir}" ]]; then
    echo "[run-anygrasp-preview-keyframes-batch] skip id=${id} missing anygrasp_dir=${anygrasp_dir}" >&2
    continue
  fi
  if [[ ! -d "${replay_dir}" ]]; then
    echo "[run-anygrasp-preview-keyframes-batch] skip id=${id} missing replay_dir=${replay_dir}" >&2
    continue
  fi
  if [[ ! -d "${base_image_dir}" ]]; then
    echo "[run-anygrasp-preview-keyframes-batch] skip id=${id} missing base_image_dir=${base_image_dir}" >&2
    continue
  fi
  if [[ ! -f "${hand_npz}" ]]; then
    echo "[run-anygrasp-preview-keyframes-batch] skip id=${id} missing hand_npz=${hand_npz}" >&2
    continue
  fi

  echo "[run-anygrasp-preview-keyframes-batch] processing id=${id} video=${video_name}"
  /home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
    /home/zaijia001/ssd/RoboTwin/code_painting/render_anygrasp_ranked_preview.py \
    --anygrasp_dir "${anygrasp_dir}" \
    --replay_dir "${replay_dir}" \
    --hand_npz "${hand_npz}" \
    --base_image_dir "${base_image_dir}" \
    --base_image_mode raw \
    --output_dir "${output_dir}" \
    --frame_selection_mode hand_keyframes_json \
    --hand_keyframes_json "${hand_keyframes_json}" \
    "${forward_args[@]}"
done
