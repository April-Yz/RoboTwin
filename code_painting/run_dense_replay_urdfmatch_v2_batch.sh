#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zaijia001/ssd/RoboTwin
INPUT_ROOT=${INPUT_ROOT:-/home/zaijia001/ssd/data/piper/hand}
OUT_ROOT=${OUT_ROOT:-${ROOT}/code_painting/human_replay/h2_pure_d435_urdfmatch_v2}
TASKS=${TASKS:-"pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray"}
IDS=${IDS:-}
GPU=${GPU:-3}
MAX_FRAMES=${MAX_FRAMES:--1}
SKIP_EXISTING=${SKIP_EXISTING:-1}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
TRANSCODE_H264=${TRANSCODE_H264:-1}

LOG_ROOT=${LOG_ROOT:-${OUT_ROOT}/_batch_logs}
mkdir -p "${LOG_ROOT}"
STATUS_FILE=${STATUS_FILE:-${LOG_ROOT}/status.tsv}
touch "${STATUS_FILE}"

id_selected() {
  local candidate=$1
  local token start end
  [[ -z "${IDS}" ]] && return 0
  for token in ${IDS//,/ }; do
    if [[ "${token}" == *-* ]]; then
      start=${token%-*}
      end=${token#*-}
      if (( candidate >= start && candidate <= end )); then
        return 0
      fi
    elif [[ "${candidate}" == "${token}" ]]; then
      return 0
    fi
  done
  return 1
}

output_complete() {
  local output_dir=$1
  local frames
  [[ -s "${output_dir}/zed_replay.mp4" ]] || return 1
  [[ -s "${output_dir}/world_targets_and_status.npz" ]] || return 1
  [[ -s "${output_dir}/dense_replay_v2_metadata.json" ]] || return 1
  [[ -s "${output_dir}/execution_audit.jsonl" ]] || return 1
  frames=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
    -of default=noprint_wrappers=1:nokey=1 "${output_dir}/zed_replay.mp4" 2>/dev/null || true)
  [[ "${frames}" =~ ^[0-9]+$ ]] && (( frames > 0 ))
}

transcode_h264() {
  local video tmp
  for video in "$@"; do
    [[ -s "${video}" ]] || continue
    if [[ "$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "${video}" 2>/dev/null || true)" == "h264" ]]; then
      continue
    fi
    tmp="${video%.mp4}.h264.tmp.mp4"
    ffmpeg -hide_banner -loglevel error -y -i "${video}" -an \
      -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "${tmp}"
    mv "${tmp}" "${video}"
  done
}

if [[ ! -s "${STATUS_FILE}" ]]; then
  printf 'timestamp\ttask\tid\tstatus\tdetail\n' > "${STATUS_FILE}"
fi

for task in ${TASKS}; do
  input_dir=${INPUT_ROOT}/${task}/harmer_output
  if [[ ! -d "${input_dir}" ]]; then
    printf '%s\t%s\t-\tmissing_input_dir\t%s\n' "$(date --iso-8601=seconds)" "${task}" "${input_dir}" >> "${STATUS_FILE}"
    continue
  fi

  while IFS= read -r input_npz; do
    name=$(basename "${input_npz}")
    id=${name#hand_detections_}
    id=${id%.npz}
    [[ "${id}" =~ ^[0-9]+$ ]] || continue
    id_selected "${id}" || continue

    output_dir=${OUT_ROOT}/${task}/id${id}_d435_z005
    if [[ "${FORCE}" != "1" && "${SKIP_EXISTING}" == "1" ]] && output_complete "${output_dir}"; then
      printf '%s\t%s\t%s\tskipped_complete\t%s\n' "$(date --iso-8601=seconds)" "${task}" "${id}" "${output_dir}" >> "${STATUS_FILE}"
      echo "[skip-complete] task=${task} id=${id} output=${output_dir}"
      continue
    fi

    command=(env TASK="${task}" ID="${id}" GPU="${GPU}" MAX_FRAMES="${MAX_FRAMES}" OUT_ROOT="${OUT_ROOT}"
      bash "${ROOT}/code_painting/run_dense_replay_urdfmatch_v2.sh")
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[dry-run]'
      printf ' %q' "${command[@]}"
      printf '\n'
      continue
    fi

    printf '%s\t%s\t%s\tstarted\t%s\n' "$(date --iso-8601=seconds)" "${task}" "${id}" "${output_dir}" >> "${STATUS_FILE}"
    echo "[start] task=${task} id=${id} gpu=${GPU} output=${output_dir}"
    if "${command[@]}"; then
      if [[ "${TRANSCODE_H264}" == "1" ]]; then
        transcode_h264 \
          "${output_dir}/zed_replay.mp4" \
          "${output_dir}/third_replay.mp4" \
          "${output_dir}/left_wrist_replay.mp4" \
          "${output_dir}/right_wrist_replay.mp4" \
          "${output_dir}/zed_depth.mp4"
      fi
      if output_complete "${output_dir}"; then
        printf '%s\t%s\t%s\tcomplete\t%s\n' "$(date --iso-8601=seconds)" "${task}" "${id}" "${output_dir}" >> "${STATUS_FILE}"
        echo "[complete] task=${task} id=${id}"
      else
        printf '%s\t%s\t%s\tincomplete_output\t%s\n' "$(date --iso-8601=seconds)" "${task}" "${id}" "${output_dir}" >> "${STATUS_FILE}"
        echo "[incomplete-output] task=${task} id=${id}" >&2
      fi
    else
      rc=$?
      printf '%s\t%s\t%s\tfailed\trc=%s output=%s\n' "$(date --iso-8601=seconds)" "${task}" "${id}" "${rc}" "${output_dir}" >> "${STATUS_FILE}"
      echo "[failed] task=${task} id=${id} rc=${rc}" >&2
    fi
  done < <(find "${input_dir}" -maxdepth 1 -type f -name 'hand_detections_*.npz' | sort -V)
done

echo "Dense Replay URDF-match v2 batch finished. Status: ${STATUS_FILE}"
