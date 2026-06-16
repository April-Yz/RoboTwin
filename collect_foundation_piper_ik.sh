#!/bin/bash

set -euo pipefail

version=${1:-}
foundation_id=${2:-}
foundation_frame=${3:-0}
gpu_id=${4:-0}
foundation_mode=${5:-o1}
run_tag=${6:-}
wrist_left_forward=${WRIST_LEFT_FORWARD_OFFSET_M:-}
wrist_right_forward=${WRIST_RIGHT_FORWARD_OFFSET_M:-}
wrist_left_roll=${WRIST_LEFT_ROLL_DEG:-}
wrist_right_roll=${WRIST_RIGHT_ROLL_DEG:-}
foundation_grasp_standoff=${FOUNDATION_GRASP_STANDOFF_M:-}

if [[ ! "$version" =~ ^v[1-4]$ ]] || [[ ! "$foundation_id" =~ ^[0-9]+$ ]] || \
   [[ ! "$foundation_frame" =~ ^[0-9]+$ ]] || \
   [[ ! "$foundation_mode" =~ ^o1(\.1|\.2)?$ ]] || \
   [[ -n "$run_tag" && ! "$run_tag" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Usage: $0 <v1|v2|v3|v4> <foundation_id> [foundation_frame] [gpu_id] [o1|o1.1|o1.2] [run_tag]" >&2
  exit 2
fi

wrist_override_count=0
for value in "$wrist_left_forward" "$wrist_right_forward" "$wrist_left_roll" "$wrist_right_roll"; do
  [[ -n "$value" ]] && wrist_override_count=$((wrist_override_count + 1))
done
if [[ "$wrist_override_count" -ne 0 && "$wrist_override_count" -ne 4 ]]; then
  echo "Set all four WRIST_LEFT_FORWARD_OFFSET_M, WRIST_RIGHT_FORWARD_OFFSET_M, WRIST_LEFT_ROLL_DEG and WRIST_RIGHT_ROLL_DEG, or set none." >&2
  exit 2
fi
if [[ "$wrist_override_count" -eq 4 ]]; then
  number_re='^-?([0-9]+([.][0-9]*)?|[.][0-9]+)$'
  for value in "$wrist_left_forward" "$wrist_right_forward" "$wrist_left_roll" "$wrist_right_roll"; do
    if [[ ! "$value" =~ $number_re ]]; then
      echo "Invalid wrist tuning number: $value" >&2
      exit 2
    fi
  done
fi
if [[ -n "$foundation_grasp_standoff" ]]; then
  number_re='^([0-9]+([.][0-9]*)?|[.][0-9]+)$'
  if [[ ! "$foundation_grasp_standoff" =~ $number_re ]]; then
    echo "Invalid FOUNDATION_GRASP_STANDOFF_M: $foundation_grasp_standoff" >&2
    exit 2
  fi
fi

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
foundation_root=/home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_replay_d435
input_dir="${foundation_root}/foundation_input_${foundation_id}"
base_name="demo_piper_ik_foundation_${version}"
mode_tag=${foundation_mode//./_}
if [[ "$foundation_mode" == "o1" ]]; then
  config_name="${base_name}_id${foundation_id}_frame${foundation_frame}"
else
  config_name="${base_name}_${mode_tag}_id${foundation_id}"
fi
if [[ -n "$run_tag" ]]; then
  config_name="${config_name}_${run_tag}"
fi
base_config="${repo_dir}/task_config/${base_name}.yml"
generated_config="${repo_dir}/task_config/${config_name}.yml"

if [[ ! -f "${input_dir}/multi_object_world_poses.npz" ]]; then
  echo "Foundation input not found: ${input_dir}/multi_object_world_poses.npz" >&2
  exit 1
fi

sed_args=(
  -e "s|^episode_num:.*|episode_num: 1|" \
  -e "s|^foundation_input_dir:.*|foundation_input_dir: \"${input_dir}\"|" \
  -e "s|^foundation_frame:.*|foundation_frame: ${foundation_frame}|" \
  -e "s|^foundation_mode:.*|foundation_mode: \"${foundation_mode}\"|"
)
if [[ "$wrist_override_count" -eq 4 ]]; then
  sed_args+=(
    -e "s|^    left:.*|    left: {forward_offset_m: ${wrist_left_forward}, image_roll_deg: ${wrist_left_roll}}|"
    -e "s|^    right:.*|    right: {forward_offset_m: ${wrist_right_forward}, image_roll_deg: ${wrist_right_roll}}|"
  )
fi
if [[ -n "$foundation_grasp_standoff" ]]; then
  sed_args+=(
    -e "s|^foundation_grasp_standoff:.*|foundation_grasp_standoff: ${foundation_grasp_standoff}|"
  )
fi
sed -E "${sed_args[@]}" "${base_config}" > "${generated_config}"

echo "[foundation-collect] config=${config_name} mode=${foundation_mode} gpu=${gpu_id} episode_num=1"
if [[ "$wrist_override_count" -eq 4 ]]; then
  echo "[foundation-collect] wrist tuning: left=(${wrist_left_forward}m, ${wrist_left_roll}deg) right=(${wrist_right_forward}m, ${wrist_right_roll}deg)"
fi
if [[ -n "$foundation_grasp_standoff" ]]; then
  echo "[foundation-collect] foundation_grasp_standoff=${foundation_grasp_standoff}m"
fi
cd "${repo_dir}"
bash collect_data.sh pick_diverse_bottles_piper_ik_foundation "${config_name}" "${gpu_id}"
