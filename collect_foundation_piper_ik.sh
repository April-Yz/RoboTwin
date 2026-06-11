#!/bin/bash

set -euo pipefail

version=${1:-}
foundation_id=${2:-}
foundation_frame=${3:-0}
gpu_id=${4:-0}
foundation_mode=${5:-o1}

if [[ ! "$version" =~ ^v[1-4]$ ]] || [[ ! "$foundation_id" =~ ^[0-9]+$ ]] || \
   [[ ! "$foundation_frame" =~ ^[0-9]+$ ]] || \
   [[ ! "$foundation_mode" =~ ^o1(\.1|\.2)?$ ]]; then
  echo "Usage: $0 <v1|v2|v3|v4> <foundation_id> [foundation_frame] [gpu_id] [o1|o1.1|o1.2]" >&2
  exit 2
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
base_config="${repo_dir}/task_config/${base_name}.yml"
generated_config="${repo_dir}/task_config/${config_name}.yml"

if [[ ! -f "${input_dir}/multi_object_world_poses.npz" ]]; then
  echo "Foundation input not found: ${input_dir}/multi_object_world_poses.npz" >&2
  exit 1
fi

sed -E \
  -e "s|^foundation_input_dir:.*|foundation_input_dir: \"${input_dir}\"|" \
  -e "s|^foundation_frame:.*|foundation_frame: ${foundation_frame}|" \
  -e "s|^foundation_mode:.*|foundation_mode: \"${foundation_mode}\"|" \
  "${base_config}" > "${generated_config}"

echo "[foundation-collect] config=${config_name} mode=${foundation_mode} gpu=${gpu_id}"
cd "${repo_dir}"
bash collect_data.sh pick_diverse_bottles_piper_ik_foundation "${config_name}" "${gpu_id}"
