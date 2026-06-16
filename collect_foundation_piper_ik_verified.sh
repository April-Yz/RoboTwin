#!/bin/bash

set -euo pipefail

foundation_task=${1:-}
version=${2:-}
foundation_id=${3:-}
gpu_id=${4:-0}
run_tag=${5:-verified_v2}
dry_run=${DRY_RUN:-0}

if [[ ! "$foundation_task" =~ ^(pick_diverse_bottles|pnp_tray)$ ]] || \
   [[ ! "$version" =~ ^v[1-4]$ ]] || [[ ! "$foundation_id" =~ ^[0-9]+$ ]] || \
   [[ ! "$gpu_id" =~ ^[0-9]+$ ]] || [[ ! "$run_tag" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Usage: $0 <pick_diverse_bottles|pnp_tray> <v1|v2|v3|v4> <foundation_id> [gpu_id] [run_tag]" >&2
  echo "Optional: DRY_RUN=1 $0 ..." >&2
  exit 2
fi

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mode_tag=o1_2

if [[ "$foundation_task" == "pick_diverse_bottles" ]]; then
  task_name=pick_diverse_bottles_piper_ik_foundation
  base_name="demo_piper_ik_foundation_${version}"
  foundation_root=/home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/foundation_replay_d435
elif [[ "$foundation_task" == "pnp_tray" ]]; then
  task_name=pnp_tray_piper_ik_foundation
  base_name="demo_pnp_tray_piper_ik_foundation_${version}"
  foundation_root=/home/zaijia001/ssd/data/piper/hand/pnp_tray/foundation_replay_d435
fi

input_dir="${foundation_root}/foundation_input_${foundation_id}"
base_config="${repo_dir}/task_config/${base_name}.yml"
config_name="${base_name}_${mode_tag}_id${foundation_id}_${run_tag}"
generated_config="${repo_dir}/task_config/${config_name}.yml"

if [[ ! -f "${base_config}" ]]; then
  echo "Base config not found: ${base_config}" >&2
  exit 1
fi
if [[ ! -f "${input_dir}/multi_object_world_poses.npz" ]]; then
  echo "Foundation input not found: ${input_dir}/multi_object_world_poses.npz" >&2
  exit 1
fi

export BASE_CONFIG="${base_config}"
export GENERATED_CONFIG="${generated_config}"
export INPUT_DIR="${input_dir}"
export FOUNDATION_TASK="${foundation_task}"
python - <<'PY'
import os
import yaml

base_config = os.environ["BASE_CONFIG"]
generated_config = os.environ["GENERATED_CONFIG"]
input_dir = os.environ["INPUT_DIR"]
foundation_task = os.environ["FOUNDATION_TASK"]

with open(base_config, "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

grasp_standoff = 0.105 if foundation_task == "pnp_tray" else 0.14

config.update(
    {
        "episode_num": 1,
        "max_seed_tries": 3,
        "render_freq": 0,
        "foundation_input_dir": input_dir,
        "foundation_mode": "o1.2",
        "foundation_frame": 0,
        "foundation_collision_mode": "support_proxy",
        "foundation_collision_radius_padding": 0.0,
        "foundation_grasp_standoff": grasp_standoff,
        "foundation_grasp_lateral_offset": 0.0,
        "foundation_pregrasp_distance": 0.12,
        "foundation_grasp_assist": True,
        "foundation_grasp_assist_max_distance": 0.16,
        "foundation_capture_radial_tolerance": 0.08,
        "foundation_grasp_require_contact": False,
        "collect_data": True,
        "eval_video_log": True,
        "skip_planner": True,
        "save_all_episodes": False,
    }
)
if foundation_task == "pnp_tray":
    config["foundation_open_after_action"] = True

camera = config.setdefault("camera", {})
camera.update(
    {
        "collect_head_camera": True,
        "collect_wrist_camera": True,
        "wrist_camera_calibration_bundle": "/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json",
        "wrist_camera_axis_mode": "legacy_r1",
        "wrist_camera_pose_reference": "urdf_end_link",
        "wrist_camera_simulation_adapter": "piper_pika_agx",
    }
)
camera["wrist_camera_tuning"] = {
    "left": {
        "forward_offset_m": 0.145,
        "image_roll_deg": -15.0,
        "parent_yaw_deg": 0.182,
        "parent_pitch_deg": 15.0,
        "parent_lateral_offset_m": -0.0207,
    },
    "right": {
        "forward_offset_m": 0.13,
        "image_roll_deg": -60.0,
        "parent_yaw_deg": 0.840,
        "parent_pitch_deg": 15.0,
        "parent_lateral_offset_m": 0.0274,
    },
}

with open(generated_config, "w", encoding="utf-8") as file:
    yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

print(f"[foundation-verified-collect] wrote {generated_config}")
print(
    "[foundation-verified-collect] verified params: "
    f"standoff={grasp_standoff}, radial=0.08, assist_max_distance=0.16, "
    "support_proxy+assist, calibrated head+wrist videos"
)
PY

echo "[foundation-verified-collect] task=${task_name} config=${config_name} gpu=${gpu_id}"
if [[ "$dry_run" == "1" ]]; then
  echo "[foundation-verified-collect] DRY_RUN=1; not running collect_data.sh"
  exit 0
fi

cd "${repo_dir}"
bash collect_data.sh "${task_name}" "${config_name}" "${gpu_id}"
