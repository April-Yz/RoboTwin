#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zaijia001/ssd/RoboTwin
TASK=${TASK:-pick_diverse_bottles}
ID=${ID:-0}
GPU=${GPU:-3}
MAX_FRAMES=${MAX_FRAMES:--1}
OUT_ROOT=${OUT_ROOT:-${ROOT}/code_painting/human_replay/h2_pure_d435_urdfmatch_v2}
OUT=${OUT_ROOT}/${TASK}/id${ID}_d435_z005
INPUT=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output/hand_detections_${ID}.npz

if [[ ! -f "${INPUT}" ]]; then
  echo "Missing input: ${INPUT}" >&2
  exit 1
fi

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
cd "${ROOT}"

CUDA_VISIBLE_DEVICES=${GPU} conda run -n RoboTwin_bw python \
  code_painting/render_hand_retarget_piper_dual_npz_urdfmatch_v2_main.py \
  --input_npz "${INPUT}" \
  --output_dir "${OUT}" \
  --image_width 640 \
  --image_height 480 \
  --fovy_deg 42.499880046655484 \
  --fps 5 \
  --max_frames "${MAX_FRAMES}" \
  --arms both \
  --piper_calibration_bundle "${ROOT}/calibration_bundle_piper_new_table_0515.json" \
  --camera_cv_axis_mode legacy_r1 \
  --require_stored_gripper_pose 1 \
  --pose_source gripper \
  --orientation_remap_label identity \
  --stored_orientation_post_rot_xyz_deg 0 0 0 \
  --target_local_forward_retreat_m 0.05 \
  --target_world_offset_xyz 0 0.1 0.1 \
  --execute_waypoint_scene_steps 5 \
  --execute_settle_scene_steps 20 \
  --urdfik_joint_interp_waypoints 10 \
  --debug_mode 0 \
  --debug_post_execute 0 \
  --debug_frame_limit -1 \
  --debug_visualize_targets 0 \
  --debug_visualize_cameras 0 \
  --clean_output 1 \
  --overlay_text_enable 0 \
  --save_png_frames 0 \
  --lighting_mode front_no_shadow

echo "Dense Replay v2 output: ${OUT}"
