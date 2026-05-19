#!/usr/bin/env bash
set -euo pipefail

# Batch replay Piper HaMeR gripper axes and overlay FoundationPose object tracks
# in the same calibrated Piper scene.

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh

HAND_INPUT=${1:-/home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2}
OBJECT_INPUT_ROOT=${2:-/home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis}
OUT_ROOT=${3:-/home/zaijia001/ssd/RoboTwin/code_painting/output_piper_replay_hamer_axes_with_objects_all}

GPU=${GPU:-2}
FPS=${FPS:-5}
MAX_FRAMES=${MAX_FRAMES:-300}
FRAME_START=${FRAME_START:-0}
ARMS=${ARMS:-both}
TARGET_DX=${TARGET_DX:-0.0}
TARGET_DY=${TARGET_DY:-0.1}
TARGET_DZ=${TARGET_DZ:-0.1}
TARGET_LOCAL_FORWARD_RETREAT_M=${TARGET_LOCAL_FORWARD_RETREAT_M:-0.0}
DEBUG_MODE=${DEBUG_MODE:-0}
DEBUG_VISUALIZE_CAMERAS=${DEBUG_VISUALIZE_CAMERAS:-0}
DEBUG_CAMERA_AXIS_LENGTH=${DEBUG_CAMERA_AXIS_LENGTH:-0.16}
DEBUG_CAMERA_AXIS_THICKNESS=${DEBUG_CAMERA_AXIS_THICKNESS:-0.006}
SAVE_PNG_FRAMES=${SAVE_PNG_FRAMES:-1}
KEEP_ONLY_ZED_THIRD=${KEEP_ONLY_ZED_THIRD:-0}
ID_FILTER=${ID_FILTER:-}
OBJECT_MISSING_FRAME_POLICY=${OBJECT_MISSING_FRAME_POLICY:-hide}
EXECUTE_WAYPOINT_SCENE_STEPS=${EXECUTE_WAYPOINT_SCENE_STEPS:-5}
EXECUTE_SETTLE_SCENE_STEPS=${EXECUTE_SETTLE_SCENE_STEPS:-20}
URDFIK_JOINT_INTERP_WAYPOINTS=${URDFIK_JOINT_INTERP_WAYPOINTS:-10}

ROBOT_CONFIG=${ROBOT_CONFIG:-/home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json}
HEAD_POS=(${HEAD_POS:-0.11210396690038413 -0.39189397826604927 0.4753892624100325})
HEAD_QUAT=(${HEAD_QUAT:-0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399})
CALIBRATION_BUNDLE=${CALIBRATION_BUNDLE:-}
OBJECT_ARGS=(
  --object pear=/home/zaijia001/ssd/data/R1/hand/obj_mesh/pear/pear.obj
  --object star_fruit=/home/zaijia001/ssd/data/R1/hand/obj_mesh/star/star.obj
)

id_selected() {
  local id="$1"
  if [[ -z "${ID_FILTER}" ]]; then
    return 0
  fi
  IFS=',' read -r -a parts <<< "${ID_FILTER}"
  for p in "${parts[@]}"; do
    if [[ "$p" =~ ^[0-9]+$ ]]; then
      [[ "$id" == "$p" ]] && return 0
    elif [[ "$p" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      local lo="${BASH_REMATCH[1]}"
      local hi="${BASH_REMATCH[2]}"
      if (( id >= lo && id <= hi )); then
        return 0
      fi
    fi
  done
  return 1
}

is_object_video_dir() {
  local dir="$1"
  find "${dir}" -mindepth 2 -maxdepth 2 -type f -name poses.npz -print -quit | grep -q .
}

find_object_dir_for_id() {
  local id="$1"
  if [[ -d "${OBJECT_INPUT_ROOT}" ]] && is_object_video_dir "${OBJECT_INPUT_ROOT}"; then
    local base
    base=$(basename "${OBJECT_INPUT_ROOT}")
    if [[ "${base}" =~ (^|[^0-9])${id}$ ]]; then
      echo "${OBJECT_INPUT_ROOT}"
      return 0
    fi
  fi
  local candidate
  while IFS= read -r -d '' candidate; do
    local base
    base=$(basename "${candidate}")
    if [[ "${base}" =~ (^|[^0-9])${id}$ ]] && is_object_video_dir "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
  done < <(find "${OBJECT_INPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
  return 1
}

run_one() {
  local npz="$1"
  local id="$2"
  local object_dir
  if ! object_dir=$(find_object_dir_for_id "${id}"); then
    echo "[skip] id=${id} missing FoundationPose object dir under ${OBJECT_INPUT_ROOT}" >&2
    return 0
  fi

  local out_dir="${OUT_ROOT}/id_${id}"
  mkdir -p "${out_dir}"
  echo "[hamer-axes-with-objects] id=${id} hand_npz=${npz} object_dir=${object_dir} out=${out_dir}"
  local calibration_args=()
  if [[ -n "${CALIBRATION_BUNDLE}" ]]; then
    calibration_args=(--piper_calibration_bundle "${CALIBRATION_BUNDLE}")
  fi

  CUDA_VISIBLE_DEVICES=${GPU} conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py \
    --input_npz "${npz}" \
    --output_dir "${out_dir}" \
    --fps "${FPS}" \
    --frame_start "${FRAME_START}" \
    --max_frames "${MAX_FRAMES}" \
    --arms "${ARMS}" \
    --robot_config "${ROBOT_CONFIG}" \
    "${calibration_args[@]}" \
    --camera_cv_axis_mode legacy_r1 \
    --head_camera_local_pos "${HEAD_POS[@]}" \
    --head_camera_local_quat_wxyz "${HEAD_QUAT[@]}" \
    --require_stored_gripper_pose 1 \
    --pose_source gripper \
    --orientation_remap_label identity \
    --stored_orientation_post_rot_xyz_deg 0 0 0 \
    --target_local_forward_retreat_m "${TARGET_LOCAL_FORWARD_RETREAT_M}" \
    --target_world_offset_xyz "${TARGET_DX}" "${TARGET_DY}" "${TARGET_DZ}" \
    --execute_waypoint_scene_steps "${EXECUTE_WAYPOINT_SCENE_STEPS}" \
    --execute_settle_scene_steps "${EXECUTE_SETTLE_SCENE_STEPS}" \
    --urdfik_joint_interp_waypoints "${URDFIK_JOINT_INTERP_WAYPOINTS}" \
    --debug_mode "${DEBUG_MODE}" \
    --debug_visualize_cameras "${DEBUG_VISUALIZE_CAMERAS}" \
    --debug_camera_axis_length "${DEBUG_CAMERA_AXIS_LENGTH}" \
    --debug_camera_axis_thickness "${DEBUG_CAMERA_AXIS_THICKNESS}" \
    --debug_post_execute 1 \
    --debug_frame_limit -1 \
    --save_png_frames "${SAVE_PNG_FRAMES}" \
    --object_replay_input_dir "${object_dir}" \
    --object_missing_frame_policy "${OBJECT_MISSING_FRAME_POLICY}" \
    "${OBJECT_ARGS[@]}" \
    --lighting_mode front_no_shadow

  if [[ "${KEEP_ONLY_ZED_THIRD}" == "1" ]]; then
    rm -f "${out_dir}/zed_depth.mp4" "${out_dir}/left_wrist_replay.mp4" "${out_dir}/right_wrist_replay.mp4"
    if [[ -d "${out_dir}/frames" ]]; then
      find "${out_dir}/frames" -maxdepth 1 -type f \( -name 'depth_*.png' -o -name 'left_wrist_*.png' -o -name 'right_wrist_*.png' \) -delete
    fi
  fi
}

mkdir -p "${OUT_ROOT}"

if [[ -d "${HAND_INPUT}" ]]; then
  shopt -s nullglob
  for npz in "${HAND_INPUT}"/hand_detections_*.npz; do
    id=$(basename "${npz}" .npz | sed 's/^hand_detections_//')
    if id_selected "${id}"; then
      run_one "${npz}" "${id}"
    else
      echo "[skip] id=${id} not in ID_FILTER=${ID_FILTER}"
    fi
  done
else
  stem=$(basename "${HAND_INPUT}" .npz)
  id="${stem#hand_detections_}"
  run_one "${HAND_INPUT}" "${id}"
fi

echo "[done] output root: ${OUT_ROOT}"
