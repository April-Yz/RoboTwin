#!/usr/bin/env bash
set -euo pipefail

# Planner runner for D435 preview summaries generated in canonical robot/replay
# frame mode. In those summaries the target blue local +Z axis is the gripper
# approach/fingertip direction, so compensation and pregrasp retreat use local Z.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh" \
  --preview_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435_robot_frame \
  --candidate_orientation_remap_label identity \
  --candidate_target_local_x_offset_m 0.0 \
  --candidate_target_local_z_offset_m -0.05 \
  --approach_axis local_z \
  --approach_offset_m 0.12 \
  --debug_gripper_actor_forward_axis local_z \
  "$@"
