#!/usr/bin/env bash
set -euo pipefail

# Replay-axis AnyGrasp runner.
#
# Direct Piper hand replay treats gripper local +Z (blue) as the approach axis.
# The raw AnyGrasp frame used by the legacy planner treats local +X as the
# finger-depth axis. This wrapper keeps the old runner unchanged by opting into
# a remapped target frame:
#   original AnyGrasp +X -> planner target +Z
# Then both target compensation and pregrasp retreat are applied along local +Z.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh" \
  --candidate_orientation_remap_label swap_red_blue \
  --candidate_target_local_x_offset_m 0.0 \
  --candidate_target_local_z_offset_m -0.05 \
  --approach_axis local_z \
  --approach_offset_m 0.12 \
  "$@"
