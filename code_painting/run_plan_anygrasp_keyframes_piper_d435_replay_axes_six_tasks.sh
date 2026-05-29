#!/usr/bin/env bash
set -euo pipefail

# Direct-replay-compatible AnyGrasp runner.
#
# Direct Piper hand replay's stored hand frame uses local +Z (blue) as the
# hand approach axis. The AnyGrasp candidate selection stage already converts
# that hand frame into the AnyGrasp/Piper execution convention:
#   AnyGrasp local +X = direct-replay hand local +Z
# Therefore execution must keep the AnyGrasp candidate frame identity and apply
# compensation/pregrasp along local +X, which matches the C-shaped gripper
# visual's fingertip direction. Remapping AnyGrasp +X to target +Z makes the
# robot follow the gripper side-normal instead of the C-plane approach.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh" \
  --candidate_orientation_remap_label identity \
  --candidate_target_local_x_offset_m -0.05 \
  --candidate_target_local_z_offset_m 0.0 \
  --approach_axis local_x \
  --approach_offset_m 0.12 \
  "$@"
