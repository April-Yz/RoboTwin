#!/usr/bin/env python3
"""Isolated planner entry for PiperCanonicalTCP-v1."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
CODE_PAINTING = THIS_DIR.parent
PROJECT_ROOT = CODE_PAINTING.parent
os.chdir(PROJECT_ROOT)
for path in (THIS_DIR, CODE_PAINTING, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_anygrasp_keyframes_r1 as base_plan
import render_hand_retarget_r1_npz as base_render
from replay_piper_dual_h5 import PiperDualReplayRenderer

from renderer import PiperCanonicalTCPV1Renderer


PIPER_CONFIG = PROJECT_ROOT / "robot_config_PiperPika_agx_dual_table_0515.json"
IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_ORIGINAL_PLANNED_EVAL = base_plan.planned_eval_pose_from_plan


def _has_flag(argv: Sequence[str], flag: str) -> bool:
    return any(str(item) == flag for item in argv)


def _prepend_default_flags(argv: Sequence[str]) -> List[str]:
    out = list(argv)
    prefix: List[str] = []
    if not _has_flag(out, "--robot_config"):
        prefix += ["--robot_config", str(PIPER_CONFIG)]
    if not _has_flag(out, "--head_camera_local_quat_wxyz"):
        prefix += ["--head_camera_local_quat_wxyz", "1", "0", "0", "0"]
    return prefix + out


def _planned_eval_pose_from_plan(
    renderer,
    arm: str,
    plan: Optional[dict],
    pose_source: str,
):
    if isinstance(renderer, PiperCanonicalTCPV1Renderer) and pose_source == "tcp":
        if not isinstance(plan, dict) or "target_joints" not in plan:
            return None
        return renderer.planned_real_tcp_pose_from_target_joints(
            arm, np.asarray(plan["target_joints"], dtype=np.float64)
        )
    return _ORIGINAL_PLANNED_EVAL(renderer, arm, plan, pose_source)


def main() -> None:
    base_plan.ReplayRenderer = PiperDualReplayRenderer
    base_plan.urdfik_base.HandRetargetR1URDFIKRenderer = PiperCanonicalTCPV1Renderer
    base_plan.planned_eval_pose_from_plan = _planned_eval_pose_from_plan
    base_plan.R1_CONFIG = PIPER_CONFIG
    base_plan.R1_WRIST_CAMERA_LOCAL_QUAT_WXYZ = IDENTITY_WXYZ.copy()
    base_plan.base.DEFAULT_ROBOT_CONFIG = PIPER_CONFIG
    base_render.DEFAULT_ROBOT_CONFIG = PIPER_CONFIG
    sys.argv = [sys.argv[0], *_prepend_default_flags(sys.argv[1:])]
    print(
        "[piper-canonical-tcp-v1-entry] isolated=1 "
        "planner_target=T_W_RTCP ik_target=T_B_L6URDF "
        "current_readback=T_W_L6SIM_to_T_W_RTCP"
    )
    base_plan.main()


if __name__ == "__main__":
    main()
