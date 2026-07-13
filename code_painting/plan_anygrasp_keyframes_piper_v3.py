#!/usr/bin/env python3
"""Piper IK V3 planner entry, isolated from the existing OursV2 entry."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import plan_anygrasp_keyframes_r1 as base_plan
import render_hand_retarget_r1_npz as base_render
import render_hand_retarget_piper_dual_npz_urdfik_v3 as piper_urdfik_v3
from replay_piper_dual_h5 import PiperDualReplayRenderer


PIPER_V3_CONFIG = PROJECT_ROOT / "robot_config_PiperPika_agx_dual_table_0515.json"
IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _has_flag(argv: Sequence[str], flag: str) -> bool:
    return any(str(item) == flag for item in argv)


def _prepend_default_flags(argv: Sequence[str]) -> List[str]:
    out = list(argv)
    prefix: List[str] = []
    if not _has_flag(out, "--robot_config"):
        prefix += ["--robot_config", str(PIPER_V3_CONFIG)]
    if not _has_flag(out, "--head_camera_local_quat_wxyz"):
        prefix += ["--head_camera_local_quat_wxyz", "1", "0", "0", "0"]
    return prefix + out


def main() -> None:
    base_plan.ReplayRenderer = PiperDualReplayRenderer
    base_plan.urdfik_base.HandRetargetR1URDFIKRenderer = (
        piper_urdfik_v3.HandRetargetPiperDualURDFIKV3Renderer
    )
    base_plan.R1_CONFIG = PIPER_V3_CONFIG
    base_plan.R1_WRIST_CAMERA_LOCAL_QUAT_WXYZ = IDENTITY_WXYZ.copy()
    base_plan.base.DEFAULT_ROBOT_CONFIG = PIPER_V3_CONFIG
    base_render.DEFAULT_ROBOT_CONFIG = PIPER_V3_CONFIG
    sys.argv = [sys.argv[0], *_prepend_default_flags(sys.argv[1:])]
    print(
        "[piper-ik-v3-entry] isolated=1 "
        f"target_semantics={os.environ.get('PIPER_IK_V3_TARGET_SEMANTICS', 'ours_tcp')}"
    )
    base_plan.main()


if __name__ == "__main__":
    main()
