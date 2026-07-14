#!/usr/bin/env python3
"""Entry point for isolated Piper Dense Replay URDF-match v2."""

from __future__ import annotations

from pathlib import Path

import render_hand_retarget_r1_npz as base
from render_hand_retarget_piper_dual_npz_urdfmatch_v2 import (
    HandRetargetPiperDualURDFMatchV2Renderer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPER_DEFAULT_CONFIG = PROJECT_ROOT / "robot_config_Piper_dual_v2.json"

base.DEFAULT_ROBOT_CONFIG = PIPER_DEFAULT_CONFIG
base.HandRetargetR1Renderer = HandRetargetPiperDualURDFMatchV2Renderer


if __name__ == "__main__":
    base.main()
