#!/usr/bin/env python3
"""Isolated Piper URDFIK V3 renderer with reversible TCP/EE conversion."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from piper_ik_v3_transforms import (
    ours_pose_to_link6_pose,
    real_piper_tcp_pose_to_link6_pose,
)
from render_hand_retarget_piper_dual_npz_urdfik import (
    HandRetargetPiperDualURDFIKRenderer,
)


VALID_TARGET_SEMANTICS = {"ours_tcp", "ours_ee", "real_piper_tcp"}


class HandRetargetPiperDualURDFIKV3Renderer(
    HandRetargetPiperDualURDFIKRenderer
):
    """Piper IK V3 without changing the existing V2 renderer.

    ``ours_tcp`` is the default because current OursV2 fields named ``*_ee_*``
    were built from ``current_*_tcp_pose_world_wxyz``.  V3 removes both the
    0.12 m TCP translation and the simulator orientation remaps before link6
    URDFIK.  ``ours_ee`` is available for future data built from the actual
    ``current_*_ee_pose_world_wxyz`` fields.  ``real_piper_tcp`` consumes the
    frame published by the real Piper FK node and removes
    ``Ry(-pi/2) @ Tx(tool_length)`` exactly as the real IK implementation does.
    """

    def __init__(self, *args, **kwargs) -> None:
        semantics = kwargs.pop(
            "piper_ik_v3_target_semantics",
            os.environ.get("PIPER_IK_V3_TARGET_SEMANTICS", "ours_tcp"),
        )
        self.piper_ik_v3_target_semantics = str(semantics).strip().lower()
        if self.piper_ik_v3_target_semantics not in VALID_TARGET_SEMANTICS:
            raise ValueError(
                "PIPER_IK_V3_TARGET_SEMANTICS must be one of "
                f"{sorted(VALID_TARGET_SEMANTICS)}, got {semantics!r}"
            )
        self.piper_ik_v3_real_tool_length_m = float(
            kwargs.pop(
                "piper_ik_v3_real_tool_length_m",
                os.environ.get("PIPER_IK_V3_REAL_TOOL_LENGTH_M", "0.19"),
            )
        )
        self.piper_ik_v3_real_tool_pitch_rad = float(
            kwargs.pop(
                "piper_ik_v3_real_tool_pitch_rad",
                os.environ.get(
                    "PIPER_IK_V3_REAL_TOOL_PITCH_RAD",
                    str(-np.pi / 2.0),
                ),
            )
        )
        super().__init__(*args, **kwargs)
        print(
            "[piper-ik-v3] "
            f"target_semantics={self.piper_ik_v3_target_semantics} "
            f"real_tool_length_m={self.piper_ik_v3_real_tool_length_m:.5f} "
            f"real_tool_pitch_rad={self.piper_ik_v3_real_tool_pitch_rad:.8f}"
        )

    def _arm_pose_parameters(
        self, arm: str
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if self.robot is None:
            raise RuntimeError("Robot is not initialized.")
        if arm == "left":
            return (
                np.asarray(self.robot.left_global_trans_matrix, dtype=np.float64),
                np.asarray(self.robot.left_delta_matrix, dtype=np.float64),
                float(self.robot.left_gripper_bias),
            )
        if arm == "right":
            return (
                np.asarray(self.robot.right_global_trans_matrix, dtype=np.float64),
                np.asarray(self.robot.right_delta_matrix, dtype=np.float64),
                float(self.robot.right_gripper_bias),
            )
        raise ValueError(f"Unsupported arm: {arm}")

    def _target_tcp_world_to_ee_base(
        self, arm: str, target_pose_world: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        target_pose_base = np.asarray(
            self.world_pose_to_base_pose_for_arm(target_pose_world, arm),
            dtype=np.float64,
        ).reshape(7)
        semantics = self.piper_ik_v3_target_semantics
        if semantics == "real_piper_tcp":
            link6_pose_base = real_piper_tcp_pose_to_link6_pose(
                target_pose_base,
                tool_length_m=self.piper_ik_v3_real_tool_length_m,
                tool_pitch_rad=self.piper_ik_v3_real_tool_pitch_rad,
            )
        else:
            global_trans, delta, gripper_bias = self._arm_pose_parameters(arm)
            link6_pose_base = ours_pose_to_link6_pose(
                target_pose_base,
                global_trans_matrix=global_trans,
                delta_matrix=delta,
                gripper_bias_m=gripper_bias,
                includes_tcp_offset=(semantics == "ours_tcp"),
            )
        return link6_pose_base[:3].copy(), link6_pose_base[3:].copy()


__all__ = ["HandRetargetPiperDualURDFIKV3Renderer"]
