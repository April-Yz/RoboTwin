#!/usr/bin/env python3
"""URDFIK renderer that uses the Real Piper TCP in both directions."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from render_hand_retarget_piper_dual_npz_urdfik import (
    HandRetargetPiperDualURDFIKRenderer,
)

from frame_contract import (
    SERVER_TOOL_LENGTH_M,
    SERVER_TOOL_PITCH_RAD,
    pose_wxyz_to_matrix,
    real_tcp_pose_to_urdf_link6_pose,
    sim_link6_pose_to_real_tcp_pose,
    sim_link6_pose_to_urdf_link6_pose,
    urdf_link6_pose_to_real_tcp_pose,
)


class PiperCanonicalTCPV1Renderer(HandRetargetPiperDualURDFIKRenderer):
    """Treat every planner target/current pose as ``T_W_RTCP``.

    This class does not call the legacy Ours TCP/EE conversion.  IK targets
    are converted with the exact Piper server tool transform.  Current TCP
    readback first maps raw SAPIEN ``L6_SIM`` axes to CuRobo/server
    ``L6_URDF`` axes and only then applies the server tool transform.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._print_link6_fk_contract_check()
        print(
            "[piper-canonical-tcp-v1] "
            "target=T_W_RTCP ik_target=T_B_L6URDF current=T_W_RTCP "
            "T_L6SIM_L6URDF=exact_Ry(+pi/2) "
            f"T_L6URDF_RTCP=Ry({SERVER_TOOL_PITCH_RAD})@Tx({SERVER_TOOL_LENGTH_M}) "
            "axes=local_RTCP(X:red:approach,Y:green:opening,Z:blue:side)"
        )

    def _print_link6_fk_contract_check(self) -> None:
        """Compare SAPIEN and URDF link6 FK for the exact same joint state."""
        for arm in ("left", "right"):
            current_joints = self._current_arm_joints(arm)
            raw_sim_link6_world = self._raw_sim_link6_world_pose(arm)
            sim_link6_base = self.world_pose_to_base_pose_for_arm(
                raw_sim_link6_world, arm
            )
            solver = getattr(self, f"{arm}_ik_solver")
            fk_pos, fk_quat, _ = solver.forward_kinematics(current_joints)
            urdf_link6_base = np.concatenate(
                [
                    np.asarray(fk_pos, dtype=np.float64).reshape(3),
                    np.asarray(fk_quat, dtype=np.float64).reshape(4),
                ]
            )
            sim_matrix = pose_wxyz_to_matrix(sim_link6_base)
            urdf_matrix = pose_wxyz_to_matrix(urdf_link6_base)
            raw_pos_err_m = float(
                np.linalg.norm(sim_matrix[:3, 3] - urdf_matrix[:3, 3])
            )
            relative_rotation = sim_matrix[:3, :3].T @ urdf_matrix[:3, :3]
            angle_cos = float(
                np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
            )
            raw_rot_err_deg = float(np.degrees(np.arccos(angle_cos)))
            adapted_sim_matrix = pose_wxyz_to_matrix(
                sim_link6_pose_to_urdf_link6_pose(sim_link6_base)
            )
            adapted_pos_err_m = float(
                np.linalg.norm(
                    adapted_sim_matrix[:3, 3] - urdf_matrix[:3, 3]
                )
            )
            adapted_relative_rotation = (
                adapted_sim_matrix[:3, :3].T @ urdf_matrix[:3, :3]
            )
            adapted_angle_cos = float(
                np.clip(
                    (np.trace(adapted_relative_rotation) - 1.0) * 0.5,
                    -1.0,
                    1.0,
                )
            )
            adapted_rot_err_deg = float(
                np.degrees(np.arccos(adapted_angle_cos))
            )
            print(
                f"[fk-contract-check] arm={arm} same_q=1 "
                f"raw_sim_vs_urdf_pos_err_m={raw_pos_err_m:.9f} "
                f"raw_sim_vs_urdf_rot_err_deg={raw_rot_err_deg:.6f} "
                "R_L6SIM_L6URDF="
                f"{np.round(relative_rotation, 9).tolist()} "
                f"adapted_pos_err_m={adapted_pos_err_m:.9f} "
                f"adapted_rot_err_deg={adapted_rot_err_deg:.6f}"
            )

    def _raw_sim_link6_world_pose(self, arm: str) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("Robot is not initialized.")
        if arm == "left":
            link = self.robot.left_ee
        elif arm == "right":
            link = self.robot.right_ee
        else:
            raise ValueError(f"Unsupported arm: {arm}")
        pose = link.global_pose
        return np.concatenate(
            [
                np.asarray(pose.p, dtype=np.float64).reshape(3),
                np.asarray(pose.q, dtype=np.float64).reshape(4),
            ]
        )

    def get_current_tcp_pose(self, arm: str) -> np.ndarray:
        """Return current ``T_W_RTCP`` from raw SAPIEN ``T_W_L6SIM``."""
        return sim_link6_pose_to_real_tcp_pose(
            self._raw_sim_link6_world_pose(arm)
        )

    def _target_tcp_world_to_ee_base(
        self, arm: str, target_pose_world: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert ``T_W_RTCP`` to the URDFIK target ``T_B_L6URDF``."""
        target_real_tcp_base = np.asarray(
            self.world_pose_to_base_pose_for_arm(target_pose_world, arm),
            dtype=np.float64,
        ).reshape(7)
        target_urdf_link6_base = real_tcp_pose_to_urdf_link6_pose(
            target_real_tcp_base
        )
        return (
            target_urdf_link6_base[:3].copy(),
            target_urdf_link6_base[3:].copy(),
        )

    def planned_real_tcp_pose_from_target_joints(
        self, arm: str, target_arm_joints: np.ndarray
    ) -> np.ndarray:
        """Evaluate a planned joint target in the same ``T_W_RTCP`` contract."""
        solver = getattr(self, f"{arm}_ik_solver", None)
        if solver is None:
            raise RuntimeError(f"{arm} IK solver is unavailable")
        link6_pos_base, link6_quat_base, _ = solver.forward_kinematics(
            np.asarray(target_arm_joints, dtype=np.float64).reshape(6)
        )
        urdf_link6_base = np.concatenate(
            [
                np.asarray(link6_pos_base, dtype=np.float64).reshape(3),
                np.asarray(link6_quat_base, dtype=np.float64).reshape(4),
            ]
        )
        real_tcp_base = urdf_link6_pose_to_real_tcp_pose(urdf_link6_base)
        return np.asarray(
            self.base_pose_to_world_pose_for_arm(real_tcp_base, arm),
            dtype=np.float64,
        ).reshape(7)


__all__ = ["PiperCanonicalTCPV1Renderer"]
