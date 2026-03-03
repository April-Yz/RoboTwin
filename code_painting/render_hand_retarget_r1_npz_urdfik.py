#!/usr/bin/env python3
"""URDF-IK variant of hand retarget replay for R1.

This keeps the same inputs / debug flow as `render_hand_retarget_r1_npz.py`,
but bypasses RoboTwin's planner and solves one IK target per arm directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import render_hand_retarget_r1_npz as base
from urdfik import DEFAULT_URDF, URDFInverseKinematics


class HandRetargetR1URDFIKRenderer(base.HandRetargetR1Renderer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ik_urdf_path = DEFAULT_URDF
        self.left_ik_solver = URDFInverseKinematics(
            urdf_file=self.ik_urdf_path,
            base_link="base_link",
            ee_link="left_gripper_link",
        )
        self.right_ik_solver = URDFInverseKinematics(
            urdf_file=self.ik_urdf_path,
            base_link="base_link",
            ee_link="right_gripper_link",
        )
        print(f"[ik-mode] solver=urdfik urdf={self.ik_urdf_path}")

    def _current_arm_joints(self, arm: str) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("Robot is not initialized.")
        if arm == "left":
            return np.asarray(self.robot.get_left_arm_real_jointState()[:7], dtype=np.float64)
        if arm == "right":
            return np.asarray(self.robot.get_right_arm_real_jointState()[:7], dtype=np.float64)
        raise ValueError(f"Unsupported arm: {arm}")

    def _target_tcp_world_to_ee_base(self, arm: str, target_pose_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.robot is None:
            raise RuntimeError("Robot is not initialized.")
        target_pose_base = self.world_pose_to_base_pose(target_pose_world)
        target_pose_ee = self.robot._trans_from_gripper_to_endlink(target_pose_base.tolist(), arm_tag=arm)
        return np.asarray(target_pose_ee.p, dtype=np.float64), np.asarray(target_pose_ee.q, dtype=np.float64)

    def plan_path(self, arm: str, target_pose_world: np.ndarray) -> Optional[Dict]:
        if self.robot is None:
            return None
        current_arm = self._current_arm_joints(arm)
        current_full = np.concatenate([self.torso_qpos, current_arm], dtype=np.float64)
        ee_pos_base, ee_quat_base = self._target_tcp_world_to_ee_base(arm, target_pose_world)
        solver = self.left_ik_solver if arm == "left" else self.right_ik_solver
        result = solver.solve_ik(
            target_position=ee_pos_base,
            target_orientation_wxyz=ee_quat_base,
            current_joints=current_full,
        )
        if result is None:
            return {
                "status": "Fail",
                "solver": "urdfik",
                "arm": arm,
                "current_joints": current_arm.copy(),
                "target_pose_world": np.asarray(target_pose_world, dtype=np.float64).copy(),
                "target_pose_ee_base": np.concatenate([ee_pos_base, ee_quat_base]).astype(np.float64),
            }

        solution = result.solution.detach().cpu().numpy().reshape(-1)
        if solution.shape[0] < 11:
            return {
                "status": "Fail",
                "solver": "urdfik",
                "arm": arm,
                "reason": f"expected >=11 joints from IK, got {solution.shape[0]}",
                "current_joints": current_arm.copy(),
                "target_pose_world": np.asarray(target_pose_world, dtype=np.float64).copy(),
                "target_pose_ee_base": np.concatenate([ee_pos_base, ee_quat_base]).astype(np.float64),
            }

        target_arm = solution[-7:].astype(np.float64)
        position_traj = np.stack([current_arm, target_arm], axis=0)
        velocity_traj = np.zeros_like(position_traj, dtype=np.float64)
        return {
            "status": "Success",
            "solver": "urdfik",
            "arm": arm,
            "current_joints": current_arm.copy(),
            "target_joints": target_arm.copy(),
            "position": position_traj,
            "velocity": velocity_traj,
            "target_pose_world": np.asarray(target_pose_world, dtype=np.float64).copy(),
            "target_pose_ee_base": np.concatenate([ee_pos_base, ee_quat_base]).astype(np.float64),
        }

    def _execute_single_ik_plan(self, arm: str, plan: Dict, num_steps: int = 20) -> str:
        status = self._plan_status(plan)
        if status != "Success":
            return status
        if self.robot is None:
            return "Missing"
        current = np.asarray(plan["current_joints"], dtype=np.float64).reshape(7)
        target = np.asarray(plan["target_joints"], dtype=np.float64).reshape(7)
        traj = np.linspace(current, target, num=max(int(num_steps), 2), endpoint=True)
        vel = np.zeros(7, dtype=np.float64)
        for q in traj:
            self.robot.set_arm_joints(q, vel, arm)
            self.step_scene(steps=1)
        self.step_scene(steps=4)
        return "Success"

    def execute_plans(self, left_plan: Optional[Dict], right_plan: Optional[Dict]) -> Tuple[str, str]:
        left_status = self._plan_status(left_plan)
        right_status = self._plan_status(right_plan)

        left_ok = left_status == "Success"
        right_ok = right_status == "Success"
        left_curr = np.asarray(left_plan["current_joints"], dtype=np.float64).reshape(7) if left_ok else None
        right_curr = np.asarray(right_plan["current_joints"], dtype=np.float64).reshape(7) if right_ok else None
        left_tgt = np.asarray(left_plan["target_joints"], dtype=np.float64).reshape(7) if left_ok else None
        right_tgt = np.asarray(right_plan["target_joints"], dtype=np.float64).reshape(7) if right_ok else None
        num_steps = 20

        for step in range(num_steps):
            alpha = float(step + 1) / float(num_steps)
            if left_ok:
                q_left = left_curr * (1.0 - alpha) + left_tgt * alpha
                self.robot.set_arm_joints(q_left, np.zeros(7, dtype=np.float64), "left")
            if right_ok:
                q_right = right_curr * (1.0 - alpha) + right_tgt * alpha
                self.robot.set_arm_joints(q_right, np.zeros(7, dtype=np.float64), "right")
            self.step_scene(steps=1)

        self.step_scene(steps=4)
        return left_status, right_status


base.HandRetargetR1Renderer = HandRetargetR1URDFIKRenderer


if __name__ == "__main__":
    base.main()
