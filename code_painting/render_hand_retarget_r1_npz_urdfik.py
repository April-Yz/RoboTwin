#!/usr/bin/env python3
"""URDF-IK variant of hand retarget replay for R1.

This keeps the same inputs / debug flow as `render_hand_retarget_r1_npz.py`,
but bypasses RoboTwin's planner and solves one IK target per arm directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import render_hand_retarget_r1_npz as base
from replay_r1_h5 import ReplayRenderer
from urdfik import URDFInverseKinematics


R1_URDF = PROJECT_ROOT / "galaxea_sim" / "assets" / "r1" / "robot.urdf"
R1_CONFIG = PROJECT_ROOT / "robot_config_R1.json"
base.DEFAULT_ROBOT_CONFIG = R1_CONFIG
DEFAULT_INIT_LEFT_ARM_JOINTS = np.array(
    [-0.16744680851063828, 2.0108510638297874, -0.6593617021276595, 2.002127659574468, 0.39382978723404255, -1.7193617021276595],
    dtype=np.float64,
)
DEFAULT_INIT_RIGHT_ARM_JOINTS = np.array(
    [0.19234042553191488, 1.8925531914893616, -0.6874468085106383, -1.6057446808510638, -0.10148936170212766, 1.3085106382978724],
    dtype=np.float64,
)
DEFAULT_INIT_GRIPPER_OPEN = 1.0


class HandRetargetR1URDFIKRenderer(ReplayRenderer):
    def __init__(self, *args, **kwargs) -> None:
        self.urdfik_trajectory_mode = str(kwargs.pop("urdfik_trajectory_mode", "joint_interp"))
        self.urdfik_cartesian_interp_steps = max(int(kwargs.pop("urdfik_cartesian_interp_steps", 8)), 2)
        kwargs["attach_planner"] = False
        if kwargs.get("init_left_arm_joints") is None:
            kwargs["init_left_arm_joints"] = DEFAULT_INIT_LEFT_ARM_JOINTS.copy()
        if kwargs.get("init_right_arm_joints") is None:
            kwargs["init_right_arm_joints"] = DEFAULT_INIT_RIGHT_ARM_JOINTS.copy()
        if kwargs.get("init_gripper_open") is None:
            kwargs["init_gripper_open"] = DEFAULT_INIT_GRIPPER_OPEN
        super().__init__(*args, **kwargs)
        self.ik_urdf_path = R1_URDF
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
        print(f"[ik-mode] robot=r1 solver=urdfik urdf={self.ik_urdf_path}")
        print(
            "[ik-trajectory] "
            f"mode={self.urdfik_trajectory_mode} "
            f"cartesian_interp_steps={self.urdfik_cartesian_interp_steps}"
        )
        print(
            "[ik-init] "
            f"left_joints={np.round(self.init_left_arm_joints, 6).tolist()} "
            f"right_joints={np.round(self.init_right_arm_joints, 6).tolist()} "
            f"gripper_open={self.init_gripper_open:.3f}"
        )

    def _current_arm_joints(self, arm: str) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("Robot is not initialized.")
        if arm == "left":
            return np.asarray(self.robot.get_left_arm_real_jointState()[:6], dtype=np.float64)
        if arm == "right":
            return np.asarray(self.robot.get_right_arm_real_jointState()[:6], dtype=np.float64)
        raise ValueError(f"Unsupported arm: {arm}")

    def _target_tcp_world_to_ee_base(self, arm: str, target_pose_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.robot is None:
            raise RuntimeError("Robot is not initialized.")
        target_pose_base = self.world_pose_to_base_pose(target_pose_world)
        target_pose_ee = self.robot._trans_from_gripper_to_endlink(target_pose_base.tolist(), arm_tag=arm)
        return np.asarray(target_pose_ee.p, dtype=np.float64), np.asarray(target_pose_ee.q, dtype=np.float64)

    @staticmethod
    def _interpolate_tcp_pose_world_series(
        start_pose_world: np.ndarray,
        target_pose_world: np.ndarray,
        num_waypoints: int,
    ) -> List[np.ndarray]:
        start_pose_world = np.asarray(start_pose_world, dtype=np.float64).reshape(7)
        target_pose_world = np.asarray(target_pose_world, dtype=np.float64).reshape(7)
        num_waypoints = max(int(num_waypoints), 2)
        start_pos = start_pose_world[:3]
        target_pos = target_pose_world[:3]
        rotations = R.from_quat(
            np.stack(
                [
                    base.quat_wxyz_to_xyzw(base.normalize_quat_wxyz(start_pose_world[3:])),
                    base.quat_wxyz_to_xyzw(base.normalize_quat_wxyz(target_pose_world[3:])),
                ],
                axis=0,
            )
        )
        slerp = Slerp([0.0, 1.0], rotations)
        waypoints: List[np.ndarray] = []
        for ratio in np.linspace(0.0, 1.0, num=num_waypoints, endpoint=True):
            interp_pos = (1.0 - ratio) * start_pos + ratio * target_pos
            interp_quat = base.quat_xyzw_to_wxyz(slerp(float(ratio)).as_quat())
            waypoints.append(np.concatenate([interp_pos, interp_quat]).astype(np.float64))
        return waypoints

    def _build_fail_plan(
        self,
        arm: str,
        current_arm: np.ndarray,
        target_pose_world: np.ndarray,
        ee_pos_base: np.ndarray,
        ee_quat_base: np.ndarray,
        reason: Optional[str] = None,
        waypoint_index: Optional[int] = None,
        waypoint_count: Optional[int] = None,
    ) -> Dict:
        payload = {
            "status": "Fail",
            "solver": "urdfik",
            "trajectory_mode": self.urdfik_trajectory_mode,
            "arm": arm,
            "current_joints": current_arm.copy(),
            "target_pose_world": np.asarray(target_pose_world, dtype=np.float64).copy(),
            "target_pose_ee_base": np.concatenate([ee_pos_base, ee_quat_base]).astype(np.float64),
        }
        if reason is not None:
            payload["reason"] = str(reason)
        if waypoint_index is not None:
            payload["failed_waypoint_index"] = int(waypoint_index)
        if waypoint_count is not None:
            payload["waypoint_count"] = int(waypoint_count)
        return payload

    def _plan_path_joint_interp(self, arm: str, target_pose_world: np.ndarray) -> Optional[Dict]:
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
            return self._build_fail_plan(arm, current_arm, target_pose_world, ee_pos_base, ee_quat_base)

        solution = result.solution.detach().cpu().numpy().reshape(-1)
        if solution.shape[0] < 10:
            return self._build_fail_plan(
                arm,
                current_arm,
                target_pose_world,
                ee_pos_base,
                ee_quat_base,
                reason=f"expected >=10 joints from IK, got {solution.shape[0]}",
            )

        target_arm = solution[-6:].astype(np.float64)
        position_traj = np.stack([current_arm, target_arm], axis=0)
        velocity_traj = np.zeros_like(position_traj, dtype=np.float64)
        return {
            "status": "Success",
            "solver": "urdfik",
            "trajectory_mode": self.urdfik_trajectory_mode,
            "arm": arm,
            "current_joints": current_arm.copy(),
            "target_joints": target_arm.copy(),
            "position": position_traj,
            "velocity": velocity_traj,
            "target_pose_world": np.asarray(target_pose_world, dtype=np.float64).copy(),
            "target_pose_ee_base": np.concatenate([ee_pos_base, ee_quat_base]).astype(np.float64),
        }

    def _plan_path_cartesian_interp_ik(self, arm: str, target_pose_world: np.ndarray) -> Optional[Dict]:
        current_arm = self._current_arm_joints(arm)
        current_full = np.concatenate([self.torso_qpos, current_arm], dtype=np.float64)
        current_tcp_world = np.asarray(self.get_current_tcp_pose(arm), dtype=np.float64).reshape(7)
        tcp_waypoints_world = self._interpolate_tcp_pose_world_series(
            current_tcp_world,
            target_pose_world,
            self.urdfik_cartesian_interp_steps,
        )
        solver = self.left_ik_solver if arm == "left" else self.right_ik_solver
        joint_waypoints: List[np.ndarray] = [current_arm.copy()]
        ee_waypoints_base: List[np.ndarray] = []
        current_seed = current_full.copy()
        for waypoint_index, tcp_waypoint_world in enumerate(tcp_waypoints_world[1:], start=1):
            ee_pos_base, ee_quat_base = self._target_tcp_world_to_ee_base(arm, tcp_waypoint_world)
            ee_waypoints_base.append(np.concatenate([ee_pos_base, ee_quat_base]).astype(np.float64))
            result = solver.solve_ik(
                target_position=ee_pos_base,
                target_orientation_wxyz=ee_quat_base,
                current_joints=current_seed,
            )
            if result is None:
                return self._build_fail_plan(
                    arm,
                    current_arm,
                    target_pose_world,
                    ee_pos_base,
                    ee_quat_base,
                    reason="failed during cartesian waypoint IK",
                    waypoint_index=waypoint_index,
                    waypoint_count=len(tcp_waypoints_world),
                )
            solution = result.solution.detach().cpu().numpy().reshape(-1)
            if solution.shape[0] < 10:
                return self._build_fail_plan(
                    arm,
                    current_arm,
                    target_pose_world,
                    ee_pos_base,
                    ee_quat_base,
                    reason=f"expected >=10 joints from IK, got {solution.shape[0]}",
                    waypoint_index=waypoint_index,
                    waypoint_count=len(tcp_waypoints_world),
                )
            target_arm = solution[-6:].astype(np.float64)
            joint_waypoints.append(target_arm.copy())
            current_seed = np.concatenate([self.torso_qpos, target_arm], dtype=np.float64)

        position_traj = np.stack(joint_waypoints, axis=0)
        velocity_traj = np.zeros_like(position_traj, dtype=np.float64)
        return {
            "status": "Success",
            "solver": "urdfik",
            "trajectory_mode": self.urdfik_trajectory_mode,
            "arm": arm,
            "current_joints": current_arm.copy(),
            "target_joints": joint_waypoints[-1].copy(),
            "position": position_traj,
            "velocity": velocity_traj,
            "target_pose_world": np.asarray(target_pose_world, dtype=np.float64).copy(),
            "target_pose_ee_base": ee_waypoints_base[-1].copy() if ee_waypoints_base else np.concatenate(self._target_tcp_world_to_ee_base(arm, target_pose_world)).astype(np.float64),
            "cartesian_waypoint_count": len(tcp_waypoints_world),
            "tcp_waypoints_world": np.stack(tcp_waypoints_world, axis=0),
            "ee_waypoints_base": np.stack(ee_waypoints_base, axis=0) if ee_waypoints_base else np.zeros((0, 7), dtype=np.float64),
        }

    def plan_path(self, arm: str, target_pose_world: np.ndarray) -> Optional[Dict]:
        if self.robot is None:
            return None
        if self.urdfik_trajectory_mode == "joint_interp":
            return self._plan_path_joint_interp(arm, target_pose_world)
        if self.urdfik_trajectory_mode == "cartesian_interp_ik":
            return self._plan_path_cartesian_interp_ik(arm, target_pose_world)
        raise ValueError(f"Unsupported urdfik_trajectory_mode: {self.urdfik_trajectory_mode}")

    def _execute_single_ik_plan(self, arm: str, plan: Dict, num_steps: int = 20) -> str:
        status = self._plan_status(plan)
        if status != "Success":
            return status
        if self.robot is None:
            return "Missing"
        current = np.asarray(plan["current_joints"], dtype=np.float64).reshape(6)
        target = np.asarray(plan["target_joints"], dtype=np.float64).reshape(6)
        traj = np.linspace(current, target, num=max(int(num_steps), 2), endpoint=True)
        vel = np.zeros(6, dtype=np.float64)
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
        left_curr = np.asarray(left_plan["current_joints"], dtype=np.float64).reshape(6) if left_ok else None
        right_curr = np.asarray(right_plan["current_joints"], dtype=np.float64).reshape(6) if right_ok else None
        left_tgt = np.asarray(left_plan["target_joints"], dtype=np.float64).reshape(6) if left_ok else None
        right_tgt = np.asarray(right_plan["target_joints"], dtype=np.float64).reshape(6) if right_ok else None
        num_steps = 20

        for step in range(num_steps):
            alpha = float(step + 1) / float(num_steps)
            if left_ok:
                q_left = left_curr * (1.0 - alpha) + left_tgt * alpha
                self.robot.set_arm_joints(q_left, np.zeros(6, dtype=np.float64), "left")
            if right_ok:
                q_right = right_curr * (1.0 - alpha) + right_tgt * alpha
                self.robot.set_arm_joints(q_right, np.zeros(6, dtype=np.float64), "right")
            self.step_scene(steps=1)

        self.step_scene(steps=4)
        return left_status, right_status


base.HandRetargetR1Renderer = HandRetargetR1URDFIKRenderer


if __name__ == "__main__":
    base.main()
