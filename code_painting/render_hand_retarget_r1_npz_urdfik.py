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
        raw_cartesian_interp_steps = int(kwargs.pop("urdfik_cartesian_interp_steps", 8))
        self.urdfik_cartesian_interp_auto_step_m = max(float(kwargs.pop("urdfik_cartesian_interp_auto_step_m", 0.05)), 1e-4)
        self.urdfik_cartesian_interp_steps = -1 if raw_cartesian_interp_steps == -1 else max(raw_cartesian_interp_steps, 2)
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
            f"cartesian_interp_steps={self.urdfik_cartesian_interp_steps} "
            f"auto_step_m={self.urdfik_cartesian_interp_auto_step_m:.4f} "
            f"ik_pos_thresh={self.left_ik_solver.default_position_threshold:.4f}m "
            f"ik_rot_thresh={self.left_ik_solver.default_rotation_threshold:.4f}rad"
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
    def _effective_cartesian_interp_steps(
        start_pose_world: np.ndarray,
        target_pose_world: np.ndarray,
        requested_waypoints: int,
        min_translation_step_m: float,
        min_rotation_step_rad: float,
    ) -> int:
        start_pose_world = np.asarray(start_pose_world, dtype=np.float64).reshape(7)
        target_pose_world = np.asarray(target_pose_world, dtype=np.float64).reshape(7)
        requested_waypoints = max(int(requested_waypoints), 2)

        translation_dist = float(np.linalg.norm(target_pose_world[:3] - start_pose_world[:3]))
        start_rot = R.from_quat(base.quat_wxyz_to_xyzw(base.normalize_quat_wxyz(start_pose_world[3:])))
        target_rot = R.from_quat(base.quat_wxyz_to_xyzw(base.normalize_quat_wxyz(target_pose_world[3:])))
        rotation_dist = float((target_rot * start_rot.inv()).magnitude())

        max_steps_from_translation = requested_waypoints
        if min_translation_step_m > 1e-9 and translation_dist > 1e-9:
            max_steps_from_translation = max(int(np.ceil(translation_dist / min_translation_step_m)) + 1, 2)

        max_steps_from_rotation = requested_waypoints
        if min_rotation_step_rad > 1e-9 and rotation_dist > 1e-9:
            max_steps_from_rotation = max(int(np.ceil(rotation_dist / min_rotation_step_rad)) + 1, 2)

        effective_waypoints = max(2, min(requested_waypoints, max_steps_from_translation, max_steps_from_rotation))
        return effective_waypoints

    @staticmethod
    def _auto_cartesian_interp_steps(
        start_pose_world: np.ndarray,
        target_pose_world: np.ndarray,
        trigger_translation_m: float,
    ) -> int:
        start_pose_world = np.asarray(start_pose_world, dtype=np.float64).reshape(7)
        target_pose_world = np.asarray(target_pose_world, dtype=np.float64).reshape(7)
        translation_dist = float(np.linalg.norm(target_pose_world[:3] - start_pose_world[:3]))
        if translation_dist <= trigger_translation_m:
            return 2
        intermediate_waypoints = max(int(np.ceil(translation_dist / trigger_translation_m)) - 1, 1)
        return intermediate_waypoints + 2

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

    def _solution_error_to_ee_target(
        self,
        arm: str,
        target_arm: np.ndarray,
        ee_pos_base: np.ndarray,
        ee_quat_base: np.ndarray,
    ) -> Tuple[float, float]:
        solver = self.left_ik_solver if arm == "left" else self.right_ik_solver
        full_joints = np.concatenate([np.asarray(self.torso_qpos, dtype=np.float64).reshape(4), target_arm.reshape(6)], dtype=np.float64)
        fk_pos_base, fk_quat_wxyz_base, _ = solver.forward_kinematics(full_joints)
        pos_err = float(np.linalg.norm(np.asarray(fk_pos_base, dtype=np.float64).reshape(3) - np.asarray(ee_pos_base, dtype=np.float64).reshape(3)))
        rot_err = float(base.quat_angle_deg_wxyz(np.asarray(fk_quat_wxyz_base, dtype=np.float64).reshape(4), np.asarray(ee_quat_base, dtype=np.float64).reshape(4)))
        return pos_err, rot_err

    def _solve_ik_best_candidate(
        self,
        arm: str,
        ee_pos_base: np.ndarray,
        ee_quat_base: np.ndarray,
        current_seed: Optional[np.ndarray],
    ):
        solver = self.left_ik_solver if arm == "left" else self.right_ik_solver
        candidates: List[Tuple[str, Optional[np.ndarray]]] = [("seeded", current_seed)]
        if current_seed is not None:
            candidates.append(("unseeded", None))

        best_result = None
        best_mode = None
        best_score = None
        for mode, seed in candidates:
            result = solver.solve_ik(
                target_position=ee_pos_base,
                target_orientation_wxyz=ee_quat_base,
                current_joints=seed,
            )
            if result is None:
                continue
            solution = result.solution.detach().cpu().numpy().reshape(-1)
            if solution.shape[0] < 10:
                continue
            target_arm = solution[-6:].astype(np.float64)
            pos_err, rot_err = self._solution_error_to_ee_target(arm, target_arm, ee_pos_base, ee_quat_base)
            score = (pos_err, rot_err)
            if best_score is None or score < best_score:
                best_result = result
                best_mode = mode
                best_score = score
        if best_result is not None and best_mode == "unseeded":
            print(
                "[ik-candidate] "
                f"arm={arm} chosen=unseeded "
                f"pos_err={best_score[0]:.4f}m rot_err={best_score[1]:.2f}deg"
            )
        return best_result

    def _plan_path_joint_interp(self, arm: str, target_pose_world: np.ndarray) -> Optional[Dict]:
        current_arm = self._current_arm_joints(arm)
        current_full = np.concatenate([self.torso_qpos, current_arm], dtype=np.float64)
        ee_pos_base, ee_quat_base = self._target_tcp_world_to_ee_base(arm, target_pose_world)
        result = self._solve_ik_best_candidate(
            arm,
            ee_pos_base=ee_pos_base,
            ee_quat_base=ee_quat_base,
            current_seed=current_full,
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
        if self.urdfik_cartesian_interp_steps == -1:
            requested_waypoints = self._auto_cartesian_interp_steps(
                current_tcp_world,
                target_pose_world,
                trigger_translation_m=self.urdfik_cartesian_interp_auto_step_m,
            )
            effective_waypoints = requested_waypoints
            if effective_waypoints != 2:
                print(
                    "[ik-waypoints] "
                    f"arm={arm} requested=auto "
                    f"step_m={self.urdfik_cartesian_interp_auto_step_m:.4f} "
                    f"effective={effective_waypoints}"
                )
        else:
            requested_waypoints = self.urdfik_cartesian_interp_steps
            effective_waypoints = self._effective_cartesian_interp_steps(
                current_tcp_world,
                target_pose_world,
                requested_waypoints,
                min_translation_step_m=self.left_ik_solver.default_position_threshold * 3.0,
                min_rotation_step_rad=self.left_ik_solver.default_rotation_threshold * 1.5,
            )
        if requested_waypoints != -1 and effective_waypoints != requested_waypoints:
            print(
                "[ik-waypoints] "
                f"arm={arm} requested={requested_waypoints} "
                f"effective={effective_waypoints}"
            )
        tcp_waypoints_world = self._interpolate_tcp_pose_world_series(
            current_tcp_world,
            target_pose_world,
            effective_waypoints,
        )
        joint_waypoints: List[np.ndarray] = [current_arm.copy()]
        ee_waypoints_base: List[np.ndarray] = []
        current_seed = current_full.copy()
        for waypoint_index, tcp_waypoint_world in enumerate(tcp_waypoints_world[1:], start=1):
            ee_pos_base, ee_quat_base = self._target_tcp_world_to_ee_base(arm, tcp_waypoint_world)
            ee_waypoints_base.append(np.concatenate([ee_pos_base, ee_quat_base]).astype(np.float64))
            result = self._solve_ik_best_candidate(
                arm,
                ee_pos_base=ee_pos_base,
                ee_quat_base=ee_quat_base,
                current_seed=current_seed,
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
        traj = np.asarray(plan.get("position"), dtype=np.float64).reshape(-1, 6)
        if traj.shape[0] == 0:
            current = np.asarray(plan["current_joints"], dtype=np.float64).reshape(6)
            target = np.asarray(plan["target_joints"], dtype=np.float64).reshape(6)
            traj = np.linspace(current, target, num=max(int(num_steps), 2), endpoint=True)
        vel_traj = np.asarray(plan.get("velocity"), dtype=np.float64).reshape(-1, 6) if "velocity" in plan else np.zeros_like(traj, dtype=np.float64)
        if vel_traj.shape[0] != traj.shape[0]:
            vel_traj = np.zeros_like(traj, dtype=np.float64)
        for q, v in zip(traj, vel_traj):
            self.robot.set_arm_joints(q, v, arm)
            self.step_scene(steps=1)
        self.step_scene(steps=4)
        return "Success"

    def execute_plans(self, left_plan: Optional[Dict], right_plan: Optional[Dict]) -> Tuple[str, str]:
        left_status = self._plan_status(left_plan)
        right_status = self._plan_status(right_plan)

        left_ok = left_status == "Success"
        right_ok = right_status == "Success"
        left_pos = np.asarray(left_plan["position"], dtype=np.float64).reshape(-1, 6) if left_ok else None
        left_vel = np.asarray(left_plan["velocity"], dtype=np.float64).reshape(-1, 6) if left_ok else None
        right_pos = np.asarray(right_plan["position"], dtype=np.float64).reshape(-1, 6) if right_ok else None
        right_vel = np.asarray(right_plan["velocity"], dtype=np.float64).reshape(-1, 6) if right_ok else None

        left_idx = 0
        right_idx = 0
        left_n = int(left_pos.shape[0]) if left_ok else 0
        right_n = int(right_pos.shape[0]) if right_ok else 0

        while left_idx < left_n or right_idx < right_n:
            if left_ok and left_idx < left_n and (not right_ok or left_idx / max(left_n, 1) <= right_idx / max(right_n, 1)):
                self.robot.set_arm_joints(left_pos[left_idx], left_vel[left_idx], "left")
                left_idx += 1
            if right_ok and right_idx < right_n and (not left_ok or right_idx / max(right_n, 1) <= left_idx / max(left_n, 1)):
                self.robot.set_arm_joints(right_pos[right_idx], right_vel[right_idx], "right")
                right_idx += 1
            self.step_scene(steps=1)

        self.step_scene(steps=4)
        return left_status, right_status


base.HandRetargetR1Renderer = HandRetargetR1URDFIKRenderer


if __name__ == "__main__":
    base.main()
