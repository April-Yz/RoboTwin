#!/usr/bin/env python3
"""URDF-IK renderer for dual-instance Piper setup."""

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
from replay_piper_dual_h5 import PiperDualReplayRenderer
from urdfik import URDFInverseKinematics

# 使用与 SAPIEN 仿真一致的 URDF（piper_pika_agx），
# 而非 piper/piper.urdf（关节原点不同！IK 求解的关节角在错误 URDF 上执行会产生巨大误差）。
# 参见 envs/robot/piper_ik.py 的相同处理。
PIPER_URDF = PROJECT_ROOT / "assets" / "embodiments" / "piper_pika_agx" / "piper_pika_agx.urdf"
DEFAULT_INIT_ARM_JOINTS = np.array([0.0, 0.8, 1.2, 0.0, -0.4, 0.0], dtype=np.float64)
DEFAULT_INIT_GRIPPER_OPEN = 1.0


class HandRetargetPiperDualURDFIKRenderer(PiperDualReplayRenderer):
    def __init__(self, *args, **kwargs) -> None:
        self.urdfik_trajectory_mode = str(kwargs.pop("urdfik_trajectory_mode", "joint_interp"))
        raw_cartesian_interp_steps = int(kwargs.pop("urdfik_cartesian_interp_steps", 8))
        self.urdfik_cartesian_interp_auto_step_m = max(float(kwargs.pop("urdfik_cartesian_interp_auto_step_m", 0.05)), 1e-4)
        self.urdfik_cartesian_interp_steps = -1 if raw_cartesian_interp_steps == -1 else max(raw_cartesian_interp_steps, 2)
        self.urdfik_joint_interp_waypoints = max(int(kwargs.pop("urdfik_joint_interp_waypoints", 2)), 2)
        self.urdfik_position_threshold_m = max(float(kwargs.pop("urdfik_position_threshold_m", 0.001)), 1e-6)
        self.urdfik_rotation_threshold_rad = max(float(kwargs.pop("urdfik_rotation_threshold_rad", 0.02)), 1e-6)
        max_pos_raw = kwargs.pop("urdfik_max_position_threshold_m", None)
        max_rot_raw = kwargs.pop("urdfik_max_rotation_threshold_rad", None)
        self.urdfik_max_position_threshold_m = None if max_pos_raw is None else max(float(max_pos_raw), self.urdfik_position_threshold_m)
        self.urdfik_max_rotation_threshold_rad = None if max_rot_raw is None else max(float(max_rot_raw), self.urdfik_rotation_threshold_rad)
        self.urdfik_num_seeds = max(int(kwargs.pop("urdfik_num_seeds", 1)), 1)
        self.urdfik_execute_partial_cartesian_plan = bool(kwargs.pop("urdfik_execute_partial_cartesian_plan", False))
        self.urdfik_apply_global_trans_to_ik = bool(kwargs.pop("urdfik_apply_global_trans_to_ik", False))
        self.urdfik_solution_selection = str(kwargs.pop("urdfik_solution_selection", "pose_error"))
        self.urdfik_seed_perturbations = max(int(kwargs.pop("urdfik_seed_perturbations", 0)), 0)
        self.urdfik_seed_perturbation_scale = max(
            float(kwargs.pop("urdfik_seed_perturbation_scale", 0.05)),
            0.0,
        )
        max_joint_step_raw = kwargs.pop("urdfik_max_joint_step_rad", None)
        self.urdfik_max_joint_step_rad = (
            None if max_joint_step_raw is None or float(max_joint_step_raw) <= 0.0 else float(max_joint_step_raw)
        )
        kwargs["attach_planner"] = False
        if kwargs.get("init_left_arm_joints") is None:
            kwargs["init_left_arm_joints"] = DEFAULT_INIT_ARM_JOINTS.copy()
        if kwargs.get("init_right_arm_joints") is None:
            kwargs["init_right_arm_joints"] = DEFAULT_INIT_ARM_JOINTS.copy()
        if kwargs.get("init_gripper_open") is None:
            kwargs["init_gripper_open"] = DEFAULT_INIT_GRIPPER_OPEN
        super().__init__(*args, **kwargs)
        self.ik_urdf_path = PIPER_URDF
        solver_kwargs = dict(
            urdf_file=self.ik_urdf_path,
            base_link="base_link",
            ee_link="link6",
            position_threshold=self.urdfik_position_threshold_m,
            rotation_threshold=self.urdfik_rotation_threshold_rad,
            max_position_threshold=self.urdfik_max_position_threshold_m,
            max_rotation_threshold=self.urdfik_max_rotation_threshold_rad,
            num_seeds=self.urdfik_num_seeds,
        )
        self.left_ik_solver = URDFInverseKinematics(**solver_kwargs)
        self.right_ik_solver = URDFInverseKinematics(**solver_kwargs)
        print(f"[ik-mode] robot=piper_v2 solver=urdfik urdf={self.ik_urdf_path}")
        print(
            "[ik-trajectory] "
            f"mode={self.urdfik_trajectory_mode} "
            f"joint_interp_waypoints={self.urdfik_joint_interp_waypoints} "
            f"cartesian_interp_steps={self.urdfik_cartesian_interp_steps} "
            f"auto_step_m={self.urdfik_cartesian_interp_auto_step_m:.4f} "
            f"pos_thresh={self.urdfik_position_threshold_m:.4f} "
            f"max_pos_thresh={float(self.left_ik_solver.max_position_threshold):.4f} "
            f"rot_thresh={self.urdfik_rotation_threshold_rad:.4f} "
            f"max_rot_thresh={float(self.left_ik_solver.max_rotation_threshold):.4f} "
            f"num_seeds={self.urdfik_num_seeds} "
            f"partial_cartesian={int(self.urdfik_execute_partial_cartesian_plan)} "
            f"solution_selection={self.urdfik_solution_selection} "
            f"seed_perturbations={self.urdfik_seed_perturbations} "
            f"seed_perturbation_scale={self.urdfik_seed_perturbation_scale:.3f} "
            f"max_joint_step_rad={self.urdfik_max_joint_step_rad} "
            f"apply_global_trans_to_ik={int(self.urdfik_apply_global_trans_to_ik)} "
            f"exec_waypoint_scene_steps={self.execute_waypoint_scene_steps} "
            f"exec_settle_scene_steps={self.execute_settle_scene_steps}"
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
        target_pose_base = self.world_pose_to_base_pose_for_arm(target_pose_world, arm)
        target_pose_ee = self.robot._trans_from_gripper_to_endlink(target_pose_base.tolist(), arm_tag=arm)
        if not self.urdfik_apply_global_trans_to_ik:
            return np.asarray(target_pose_ee.p, dtype=np.float64), np.asarray(target_pose_ee.q, dtype=np.float64)
        # Optional diagnostic mode only. The default path intentionally matches
        # the direct Piper hand replay convention, which already produces the
        # visually correct gripper orientation for stored gripper poses.
        global_trans = np.asarray(
            self.robot.left_global_trans_matrix if arm == "left" else self.robot.right_global_trans_matrix,
            dtype=np.float64,
        ).reshape(3, 3)
        ee_rot_base = base.orthonormalize_rotation(
            np.asarray(R.from_quat(base.quat_wxyz_to_xyzw(target_pose_ee.q)).as_matrix(), dtype=np.float64)
            @ np.linalg.inv(global_trans)
        )
        ee_quat_base = base.quat_xyzw_to_wxyz(R.from_matrix(ee_rot_base).as_quat())
        return np.asarray(target_pose_ee.p, dtype=np.float64), np.asarray(ee_quat_base, dtype=np.float64)

    @staticmethod
    def _effective_cartesian_interp_steps(start_pose_world, target_pose_world, requested_waypoints, min_translation_step_m, min_rotation_step_rad):
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
        return max(2, min(requested_waypoints, max_steps_from_translation, max_steps_from_rotation))

    @staticmethod
    def _auto_cartesian_interp_steps(start_pose_world, target_pose_world, trigger_translation_m):
        start_pose_world = np.asarray(start_pose_world, dtype=np.float64).reshape(7)
        target_pose_world = np.asarray(target_pose_world, dtype=np.float64).reshape(7)
        translation_dist = float(np.linalg.norm(target_pose_world[:3] - start_pose_world[:3]))
        if translation_dist <= trigger_translation_m:
            return 2
        intermediate_waypoints = max(int(np.ceil(translation_dist / trigger_translation_m)) - 1, 1)
        return intermediate_waypoints + 2

    @staticmethod
    def _interpolate_tcp_pose_world_series(start_pose_world, target_pose_world, num_waypoints):
        start_pose_world = np.asarray(start_pose_world, dtype=np.float64).reshape(7)
        target_pose_world = np.asarray(target_pose_world, dtype=np.float64).reshape(7)
        num_waypoints = max(int(num_waypoints), 2)
        start_pos = start_pose_world[:3]
        target_pos = target_pose_world[:3]
        rotations = R.from_quat(np.stack([
            base.quat_wxyz_to_xyzw(base.normalize_quat_wxyz(start_pose_world[3:])),
            base.quat_wxyz_to_xyzw(base.normalize_quat_wxyz(target_pose_world[3:])),
        ], axis=0))
        slerp = Slerp([0.0, 1.0], rotations)
        waypoints = []
        for ratio in np.linspace(0.0, 1.0, num=num_waypoints, endpoint=True):
            interp_pos = (1.0 - ratio) * start_pos + ratio * target_pos
            interp_quat = base.quat_xyzw_to_wxyz(slerp(float(ratio)).as_quat())
            waypoints.append(np.concatenate([interp_pos, interp_quat]).astype(np.float64))
        return waypoints

    def _build_fail_plan(self, arm, current_arm, target_pose_world, ee_pos_base, ee_quat_base, reason=None, waypoint_index=None, waypoint_count=None):
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

    def _build_partial_cartesian_plan(
        self,
        arm,
        current_arm,
        target_pose_world,
        failed_ee_pos_base,
        failed_ee_quat_base,
        joint_waypoints,
        ee_waypoints_base,
        tcp_waypoints_world,
        reason,
        waypoint_index,
        waypoint_count,
    ):
        if len(joint_waypoints) < 2:
            return self._build_fail_plan(
                arm,
                current_arm,
                target_pose_world,
                failed_ee_pos_base,
                failed_ee_quat_base,
                reason=reason,
                waypoint_index=waypoint_index,
                waypoint_count=waypoint_count,
            )
        position_traj = np.stack(joint_waypoints, axis=0)
        velocity_traj = np.zeros_like(position_traj, dtype=np.float64)
        solved_waypoint_count = len(joint_waypoints)
        return {
            "status": "Partial",
            "solver": "urdfik",
            "trajectory_mode": self.urdfik_trajectory_mode,
            "arm": arm,
            "current_joints": current_arm.copy(),
            "target_joints": joint_waypoints[-1].copy(),
            "position": position_traj,
            "velocity": velocity_traj,
            "target_pose_world": np.asarray(target_pose_world, dtype=np.float64).copy(),
            "target_pose_ee_base": ee_waypoints_base[-1].copy() if ee_waypoints_base else np.concatenate([failed_ee_pos_base, failed_ee_quat_base]).astype(np.float64),
            "failed_target_pose_ee_base": np.concatenate([failed_ee_pos_base, failed_ee_quat_base]).astype(np.float64),
            "cartesian_waypoint_count": len(tcp_waypoints_world),
            "tcp_waypoints_world": np.stack(tcp_waypoints_world, axis=0),
            "solved_tcp_waypoints_world": np.stack(tcp_waypoints_world[:solved_waypoint_count], axis=0),
            "ee_waypoints_base": np.stack(ee_waypoints_base, axis=0) if ee_waypoints_base else np.zeros((0, 7), dtype=np.float64),
            "reason": str(reason),
            "failed_waypoint_index": int(waypoint_index),
            "waypoint_count": int(waypoint_count),
            "solved_waypoint_count": int(solved_waypoint_count),
        }

    def _try_first_segment_partial_waypoint(self, arm, current_tcp_world, failed_tcp_waypoint_world, current_seed):
        sub_waypoints = self._interpolate_tcp_pose_world_series(current_tcp_world, failed_tcp_waypoint_world, 5)
        for tcp_sub_waypoint_world in reversed(sub_waypoints[1:-1]):
            ee_pos_base, ee_quat_base = self._target_tcp_world_to_ee_base(arm, tcp_sub_waypoint_world)
            result = self._solve_ik_best_candidate(arm, ee_pos_base, ee_quat_base, current_seed=current_seed)
            if result is None:
                continue
            solution = result.solution.detach().cpu().numpy().reshape(-1)
            if solution.shape[0] < 6:
                continue
            target_arm = solution[-6:].astype(np.float64)
            return target_arm, np.concatenate([ee_pos_base, ee_quat_base]).astype(np.float64), tcp_sub_waypoint_world
        return None

    def _solution_error_to_ee_target(self, arm, target_arm, ee_pos_base, ee_quat_base):
        solver = self.left_ik_solver if arm == "left" else self.right_ik_solver
        fk_pos_base, fk_quat_wxyz_base, _ = solver.forward_kinematics(target_arm.reshape(6))
        pos_err = float(np.linalg.norm(np.asarray(fk_pos_base, dtype=np.float64).reshape(3) - np.asarray(ee_pos_base, dtype=np.float64).reshape(3)))
        rot_err = float(base.quat_angle_deg_wxyz(np.asarray(fk_quat_wxyz_base, dtype=np.float64).reshape(4), np.asarray(ee_quat_base, dtype=np.float64).reshape(4)))
        return pos_err, rot_err

    def _solve_ik_best_candidate(self, arm, ee_pos_base, ee_quat_base, current_seed):
        solver = self.left_ik_solver if arm == "left" else self.right_ik_solver
        candidates = [("seeded", current_seed)]
        if current_seed is not None:
            rng = np.random.RandomState(42)
            for perturb_idx in range(self.urdfik_seed_perturbations):
                perturbation = rng.normal(
                    0.0,
                    self.urdfik_seed_perturbation_scale,
                    size=np.asarray(current_seed).shape,
                )
                candidates.append(
                    (
                        f"perturbed_{perturb_idx + 1}",
                        np.asarray(current_seed, dtype=np.float64) + perturbation,
                    )
                )
            candidates.append(("unseeded", None))
        best_result = None
        best_mode = None
        best_score = None
        for mode, seed in candidates:
            result = solver.solve_ik(target_position=ee_pos_base, target_orientation_wxyz=ee_quat_base, current_joints=seed)
            if result is None:
                continue
            solution = result.solution.detach().cpu().numpy().reshape(-1)
            if solution.shape[0] < 6:
                continue
            target_arm = solution[-6:].astype(np.float64)
            pos_err, rot_err = self._solution_error_to_ee_target(arm, target_arm, ee_pos_base, ee_quat_base)
            if current_seed is None:
                joint_l2 = 0.0
                joint_max = 0.0
            else:
                joint_delta = target_arm - np.asarray(current_seed, dtype=np.float64).reshape(6)
                joint_l2 = float(np.linalg.norm(joint_delta))
                joint_max = float(np.max(np.abs(joint_delta)))
            continuity_violation = (
                self.urdfik_max_joint_step_rad is not None
                and joint_max > float(self.urdfik_max_joint_step_rad)
            )
            if self.urdfik_solution_selection == "joint_continuity":
                score = (int(continuity_violation), joint_l2, joint_max, pos_err, rot_err)
            else:
                score = (int(continuity_violation), pos_err, rot_err, joint_l2, joint_max)
            if best_score is None or score < best_score:
                best_result = result
                best_mode = mode
                best_score = score
        if best_result is None:
            return None
        best_solution = best_result.solution.detach().cpu().numpy().reshape(-1)[-6:].astype(np.float64)
        best_pos_err, best_rot_err = self._solution_error_to_ee_target(
            arm, best_solution, ee_pos_base, ee_quat_base
        )
        if current_seed is None:
            best_joint_l2 = 0.0
            best_joint_max = 0.0
        else:
            best_joint_delta = best_solution - np.asarray(current_seed, dtype=np.float64).reshape(6)
            best_joint_l2 = float(np.linalg.norm(best_joint_delta))
            best_joint_max = float(np.max(np.abs(best_joint_delta)))
        if (
            self.urdfik_max_joint_step_rad is not None
            and best_joint_max > float(self.urdfik_max_joint_step_rad)
        ):
            print(
                f"[ik-continuity-reject] arm={arm} mode={best_mode} "
                f"joint_max={best_joint_max:.3f}rad joint_l2={best_joint_l2:.3f}rad "
                f"limit={float(self.urdfik_max_joint_step_rad):.3f}rad "
                f"pos_err={best_pos_err:.4f}m rot_err={best_rot_err:.2f}deg"
            )
            return None
        if best_mode == "unseeded" or self.urdfik_solution_selection == "joint_continuity":
            print(
                f"[ik-candidate] arm={arm} chosen={best_mode} "
                f"joint_max={best_joint_max:.3f}rad joint_l2={best_joint_l2:.3f}rad "
                f"pos_err={best_pos_err:.4f}m rot_err={best_rot_err:.2f}deg"
            )
        return best_result

    def _plan_path_joint_interp(self, arm, target_pose_world):
        current_arm = self._current_arm_joints(arm)
        ee_pos_base, ee_quat_base = self._target_tcp_world_to_ee_base(arm, target_pose_world)
        result = self._solve_ik_best_candidate(arm, ee_pos_base, ee_quat_base, current_seed=current_arm.copy())
        if result is None:
            return self._build_fail_plan(arm, current_arm, target_pose_world, ee_pos_base, ee_quat_base)
        solution = result.solution.detach().cpu().numpy().reshape(-1)
        if solution.shape[0] < 6:
            return self._build_fail_plan(arm, current_arm, target_pose_world, ee_pos_base, ee_quat_base, reason=f"expected >=6 joints from IK, got {solution.shape[0]}")
        target_arm = solution[-6:].astype(np.float64)
        position_traj = np.linspace(current_arm, target_arm, num=self.urdfik_joint_interp_waypoints, endpoint=True, dtype=np.float64)
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

    def _plan_path_cartesian_interp_ik(self, arm, target_pose_world):
        current_arm = self._current_arm_joints(arm)
        current_tcp_world = np.asarray(self.get_current_tcp_pose(arm), dtype=np.float64).reshape(7)
        if self.urdfik_cartesian_interp_steps == -1:
            requested_waypoints = self._auto_cartesian_interp_steps(current_tcp_world, target_pose_world, trigger_translation_m=self.urdfik_cartesian_interp_auto_step_m)
            effective_waypoints = requested_waypoints
        else:
            requested_waypoints = self.urdfik_cartesian_interp_steps
            effective_waypoints = self._effective_cartesian_interp_steps(current_tcp_world, target_pose_world, requested_waypoints, self.left_ik_solver.default_position_threshold * 3.0, self.left_ik_solver.default_rotation_threshold * 1.5)
        tcp_waypoints_world = self._interpolate_tcp_pose_world_series(current_tcp_world, target_pose_world, effective_waypoints)
        joint_waypoints = [current_arm.copy()]
        ee_waypoints_base = []
        current_seed = current_arm.copy()
        for waypoint_index, tcp_waypoint_world in enumerate(tcp_waypoints_world[1:], start=1):
            ee_pos_base, ee_quat_base = self._target_tcp_world_to_ee_base(arm, tcp_waypoint_world)
            ee_waypoints_base.append(np.concatenate([ee_pos_base, ee_quat_base]).astype(np.float64))
            result = self._solve_ik_best_candidate(arm, ee_pos_base, ee_quat_base, current_seed=current_seed)
            if result is None:
                if self.urdfik_execute_partial_cartesian_plan:
                    if len(joint_waypoints) < 2:
                        sub_result = self._try_first_segment_partial_waypoint(arm, current_tcp_world, tcp_waypoint_world, current_seed)
                        if sub_result is not None:
                            sub_target_arm, sub_ee_base, sub_tcp_world = sub_result
                            joint_waypoints.append(sub_target_arm.copy())
                            ee_waypoints_base[-1] = sub_ee_base.copy()
                            tcp_waypoints_world = [tcp_waypoints_world[0], sub_tcp_world, *tcp_waypoints_world[1:]]
                            solved_ee_waypoints_base = ee_waypoints_base
                        else:
                            solved_ee_waypoints_base = ee_waypoints_base[:-1]
                    else:
                        solved_ee_waypoints_base = ee_waypoints_base[:-1]
                    return self._build_partial_cartesian_plan(
                        arm,
                        current_arm,
                        target_pose_world,
                        ee_pos_base,
                        ee_quat_base,
                        joint_waypoints,
                        solved_ee_waypoints_base,
                        tcp_waypoints_world,
                        reason="failed during cartesian waypoint IK",
                        waypoint_index=waypoint_index,
                        waypoint_count=len(tcp_waypoints_world),
                    )
                return self._build_fail_plan(arm, current_arm, target_pose_world, ee_pos_base, ee_quat_base, reason="failed during cartesian waypoint IK", waypoint_index=waypoint_index, waypoint_count=len(tcp_waypoints_world))
            solution = result.solution.detach().cpu().numpy().reshape(-1)
            if solution.shape[0] < 6:
                if self.urdfik_execute_partial_cartesian_plan:
                    if len(joint_waypoints) < 2:
                        sub_result = self._try_first_segment_partial_waypoint(arm, current_tcp_world, tcp_waypoint_world, current_seed)
                        if sub_result is not None:
                            sub_target_arm, sub_ee_base, sub_tcp_world = sub_result
                            joint_waypoints.append(sub_target_arm.copy())
                            ee_waypoints_base[-1] = sub_ee_base.copy()
                            tcp_waypoints_world = [tcp_waypoints_world[0], sub_tcp_world, *tcp_waypoints_world[1:]]
                            solved_ee_waypoints_base = ee_waypoints_base
                        else:
                            solved_ee_waypoints_base = ee_waypoints_base[:-1]
                    else:
                        solved_ee_waypoints_base = ee_waypoints_base[:-1]
                    return self._build_partial_cartesian_plan(
                        arm,
                        current_arm,
                        target_pose_world,
                        ee_pos_base,
                        ee_quat_base,
                        joint_waypoints,
                        solved_ee_waypoints_base,
                        tcp_waypoints_world,
                        reason=f"expected >=6 joints from IK, got {solution.shape[0]}",
                        waypoint_index=waypoint_index,
                        waypoint_count=len(tcp_waypoints_world),
                    )
                return self._build_fail_plan(arm, current_arm, target_pose_world, ee_pos_base, ee_quat_base, reason=f"expected >=6 joints from IK, got {solution.shape[0]}", waypoint_index=waypoint_index, waypoint_count=len(tcp_waypoints_world))
            target_arm = solution[-6:].astype(np.float64)
            joint_waypoints.append(target_arm.copy())
            current_seed = target_arm.copy()
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

    def plan_path(self, arm, target_pose_world):
        if self.robot is None:
            return None
        if self.urdfik_trajectory_mode == "joint_interp":
            return self._plan_path_joint_interp(arm, target_pose_world)
        if self.urdfik_trajectory_mode == "cartesian_interp_ik":
            return self._plan_path_cartesian_interp_ik(arm, target_pose_world)
        raise ValueError(f"Unsupported urdfik_trajectory_mode: {self.urdfik_trajectory_mode}")

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
            self.step_scene(steps=self.execute_waypoint_scene_steps)
        if self.execute_settle_scene_steps > 0:
            self.step_scene(steps=self.execute_settle_scene_steps)
        return left_status, right_status


__all__ = ["HandRetargetPiperDualURDFIKRenderer"]
