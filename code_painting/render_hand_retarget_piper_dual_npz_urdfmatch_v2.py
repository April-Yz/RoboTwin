#!/usr/bin/env python3
"""Isolated Dense Replay v2 renderer with matched Piper kinematics.

This module deliberately does not change the legacy dense-retarget renderer.
It fixes four version-specific mismatches:

1. IK and SAPIEN both use ``piper_pika_agx.urdf``.
2. The requested joint interpolation count is restored after the legacy
   constructor overwrites it with its default value.
3. The HaMeR fingertip midpoint is treated as the robot TCP, so the link6 IK
   target is obtained by exactly inverting ``Robot._trans_endpose``.
4. Successful plans are held until the measured joints converge, matching the
   execution discipline used by Ours v2.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

import render_hand_retarget_r1_npz as base
from render_hand_retarget_piper_dual_npz_urdfik import (
    HandRetargetPiperDualURDFIKRenderer,
    PIPER_URDF,
)


class HandRetargetPiperDualURDFMatchV2Renderer(HandRetargetPiperDualURDFIKRenderer):
    """Dense-retarget renderer that keeps planning and execution conventions aligned."""

    VERSION = "dense_replay_urdfmatch_v2"
    JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
    # Ours v2 spends roughly 295 simulation steps on a commanded target
    # (24 * 10 interpolation/scene steps + settling/waiting).  The legacy
    # Dense path only spent about 70 before its fixed post-wait, which leaves
    # a repeatable first-frame servo lag.  Keep this as a *maximum*: the loop
    # exits as soon as every commanded joint is within tolerance.
    JOINT_WAIT_STEPS = 240
    JOINT_WAIT_TOL_RAD = 0.01
    TCP_POSITION_CORRECTION_ITERS = 5
    TCP_POSITION_CORRECTION_TOL_M = 0.003
    # For the same q, SAPIEN link6 axes equal Curobo FK link6 axes followed by
    # a local Ry(-90 deg). Link origins coincide; only the local axes differ.
    CUROBO_TO_SAPIEN_LINK_ROTATION = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    def __init__(self, *args, **kwargs) -> None:
        # Capture this before the parent pops it. The legacy parent then calls a
        # base constructor whose default resets the value to 2.
        requested_joint_waypoints = max(int(kwargs.get("urdfik_joint_interp_waypoints", 10)), 2)

        # Ours-v2-style robust positional IK. Dense human wrist orientations are
        # often not robot-reachable, so rotation is optimized but is not allowed
        # to make an otherwise valid positional target disappear.
        kwargs["urdfik_position_threshold_m"] = 0.001
        kwargs["urdfik_max_position_threshold_m"] = 0.02
        kwargs["urdfik_rotation_threshold_rad"] = 0.02
        kwargs["urdfik_max_rotation_threshold_rad"] = math.pi
        kwargs["urdfik_num_seeds"] = 1
        kwargs["urdfik_solution_selection"] = "joint_continuity"
        kwargs["urdfik_seed_perturbations"] = 6
        kwargs["urdfik_seed_perturbation_scale"] = 0.05

        super().__init__(*args, **kwargs)

        # Restore the actual CLI request after the inherited overwrite.
        self.urdfik_joint_interp_waypoints = requested_joint_waypoints
        self._pending_plans: Dict[str, Optional[Dict]] = {"left": None, "right": None}
        self._execution_index = 0
        self._audit_path = Path(self.robot_config_path).resolve().parent / "execution_audit.jsonl"
        self._metadata_path = self._audit_path.parent / "dense_replay_v2_metadata.json"
        self._audit_path.write_text("", encoding="utf-8")

        self._assert_model_and_joint_match()
        self._write_metadata()
        print(
            "[dense-v2] "
            f"version={self.VERSION} "
            f"joint_interp_waypoints={self.urdfik_joint_interp_waypoints} "
            f"joint_wait_steps={self.JOINT_WAIT_STEPS} "
            f"joint_wait_tol_rad={self.JOINT_WAIT_TOL_RAD:.4f} "
            "target_reference=hamer_fingertip_midpoint_as_tcp"
        )

    def _assert_model_and_joint_match(self) -> None:
        ik_path = Path(self.ik_urdf_path).resolve()
        expected_path = Path(PIPER_URDF).resolve()
        if ik_path != expected_path:
            raise RuntimeError(f"Dense Replay v2 requires IK URDF {expected_path}, got {ik_path}")

        with Path(self.robot_config_path).open("r", encoding="utf-8") as f:
            robot_config = json.load(f)
        for side in ("left", "right"):
            robot_file = str(robot_config.get(f"{side}_robot_file", ""))
            embodiment = robot_config.get(f"{side}_embodiment_config", {})
            urdf_name = str(embodiment.get("urdf_path", ""))
            configured_joint_names = list(embodiment.get("arm_joints_name", [[]])[0])
            if "piper_pika_agx" not in robot_file or urdf_name != "piper_pika_agx.urdf":
                raise RuntimeError(
                    f"Dense Replay v2 requires the SAPIEN piper_pika_agx model for {side}; "
                    f"got robot_file={robot_file!r}, urdf_path={urdf_name!r}"
                )
            if configured_joint_names != self.JOINT_NAMES:
                raise RuntimeError(
                    f"Dense Replay v2 joint order mismatch for {side}: "
                    f"expected {self.JOINT_NAMES}, got {configured_joint_names}"
                )

    def _write_metadata(self) -> None:
        payload = {
            "schema": "dense_replay_urdfmatch_v2.v1",
            "version": self.VERSION,
            "ik_urdf": str(Path(self.ik_urdf_path).resolve()),
            "simulation_robot_config": str(Path(self.robot_config_path).resolve()),
            "joint_order": self.JOINT_NAMES,
            "joint_interp_waypoints": int(self.urdfik_joint_interp_waypoints),
            "joint_wait_steps": int(self.JOINT_WAIT_STEPS),
            "joint_wait_tol_rad": float(self.JOINT_WAIT_TOL_RAD),
            "tcp_position_correction_iters": int(self.TCP_POSITION_CORRECTION_ITERS),
            "tcp_position_correction_tol_m": float(self.TCP_POSITION_CORRECTION_TOL_M),
            "target_reference": "HaMeR midpoint(thumb_tip,index_tip) == robot TCP",
            "tcp_to_link6": {
                "position": "p_link6 = p_tcp - R_tcp @ [gripper_bias, 0, 0]",
                "sapien_rotation": "R_sapien_link6 = R_tcp @ inv(delta_matrix) @ inv(global_trans_matrix)",
                "curobo_rotation": "R_curobo_link6 = R_sapien_link6 @ inv(Ry(-90deg))",
            },
            "curobo_to_sapien_link_rotation": self.CUROBO_TO_SAPIEN_LINK_ROTATION.tolist(),
            "ik_policy": {
                "position_threshold_m": float(self.urdfik_position_threshold_m),
                "max_position_threshold_m": float(self.left_ik_solver.max_position_threshold),
                "rotation_threshold_rad": float(self.urdfik_rotation_threshold_rad),
                "max_rotation_threshold_rad": float(self.left_ik_solver.max_rotation_threshold),
                "solution_selection": self.urdfik_solution_selection,
                "seed_perturbations": int(self.urdfik_seed_perturbations),
            },
            "legacy_entry_unchanged": True,
        }
        self._metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _target_tcp_world_to_ee_base(
        self,
        arm: str,
        target_pose_world: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Invert ``Robot._trans_endpose(..., is_endpose=True)`` exactly.

        The source point stored by HaMeR is the midpoint of thumb and index
        fingertips. It is therefore a TCP point, not link6 and not the legacy
        12 cm-shifted EE reference.
        """

        target_tcp_base = self.world_pose_to_base_pose_for_arm(target_pose_world, arm)
        tcp_position = np.asarray(target_tcp_base[:3], dtype=np.float64)
        tcp_rotation = R.from_quat(base.quat_wxyz_to_xyzw(target_tcp_base[3:])).as_matrix()

        robot = self.robot
        if robot is None:
            raise RuntimeError("Robot is unavailable.")
        gripper_bias = float(getattr(robot, f"{arm}_gripper_bias"))
        delta_matrix = np.asarray(getattr(robot, f"{arm}_delta_matrix"), dtype=np.float64).reshape(3, 3)
        global_matrix = np.asarray(getattr(robot, f"{arm}_global_trans_matrix"), dtype=np.float64).reshape(3, 3)

        link_position = tcp_position - tcp_rotation @ np.array([gripper_bias, 0.0, 0.0], dtype=np.float64)
        sapien_link_rotation = base.orthonormalize_rotation(
            tcp_rotation @ np.linalg.inv(delta_matrix) @ np.linalg.inv(global_matrix)
        )
        curobo_link_rotation = base.orthonormalize_rotation(
            sapien_link_rotation @ np.linalg.inv(self.CUROBO_TO_SAPIEN_LINK_ROTATION)
        )
        link_quat_wxyz = base.quat_xyzw_to_wxyz(R.from_matrix(curobo_link_rotation).as_quat())
        return link_position, link_quat_wxyz

    def plan_path(self, arm: str, target_pose_world: np.ndarray) -> Optional[Dict]:
        plan = super().plan_path(arm, target_pose_world)
        self._pending_plans[arm] = plan
        return plan

    def _solver_fk_tcp_base(self, arm: str, joints: np.ndarray) -> np.ndarray:
        solver = self.left_ik_solver if arm == "left" else self.right_ik_solver
        link_pos_base, link_quat_base, _ = solver.forward_kinematics(joints)
        curobo_link_rotation_base = R.from_quat(base.quat_wxyz_to_xyzw(link_quat_base)).as_matrix()
        sapien_link_rotation_base = base.orthonormalize_rotation(
            curobo_link_rotation_base @ self.CUROBO_TO_SAPIEN_LINK_ROTATION
        )
        global_matrix = np.asarray(getattr(self.robot, f"{arm}_global_trans_matrix"), dtype=np.float64).reshape(3, 3)
        delta_matrix = np.asarray(getattr(self.robot, f"{arm}_delta_matrix"), dtype=np.float64).reshape(3, 3)
        gripper_bias = float(getattr(self.robot, f"{arm}_gripper_bias"))
        tcp_rotation_base = base.orthonormalize_rotation(
            sapien_link_rotation_base @ global_matrix @ delta_matrix
        )
        tcp_position_base = np.asarray(link_pos_base, dtype=np.float64) + tcp_rotation_base @ np.array(
            [gripper_bias, 0.0, 0.0], dtype=np.float64
        )
        tcp_quat_base = base.quat_xyzw_to_wxyz(R.from_matrix(tcp_rotation_base).as_quat())
        return np.concatenate([tcp_position_base, tcp_quat_base]).astype(np.float64)

    def _plan_path_joint_interp(self, arm: str, target_pose_world: np.ndarray) -> Dict:
        """Solve link6 IK while iteratively keeping the reported TCP on target.

        With relaxed rotation tolerance, the solved TCP orientation can differ
        from the human orientation. Since TCP is 12 cm from link6, that rotation
        difference otherwise becomes a large apparent translation. Re-solving
        the link6 position against the FK TCP removes that coupling.
        """

        current_arm = self._current_arm_joints(arm)
        target_tcp_base = self.world_pose_to_base_pose_for_arm(target_pose_world, arm)
        ee_pos_base, ee_quat_base = self._target_tcp_world_to_ee_base(arm, target_pose_world)
        seed = current_arm.copy()
        target_arm = None
        correction_history = []

        for correction_iter in range(self.TCP_POSITION_CORRECTION_ITERS):
            result = self._solve_ik_best_candidate(
                arm,
                ee_pos_base,
                ee_quat_base,
                current_seed=seed,
            )
            if result is None:
                break
            solution = result.solution.detach().cpu().numpy().reshape(-1)
            if solution.shape[0] < 6:
                break
            target_arm = solution[-6:].astype(np.float64)
            planned_tcp_base = self._solver_fk_tcp_base(arm, target_arm)
            tcp_position_error = np.asarray(target_tcp_base[:3], dtype=np.float64) - planned_tcp_base[:3]
            correction_history.append(
                {
                    "iteration": int(correction_iter + 1),
                    "tcp_position_error_m": tcp_position_error.tolist(),
                    "tcp_position_error_norm_m": float(np.linalg.norm(tcp_position_error)),
                }
            )
            if np.linalg.norm(tcp_position_error) <= self.TCP_POSITION_CORRECTION_TOL_M:
                break
            ee_pos_base = np.asarray(ee_pos_base, dtype=np.float64) + tcp_position_error
            seed = target_arm.copy()

        if target_arm is None:
            plan = self._build_fail_plan(arm, current_arm, target_pose_world, ee_pos_base, ee_quat_base)
            plan["tcp_position_correction_history"] = correction_history
            return plan

        position_traj = np.linspace(
            current_arm,
            target_arm,
            num=self.urdfik_joint_interp_waypoints,
            endpoint=True,
            dtype=np.float64,
        )
        velocity_traj = np.zeros_like(position_traj, dtype=np.float64)
        return {
            "status": "Success",
            "solver": "urdfik_tcp_corrected_v2",
            "trajectory_mode": self.urdfik_trajectory_mode,
            "arm": arm,
            "current_joints": current_arm.copy(),
            "target_joints": target_arm.copy(),
            "position": position_traj,
            "velocity": velocity_traj,
            "target_pose_world": np.asarray(target_pose_world, dtype=np.float64).copy(),
            "target_pose_ee_base": np.concatenate([ee_pos_base, ee_quat_base]).astype(np.float64),
            "tcp_position_correction_history": correction_history,
            "planned_tcp_base": self._solver_fk_tcp_base(arm, target_arm),
        }

    def _current_joints(self, arm: str) -> np.ndarray:
        if arm == "left":
            values = self.robot.get_left_arm_real_jointState()[:6]
        else:
            values = self.robot.get_right_arm_real_jointState()[:6]
        return np.asarray(values, dtype=np.float64).reshape(6)

    @staticmethod
    def _quat_error_deg(lhs_wxyz: np.ndarray, rhs_wxyz: np.ndarray) -> float:
        lhs = R.from_quat(base.quat_wxyz_to_xyzw(base.normalize_quat_wxyz(lhs_wxyz)))
        rhs = R.from_quat(base.quat_wxyz_to_xyzw(base.normalize_quat_wxyz(rhs_wxyz)))
        return float(np.rad2deg(np.linalg.norm((lhs.inv() * rhs).as_rotvec())))

    def _solver_fk_tcp_world(self, arm: str, joints: np.ndarray) -> np.ndarray:
        tcp_base = self._solver_fk_tcp_base(arm, joints)
        return self.base_pose_to_world_pose_for_arm(tcp_base, arm)

    def _settle_successful_targets(self, plans: Dict[str, Optional[Dict]]) -> Tuple[int, Dict[str, Dict[str, float]]]:
        targets: Dict[str, np.ndarray] = {}
        for arm, plan in plans.items():
            if not isinstance(plan, dict) or str(plan.get("status", "")) != "Success":
                continue
            if "target_joints" not in plan:
                continue
            targets[arm] = np.asarray(plan["target_joints"], dtype=np.float64).reshape(6)

        if not targets:
            return 0, {}

        zero_velocity = np.zeros(6, dtype=np.float64)
        final_metrics: Dict[str, Dict[str, float]] = {}
        steps_used = 0
        for step_idx in range(self.JOINT_WAIT_STEPS + 1):
            final_metrics = {}
            for arm, target in targets.items():
                delta = target - self._current_joints(arm)
                final_metrics[arm] = {
                    "max_abs_err_rad": float(np.max(np.abs(delta))),
                    "l2_err_rad": float(np.linalg.norm(delta)),
                }
            if all(item["max_abs_err_rad"] <= self.JOINT_WAIT_TOL_RAD for item in final_metrics.values()):
                break
            if step_idx >= self.JOINT_WAIT_STEPS:
                break
            for arm, target in targets.items():
                self.robot.set_arm_joints(target, zero_velocity, arm)
            self.step_scene(steps=1)
            steps_used += 1
        return steps_used, final_metrics

    def _write_execution_audit(
        self,
        plans: Dict[str, Optional[Dict]],
        statuses: Dict[str, str],
        settle_steps_used: int,
        joint_metrics: Dict[str, Dict[str, float]],
    ) -> None:
        for arm in ("left", "right"):
            plan = plans.get(arm)
            record: Dict[str, object] = {
                "execution_index": int(self._execution_index),
                "arm": arm,
                "status": statuses.get(arm, "Missing"),
                "settle_steps_used": int(settle_steps_used),
                "joint_metrics_after_execute": joint_metrics.get(arm),
            }
            if isinstance(plan, dict):
                if "target_pose_world" in plan:
                    target = np.asarray(plan["target_pose_world"], dtype=np.float64).reshape(7)
                    record["target_tcp_world_wxyz"] = target.tolist()
                else:
                    target = None
                if "target_joints" in plan:
                    target_joints = np.asarray(plan["target_joints"], dtype=np.float64).reshape(6)
                    actual_joints = self._current_joints(arm)
                    planned_tcp = self._solver_fk_tcp_world(arm, target_joints)
                    actual_tcp = np.asarray(self.get_current_tcp_pose(arm), dtype=np.float64).reshape(7)
                    solver_fk_at_actual_tcp = self._solver_fk_tcp_world(arm, actual_joints)
                    sapien_link = self.robot.left_ee.global_pose if arm == "left" else self.robot.right_ee.global_pose
                    sapien_link_world = np.concatenate(
                        [
                            np.asarray(sapien_link.p, dtype=np.float64).reshape(3),
                            np.asarray(sapien_link.q, dtype=np.float64).reshape(4),
                        ]
                    )
                    record.update(
                        {
                            "target_joints": target_joints.tolist(),
                            "actual_joints_after_execute": actual_joints.tolist(),
                            "planned_fk_tcp_world_wxyz": planned_tcp.tolist(),
                            "actual_tcp_world_wxyz": actual_tcp.tolist(),
                            "solver_fk_at_actual_joints_tcp_world_wxyz": solver_fk_at_actual_tcp.tolist(),
                            "solver_fk_vs_sapien_tcp_pos_err_m": float(
                                np.linalg.norm(solver_fk_at_actual_tcp[:3] - actual_tcp[:3])
                            ),
                            "solver_fk_vs_sapien_tcp_rot_err_deg": self._quat_error_deg(
                                solver_fk_at_actual_tcp[3:], actual_tcp[3:]
                            ),
                            "sapien_link6_world_wxyz": sapien_link_world.tolist(),
                            "tcp_position_correction_history": plan.get("tcp_position_correction_history", []),
                        }
                    )
                    if target is not None:
                        record["planned_fk_vs_target_pos_err_m"] = float(np.linalg.norm(planned_tcp[:3] - target[:3]))
                        record["planned_fk_vs_target_rot_err_deg"] = self._quat_error_deg(planned_tcp[3:], target[3:])
                        record["actual_vs_target_pos_err_m"] = float(np.linalg.norm(actual_tcp[:3] - target[:3]))
                        record["actual_vs_target_rot_err_deg"] = self._quat_error_deg(actual_tcp[3:], target[3:])
            with self._audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def execute_plans(
        self,
        left_plan: Optional[Dict],
        right_plan: Optional[Dict],
    ) -> Tuple[str, str]:
        plans = {"left": left_plan, "right": right_plan}
        left_status, right_status = super().execute_plans(left_plan, right_plan)
        settle_steps_used, joint_metrics = self._settle_successful_targets(plans)
        self._write_execution_audit(
            plans,
            {"left": left_status, "right": right_status},
            settle_steps_used,
            joint_metrics,
        )
        self._execution_index += 1
        return left_status, right_status


__all__ = ["HandRetargetPiperDualURDFMatchV2Renderer"]
