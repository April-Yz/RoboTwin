#!/usr/bin/env python3
"""Replay H5 trajectories on the 6-DoF R1 robot model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import render_hand_retarget_r1_npz as base
from replay_r1_h5 import (
    H5ReplayTrajectory,
    ReplayRenderer,
    adapt_joint_dim,
    base_pose_to_world_pose,
    build_frame_indices,
    execute_joint_targets,
    parse_optional_base_pose,
    resolve_arms,
)
from urdfik import URDFInverseKinematics


DEFAULT_R1_URDF = PROJECT_ROOT / "galaxea_sim" / "assets" / "r1" / "robot.urdf"
DEFAULT_R1_CONFIG = PROJECT_ROOT / "robot_config_R1.json"


class ReplayR1URDFIKRenderer(ReplayRenderer):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["attach_planner"] = False
        super().__init__(*args, **kwargs)
        self.ik_urdf_path = DEFAULT_R1_URDF
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
                "solver": "urdfik_r1",
                "arm": arm,
                "current_joints": current_arm.copy(),
                "target_pose_world": np.asarray(target_pose_world, dtype=np.float64).copy(),
                "target_pose_ee_base": np.concatenate([ee_pos_base, ee_quat_base]).astype(np.float64),
            }

        solution = result.solution.detach().cpu().numpy().reshape(-1)
        if solution.shape[0] < 10:
            return {
                "status": "Fail",
                "solver": "urdfik_r1",
                "arm": arm,
                "reason": f"expected >=10 joints from IK, got {solution.shape[0]}",
                "current_joints": current_arm.copy(),
                "target_pose_world": np.asarray(target_pose_world, dtype=np.float64).copy(),
                "target_pose_ee_base": np.concatenate([ee_pos_base, ee_quat_base]).astype(np.float64),
            }

        target_arm = solution[-6:].astype(np.float64)
        position_traj = np.stack([current_arm, target_arm], axis=0)
        velocity_traj = np.zeros_like(position_traj, dtype=np.float64)
        return {
            "status": "Success",
            "solver": "urdfik_r1",
            "arm": arm,
            "current_joints": current_arm.copy(),
            "target_joints": target_arm.copy(),
            "position": position_traj,
            "velocity": velocity_traj,
            "target_pose_world": np.asarray(target_pose_world, dtype=np.float64).copy(),
            "target_pose_ee_base": np.concatenate([ee_pos_base, ee_quat_base]).astype(np.float64),
        }

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


def build_renderer(args: argparse.Namespace) -> ReplayR1URDFIKRenderer:
    return ReplayR1URDFIKRenderer(
        robot_config_path=args.robot_config,
        image_width=args.image_width,
        image_height=args.image_height,
        fovy_deg=args.fovy_deg,
        torso_qpos=args.torso_qpos,
        robot_base_pose_override=parse_optional_base_pose(args.robot_base_pose),
        third_person_view=bool(args.third_person_view),
        need_topp=False,
        link_cam_debug_enable=False,
        link_cam_axis_mode="none",
        link_cam_debug_rot_xyz_deg=[0.0, 0.0, 0.0],
        link_cam_debug_shift_fru=[0.0, 0.0, 0.0],
        camera_cv_axis_mode="legacy_r1",
        head_camera_local_pos=base.DEFAULT_HEAD_CAMERA_LOCAL_POS,
        head_camera_local_quat_wxyz=base.DEFAULT_HEAD_CAMERA_LOCAL_QUAT_WXYZ,
        wrist_camera_local_pos=base.DEFAULT_WRIST_CAMERA_LOCAL_POS,
        wrist_camera_local_quat_wxyz=base.DEFAULT_WRIST_CAMERA_LOCAL_QUAT_WXYZ,
        camera_debug_target="head",
        enable_viewer=bool(args.enable_viewer),
        viewer_frame_delay=args.viewer_frame_delay,
        viewer_wait_at_end=bool(args.viewer_wait_at_end),
        debug_mode=False,
        debug_force_orientation="none",
        debug_visualize_targets=bool(args.debug_visualize_targets),
        debug_target_axis_length=args.debug_target_axis_length,
        debug_target_axis_thickness=args.debug_target_axis_thickness,
        orientation_remap_label="identity",
        orientation_remap_matrix=np.eye(3, dtype=np.float64),
        stored_orientation_post_rot_xyz_deg=[0.0, 0.0, 0.0],
        target_world_offset_xyz=args.target_world_offset_xyz,
        left_target_world_offset_xyz=args.left_target_world_offset_xyz,
        right_target_world_offset_xyz=args.right_target_world_offset_xyz,
        target_world_z_offset=args.target_world_z_offset,
        disable_table=bool(args.disable_table),
        camera_sweep_enable=False,
        camera_sweep_steps_deg=[0.0],
        init_left_arm_joints=args.init_left_arm_joints,
        init_right_arm_joints=args.init_right_arm_joints,
        init_gripper_open=args.init_gripper_open,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay H5 trajectories for the 6-DoF R1 robot.")
    parser.add_argument("--input_h5", type=Path, required=True, help="Input h5 file.")
    parser.add_argument("--control_mode", choices=["joint", "eepose"], default="joint")
    parser.add_argument("--data_group", choices=["obs", "action"], default="obs")
    parser.add_argument("--arms", choices=["left", "right", "both"], default="both")
    parser.add_argument("--robot_config", type=Path, default=DEFAULT_R1_CONFIG)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--fovy_deg", type=float, default=90.0)
    parser.add_argument("--torso_qpos", type=float, nargs=4, default=base.DEFAULT_TORSO_QPOS.tolist())
    parser.add_argument("--robot_base_pose", type=float, nargs=7, default=None, metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"))
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=-1)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--joint_interp_steps", type=int, default=10)
    parser.add_argument("--settle_steps", type=int, default=2)
    parser.add_argument("--use_gripper", type=int, default=1)
    parser.add_argument("--third_person_view", type=int, default=1)
    parser.add_argument("--enable_viewer", type=int, default=1)
    parser.add_argument("--viewer_frame_delay", type=float, default=0.0)
    parser.add_argument("--viewer_wait_at_end", type=int, default=1)
    parser.add_argument("--disable_table", type=int, default=1)
    parser.add_argument("--debug_visualize_targets", type=int, default=1)
    parser.add_argument("--debug_target_axis_length", type=float, default=0.08)
    parser.add_argument("--debug_target_axis_thickness", type=float, default=0.004)
    parser.add_argument("--target_world_offset_xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"))
    parser.add_argument("--left_target_world_offset_xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"))
    parser.add_argument("--right_target_world_offset_xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"))
    parser.add_argument("--target_world_z_offset", type=float, default=0.0)
    parser.add_argument("--init_left_arm_joints", type=float, nargs=6, default=None, metavar=("J1", "J2", "J3", "J4", "J5", "J6"))
    parser.add_argument("--init_right_arm_joints", type=float, nargs=6, default=None, metavar=("J1", "J2", "J3", "J4", "J5", "J6"))
    parser.add_argument("--init_gripper_open", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arms = resolve_arms(args.arms)
    trajectory = H5ReplayTrajectory(args.input_h5, args.data_group)
    frame_indices = build_frame_indices(
        trajectory.length,
        args.frame_start,
        args.frame_end,
        args.frame_stride,
        args.max_frames,
    )
    if not frame_indices:
        raise ValueError("No frames selected. Check --frame_start/--frame_end/--frame_stride/--max_frames.")

    renderer = build_renderer(args)
    warned_joint_dims: set = set()
    expected_dims = {
        "left": len(renderer.robot.left_arm_joints) if renderer.robot is not None else 6,
        "right": len(renderer.robot.right_arm_joints) if renderer.robot is not None else 6,
    }
    print(
        "[replay-r1] "
        f"file={trajectory.path} group={args.data_group} control_mode={args.control_mode} "
        f"frames={len(frame_indices)} arms={','.join(arms)}"
    )

    for replay_idx, frame_idx in enumerate(frame_indices):
        left_gripper = trajectory.gripper_frame("left", frame_idx) if bool(args.use_gripper) and "left" in arms else None
        right_gripper = trajectory.gripper_frame("right", frame_idx) if bool(args.use_gripper) and "right" in arms else None

        if args.control_mode == "joint":
            left_target = None
            right_target = None
            if "left" in arms:
                left_target = adapt_joint_dim(
                    trajectory.joint_frame("left", frame_idx),
                    expected_dims["left"],
                    trajectory.fallback_joint_frame("left", frame_idx),
                    "left",
                    warned_joint_dims,
                )
            if "right" in arms:
                right_target = adapt_joint_dim(
                    trajectory.joint_frame("right", frame_idx),
                    expected_dims["right"],
                    trajectory.fallback_joint_frame("right", frame_idx),
                    "right",
                    warned_joint_dims,
                )
            renderer.update_target_axis_visuals(None, None)
            execute_joint_targets(
                renderer=renderer,
                left_target=left_target,
                right_target=right_target,
                interp_steps=args.joint_interp_steps,
                settle_steps=args.settle_steps,
            )
            if bool(args.use_gripper):
                renderer.set_grippers(left_gripper, right_gripper)
            print(f"[frame {frame_idx:04d}] replay_step={replay_idx + 1}/{len(frame_indices)} mode=joint")
            continue

        left_world = None
        right_world = None
        if "left" in arms:
            left_world = base_pose_to_world_pose(renderer, trajectory.eepose_frame("left", frame_idx))
            left_world = renderer.apply_target_world_offset("left", left_world)
        if "right" in arms:
            right_world = base_pose_to_world_pose(renderer, trajectory.eepose_frame("right", frame_idx))
            right_world = renderer.apply_target_world_offset("right", right_world)

        renderer.update_target_axis_visuals(left_world, right_world)
        left_plan = renderer.plan_path("left", left_world) if left_world is not None else None
        right_plan = renderer.plan_path("right", right_world) if right_world is not None else None
        left_status, right_status = renderer.execute_plans(left_plan, right_plan)
        if bool(args.use_gripper):
            renderer.set_grippers(left_gripper, right_gripper)
        print(
            f"[frame {frame_idx:04d}] "
            f"left={base.short_status(left_status)} right={base.short_status(right_status)}"
        )

    renderer.hold_viewer()


if __name__ == "__main__":
    main()
