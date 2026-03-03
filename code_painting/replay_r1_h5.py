#!/usr/bin/env python3
"""Replay R1 H5 trajectories in RoboTwin with joint or EE-pose control."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import sapien.core as sapien

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import render_hand_retarget_r1_npz as base


def parse_optional_base_pose(values: Optional[Sequence[float]]) -> Optional[List[float]]:
    if values is None:
        return None
    if len(values) != 7:
        raise ValueError("--robot_base_pose expects 7 numbers: x y z qw qx qy qz")
    return [float(v) for v in values]


def resolve_arms(label: str) -> Tuple[str, ...]:
    if label == "both":
        return ("left", "right")
    return (label,)


def build_frame_indices(length: int, frame_start: int, frame_end: int, frame_stride: int, max_frames: int) -> List[int]:
    start = max(int(frame_start), 0)
    end = length if int(frame_end) < 0 else min(int(frame_end) + 1, length)
    stride = max(int(frame_stride), 1)
    indices = list(range(start, end, stride))
    if max_frames > 0:
        indices = indices[: max_frames]
    return indices


def base_pose_to_world_pose(renderer: base.HandRetargetR1Renderer, pose_base: np.ndarray) -> np.ndarray:
    if renderer._base_pose is None:
        raise RuntimeError("Renderer base pose is unavailable.")
    pose_base = np.asarray(pose_base, dtype=np.float64).reshape(7)
    base_mat = base.pose_to_matrix(renderer._base_pose)
    local_pose = sapien.Pose(pose_base[:3], base.normalize_quat_wxyz(pose_base[3:]))
    world_pose = base.matrix_to_pose(base_mat @ base.pose_to_matrix(local_pose))
    return np.concatenate(
        [
            np.asarray(world_pose.p, dtype=np.float64),
            base.normalize_quat_wxyz(world_pose.q),
        ]
    )


class ReplayRenderer(base.HandRetargetR1Renderer):
    def __init__(self, *args, attach_planner: bool = True, **kwargs) -> None:
        self.attach_planner = bool(attach_planner)
        super().__init__(*args, **kwargs)

    def _load_robot(self) -> None:
        from envs.robot import Robot

        with self.robot_config_path.open("r", encoding="utf-8") as f:
            robot_cfg = json.load(f)

        self.robot = Robot(self.scene, self.need_topp, **robot_cfg)
        self.robot.init_joints()
        self.robot.move_to_homestate()
        self._set_fixed_torso()

        base_pose_raw = self.robot_base_pose_override
        if base_pose_raw is None:
            pose_cfg = robot_cfg["left_embodiment_config"]["robot_pose"][0]
            base_pose_raw = pose_cfg[:3] + pose_cfg[-4:]
        base_pose_arr = np.asarray(base_pose_raw, dtype=np.float64).reshape(7)
        self._base_pose = sapien.Pose(base_pose_arr[:3], base.normalize_quat_wxyz(base_pose_arr[3:]))
        self.robot.left_entity.set_root_pose(self._base_pose)
        self.robot.right_entity.set_root_pose(self._base_pose)
        self.robot.left_entity_origion_pose = self._base_pose
        self.robot.right_entity_origion_pose = self._base_pose

        self.robot.left_gripper_val = 0.8
        self.robot.right_gripper_val = 0.8
        if self.attach_planner:
            print("[init] attaching planner")
            self.robot.set_planner(self.scene)
        else:
            print("[init] skipping planner for joint replay")

        self._head_camera_link = self._find_robot_link(["zed_link", "head_camera", "head", "camera_link"])
        if self._head_camera_link is None:
            raise RuntimeError("Could not find zed/head camera link on R1.")
        self._left_wrist_camera_link = self._find_robot_link(["left_realsense_link", "left_D405_link", "left_camera"])
        self._right_wrist_camera_link = self._find_robot_link(["right_realsense_link", "right_D405_link", "right_camera"])

        if self.debug_visualize_targets:
            self._left_target_axis_actor = self._create_debug_axis_actor("left_target_axis")
            self._right_target_axis_actor = self._create_debug_axis_actor("right_target_axis")

        self._update_table_pose()
        self.update_robot_link_cameras()


class H5ReplayTrajectory:
    def __init__(self, h5_path: Path, data_group: str) -> None:
        self.path = Path(h5_path)
        self.data_group = str(data_group)
        self.joints: Dict[str, np.ndarray] = {}
        self.eeposes: Dict[str, np.ndarray] = {}
        self.grippers: Dict[str, Optional[np.ndarray]] = {}
        self.obs_joint_fallback: Dict[str, Optional[np.ndarray]] = {}

        if not self.path.is_file():
            raise FileNotFoundError(f"H5 file not found: {self.path}")

        with h5py.File(self.path, "r") as f:
            for side in ("left", "right"):
                joint_key = f"{self.data_group}/arm_{side}/joint_pos"
                eef_pos_key = f"{self.data_group}/arm_{side}/eef_pos"
                eef_quat_key = f"{self.data_group}/arm_{side}/eef_quat"
                gripper_key = f"{self.data_group}/gripper_{side}/joint_pos"
                obs_joint_key = f"obs/arm_{side}/joint_pos"

                if joint_key not in f:
                    raise KeyError(f"Missing dataset: {joint_key}")
                if eef_pos_key not in f or eef_quat_key not in f:
                    raise KeyError(f"Missing dataset: {eef_pos_key} or {eef_quat_key}")

                self.joints[side] = np.asarray(f[joint_key][:], dtype=np.float64)
                eef_pos = np.asarray(f[eef_pos_key][:], dtype=np.float64)
                eef_quat = np.asarray(f[eef_quat_key][:], dtype=np.float64)
                self.eeposes[side] = np.concatenate([eef_pos, eef_quat], axis=1)
                self.grippers[side] = np.asarray(f[gripper_key][:], dtype=np.float64) if gripper_key in f else None
                self.obs_joint_fallback[side] = np.asarray(f[obs_joint_key][:], dtype=np.float64) if obs_joint_key in f else None

        lengths = []
        for side in ("left", "right"):
            lengths.append(int(self.joints[side].shape[0]))
            lengths.append(int(self.eeposes[side].shape[0]))
            if self.grippers[side] is not None:
                lengths.append(int(self.grippers[side].shape[0]))
        self.length = min(lengths)

    def joint_frame(self, side: str, frame_idx: int) -> np.ndarray:
        return np.asarray(self.joints[side][frame_idx], dtype=np.float64).reshape(-1)

    def eepose_frame(self, side: str, frame_idx: int) -> np.ndarray:
        return np.asarray(self.eeposes[side][frame_idx], dtype=np.float64).reshape(7)

    def gripper_frame(self, side: str, frame_idx: int) -> Optional[float]:
        values = self.grippers.get(side)
        if values is None:
            return None
        return float(np.asarray(values[frame_idx], dtype=np.float64).reshape(-1)[0])

    def fallback_joint_frame(self, side: str, frame_idx: int) -> Optional[np.ndarray]:
        values = self.obs_joint_fallback.get(side)
        if values is None:
            return None
        return np.asarray(values[frame_idx], dtype=np.float64).reshape(-1)


def adapt_joint_dim(
    raw_joint: np.ndarray,
    expected_dim: int,
    fallback_joint: Optional[np.ndarray],
    side: str,
    warned: set,
) -> np.ndarray:
    raw_joint = np.asarray(raw_joint, dtype=np.float64).reshape(-1)
    if raw_joint.size == expected_dim:
        return raw_joint
    if raw_joint.size > expected_dim:
        if (side, "truncate") not in warned:
            print(
                f"[joint-adapt] {side}: input dim={raw_joint.size}, expected={expected_dim}; "
                f"truncating to first {expected_dim} values."
            )
            warned.add((side, "truncate"))
        return raw_joint[:expected_dim]

    pad = np.zeros(expected_dim - raw_joint.size, dtype=np.float64)
    if fallback_joint is not None and fallback_joint.size >= expected_dim:
        pad = np.asarray(fallback_joint[raw_joint.size:expected_dim], dtype=np.float64)
    if (side, "pad") not in warned:
        print(
            f"[joint-adapt] {side}: input dim={raw_joint.size}, expected={expected_dim}; "
            f"padding missing dims with obs fallback if available."
        )
        warned.add((side, "pad"))
    return np.concatenate([raw_joint, pad], axis=0)


def execute_joint_targets(
    renderer: base.HandRetargetR1Renderer,
    left_target: Optional[np.ndarray],
    right_target: Optional[np.ndarray],
    interp_steps: int,
    settle_steps: int,
) -> None:
    if renderer.robot is None:
        raise RuntimeError("Robot is not initialized.")

    left_current = None
    right_current = None
    if left_target is not None:
        left_current = np.asarray(renderer.robot.get_left_arm_real_jointState()[: len(left_target)], dtype=np.float64)
    if right_target is not None:
        right_current = np.asarray(renderer.robot.get_right_arm_real_jointState()[: len(right_target)], dtype=np.float64)

    steps = max(int(interp_steps), 1)
    for step in range(steps):
        alpha = float(step + 1) / float(steps)
        if left_target is not None and left_current is not None:
            q_left = left_current * (1.0 - alpha) + left_target * alpha
            renderer.robot.set_arm_joints(q_left, np.zeros_like(q_left), "left")
        if right_target is not None and right_current is not None:
            q_right = right_current * (1.0 - alpha) + right_target * alpha
            renderer.robot.set_arm_joints(q_right, np.zeros_like(q_right), "right")
        renderer.step_scene(steps=1)

    if settle_steps > 0:
        renderer.step_scene(steps=settle_steps)


def build_renderer(args: argparse.Namespace) -> ReplayRenderer:
    return ReplayRenderer(
        robot_config_path=args.robot_config,
        image_width=args.image_width,
        image_height=args.image_height,
        fovy_deg=args.fovy_deg,
        torso_qpos=args.torso_qpos,
        robot_base_pose_override=parse_optional_base_pose(args.robot_base_pose),
        third_person_view=bool(args.third_person_view),
        need_topp=bool(args.need_topp),
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
        attach_planner=(args.control_mode == "eepose"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay H5 joint or EE-pose trajectories for R1 in RoboTwin.")
    parser.add_argument("--input_h5", type=Path, required=True, help="Input h5 file.")
    parser.add_argument("--control_mode", choices=["joint", "eepose"], default="joint")
    parser.add_argument("--data_group", choices=["obs", "action"], default="obs")
    parser.add_argument("--arms", choices=["left", "right", "both"], default="both")
    parser.add_argument("--robot_config", type=Path, default=base.DEFAULT_ROBOT_CONFIG)
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
    parser.add_argument("--need_topp", type=int, default=0)
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
        "left": len(renderer.robot.left_arm_joints) if renderer.robot is not None else 7,
        "right": len(renderer.robot.right_arm_joints) if renderer.robot is not None else 7,
    }

    print(
        "[replay] "
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
        else:
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

        if args.control_mode == "joint":
            print(f"[frame {frame_idx:04d}] replay_step={replay_idx + 1}/{len(frame_indices)} mode=joint")

    renderer.hold_viewer()


if __name__ == "__main__":
    main()
