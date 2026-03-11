#!/usr/bin/env python3
"""Replay FoundationPose object poses in RoboTwin R1 and record camera videos."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import render_hand_retarget_r1_npz as base
from replay_r1_h5 import ReplayRenderer, build_frame_indices, parse_optional_base_pose


R1_CONFIG = PROJECT_ROOT / "robot_config_R1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay FoundationPose poses.npz object trajectory and record head/third camera videos in RoboTwin R1."
    )
    parser.add_argument("--input_npz", type=Path, required=True, help="FoundationPose poses.npz path.")
    parser.add_argument("--mesh_file", type=Path, required=True, help="Object mesh file (e.g., bottle.obj).")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for videos and converted poses.")
    parser.add_argument("--robot_config", type=Path, default=R1_CONFIG)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--fovy_deg", type=float, default=90.0)
    parser.add_argument("--torso_qpos", type=float, nargs=4, default=base.DEFAULT_TORSO_QPOS.tolist())
    parser.add_argument("--robot_base_pose", type=float, nargs=7, default=None, metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"))
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=-1)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--head_only", type=int, default=0, help="If 1, only save head_cam video and disable third_cam output.")
    parser.add_argument("--overlay_text", type=int, default=1, help="If 1, draw overlay text on replay frames.")
    parser.add_argument("--third_person_view", type=int, default=1)
    parser.add_argument("--save_png_frames", type=int, default=0)
    parser.add_argument("--save_head_depth", type=int, default=0, help="If 1, save head camera depth PNGs in uint16 millimeters.")
    parser.add_argument(
        "--save_anygrasp_frames",
        type=int,
        default=0,
        help="If 1, save per-frame color/depth PNGs and camera intrinsics for AnyGrasp.",
    )
    parser.add_argument("--enable_viewer", type=int, default=0)
    parser.add_argument("--viewer_frame_delay", type=float, default=0.0)
    parser.add_argument("--viewer_wait_at_end", type=int, default=0)
    parser.add_argument("--disable_table", type=int, default=1)
    parser.add_argument("--need_topp", type=int, default=0)
    parser.add_argument("--lighting_mode", choices=["default", "front", "front_no_shadow"], default="default")
    parser.add_argument("--camera_cv_axis_mode", choices=sorted(base.CV_TO_WORLD_CAMERA_PRESETS.keys()), default="legacy_r1")
    parser.add_argument("--head_camera_local_pos", type=float, nargs=3, default=base.DEFAULT_HEAD_CAMERA_LOCAL_POS.tolist())
    parser.add_argument("--head_camera_local_quat_wxyz", type=float, nargs=4, default=base.DEFAULT_HEAD_CAMERA_LOCAL_QUAT_WXYZ.tolist())
    parser.add_argument("--object_actor_name", type=str, default="tracked_object")
    return parser.parse_args()


def load_pose_sequence(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(str(npz_path), allow_pickle=True)
    if "poses" not in data.files:
        raise KeyError(f"Missing 'poses' in NPZ: {npz_path}")

    poses = np.asarray(data["poses"], dtype=np.float64)
    if poses.ndim == 2 and poses.shape == (4, 4):
        poses = poses[None, ...]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"'poses' must have shape [N, 4, 4], got {poses.shape}")

    if "frame_indices" in data.files:
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int32).reshape(-1)
        if frame_indices.shape[0] != poses.shape[0]:
            raise ValueError(
                f"'frame_indices' length {frame_indices.shape[0]} does not match poses length {poses.shape[0]}"
            )
    else:
        frame_indices = np.arange(poses.shape[0], dtype=np.int32)
    return poses, frame_indices


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
        camera_cv_axis_mode=args.camera_cv_axis_mode,
        head_camera_local_pos=args.head_camera_local_pos,
        head_camera_local_quat_wxyz=args.head_camera_local_quat_wxyz,
        wrist_camera_local_pos=base.DEFAULT_WRIST_CAMERA_LOCAL_POS,
        wrist_camera_local_quat_wxyz=base.DEFAULT_WRIST_CAMERA_LOCAL_QUAT_WXYZ,
        camera_debug_target="head",
        enable_viewer=bool(args.enable_viewer),
        viewer_frame_delay=args.viewer_frame_delay,
        viewer_wait_at_end=bool(args.viewer_wait_at_end),
        debug_mode=False,
        debug_force_orientation="none",
        debug_visualize_targets=False,
        debug_target_axis_length=0.08,
        debug_target_axis_thickness=0.004,
        orientation_remap_label="identity",
        orientation_remap_matrix=np.eye(3, dtype=np.float64),
        stored_orientation_post_rot_xyz_deg=[0.0, 0.0, 0.0],
        target_world_offset_xyz=[0.0, 0.0, 0.0],
        left_target_world_offset_xyz=[0.0, 0.0, 0.0],
        right_target_world_offset_xyz=[0.0, 0.0, 0.0],
        target_world_z_offset=0.0,
        disable_table=bool(args.disable_table),
        camera_sweep_enable=False,
        camera_sweep_steps_deg=[0.0],
        init_left_arm_joints=None,
        init_right_arm_joints=None,
        init_gripper_open=None,
        lighting_mode=args.lighting_mode,
        attach_planner=False,
    )


def create_object_actor(scene: sapien.Scene, mesh_file: Path, actor_name: str):
    builder = scene.create_actor_builder()
    try:
        builder.add_visual_from_file(str(mesh_file))
    except Exception as exc:
        raise RuntimeError(f"Failed to load mesh visual: {mesh_file}") from exc
    try:
        builder.add_convex_collision_from_file(str(mesh_file))
    except Exception as exc:
        print(f"[mesh-warning] failed to add convex collision for {mesh_file}: {exc}")
    return builder.build_kinematic(name=actor_name)


def camera_pose_matrix_to_world_pose(renderer: ReplayRenderer, pose_cam_matrix: np.ndarray) -> np.ndarray:
    pose_cam_matrix = np.asarray(pose_cam_matrix, dtype=np.float64).reshape(4, 4)
    position_cam = pose_cam_matrix[:3, 3]
    rotation_cam = base.orthonormalize_rotation(pose_cam_matrix[:3, :3])
    return renderer.camera_to_world_pose(position_cam, rotation_cam)


def pose_wxyz_to_matrix(pose_wxyz: np.ndarray) -> np.ndarray:
    pose_wxyz = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    quat_wxyz = base.normalize_quat_wxyz(pose_wxyz[3:])
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_quat(base.quat_wxyz_to_xyzw(quat_wxyz)).as_matrix()
    mat[:3, 3] = pose_wxyz[:3]
    return mat


def sanitize_depth_mm(depth_mm: np.ndarray) -> np.ndarray:
    depth_safe = np.nan_to_num(np.asarray(depth_mm, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    depth_safe = np.clip(depth_safe, 0.0, float(np.iinfo(np.uint16).max))
    return np.rint(depth_safe).astype(np.uint16)


def get_camera_intrinsic_matrix(camera, image_width: int, image_height: int, fovy_deg: float) -> np.ndarray:
    try:
        intrinsic = np.asarray(camera.get_intrinsic_matrix(), dtype=np.float64)
        if intrinsic.shape == (4, 4):
            intrinsic = intrinsic[:3, :3]
        if intrinsic.shape == (3, 3) and np.isfinite(intrinsic).all():
            return intrinsic
    except Exception:
        pass

    fovy_rad = np.deg2rad(float(fovy_deg))
    fy = float(image_height) / (2.0 * np.tan(fovy_rad / 2.0))
    fx = fy
    cx = float(image_width) / 2.0
    cy = float(image_height) / 2.0
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def save_anygrasp_camera_info(output_dir: Path, camera, image_width: int, image_height: int, fovy_deg: float) -> None:
    intrinsic = get_camera_intrinsic_matrix(camera, image_width=image_width, image_height=image_height, fovy_deg=fovy_deg)
    info = {
        "format": "anygrasp_rgbd_sequence_v1",
        "camera_name": "head_cam",
        "image_width": int(image_width),
        "image_height": int(image_height),
        "depth_unit": "millimeter",
        "depth_scale": 1000.0,
        "intrinsic_matrix": intrinsic.tolist(),
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "color_pattern": "color_{source_frame:06d}.png",
        "depth_pattern": "depth_{source_frame:06d}.png",
    }
    with (output_dir / "camera_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def main() -> None:
    args = parse_args()
    args.input_npz = args.input_npz.resolve()
    args.mesh_file = args.mesh_file.resolve()
    args.output_dir = args.output_dir.resolve()
    args.robot_config = args.robot_config.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_npz.is_file():
        raise FileNotFoundError(f"Input NPZ not found: {args.input_npz}")
    if not args.mesh_file.is_file():
        raise FileNotFoundError(f"Mesh file not found: {args.mesh_file}")
    if not args.robot_config.is_file():
        raise FileNotFoundError(f"Robot config not found: {args.robot_config}")

    poses_cam, source_frame_indices = load_pose_sequence(args.input_npz)
    selected_npz_indices = build_frame_indices(
        length=poses_cam.shape[0],
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    if not selected_npz_indices:
        raise ValueError("No frames selected. Check --frame_start/--frame_end/--frame_stride/--max_frames.")

    renderer = build_renderer(args)
    object_actor = create_object_actor(renderer.scene, args.mesh_file, args.object_actor_name)
    renderer.update_robot_link_cameras()
    renderer.step_scene(steps=1)

    third_enabled = bool(args.third_person_view) and not bool(args.head_only)
    use_overlay = bool(args.overlay_text)

    head_video_path = args.output_dir / "head_cam_replay.mp4"
    third_video_path = args.output_dir / "third_cam_replay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    head_writer = cv2.VideoWriter(str(head_video_path), fourcc, args.fps, (args.image_width, args.image_height))
    if not head_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {head_video_path}")
    third_writer = None
    if third_enabled:
        third_writer = cv2.VideoWriter(str(third_video_path), fourcc, args.fps, (args.image_width, args.image_height))
        if not third_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {third_video_path}")

    frames_dir = args.output_dir / "frames" if bool(args.save_png_frames) else None
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
    head_depth_dir = args.output_dir / "head_depth_frames" if bool(args.save_head_depth) else None
    if head_depth_dir is not None:
        head_depth_dir.mkdir(parents=True, exist_ok=True)
    anygrasp_dir = args.output_dir / "head_anygrasp_frames" if bool(args.save_anygrasp_frames) else None
    if anygrasp_dir is not None:
        anygrasp_dir.mkdir(parents=True, exist_ok=True)
        save_anygrasp_camera_info(
            anygrasp_dir,
            renderer.zed_camera,
            image_width=args.image_width,
            image_height=args.image_height,
            fovy_deg=args.fovy_deg,
        )

    world_pose_list = []
    used_source_frames = []
    try:
        for replay_idx, npz_idx in enumerate(selected_npz_indices):
            pose_world = camera_pose_matrix_to_world_pose(renderer, poses_cam[npz_idx])
            object_actor.set_pose(sapien.Pose(pose_world[:3], base.normalize_quat_wxyz(pose_world[3:])))

            renderer.update_robot_link_cameras()
            renderer.step_scene(steps=1)

            source_frame = int(source_frame_indices[npz_idx])
            overlay_lines = [
                f"npz_idx={npz_idx}",
                f"source_frame={source_frame}",
                f"replay_step={replay_idx + 1}/{len(selected_npz_indices)}",
                f"obj_xyz={np.round(pose_world[:3], 4).tolist()}",
            ]

            head_rgb, head_depth_mm = renderer.capture_camera(renderer.zed_camera)
            if use_overlay:
                head_bgr = base.overlay_text(head_rgb, overlay_lines)
            else:
                head_bgr = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
            head_writer.write(head_bgr)
            head_depth_uint16 = sanitize_depth_mm(head_depth_mm)

            third_bgr = None
            if third_writer is not None:
                third_rgb, _ = renderer.capture_camera(renderer.third_camera)
                if use_overlay:
                    third_bgr = base.overlay_text(third_rgb, overlay_lines)
                else:
                    third_bgr = cv2.cvtColor(third_rgb, cv2.COLOR_RGB2BGR)
                third_writer.write(third_bgr)

            if frames_dir is not None:
                cv2.imwrite(str(frames_dir / f"head_{source_frame:06d}.png"), head_bgr)
                if third_bgr is not None:
                    cv2.imwrite(str(frames_dir / f"third_{source_frame:06d}.png"), third_bgr)
            if head_depth_dir is not None:
                cv2.imwrite(str(head_depth_dir / f"depth_{source_frame:06d}.png"), head_depth_uint16)
            if anygrasp_dir is not None:
                cv2.imwrite(str(anygrasp_dir / f"color_{source_frame:06d}.png"), head_bgr)
                cv2.imwrite(str(anygrasp_dir / f"depth_{source_frame:06d}.png"), head_depth_uint16)

            world_pose_list.append(pose_world.astype(np.float64))
            used_source_frames.append(source_frame)
    finally:
        head_writer.release()
        if third_writer is not None:
            third_writer.release()

    world_pose_array = np.asarray(world_pose_list, dtype=np.float64).reshape(-1, 7)
    world_pose_matrix = np.stack([pose_wxyz_to_matrix(p) for p in world_pose_array], axis=0)
    np.savez_compressed(
        args.output_dir / "object_world_poses.npz",
        source_npz=str(args.input_npz),
        mesh_file=str(args.mesh_file),
        robot_config=str(args.robot_config),
        selected_npz_indices=np.asarray(selected_npz_indices, dtype=np.int32),
        source_frame_indices=np.asarray(used_source_frames, dtype=np.int32),
        pose_world_wxyz=world_pose_array,
        pose_world_matrix=world_pose_matrix,
    )

    print(
        "[done] "
        f"frames={len(selected_npz_indices)} "
        f"head_video={head_video_path} "
        f"third_video={third_video_path if third_enabled else 'disabled'} "
        f"overlay_text={int(use_overlay)}"
    )
    print(f"[done] world poses saved to: {args.output_dir / 'object_world_poses.npz'}")
    renderer.hold_viewer()


if __name__ == "__main__":
    main()
