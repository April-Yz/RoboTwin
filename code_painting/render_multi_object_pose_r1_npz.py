#!/usr/bin/env python3
"""Replay multiple FoundationPose object trajectories in one RoboTwin scene."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import sapien.core as sapien

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import render_hand_retarget_r1_npz as base
from render_object_pose_r1_npz import (
    build_renderer,
    camera_pose_matrix_to_world_pose,
    create_object_actor,
    get_camera_intrinsic_matrix,
    load_pose_sequence,
    pose_wxyz_to_matrix,
    sanitize_depth_mm,
    save_anygrasp_camera_info,
)
from replay_r1_h5 import build_frame_indices


@dataclass
class ObjectReplayData:
    name: str
    pose_dir: Path
    mesh_file: Path
    poses_cam: np.ndarray
    source_frame_indices: np.ndarray
    actor: sapien.Entity
    frame_to_npz_index: Dict[int, int]
    last_pose_world: Optional[np.ndarray] = None
    aligned_pose_world: List[np.ndarray] = field(default_factory=list)
    aligned_visible: List[bool] = field(default_factory=list)


def parse_object_mesh_overrides(specs: Optional[List[str]]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    if not specs:
        return overrides
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --object value '{spec}'. Expected NAME=/path/to/mesh.obj")
        name, mesh_file = spec.split("=", 1)
        name = name.strip()
        mesh_path = Path(mesh_file.strip()).resolve()
        if not name:
            raise ValueError(f"Invalid --object value '{spec}'. Empty object name.")
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Override mesh file does not exist: {mesh_path}")
        overrides[name] = mesh_path
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay multiple FoundationPose object folders into one RoboTwin scene.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Video-level directory containing object subdirs, each with poses.npz.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for replay videos and combined world poses.")
    parser.add_argument("--objects", type=str, nargs="*", default=None, help="Optional subset of object subdir names to replay.")
    parser.add_argument(
        "--object",
        action="append",
        default=None,
        help="Repeatable mesh override in the form NAME=/path/to/mesh.obj. If omitted, mesh_file is read from each object's run_config.json.",
    )
    parser.add_argument("--missing_frame_policy", choices=["hide", "hold_last"], default="hide")
    parser.add_argument("--robot_config", type=Path, default=(PROJECT_ROOT / "robot_config_R1.json"))
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
    parser.add_argument("--head_only", type=int, default=1)
    parser.add_argument("--hide_robot", type=int, default=0, help="If 1, freeze the head camera pose and move robot visuals out of view.")
    parser.add_argument("--overlay_text", type=int, default=0)
    parser.add_argument("--third_person_view", type=int, default=0)
    parser.add_argument("--save_png_frames", type=int, default=0)
    parser.add_argument("--save_head_depth", type=int, default=0, help="If 1, save head camera depth PNGs in uint16 millimeters.")
    parser.add_argument(
        "--save_anygrasp_frames",
        type=int,
        default=0,
        help="If 1, save per-frame head color/depth PNGs and camera intrinsics for AnyGrasp.",
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
    return parser.parse_args()


def load_mesh_from_run_config(pose_dir: Path) -> Path:
    run_config_path = pose_dir / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Missing run_config.json in {pose_dir}")
    with run_config_path.open("r", encoding="utf-8") as f:
        run_config = json.load(f)
    mesh_file = run_config.get("mesh_file")
    if not mesh_file:
        raise KeyError(f"run_config.json in {pose_dir} does not contain mesh_file")
    mesh_path = Path(mesh_file).resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file from run_config does not exist: {mesh_path}")
    return mesh_path


def discover_object_dirs(input_dir: Path, selected_names: Optional[List[str]]) -> List[Path]:
    root = input_dir.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {root}")
    allowed = None if not selected_names else {name.strip() for name in selected_names if name.strip()}
    object_dirs = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "poses.npz").is_file():
            continue
        if allowed is not None and child.name not in allowed:
            continue
        object_dirs.append(child)
    if not object_dirs:
        raise RuntimeError(f"No object pose folders found under {root}")
    return object_dirs


def to_npz_key(name: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_")
    return key or "object"


def set_actor_pose(actor: sapien.Entity, pose_world: Optional[np.ndarray]) -> None:
    if pose_world is None:
        actor.set_pose(base.HIDDEN_DEBUG_POSE)
        return
    actor.set_pose(sapien.Pose(pose_world[:3], base.normalize_quat_wxyz(pose_world[3:])))


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.robot_config = args.robot_config.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    object_mesh_overrides = parse_object_mesh_overrides(args.object)

    if not args.robot_config.is_file():
        raise FileNotFoundError(f"Robot config not found: {args.robot_config}")

    object_dirs = discover_object_dirs(args.input_dir, args.objects)
    renderer = build_renderer(args)

    object_entries: List[ObjectReplayData] = []
    for object_dir in object_dirs:
        poses_cam, source_frame_indices = load_pose_sequence(object_dir / "poses.npz")
        mesh_file = object_mesh_overrides.get(object_dir.name, load_mesh_from_run_config(object_dir))
        actor = create_object_actor(renderer.scene, mesh_file, f"tracked_object_{object_dir.name}")
        frame_to_npz_index = {int(frame_idx): idx for idx, frame_idx in enumerate(source_frame_indices.tolist())}
        object_entries.append(
            ObjectReplayData(
                name=object_dir.name,
                pose_dir=object_dir,
                mesh_file=mesh_file,
                poses_cam=poses_cam,
                source_frame_indices=source_frame_indices,
                actor=actor,
                frame_to_npz_index=frame_to_npz_index,
            )
        )
        actor.set_pose(base.HIDDEN_DEBUG_POSE)

    all_source_frames = sorted(
        {
            int(frame_idx)
            for entry in object_entries
            for frame_idx in entry.source_frame_indices.tolist()
        }
    )
    selected_frame_pos = build_frame_indices(
        length=len(all_source_frames),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    if not selected_frame_pos:
        raise ValueError("No frames selected. Check --frame_start/--frame_end/--frame_stride/--max_frames.")
    selected_source_frames = [all_source_frames[idx] for idx in selected_frame_pos]

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
        head_intrinsic = get_camera_intrinsic_matrix(
            renderer.zed_camera,
            image_width=args.image_width,
            image_height=args.image_height,
            fovy_deg=args.fovy_deg,
        )
        np.savez_compressed(
            head_depth_dir / "camera_info.npz",
            intrinsic_matrix=head_intrinsic.astype(np.float64),
            fx=np.float64(head_intrinsic[0, 0]),
            fy=np.float64(head_intrinsic[1, 1]),
            cx=np.float64(head_intrinsic[0, 2]),
            cy=np.float64(head_intrinsic[1, 2]),
            depth_scale=np.float64(1000.0),
            image_width=np.int32(args.image_width),
            image_height=np.int32(args.image_height),
        )
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

    try:
        for replay_idx, source_frame in enumerate(selected_source_frames):
            visible_names = []
            for entry in object_entries:
                pose_world = None
                npz_idx = entry.frame_to_npz_index.get(int(source_frame))
                if npz_idx is not None:
                    pose_world = camera_pose_matrix_to_world_pose(renderer, entry.poses_cam[npz_idx])
                    entry.last_pose_world = pose_world.copy()
                elif args.missing_frame_policy == "hold_last":
                    pose_world = None if entry.last_pose_world is None else entry.last_pose_world.copy()

                visible = pose_world is not None
                set_actor_pose(entry.actor, pose_world)
                entry.aligned_visible.append(visible)
                if visible:
                    visible_names.append(entry.name)
                    entry.aligned_pose_world.append(np.asarray(pose_world, dtype=np.float64))
                else:
                    entry.aligned_pose_world.append(np.full(7, np.nan, dtype=np.float64))

            renderer.update_robot_link_cameras()
            renderer.step_scene(steps=1)

            overlay_lines = [
                f"source_frame={source_frame}",
                f"replay_step={replay_idx + 1}/{len(selected_source_frames)}",
                f"visible={','.join(visible_names) if visible_names else 'none'}",
            ]

            head_rgb, head_depth_mm = renderer.capture_camera(renderer.zed_camera)
            head_bgr = base.overlay_text(head_rgb, overlay_lines) if use_overlay else cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
            head_writer.write(head_bgr)
            head_depth_uint16 = sanitize_depth_mm(head_depth_mm)

            third_bgr = None
            if third_writer is not None:
                third_rgb, _ = renderer.capture_camera(renderer.third_camera)
                third_bgr = base.overlay_text(third_rgb, overlay_lines) if use_overlay else cv2.cvtColor(third_rgb, cv2.COLOR_RGB2BGR)
                third_writer.write(third_bgr)

            if frames_dir is not None:
                cv2.imwrite(str(frames_dir / f"head_{int(source_frame):06d}.png"), head_bgr)
                if third_bgr is not None:
                    cv2.imwrite(str(frames_dir / f"third_{int(source_frame):06d}.png"), third_bgr)
            if head_depth_dir is not None:
                cv2.imwrite(str(head_depth_dir / f"depth_{int(source_frame):06d}.png"), head_depth_uint16)
            if anygrasp_dir is not None:
                cv2.imwrite(str(anygrasp_dir / f"color_{int(source_frame):06d}.png"), head_bgr)
                cv2.imwrite(str(anygrasp_dir / f"depth_{int(source_frame):06d}.png"), head_depth_uint16)
    finally:
        head_writer.release()
        if third_writer is not None:
            third_writer.release()

    save_kwargs = {
        "input_dir": str(args.input_dir),
        "selected_source_frame_indices": np.asarray(selected_source_frames, dtype=np.int32),
        "object_names": np.asarray([entry.name for entry in object_entries], dtype=object),
    }
    for entry in object_entries:
        key = to_npz_key(entry.name)
        pose_world_arr = np.asarray(entry.aligned_pose_world, dtype=np.float64).reshape(-1, 7)
        pose_world_matrix_arr = np.full((pose_world_arr.shape[0], 4, 4), np.nan, dtype=np.float64)
        valid_mask = np.isfinite(pose_world_arr).all(axis=1)
        for idx, valid in enumerate(valid_mask.tolist()):
            if valid:
                pose_world_matrix_arr[idx] = pose_wxyz_to_matrix(pose_world_arr[idx])
        save_kwargs[f"{key}__mesh_file"] = np.asarray(str(entry.mesh_file), dtype=object)
        save_kwargs[f"{key}__pose_dir"] = np.asarray(str(entry.pose_dir), dtype=object)
        save_kwargs[f"{key}__pose_world_wxyz"] = pose_world_arr
        save_kwargs[f"{key}__pose_world_matrix"] = pose_world_matrix_arr
        save_kwargs[f"{key}__visible"] = np.asarray(entry.aligned_visible, dtype=bool)
    np.savez_compressed(args.output_dir / "multi_object_world_poses.npz", **save_kwargs)

    print(
        "[done] "
        f"objects={[entry.name for entry in object_entries]} "
        f"frames={len(selected_source_frames)} "
        f"head_video={head_video_path} "
        f"third_video={third_video_path if third_enabled else 'disabled'}"
    )
    print(f"[done] combined world poses saved to: {args.output_dir / 'multi_object_world_poses.npz'}")
    renderer.hold_viewer()


if __name__ == "__main__":
    main()
