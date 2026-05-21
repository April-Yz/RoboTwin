#!/usr/bin/env python3
"""Plot world-axis distances from HaMeR gripper/wrist-retreat points to objects."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import render_hand_retarget_r1_npz as base
from render_hand_retarget_piper_dual_npz_urdfik import HandRetargetPiperDualURDFIKRenderer
from render_object_pose_r1_npz import camera_pose_matrix_to_world_pose, load_pose_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand_npz", type=Path, required=True)
    parser.add_argument("--object_dir", type=Path, required=True, help="FoundationPose video-level dir containing pear/star_fruit subdirs.")
    parser.add_argument("--output_png", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=-1)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--robot_config", type=Path, default=PROJECT_ROOT / "robot_config_PiperPika_agx_dual_table_0515.json")
    parser.add_argument("--camera_cv_axis_mode", choices=sorted(base.CV_TO_WORLD_CAMERA_PRESETS.keys()), default="legacy_r1")
    parser.add_argument("--head_camera_local_pos", type=float, nargs=3, default=[0.11210396690038413, -0.39189397826604927, 0.4753892624100325])
    parser.add_argument("--head_camera_local_quat_wxyz", type=float, nargs=4, default=[0.8524694864910365, -0.0011011947849308937, 0.5226654778798345, 0.010740586780925399])
    parser.add_argument("--target_world_offset_xyz", type=float, nargs=3, default=[0.0, 0.1, 0.1])
    parser.add_argument("--retreat_distance", type=float, default=0.11)
    parser.add_argument("--left_object", type=str, default="pear")
    parser.add_argument("--right_object", type=str, default="star_fruit")
    parser.add_argument("--absolute", type=int, default=0, help="If 1, plot absolute axis distances instead of signed deltas.")
    parser.add_argument(
        "--plot_clip_abs_m",
        type=float,
        default=0.5,
        help="Clip plotted values to +/- this many meters while keeping CSV values raw. Use <=0 to disable plot clipping.",
    )
    return parser.parse_args()


def build_renderer(args: argparse.Namespace) -> HandRetargetPiperDualURDFIKRenderer:
    return HandRetargetPiperDualURDFIKRenderer(
        robot_config_path=args.robot_config,
        image_width=640,
        image_height=360,
        fovy_deg=90.0,
        torso_qpos=base.DEFAULT_TORSO_QPOS.tolist(),
        robot_base_pose_override=None,
        third_person_view=False,
        need_topp=False,
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
        enable_viewer=False,
        viewer_frame_delay=0.0,
        viewer_wait_at_end=False,
        debug_mode=False,
        debug_force_orientation="none",
        debug_visualize_targets=False,
        debug_target_axis_length=0.08,
        debug_target_axis_thickness=0.004,
        debug_visualize_cameras=False,
        debug_camera_axis_length=0.16,
        debug_camera_axis_thickness=0.006,
        orientation_remap_label="identity",
        orientation_remap_matrix=np.eye(3, dtype=np.float64),
        stored_orientation_post_rot_xyz_deg=[0.0, 0.0, 0.0],
        target_local_forward_retreat_m=0.0,
        target_world_offset_xyz=args.target_world_offset_xyz,
        left_target_world_offset_xyz=[0.0, 0.0, 0.0],
        right_target_world_offset_xyz=[0.0, 0.0, 0.0],
        target_world_z_offset=0.0,
        disable_table=True,
        base_occluder_enable=False,
        base_occluder_local_pos=[0.0, 0.0, 0.4],
        base_occluder_half_size=[0.28, 0.32, 0.02],
        base_occluder_color=[1.0, 1.0, 1.0],
        camera_sweep_enable=False,
        camera_sweep_steps_deg=[0.0],
        init_left_arm_joints=None,
        init_right_arm_joints=None,
        init_gripper_open=None,
        execute_waypoint_scene_steps=5,
        execute_settle_scene_steps=20,
        urdfik_joint_interp_waypoints=10,
        lighting_mode="front_no_shadow",
    )


def frame_indices(length: int, args: argparse.Namespace) -> List[int]:
    end = length if args.frame_end < 0 else min(length, args.frame_end + 1)
    indices = list(range(max(args.frame_start, 0), end, max(args.frame_stride, 1)))
    if args.max_frames > 0:
        indices = indices[: args.max_frames]
    if not indices:
        raise ValueError("No frames selected.")
    return indices


def load_object_track(object_dir: Path, object_name: str) -> Tuple[Optional[np.ndarray], Dict[int, int]]:
    npz_path = object_dir / object_name / "poses.npz"
    if not npz_path.is_file():
        print(f"[warn] missing object pose track: {npz_path}")
        return None, {}
    poses, source_frames = load_pose_sequence(npz_path)
    return poses, {int(frame_idx): idx for idx, frame_idx in enumerate(source_frames.tolist())}


def target_world_pos(renderer, trajectory, side: str, frame_idx: int, pose_source: str) -> Optional[np.ndarray]:
    target = trajectory.get_side_target(side, frame_idx, pose_source)
    if not target.valid:
        return None
    rot_cam = renderer.remap_target_rotation(target.rotation_cam)
    pose_world = renderer.camera_to_world_pose(target.position_cam, rot_cam)
    pose_world = renderer.apply_target_world_offset(side, pose_world)
    return pose_world[:3].copy()


def object_world_pos(renderer, poses: Optional[np.ndarray], frame_to_idx: Dict[int, int], frame_idx: int) -> Optional[np.ndarray]:
    if poses is None:
        return None
    npz_idx = frame_to_idx.get(int(frame_idx))
    if npz_idx is None:
        return None
    pose_world = camera_pose_matrix_to_world_pose(renderer, np.asarray(poses[npz_idx], dtype=np.float64).reshape(4, 4))
    return pose_world[:3].copy()


def values_for_plot(values: List[float], clip_abs_m: float) -> Tuple[np.ndarray, int]:
    arr = np.asarray(values, dtype=np.float64)
    if clip_abs_m <= 0.0:
        return arr, 0
    finite = np.isfinite(arr)
    clipped_count = int(np.count_nonzero(finite & (np.abs(arr) > clip_abs_m)))
    return np.clip(arr, -clip_abs_m, clip_abs_m), clipped_count


def main() -> None:
    args = parse_args()
    args.output_png = args.output_png.resolve()
    if args.output_csv is None:
        args.output_csv = args.output_png.with_suffix(".csv")
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    trajectory = base.HandRetargetTrajectory(
        npz_path=args.hand_npz,
        retreat_distance=args.retreat_distance,
        thumb_tip_idx=4,
        index_tip_idx=8,
        index_joint_idx=6,
    )
    renderer = build_renderer(args)
    renderer.update_robot_link_cameras()
    renderer.step_scene(steps=1)

    tracks = {
        "left": (args.left_object, *load_object_track(args.object_dir, args.left_object)),
        "right": (args.right_object, *load_object_track(args.object_dir, args.right_object)),
    }
    indices = frame_indices(trajectory.length, args)

    rows: List[Dict[str, float]] = []
    series: Dict[str, Dict[str, List[float]]] = {"left": {}, "right": {}}
    for side in ("left", "right"):
        for key in ("gripper_dx", "gripper_dy", "gripper_dz", "wrist_dx", "wrist_dy", "wrist_dz"):
            series[side][key] = []

    for frame_idx in indices:
        row: Dict[str, float] = {"frame": float(frame_idx)}
        for side in ("left", "right"):
            object_name, poses, frame_to_idx = tracks[side]
            obj = object_world_pos(renderer, poses, frame_to_idx, frame_idx)
            grip = target_world_pos(renderer, trajectory, side, frame_idx, "gripper")
            wrist = target_world_pos(renderer, trajectory, side, frame_idx, "retreat")
            for prefix, point in (("gripper", grip), ("wrist", wrist)):
                if obj is None or point is None:
                    delta = np.full(3, np.nan, dtype=np.float64)
                else:
                    delta = point - obj
                    if bool(args.absolute):
                        delta = np.abs(delta)
                for axis, value in zip(("x", "y", "z"), delta.tolist()):
                    key = f"{prefix}_d{axis}"
                    series[side][key].append(float(value))
                    row[f"{side}_{object_name}_{key}_m"] = float(value)
        rows.append(row)

    with args.output_csv.open("w", encoding="utf-8") as f:
        headers = ["frame"]
        for side in ("left", "right"):
            object_name = tracks[side][0]
            for prefix in ("gripper", "wrist"):
                for axis in ("x", "y", "z"):
                    headers.append(f"{side}_{object_name}_{prefix}_d{axis}_m")
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join("" if not np.isfinite(row.get(h, np.nan)) else f"{row[h]:.8f}" for h in headers) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    colors = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}
    x_values = np.asarray(indices, dtype=np.int32)
    clip_abs_m = float(args.plot_clip_abs_m)
    for ax, side in zip(axes, ("left", "right")):
        object_name = tracks[side][0]
        clipped_total = 0
        for axis in ("x", "y", "z"):
            gripper_values, gripper_clipped = values_for_plot(series[side][f"gripper_d{axis}"], clip_abs_m)
            wrist_values, wrist_clipped = values_for_plot(series[side][f"wrist_d{axis}"], clip_abs_m)
            clipped_total += gripper_clipped + wrist_clipped
            ax.plot(x_values, gripper_values, color=colors[axis], linestyle="-", label=f"gripper d{axis}")
            ax.plot(x_values, wrist_values, color=colors[axis], linestyle="--", label=f"wrist-retreat d{axis}")
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        if clip_abs_m > 0.0:
            ax.axhline(clip_abs_m, color="black", linewidth=0.7, alpha=0.25, linestyle=":")
            ax.axhline(-clip_abs_m, color="black", linewidth=0.7, alpha=0.25, linestyle=":")
            ax.set_ylim(-clip_abs_m * 1.08, clip_abs_m * 1.08)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("axis distance (m)")
        clip_note = "" if clip_abs_m <= 0.0 else f", plot clipped to +/-{clip_abs_m:.2f}m ({clipped_total} values)"
        ax.set_title(f"{side} hand vs {object_name}: {'absolute' if bool(args.absolute) else 'signed'} world-axis distance{clip_note}")
        ax.legend(loc="upper right", ncol=3, fontsize=9)
    axes[-1].set_xlabel("frame index")
    title = "HaMeR gripper / wrist-retreat point to FoundationPose object distance by world axis"
    if clip_abs_m > 0.0:
        title += " (CSV raw, plot clipped)"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(args.output_png, dpi=160)
    print(f"[done] png={args.output_png}")
    print(f"[done] csv={args.output_csv}")


if __name__ == "__main__":
    main()
