#!/usr/bin/env python3
"""Visualize Mode O Piper-vs-ALOHA target-frame conventions.

This script is intentionally independent from IK execution. It loads the
FoundationPose object positions for one `pick_diverse_bottles` episode and
draws the same physical side-grasp target under three local-frame conventions:

- `piper_local_z`: current Piper/replay convention, local +Z is approach.
- `aloha_local_x_y_up`: ALOHA-style local +X is approach, local +Y prefers up.
- `aloha_local_x_z_up`: ALOHA-style local +X is approach, local +Z prefers up.

The output PNG and JSON are useful for checking whether the planned gripper
axis convention matches the intended physical side approach before running IK.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plan_first_frame_foundation_pick_diverse_bottles import (
    APPROACH_AXIS_BY_ARM,
    OBJECT_BY_ARM,
    fixed_side_grasp_rotation,
    object_position,
    planner_approach_axis_for_convention,
)


CONVENTIONS = ("piper_local_z", "aloha_local_x_y_up", "aloha_local_x_z_up")
AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")
AXIS_NAMES = ("local +X", "local +Y", "local +Z")


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    return float(math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))))


def draw_frame(ax, origin: np.ndarray, rot_world: np.ndarray, *, label: str, axis_len: float) -> None:
    for axis_idx, (axis_name, color) in enumerate(zip(AXIS_NAMES, AXIS_COLORS)):
        vec = rot_world[:, axis_idx] * axis_len
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            vec[0],
            vec[1],
            vec[2],
            color=color,
            linewidth=2.0,
            arrow_length_ratio=0.18,
        )
        tip = origin + vec * 1.08
        ax.text(tip[0], tip[1], tip[2], f"{label} {axis_name}", fontsize=7, color=color)


def equalize_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float(np.max(maxs - mins)) * 0.62, 0.18)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Mode O gripper frame conventions.")
    parser.add_argument("--task", default="pick_diverse_bottles")
    parser.add_argument("--video_id", type=int, default=0)
    parser.add_argument("--foundation_frame", type=int, default=0)
    parser.add_argument("--replay_dir", type=Path, default=None)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/zaijia001/ssd/RoboTwin/code_painting/mode_o_frame_convention_debug"),
    )
    parser.add_argument("--grasp_surface_retreat_m", type=float, default=0.03)
    parser.add_argument("--axis_len", type=float, default=0.08)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.replay_dir is None:
        args.replay_dir = Path(
            f"/home/zaijia001/ssd/data/piper/hand/{args.task}/foundation_replay_d435/foundation_input_{args.video_id}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    obj_npz_path = args.replay_dir / "multi_object_world_poses.npz"
    if not obj_npz_path.is_file():
        raise FileNotFoundError(f"Object pose NPZ not found: {obj_npz_path}")
    obj_data: Dict[str, np.ndarray] = dict(np.load(str(obj_npz_path), allow_pickle=True))

    fig = plt.figure(figsize=(14, 7), dpi=160)
    axes = {
        "left": fig.add_subplot(1, 2, 1, projection="3d"),
        "right": fig.add_subplot(1, 2, 2, projection="3d"),
    }
    summary = {
        "task": args.task,
        "video_id": int(args.video_id),
        "foundation_frame": int(args.foundation_frame),
        "replay_dir": str(args.replay_dir),
        "grasp_surface_retreat_m": float(args.grasp_surface_retreat_m),
        "conventions": {},
    }
    all_points = []

    for arm, ax in axes.items():
        object_name = OBJECT_BY_ARM[arm]
        obj_pos = object_position(obj_data, object_name, args.foundation_frame)
        approach_axis = APPROACH_AXIS_BY_ARM[arm]
        grasp_pos = obj_pos - float(args.grasp_surface_retreat_m) * approach_axis
        all_points.extend([obj_pos, grasp_pos])

        ax.scatter([obj_pos[0]], [obj_pos[1]], [obj_pos[2]], s=70, color="black", label="Foundation object center")
        ax.scatter([grasp_pos[0]], [grasp_pos[1]], [grasp_pos[2]], s=45, color="tab:orange", label="grasp target")
        ax.plot(
            [obj_pos[0], grasp_pos[0]],
            [obj_pos[1], grasp_pos[1]],
            [obj_pos[2], grasp_pos[2]],
            color="0.35",
            linestyle="--",
            linewidth=1.0,
        )
        ax.quiver(
            grasp_pos[0],
            grasp_pos[1],
            grasp_pos[2],
            approach_axis[0] * args.axis_len,
            approach_axis[1] * args.axis_len,
            approach_axis[2] * args.axis_len,
            color="black",
            linewidth=2.5,
            arrow_length_ratio=0.15,
        )
        ax.text(*(grasp_pos + approach_axis * args.axis_len * 1.12), "physical approach", fontsize=8, color="black")

        arm_summary = {
            "object": object_name,
            "object_pos_xyz": obj_pos.tolist(),
            "grasp_pos_xyz": grasp_pos.tolist(),
            "physical_approach_axis_world": approach_axis.tolist(),
            "frames": {},
        }
        offsets = {
            "piper_local_z": np.array([0.0, 0.0, 0.0]),
            "aloha_local_x_y_up": np.array([0.0, 0.025, 0.0]),
            "aloha_local_x_z_up": np.array([0.0, -0.025, 0.0]),
        }
        for convention in CONVENTIONS:
            rot_world = fixed_side_grasp_rotation(approach_axis, target_frame_convention=convention)
            origin = grasp_pos + offsets[convention]
            draw_frame(ax, origin, rot_world, label=convention.replace("_", "\n"), axis_len=args.axis_len)
            all_points.append(origin)
            arm_summary["frames"][convention] = {
                "planner_approach_axis": planner_approach_axis_for_convention(convention),
                "local_x_world": rot_world[:, 0].tolist(),
                "local_y_world": rot_world[:, 1].tolist(),
                "local_z_world": rot_world[:, 2].tolist(),
                "angle_local_x_to_physical_approach_deg": angle_deg(rot_world[:, 0], approach_axis),
                "angle_local_y_to_physical_approach_deg": angle_deg(rot_world[:, 1], approach_axis),
                "angle_local_z_to_physical_approach_deg": angle_deg(rot_world[:, 2], approach_axis),
            }
        summary["conventions"][arm] = arm_summary

        ax.set_title(f"{arm} arm / {object_name}")
        ax.set_xlabel("world X")
        ax.set_ylabel("world Y")
        ax.set_zlabel("world Z")
        ax.view_init(elev=24, azim=-62)

    equalize_axes(axes["left"], np.asarray(all_points, dtype=np.float64))
    equalize_axes(axes["right"], np.asarray(all_points, dtype=np.float64))
    handles, labels = axes["left"].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(
        "Mode O gripper frame convention check: red=X, green=Y, blue=Z; black arrow=physical side approach",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    stem = f"{args.task}_id{args.video_id}_frame{args.foundation_frame}"
    png_path = args.output_dir / f"{stem}_gripper_frame_conventions.png"
    json_path = args.output_dir / f"{stem}_gripper_frame_conventions.json"
    fig.savefig(png_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[frame-check] wrote image: {png_path}")
    print(f"[frame-check] wrote metrics: {json_path}")
    for arm in ("left", "right"):
        print(f"[{arm}] physical_approach={summary['conventions'][arm]['physical_approach_axis_world']}")
        for convention in CONVENTIONS:
            frame = summary["conventions"][arm]["frames"][convention]
            print(
                f"  {convention}: planner_axis={frame['planner_approach_axis']} "
                f"angle_x={frame['angle_local_x_to_physical_approach_deg']:.1f} "
                f"angle_y={frame['angle_local_y_to_physical_approach_deg']:.1f} "
                f"angle_z={frame['angle_local_z_to_physical_approach_deg']:.1f}"
            )


if __name__ == "__main__":
    main()
