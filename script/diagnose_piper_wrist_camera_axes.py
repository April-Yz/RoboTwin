"""Diagnose Piper/Pika wrist camera axes against gripper-frame conventions.

This script is intentionally simulation-free.  It uses the calibration bundle and
the same axis conversion as envs/camera/camera.py to report whether each wrist
camera forward axis lies in the plane perpendicular to the finger-opening axis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


AXIS_ROTATIONS = {
    "legacy_r1": np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    ),
    "diag_flip_yz": np.diag([1.0, -1.0, -1.0]).astype(np.float64),
}


def unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError(f"cannot normalize vector {vec!r}")
    return vec / norm


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = unit(a)
    b = unit(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def signed_angle_about_axis(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    a = unit(a)
    b = unit(b)
    axis = unit(axis)
    a_proj = unit(a - axis * np.dot(a, axis))
    b_proj = unit(b - axis * np.dot(b, axis))
    return math.degrees(
        math.atan2(np.dot(axis, np.cross(a_proj, b_proj)), np.dot(a_proj, b_proj))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        default="calibration_bundle_piper_new_table_0515.json",
        help="Piper calibration bundle JSON path.",
    )
    parser.add_argument("--adapter", default="piper_pika_agx")
    parser.add_argument("--axis_mode", default="legacy_r1", choices=sorted(AXIS_ROTATIONS))
    args = parser.parse_args()

    with Path(args.bundle).open("r", encoding="utf-8") as file:
        bundle = json.load(file)
    wrist_data = bundle["wrist_cameras"]
    adapter = np.asarray(
        wrist_data["simulation_adapters"][args.adapter]["matrix"], dtype=np.float64
    )
    cv_to_render = AXIS_ROTATIONS[args.axis_mode]

    physical_forward_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    debug_forward_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    finger_open_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    print(f"bundle={Path(args.bundle).resolve()}")
    print(f"adapter={args.adapter} axis_mode={args.axis_mode}")
    print("conventions:")
    print("  SAPIEN camera forward = render local +X")
    print("  Pika CAD / Robot TCP physical forward = gripper local +X")
    print("  legacy debug blue axis forward label = gripper local +Z")
    print("  finger opening axis = gripper local Y")

    for side, key in (
        ("left", "left_gripper_T_camera"),
        ("right", "right_gripper_T_camera"),
    ):
        matrix = np.asarray(wrist_data[key]["matrix"], dtype=np.float64)
        render_rotation = (adapter @ matrix)[:3, :3] @ cv_to_render.T
        forward = unit(render_rotation[:, 0])
        left = unit(render_rotation[:, 1])
        up = unit(render_rotation[:, 2])

        projected = unit(forward - finger_open_y * np.dot(forward, finger_open_y))
        plane_error = math.degrees(
            math.asin(float(np.clip(np.dot(forward, finger_open_y), -1.0, 1.0)))
        )
        rotate_about_debug_z = math.degrees(math.atan2(-forward[1], forward[0]))
        pitch_projected_to_debug_z = signed_angle_about_axis(
            projected, debug_forward_z, finger_open_y
        )

        print(f"\n[{side}]")
        print(f"  camera_forward(+X render) = {np.round(forward, 6).tolist()}")
        print(f"  camera_left(+Y render)    = {np.round(left, 6).tolist()}")
        print(f"  camera_up(+Z render)      = {np.round(up, 6).tolist()}")
        print(
            "  opening-plane error: "
            f"dot(forward, gripper_Y)={np.dot(forward, finger_open_y):.6f}, "
            f"signed={plane_error:.3f} deg"
        )
        print(
            "  angle to physical gripper +X = "
            f"{angle_deg(forward, physical_forward_x):.3f} deg"
        )
        print(
            "  angle to legacy debug +Z     = "
            f"{angle_deg(forward, debug_forward_z):.3f} deg"
        )
        print(
            "  tiny plane-only yaw about gripper +Z to zero Y = "
            f"{rotate_about_debug_z:.3f} deg"
        )
        print(
            "  if forcing forward to legacy +Z, pitch about gripper Y = "
            f"{pitch_projected_to_debug_z:.3f} deg (not recommended unless frame convention changes)"
        )


if __name__ == "__main__":
    main()
