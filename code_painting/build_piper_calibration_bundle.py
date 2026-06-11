#!/usr/bin/env python3
"""Build a self-contained Piper calibration bundle from hand-eye JSON files."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


DEFAULT_LEFT_BASE_WORLD_POSE = [-0.3, -0.25, 0.75, 0.70710678, 0.0, 0.0, 0.70710678]
LEGACY_R1_CV_TO_RENDER_CAMERA = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)
# Simulation-only stand clearance; the measured hand-eye rotations remain unchanged.
PIPER_PIKA_AGX_URDF_LINK_T_REAL_TCP = np.array(
    [
        [1.0, 0.0, 0.0, 0.075],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.05],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _as_vec3(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        return np.array([value["x"], value["y"], value["z"]], dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(3)


def _as_quat_wxyz(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        return np.array([value["w"], value["x"], value["y"], value["z"]], dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(4)
    return np.array([arr[3], arr[0], arr[1], arr[2]], dtype=np.float64)


def _quat_wxyz_to_matrix(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                0.25 * scale,
                (rot[2, 1] - rot[1, 2]) / scale,
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[1, 0] - rot[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        scale = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        q = np.array(
            [
                (rot[2, 1] - rot[1, 2]) / scale,
                0.25 * scale,
                (rot[0, 1] + rot[1, 0]) / scale,
                (rot[0, 2] + rot[2, 0]) / scale,
            ],
            dtype=np.float64,
        )
    elif rot[1, 1] > rot[2, 2]:
        scale = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        q = np.array(
            [
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[0, 1] + rot[1, 0]) / scale,
                0.25 * scale,
                (rot[1, 2] + rot[2, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        scale = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        q = np.array(
            [
                (rot[1, 0] - rot[0, 1]) / scale,
                (rot[0, 2] + rot[2, 0]) / scale,
                (rot[1, 2] + rot[2, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    return q / np.linalg.norm(q)


def _pose_to_matrix(pose: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(pose), dtype=np.float64).reshape(7)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = _quat_wxyz_to_matrix(values[3:])
    mat[:3, 3] = values[:3]
    return mat


def _matrix_to_pose(mat: np.ndarray) -> List[float]:
    q = _matrix_to_quat_wxyz(mat[:3, :3])
    return [float(x) for x in [*mat[:3, 3].tolist(), *q.tolist()]]


def _transform_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    mat = np.asarray(entry["matrix"], dtype=np.float64)
    t = _as_vec3(entry["translation"])
    q = _as_quat_wxyz(entry["rotation_quaternion_xyzw"])
    return {
        "translation": [float(x) for x in t.tolist()],
        "quat_wxyz": [float(x) for x in q.tolist()],
        "matrix": [[float(x) for x in row] for row in mat.tolist()],
    }


def _transform_entry_from_matrix(mat: np.ndarray) -> Dict[str, Any]:
    mat = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    return {
        "translation": [float(x) for x in mat[:3, 3].tolist()],
        "quat_wxyz": [float(x) for x in _matrix_to_quat_wxyz(mat[:3, :3]).tolist()],
        "matrix": [[float(x) for x in row] for row in mat.tolist()],
    }


def _set_robot_poses(robot_config: Dict[str, Any], left_pose: List[float], right_pose: List[float]) -> Dict[str, Any]:
    cfg = deepcopy(robot_config)
    cfg["embodiment_dis"] = 0.0
    for key in ("left_embodiment_config", "right_embodiment_config"):
        if key in cfg:
            cfg[key]["robot_pose"] = [left_pose, right_pose]
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose Piper hand-eye calibration files into one replay bundle.")
    parser.add_argument("--head", type=Path, required=True, help="head_d435_*_head_from_wrist.json")
    parser.add_argument("--base", type=Path, required=True, help="left_base_T_right_base*.json")
    parser.add_argument("--left_wrist", type=Path, required=True, help="left_wrist_*_eye_in_hand.json")
    parser.add_argument("--right_wrist", type=Path, required=True, help="right_wrist_*_eye_in_hand.json")
    parser.add_argument("--template_robot_config", type=Path, required=True, help="Robot config used as template.")
    parser.add_argument("--output", type=Path, required=True, help="Output bundle JSON.")
    parser.add_argument("--left_base_world_pose", type=float, nargs=7, default=DEFAULT_LEFT_BASE_WORLD_POSE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    head = json.loads(args.head.read_text(encoding="utf-8"))
    base = json.loads(args.base.read_text(encoding="utf-8"))
    left_wrist = json.loads(args.left_wrist.read_text(encoding="utf-8"))
    right_wrist = json.loads(args.right_wrist.read_text(encoding="utf-8"))
    template_robot_config = json.loads(args.template_robot_config.read_text(encoding="utf-8"))

    left_base_pose = [float(x) for x in args.left_base_world_pose]
    left_base_world = _pose_to_matrix(left_base_pose)
    left_base_t_right_base = _transform_entry(base["left_base_T_right_base"])
    right_base_world = left_base_world @ np.asarray(left_base_t_right_base["matrix"], dtype=np.float64)
    right_base_pose = _matrix_to_pose(right_base_world)

    head_raw_optical = _transform_entry(head["left_base_T_head_camera"])
    head_render_matrix = np.asarray(head_raw_optical["matrix"], dtype=np.float64).copy()
    # Existing replay code uses camera_cv_axis_mode=legacy_r1 and expects:
    #   T_world_render_camera @ legacy_r1 == T_world_raw_optical_camera
    # so the render/SAPIEN camera pose must be raw_optical @ legacy_r1.T.
    head_render_matrix[:3, :3] = head_render_matrix[:3, :3] @ LEGACY_R1_CV_TO_RENDER_CAMERA.T
    head_local = _transform_entry_from_matrix(head_render_matrix)
    left_wrist_local = _transform_entry(left_wrist["gripper_T_camera"])
    right_wrist_local = _transform_entry(right_wrist["gripper_T_camera"])
    robot_config = _set_robot_poses(template_robot_config, left_base_pose, right_base_pose)
    head_world = left_base_world @ np.asarray(head_local["matrix"], dtype=np.float64)

    bundle = {
        "schema": "piper_calibration_bundle.v1",
        "name": "piper_new_table_0515",
        "source_files": {
            "head": str(args.head),
            "base": str(args.base),
            "left_wrist": str(args.left_wrist),
            "right_wrist": str(args.right_wrist),
            "template_robot_config": str(args.template_robot_config),
        },
        "robot_config": robot_config,
        "left_base_world_pose": left_base_pose,
        "right_base_world_pose": right_base_pose,
        "head_camera": {
            "axis_conversion": "render_camera = raw_optical @ legacy_r1.T",
            "raw_left_base_T_head_camera": head_raw_optical,
            "left_base_T_head_camera": head_local,
            "world_pose": _matrix_to_pose(head_world),
        },
        "wrist_cameras": {
            "parent_frame": "urdf_end_pose_orient_tcp",
            "camera_frame": "opencv_color_optical",
            "axis_conversion": "render_camera = raw_optical @ legacy_r1.T",
            "left_gripper_T_camera": left_wrist_local,
            "right_gripper_T_camera": right_wrist_local,
            "simulation_adapters": {
                "piper_pika_agx": {
                    "from_frame": "real_urdf_end_pose_orient_tcp",
                    "to_frame": "simulation_link6",
                    "matrix": PIPER_PIKA_AGX_URDF_LINK_T_REAL_TCP.tolist(),
                    "note": (
                        "Keeps the calibrated real TCP axes aligned with the simulation "
                        "link6 task frame. The translation only converts the real TCP "
                        "origin to a collision-free virtual camera stand above the Pika "
                        "gripper shell; calibrated left/right lateral signs, optical "
                        "axis, and camera roll are preserved."
                    ),
                }
            },
        },
        "derived_summary": {
            "base_distance_m": float(np.linalg.norm(right_base_world[:3, 3] - left_base_world[:3, 3])),
            "head_distance_from_left_base_m": float(np.linalg.norm(head_world[:3, 3] - left_base_world[:3, 3])),
            "head_distance_from_right_base_m": float(np.linalg.norm(head_world[:3, 3] - right_base_world[:3, 3])),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[piper-calibration-bundle] wrote {args.output}")
    print(f"[piper-calibration-bundle] base_distance_m={bundle['derived_summary']['base_distance_m']:.6f}")
    print(f"[piper-calibration-bundle] head_world_pose={np.round(bundle['head_camera']['world_pose'], 6).tolist()}")


if __name__ == "__main__":
    main()
