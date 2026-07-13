#!/usr/bin/env python3
"""Explicit Piper IK V3 pose conversions.

V3 keeps pose semantics explicit instead of relying on the historical
``0.12 - gripper_bias`` shortcut.  It supports the current RoboTwin Ours TCP,
the corresponding Ours EE-labelled/link6-origin pose, and the real Piper TCP
published by ``piper_FK.py``.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def _rotation_from_wxyz(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    quat /= norm
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64)).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def pose_wxyz_to_matrix(pose_wxyz: Sequence[float]) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation_from_wxyz(pose[3:])
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_pose_wxyz(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return np.concatenate(
        [transform[:3, 3], _wxyz_from_rotation(transform[:3, :3])]
    )


def link6_pose_to_ours_pose(
    link6_pose_wxyz: Sequence[float],
    global_trans_matrix: np.ndarray,
    delta_matrix: np.ndarray,
    gripper_bias_m: float,
    *,
    include_tcp_offset: bool,
) -> np.ndarray:
    """Mirror RoboTwin ``get_*_tcp_pose`` / ``get_*_ee_pose`` exactly."""
    link6 = pose_wxyz_to_matrix(link6_pose_wxyz)
    global_trans = np.asarray(global_trans_matrix, dtype=np.float64).reshape(3, 3)
    delta = np.asarray(delta_matrix, dtype=np.float64).reshape(3, 3)
    ours_rotation = link6[:3, :3] @ global_trans @ delta
    ours_position = link6[:3, 3].copy()
    if include_tcp_offset:
        ours_position += ours_rotation @ np.array(
            [float(gripper_bias_m), 0.0, 0.0], dtype=np.float64
        )
    return np.concatenate([ours_position, _wxyz_from_rotation(ours_rotation)])


def ours_pose_to_link6_pose(
    ours_pose_wxyz: Sequence[float],
    global_trans_matrix: np.ndarray,
    delta_matrix: np.ndarray,
    gripper_bias_m: float,
    *,
    includes_tcp_offset: bool,
) -> np.ndarray:
    """Exact inverse of the current RoboTwin Ours pose readback convention."""
    ours = pose_wxyz_to_matrix(ours_pose_wxyz)
    global_trans = np.asarray(global_trans_matrix, dtype=np.float64).reshape(3, 3)
    delta = np.asarray(delta_matrix, dtype=np.float64).reshape(3, 3)
    link6_rotation = ours[:3, :3] @ np.linalg.inv(delta) @ np.linalg.inv(global_trans)
    link6_position = ours[:3, 3].copy()
    if includes_tcp_offset:
        link6_position -= ours[:3, :3] @ np.array(
            [float(gripper_bias_m), 0.0, 0.0], dtype=np.float64
        )
    return np.concatenate([link6_position, _wxyz_from_rotation(link6_rotation)])


def real_piper_link6_to_tcp_transform(
    tool_length_m: float = 0.19,
    tool_pitch_rad: float = -math.pi / 2.0,
) -> np.ndarray:
    """Return the real Piper ``first_matrix @ second_matrix`` tool transform.

    The server implementation uses ``Ry(-1.57) @ Tx(tool_length)``.  The
    default here uses the exact -pi/2 equivalent while keeping both values
    configurable for reproduction tests.
    """
    rotation = Rotation.from_euler("y", float(tool_pitch_rad)).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = rotation @ np.array(
        [float(tool_length_m), 0.0, 0.0], dtype=np.float64
    )
    return transform


def link6_pose_to_real_piper_tcp_pose(
    link6_pose_wxyz: Sequence[float],
    tool_length_m: float = 0.19,
    tool_pitch_rad: float = -math.pi / 2.0,
) -> np.ndarray:
    link6 = pose_wxyz_to_matrix(link6_pose_wxyz)
    tcp = link6 @ real_piper_link6_to_tcp_transform(tool_length_m, tool_pitch_rad)
    return matrix_to_pose_wxyz(tcp)


def real_piper_tcp_pose_to_link6_pose(
    tcp_pose_wxyz: Sequence[float],
    tool_length_m: float = 0.19,
    tool_pitch_rad: float = -math.pi / 2.0,
) -> np.ndarray:
    """Mirror the real Piper IK by removing the same tool transform as FK."""
    tcp = pose_wxyz_to_matrix(tcp_pose_wxyz)
    link6 = tcp @ np.linalg.inv(
        real_piper_link6_to_tcp_transform(tool_length_m, tool_pitch_rad)
    )
    return matrix_to_pose_wxyz(link6)


__all__ = [
    "link6_pose_to_ours_pose",
    "ours_pose_to_link6_pose",
    "link6_pose_to_real_piper_tcp_pose",
    "real_piper_tcp_pose_to_link6_pose",
    "real_piper_link6_to_tcp_transform",
    "matrix_to_pose_wxyz",
    "pose_wxyz_to_matrix",
]
