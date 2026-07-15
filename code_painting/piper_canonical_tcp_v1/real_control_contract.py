#!/usr/bin/env python3
"""Pure transform helpers for OursV2/Canonical/Real control comparisons.

All positions in this module are expressed in an arm base frame unless a
function name explicitly says ``world``.  The common EE-pose experiment input
is the numeric Piper server Real-TCP pose ``T_B_RTCP``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from frame_contract import server_urdf_link6_to_real_tcp_transform


SCHEMA = "piper_canonical_tcp_v1.real_control_compare.v1"
OURS_V2_DEFAULT_GLOBAL_TRANSFORM = np.diag([1.0, -1.0, -1.0])
OURS_V2_DEFAULT_GRIPPER_BIAS_M = 0.12


def make_transform(position: Sequence[float], rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def split_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return transform[:3, 3].copy(), transform[:3, :3].copy()


def oursv2_tcp_from_urdf_link6(
    link6_position: Sequence[float],
    link6_rotation: np.ndarray,
    global_transform: np.ndarray = OURS_V2_DEFAULT_GLOBAL_TRANSFORM,
    gripper_bias_m: float = OURS_V2_DEFAULT_GRIPPER_BIAS_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the current OursV2 q->TCP convention.

    ``R_B_OTCP = R_B_L6URDF @ diag(1,-1,-1)`` and the TCP origin is
    ``gripper_bias_m`` along local OursV2 TCP +X.
    """
    link_position = np.asarray(link6_position, dtype=np.float64).reshape(3)
    link_rotation = np.asarray(link6_rotation, dtype=np.float64).reshape(3, 3)
    global_transform = np.asarray(global_transform, dtype=np.float64).reshape(3, 3)
    tcp_rotation = link_rotation @ global_transform
    tcp_position = link_position + tcp_rotation @ np.array(
        [float(gripper_bias_m), 0.0, 0.0], dtype=np.float64
    )
    return tcp_position, tcp_rotation


def canonical_rtcp_from_urdf_link6(
    link6_position: Sequence[float], link6_rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    link6 = make_transform(link6_position, link6_rotation)
    return split_transform(link6 @ server_urdf_link6_to_real_tcp_transform())


def canonical_link6_target_from_real_tcp(
    real_tcp_position: Sequence[float], real_tcp_rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    real_tcp = make_transform(real_tcp_position, real_tcp_rotation)
    link6 = real_tcp @ np.linalg.inv(server_urdf_link6_to_real_tcp_transform())
    return split_transform(link6)


def oursv2_legacy_link6_target_from_real_tcp_numeric(
    real_tcp_position: Sequence[float], real_tcp_rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the old default EE-pose IK input semantics.

    With target_retreat=0 and piper_urdfik_apply_global_trans_to_ik=0, the
    numeric pose is sent to URDF link6 IK unchanged.  In this experiment the
    numeric input is a Piper server ``T_B_RTCP``; intentionally *no* canonical
    inverse-tool conversion is applied in this branch.
    """
    return (
        np.asarray(real_tcp_position, dtype=np.float64).reshape(3).copy(),
        np.asarray(real_tcp_rotation, dtype=np.float64).reshape(3, 3).copy(),
    )


def rotation_error_rad(actual: np.ndarray, target: np.ndarray) -> float:
    relative = np.asarray(target, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        actual, dtype=np.float64
    ).reshape(3, 3)
    return float(Rotation.from_matrix(relative).magnitude())
