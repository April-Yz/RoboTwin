#!/usr/bin/env python3
"""Frame contract for the isolated PiperCanonicalTCP-v1 experiment line.

The authoritative tool transform intentionally mirrors the Piper server
literal values instead of replacing ``-1.57`` with ``-pi/2``::

    T_L6URDF_RTCP = Ry(-1.57) @ Tx(0.19)

SAPIEN and the CuRobo URDF expose different local axes for the entity named
``link6``.  Runtime same-q FK establishes the exact adapter::

    T_L6SIM_L6URDF = Ry(+pi/2)  # exact signed axis permutation

Pose names use ``T_A_B`` for the pose of frame B expressed in frame A.
Rotation names use ``R_A_B`` for mapping a vector from local B into A.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA = "piper_canonical_tcp_v1.frame_contract.v1"
SERVER_TOOL_PITCH_RAD = -1.57
SERVER_TOOL_LENGTH_M = 0.19

# The robot-frame preview stores each raw AnyGrasp/Real-TCP orientation as
# ``R_W_CGRASP = R_W_RTCP @ R_RTCP_CGRASP``.  This is an exact signed axis
# permutation, deliberately kept separate from the Piper server's approximate
# -1.57-radian tool pitch above.
R_RTCP_CGRASP = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
R_CGRASP_RTCP = R_RTCP_CGRASP.T

# SAPIEN's raw link6 actor and CuRobo's URDF link6 have the same origin but
# different local axes.  This has the same numeric matrix as
# R_RTCP_CGRASP above, but it is a separate physical/frame relationship.
R_L6SIM_L6URDF = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def _rotation_from_wxyz(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    quat /= norm
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(
        np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    ).as_quat()
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


def rotation_y(angle_rad: float) -> np.ndarray:
    return Rotation.from_euler("y", float(angle_rad)).as_matrix()


def sim_link6_to_urdf_link6_transform() -> np.ndarray:
    """Return the exact same-origin SAPIEN-link6 to URDF-link6 adapter."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R_L6SIM_L6URDF
    return transform


def server_urdf_link6_to_real_tcp_transform() -> np.ndarray:
    """Return the exact server transform ``Ry(-1.57) @ Tx(0.19)``."""
    rotation = rotation_y(SERVER_TOOL_PITCH_RAD)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = rotation @ np.array(
        [SERVER_TOOL_LENGTH_M, 0.0, 0.0], dtype=np.float64
    )
    return transform


def canonical_grasp_pose_to_real_tcp_pose(
    canonical_grasp_pose_wxyz: Sequence[float],
) -> np.ndarray:
    """Undo the robot-frame preview's local-axis canonicalization.

    The origins are identical; only the local orientation basis changes:
    ``T_W_RTCP = T_W_CGRASP @ T_CGRASP_RTCP``.
    """
    world_canonical = pose_wxyz_to_matrix(canonical_grasp_pose_wxyz)
    canonical_to_real = np.eye(4, dtype=np.float64)
    canonical_to_real[:3, :3] = R_CGRASP_RTCP
    return matrix_to_pose_wxyz(world_canonical @ canonical_to_real)


def urdf_link6_pose_to_real_tcp_pose(
    urdf_link6_pose_wxyz: Sequence[float],
) -> np.ndarray:
    urdf_link6 = pose_wxyz_to_matrix(urdf_link6_pose_wxyz)
    real_tcp = urdf_link6 @ server_urdf_link6_to_real_tcp_transform()
    return matrix_to_pose_wxyz(real_tcp)


def real_tcp_pose_to_urdf_link6_pose(
    real_tcp_pose_wxyz: Sequence[float],
) -> np.ndarray:
    real_tcp = pose_wxyz_to_matrix(real_tcp_pose_wxyz)
    urdf_link6 = real_tcp @ np.linalg.inv(
        server_urdf_link6_to_real_tcp_transform()
    )
    return matrix_to_pose_wxyz(urdf_link6)


def sim_link6_pose_to_urdf_link6_pose(
    sim_link6_pose_wxyz: Sequence[float],
) -> np.ndarray:
    sim_link6 = pose_wxyz_to_matrix(sim_link6_pose_wxyz)
    urdf_link6 = sim_link6 @ sim_link6_to_urdf_link6_transform()
    return matrix_to_pose_wxyz(urdf_link6)


def sim_link6_pose_to_real_tcp_pose(
    sim_link6_pose_wxyz: Sequence[float],
) -> np.ndarray:
    """Apply the complete SAPIEN-readback to Real-Piper-TCP chain."""
    sim_link6 = pose_wxyz_to_matrix(sim_link6_pose_wxyz)
    real_tcp = (
        sim_link6
        @ sim_link6_to_urdf_link6_transform()
        @ server_urdf_link6_to_real_tcp_transform()
    )
    return matrix_to_pose_wxyz(real_tcp)


def world_real_tcp_to_base_urdf_link6_pose(
    world_real_tcp_pose_wxyz: Sequence[float],
    world_base_pose_wxyz: Sequence[float],
) -> np.ndarray:
    """Apply ``T_B_L6URDF = inv(T_W_B) @ T_W_RTCP @ inv(T_L6URDF_RTCP)``."""
    world_real_tcp = pose_wxyz_to_matrix(world_real_tcp_pose_wxyz)
    world_base = pose_wxyz_to_matrix(world_base_pose_wxyz)
    base_urdf_link6 = (
        np.linalg.inv(world_base)
        @ world_real_tcp
        @ np.linalg.inv(server_urdf_link6_to_real_tcp_transform())
    )
    return matrix_to_pose_wxyz(base_urdf_link6)


def frame_contract_payload() -> dict[str, Any]:
    sim_to_urdf = sim_link6_to_urdf_link6_transform()
    tool = server_urdf_link6_to_real_tcp_transform()
    return {
        "schema": SCHEMA,
        "pipeline": "PiperCanonicalTCP-v1",
        "authoritative_source": (
            "/home/piper/pika_ros/src/PikaAnyArm/piper/"
            "pika_remote_piper/scripts/forward_inverse_kinematics.py"
        ),
        "server_literals": {
            "tool_pitch_rad": SERVER_TOOL_PITCH_RAD,
            "tool_length_m": SERVER_TOOL_LENGTH_M,
            "formula": "T_L6URDF_RTCP = Ry(-1.57) @ Tx(0.19)",
        },
        "notation": {
            "T_A_B": "pose of frame B expressed in frame A",
            "R_A_B": "rotation mapping a local-B vector into frame A",
            "p_A_B": "origin of frame B expressed in frame A",
            "W": "shared Piper0515 world frame",
            "B_L": "left-arm base frame",
            "B_R": "right-arm base frame",
            "L6_SIM": "raw SAPIEN link6 actor frame used for simulator readback",
            "L6_URDF": (
                "CuRobo URDF link6 frame used by IK/FK and by the Piper "
                "server tool formula"
            ),
            "RTCP": "Real Piper TCP frame published/consumed by the server",
            "CGRASP": (
                "canonical local grasp axes stored by the robot-frame preview; "
                "its origin equals RTCP but its local axes do not"
            ),
        },
        "axes": {
            "local_RTCP_+X": {
                "color": "red",
                "physical_role": "approach/forward",
            },
            "local_RTCP_+Y": {
                "color": "green",
                "physical_role": "finger opening/lateral",
            },
            "local_RTCP_+Z": {
                "color": "blue",
                "physical_role": "side normal",
            },
            "world_+X": {"color": "red", "physical_role": "Piper0515 world X"},
            "world_+Y": {"color": "green", "physical_role": "Piper0515 world Y"},
            "world_+Z": {"color": "blue", "physical_role": "vertical/up"},
        },
        "transforms": {
            "R_RTCP_CGRASP": R_RTCP_CGRASP.tolist(),
            "R_CGRASP_RTCP": R_CGRASP_RTCP.tolist(),
            "preview_axis_formula": (
                "R_W_CGRASP = R_W_RTCP @ R_RTCP_CGRASP; "
                "R_W_RTCP = R_W_CGRASP @ R_CGRASP_RTCP"
            ),
            "preview_origin_formula": "p_W_CGRASP = p_W_RTCP",
            "T_L6SIM_L6URDF": sim_to_urdf.tolist(),
            "sim_urdf_formula": (
                "T_B_L6URDF = T_B_L6SIM @ T_L6SIM_L6URDF; "
                "R_L6SIM_L6URDF is exact Ry(+pi/2)"
            ),
            "T_L6URDF_RTCP": tool.tolist(),
            "sim_forward": (
                "T_W_RTCP = T_W_B @ T_B_L6SIM @ "
                "T_L6SIM_L6URDF @ T_L6URDF_RTCP"
            ),
            "urdf_forward": (
                "T_W_RTCP = T_W_B @ T_B_L6URDF @ T_L6URDF_RTCP"
            ),
            "ik_inverse": (
                "T_B_L6URDF = inv(T_W_B) @ T_W_RTCP @ "
                "inv(T_L6URDF_RTCP)"
            ),
        },
        "input_contract": {
            "planner_target": "T_W_RTCP",
            "position_field": "p_W_RTCP_m",
            "rotation_field": "R_W_RTCP",
            "quaternion_order": "wxyz",
            "position_unit": "m",
        },
        "legacy_warning": (
            "Legacy OursV2 fields named *_ee_* may store an Ours TCP. "
            "They are never accepted implicitly by this pipeline."
        ),
    }


def write_frame_contract(path: Path, extra: Mapping[str, Any] | None = None) -> None:
    payload = frame_contract_payload()
    if extra:
        payload["run"] = dict(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, required=True)
    parser.add_argument("--task", default="")
    parser.add_argument("--episode-id", type=int, default=-1)
    parser.add_argument("--strategy", default="")
    parser.add_argument("--source-semantics", default="")
    args = parser.parse_args()
    write_frame_contract(
        args.write,
        {
            "task": args.task,
            "episode_id": args.episode_id,
            "strategy": args.strategy,
            "source_semantics": args.source_semantics,
        },
    )
    print(args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
