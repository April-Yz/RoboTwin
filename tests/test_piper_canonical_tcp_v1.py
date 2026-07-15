#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO / "code_painting/piper_canonical_tcp_v1"
sys.path.insert(0, str(MODULE_DIR))

from frame_contract import (  # noqa: E402
    R_CGRASP_RTCP,
    R_L6SIM_L6URDF,
    R_RTCP_CGRASP,
    SERVER_TOOL_LENGTH_M,
    SERVER_TOOL_PITCH_RAD,
    canonical_grasp_pose_to_real_tcp_pose,
    frame_contract_payload,
    matrix_to_pose_wxyz,
    pose_wxyz_to_matrix,
    real_tcp_pose_to_urdf_link6_pose,
    server_urdf_link6_to_real_tcp_transform,
    sim_link6_pose_to_real_tcp_pose,
    sim_link6_pose_to_urdf_link6_pose,
    sim_link6_to_urdf_link6_transform,
    urdf_link6_pose_to_real_tcp_pose,
    world_real_tcp_to_base_urdf_link6_pose,
)


def random_pose(rng: np.random.Generator) -> np.ndarray:
    quat_xyzw = Rotation.random(random_state=rng).as_quat()
    return np.array(
        [
            *rng.uniform(-1.0, 1.0, size=3),
            quat_xyzw[3],
            quat_xyzw[0],
            quat_xyzw[1],
            quat_xyzw[2],
        ],
        dtype=np.float64,
    )


class PiperCanonicalTCPV1Test(unittest.TestCase):
    def test_server_literals_and_translation_order(self) -> None:
        self.assertEqual(SERVER_TOOL_PITCH_RAD, -1.57)
        self.assertEqual(SERVER_TOOL_LENGTH_M, 0.19)
        tool = server_urdf_link6_to_real_tcp_transform()
        expected = tool[:3, :3] @ np.array([0.19, 0.0, 0.0])
        np.testing.assert_allclose(tool[:3, 3], expected, atol=1e-15)

    def test_urdf_link6_real_tcp_roundtrip(self) -> None:
        rng = np.random.default_rng(20260715)
        for _ in range(200):
            urdf_link6 = random_pose(rng)
            recovered = real_tcp_pose_to_urdf_link6_pose(
                urdf_link6_pose_to_real_tcp_pose(urdf_link6)
            )
            np.testing.assert_allclose(
                pose_wxyz_to_matrix(recovered),
                pose_wxyz_to_matrix(urdf_link6),
                atol=1e-12,
            )

    def test_sim_link6_adapter_and_complete_forward_chain(self) -> None:
        np.testing.assert_array_equal(
            sim_link6_to_urdf_link6_transform()[:3, :3],
            R_L6SIM_L6URDF,
        )
        rng = np.random.default_rng(600)
        for _ in range(100):
            sim_link6 = random_pose(rng)
            expected_urdf = (
                pose_wxyz_to_matrix(sim_link6)
                @ sim_link6_to_urdf_link6_transform()
            )
            actual_urdf = pose_wxyz_to_matrix(
                sim_link6_pose_to_urdf_link6_pose(sim_link6)
            )
            np.testing.assert_allclose(actual_urdf, expected_urdf, atol=1e-12)
            expected_real = (
                expected_urdf @ server_urdf_link6_to_real_tcp_transform()
            )
            actual_real = pose_wxyz_to_matrix(
                sim_link6_pose_to_real_tcp_pose(sim_link6)
            )
            np.testing.assert_allclose(actual_real, expected_real, atol=1e-12)

    def test_preview_canonical_axes_are_undone_exactly(self) -> None:
        np.testing.assert_allclose(
            R_RTCP_CGRASP @ R_CGRASP_RTCP,
            np.eye(3),
            atol=0.0,
        )
        rng = np.random.default_rng(90)
        for _ in range(100):
            real_tcp = random_pose(rng)
            world_real = pose_wxyz_to_matrix(real_tcp)
            world_canonical = world_real.copy()
            world_canonical[:3, :3] = world_real[:3, :3] @ R_RTCP_CGRASP
            recovered = canonical_grasp_pose_to_real_tcp_pose(
                matrix_to_pose_wxyz(world_canonical)
            )
            np.testing.assert_allclose(
                pose_wxyz_to_matrix(recovered),
                world_real,
                atol=1e-12,
            )

    def test_world_base_inverse_chain(self) -> None:
        rng = np.random.default_rng(190)
        for _ in range(100):
            world_base = random_pose(rng)
            base_link6 = random_pose(rng)
            world_real_tcp = (
                pose_wxyz_to_matrix(world_base)
                @ pose_wxyz_to_matrix(base_link6)
                @ server_urdf_link6_to_real_tcp_transform()
            )
            recovered = world_real_tcp_to_base_urdf_link6_pose(
                matrix_to_pose_wxyz(world_real_tcp), world_base
            )
            np.testing.assert_allclose(
                pose_wxyz_to_matrix(recovered),
                pose_wxyz_to_matrix(base_link6),
                atol=1e-12,
            )

    def test_contract_has_world_and_local_axis_labels(self) -> None:
        contract = frame_contract_payload()
        self.assertEqual(contract["input_contract"]["planner_target"], "T_W_RTCP")
        self.assertEqual(
            contract["axes"]["local_RTCP_+X"]["physical_role"],
            "approach/forward",
        )
        self.assertEqual(contract["axes"]["world_+Z"]["physical_role"], "vertical/up")


if __name__ == "__main__":
    unittest.main()
