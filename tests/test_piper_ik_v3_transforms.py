#!/usr/bin/env python3
"""Round-trip tests for the isolated Piper IK V3 pose conversions."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from piper_ik_v3_transforms import (
    link6_pose_to_ours_pose,
    link6_pose_to_real_piper_tcp_pose,
    ours_pose_to_link6_pose,
    pose_wxyz_to_matrix,
    real_piper_tcp_pose_to_link6_pose,
)


GLOBAL_TRANS = np.diag([1.0, -1.0, -1.0])
DELTA = np.eye(3, dtype=np.float64)


def random_pose(rng: np.random.RandomState) -> np.ndarray:
    quat_xyzw = Rotation.random(random_state=rng).as_quat()
    return np.concatenate(
        [
            rng.uniform([-0.5, -0.4, 0.1], [0.7, 0.5, 1.2]),
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        ]
    )


class PiperIKV3TransformTest(unittest.TestCase):
    def assert_pose_close(self, actual: np.ndarray, expected: np.ndarray) -> None:
        actual_matrix = pose_wxyz_to_matrix(actual)
        expected_matrix = pose_wxyz_to_matrix(expected)
        np.testing.assert_allclose(actual_matrix, expected_matrix, atol=1e-10, rtol=0.0)

    def test_ours_tcp_roundtrip(self) -> None:
        rng = np.random.RandomState(7)
        for _ in range(100):
            link6 = random_pose(rng)
            tcp = link6_pose_to_ours_pose(
                link6,
                GLOBAL_TRANS,
                DELTA,
                0.12,
                include_tcp_offset=True,
            )
            recovered = ours_pose_to_link6_pose(
                tcp,
                GLOBAL_TRANS,
                DELTA,
                0.12,
                includes_tcp_offset=True,
            )
            self.assert_pose_close(recovered, link6)

    def test_ours_ee_roundtrip(self) -> None:
        rng = np.random.RandomState(11)
        for _ in range(100):
            link6 = random_pose(rng)
            ee = link6_pose_to_ours_pose(
                link6,
                GLOBAL_TRANS,
                DELTA,
                0.12,
                include_tcp_offset=False,
            )
            recovered = ours_pose_to_link6_pose(
                ee,
                GLOBAL_TRANS,
                DELTA,
                0.12,
                includes_tcp_offset=False,
            )
            self.assert_pose_close(recovered, link6)

    def test_real_piper_tcp_roundtrip(self) -> None:
        rng = np.random.RandomState(13)
        for _ in range(100):
            link6 = random_pose(rng)
            tcp = link6_pose_to_real_piper_tcp_pose(link6, tool_length_m=0.19)
            recovered = real_piper_tcp_pose_to_link6_pose(tcp, tool_length_m=0.19)
            self.assert_pose_close(recovered, link6)


if __name__ == "__main__":
    unittest.main()
