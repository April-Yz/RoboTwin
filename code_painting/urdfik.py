#!/usr/bin/env python3
"""Curobo-based IK solver for R1, adapted for the local RoboTwin workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import Pose as CuroboPose
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DEFAULT_URDF = PROJECT_ROOT / "galaxea_sim" / "assets" / "r1_pro" / "robot.urdf"


class URDFInverseKinematics:
    def __init__(
        self,
        urdf_file: str | Path = DEFAULT_URDF,
        base_link: str = "base_link",
        ee_link: str = "left_gripper_link",
    ) -> None:
        self.urdf_file = str(Path(urdf_file).resolve())
        self.base_link = str(base_link)
        self.ee_link = str(ee_link)

        self.tensor_args = TensorDeviceType()
        self.robot_cfg = RobotConfig.from_basic(self.urdf_file, self.base_link, self.ee_link, self.tensor_args)
        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=1,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
        )
        self.ik_solver = IKSolver(self.ik_config)
        self.fk_model = CudaRobotModel(self.robot_cfg.kinematics)

    def solve_ik(
        self,
        target_position: Sequence[float],
        target_orientation_wxyz: Sequence[float],
        current_joints: Optional[Sequence[float]] = None,
    ):
        quat = np.asarray(target_orientation_wxyz, dtype=np.float64).reshape(4)
        norm = np.linalg.norm(quat)
        if norm <= 1e-12:
            print("[IK] Invalid quaternion: norm is zero")
            return None
        quat = quat / norm

        target_position_tensor = torch.tensor(list(target_position), device=self.tensor_args.device, dtype=torch.float32)
        target_orientation_tensor = torch.tensor(list(quat), device=self.tensor_args.device, dtype=torch.float32)
        goal = CuroboPose(target_position_tensor, target_orientation_tensor)

        seed_tensor = None
        if current_joints is not None:
            seed_tensor = torch.tensor(current_joints, device=self.tensor_args.device, dtype=torch.float32).view(1, -1)

        if seed_tensor is not None:
            result = self.ik_solver.solve_batch(goal, seed_config=seed_tensor)
        else:
            result = self.ik_solver.solve_batch(goal)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        is_success = bool(result.success.cpu().numpy().all())
        original_pos_thresh = self.ik_solver.position_threshold
        original_rot_thresh = self.ik_solver.rotation_threshold

        while not is_success:
            self.ik_solver.position_threshold *= 5
            self.ik_solver.rotation_threshold *= 2
            if seed_tensor is not None:
                result = self.ik_solver.solve_batch(goal, seed_config=seed_tensor)
            else:
                result = self.ik_solver.solve_batch(goal)
            is_success = bool(result.success.cpu().numpy().all())
            if self.ik_solver.position_threshold > 0.1:
                pos_err = float(result.position_error.cpu().numpy()[0, 0])
                print(f"[IK] Failed to converge (ee_link={self.ee_link}, pos_err={pos_err:.4f}m)")
                break

        self.ik_solver.position_threshold = original_pos_thresh
        self.ik_solver.rotation_threshold = original_rot_thresh
        return result if is_success else None

    def forward_kinematics(self, joint_angles: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint_tensor = torch.tensor(joint_angles, device=self.tensor_args.device, dtype=torch.float32).view(1, -1)
        kin_state = self.fk_model.get_state(joint_tensor)
        position = kin_state.ee_position.cpu().numpy()[0]
        quaternion_wxyz = kin_state.ee_quaternion.cpu().numpy()[0]
        quat_xyzw = [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]]
        euler_zyx = R.from_quat(quat_xyzw).as_euler("zyx")
        return position, quaternion_wxyz, euler_zyx
