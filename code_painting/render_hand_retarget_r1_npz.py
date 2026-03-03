#!/usr/bin/env python3
"""Replay camera-space hand-retargeted gripper poses on R1 inside RoboTwin."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from envs.robot import Robot


DEFAULT_ROBOT_CONFIG = PROJECT_ROOT / "robot_config_R1_pro.json"
CV_TO_WORLD_CAMERA_PRESETS = {
    "legacy_r1": np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64),
    "diag_flip_yz": np.diag([1.0, -1.0, -1.0]).astype(np.float64),
}
DEFAULT_HEAD_CAMERA_LOCAL_POS = np.array([0.0, 0.0, 0.0], dtype=np.float64)
DEFAULT_HEAD_CAMERA_LOCAL_QUAT_WXYZ = np.array([1.0, 1.0, -1.0, 1.0], dtype=np.float64)
DEFAULT_WRIST_CAMERA_LOCAL_POS = np.array([0.0, 0.0, 0.0], dtype=np.float64)
DEFAULT_WRIST_CAMERA_LOCAL_QUAT_WXYZ = None
TORSO_JOINT_NAMES = ("torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4")
DEFAULT_TORSO_QPOS = np.array([0.25, -0.4, -0.85, 0.0], dtype=np.float64)
HIDDEN_DEBUG_POSE = sapien.Pose([0.0, 0.0, -10.0], [1.0, 0.0, 0.0, 0.0])


def vec_norm(v: np.ndarray) -> float:
    return math.sqrt(float(np.dot(v, v)))


def normalize_quat_wxyz(quat_wxyz: Sequence[float]) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("Zero-norm quaternion.")
    return quat / norm


def quat_wxyz_to_xyzw(quat_wxyz: Sequence[float]) -> np.ndarray:
    quat = normalize_quat_wxyz(quat_wxyz)
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(quat_xyzw: Sequence[float]) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("Zero-norm quaternion.")
    quat = quat / norm
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def quat_multiply_wxyz(quat_a_wxyz: Sequence[float], quat_b_wxyz: Sequence[float]) -> np.ndarray:
    rot_a = R.from_quat(quat_wxyz_to_xyzw(quat_a_wxyz))
    rot_b = R.from_quat(quat_wxyz_to_xyzw(quat_b_wxyz))
    return quat_xyzw_to_wxyz((rot_a * rot_b).as_quat())


def quat_from_euler_wxyz(seq: str, angles, degrees: bool = False) -> np.ndarray:
    return quat_xyzw_to_wxyz(R.from_euler(seq, angles, degrees=degrees).as_quat())


DEFAULT_WRIST_CAMERA_LOCAL_QUAT_WXYZ = quat_multiply_wxyz(
    quat_from_euler_wxyz("xyz", [math.radians(-10.0), 0.0, -math.pi / 2], degrees=False),
    [0.5, 0.5, -0.5, 0.5],
)


def pose_to_matrix(pose: sapien.Pose) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_quat(quat_wxyz_to_xyzw(pose.q)).as_matrix()
    mat[:3, 3] = np.asarray(pose.p, dtype=np.float64)
    return mat


def matrix_to_pose(mat: np.ndarray) -> sapien.Pose:
    mat = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    rot = orthonormalize_rotation(mat[:3, :3])
    quat = quat_xyzw_to_wxyz(R.from_matrix(rot).as_quat())
    return sapien.Pose(mat[:3, 3], quat)


def camera_pose_from_forward_left(position: np.ndarray, forward: np.ndarray, left: np.ndarray) -> sapien.Pose:
    position = np.asarray(position, dtype=np.float64).reshape(3)
    forward = np.asarray(forward, dtype=np.float64).reshape(3)
    left = np.asarray(left, dtype=np.float64).reshape(3)
    forward = forward / max(np.linalg.norm(forward), 1e-12)
    left = left - forward * float(np.dot(left, forward))
    left = left / max(np.linalg.norm(left), 1e-12)
    up = np.cross(forward, left)
    up = up / max(np.linalg.norm(up), 1e-12)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.column_stack([forward, left, up])
    mat[:3, 3] = position
    return sapien.Pose(mat)


def invert_pose_matrix(mat: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    inv[:3, :3] = rot.T
    inv[:3, 3] = -(rot.T @ trans)
    return inv


def orthonormalize_rotation(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    u, _, vh = np.linalg.svd(rot)
    rot_ortho = u @ vh
    if np.linalg.det(rot_ortho) < 0:
        u[:, -1] *= -1.0
        rot_ortho = u @ vh
    return rot_ortho


def quat_angle_deg_wxyz(quat_a_wxyz: Sequence[float], quat_b_wxyz: Sequence[float]) -> float:
    rot_a = R.from_quat(quat_wxyz_to_xyzw(quat_a_wxyz))
    rot_b = R.from_quat(quat_wxyz_to_xyzw(quat_b_wxyz))
    delta = rot_a.inv() * rot_b
    return float(np.rad2deg(delta.magnitude()))


def axis_angle_errors_deg_wxyz(quat_actual_wxyz: Sequence[float], quat_target_wxyz: Sequence[float]) -> Dict[str, float]:
    rot_actual = R.from_quat(quat_wxyz_to_xyzw(quat_actual_wxyz)).as_matrix()
    rot_target = R.from_quat(quat_wxyz_to_xyzw(quat_target_wxyz)).as_matrix()
    errors: Dict[str, float] = {}
    for idx, axis_name in enumerate(("x", "y", "z")):
        v_actual = rot_actual[:, idx]
        v_target = rot_target[:, idx]
        dot = float(np.clip(np.dot(v_actual, v_target), -1.0, 1.0))
        errors[axis_name] = float(np.rad2deg(np.arccos(dot)))
    return errors


def format_vec3(vec: Sequence[float]) -> str:
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    return f"[{arr[0]:+.3f}, {arr[1]:+.3f}, {arr[2]:+.3f}]"


def position_axis_errors(actual_pos: Sequence[float], target_pos: Sequence[float]) -> Dict[str, float]:
    actual = np.asarray(actual_pos, dtype=np.float64).reshape(3)
    target = np.asarray(target_pos, dtype=np.float64).reshape(3)
    delta = actual - target
    return {
        "x": float(delta[0]),
        "y": float(delta[1]),
        "z": float(delta[2]),
    }


def rotation_axes_world_wxyz(quat_wxyz: Sequence[float]) -> Dict[str, np.ndarray]:
    rot = R.from_quat(quat_wxyz_to_xyzw(quat_wxyz)).as_matrix()
    return {
        "x": rot[:, 0].copy(),
        "y": rot[:, 1].copy(),
        "z": rot[:, 2].copy(),
    }


def short_status(status: str) -> str:
    s = str(status).strip().lower()
    if s == "success":
        return "succ"
    if s == "fail":
        return "fail"
    if s == "skipped":
        return "skip"
    if s == "missing":
        return "miss"
    return s[:4] if s else "unk"


def make_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    return obj


def format_offset_label(prefix: str, value: float) -> str:
    text = f"{float(value):+0.2f}"
    text = text.replace("+", "p").replace("-", "m").replace(".", "p")
    return f"{prefix}_{text}"


def calc_gripper_pose_from_keypoints(
    thumb_tip_pos: np.ndarray,
    index_tip_pos: np.ndarray,
    index_joint_pos: np.ndarray,
    retreat_distance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    thumb_tip_pos = np.asarray(thumb_tip_pos, dtype=np.float64)
    index_tip_pos = np.asarray(index_tip_pos, dtype=np.float64)
    index_joint_pos = np.asarray(index_joint_pos, dtype=np.float64)

    gripper_position = 0.5 * (thumb_tip_pos + index_tip_pos)

    y_axis = thumb_tip_pos - index_tip_pos
    y_norm = vec_norm(y_axis)
    if y_norm < 1e-9:
        raise ValueError("Degenerate y-axis.")
    y_axis = y_axis / y_norm

    v_temp = index_joint_pos - index_tip_pos
    x_axis = np.cross(v_temp, y_axis)
    x_norm = vec_norm(x_axis)
    if x_norm < 1e-9:
        raise ValueError("Degenerate x-axis.")
    x_axis = x_axis / x_norm

    z_axis = np.cross(x_axis, y_axis)
    z_norm = vec_norm(z_axis)
    if z_norm < 1e-9:
        raise ValueError("Degenerate z-axis.")
    z_axis = z_axis / z_norm

    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

    rz_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    rx_90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    rz_270 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
    rotation_matrix = rz_90 @ rotation_matrix @ rx_90.T @ rz_270.T

    retreat_position = gripper_position - retreat_distance * z_axis
    return gripper_position, retreat_position, rotation_matrix


@dataclass
class SideFrameTarget:
    valid: bool
    position_cam: np.ndarray
    rotation_cam: np.ndarray
    finger_distance: Optional[float]


class HandRetargetTrajectory:
    def __init__(
        self,
        npz_path: Path,
        retreat_distance: float,
        thumb_tip_idx: int,
        index_tip_idx: int,
        index_joint_idx: int,
    ):
        self.path = npz_path
        self.data = np.load(str(npz_path), allow_pickle=True)
        self.retreat_distance = float(retreat_distance)
        self.thumb_tip_idx = int(thumb_tip_idx)
        self.index_tip_idx = int(index_tip_idx)
        self.index_joint_idx = int(index_joint_idx)
        self.length = self._infer_length()

    def has_stored_gripper_pose(self, side: str, pose_source: str) -> bool:
        pos_key = f"{side}_{'gripper_position' if pose_source == 'gripper' else 'wrist_position_retreat'}"
        rot_key = f"{side}_gripper_rotation_matrix"
        return pos_key in self.data.files and rot_key in self.data.files

    def describe_pose_source(self, side: str, pose_source: str) -> str:
        if self.has_stored_gripper_pose(side, pose_source):
            return f"{side}: stored_npz"
        return f"{side}: recompute_from_keypoints"

    def _infer_length(self) -> int:
        candidates = [
            "left_gripper_position",
            "right_gripper_position",
            "left_kpts_3d_cam",
            "right_kpts_3d_cam",
            "left_hand_detected",
            "right_hand_detected",
        ]
        for key in candidates:
            if key in self.data.files:
                return int(np.asarray(self.data[key]).shape[0])
        raise ValueError(f"Could not infer frame count from {self.path}")

    def _get_optional(self, key: str) -> Optional[np.ndarray]:
        if key not in self.data.files:
            return None
        return np.asarray(self.data[key])

    def _compute_from_keypoints(self, side: str, frame_idx: int, pose_source: str) -> SideFrameTarget:
        detected = self._get_optional(f"{side}_hand_detected")
        if detected is not None and not bool(detected[frame_idx]):
            return SideFrameTarget(False, np.full(3, np.nan), np.full((3, 3), np.nan), None)

        kpts_key = f"{side}_kpts_3d_cam" if f"{side}_kpts_3d_cam" in self.data.files else f"{side}_kpts_3d"
        if kpts_key not in self.data.files:
            return SideFrameTarget(False, np.full(3, np.nan), np.full((3, 3), np.nan), None)
        kpts = np.asarray(self.data[kpts_key])[frame_idx]
        thumb_tip = kpts[self.thumb_tip_idx]
        index_tip = kpts[self.index_tip_idx]
        index_joint = kpts[self.index_joint_idx]
        if not (np.isfinite(thumb_tip).all() and np.isfinite(index_tip).all() and np.isfinite(index_joint).all()):
            return SideFrameTarget(False, np.full(3, np.nan), np.full((3, 3), np.nan), None)
        try:
            gripper_position, retreat_position, rotation = calc_gripper_pose_from_keypoints(
                thumb_tip,
                index_tip,
                index_joint,
                retreat_distance=self.retreat_distance,
            )
        except ValueError:
            return SideFrameTarget(False, np.full(3, np.nan), np.full((3, 3), np.nan), None)
        position = gripper_position if pose_source == "gripper" else retreat_position
        finger_distance = vec_norm(thumb_tip - index_tip)
        return SideFrameTarget(True, position, rotation, finger_distance)

    def get_side_target(self, side: str, frame_idx: int, pose_source: str) -> SideFrameTarget:
        pos_key = f"{side}_{'gripper_position' if pose_source == 'gripper' else 'wrist_position_retreat'}"
        rot_key = f"{side}_gripper_rotation_matrix"
        valid_key = f"{side}_gripper_valid"
        finger_key = f"{side}_gripper_finger_distance"

        if pos_key in self.data.files and rot_key in self.data.files:
            positions = np.asarray(self.data[pos_key])
            rotations = np.asarray(self.data[rot_key])
            valid = np.asarray(self.data[valid_key])[frame_idx] if valid_key in self.data.files else True
            finger_distance = None
            if finger_key in self.data.files:
                finger_value = float(np.asarray(self.data[finger_key])[frame_idx])
                if np.isfinite(finger_value):
                    finger_distance = finger_value
            position = np.asarray(positions[frame_idx], dtype=np.float64)
            rotation = np.asarray(rotations[frame_idx], dtype=np.float64)
            is_valid = bool(valid) and np.isfinite(position).all() and np.isfinite(rotation).all()
            return SideFrameTarget(is_valid, position, rotation, finger_distance)

        return self._compute_from_keypoints(side, frame_idx, pose_source)

    def build_gripper_series(self, side: str) -> np.ndarray:
        key = f"{side}_gripper_finger_distance"
        if key in self.data.files:
            values = np.asarray(self.data[key], dtype=np.float64)
            valid_key = f"{side}_gripper_valid"
            if valid_key in self.data.files:
                valid = np.asarray(self.data[valid_key], dtype=bool)
                values = values[valid]
            values = values[np.isfinite(values)]
            if values.size > 0:
                return values

        values: List[float] = []
        for frame_idx in range(self.length):
            target = self._compute_from_keypoints(side, frame_idx, pose_source="gripper")
            if target.valid and target.finger_distance is not None and math.isfinite(target.finger_distance):
                values.append(float(target.finger_distance))
        return np.asarray(values, dtype=np.float64)


class HandRetargetR1Renderer:
    def __init__(
        self,
        robot_config_path: Path,
        image_width: int,
        image_height: int,
        fovy_deg: float,
        torso_qpos: Sequence[float],
        robot_base_pose_override: Optional[Sequence[float]],
        third_person_view: bool,
        need_topp: bool,
        link_cam_debug_enable: bool,
        link_cam_axis_mode: str,
        link_cam_debug_rot_xyz_deg: Sequence[float],
        link_cam_debug_shift_fru: Sequence[float],
        camera_cv_axis_mode: str,
        head_camera_local_pos: Sequence[float],
        head_camera_local_quat_wxyz: Sequence[float],
        wrist_camera_local_pos: Sequence[float],
        wrist_camera_local_quat_wxyz: Sequence[float],
        camera_debug_target: str,
        enable_viewer: bool,
        viewer_frame_delay: float,
        viewer_wait_at_end: bool,
        debug_mode: bool,
        debug_force_orientation: str,
        debug_visualize_targets: bool,
        debug_target_axis_length: float,
        debug_target_axis_thickness: float,
        target_world_offset_xyz: Sequence[float],
        target_world_z_offset: float,
        disable_table: bool,
        camera_sweep_enable: bool,
        camera_sweep_steps_deg: Sequence[float],
    ):
        self.robot_config_path = robot_config_path
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.fovy = np.deg2rad(float(fovy_deg))
        self.torso_qpos = np.asarray(torso_qpos, dtype=np.float64).reshape(4)
        self.robot_base_pose_override = robot_base_pose_override
        self.third_person_view = bool(third_person_view)
        self.need_topp = bool(need_topp)
        self.link_cam_debug_enable = bool(link_cam_debug_enable)
        self.link_cam_axis_mode = str(link_cam_axis_mode).strip().lower()
        self.link_cam_debug_rot_xyz_deg = np.asarray(link_cam_debug_rot_xyz_deg, dtype=np.float64).reshape(3)
        self.link_cam_debug_shift_fru = np.asarray(link_cam_debug_shift_fru, dtype=np.float64).reshape(3)
        self.camera_cv_axis_mode = str(camera_cv_axis_mode).strip().lower()
        self.head_camera_local_pos = np.asarray(head_camera_local_pos, dtype=np.float64).reshape(3)
        self.head_camera_local_quat_wxyz = normalize_quat_wxyz(head_camera_local_quat_wxyz)
        self.wrist_camera_local_pos = np.asarray(wrist_camera_local_pos, dtype=np.float64).reshape(3)
        self.wrist_camera_local_quat_wxyz = normalize_quat_wxyz(wrist_camera_local_quat_wxyz)
        self.camera_debug_target = str(camera_debug_target).strip().lower()
        self.enable_viewer = bool(enable_viewer)
        self.viewer_frame_delay = max(float(viewer_frame_delay), 0.0)
        self.viewer_wait_at_end = bool(viewer_wait_at_end)
        self.debug_mode = bool(debug_mode)
        self.debug_force_orientation = str(debug_force_orientation).strip().lower()
        self.debug_visualize_targets = bool(debug_visualize_targets)
        self.debug_target_axis_length = max(float(debug_target_axis_length), 0.01)
        self.debug_target_axis_thickness = max(float(debug_target_axis_thickness), 0.001)
        self.target_world_offset_xyz = np.asarray(target_world_offset_xyz, dtype=np.float64).reshape(3)
        self.target_world_z_offset = float(target_world_z_offset)
        self.disable_table = bool(disable_table)
        self.camera_sweep_enable = bool(camera_sweep_enable)
        self.camera_sweep_steps_deg = tuple(float(v) for v in camera_sweep_steps_deg)

        self.robot: Optional["Robot"] = None
        self._head_camera_link = None
        self._left_wrist_camera_link = None
        self._right_wrist_camera_link = None
        self._table = None
        self._base_pose = None
        self._left_target_axis_actor = None
        self._right_target_axis_actor = None

        self._setup_scene()
        self._setup_cameras()
        if not self.disable_table:
            self._create_table()
        self._load_robot()

    def _setup_scene(self) -> None:
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)

        self.scene = self.engine.create_scene(sapien.SceneConfig())
        self.scene.set_timestep(1.0 / 250.0)
        self.scene.add_ground(0.0)
        self.scene.default_physical_material = self.scene.create_physical_material(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        )
        self.scene.set_ambient_light([0.45, 0.45, 0.45])
        self.scene.add_directional_light([0.0, 0.5, -1.0], [0.7, 0.7, 0.7], shadow=True)
        self.scene.add_point_light([1.0, -0.5, 1.8], [1.0, 1.0, 1.0], shadow=True)
        self.scene.add_point_light([-1.0, -0.5, 1.8], [1.0, 1.0, 1.0], shadow=True)

        self.viewer = None
        if self.enable_viewer:
            from sapien.utils.viewer import Viewer

            self.viewer = Viewer(self.renderer)
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(x=0.55, y=0.2, z=1.45)
            self.viewer.set_camera_rpy(r=0.0, p=-0.6, y=2.6)

    def _setup_cameras(self) -> None:
        self.zed_camera = self.scene.add_camera(
            name="zed_camera",
            width=self.image_width,
            height=self.image_height,
            fovy=self.fovy,
            near=0.01,
            far=100.0,
        )
        self.third_camera = self.scene.add_camera(
            name="third_camera",
            width=self.image_width,
            height=self.image_height,
            fovy=self.fovy,
            near=0.01,
            far=100.0,
        )
        self.left_wrist_camera = self.scene.add_camera(
            name="left_wrist_camera",
            width=self.image_width,
            height=self.image_height,
            fovy=np.deg2rad(43.973014784873506),
            near=0.01,
            far=100.0,
        )
        self.right_wrist_camera = self.scene.add_camera(
            name="right_wrist_camera",
            width=self.image_width,
            height=self.image_height,
            fovy=np.deg2rad(44.5846756133851),
            near=0.01,
            far=100.0,
        )
        identity_pose = sapien.Pose([0.0, 0.0, 1.5], [1.0, 0.0, 0.0, 0.0])
        self.zed_camera.set_entity_pose(identity_pose)
        self.third_camera.set_entity_pose(identity_pose)
        self.left_wrist_camera.set_entity_pose(identity_pose)
        self.right_wrist_camera.set_entity_pose(identity_pose)

    def _create_table(self) -> None:
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.2, 0.4, 0.725])
        builder.add_box_visual(half_size=[0.2, 0.4, 0.725], material=[0.6, 0.4, 0.2])
        self._table = builder.build_kinematic(name="table")
        self._table.set_pose(sapien.Pose([0.0, 0.0, -0.025]))

    def _create_debug_axis_actor(self, name: str) -> sapien.Entity:
        builder = self.scene.create_actor_builder()
        axis_len = self.debug_target_axis_length
        axis_half = axis_len * 0.5
        thick = self.debug_target_axis_thickness
        builder.add_sphere_visual(radius=thick * 1.8, material=[1.0, 1.0, 1.0])
        builder.add_box_visual(pose=sapien.Pose([axis_half, 0.0, 0.0]), half_size=[axis_half, thick, thick], material=[1.0, 0.0, 0.0])
        builder.add_box_visual(pose=sapien.Pose([0.0, axis_half, 0.0]), half_size=[thick, axis_half, thick], material=[0.0, 1.0, 0.0])
        builder.add_box_visual(pose=sapien.Pose([0.0, 0.0, axis_half]), half_size=[thick, thick, axis_half], material=[0.0, 0.4, 1.0])
        actor = builder.build_kinematic(name=name)
        actor.set_pose(HIDDEN_DEBUG_POSE)
        return actor

    def _load_robot(self) -> None:
        from envs.robot import Robot

        with self.robot_config_path.open("r", encoding="utf-8") as f:
            robot_cfg = json.load(f)

        self.robot = Robot(self.scene, self.need_topp, **robot_cfg)
        self.robot.init_joints()
        self.robot.move_to_homestate()
        self._set_fixed_torso()

        base_pose_raw = self.robot_base_pose_override
        if base_pose_raw is None:
            pose_cfg = robot_cfg["left_embodiment_config"]["robot_pose"][0]
            base_pose_raw = pose_cfg[:3] + pose_cfg[-4:]
        base_pose_arr = np.asarray(base_pose_raw, dtype=np.float64).reshape(7)
        self._base_pose = sapien.Pose(base_pose_arr[:3], normalize_quat_wxyz(base_pose_arr[3:]))
        self.robot.left_entity.set_root_pose(self._base_pose)
        self.robot.right_entity.set_root_pose(self._base_pose)
        self.robot.left_entity_origion_pose = self._base_pose
        self.robot.right_entity_origion_pose = self._base_pose

        self.robot.left_gripper_val = 0.8
        self.robot.right_gripper_val = 0.8
        self.robot.set_planner(self.scene)

        self._head_camera_link = self._find_robot_link(["zed_link", "head_camera", "head", "camera_link"])
        if self._head_camera_link is None:
            raise RuntimeError("Could not find zed/head camera link on R1.")
        self._left_wrist_camera_link = self._find_robot_link(["left_realsense_link", "left_D405_link", "left_camera"])
        self._right_wrist_camera_link = self._find_robot_link(["right_realsense_link", "right_D405_link", "right_camera"])

        if self.debug_visualize_targets:
            self._left_target_axis_actor = self._create_debug_axis_actor("left_target_axis")
            self._right_target_axis_actor = self._create_debug_axis_actor("right_target_axis")

        self._update_table_pose()
        self.update_robot_link_cameras()
        if self.debug_mode:
            self.print_head_camera_summary()
            self.print_wrist_camera_summary("left")
            self.print_wrist_camera_summary("right")

    def _find_robot_link(self, names: Sequence[str]):
        if self.robot is None:
            return None
        for name in names:
            link = self.robot.left_entity.find_link_by_name(name)
            if link is not None:
                return link
        return None

    def _set_fixed_torso(self) -> None:
        if self.robot is None:
            return
        entity = self.robot.left_entity
        active_joints = entity.get_active_joints()
        joint_map = {joint.get_name(): idx for idx, joint in enumerate(active_joints)}
        qpos = entity.get_qpos()
        for joint_name, target in zip(TORSO_JOINT_NAMES, self.torso_qpos):
            if joint_name not in joint_map:
                continue
            idx = joint_map[joint_name]
            qpos[idx] = float(target)
            active_joints[idx].set_drive_target(float(target))
            active_joints[idx].set_drive_velocity_target(0.0)
        entity.set_qpos(qpos)

    def _update_table_pose(self) -> None:
        if self._table is None or self._base_pose is None:
            return
        base_rot = R.from_quat(quat_wxyz_to_xyzw(self._base_pose.q)).as_matrix()
        forward = base_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        table_pos = np.asarray(self._base_pose.p, dtype=np.float64) + 0.65 * forward
        table_pos[2] = 0.0
        self._table.set_pose(sapien.Pose(table_pos, self._base_pose.q))

    def _camera_axis_mode_rotation(self) -> np.ndarray:
        presets = {
            "none": np.eye(3, dtype=np.float64),
            "yaw_p90": R.from_euler("z", 90.0, degrees=True).as_matrix(),
            "yaw_n90": R.from_euler("z", -90.0, degrees=True).as_matrix(),
            "pitch_p90": R.from_euler("y", 90.0, degrees=True).as_matrix(),
            "pitch_n90": R.from_euler("y", -90.0, degrees=True).as_matrix(),
            "roll_p90": R.from_euler("x", 90.0, degrees=True).as_matrix(),
            "roll_n90": R.from_euler("x", -90.0, degrees=True).as_matrix(),
            "swap_xy": np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64),
            "swap_xz": np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=np.float64),
            "swap_yz": np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64),
        }
        return presets.get(self.link_cam_axis_mode, presets["none"])

    def _should_apply_camera_debug(self, target: str) -> bool:
        if not self.link_cam_debug_enable:
            return False
        return self.camera_debug_target in ("all", target)

    def _apply_camera_debug(self, pose: sapien.Pose, target: str) -> sapien.Pose:
        if not self._should_apply_camera_debug(target):
            return pose
        rot_base = R.from_quat(quat_wxyz_to_xyzw(pose.q)).as_matrix()
        rot_axis = self._camera_axis_mode_rotation()
        rot_delta = R.from_euler("xyz", self.link_cam_debug_rot_xyz_deg, degrees=True).as_matrix()
        rot_new = rot_base @ rot_axis @ rot_delta
        forward, right, up = self.link_cam_debug_shift_fru.tolist()
        shift_local = np.array([right, up, -forward], dtype=np.float64)
        pos_new = np.asarray(pose.p, dtype=np.float64) + rot_base @ shift_local
        return sapien.Pose(pos_new, quat_xyzw_to_wxyz(R.from_matrix(rot_new).as_quat()))

    def get_head_camera_pose(self) -> sapien.Pose:
        if self._head_camera_link is None:
            raise RuntimeError("Head camera link is unavailable.")
        link_pose = self._head_camera_link.get_entity_pose()
        local_pose = sapien.Pose(self.head_camera_local_pos, self.head_camera_local_quat_wxyz)
        head_pose = matrix_to_pose(pose_to_matrix(link_pose) @ pose_to_matrix(local_pose))
        return self._apply_camera_debug(head_pose, "head")

    def get_wrist_camera_pose(self, side: str) -> sapien.Pose:
        if side not in ("left", "right"):
            raise ValueError(f"Unsupported side: {side}")
        link = self._left_wrist_camera_link if side == "left" else self._right_wrist_camera_link
        if link is None:
            raise RuntimeError(f"{side} wrist camera link is unavailable.")
        link_pose = link.get_entity_pose()
        local_pose = sapien.Pose(self.wrist_camera_local_pos, self.wrist_camera_local_quat_wxyz)
        pose = matrix_to_pose(pose_to_matrix(link_pose) @ pose_to_matrix(local_pose))
        return self._apply_camera_debug(pose, f"{side}_wrist")

    def _camera_cv_rotation(self) -> np.ndarray:
        return CV_TO_WORLD_CAMERA_PRESETS.get(self.camera_cv_axis_mode, CV_TO_WORLD_CAMERA_PRESETS["legacy_r1"])

    def print_head_camera_summary(self) -> None:
        head_pose = self.get_head_camera_pose()
        rot = R.from_quat(quat_wxyz_to_xyzw(head_pose.q)).as_matrix()
        right = rot[:, 0]
        up = rot[:, 1]
        back = rot[:, 2]
        forward = -back
        base_rot = R.from_quat(quat_wxyz_to_xyzw(self._base_pose.q)).as_matrix()
        base_forward = base_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        base_left = base_rot @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        base_up = base_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        print(
            "[camera-debug] "
            f"head_link={self._head_camera_link.get_name() if self._head_camera_link is not None else 'None'} "
            f"camera_cv_axis_mode={self.camera_cv_axis_mode} "
            f"pose_p={format_vec3(head_pose.p)} "
            f"pose_q_wxyz={np.asarray(head_pose.q, dtype=np.float64).round(6).tolist()}"
        )
        print(
            "[camera-debug] "
            f"forward={format_vec3(forward)} right={format_vec3(right)} up={format_vec3(up)} "
            f"base_q_wxyz={np.asarray(self._base_pose.q, dtype=np.float64).round(6).tolist()} "
            f"head_local_q_wxyz={self.head_camera_local_quat_wxyz.round(6).tolist()}"
        )
        print(
            "[camera-debug] "
            f"robot_base_p={format_vec3(self._base_pose.p)} "
            f"robot_forward={format_vec3(base_forward)} "
            f"robot_left={format_vec3(base_left)} "
            f"robot_up={format_vec3(base_up)}"
        )

    def print_wrist_camera_summary(self, side: str) -> None:
        link = self._left_wrist_camera_link if side == "left" else self._right_wrist_camera_link
        if link is None:
            print(f"[camera-debug] {side}_wrist_link=None")
            return
        pose = self.get_wrist_camera_pose(side)
        rot = R.from_quat(quat_wxyz_to_xyzw(pose.q)).as_matrix()
        right = rot[:, 0]
        up = rot[:, 1]
        back = rot[:, 2]
        forward = -back
        print(
            "[camera-debug] "
            f"{side}_wrist_link={link.get_name()} pose_p={format_vec3(pose.p)} "
            f"pose_q_wxyz={np.asarray(pose.q, dtype=np.float64).round(6).tolist()}"
        )
        print(
            "[camera-debug] "
            f"{side}_wrist_forward={format_vec3(forward)} right={format_vec3(right)} up={format_vec3(up)} "
            f"wrist_local_q_wxyz={self.wrist_camera_local_quat_wxyz.round(6).tolist()}"
        )

    def get_current_tcp_pose(self, arm: str) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("Robot is unavailable.")
        pose = self.robot.get_left_tcp_pose() if arm == "left" else self.robot.get_right_tcp_pose()
        return np.asarray(pose, dtype=np.float64)

    def snapshot_robot_state(self) -> Dict[str, np.ndarray]:
        if self.robot is None:
            raise RuntimeError("Robot is unavailable.")
        entity = self.robot.left_entity
        return {
            "qpos": np.asarray(entity.get_qpos(), dtype=np.float64).copy(),
            "qvel": np.asarray(entity.get_qvel(), dtype=np.float64).copy(),
        }

    def restore_robot_state(self, state: Dict[str, np.ndarray]) -> None:
        if self.robot is None:
            raise RuntimeError("Robot is unavailable.")
        entity = self.robot.left_entity
        qpos = np.asarray(state["qpos"], dtype=np.float64).copy()
        qvel = np.asarray(state["qvel"], dtype=np.float64).copy()
        entity.set_qpos(qpos)
        entity.set_qvel(qvel)
        self._set_fixed_torso()
        self.scene.update_render()

    def align_target_orientation(self, arm: str, target_pose_world: np.ndarray) -> np.ndarray:
        target = np.asarray(target_pose_world, dtype=np.float64).reshape(7).copy()
        mode = self.debug_force_orientation
        if mode == "none":
            return target
        if mode == "current_tcp":
            target[3:] = normalize_quat_wxyz(self.get_current_tcp_pose(arm)[3:])
            return target
        if mode in ("base", "robot_forward_y"):
            target[3:] = normalize_quat_wxyz(self._base_pose.q)
            return target
        raise ValueError(f"Unsupported debug force orientation mode: {mode}")

    def apply_target_world_offset(self, target_pose_world: np.ndarray) -> np.ndarray:
        target = np.asarray(target_pose_world, dtype=np.float64).reshape(7).copy()
        target[:3] += self.target_world_offset_xyz
        target[2] += self.target_world_z_offset
        return target

    def world_pose_to_base_pose(self, pose_world: np.ndarray) -> np.ndarray:
        pose_world = np.asarray(pose_world, dtype=np.float64).reshape(7)
        base_inv = invert_pose_matrix(pose_to_matrix(self._base_pose))
        pose_base = base_inv @ pose_to_matrix(sapien.Pose(pose_world[:3], normalize_quat_wxyz(pose_world[3:])))
        return np.concatenate([pose_base[:3, 3], quat_xyzw_to_wxyz(R.from_matrix(orthonormalize_rotation(pose_base[:3, :3])).as_quat())])

    def _set_axis_actor_pose(self, actor, pose_world: Optional[np.ndarray]) -> None:
        if actor is None:
            return
        if pose_world is None:
            actor.set_pose(HIDDEN_DEBUG_POSE)
            return
        pose_world = np.asarray(pose_world, dtype=np.float64).reshape(7)
        if not np.isfinite(pose_world).all():
            actor.set_pose(HIDDEN_DEBUG_POSE)
            return
        actor.set_pose(sapien.Pose(pose_world[:3], normalize_quat_wxyz(pose_world[3:])))

    def update_target_axis_visuals(self, left_pose_world: Optional[np.ndarray], right_pose_world: Optional[np.ndarray]) -> None:
        self._set_axis_actor_pose(self._left_target_axis_actor, left_pose_world)
        self._set_axis_actor_pose(self._right_target_axis_actor, right_pose_world)

    def update_robot_link_cameras(self) -> None:
        head_pose = self.get_head_camera_pose()
        self.zed_camera.set_entity_pose(head_pose)
        if self._left_wrist_camera_link is not None:
            self.left_wrist_camera.set_entity_pose(self.get_wrist_camera_pose("left"))
        if self._right_wrist_camera_link is not None:
            self.right_wrist_camera.set_entity_pose(self.get_wrist_camera_pose("right"))
        if self.third_person_view:
            self.third_camera.set_entity_pose(self._build_third_camera_pose(head_pose))

    def _build_third_camera_pose(self, head_pose: sapien.Pose) -> sapien.Pose:
        base_rot = R.from_quat(quat_wxyz_to_xyzw(self._base_pose.q)).as_matrix()
        robot_forward = base_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        eye = np.asarray(head_pose.p, dtype=np.float64) + 1.0 * robot_forward
        eye = eye + 0.18 * world_up
        target = np.asarray(head_pose.p, dtype=np.float64)
        forward = target - eye
        if np.linalg.norm(forward) < 1e-12:
            forward = -robot_forward
        left = np.cross(world_up, forward)
        if np.linalg.norm(left) < 1e-12:
            left = base_rot @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return camera_pose_from_forward_left(eye, forward, left)

    def step_scene(self, steps: int = 1) -> None:
        for _ in range(int(steps)):
            self._set_fixed_torso()
            self.scene.step()
        self.scene.update_render()
        if self.viewer is not None and not self.viewer.closed:
            self.viewer.render()
            if self.viewer_frame_delay > 0.0:
                time.sleep(self.viewer_frame_delay)

    def hold_viewer(self) -> None:
        if self.viewer is None or self.viewer.closed or not self.viewer_wait_at_end:
            return
        print("Viewer debug mode active. Close the viewer window to exit.")
        while not self.viewer.closed:
            self.scene.update_render()
            self.viewer.render()
            time.sleep(max(self.viewer_frame_delay, 0.01))

    def camera_to_world_pose(self, position_cam: np.ndarray, rotation_cam: np.ndarray) -> np.ndarray:
        head_pose = self.get_head_camera_pose()
        rot_world_from_head = R.from_quat(quat_wxyz_to_xyzw(head_pose.q)).as_matrix()
        cam_cv_to_local = self._camera_cv_rotation()
        pos_world = np.asarray(head_pose.p, dtype=np.float64) + rot_world_from_head @ (cam_cv_to_local @ position_cam)
        rot_world = rot_world_from_head @ cam_cv_to_local @ orthonormalize_rotation(rotation_cam)
        quat_world = quat_xyzw_to_wxyz(R.from_matrix(rot_world).as_quat())
        return np.concatenate([pos_world, quat_world]).astype(np.float64)

    def plan_path(self, arm: str, target_pose_world: np.ndarray) -> Optional[Dict]:
        if self.robot is None:
            return None
        pose_list = target_pose_world.tolist()
        if arm == "left":
            return self.robot.left_plan_path(pose_list)
        if arm == "right":
            return self.robot.right_plan_path(pose_list)
        raise ValueError(f"Unsupported arm: {arm}")

    def execute_plans(self, left_plan: Optional[Dict], right_plan: Optional[Dict]) -> Tuple[str, str]:
        left_status = self._plan_status(left_plan)
        right_status = self._plan_status(right_plan)

        left_ok = left_status == "Success"
        right_ok = right_status == "Success"
        left_pos = left_plan["position"] if left_ok else None
        left_vel = left_plan["velocity"] if left_ok else None
        right_pos = right_plan["position"] if right_ok else None
        right_vel = right_plan["velocity"] if right_ok else None

        left_idx = 0
        right_idx = 0
        left_n = int(left_pos.shape[0]) if left_ok else 0
        right_n = int(right_pos.shape[0]) if right_ok else 0

        while left_idx < left_n or right_idx < right_n:
            if left_ok and left_idx < left_n and (not right_ok or left_idx / left_n <= right_idx / max(right_n, 1)):
                self.robot.set_arm_joints(left_pos[left_idx], left_vel[left_idx], "left")
                left_idx += 1
            if right_ok and right_idx < right_n and (not left_ok or right_idx / right_n <= left_idx / max(left_n, 1)):
                self.robot.set_arm_joints(right_pos[right_idx], right_vel[right_idx], "right")
                right_idx += 1
            self.step_scene(steps=1)

        self.step_scene(steps=4)
        return left_status, right_status

    def set_grippers(self, left_value: Optional[float], right_value: Optional[float]) -> None:
        if self.robot is None:
            return
        if left_value is not None:
            self.robot.set_gripper(float(left_value), "left", gripper_eps=0.0)
        if right_value is not None:
            self.robot.set_gripper(float(right_value), "right", gripper_eps=0.0)
        self.step_scene(steps=2)

    def capture_camera(self, camera) -> Tuple[np.ndarray, np.ndarray]:
        self.scene.update_render()
        camera.take_picture()
        rgba = camera.get_picture("Color")
        rgb = (rgba * 255).clip(0, 255).astype(np.uint8)[..., :3]
        position = camera.get_picture("Position")
        depth = (-position[..., 2] * 1000.0).astype(np.float64)
        return rgb, depth

    def run_camera_rotation_sweep(self, output_dir: Path, overlay_lines: Sequence[str]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        base_enable = self.link_cam_debug_enable
        base_rot = self.link_cam_debug_rot_xyz_deg.copy()
        cases: List[Tuple[float, float, float]] = []
        for rx in self.camera_sweep_steps_deg:
            for ry in self.camera_sweep_steps_deg:
                for rz in self.camera_sweep_steps_deg:
                    cases.append((float(rx), float(ry), float(rz)))

        try:
            self.link_cam_debug_enable = True
            for case_idx, (rx, ry, rz) in enumerate(cases):
                self.link_cam_debug_rot_xyz_deg = base_rot + np.array([rx, ry, rz], dtype=np.float64)
                self.update_robot_link_cameras()
                self.step_scene(steps=1)
                head_pose = self.get_head_camera_pose()
                pose_line = (
                    f"p={format_vec3(head_pose.p)} "
                    f"q={np.asarray(head_pose.q, dtype=np.float64).round(4).tolist()}"
                )
                case_name = f"{case_idx:03d}_rx{int(rx):+03d}_ry{int(ry):+03d}_rz{int(rz):+03d}"
                zed_rgb, _ = self.capture_camera(self.zed_camera)
                zed_img = overlay_text(zed_rgb, list(overlay_lines) + [case_name, pose_line])
                cv2.imwrite(str(output_dir / f"{case_name}_zed.png"), zed_img)
                if self._left_wrist_camera_link is not None:
                    left_rgb, _ = self.capture_camera(self.left_wrist_camera)
                    left_img = overlay_text(left_rgb, list(overlay_lines) + [case_name, "left_wrist"])
                    cv2.imwrite(str(output_dir / f"{case_name}_left_wrist.png"), left_img)
                if self._right_wrist_camera_link is not None:
                    right_rgb, _ = self.capture_camera(self.right_wrist_camera)
                    right_img = overlay_text(right_rgb, list(overlay_lines) + [case_name, "right_wrist"])
                    cv2.imwrite(str(output_dir / f"{case_name}_right_wrist.png"), right_img)
                if self.third_person_view:
                    third_rgb, _ = self.capture_camera(self.third_camera)
                    third_img = overlay_text(third_rgb, list(overlay_lines) + [case_name, pose_line])
                    cv2.imwrite(str(output_dir / f"{case_name}_third.png"), third_img)
                print(f"[camera-sweep] saved {case_name}")
        finally:
            self.link_cam_debug_enable = base_enable
            self.link_cam_debug_rot_xyz_deg = base_rot
            self.update_robot_link_cameras()
            self.step_scene(steps=1)

    def build_orientation_candidates(
        self,
        base_quat_wxyz: Sequence[float],
        steps_deg: Sequence[float],
    ) -> List[Tuple[str, np.ndarray]]:
        base_rot = R.from_quat(quat_wxyz_to_xyzw(base_quat_wxyz))
        cases: List[Tuple[str, np.ndarray]] = []
        seen = set()
        for rx in steps_deg:
            for ry in steps_deg:
                for rz in steps_deg:
                    delta = R.from_euler("xyz", [float(rx), float(ry), float(rz)], degrees=True)
                    quat = quat_xyzw_to_wxyz((base_rot * delta).as_quat())
                    key = tuple(np.round(quat, 6).tolist())
                    if key in seen:
                        continue
                    seen.add(key)
                    label = f"rx{int(rx):+03d}_ry{int(ry):+03d}_rz{int(rz):+03d}"
                    cases.append((label, quat))
        return cases

    def run_orientation_sweep(
        self,
        output_dir: Path,
        frame_idx: int,
        arm: str,
        position_world: np.ndarray,
        target_quat_wxyz: np.ndarray,
        base_quat_wxyz: np.ndarray,
        steps_deg: Sequence[float],
        execute_success: bool = False,
        reset_each_case: bool = True,
        save_images: bool = True,
    ) -> List[Dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results: List[Dict] = []
        current_tcp = self.get_current_tcp_pose(arm)
        initial_state = self.snapshot_robot_state()
        candidates = self.build_orientation_candidates(base_quat_wxyz, steps_deg)
        for case_idx, (label, quat_wxyz) in enumerate(candidates):
            if reset_each_case:
                self.restore_robot_state(initial_state)
                self.update_robot_link_cameras()
                self.step_scene(steps=1)
            pose_world = np.concatenate([np.asarray(position_world, dtype=np.float64), normalize_quat_wxyz(quat_wxyz)])
            if arm == "left":
                self.update_target_axis_visuals(pose_world, None)
            else:
                self.update_target_axis_visuals(None, pose_world)
            self.update_robot_link_cameras()
            self.step_scene(steps=1)
            plan = self.plan_path(arm, pose_world)
            status = self._plan_status(plan)
            rot_err_current_deg = quat_angle_deg_wxyz(current_tcp[3:], pose_world[3:])
            rot_err_target_deg = quat_angle_deg_wxyz(target_quat_wxyz, pose_world[3:])
            item = {
                "frame_idx": int(frame_idx),
                "arm": arm,
                "case_idx": int(case_idx),
                "label": label,
                "status": status,
                "position_world": np.asarray(position_world, dtype=np.float64),
                "quat_world_wxyz": np.asarray(pose_world[3:], dtype=np.float64),
                "rot_err_vs_current_tcp_deg": float(rot_err_current_deg),
                "rot_err_vs_target_deg": float(rot_err_target_deg),
            }
            if execute_success and status == "Success":
                if arm == "left":
                    left_status, right_status = self.execute_plans(plan, None)
                else:
                    left_status, right_status = self.execute_plans(None, plan)
                item["executed_status"] = left_status if arm == "left" else right_status
                actual_tcp = self.get_current_tcp_pose(arm)
                item["actual_tcp_pose_world"] = actual_tcp.copy()
                item["pos_err_after_execute_m"] = float(np.linalg.norm(actual_tcp[:3] - pose_world[:3]))
                item["pos_axis_err_after_execute_m"] = position_axis_errors(actual_tcp[:3], pose_world[:3])
                item["rot_err_after_execute_deg"] = float(quat_angle_deg_wxyz(actual_tcp[3:], pose_world[3:]))
                item["axis_err_after_execute_deg"] = axis_angle_errors_deg_wxyz(actual_tcp[3:], pose_world[3:])
                print(
                    f"[orientation-sweep-exec] frame={frame_idx:04d} arm={arm} case={label} "
                    f"executed={item['executed_status']} pos_err={item['pos_err_after_execute_m']:.3f}m "
                    f"rot_err={item['rot_err_after_execute_deg']:.1f}deg"
                )
                pe = item["pos_axis_err_after_execute_m"]
                print(
                    f"[orientation-sweep-exec-pos-axis] frame={frame_idx:04d} arm={arm} "
                    f"dx={pe['x']:+.3f}m dy={pe['y']:+.3f}m dz={pe['z']:+.3f}m"
                )
                ae = item["axis_err_after_execute_deg"]
                print(
                    f"[orientation-sweep-exec-axis] frame={frame_idx:04d} arm={arm} "
                    f"x={ae['x']:.1f}deg y={ae['y']:.1f}deg z={ae['z']:.1f}deg"
                )
            else:
                item["executed_status"] = "Skipped"
            results.append(item)
            print(
                f"[orientation-sweep] frame={frame_idx:04d} arm={arm} case={label} "
                f"status={status} rot_vs_current={rot_err_current_deg:.1f}deg rot_vs_target={rot_err_target_deg:.1f}deg"
            )
            if save_images:
                self.update_robot_link_cameras()
                status_suffix = f"{arm[0].upper()}{short_status(status)}"
                overlay_lines = [
                    f"Frame {frame_idx} arm={arm}",
                    f"case={label}",
                    f"status={status}",
                    f"executed={item['executed_status']}",
                    f"rot_vs_current={rot_err_current_deg:.1f}deg",
                    f"rot_vs_target={rot_err_target_deg:.1f}deg",
                ]
                if "pos_err_after_execute_m" in item:
                    overlay_lines.append(f"pos_err_exec={item['pos_err_after_execute_m']:.3f}m")
                if "pos_axis_err_after_execute_m" in item:
                    pe = item["pos_axis_err_after_execute_m"]
                    overlay_lines.append(f"pos dx={pe['x']:+.3f} dy={pe['y']:+.3f} dz={pe['z']:+.3f}")
                if "axis_err_after_execute_deg" in item:
                    ae = item["axis_err_after_execute_deg"]
                    overlay_lines.append(f"axis x={ae['x']:.1f} y={ae['y']:.1f} z={ae['z']:.1f}")
                zed_rgb, _ = self.capture_camera(self.zed_camera)
                cv2.imwrite(str(output_dir / f"{case_idx:03d}_{label}_{status_suffix}_zed.png"), overlay_text(zed_rgb, overlay_lines))
                if self._left_wrist_camera_link is not None:
                    left_rgb, _ = self.capture_camera(self.left_wrist_camera)
                    cv2.imwrite(str(output_dir / f"{case_idx:03d}_{label}_{status_suffix}_left_wrist.png"), overlay_text(left_rgb, overlay_lines))
                if self._right_wrist_camera_link is not None:
                    right_rgb, _ = self.capture_camera(self.right_wrist_camera)
                    cv2.imwrite(str(output_dir / f"{case_idx:03d}_{label}_{status_suffix}_right_wrist.png"), overlay_text(right_rgb, overlay_lines))
                if self.third_person_view:
                    third_rgb, _ = self.capture_camera(self.third_camera)
                    cv2.imwrite(str(output_dir / f"{case_idx:03d}_{label}_{status_suffix}_third.png"), overlay_text(third_rgb, overlay_lines))

        with (output_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(make_json_safe(results), f, ensure_ascii=False, indent=2)
        return results

    def run_dual_orientation_sweep(
        self,
        output_dir: Path,
        frame_idx: int,
        left_position_world: np.ndarray,
        right_position_world: np.ndarray,
        left_target_quat_wxyz: np.ndarray,
        right_target_quat_wxyz: np.ndarray,
        left_base_quat_wxyz: np.ndarray,
        right_base_quat_wxyz: np.ndarray,
        steps_deg: Sequence[float],
        pair_mode: str = "paired",
        execute_success: bool = False,
        reset_each_case: bool = True,
        save_images: bool = True,
    ) -> List[Dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        results: List[Dict] = []
        initial_state = self.snapshot_robot_state()
        left_current_tcp = self.get_current_tcp_pose("left")
        right_current_tcp = self.get_current_tcp_pose("right")
        left_candidates = self.build_orientation_candidates(left_base_quat_wxyz, steps_deg)
        right_candidates = self.build_orientation_candidates(right_base_quat_wxyz, steps_deg)

        if pair_mode == "paired":
            case_pairs = []
            max_len = min(len(left_candidates), len(right_candidates))
            for idx in range(max_len):
                case_pairs.append((idx, left_candidates[idx], right_candidates[idx]))
        elif pair_mode == "cartesian":
            case_pairs = []
            idx = 0
            for left_case in left_candidates:
                for right_case in right_candidates:
                    case_pairs.append((idx, left_case, right_case))
                    idx += 1
        else:
            raise ValueError(f"Unsupported dual orientation sweep pair mode: {pair_mode}")

        for case_idx, left_case, right_case in case_pairs:
            if reset_each_case:
                self.restore_robot_state(initial_state)
                self.update_robot_link_cameras()
                self.step_scene(steps=1)

            left_label, left_quat = left_case
            right_label, right_quat = right_case
            left_pose_world = np.concatenate([np.asarray(left_position_world, dtype=np.float64), normalize_quat_wxyz(left_quat)])
            right_pose_world = np.concatenate([np.asarray(right_position_world, dtype=np.float64), normalize_quat_wxyz(right_quat)])

            self.update_target_axis_visuals(left_pose_world, right_pose_world)
            self.update_robot_link_cameras()
            self.step_scene(steps=1)

            left_plan = self.plan_path("left", left_pose_world)
            right_plan = self.plan_path("right", right_pose_world)
            left_status = self._plan_status(left_plan)
            right_status = self._plan_status(right_plan)
            item = {
                "frame_idx": int(frame_idx),
                "case_idx": int(case_idx),
                "left_label": left_label,
                "right_label": right_label,
                "left_status": left_status,
                "right_status": right_status,
                "left_position_world": np.asarray(left_position_world, dtype=np.float64),
                "right_position_world": np.asarray(right_position_world, dtype=np.float64),
                "left_quat_world_wxyz": np.asarray(left_pose_world[3:], dtype=np.float64),
                "right_quat_world_wxyz": np.asarray(right_pose_world[3:], dtype=np.float64),
                "left_rot_err_vs_current_tcp_deg": float(quat_angle_deg_wxyz(left_current_tcp[3:], left_pose_world[3:])),
                "right_rot_err_vs_current_tcp_deg": float(quat_angle_deg_wxyz(right_current_tcp[3:], right_pose_world[3:])),
                "left_rot_err_vs_target_deg": float(quat_angle_deg_wxyz(left_target_quat_wxyz, left_pose_world[3:])),
                "right_rot_err_vs_target_deg": float(quat_angle_deg_wxyz(right_target_quat_wxyz, right_pose_world[3:])),
            }
            if execute_success and (left_status == "Success" or right_status == "Success"):
                exec_left_status, exec_right_status = self.execute_plans(left_plan, right_plan)
                item["executed_left_status"] = exec_left_status
                item["executed_right_status"] = exec_right_status
                actual_left_tcp = self.get_current_tcp_pose("left")
                actual_right_tcp = self.get_current_tcp_pose("right")
                if left_status == "Success":
                    item["left_pos_err_after_execute_m"] = float(np.linalg.norm(actual_left_tcp[:3] - left_pose_world[:3]))
                    item["left_pos_axis_err_after_execute_m"] = position_axis_errors(actual_left_tcp[:3], left_pose_world[:3])
                    item["left_rot_err_after_execute_deg"] = float(quat_angle_deg_wxyz(actual_left_tcp[3:], left_pose_world[3:]))
                    item["left_axis_err_after_execute_deg"] = axis_angle_errors_deg_wxyz(actual_left_tcp[3:], left_pose_world[3:])
                if right_status == "Success":
                    item["right_pos_err_after_execute_m"] = float(np.linalg.norm(actual_right_tcp[:3] - right_pose_world[:3]))
                    item["right_pos_axis_err_after_execute_m"] = position_axis_errors(actual_right_tcp[:3], right_pose_world[:3])
                    item["right_rot_err_after_execute_deg"] = float(quat_angle_deg_wxyz(actual_right_tcp[3:], right_pose_world[3:]))
                    item["right_axis_err_after_execute_deg"] = axis_angle_errors_deg_wxyz(actual_right_tcp[3:], right_pose_world[3:])
                print(
                    f"[orientation-sweep-exec] frame={frame_idx:04d} both case={case_idx:03d} "
                    f"left={exec_left_status} right={exec_right_status} "
                    f"left_pos_err={item.get('left_pos_err_after_execute_m', float('nan')):.3f}m "
                    f"right_pos_err={item.get('right_pos_err_after_execute_m', float('nan')):.3f}m"
                )
                if "left_pos_axis_err_after_execute_m" in item:
                    lpe = item["left_pos_axis_err_after_execute_m"]
                    print(
                        f"[orientation-sweep-exec-pos-axis] frame={frame_idx:04d} left "
                        f"dx={lpe['x']:+.3f}m dy={lpe['y']:+.3f}m dz={lpe['z']:+.3f}m"
                    )
                if "right_pos_axis_err_after_execute_m" in item:
                    rpe = item["right_pos_axis_err_after_execute_m"]
                    print(
                        f"[orientation-sweep-exec-pos-axis] frame={frame_idx:04d} right "
                        f"dx={rpe['x']:+.3f}m dy={rpe['y']:+.3f}m dz={rpe['z']:+.3f}m"
                    )
                if "left_axis_err_after_execute_deg" in item:
                    la = item["left_axis_err_after_execute_deg"]
                    print(
                        f"[orientation-sweep-exec-axis] frame={frame_idx:04d} left "
                        f"x={la['x']:.1f}deg y={la['y']:.1f}deg z={la['z']:.1f}deg"
                    )
                if "right_axis_err_after_execute_deg" in item:
                    ra = item["right_axis_err_after_execute_deg"]
                    print(
                        f"[orientation-sweep-exec-axis] frame={frame_idx:04d} right "
                        f"x={ra['x']:.1f}deg y={ra['y']:.1f}deg z={ra['z']:.1f}deg"
                    )
            else:
                item["executed_left_status"] = "Skipped"
                item["executed_right_status"] = "Skipped"

            results.append(item)
            print(
                f"[orientation-sweep] frame={frame_idx:04d} both case={case_idx:03d} "
                f"left={left_label}:{left_status} right={right_label}:{right_status}"
            )

            if save_images:
                self.update_robot_link_cameras()
                status_suffix = f"L{short_status(left_status)}_R{short_status(right_status)}"
                overlay_lines = [
                    f"Frame {frame_idx} both case={case_idx:03d}",
                    f"L {left_label}: {left_status}",
                    f"R {right_label}: {right_status}",
                    f"exec L={item['executed_left_status']} R={item['executed_right_status']}",
                ]
                if "left_pos_err_after_execute_m" in item or "right_pos_err_after_execute_m" in item:
                    left_err = item.get("left_pos_err_after_execute_m", float("nan"))
                    right_err = item.get("right_pos_err_after_execute_m", float("nan"))
                    overlay_lines.append(f"exec err L={left_err:.3f}m R={right_err:.3f}m")
                if "left_pos_axis_err_after_execute_m" in item:
                    lpe = item["left_pos_axis_err_after_execute_m"]
                    overlay_lines.append(f"L pos dx={lpe['x']:+.3f} dy={lpe['y']:+.3f} dz={lpe['z']:+.3f}")
                if "right_pos_axis_err_after_execute_m" in item:
                    rpe = item["right_pos_axis_err_after_execute_m"]
                    overlay_lines.append(f"R pos dx={rpe['x']:+.3f} dy={rpe['y']:+.3f} dz={rpe['z']:+.3f}")
                if "left_axis_err_after_execute_deg" in item:
                    la = item["left_axis_err_after_execute_deg"]
                    overlay_lines.append(f"L axis x={la['x']:.1f} y={la['y']:.1f} z={la['z']:.1f}")
                if "right_axis_err_after_execute_deg" in item:
                    ra = item["right_axis_err_after_execute_deg"]
                    overlay_lines.append(f"R axis x={ra['x']:.1f} y={ra['y']:.1f} z={ra['z']:.1f}")
                zed_rgb, _ = self.capture_camera(self.zed_camera)
                cv2.imwrite(str(output_dir / f"{case_idx:03d}_{status_suffix}_zed.png"), overlay_text(zed_rgb, overlay_lines))
                if self._left_wrist_camera_link is not None:
                    left_rgb, _ = self.capture_camera(self.left_wrist_camera)
                    cv2.imwrite(str(output_dir / f"{case_idx:03d}_{status_suffix}_left_wrist.png"), overlay_text(left_rgb, overlay_lines))
                if self._right_wrist_camera_link is not None:
                    right_rgb, _ = self.capture_camera(self.right_wrist_camera)
                    cv2.imwrite(str(output_dir / f"{case_idx:03d}_{status_suffix}_right_wrist.png"), overlay_text(right_rgb, overlay_lines))
                if self.third_person_view:
                    third_rgb, _ = self.capture_camera(self.third_camera)
                    cv2.imwrite(str(output_dir / f"{case_idx:03d}_{status_suffix}_third.png"), overlay_text(third_rgb, overlay_lines))

        with (output_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(make_json_safe(results), f, ensure_ascii=False, indent=2)
        return results

    @staticmethod
    def _plan_status(plan: Optional[Dict]) -> str:
        if not isinstance(plan, dict):
            return "Missing"
        return str(plan.get("status", "Missing"))

    def print_frame_debug(
        self,
        frame_idx: int,
        arm: str,
        target_cam: SideFrameTarget,
        target_world: Optional[np.ndarray],
        plan: Optional[Dict],
    ) -> None:
        status = self._plan_status(plan)
        if not target_cam.valid or target_world is None:
            print(f"[debug frame {frame_idx:04d}] {arm}: invalid target, status={status}")
            return
        current_tcp = self.get_current_tcp_pose(arm)
        target_base = self.world_pose_to_base_pose(target_world)
        current_base = self.world_pose_to_base_pose(current_tcp)
        pos_err = float(np.linalg.norm(target_world[:3] - current_tcp[:3]))
        pos_axis_err = position_axis_errors(current_tcp[:3], target_world[:3])
        base_pos_axis_err = position_axis_errors(current_base[:3], target_base[:3])
        rot_err_deg = quat_angle_deg_wxyz(current_tcp[3:], target_world[3:])
        axis_err_deg = axis_angle_errors_deg_wxyz(current_tcp[3:], target_world[3:])
        target_axes = rotation_axes_world_wxyz(target_world[3:])
        current_axes = rotation_axes_world_wxyz(current_tcp[3:])
        print(
            f"[debug frame {frame_idx:04d}] {arm}: "
            f"cam_pos={format_vec3(target_cam.position_cam)} "
            f"world_pos={format_vec3(target_world[:3])} "
            f"current_tcp={format_vec3(current_tcp[:3])} "
            f"pos_err={pos_err:.3f}m rot_err={rot_err_deg:.1f}deg "
            f"status={status}"
        )
        print(
            f"[debug frame {frame_idx:04d}] {arm}: "
            f"target_q_wxyz={np.asarray(target_world[3:], dtype=np.float64).round(6).tolist()} "
            f"current_q_wxyz={np.asarray(current_tcp[3:], dtype=np.float64).round(6).tolist()}"
        )
        print(
            f"[debug frame {frame_idx:04d}] {arm}: "
            f"world_dxyz=[{pos_axis_err['x']:+.3f}, {pos_axis_err['y']:+.3f}, {pos_axis_err['z']:+.3f}]m "
            f"axis_err=[x={axis_err_deg['x']:.1f}, y={axis_err_deg['y']:.1f}, z={axis_err_deg['z']:.1f}]deg"
        )
        print(
            f"[debug frame {frame_idx:04d}] {arm}: "
            f"target_base={format_vec3(target_base[:3])} "
            f"current_base={format_vec3(current_base[:3])} "
            f"base_dxyz=[{base_pos_axis_err['x']:+.3f}, {base_pos_axis_err['y']:+.3f}, {base_pos_axis_err['z']:+.3f}]m"
        )
        print(
            f"[debug frame {frame_idx:04d}] {arm}: "
            f"target_axes x={format_vec3(target_axes['x'])} y={format_vec3(target_axes['y'])} z={format_vec3(target_axes['z'])}"
        )
        print(
            f"[debug frame {frame_idx:04d}] {arm}: "
            f"current_axes x={format_vec3(current_axes['x'])} y={format_vec3(current_axes['y'])} z={format_vec3(current_axes['z'])}"
        )


def look_at_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> sapien.Pose:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward = forward / max(np.linalg.norm(forward), 1e-12)
    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-12)
    camera_up = np.cross(right, forward)
    camera_up = camera_up / max(np.linalg.norm(camera_up), 1e-12)

    rot = np.column_stack([right, camera_up, -forward])
    rot = orthonormalize_rotation(rot)
    quat = quat_xyzw_to_wxyz(R.from_matrix(rot).as_quat())
    return sapien.Pose(eye, quat)


def build_gripper_mapper(
    series: np.ndarray,
    fixed_value: Optional[float],
    close_percentile: float,
    open_percentile: float,
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    if fixed_value is not None:
        return float(np.clip(fixed_value, 0.0, 1.0)), None
    if series.size == 0:
        return 0.8, None
    close_dist = float(np.percentile(series, close_percentile))
    open_dist = float(np.percentile(series, open_percentile))
    if abs(open_dist - close_dist) < 1e-6:
        return 0.8, None
    return None, (close_dist, open_dist)


def map_gripper_value(
    finger_distance: Optional[float],
    fixed_value: Optional[float],
    distance_range: Optional[Tuple[float, float]],
    fallback: float = 0.8,
) -> float:
    if fixed_value is not None:
        return float(np.clip(fixed_value, 0.0, 1.0))
    if finger_distance is None or distance_range is None or not math.isfinite(float(finger_distance)):
        return fallback
    close_dist, open_dist = distance_range
    value = (float(finger_distance) - close_dist) / max(open_dist - close_dist, 1e-6)
    return float(np.clip(value, 0.0, 1.0))


def overlay_text(rgb: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    y = 26
    for line in lines:
        cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 40), 1, cv2.LINE_AA)
        y += 24
    return bgr


def parse_optional_base_pose(values: Optional[Sequence[float]]) -> Optional[List[float]]:
    if values is None:
        return None
    if len(values) != 7:
        raise ValueError("--robot_base_pose expects 7 numbers: x y z qw qx qy qz")
    return [float(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay hand-retargeted gripper poses in RoboTwin R1.")
    parser.add_argument("--input_npz", type=Path, required=True, help="hand_detections_*.npz or *_with_gripper.npz")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for videos, frames, and world targets")
    parser.add_argument("--robot_config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--fovy_deg", type=float, default=90.0)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=-1)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--arms", choices=["left", "right", "both"], default="both")
    parser.add_argument("--pose_source", choices=["gripper", "retreat"], default="gripper")
    parser.add_argument("--retreat_distance", type=float, default=0.11)
    parser.add_argument("--thumb_tip_idx", type=int, default=4)
    parser.add_argument("--index_tip_idx", type=int, default=8)
    parser.add_argument("--index_joint_idx", type=int, default=6)
    parser.add_argument("--left_gripper_value", type=float, default=None)
    parser.add_argument("--right_gripper_value", type=float, default=None)
    parser.add_argument("--gripper_close_percentile", type=float, default=5.0)
    parser.add_argument("--gripper_open_percentile", type=float, default=95.0)
    parser.add_argument("--torso_qpos", type=float, nargs=4, default=DEFAULT_TORSO_QPOS.tolist())
    parser.add_argument("--robot_base_pose", type=float, nargs=7, default=None, metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"))
    parser.add_argument("--third_person_view", type=int, default=1)
    parser.add_argument("--save_png_frames", type=int, default=0)
    parser.add_argument("--need_topp", type=int, default=1)
    parser.add_argument("--camera_cv_axis_mode", choices=sorted(CV_TO_WORLD_CAMERA_PRESETS.keys()), default="legacy_r1")
    parser.add_argument("--head_camera_local_pos", type=float, nargs=3, default=DEFAULT_HEAD_CAMERA_LOCAL_POS.tolist())
    parser.add_argument("--head_camera_local_quat_wxyz", type=float, nargs=4, default=DEFAULT_HEAD_CAMERA_LOCAL_QUAT_WXYZ.tolist())
    parser.add_argument("--wrist_camera_local_pos", type=float, nargs=3, default=DEFAULT_WRIST_CAMERA_LOCAL_POS.tolist())
    parser.add_argument("--wrist_camera_local_quat_wxyz", type=float, nargs=4, default=DEFAULT_WRIST_CAMERA_LOCAL_QUAT_WXYZ.tolist())
    parser.add_argument("--enable_viewer", type=int, default=0)
    parser.add_argument("--viewer_frame_delay", type=float, default=0.0, help="Seconds to sleep after each viewer render for slow-motion debugging")
    parser.add_argument("--viewer_wait_at_end", type=int, default=0, help="Keep the viewer open after replay finishes until the window is closed")
    parser.add_argument("--link_cam_debug_enable", type=int, default=0)
    parser.add_argument("--camera_debug_target", choices=["head", "left_wrist", "right_wrist", "all"], default="head")
    parser.add_argument("--link_cam_axis_mode", type=str, default="none")
    parser.add_argument("--link_cam_debug_rot_x_deg", type=float, default=0.0)
    parser.add_argument("--link_cam_debug_rot_y_deg", type=float, default=0.0)
    parser.add_argument("--link_cam_debug_rot_z_deg", type=float, default=0.0)
    parser.add_argument("--link_cam_debug_forward", type=float, default=0.0)
    parser.add_argument("--link_cam_debug_right", type=float, default=0.0)
    parser.add_argument("--link_cam_debug_up", type=float, default=0.0)
    parser.add_argument("--debug_mode", type=int, default=0)
    parser.add_argument("--debug_frame_limit", type=int, default=5)
    parser.add_argument("--debug_force_orientation", choices=["none", "current_tcp", "base", "robot_forward_y"], default="none")
    parser.add_argument("--debug_visualize_targets", type=int, default=1)
    parser.add_argument("--debug_target_axis_length", type=float, default=0.08)
    parser.add_argument("--debug_target_axis_thickness", type=float, default=0.004)
    parser.add_argument("--target_world_offset_xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"), help="Meters to add to every world-space target position before planning")
    parser.add_argument("--target_world_z_offset", type=float, default=0.0, help="Meters to add to every world-space target z position")
    parser.add_argument("--target_world_offset_z_sweep_enable", type=int, default=0)
    parser.add_argument("--target_world_offset_z_sweep_start", type=float, default=-0.1)
    parser.add_argument("--target_world_offset_z_sweep_end", type=float, default=0.5)
    parser.add_argument("--target_world_offset_z_sweep_step", type=float, default=0.05)
    parser.add_argument("--disable_table", type=int, default=1)
    parser.add_argument("--camera_sweep_enable", type=int, default=0)
    parser.add_argument("--camera_sweep_steps_deg", type=float, nargs="+", default=[-180.0, -90.0, 0.0, 90.0])
    parser.add_argument("--orientation_sweep_enable", type=int, default=0)
    parser.add_argument("--orientation_sweep_arm", choices=["left", "right", "both"], default="both")
    parser.add_argument("--orientation_sweep_base", choices=["current_tcp", "base", "target"], default="current_tcp")
    parser.add_argument("--orientation_sweep_steps_deg", type=float, nargs="+", default=[-180.0, -90.0, 0.0, 90.0])
    parser.add_argument("--orientation_sweep_both_mode", choices=["paired", "cartesian"], default="paired")
    parser.add_argument("--orientation_sweep_execute", type=int, default=0)
    parser.add_argument("--orientation_sweep_reset_each_case", type=int, default=1)
    parser.add_argument("--orientation_sweep_save_images", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    if args.save_png_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    trajectory = HandRetargetTrajectory(
        npz_path=args.input_npz,
        retreat_distance=args.retreat_distance,
        thumb_tip_idx=args.thumb_tip_idx,
        index_tip_idx=args.index_tip_idx,
        index_joint_idx=args.index_joint_idx,
    )
    print(
        "[pose-source] "
        f"{trajectory.describe_pose_source('left', args.pose_source)}, "
        f"{trajectory.describe_pose_source('right', args.pose_source)}"
    )
    if not trajectory.has_stored_gripper_pose("left", args.pose_source) or not trajectory.has_stored_gripper_pose("right", args.pose_source):
        print(
            "[pose-source] missing stored gripper pose fields in input npz; "
            "recomputing from hand keypoints with the same formula used in compute_gripper_pose_from_npz.py"
        )

    fixed_left, left_range = build_gripper_mapper(
        trajectory.build_gripper_series("left"),
        fixed_value=args.left_gripper_value,
        close_percentile=args.gripper_close_percentile,
        open_percentile=args.gripper_open_percentile,
    )
    fixed_right, right_range = build_gripper_mapper(
        trajectory.build_gripper_series("right"),
        fixed_value=args.right_gripper_value,
        close_percentile=args.gripper_close_percentile,
        open_percentile=args.gripper_open_percentile,
    )

    renderer = HandRetargetR1Renderer(
        robot_config_path=args.robot_config,
        image_width=args.image_width,
        image_height=args.image_height,
        fovy_deg=args.fovy_deg,
        torso_qpos=args.torso_qpos,
        robot_base_pose_override=parse_optional_base_pose(args.robot_base_pose),
        third_person_view=bool(args.third_person_view),
        need_topp=bool(args.need_topp),
        link_cam_debug_enable=bool(args.link_cam_debug_enable),
        link_cam_axis_mode=args.link_cam_axis_mode,
        link_cam_debug_rot_xyz_deg=[
            args.link_cam_debug_rot_x_deg,
            args.link_cam_debug_rot_y_deg,
            args.link_cam_debug_rot_z_deg,
        ],
        link_cam_debug_shift_fru=[
            args.link_cam_debug_forward,
            args.link_cam_debug_right,
            args.link_cam_debug_up,
        ],
        camera_cv_axis_mode=args.camera_cv_axis_mode,
        head_camera_local_pos=args.head_camera_local_pos,
        head_camera_local_quat_wxyz=args.head_camera_local_quat_wxyz,
        wrist_camera_local_pos=args.wrist_camera_local_pos,
        wrist_camera_local_quat_wxyz=args.wrist_camera_local_quat_wxyz,
        camera_debug_target=args.camera_debug_target,
        enable_viewer=bool(args.enable_viewer),
        viewer_frame_delay=args.viewer_frame_delay,
        viewer_wait_at_end=bool(args.viewer_wait_at_end),
        debug_mode=bool(args.debug_mode),
        debug_force_orientation=args.debug_force_orientation,
        debug_visualize_targets=bool(args.debug_visualize_targets),
        debug_target_axis_length=args.debug_target_axis_length,
        debug_target_axis_thickness=args.debug_target_axis_thickness,
        target_world_offset_xyz=args.target_world_offset_xyz,
        target_world_z_offset=args.target_world_z_offset,
        disable_table=bool(args.disable_table),
        camera_sweep_enable=bool(args.camera_sweep_enable),
        camera_sweep_steps_deg=args.camera_sweep_steps_deg,
    )

    indices = list(range(max(args.frame_start, 0), trajectory.length if args.frame_end < 0 else min(args.frame_end + 1, trajectory.length), max(args.frame_stride, 1)))
    if args.debug_mode and args.debug_frame_limit > 0:
        indices = indices[: args.debug_frame_limit]
    if args.max_frames > 0:
        indices = indices[: args.max_frames]
    if not indices:
        raise ValueError("No frames selected.")

    if args.camera_sweep_enable:
        frame_idx = indices[0]
        use_left = args.arms in ("left", "both")
        use_right = args.arms in ("right", "both")
        left_target = trajectory.get_side_target("left", frame_idx, args.pose_source) if use_left else SideFrameTarget(False, np.full(3, np.nan), np.full((3, 3), np.nan), None)
        right_target = trajectory.get_side_target("right", frame_idx, args.pose_source) if use_right else SideFrameTarget(False, np.full(3, np.nan), np.full((3, 3), np.nan), None)
        left_world = None
        right_world = None
        if left_target.valid:
            left_world = renderer.camera_to_world_pose(left_target.position_cam, left_target.rotation_cam)
            left_world = renderer.apply_target_world_offset(left_world)
            left_world = renderer.align_target_orientation("left", left_world)
        if right_target.valid:
            right_world = renderer.camera_to_world_pose(right_target.position_cam, right_target.rotation_cam)
            right_world = renderer.apply_target_world_offset(right_world)
            right_world = renderer.align_target_orientation("right", right_world)
        renderer.update_target_axis_visuals(left_world, right_world)
        renderer.update_robot_link_cameras()
        renderer.step_scene(steps=1)
        sweep_dir = args.output_dir / "camera_sweep"
        renderer.run_camera_rotation_sweep(
            sweep_dir,
            overlay_lines=[
                f"Frame {frame_idx}",
                f"camera_cv_axis_mode={args.camera_cv_axis_mode}",
                f"head_local_q={np.asarray(args.head_camera_local_quat_wxyz, dtype=np.float64).round(4).tolist()}",
            ],
        )
        print(f"Saved camera sweep images to: {sweep_dir}")
        renderer.hold_viewer()
        return

    if args.orientation_sweep_enable:
        sweep_root = args.output_dir / "orientation_sweep"
        sweep_root.mkdir(parents=True, exist_ok=True)
        selected_arms = ("left", "right") if args.orientation_sweep_arm == "both" else (args.orientation_sweep_arm,)
        for frame_idx in indices:
            frame_dir = sweep_root / f"frame_{frame_idx:04d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            left_target = trajectory.get_side_target("left", frame_idx, args.pose_source)
            right_target = trajectory.get_side_target("right", frame_idx, args.pose_source)
            raw_targets = {"left": left_target, "right": right_target}
            world_targets: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
            for arm in selected_arms:
                target = raw_targets[arm]
                if not target.valid:
                    print(f"[orientation-sweep] frame={frame_idx:04d} arm={arm} skipped: invalid target")
                    continue
                pose_world = renderer.camera_to_world_pose(target.position_cam, target.rotation_cam)
                pose_world = renderer.apply_target_world_offset(pose_world)
                world_targets[arm] = pose_world
            renderer.update_target_axis_visuals(world_targets["left"], world_targets["right"])
            renderer.update_robot_link_cameras()
            renderer.step_scene(steps=1)
            if args.orientation_sweep_arm == "both":
                if world_targets["left"] is None or world_targets["right"] is None:
                    print(f"[orientation-sweep] frame={frame_idx:04d} both skipped: one or both targets invalid")
                    continue
                if args.orientation_sweep_base == "current_tcp":
                    left_base_quat = renderer.get_current_tcp_pose("left")[3:]
                    right_base_quat = renderer.get_current_tcp_pose("right")[3:]
                elif args.orientation_sweep_base == "base":
                    left_base_quat = renderer._base_pose.q
                    right_base_quat = renderer._base_pose.q
                elif args.orientation_sweep_base == "target":
                    left_base_quat = world_targets["left"][3:]
                    right_base_quat = world_targets["right"][3:]
                else:
                    raise ValueError(f"Unsupported orientation sweep base: {args.orientation_sweep_base}")
                both_dir = frame_dir / "both"
                results = renderer.run_dual_orientation_sweep(
                    output_dir=both_dir,
                    frame_idx=frame_idx,
                    left_position_world=world_targets["left"][:3],
                    right_position_world=world_targets["right"][:3],
                    left_target_quat_wxyz=world_targets["left"][3:],
                    right_target_quat_wxyz=world_targets["right"][3:],
                    left_base_quat_wxyz=left_base_quat,
                    right_base_quat_wxyz=right_base_quat,
                    steps_deg=args.orientation_sweep_steps_deg,
                    pair_mode=args.orientation_sweep_both_mode,
                    execute_success=bool(args.orientation_sweep_execute),
                    reset_each_case=bool(args.orientation_sweep_reset_each_case),
                    save_images=bool(args.orientation_sweep_save_images),
                )
                both_success = sum(1 for item in results if item["left_status"] == "Success" and item["right_status"] == "Success")
                any_success = sum(1 for item in results if item["left_status"] == "Success" or item["right_status"] == "Success")
                print(
                    f"[orientation-sweep] frame={frame_idx:04d} both "
                    f"any_success={any_success}/{len(results)} "
                    f"dual_success={both_success}/{len(results)} saved_to={both_dir}"
                )
                continue
            for arm in selected_arms:
                pose_world = world_targets[arm]
                if pose_world is None:
                    continue
                if args.orientation_sweep_base == "current_tcp":
                    base_quat = renderer.get_current_tcp_pose(arm)[3:]
                elif args.orientation_sweep_base == "base":
                    base_quat = renderer._base_pose.q
                elif args.orientation_sweep_base == "target":
                    base_quat = pose_world[3:]
                else:
                    raise ValueError(f"Unsupported orientation sweep base: {args.orientation_sweep_base}")
                arm_dir = frame_dir / arm
                results = renderer.run_orientation_sweep(
                    output_dir=arm_dir,
                    frame_idx=frame_idx,
                    arm=arm,
                    position_world=pose_world[:3],
                    target_quat_wxyz=pose_world[3:],
                    base_quat_wxyz=base_quat,
                    steps_deg=args.orientation_sweep_steps_deg,
                    execute_success=bool(args.orientation_sweep_execute),
                    reset_each_case=bool(args.orientation_sweep_reset_each_case),
                    save_images=bool(args.orientation_sweep_save_images),
                )
                success_count = sum(1 for item in results if item["status"] == "Success")
                print(
                    f"[orientation-sweep] frame={frame_idx:04d} arm={arm} "
                    f"success={success_count}/{len(results)} saved_to={arm_dir}"
                )
        print(f"Saved orientation sweep results to: {sweep_root}")
        renderer.hold_viewer()
        return

    if args.target_world_offset_z_sweep_enable:
        if args.target_world_offset_z_sweep_step <= 0.0:
            raise ValueError("--target_world_offset_z_sweep_step must be > 0")
        sweep_root = args.output_dir / "target_world_offset_z_sweep"
        sweep_root.mkdir(parents=True, exist_ok=True)
        original_offset = renderer.target_world_offset_xyz.copy()
        initial_state = renderer.snapshot_robot_state()
        z_values: List[float] = []
        z_value = float(args.target_world_offset_z_sweep_start)
        z_end = float(args.target_world_offset_z_sweep_end)
        z_step = float(args.target_world_offset_z_sweep_step)
        while z_value <= z_end + 1e-9:
            z_values.append(round(z_value, 10))
            z_value += z_step

        use_left = args.arms in ("left", "both")
        use_right = args.arms in ("right", "both")
        sweep_summary: List[Dict] = []
        for z_value in z_values:
            renderer.restore_robot_state(initial_state)
            renderer.target_world_offset_xyz = original_offset.copy()
            renderer.target_world_offset_xyz[2] = float(z_value)
            z_dir = sweep_root / format_offset_label("z", z_value)
            z_dir.mkdir(parents=True, exist_ok=True)
            frame_results: List[Dict] = []
            success_left = 0
            success_right = 0
            for frame_idx in indices:
                renderer.update_robot_link_cameras()
                left_target = trajectory.get_side_target("left", frame_idx, args.pose_source) if use_left else SideFrameTarget(False, np.full(3, np.nan), np.full((3, 3), np.nan), None)
                right_target = trajectory.get_side_target("right", frame_idx, args.pose_source) if use_right else SideFrameTarget(False, np.full(3, np.nan), np.full((3, 3), np.nan), None)

                left_world = None
                right_world = None
                if left_target.valid:
                    left_world = renderer.camera_to_world_pose(left_target.position_cam, left_target.rotation_cam)
                    left_world = renderer.apply_target_world_offset(left_world)
                    left_world = renderer.align_target_orientation("left", left_world)
                if right_target.valid:
                    right_world = renderer.camera_to_world_pose(right_target.position_cam, right_target.rotation_cam)
                    right_world = renderer.apply_target_world_offset(right_world)
                    right_world = renderer.align_target_orientation("right", right_world)

                renderer.update_target_axis_visuals(left_world, right_world)
                renderer.step_scene(steps=1)

                left_plan = renderer.plan_path("left", left_world) if left_world is not None else None
                right_plan = renderer.plan_path("right", right_world) if right_world is not None else None

                if args.debug_mode:
                    if use_left:
                        renderer.print_frame_debug(frame_idx, "left", left_target, left_world, left_plan)
                    if use_right:
                        renderer.print_frame_debug(frame_idx, "right", right_target, right_world, right_plan)

                left_status, right_status = renderer.execute_plans(left_plan, right_plan)
                if left_status == "Success":
                    success_left += 1
                if right_status == "Success":
                    success_right += 1

                left_gripper = map_gripper_value(left_target.finger_distance, fixed_left, left_range) if use_left else None
                right_gripper = map_gripper_value(right_target.finger_distance, fixed_right, right_range) if use_right else None
                renderer.set_grippers(left_gripper, right_gripper)
                renderer.update_robot_link_cameras()

                overlay_lines = [
                    f"z_offset={z_value:+.2f}m",
                    f"frame={frame_idx}",
                    f"left={left_status}",
                    f"right={right_status}",
                ]
                zed_rgb, _ = renderer.capture_camera(renderer.zed_camera)
                cv2.imwrite(str(z_dir / f"frame_{frame_idx:04d}_zed.png"), overlay_text(zed_rgb, overlay_lines))
                if renderer.third_person_view:
                    third_rgb, _ = renderer.capture_camera(renderer.third_camera)
                    cv2.imwrite(str(z_dir / f"frame_{frame_idx:04d}_third.png"), overlay_text(third_rgb, overlay_lines))

                frame_results.append(
                    {
                        "frame_idx": int(frame_idx),
                        "left_status": left_status,
                        "right_status": right_status,
                        "left_target_world": None if left_world is None else left_world.copy(),
                        "right_target_world": None if right_world is None else right_world.copy(),
                    }
                )

            summary_item = {
                "z_offset_m": float(z_value),
                "num_frames": int(len(indices)),
                "left_success_count": int(success_left),
                "right_success_count": int(success_right),
                "frame_results": frame_results,
            }
            sweep_summary.append(summary_item)
            with (z_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(make_json_safe(summary_item), f, ensure_ascii=False, indent=2)
            print(
                f"[z-sweep] z={z_value:+.2f}m "
                f"left_success={success_left}/{len(indices)} "
                f"right_success={success_right}/{len(indices)} saved_to={z_dir}"
            )

        renderer.target_world_offset_xyz = original_offset
        with (sweep_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(make_json_safe(sweep_summary), f, ensure_ascii=False, indent=2)
        print(f"Saved z-offset sweep results to: {sweep_root}")
        renderer.hold_viewer()
        return

    main_video_path = args.output_dir / "zed_replay.mp4"
    third_video_path = args.output_dir / "third_replay.mp4"
    depth_video_path = args.output_dir / "zed_depth.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    main_writer = cv2.VideoWriter(str(main_video_path), fourcc, args.fps, (args.image_width, args.image_height))
    third_writer = None
    if args.third_person_view:
        third_writer = cv2.VideoWriter(str(third_video_path), fourcc, args.fps, (args.image_width, args.image_height))
    depth_writer = cv2.VideoWriter(str(depth_video_path), fourcc, args.fps, (args.image_width, args.image_height), False)

    n_total = trajectory.length
    left_world_targets = np.full((n_total, 7), np.nan, dtype=np.float64)
    right_world_targets = np.full((n_total, 7), np.nan, dtype=np.float64)
    left_statuses = np.full((n_total,), "", dtype="<U32")
    right_statuses = np.full((n_total,), "", dtype="<U32")
    left_values = np.full((n_total,), np.nan, dtype=np.float64)
    right_values = np.full((n_total,), np.nan, dtype=np.float64)

    use_left = args.arms in ("left", "both")
    use_right = args.arms in ("right", "both")

    try:
        for out_idx, frame_idx in enumerate(indices):
            renderer.update_robot_link_cameras()

            left_target = trajectory.get_side_target("left", frame_idx, args.pose_source) if use_left else SideFrameTarget(False, np.full(3, np.nan), np.full((3, 3), np.nan), None)
            right_target = trajectory.get_side_target("right", frame_idx, args.pose_source) if use_right else SideFrameTarget(False, np.full(3, np.nan), np.full((3, 3), np.nan), None)

            left_world = None
            right_world = None
            if left_target.valid:
                left_world = renderer.camera_to_world_pose(left_target.position_cam, left_target.rotation_cam)
                left_world = renderer.apply_target_world_offset(left_world)
                left_world = renderer.align_target_orientation("left", left_world)
                left_world_targets[frame_idx] = left_world
            if right_target.valid:
                right_world = renderer.camera_to_world_pose(right_target.position_cam, right_target.rotation_cam)
                right_world = renderer.apply_target_world_offset(right_world)
                right_world = renderer.align_target_orientation("right", right_world)
                right_world_targets[frame_idx] = right_world

            renderer.update_target_axis_visuals(left_world, right_world)
            if args.debug_mode:
                renderer.step_scene(steps=1)

            left_plan = renderer.plan_path("left", left_world) if left_world is not None else None
            right_plan = renderer.plan_path("right", right_world) if right_world is not None else None

            if args.debug_mode:
                if use_left:
                    renderer.print_frame_debug(frame_idx, "left", left_target, left_world, left_plan)
                if use_right:
                    renderer.print_frame_debug(frame_idx, "right", right_target, right_world, right_plan)

            left_status, right_status = renderer.execute_plans(left_plan, right_plan)
            left_statuses[frame_idx] = left_status
            right_statuses[frame_idx] = right_status

            left_gripper = map_gripper_value(left_target.finger_distance, fixed_left, left_range)
            right_gripper = map_gripper_value(right_target.finger_distance, fixed_right, right_range)
            if not use_left:
                left_gripper = None
            if not use_right:
                right_gripper = None
            renderer.set_grippers(left_gripper, right_gripper)
            if left_gripper is not None:
                left_values[frame_idx] = left_gripper
            if right_gripper is not None:
                right_values[frame_idx] = right_gripper

            renderer.update_robot_link_cameras()
            rgb, depth = renderer.capture_camera(renderer.zed_camera)
            overlay_lines = [
                f"Frame {frame_idx} ({out_idx + 1}/{len(indices)})",
                f"Left plan: {left_statuses[frame_idx] or 'Skipped'}",
                f"Right plan: {right_statuses[frame_idx] or 'Skipped'}",
            ]
            main_bgr = overlay_text(rgb, overlay_lines)
            main_writer.write(main_bgr)

            depth_safe = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            if depth_safe.max() > depth_safe.min():
                depth_norm = ((depth_safe - depth_safe.min()) / (depth_safe.max() - depth_safe.min()) * 255.0).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth_safe, dtype=np.uint8)
            depth_writer.write(depth_norm)

            if third_writer is not None:
                third_rgb, _ = renderer.capture_camera(renderer.third_camera)
                third_bgr = overlay_text(third_rgb, overlay_lines)
                third_writer.write(third_bgr)

            if args.save_png_frames:
                cv2.imwrite(str(frames_dir / f"zed_{frame_idx:04d}.png"), main_bgr)
                cv2.imwrite(str(frames_dir / f"depth_{frame_idx:04d}.png"), depth_safe.astype(np.uint16))
                if third_writer is not None:
                    cv2.imwrite(str(frames_dir / f"third_{frame_idx:04d}.png"), third_bgr)

            print(
                f"[frame {frame_idx:04d}] "
                f"left={left_statuses[frame_idx] or 'Skipped'} "
                f"right={right_statuses[frame_idx] or 'Skipped'} "
                f"left_gripper={left_gripper if left_gripper is not None else 'N/A'} "
                f"right_gripper={right_gripper if right_gripper is not None else 'N/A'}"
            )
    finally:
        main_writer.release()
        depth_writer.release()
        if third_writer is not None:
            third_writer.release()

    np.savez_compressed(
        str(args.output_dir / "world_targets_and_status.npz"),
        source_npz=str(args.input_npz),
        selected_indices=np.asarray(indices, dtype=np.int32),
        left_world_targets=left_world_targets,
        right_world_targets=right_world_targets,
        left_plan_status=left_statuses,
        right_plan_status=right_statuses,
        left_gripper_value=left_values,
        right_gripper_value=right_values,
        pose_source=np.array(args.pose_source),
        robot_base_pose=np.asarray(args.robot_base_pose if args.robot_base_pose is not None else [], dtype=np.float64),
        torso_qpos=np.asarray(args.torso_qpos, dtype=np.float64),
        camera_axis_conversion=np.array(f"world = zed_link_pose * {renderer.camera_cv_axis_mode} * camera_cv", dtype=object),
    )

    print(f"Saved zed replay video to: {main_video_path}")
    print(f"Saved depth replay video to: {depth_video_path}")
    if args.third_person_view:
        print(f"Saved third-person replay video to: {third_video_path}")
    print(f"Saved world-space targets to: {args.output_dir / 'world_targets_and_status.npz'}")
    renderer.hold_viewer()


if __name__ == "__main__":
    main()
