#!/usr/bin/env python3
"""Plan a two-keyframe RoboTwin demo from AnyGrasp candidates and hand gripper poses."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import render_hand_retarget_r1_npz as base
import render_hand_retarget_r1_npz_urdfik as urdfik_base
from render_object_pose_r1_npz import create_object_actor, get_camera_intrinsic_matrix, pose_wxyz_to_matrix
from replay_r1_h5 import ReplayRenderer, parse_optional_base_pose


R1_CONFIG = PROJECT_ROOT / "robot_config_R1.json"


@dataclass
class ObjectState:
    name: str
    mesh_file: Path
    pose_world_wxyz: np.ndarray
    pose_world_matrix: np.ndarray
    visible: bool
    actor: Optional[sapien.Entity] = None


@dataclass
class ObjectTrack:
    name: str
    mesh_file: Path
    frame_indices: np.ndarray
    pose_world_wxyz: np.ndarray
    pose_world_matrix: np.ndarray
    visible: np.ndarray
    actor: Optional[sapien.Entity] = None


@dataclass
class CandidatePose:
    candidate_idx: int
    score: float
    translation_cam: np.ndarray
    rotation_cam: np.ndarray
    width_m: float
    depth_m: float
    raw_pose_world_wxyz: np.ndarray
    raw_pose_world_matrix: np.ndarray
    pose_world_wxyz: np.ndarray
    pose_world_matrix: np.ndarray
    nearest_object: str
    nearest_object_distance_m: float
    rotation_distance_deg: float
    top_axis_up_dot: float
    original_top_axis_up_dot: float
    camera_up_flip_applied: int
    forward_axis_change_deg: float


@dataclass
class SelectedKeyframe:
    source_frame: int
    arm: str
    candidate: CandidatePose
    hand_rotation_cam: np.ndarray


@dataclass
class ArmSelectionResult:
    arm: str
    expected_object: Optional[str]
    selected_keyframes: List[SelectedKeyframe]
    ranked_candidates_per_frame: Dict[int, List[CandidatePose]]
    all_candidates_per_frame: Dict[int, List[CandidatePose]]
    diagnostics: Dict[int, Dict[str, int]]


@dataclass
class ArmDebugInfo:
    arm: str
    expected_object: Optional[str]
    ranked_candidates_per_frame: Dict[int, List[CandidatePose]]
    all_candidates_per_frame: Dict[int, List[CandidatePose]]
    diagnostics: Dict[int, Dict[str, int]]
    selected_keyframes: Optional[List[SelectedKeyframe]] = None


@dataclass
class DebugVisualBundle:
    keyframe_axis_actors: Dict[Tuple[int, str], sapien.Entity]
    common_candidate_actors: Dict[int, List[sapien.Entity]]
    arm_candidate_actors: Dict[str, Dict[int, List[sapien.Entity]]]
    arm_candidate_axis_actors: Dict[str, Dict[int, List[sapien.Entity]]]


@dataclass
class DebugExecutionState:
    writer: Optional[cv2.VideoWriter]
    selected_keyframes: List[SelectedKeyframe]
    common_candidates_per_frame: Dict[int, List[CandidatePose]]
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]]
    head_intrinsic: np.ndarray
    active_frame: Optional[int] = None
    pose_debug_path: Optional[Path] = None
    replay_head_camera_pose_by_frame: Optional[Dict[int, np.ndarray]] = None
    object_tracks: Optional[Dict[str, ObjectTrack]] = None


@dataclass
class RankPreviewRecord:
    frame: int
    rank: int
    image_path: str
    left_candidate_idx: Optional[int]
    right_candidate_idx: Optional[int]


@dataclass
class ExecutionObjectReplayConfig:
    replay_frame_indices: np.ndarray
    object_tracks: Dict[str, ObjectTrack]
    start_frame: int
    end_frame: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use AnyGrasp keyframes + hand orientation to plan a two-keyframe RoboTwin demo.")
    parser.add_argument("--anygrasp_dir", type=Path, required=True, help="Per-video AnyGrasp result dir, e.g. anygrasp_batch_results/d_pour_blue_1.")
    parser.add_argument("--replay_dir", type=Path, required=True, help="Per-video replay dir containing multi_object_world_poses.npz.")
    parser.add_argument("--hand_npz", type=Path, required=True, help="Per-video hand_detections_*.npz with gripper pose fields.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output dir for planned demo video and metadata.")
    parser.add_argument("--keyframes", type=int, nargs=2, default=[1, 22], metavar=("GRASP_FRAME", "ACTION_FRAME"))
    parser.add_argument("--arm", choices=["auto", "left", "right"], default="auto")
    parser.add_argument("--execute_both_arms", type=int, default=0, help="If 1 and --arm auto, execute synchronized dual-arm stages and advance only when both arms satisfy reach checks.")
    parser.add_argument("--planner_backend", choices=["urdfik", "curobo"], default="urdfik")
    parser.add_argument("--left_target_object", type=str, default="cup")
    parser.add_argument("--right_target_object", type=str, default="bottle")
    parser.add_argument("--candidate_object_max_distance_m", type=float, default=0.12)
    parser.add_argument("--enforce_target_object_constraint", type=int, default=1)
    parser.add_argument("--enforce_candidate_distance_constraint", type=int, default=1)
    parser.add_argument("--debug_candidate_top_k", type=int, default=5)
    parser.add_argument("--debug_show_all_candidates", type=int, default=1)
    parser.add_argument("--debug_common_candidate_top_k", type=int, default=0, help="How many raw-score candidates to show in green per keyframe. 0 hides them.")
    parser.add_argument("--candidate_orientation_remap_label", type=str, default="identity")
    parser.add_argument("--candidate_post_rot_xyz_deg", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--candidate_keep_camera_up", type=int, default=0, help="If 1, keep the gripper/camera top side facing upward overall while preserving the original grasp direction. The planner only resolves the redundant roll about the gripper forward axis.")
    parser.add_argument("--candidate_camera_top_axis", choices=["y", "z"], default="z", help="Which local gripper axis should be treated as the camera/top direction when --candidate_keep_camera_up=1.")
    parser.add_argument("--candidate_target_local_x_offset_m", type=float, default=0.0, help="Additional translation applied to each AnyGrasp world target along the gripper local +X axis before planning/visualization. Use -0.12 if the raw candidate behaves like a wrist/endlink pose and you want a fingertip TCP target.")
    parser.add_argument("--manual_candidate", type=str, nargs=3, action="append", default=[], metavar=("FRAME", "ARM", "CANDIDATE_IDX"), help="Optional manual candidate override, e.g. --manual_candidate 1 left 5. Partial overrides only reorder debug display; full two-frame overrides for one arm drive selection directly.")
    parser.add_argument("--object_mesh_override", action="append", default=[], help="Repeatable mesh override in the form NAME=/abs/path/to/mesh.obj, e.g. cup=/.../blue_cup.obj")
    parser.add_argument("--robot_config", type=Path, default=R1_CONFIG)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--fovy_deg", type=float, default=90.0)
    parser.add_argument("--torso_qpos", type=float, nargs=4, default=base.DEFAULT_TORSO_QPOS.tolist())
    parser.add_argument("--robot_base_pose", type=float, nargs=7, default=None, metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"))
    parser.add_argument("--open_gripper", type=float, default=1.0)
    parser.add_argument("--close_gripper", type=float, default=0.0)
    parser.add_argument("--approach_offset_m", type=float, default=0.08)
    parser.add_argument("--settle_steps", type=int, default=4)
    parser.add_argument("--execute_interp_steps", type=int, default=24)
    parser.add_argument("--reach_pos_tol_m", type=float, default=0.03)
    parser.add_argument("--reach_rot_tol_deg", type=float, default=20.0)
    parser.add_argument("--reach_error_pose_source", choices=["tcp", "ee"], default="tcp", help="Which arm pose to use when computing reach error against the target.")
    parser.add_argument("--max_stage_replans", type=int, default=3)
    parser.add_argument("--replan_until_reached", type=int, default=0, help="If 1, keep replanning from the current state until the stage reaches tolerance or the extended attempt budget is exhausted.")
    parser.add_argument("--replan_until_reached_max_attempts", type=int, default=20)
    parser.add_argument("--hold_frames_after_stage", type=int, default=2)
    parser.add_argument("--init_prefix_frames", type=int, default=0, help="Emit fixed init-pose frames before moving to keyframe-1; useful for downstream trimming.")
    parser.add_argument("--pause_after_keyframe1_seconds", type=float, default=0.0, help="After reaching keyframe-1 and closing the gripper, hold the robot at that pose for N seconds before planning/executing the next target.")
    parser.add_argument("--replay_objects_during_action", type=int, default=0, help="If 1, replay object tracks from keyframe-1 to keyframe-2 during the action stage instead of attaching selected objects to the TCP.")
    parser.add_argument("--replay_objects_ignore_collision", type=int, default=1, help="If 1, replayed objects are created as visual-only kinematic actors without collision.")
    parser.add_argument("--save_debug_preview", type=int, default=1)
    parser.add_argument("--debug_preview_fps", type=int, default=10)
    parser.add_argument("--debug_keyframe_hold_frames", type=int, default=12)
    parser.add_argument("--save_debug_execution_preview", type=int, default=1)
    parser.add_argument("--debug_execution_fps", type=int, default=10)
    parser.add_argument("--save_pose_debug", type=int, default=0, help="If 1, dump per-frame planner camera/TCP/object poses to pose_debug.jsonl.")
    parser.add_argument("--save_rank_preview_images", type=int, default=1)
    parser.add_argument("--rank_preview_top_n", type=int, default=3, help="Save per-keyframe rank preview PNGs for left/right rank 1..N.")
    parser.add_argument("--debug_target_axis_length", type=float, default=0.08)
    parser.add_argument("--debug_target_axis_thickness", type=float, default=0.004)
    parser.add_argument("--head_only", type=int, default=1)
    parser.add_argument("--overlay_text", type=int, default=1)
    parser.add_argument("--third_person_view", type=int, default=0)
    parser.add_argument("--enable_viewer", type=int, default=0)
    parser.add_argument("--viewer_frame_delay", type=float, default=0.0)
    parser.add_argument("--viewer_wait_at_end", type=int, default=0)
    parser.add_argument("--disable_table", type=int, default=1)
    parser.add_argument("--lighting_mode", choices=["default", "front", "front_no_shadow"], default="front_no_shadow")
    parser.add_argument("--camera_cv_axis_mode", choices=sorted(base.CV_TO_WORLD_CAMERA_PRESETS.keys()), default="legacy_r1")
    parser.add_argument("--head_camera_local_pos", type=float, nargs=3, default=base.DEFAULT_HEAD_CAMERA_LOCAL_POS.tolist())
    parser.add_argument("--head_camera_local_quat_wxyz", type=float, nargs=4, default=base.DEFAULT_HEAD_CAMERA_LOCAL_QUAT_WXYZ.tolist())
    return parser.parse_args()


def build_renderer(args: argparse.Namespace) -> ReplayRenderer:
    renderer_cls = ReplayRenderer if args.planner_backend == "curobo" else urdfik_base.HandRetargetR1URDFIKRenderer
    attach_planner = args.planner_backend == "curobo"
    return renderer_cls(
        robot_config_path=args.robot_config,
        image_width=args.image_width,
        image_height=args.image_height,
        fovy_deg=args.fovy_deg,
        torso_qpos=args.torso_qpos,
        robot_base_pose_override=parse_optional_base_pose(args.robot_base_pose),
        third_person_view=bool(args.third_person_view),
        need_topp=False,
        link_cam_debug_enable=False,
        link_cam_axis_mode="none",
        link_cam_debug_rot_xyz_deg=[0.0, 0.0, 0.0],
        link_cam_debug_shift_fru=[0.0, 0.0, 0.0],
        camera_cv_axis_mode=args.camera_cv_axis_mode,
        head_camera_local_pos=args.head_camera_local_pos,
        head_camera_local_quat_wxyz=args.head_camera_local_quat_wxyz,
        wrist_camera_local_pos=base.DEFAULT_WRIST_CAMERA_LOCAL_POS,
        wrist_camera_local_quat_wxyz=base.DEFAULT_WRIST_CAMERA_LOCAL_QUAT_WXYZ,
        camera_debug_target="head",
        enable_viewer=bool(args.enable_viewer),
        viewer_frame_delay=args.viewer_frame_delay,
        viewer_wait_at_end=bool(args.viewer_wait_at_end),
        debug_mode=False,
        debug_force_orientation="none",
        debug_visualize_targets=False,
        debug_target_axis_length=args.debug_target_axis_length,
        debug_target_axis_thickness=args.debug_target_axis_thickness,
        orientation_remap_label="identity",
        orientation_remap_matrix=np.eye(3, dtype=np.float64),
        stored_orientation_post_rot_xyz_deg=[0.0, 0.0, 0.0],
        target_world_offset_xyz=[0.0, 0.0, 0.0],
        left_target_world_offset_xyz=[0.0, 0.0, 0.0],
        right_target_world_offset_xyz=[0.0, 0.0, 0.0],
        target_world_z_offset=0.0,
        disable_table=bool(args.disable_table),
        camera_sweep_enable=False,
        camera_sweep_steps_deg=[0.0],
        init_left_arm_joints=None,
        init_right_arm_joints=None,
        init_gripper_open=None,
        lighting_mode=args.lighting_mode,
        attach_planner=attach_planner,
        hide_robot=False,
    )


def load_hand_data(hand_npz: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(hand_npz), allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def load_anygrasp_candidates(anygrasp_dir: Path, source_frame: int) -> List[Dict]:
    grasp_json = anygrasp_dir / "grasps" / f"grasp_{int(source_frame):06d}.json"
    if not grasp_json.is_file():
        raise FileNotFoundError(f"Missing AnyGrasp grasp json: {grasp_json}")
    with grasp_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("grasps", []))


def parse_object_mesh_overrides(specs: Sequence[str]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for spec in specs:
        if "=" not in str(spec):
            raise ValueError(f"Invalid --object_mesh_override value '{spec}'. Expected NAME=/abs/path/to/mesh.obj")
        name, mesh_file = str(spec).split("=", 1)
        obj_name = name.strip()
        mesh_path = Path(mesh_file.strip()).resolve()
        if not obj_name:
            raise ValueError(f"Invalid --object_mesh_override value '{spec}'. Empty object name.")
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Override mesh file does not exist: {mesh_path}")
        overrides[obj_name] = mesh_path
    return overrides


def load_object_states(replay_dir: Path, source_frame: int, mesh_overrides: Optional[Dict[str, Path]] = None) -> Dict[str, ObjectState]:
    pose_path = replay_dir / "multi_object_world_poses.npz"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing multi_object_world_poses.npz: {pose_path}")
    data = np.load(str(pose_path), allow_pickle=True)
    frame_indices = np.asarray(data["selected_source_frame_indices"], dtype=np.int32).reshape(-1)
    matches = np.where(frame_indices == int(source_frame))[0]
    if matches.size == 0:
        raise KeyError(f"source_frame={source_frame} not found in {pose_path}")
    idx = int(matches[0])

    object_names = [str(name) for name in np.asarray(data["object_names"], dtype=object).tolist()]
    states: Dict[str, ObjectState] = {}
    for object_name in object_names:
        key = object_name
        pose_world_wxyz = np.asarray(data[f"{key}__pose_world_wxyz"], dtype=np.float64)[idx]
        pose_world_matrix = np.asarray(data[f"{key}__pose_world_matrix"], dtype=np.float64)[idx]
        visible = bool(np.asarray(data[f"{key}__visible"], dtype=bool)[idx])
        mesh_file = Path(str(np.asarray(data[f"{key}__mesh_file"], dtype=object).reshape(()))).resolve()
        if mesh_overrides and object_name in mesh_overrides:
            mesh_file = mesh_overrides[object_name]
        states[object_name] = ObjectState(
            name=object_name,
            mesh_file=mesh_file,
            pose_world_wxyz=pose_world_wxyz,
            pose_world_matrix=pose_world_matrix,
            visible=visible,
        )
    return states


def load_object_tracks(replay_dir: Path, mesh_overrides: Optional[Dict[str, Path]] = None) -> Tuple[np.ndarray, Dict[str, ObjectTrack]]:
    pose_path = replay_dir / "multi_object_world_poses.npz"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing multi_object_world_poses.npz: {pose_path}")
    data = np.load(str(pose_path), allow_pickle=True)
    frame_indices = np.asarray(data["selected_source_frame_indices"], dtype=np.int32).reshape(-1)
    object_names = [str(name) for name in np.asarray(data["object_names"], dtype=object).tolist()]
    tracks: Dict[str, ObjectTrack] = {}
    for object_name in object_names:
        key = object_name
        mesh_file = Path(str(np.asarray(data[f"{key}__mesh_file"], dtype=object).reshape(()))).resolve()
        if mesh_overrides and object_name in mesh_overrides:
            mesh_file = mesh_overrides[object_name]
        tracks[object_name] = ObjectTrack(
            name=object_name,
            mesh_file=mesh_file,
            frame_indices=frame_indices.copy(),
            pose_world_wxyz=np.asarray(data[f"{key}__pose_world_wxyz"], dtype=np.float64),
            pose_world_matrix=np.asarray(data[f"{key}__pose_world_matrix"], dtype=np.float64),
            visible=np.asarray(data[f"{key}__visible"], dtype=bool),
        )
    return frame_indices, tracks


def load_replay_head_camera_poses(replay_dir: Path) -> Dict[int, np.ndarray]:
    pose_path = replay_dir / "multi_object_world_poses.npz"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing multi_object_world_poses.npz: {pose_path}")
    data = np.load(str(pose_path), allow_pickle=True)
    if "head_camera_pose_world_wxyz" not in data.files:
        return {}
    frame_indices = np.asarray(data["selected_source_frame_indices"], dtype=np.int32).reshape(-1)
    head_poses = np.asarray(data["head_camera_pose_world_wxyz"], dtype=np.float64).reshape(-1, 7)
    return {int(frame): head_poses[idx] for idx, frame in enumerate(frame_indices.tolist())}


def create_execution_object_actor(scene: sapien.Scene, mesh_file: Path, actor_name: str, ignore_collision: bool) -> sapien.Entity:
    if not bool(ignore_collision):
        return create_object_actor(scene, mesh_file, actor_name)
    builder = scene.create_actor_builder()
    try:
        builder.add_visual_from_file(str(mesh_file))
    except Exception as exc:
        raise RuntimeError(f"Failed to load mesh visual: {mesh_file}") from exc
    return builder.build_kinematic(name=actor_name)


def nearest_track_index(replay_frame_indices: np.ndarray, source_frame: int) -> int:
    frame_indices = np.asarray(replay_frame_indices, dtype=np.int32).reshape(-1)
    if frame_indices.size == 0:
        raise ValueError("replay_frame_indices is empty.")
    return int(np.argmin(np.abs(frame_indices - int(source_frame))))


def update_execution_object_replay(config: ExecutionObjectReplayConfig, progress_01: float) -> int:
    alpha = float(np.clip(progress_01, 0.0, 1.0))
    source_frame = int(round(float(config.start_frame) + alpha * float(config.end_frame - config.start_frame)))
    idx = nearest_track_index(config.replay_frame_indices, source_frame)
    for track in config.object_tracks.values():
        if track.actor is None:
            continue
        if bool(track.visible[idx]):
            set_actor_pose(track.actor, track.pose_world_wxyz[idx])
        else:
            hide_actor(track.actor)
    return int(config.replay_frame_indices[idx])


def expected_object_for_arm(args: argparse.Namespace, arm: str) -> Optional[str]:
    if not bool(args.enforce_target_object_constraint):
        return None
    if arm == "left":
        value = str(args.left_target_object).strip()
    elif arm == "right":
        value = str(args.right_target_object).strip()
    else:
        value = ""
    return value or None


def rotation_distance_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    delta = R.from_matrix(np.asarray(rot_a, dtype=np.float64).reshape(3, 3)).inv() * R.from_matrix(np.asarray(rot_b, dtype=np.float64).reshape(3, 3))
    return float(np.rad2deg(delta.magnitude()))


def shift_pose_along_local_x(pose_world_wxyz: np.ndarray, delta_m: float) -> np.ndarray:
    pose_world_wxyz = np.asarray(pose_world_wxyz, dtype=np.float64).reshape(7)
    pose_world_matrix = pose_wxyz_to_matrix(pose_world_wxyz)
    pose_world_matrix[:3, 3] += pose_world_matrix[:3, 0] * float(delta_m)
    quat = base.quat_xyzw_to_wxyz(R.from_matrix(base.orthonormalize_rotation(pose_world_matrix[:3, :3])).as_quat())
    return np.concatenate([pose_world_matrix[:3, 3], quat]).astype(np.float64)


def candidate_to_world_pose(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    grasp: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    rot_cam = np.asarray(grasp["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    trans_cam = np.asarray(grasp["translation"], dtype=np.float64).reshape(3)
    remapped_rotation = base.orthonormalize_rotation(
        base.orthonormalize_rotation(rot_cam) @ args.candidate_post_rot_matrix @ args.candidate_orientation_remap_matrix
    )
    raw_pose_world_wxyz = renderer.camera_to_world_pose(trans_cam, remapped_rotation)
    raw_pose_world_matrix = pose_wxyz_to_matrix(raw_pose_world_wxyz)
    pose_world_wxyz = raw_pose_world_wxyz.copy()
    if abs(float(args.candidate_target_local_x_offset_m)) > 1e-12:
        pose_world_wxyz = shift_pose_along_local_x(pose_world_wxyz, float(args.candidate_target_local_x_offset_m))
    pose_world_matrix = pose_wxyz_to_matrix(pose_world_wxyz)
    original_rotation_world = raw_pose_world_matrix[:3, :3].copy()
    roll_debug = {
        "original_top_axis_up_dot": top_axis_up_dot(original_rotation_world, args.candidate_camera_top_axis),
        "camera_up_flip_applied": 0,
        "forward_axis_change_deg": 0.0,
    }
    if bool(args.candidate_keep_camera_up):
        pose_world_matrix[:3, :3], roll_debug = constrain_roll_keep_top_axis_up(
            pose_world_matrix[:3, :3],
            top_axis=args.candidate_camera_top_axis,
        )
        quat = base.quat_xyzw_to_wxyz(R.from_matrix(base.orthonormalize_rotation(pose_world_matrix[:3, :3])).as_quat())
        pose_world_wxyz = np.concatenate([pose_world_matrix[:3, 3], quat]).astype(np.float64)
    return raw_pose_world_wxyz, raw_pose_world_matrix, pose_world_wxyz, pose_world_matrix, roll_debug


def top_axis_up_dot(rotation_world: np.ndarray, top_axis: str) -> float:
    axis_idx = 1 if top_axis == "y" else 2
    axis_vec = np.asarray(rotation_world, dtype=np.float64).reshape(3, 3)[:, axis_idx]
    return float(np.dot(axis_vec, np.array([0.0, 0.0, 1.0], dtype=np.float64)))


def forward_axis_change_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    x_a = np.asarray(rotation_a, dtype=np.float64).reshape(3, 3)[:, 0]
    x_b = np.asarray(rotation_b, dtype=np.float64).reshape(3, 3)[:, 0]
    dot = float(np.clip(np.dot(x_a, x_b) / max(np.linalg.norm(x_a) * np.linalg.norm(x_b), 1e-12), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot)))


def constrain_roll_keep_top_axis_up(rotation_world: np.ndarray, top_axis: str) -> Tuple[np.ndarray, Dict[str, float]]:
    rot = base.orthonormalize_rotation(rotation_world)
    roll_flip_180 = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    rot_flipped = base.orthonormalize_rotation(rot @ roll_flip_180)
    base_dot = top_axis_up_dot(rot, top_axis)
    flipped_dot = top_axis_up_dot(rot_flipped, top_axis)
    if flipped_dot > base_dot + 1e-9:
        chosen = rot_flipped
        flip_applied = 1
    else:
        chosen = rot
        flip_applied = 0
    return chosen, {
        "original_top_axis_up_dot": float(base_dot),
        "camera_up_flip_applied": int(flip_applied),
        "forward_axis_change_deg": float(forward_axis_change_deg(rot, chosen)),
    }


def nearest_object_name(candidate_world_matrix: np.ndarray, object_states: Dict[str, ObjectState]) -> Tuple[str, float]:
    pos = np.asarray(candidate_world_matrix[:3, 3], dtype=np.float64)
    best_name = ""
    best_dist = float("inf")
    for name, state in object_states.items():
        if not state.visible or not np.isfinite(state.pose_world_matrix).all():
            continue
        obj_pos = np.asarray(state.pose_world_matrix[:3, 3], dtype=np.float64)
        dist = float(np.linalg.norm(pos - obj_pos))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    if not best_name:
        raise RuntimeError("No visible objects found for nearest-object matching.")
    return best_name, best_dist


def nearest_valid_hand_frame(frame: int, hand_valid: np.ndarray) -> Optional[int]:
    valid_indices = np.where(np.asarray(hand_valid, dtype=bool))[0]
    if valid_indices.size == 0:
        return None
    return int(valid_indices[np.argmin(np.abs(valid_indices - int(frame)))])


def build_ranked_candidates_for_arm(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    anygrasp_dir: Path,
    hand_data: Dict[str, np.ndarray],
    keyframes: Sequence[int],
    arm: str,
    object_states_per_frame: Dict[int, Dict[str, ObjectState]],
) -> Tuple[Optional[Dict[int, List[CandidatePose]]], Dict[int, Dict[str, int]], Dict[int, List[CandidatePose]]]:
    hand_valid = np.asarray(hand_data[f"{arm}_gripper_valid"], dtype=bool)
    hand_rotations = np.asarray(hand_data[f"{arm}_gripper_rotation_matrix"], dtype=np.float64)

    expected_object = expected_object_for_arm(args, arm)
    candidates_per_frame: Dict[int, List[CandidatePose]] = {}
    all_candidates_per_frame: Dict[int, List[CandidatePose]] = {}
    diagnostics: Dict[int, Dict[str, int]] = {}
    for frame in keyframes:
        ref_frame = int(frame) if bool(hand_valid[frame]) else nearest_valid_hand_frame(int(frame), hand_valid)
        if ref_frame is None:
            diagnostics[int(frame)] = {"hand_valid": 0, "reference_hand_frame": -1}
            return None, diagnostics, all_candidates_per_frame
        ref_rotation = np.asarray(hand_rotations[ref_frame], dtype=np.float64)
        all_candidates = []
        raw_candidates = []
        total = 0
        object_pass = 0
        distance_pass = 0
        for candidate_idx, grasp in enumerate(load_anygrasp_candidates(anygrasp_dir, frame)):
            total += 1
            raw_pose_world_wxyz, raw_pose_world_matrix, pose_world_wxyz, pose_world_matrix, roll_debug = candidate_to_world_pose(renderer, args, grasp)
            nearest_name, nearest_dist = nearest_object_name(pose_world_matrix, object_states_per_frame[frame])
            candidate = CandidatePose(
                candidate_idx=candidate_idx,
                score=float(grasp["score"]),
                translation_cam=np.asarray(grasp["translation"], dtype=np.float64).reshape(3),
                rotation_cam=np.asarray(grasp["rotation_matrix"], dtype=np.float64).reshape(3, 3),
                width_m=float(grasp.get("width", 0.08)),
                depth_m=float(grasp.get("depth", 0.04)),
                raw_pose_world_wxyz=raw_pose_world_wxyz,
                raw_pose_world_matrix=raw_pose_world_matrix,
                pose_world_wxyz=pose_world_wxyz,
                pose_world_matrix=pose_world_matrix,
                nearest_object=nearest_name,
                nearest_object_distance_m=nearest_dist,
                rotation_distance_deg=rotation_distance_deg(ref_rotation, grasp["rotation_matrix"]),
                top_axis_up_dot=top_axis_up_dot(pose_world_matrix[:3, :3], args.candidate_camera_top_axis),
                original_top_axis_up_dot=float(roll_debug["original_top_axis_up_dot"]),
                camera_up_flip_applied=int(roll_debug["camera_up_flip_applied"]),
                forward_axis_change_deg=float(roll_debug["forward_axis_change_deg"]),
            )
            raw_candidates.append(candidate)
            object_ok = expected_object is None or nearest_name == expected_object
            if not object_ok:
                continue
            object_pass += 1
            if bool(args.enforce_candidate_distance_constraint) and nearest_dist > float(args.candidate_object_max_distance_m):
                continue
            distance_pass += 1
            all_candidates.append(candidate)
        raw_candidates.sort(key=lambda cand: (cand.rotation_distance_deg, cand.nearest_object_distance_m, -cand.score))
        all_candidates.sort(key=lambda cand: (cand.rotation_distance_deg, cand.nearest_object_distance_m, -cand.score))
        all_candidates_per_frame[frame] = raw_candidates
        candidates_per_frame[frame] = all_candidates
        diagnostics[int(frame)] = {
            "hand_valid": int(bool(hand_valid[frame])),
            "reference_hand_frame": int(ref_frame),
            "total_candidates": total,
            "object_pass": object_pass,
            "distance_pass": distance_pass,
            "raw_candidates": len(raw_candidates),
            "final_ranked": len(all_candidates),
        }
        if not all_candidates:
            return None, diagnostics, all_candidates_per_frame
    return candidates_per_frame, diagnostics, all_candidates_per_frame


def parse_manual_candidate_overrides(specs: Sequence[Sequence[str]]) -> Dict[str, Dict[int, int]]:
    overrides: Dict[str, Dict[int, int]] = {"left": {}, "right": {}}
    for spec in specs:
        if len(spec) != 3:
            raise ValueError(f"Invalid --manual_candidate entry: {spec}")
        frame = int(spec[0])
        arm = str(spec[1]).strip().lower()
        candidate_idx = int(spec[2])
        if arm not in overrides:
            raise ValueError(f"Unsupported arm in --manual_candidate: {arm}")
        overrides[arm][frame] = candidate_idx
    return overrides


def find_candidate_by_idx(candidates: Sequence[CandidatePose], candidate_idx: int) -> Optional[CandidatePose]:
    for cand in candidates:
        if int(cand.candidate_idx) == int(candidate_idx):
            return cand
    return None


def reorder_candidates_with_manual_choice(candidates: Sequence[CandidatePose], candidate_idx: Optional[int]) -> List[CandidatePose]:
    ordered = list(candidates)
    if candidate_idx is None:
        return ordered
    chosen = find_candidate_by_idx(ordered, int(candidate_idx))
    if chosen is None:
        return ordered
    return [chosen] + [cand for cand in ordered if int(cand.candidate_idx) != int(candidate_idx)]


def select_keyframes_for_arm(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    anygrasp_dir: Path,
    hand_data: Dict[str, np.ndarray],
    keyframes: Sequence[int],
    arm: str,
    object_states_per_frame: Dict[int, Dict[str, ObjectState]],
) -> Optional[ArmSelectionResult]:
    hand_rotations = np.asarray(hand_data[f"{arm}_gripper_rotation_matrix"], dtype=np.float64)
    candidates_per_frame, diagnostics, all_candidates_per_frame = build_ranked_candidates_for_arm(
        renderer=renderer,
        args=args,
        anygrasp_dir=anygrasp_dir,
        hand_data=hand_data,
        keyframes=keyframes,
        arm=arm,
        object_states_per_frame=object_states_per_frame,
    )
    if candidates_per_frame is None:
        return None

    manual_overrides = getattr(args, "manual_candidate_overrides", {}).get(arm, {})
    for frame in keyframes:
        frame = int(frame)
        manual_idx = manual_overrides.get(frame)
        candidates_per_frame[frame] = reorder_candidates_with_manual_choice(candidates_per_frame.get(frame, []), manual_idx)
        all_candidates_per_frame[frame] = reorder_candidates_with_manual_choice(all_candidates_per_frame.get(frame, []), manual_idx)
        diagnostics.setdefault(frame, {})["manual_candidate_idx"] = -1 if manual_idx is None else int(manual_idx)
        diagnostics[frame]["manual_candidate_found"] = int(find_candidate_by_idx(all_candidates_per_frame.get(frame, []), manual_idx) is not None) if manual_idx is not None else 0

    if all(int(frame) in manual_overrides for frame in keyframes):
        manual_selection: List[SelectedKeyframe] = []
        for frame in keyframes:
            frame = int(frame)
            chosen = find_candidate_by_idx(all_candidates_per_frame.get(frame, []), int(manual_overrides[frame]))
            if chosen is None:
                return None
            manual_selection.append(
                SelectedKeyframe(
                    source_frame=frame,
                    arm=arm,
                    candidate=chosen,
                    hand_rotation_cam=np.asarray(hand_rotations[frame], dtype=np.float64),
                )
            )
        return ArmSelectionResult(
            arm=arm,
            expected_object=expected_object_for_arm(args, arm),
            selected_keyframes=manual_selection,
            ranked_candidates_per_frame=candidates_per_frame,
            all_candidates_per_frame=all_candidates_per_frame,
            diagnostics=diagnostics,
        )

    best_selection: Optional[List[SelectedKeyframe]] = None
    best_cost = float("inf")
    for first_candidate in candidates_per_frame[keyframes[0]]:
        object_name = first_candidate.nearest_object
        second_candidates = [cand for cand in candidates_per_frame[keyframes[1]] if cand.nearest_object == object_name]
        if not second_candidates:
            continue
        best_second = second_candidates[0]
        total_cost = (
            first_candidate.rotation_distance_deg
            + best_second.rotation_distance_deg
            + 10.0 * (first_candidate.nearest_object_distance_m + best_second.nearest_object_distance_m)
        )
        if total_cost < best_cost:
            best_cost = total_cost
            best_selection = [
                SelectedKeyframe(
                    source_frame=int(keyframes[0]),
                    arm=arm,
                    candidate=first_candidate,
                    hand_rotation_cam=np.asarray(hand_rotations[keyframes[0]], dtype=np.float64),
                ),
                SelectedKeyframe(
                    source_frame=int(keyframes[1]),
                    arm=arm,
                    candidate=best_second,
                    hand_rotation_cam=np.asarray(hand_rotations[keyframes[1]], dtype=np.float64),
                ),
            ]
    if best_selection is None:
        return None
    return ArmSelectionResult(
        arm=arm,
        expected_object=expected_object_for_arm(args, arm),
        selected_keyframes=best_selection,
        ranked_candidates_per_frame=candidates_per_frame,
        all_candidates_per_frame=all_candidates_per_frame,
        diagnostics=diagnostics,
    )


def choose_keyframes(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    anygrasp_dir: Path,
    hand_data: Dict[str, np.ndarray],
    keyframes: Sequence[int],
    arm_mode: str,
    object_states_per_frame: Dict[int, Dict[str, ObjectState]],
) -> Tuple[ArmSelectionResult, Dict[str, ArmDebugInfo]]:
    candidate_arms = [arm_mode] if arm_mode in ("left", "right") else ["left", "right"]
    best: Optional[ArmSelectionResult] = None
    best_cost = float("inf")
    diagnostic_lines: List[str] = []
    arm_debugs: Dict[str, ArmDebugInfo] = {}
    for arm in candidate_arms:
        selection = select_keyframes_for_arm(renderer, args, anygrasp_dir, hand_data, keyframes, arm, object_states_per_frame)
        if selection is None:
            expected_object = expected_object_for_arm(args, arm)
            ranked, diagnostics, all_candidates = build_ranked_candidates_for_arm(
                renderer=renderer,
                args=args,
                anygrasp_dir=anygrasp_dir,
                hand_data=hand_data,
                keyframes=keyframes,
                arm=arm,
                object_states_per_frame=object_states_per_frame,
            )
            arm_debugs[arm] = ArmDebugInfo(
                arm=arm,
                expected_object=expected_object,
                ranked_candidates_per_frame=ranked or {},
                all_candidates_per_frame=all_candidates,
                diagnostics=diagnostics,
                selected_keyframes=None,
            )
            diag_chunks = []
            for frame in keyframes:
                info = diagnostics.get(int(frame), {})
                diag_chunks.append(
                    f"frame={int(frame)} hand_valid={info.get('hand_valid', 0)} "
                    f"total={info.get('total_candidates', 0)} "
                    f"object_pass={info.get('object_pass', 0)} "
                    f"distance_pass={info.get('distance_pass', 0)} "
                    f"final={info.get('final_ranked', 0)}"
                )
            diagnostic_lines.append(
                f"arm={arm} expected_object={expected_object} " + " | ".join(diag_chunks)
            )
            continue
        arm_debugs[arm] = ArmDebugInfo(
            arm=arm,
            expected_object=selection.expected_object,
            ranked_candidates_per_frame=selection.ranked_candidates_per_frame,
            all_candidates_per_frame=selection.all_candidates_per_frame,
            diagnostics=selection.diagnostics,
            selected_keyframes=selection.selected_keyframes,
        )
        total_cost = sum(item.candidate.rotation_distance_deg for item in selection.selected_keyframes)
        if total_cost < best_cost:
            best_cost = total_cost
            best = selection
    if best is None:
        diag_msg = "; ".join(diagnostic_lines) if diagnostic_lines else "no diagnostics"
        raise RuntimeError(
            "Failed to find valid AnyGrasp candidates after object/distance/orientation filtering. "
            f"Diagnostics: {diag_msg}"
        )
    return best, arm_debugs


def set_actor_pose(actor: sapien.Entity, pose_world_wxyz: np.ndarray) -> None:
    pose_world_wxyz = np.asarray(pose_world_wxyz, dtype=np.float64).reshape(7)
    actor.set_pose(sapien.Pose(pose_world_wxyz[:3], base.normalize_quat_wxyz(pose_world_wxyz[3:])))


def hide_actor(actor: sapien.Entity) -> None:
    actor.set_pose(base.HIDDEN_DEBUG_POSE)


def create_colored_axis_actor(
    scene: sapien.Scene,
    name: str,
    axis_length: float,
    thickness: float,
    colors: Tuple[Sequence[float], Sequence[float], Sequence[float]],
) -> sapien.Entity:
    builder = scene.create_actor_builder()
    axis_half = axis_length * 0.5
    builder.add_sphere_visual(radius=thickness * 1.8, material=[1.0, 1.0, 1.0])
    builder.add_box_visual(pose=sapien.Pose([axis_half, 0.0, 0.0]), half_size=[axis_half, thickness, thickness], material=list(colors[0]))
    builder.add_box_visual(pose=sapien.Pose([0.0, axis_half, 0.0]), half_size=[thickness, axis_half, thickness], material=list(colors[1]))
    builder.add_box_visual(pose=sapien.Pose([0.0, 0.0, axis_half]), half_size=[thickness, thickness, axis_half], material=list(colors[2]))
    actor = builder.build_kinematic(name=name)
    hide_actor(actor)
    return actor


def create_gripper_candidate_actor(
    scene: sapien.Scene,
    name: str,
    color: Sequence[float],
    marker_side: str = "none",
    scale: float = 1.0,
    opening_width_m: float = 0.04,
) -> sapien.Entity:
    builder = scene.create_actor_builder()
    scale = float(scale)
    body_half = [0.008 * scale, 0.026 * scale, 0.004 * scale]
    finger_half = [0.018 * scale, 0.0035 * scale, 0.0035 * scale]
    finger_gap = float(np.clip(opening_width_m, 0.0, 0.12)) * 0.5 + finger_half[1]
    builder.add_box_visual(pose=sapien.Pose([-0.012 * scale, 0.0, 0.0]), half_size=body_half, material=list(color))
    builder.add_box_visual(pose=sapien.Pose([0.012 * scale, finger_gap, 0.0]), half_size=finger_half, material=list(color))
    builder.add_box_visual(pose=sapien.Pose([0.012 * scale, -finger_gap, 0.0]), half_size=finger_half, material=list(color))
    marker_color = [0.05, 0.05, 0.05]
    if marker_side == "left":
        builder.add_box_visual(pose=sapien.Pose([0.0, 0.0, 0.012]), half_size=[0.004, 0.004, 0.002], material=marker_color)
    elif marker_side == "right":
        builder.add_box_visual(pose=sapien.Pose([0.0, 0.0, -0.012]), half_size=[0.004, 0.004, 0.002], material=marker_color)
    actor = builder.build_kinematic(name=name)
    hide_actor(actor)
    return actor



def capture_head_bgr(renderer: ReplayRenderer) -> np.ndarray:
    head_rgb, _ = renderer.capture_camera(renderer.zed_camera)
    return cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)


def project_world_point_to_image(
    camera,
    intrinsic: np.ndarray,
    point_world: np.ndarray,
    image_width: int,
    image_height: int,
) -> Optional[Tuple[int, int]]:
    try:
        extrinsic = np.asarray(camera.get_extrinsic_matrix(), dtype=np.float64)
    except Exception:
        return None
    if extrinsic.shape == (3, 4):
        extrinsic_4x4 = np.eye(4, dtype=np.float64)
        extrinsic_4x4[:3, :4] = extrinsic
        extrinsic = extrinsic_4x4
    if extrinsic.shape != (4, 4):
        return None

    point_h = np.ones(4, dtype=np.float64)
    point_h[:3] = np.asarray(point_world, dtype=np.float64).reshape(3)
    point_cam = extrinsic @ point_h
    z = float(point_cam[2])
    if not np.isfinite(z) or z <= 1e-6:
        return None

    pixel = np.asarray(intrinsic, dtype=np.float64) @ point_cam[:3]
    if not np.isfinite(pixel).all() or abs(pixel[2]) <= 1e-6:
        return None
    u = int(round(float(pixel[0] / pixel[2])))
    v = int(round(float(pixel[1] / pixel[2])))
    if u < -32 or u >= image_width + 32 or v < -32 or v >= image_height + 32:
        return None
    return u, v


def draw_small_candidate_label(
    image_bgr: np.ndarray,
    text: str,
    pixel_xy: Tuple[int, int],
    color_bgr: Tuple[int, int, int],
    font_scale: float = 0.20,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    x, y = int(pixel_xy[0]), int(pixel_xy[1])
    cv2.putText(image_bgr, text, (x, y), font, float(font_scale), color_bgr, thickness, cv2.LINE_AA)


def annotate_candidate_labels(
    image_bgr: np.ndarray,
    camera,
    intrinsic: np.ndarray,
    active_frame: Optional[int],
    common_candidates_per_frame: Dict[int, List[CandidatePose]],
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]],
    selected_keyframes: List[SelectedKeyframe],
) -> None:
    if active_frame is None:
        return

    width = int(image_bgr.shape[1])
    height = int(image_bgr.shape[0])
    frame = int(active_frame)

    selected_by_frame_and_arm = {
        (int(item.source_frame), item.arm): int(item.candidate.candidate_idx) for item in selected_keyframes
    }

    common_candidates = common_candidates_per_frame.get(frame, [])
    for cand in common_candidates:
        origin = np.asarray(cand.pose_world_wxyz[:3], dtype=np.float64)
        pixel = project_world_point_to_image(camera, intrinsic, origin, width, height)
        if pixel is None:
            continue
        draw_small_candidate_label(image_bgr, str(int(cand.candidate_idx)), (pixel[0] + 2, pixel[1] - 2), (0, 150, 0), font_scale=0.15)

    arm_styles = {
        "left": ((220, 120, 20), np.array([0.0, 0.0, 0.018], dtype=np.float64)),
        "right": ((0, 140, 255), np.array([0.0, 0.0, -0.018], dtype=np.float64)),
    }
    for arm_name, (label_color, local_offset) in arm_styles.items():
        frame_candidates = arm_display_candidates.get(arm_name, {}).get(frame, [])
        selected_idx = selected_by_frame_and_arm.get((frame, arm_name))
        for cand in frame_candidates:
            pose_world = np.asarray(cand.pose_world_matrix, dtype=np.float64)
            label_world = pose_world[:3, 3] + pose_world[:3, :3] @ local_offset
            pixel = project_world_point_to_image(camera, intrinsic, label_world, width, height)
            if pixel is None:
                continue
            color = (0, 0, 255) if selected_idx is not None and int(cand.candidate_idx) == selected_idx else label_color
            draw_small_candidate_label(image_bgr, str(int(cand.candidate_idx)), (pixel[0] + 2, pixel[1] - 2), color)


def selected_keyframes_for_active_frame(
    selected_keyframes: Sequence[SelectedKeyframe],
    active_frame: Optional[int],
) -> List[SelectedKeyframe]:
    if active_frame is None:
        return []
    frame = int(active_frame)
    return [item for item in selected_keyframes if int(item.source_frame) == frame]


def build_display_candidates_per_frame(
    args: argparse.Namespace,
    keyframes: Sequence[int],
    ranked_candidates_per_frame: Dict[int, List[CandidatePose]],
    all_candidates_per_frame: Dict[int, List[CandidatePose]],
) -> Dict[int, List[CandidatePose]]:
    display: Dict[int, List[CandidatePose]] = {}
    common_top_k = int(args.debug_common_candidate_top_k)
    for frame in keyframes:
        frame = int(frame)
        if bool(args.debug_show_all_candidates):
            ranked_common = sorted(all_candidates_per_frame.get(frame, []), key=lambda item: float(item.score), reverse=True)
            display[frame] = ranked_common if common_top_k < 0 else ranked_common[: max(common_top_k, 0)]
        else:
            display[frame] = []
    return display


def create_debug_visual_bundle(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    keyframes: Sequence[int],
    common_candidates_per_frame: Dict[int, List[CandidatePose]],
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]],
    selected_keyframes: List[SelectedKeyframe],
) -> DebugVisualBundle:
    axis_colors = {
        int(keyframes[0]): ([1.0, 0.35, 0.10], [1.0, 0.65, 0.15], [1.0, 0.90, 0.20]),
        int(keyframes[1]): ([0.15, 0.85, 1.0], [0.15, 0.45, 1.0], [0.55, 0.25, 1.0]),
    }
    keyframe_axis_actors = {
        (int(frame), arm_name): create_colored_axis_actor(
            renderer.scene,
            f"debug_axis_keyframe_{int(frame)}_{arm_name}",
            axis_length=float(args.debug_target_axis_length),
            thickness=float(args.debug_target_axis_thickness),
            colors=axis_colors[int(frame)],
        )
        for frame in keyframes
        for arm_name in ("left", "right")
    }
    selected_by_frame_and_arm = {
        (int(item.source_frame), item.arm): item.candidate.candidate_idx for item in selected_keyframes
    }
    common_candidate_actors = {
        int(frame): [
            create_gripper_candidate_actor(
                renderer.scene,
                f"debug_common_candidate_{int(frame)}_{rank}",
                color=[0.1, 0.85, 0.2],
                marker_side="none",
                scale=0.82,
                opening_width_m=float(_cand.width_m),
            )
            for rank, _cand in enumerate(common_candidates_per_frame.get(int(frame), []))
        ]
        for frame in keyframes
    }
    arm_candidate_actors: Dict[str, Dict[int, List[sapien.Entity]]] = {}
    arm_candidate_axis_actors: Dict[str, Dict[int, List[sapien.Entity]]] = {}
    standard_axis_colors = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.4, 1.0])
    for arm_name, frame_candidates in arm_display_candidates.items():
        arm_candidate_actors[arm_name] = {}
        arm_candidate_axis_actors[arm_name] = {}
        for frame in keyframes:
            frame = int(frame)
            selected_idx = selected_by_frame_and_arm.get((frame, arm_name))
            actors = []
            axis_actors = []
            for rank, cand in enumerate(frame_candidates.get(frame, [])):
                actors.append(
                    create_gripper_candidate_actor(
                        renderer.scene,
                        f"debug_{arm_name}_candidate_{frame}_{rank}",
                        color=[1.0, 0.1, 0.1] if cand.candidate_idx == selected_idx else ([0.1, 0.45, 1.0] if arm_name == "left" else [1.0, 0.55, 0.0]),
                        marker_side="none",
                        scale=1.65 if cand.candidate_idx == selected_idx else 1.0,
                        opening_width_m=float(cand.width_m),
                    )
                )
                axis_actors.append(
                    create_colored_axis_actor(
                        renderer.scene,
                        f"debug_{arm_name}_candidate_axis_{frame}_{rank}",
                        axis_length=float(args.debug_target_axis_length) * 0.72,
                        thickness=float(args.debug_target_axis_thickness) * 0.72,
                        colors=standard_axis_colors,
                    )
                )
            arm_candidate_actors[arm_name][frame] = actors
            arm_candidate_axis_actors[arm_name][frame] = axis_actors
    return DebugVisualBundle(
        keyframe_axis_actors=keyframe_axis_actors,
        common_candidate_actors=common_candidate_actors,
        arm_candidate_actors=arm_candidate_actors,
        arm_candidate_axis_actors=arm_candidate_axis_actors,
    )


def set_single_arm_target_visual(renderer: ReplayRenderer, arm: str, pose_world_wxyz: Optional[np.ndarray]) -> None:
    if pose_world_wxyz is None:
        renderer.update_target_axis_visuals(None, None)
        return
    if arm == "left":
        renderer.update_target_axis_visuals(pose_world_wxyz, None)
    elif arm == "right":
        renderer.update_target_axis_visuals(None, pose_world_wxyz)
    else:
        renderer.update_target_axis_visuals(None, None)


def set_dual_arm_target_visuals(
    renderer: ReplayRenderer,
    left_pose_world_wxyz: Optional[np.ndarray],
    right_pose_world_wxyz: Optional[np.ndarray],
) -> None:
    renderer.update_target_axis_visuals(left_pose_world_wxyz, right_pose_world_wxyz)


def record_frame(
    renderer: ReplayRenderer,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    overlay_lines: Sequence[str],
    use_overlay: bool,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
) -> None:
    viewer_active = bool(getattr(renderer, "viewer", None) is not None and not renderer.viewer.closed)
    if debug_visuals is not None:
        for actor in debug_visuals.keyframe_axis_actors.values():
            hide_actor(actor)
        for actors in debug_visuals.common_candidate_actors.values():
            for actor in actors:
                hide_actor(actor)
        for per_frame in debug_visuals.arm_candidate_actors.values():
            for actors in per_frame.values():
                for actor in actors:
                    hide_actor(actor)
        for per_frame in debug_visuals.arm_candidate_axis_actors.values():
            for actors in per_frame.values():
                for actor in actors:
                    hide_actor(actor)
        if debug_execution_state is not None:
            for item in selected_keyframes_for_active_frame(
                debug_execution_state.selected_keyframes,
                debug_execution_state.active_frame,
            ):
                actor = debug_visuals.keyframe_axis_actors.get((int(item.source_frame), item.arm))
                if actor is not None:
                    set_actor_pose(actor, item.candidate.pose_world_wxyz)
            update_candidate_debug_visuals(
                debug_visuals,
                debug_execution_state.active_frame,
                debug_execution_state.common_candidates_per_frame,
                debug_execution_state.arm_display_candidates,
            )
    renderer.update_robot_link_cameras()
    renderer.scene.update_render()
    if viewer_active:
        renderer.viewer.render()
    head_rgb, _ = renderer.capture_camera(renderer.zed_camera)
    head_bgr = base.overlay_text(head_rgb, overlay_lines) if use_overlay else cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
    head_writer.write(head_bgr)
    if third_writer is not None:
        third_rgb, _ = renderer.capture_camera(renderer.third_camera)
        third_bgr = base.overlay_text(third_rgb, overlay_lines) if use_overlay else cv2.cvtColor(third_rgb, cv2.COLOR_RGB2BGR)
        third_writer.write(third_bgr)
    if debug_execution_state is not None and debug_execution_state.writer is not None and debug_visuals is not None:
        renderer.update_robot_link_cameras()
        renderer.scene.update_render()
        debug_overlay = list(overlay_lines)
        debug_overlay.extend(
            [
                "mode=debug_execution",
                (
                    f"active_keyframe={int(debug_execution_state.active_frame)}"
                    if debug_execution_state.active_frame is not None
                    else "active_keyframe=none"
                ),
                "color=green:raw blue:left orange:right red:selected",
                "labels=tiny candidate_idx",
            ]
        )
        debug_rgb, _ = renderer.capture_camera(renderer.zed_camera)
        debug_bgr = base.overlay_text(debug_rgb, debug_overlay) if use_overlay else cv2.cvtColor(debug_rgb, cv2.COLOR_RGB2BGR)
        annotate_candidate_labels(
            debug_bgr,
            renderer.zed_camera,
            debug_execution_state.head_intrinsic,
            debug_execution_state.active_frame,
            debug_execution_state.common_candidates_per_frame,
            debug_execution_state.arm_display_candidates,
            selected_keyframes_for_active_frame(
                debug_execution_state.selected_keyframes,
                debug_execution_state.active_frame,
            ),
        )
        debug_execution_state.writer.write(debug_bgr)
    if debug_execution_state is not None and debug_execution_state.pose_debug_path is not None:
        current_head_pose = renderer.get_head_camera_pose()
        object_actor_poses = {}
        if debug_execution_state.object_tracks is not None:
            for name, track in debug_execution_state.object_tracks.items():
                if track.actor is None:
                    continue
                actor_pose = track.actor.get_pose()
                object_actor_poses[name] = {
                    "actor_pose_world_wxyz": (
                        np.asarray(actor_pose.p, dtype=np.float64).reshape(3).tolist()
                        + base.normalize_quat_wxyz(np.asarray(actor_pose.q, dtype=np.float64).reshape(4)).tolist()
                    )
                }
                if debug_execution_state.active_frame is not None:
                    frame_to_idx = {int(frame): idx for idx, frame in enumerate(track.frame_indices.tolist())}
                    idx = frame_to_idx.get(int(debug_execution_state.active_frame))
                    if idx is not None:
                        object_actor_poses[name]["replay_pose_world_wxyz"] = np.asarray(track.pose_world_wxyz[idx], dtype=np.float64).tolist()
        pose_debug = {
            "active_frame": None if debug_execution_state.active_frame is None else int(debug_execution_state.active_frame),
            "overlay_lines": list(overlay_lines),
            "current_head_camera_pose_world_wxyz": (
                np.asarray(current_head_pose.p, dtype=np.float64).reshape(3).tolist()
                + base.normalize_quat_wxyz(np.asarray(current_head_pose.q, dtype=np.float64).reshape(4)).tolist()
            ),
            "current_left_tcp_pose_world_wxyz": np.asarray(renderer.get_current_tcp_pose("left"), dtype=np.float64).tolist(),
            "current_right_tcp_pose_world_wxyz": np.asarray(renderer.get_current_tcp_pose("right"), dtype=np.float64).tolist(),
            "replay_head_camera_pose_world_wxyz": None
            if debug_execution_state.replay_head_camera_pose_by_frame is None or debug_execution_state.active_frame is None
            else np.asarray(
                debug_execution_state.replay_head_camera_pose_by_frame.get(int(debug_execution_state.active_frame), np.full(7, np.nan)),
                dtype=np.float64,
            ).tolist(),
            "object_actor_poses": object_actor_poses,
        }
        with debug_execution_state.pose_debug_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(pose_debug, ensure_ascii=False) + "\n")
    if debug_visuals is not None and not viewer_active:
        for actors in debug_visuals.common_candidate_actors.values():
            for actor in actors:
                hide_actor(actor)
        for per_frame in debug_visuals.arm_candidate_actors.values():
            for actors in per_frame.values():
                for actor in actors:
                    hide_actor(actor)
        for per_frame in debug_visuals.arm_candidate_axis_actors.values():
            for actors in per_frame.values():
                for actor in actors:
                    hide_actor(actor)
        for actor in debug_visuals.keyframe_axis_actors.values():
            hide_actor(actor)


def pose_with_offset_along_local_x(pose_world_wxyz: np.ndarray, offset_m: float) -> np.ndarray:
    pose_world_wxyz = np.asarray(pose_world_wxyz, dtype=np.float64).reshape(7)
    pose_world_matrix = pose_wxyz_to_matrix(pose_world_wxyz)
    pose_world_matrix[:3, 3] -= pose_world_matrix[:3, 0] * float(offset_m)
    rot = pose_world_matrix[:3, :3]
    quat = base.quat_xyzw_to_wxyz(R.from_matrix(rot).as_quat())
    return np.concatenate([pose_world_matrix[:3, 3], quat]).astype(np.float64)


def sapien_pose_to_wxyz(pose: sapien.Pose) -> np.ndarray:
    return np.concatenate([np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64)]).astype(np.float64)


def get_current_pose_for_error(renderer: ReplayRenderer, arm: str, pose_source: str) -> np.ndarray:
    if renderer.robot is None:
        raise RuntimeError("Robot is unavailable.")
    if pose_source == "tcp":
        return renderer.get_current_tcp_pose(arm)
    if pose_source == "ee":
        pose = renderer.robot.get_left_ee_pose() if arm == "left" else renderer.robot.get_right_ee_pose()
        return np.asarray(pose, dtype=np.float64)
    raise ValueError(f"Unsupported pose_source: {pose_source}")


def target_pose_for_error(renderer: ReplayRenderer, arm: str, target_tcp_pose_world_wxyz: np.ndarray, pose_source: str) -> np.ndarray:
    if renderer.robot is None:
        raise RuntimeError("Robot is unavailable.")
    target_tcp_pose_world_wxyz = np.asarray(target_tcp_pose_world_wxyz, dtype=np.float64).reshape(7)
    if pose_source == "tcp":
        return target_tcp_pose_world_wxyz
    if pose_source == "ee":
        target_pose_base = renderer.world_pose_to_base_pose(target_tcp_pose_world_wxyz)
        target_pose_ee_base = renderer.robot._trans_from_gripper_to_endlink(target_pose_base.tolist(), arm_tag=arm)
        base_world = base.pose_to_matrix(renderer._base_pose)
        ee_base = base.pose_to_matrix(target_pose_ee_base)
        ee_world = base_world @ ee_base
        quat = base.quat_xyzw_to_wxyz(R.from_matrix(base.orthonormalize_rotation(ee_world[:3, :3])).as_quat())
        return np.concatenate([ee_world[:3, 3], quat]).astype(np.float64)
    raise ValueError(f"Unsupported pose_source: {pose_source}")


def build_supervision_targets(
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]],
    selected_keyframes: List[SelectedKeyframe],
) -> Dict[str, np.ndarray]:
    targets: Dict[str, np.ndarray] = {}
    active_frame = int(selected_keyframes[0].source_frame)
    for arm_name, frame_map in arm_display_candidates.items():
        frame_candidates = frame_map.get(active_frame, [])
        if frame_candidates:
            targets[arm_name] = np.asarray(frame_candidates[0].pose_world_wxyz, dtype=np.float64)
    for item in selected_keyframes:
        if int(item.source_frame) == active_frame:
            targets[item.arm] = np.asarray(item.candidate.pose_world_wxyz, dtype=np.float64)
    return targets


def compute_supervision_errors(
    renderer: ReplayRenderer,
    supervision_targets: Dict[str, np.ndarray],
    pose_source: str,
) -> Dict[str, Dict[str, float]]:
    errors: Dict[str, Dict[str, float]] = {}
    for arm_name, target_pose in supervision_targets.items():
        current_pose = get_current_pose_for_error(renderer, arm_name, pose_source)
        target_eval_pose = target_pose_for_error(renderer, arm_name, target_pose, pose_source)
        pos_err, rot_err = tcp_pose_errors(target_eval_pose, current_pose)
        errors[arm_name] = {
            "pose_source": pose_source,
            "pos_err_m": float(pos_err),
            "rot_err_deg": float(rot_err),
        }
    return errors


def tcp_pose_errors(target_pose_world_wxyz: np.ndarray, current_pose_world_wxyz: np.ndarray) -> Tuple[float, float]:
    target_pose_world_wxyz = np.asarray(target_pose_world_wxyz, dtype=np.float64).reshape(7)
    current_pose_world_wxyz = np.asarray(current_pose_world_wxyz, dtype=np.float64).reshape(7)
    pos_err = float(np.linalg.norm(target_pose_world_wxyz[:3] - current_pose_world_wxyz[:3]))
    rot_err = float(base.quat_angle_deg_wxyz(current_pose_world_wxyz[3:], target_pose_world_wxyz[3:]))
    return pos_err, rot_err


def interpolate_joint_trajectory(position: np.ndarray, velocity: np.ndarray, num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    if position.shape[0] < 2 or int(num_steps) <= position.shape[0]:
        return position, velocity

    out_pos = []
    dof = position.shape[1]
    zero_vel = np.zeros(dof, dtype=np.float64)
    for idx in range(position.shape[0] - 1):
        start = position[idx]
        end = position[idx + 1]
        seg = np.linspace(start, end, num=max(int(num_steps / max(position.shape[0] - 1, 1)), 2), endpoint=False)
        out_pos.extend(seg.tolist())
    out_pos.append(position[-1].tolist())
    out_pos_arr = np.asarray(out_pos, dtype=np.float64)
    out_vel_arr = np.tile(zero_vel[None, :], (out_pos_arr.shape[0], 1))
    return out_pos_arr, out_vel_arr


def apply_robot_init_pose(renderer: ReplayRenderer, open_gripper: float, settle_steps: int) -> Dict[str, object]:
    left_init = getattr(renderer, "init_left_arm_joints", None)
    right_init = getattr(renderer, "init_right_arm_joints", None)
    init_gripper_open = getattr(renderer, "init_gripper_open", None)
    if init_gripper_open is None:
        init_gripper_open = float(open_gripper)

    vel = np.zeros(6, dtype=np.float64)
    left_applied = False
    right_applied = False
    if left_init is not None:
        renderer.robot.set_arm_joints(np.asarray(left_init, dtype=np.float64).reshape(6), vel, "left")
        left_applied = True
    if right_init is not None:
        renderer.robot.set_arm_joints(np.asarray(right_init, dtype=np.float64).reshape(6), vel, "right")
        right_applied = True

    if left_applied or right_applied:
        renderer.step_scene(steps=max(int(settle_steps), 1))
    renderer.set_grippers(float(init_gripper_open), float(init_gripper_open))
    renderer.step_scene(steps=1)

    return {
        "left_applied": bool(left_applied),
        "right_applied": bool(right_applied),
        "left_joints": None if left_init is None else np.asarray(left_init, dtype=np.float64).reshape(6),
        "right_joints": None if right_init is None else np.asarray(right_init, dtype=np.float64).reshape(6),
        "gripper_open": float(init_gripper_open),
    }


def emit_init_prefix_frames(
    renderer: ReplayRenderer,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    use_overlay: bool,
    init_info: Dict[str, object],
    fixed_frames: int,
    arm_label: str,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
) -> int:
    total = max(int(fixed_frames), 0)
    if total <= 0:
        return 0
    for idx in range(total):
        record_frame(
            renderer,
            head_writer,
            third_writer,
            [
                "stage=init_prefix",
                f"arm={arm_label}",
                f"fixed_frame={idx + 1}/{total}",
                f"gripper_open={float(init_info['gripper_open']):.3f}",
            ],
            use_overlay,
            debug_visuals,
            debug_execution_state,
        )
    return total


def execute_single_arm_plan(
    renderer: ReplayRenderer,
    arm: str,
    plan: Optional[Dict],
    label: str,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    use_overlay: bool,
    attached_actor: Optional[sapien.Entity] = None,
    tcp_to_object: Optional[np.ndarray] = None,
    execute_interp_steps: int = 24,
    settle_steps: int = 4,
    hold_frames_after_stage: int = 0,
    target_visual_pose: Optional[np.ndarray] = None,
    target_visual_label: Optional[str] = None,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
    object_replay: Optional[ExecutionObjectReplayConfig] = None,
) -> str:
    status = renderer._plan_status(plan)
    overlay_lines = [f"stage={label}", f"arm={arm}", f"status={status}"]
    if target_visual_label:
        overlay_lines.append(f"goal={target_visual_label}")
    set_single_arm_target_visual(renderer, arm, target_visual_pose)
    if status != "Success":
        record_frame(renderer, head_writer, third_writer, overlay_lines, use_overlay, debug_visuals, debug_execution_state)
        return status

    position = np.asarray(plan["position"], dtype=np.float64)
    velocity = np.asarray(plan["velocity"], dtype=np.float64)
    position, velocity = interpolate_joint_trajectory(position, velocity, execute_interp_steps)
    for idx in range(position.shape[0]):
        renderer.robot.set_arm_joints(position[idx], velocity[idx], arm)
        if object_replay is not None:
            update_execution_object_replay(object_replay, 0.0 if position.shape[0] <= 1 else float(idx) / float(position.shape[0] - 1))
        elif attached_actor is not None and tcp_to_object is not None:
            tcp_pose = renderer.get_current_tcp_pose(arm)
            object_world = pose_wxyz_to_matrix(tcp_pose) @ tcp_to_object
            quat = base.quat_xyzw_to_wxyz(R.from_matrix(object_world[:3, :3]).as_quat())
            set_actor_pose(attached_actor, np.concatenate([object_world[:3, 3], quat]))
        renderer.step_scene(steps=1)
        record_frame(
            renderer,
            head_writer,
            third_writer,
            overlay_lines + [f"plan_step={idx + 1}/{position.shape[0]}"],
            use_overlay,
            debug_visuals,
            debug_execution_state,
        )
    renderer.step_scene(steps=max(int(settle_steps), 0))
    if object_replay is not None:
        update_execution_object_replay(object_replay, 1.0)
    record_frame(renderer, head_writer, third_writer, overlay_lines + ["plan_step=done"], use_overlay, debug_visuals, debug_execution_state)
    for hold_idx in range(max(int(hold_frames_after_stage), 0)):
        record_frame(
            renderer,
            head_writer,
            third_writer,
            overlay_lines + [f"hold={hold_idx + 1}/{hold_frames_after_stage}"],
            use_overlay,
            debug_visuals,
            debug_execution_state,
        )
    return status


def execute_stage_until_reached(
    renderer: ReplayRenderer,
    arm: str,
    target_pose_world_wxyz: np.ndarray,
    label: str,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    use_overlay: bool,
    args: argparse.Namespace,
    attached_actor: Optional[sapien.Entity] = None,
    tcp_to_object: Optional[np.ndarray] = None,
    target_visual_pose: Optional[np.ndarray] = None,
    target_visual_label: Optional[str] = None,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
    supervision_targets: Optional[Dict[str, np.ndarray]] = None,
    object_replay: Optional[ExecutionObjectReplayConfig] = None,
) -> Dict[str, object]:
    last_status = "Missing"
    last_pos_err = float("inf")
    last_rot_err = float("inf")
    attempts = 0
    attempt_history = []
    max_attempts = max(int(args.max_stage_replans), 1)
    if bool(args.replan_until_reached):
        max_attempts = max(max_attempts, int(args.replan_until_reached_max_attempts))
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        plan = renderer.plan_path(arm, target_pose_world_wxyz)
        last_status = execute_single_arm_plan(
            renderer=renderer,
            arm=arm,
            plan=plan,
            label=f"{label}_try{attempt}",
            head_writer=head_writer,
            third_writer=third_writer,
            use_overlay=use_overlay,
            attached_actor=attached_actor,
            tcp_to_object=tcp_to_object,
            execute_interp_steps=args.execute_interp_steps,
            settle_steps=args.settle_steps,
            hold_frames_after_stage=args.hold_frames_after_stage,
            target_visual_pose=target_visual_pose,
            target_visual_label=target_visual_label,
            debug_visuals=debug_visuals,
            debug_execution_state=debug_execution_state,
            object_replay=object_replay,
        )
        current_eval_pose = get_current_pose_for_error(renderer, arm, args.reach_error_pose_source)
        target_eval_pose = target_pose_for_error(renderer, arm, target_pose_world_wxyz, args.reach_error_pose_source)
        last_pos_err, last_rot_err = tcp_pose_errors(target_eval_pose, current_eval_pose)
        reached = (
            last_status == "Success"
            and last_pos_err <= float(args.reach_pos_tol_m)
            and last_rot_err <= float(args.reach_rot_tol_deg)
        )
        supervision_errors = compute_supervision_errors(renderer, supervision_targets or {}, args.reach_error_pose_source) if supervision_targets else {}
        attempt_history.append(
            {
                "attempt": attempt,
                "status": last_status,
                "pos_err_m": last_pos_err,
                "rot_err_deg": last_rot_err,
                "reached": bool(reached),
                "supervision_errors": supervision_errors,
            }
        )
        record_frame(
            renderer,
            head_writer,
            third_writer,
            [
                f"stage={label}",
                f"arm={arm}",
                f"attempt={attempt}/{max_attempts}",
                f"status={last_status}",
                f"pos_err={last_pos_err:.4f}m",
                f"rot_err={last_rot_err:.2f}deg",
                f"reached={int(reached)}",
                *([f"right_sup_pos={supervision_errors['right']['pos_err_m']:.4f}m", f"right_sup_rot={supervision_errors['right']['rot_err_deg']:.2f}deg"] if "right" in supervision_errors and arm != "right" else []),
            ],
            use_overlay,
            debug_visuals,
            debug_execution_state,
        )
        if reached:
            return {
                "status": last_status,
                "attempts": attempts,
                "reached": True,
                "pos_err_m": last_pos_err,
                "rot_err_deg": last_rot_err,
                "attempt_history": attempt_history,
            }

    return {
        "status": last_status,
        "attempts": attempts,
        "reached": False,
        "pos_err_m": last_pos_err,
        "rot_err_deg": last_rot_err,
        "attempt_history": attempt_history,
    }


def execute_dual_arm_plan(
    renderer: ReplayRenderer,
    plans_by_arm: Dict[str, Optional[Dict]],
    label: str,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    use_overlay: bool,
    execute_interp_steps: int = 24,
    settle_steps: int = 4,
    hold_frames_after_stage: int = 0,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
    attached_actor_by_arm: Optional[Dict[str, Optional[sapien.Entity]]] = None,
    tcp_to_object_by_arm: Optional[Dict[str, Optional[np.ndarray]]] = None,
    object_replay: Optional[ExecutionObjectReplayConfig] = None,
) -> Dict[str, str]:
    arms = [arm for arm in ("left", "right") if arm in plans_by_arm]
    statuses: Dict[str, str] = {arm: renderer._plan_status(plans_by_arm.get(arm)) for arm in arms}
    overlay_lines = [
        f"stage={label}",
        f"left_status={statuses.get('left', 'NA')}",
        f"right_status={statuses.get('right', 'NA')}",
    ]

    trajectories: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    max_steps = 0
    for arm in arms:
        if statuses[arm] != "Success":
            continue
        plan = plans_by_arm[arm]
        position = np.asarray(plan["position"], dtype=np.float64)
        velocity = np.asarray(plan["velocity"], dtype=np.float64)
        position, velocity = interpolate_joint_trajectory(position, velocity, execute_interp_steps)
        trajectories[arm] = (position, velocity)
        max_steps = max(max_steps, int(position.shape[0]))

    if max_steps <= 0:
        record_frame(renderer, head_writer, third_writer, overlay_lines, use_overlay, debug_visuals, debug_execution_state)
        return statuses

    for step_idx in range(max_steps):
        for arm in arms:
            if arm not in trajectories:
                continue
            position, velocity = trajectories[arm]
            local_idx = min(step_idx, int(position.shape[0]) - 1)
            renderer.robot.set_arm_joints(position[local_idx], velocity[local_idx], arm)
        for arm in arms:
            actor = None if attached_actor_by_arm is None else attached_actor_by_arm.get(arm)
            tcp_to_object = None if tcp_to_object_by_arm is None else tcp_to_object_by_arm.get(arm)
            if object_replay is not None:
                continue
            if actor is not None and tcp_to_object is not None:
                tcp_pose = renderer.get_current_tcp_pose(arm)
                object_world = pose_wxyz_to_matrix(tcp_pose) @ tcp_to_object
                quat = base.quat_xyzw_to_wxyz(R.from_matrix(object_world[:3, :3]).as_quat())
                set_actor_pose(actor, np.concatenate([object_world[:3, 3], quat]))
        if object_replay is not None:
            update_execution_object_replay(object_replay, 0.0 if max_steps <= 1 else float(step_idx) / float(max_steps - 1))
        renderer.step_scene(steps=1)
        record_frame(
            renderer,
            head_writer,
            third_writer,
            overlay_lines + [f"plan_step={step_idx + 1}/{max_steps}"],
            use_overlay,
            debug_visuals,
            debug_execution_state,
        )

    renderer.step_scene(steps=max(int(settle_steps), 0))
    if object_replay is not None:
        update_execution_object_replay(object_replay, 1.0)
    record_frame(renderer, head_writer, third_writer, overlay_lines + ["plan_step=done"], use_overlay, debug_visuals, debug_execution_state)
    for hold_idx in range(max(int(hold_frames_after_stage), 0)):
        record_frame(
            renderer,
            head_writer,
            third_writer,
            overlay_lines + [f"hold={hold_idx + 1}/{hold_frames_after_stage}"],
            use_overlay,
            debug_visuals,
            debug_execution_state,
        )
    return statuses


def execute_dual_stage_until_reached(
    renderer: ReplayRenderer,
    target_pose_world_wxyz_by_arm: Dict[str, np.ndarray],
    label: str,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    use_overlay: bool,
    args: argparse.Namespace,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
    attached_actor_by_arm: Optional[Dict[str, Optional[sapien.Entity]]] = None,
    tcp_to_object_by_arm: Optional[Dict[str, Optional[np.ndarray]]] = None,
    object_replay: Optional[ExecutionObjectReplayConfig] = None,
) -> Dict[str, object]:
    arms = [arm for arm in ("left", "right") if arm in target_pose_world_wxyz_by_arm]
    max_attempts = max(int(args.max_stage_replans), 1)
    if bool(args.replan_until_reached):
        max_attempts = max(max_attempts, int(args.replan_until_reached_max_attempts))

    attempt_history: List[Dict[str, object]] = []
    last_arm_metrics: Dict[str, Dict[str, object]] = {}
    for attempt in range(1, max_attempts + 1):
        plans_by_arm: Dict[str, Optional[Dict]] = {
            arm: renderer.plan_path(arm, target_pose_world_wxyz_by_arm[arm]) for arm in arms
        }
        statuses = execute_dual_arm_plan(
            renderer=renderer,
            plans_by_arm=plans_by_arm,
            label=f"{label}_try{attempt}",
            head_writer=head_writer,
            third_writer=third_writer,
            use_overlay=use_overlay,
            execute_interp_steps=args.execute_interp_steps,
            settle_steps=args.settle_steps,
            hold_frames_after_stage=args.hold_frames_after_stage,
            debug_visuals=debug_visuals,
            debug_execution_state=debug_execution_state,
            attached_actor_by_arm=attached_actor_by_arm,
            tcp_to_object_by_arm=tcp_to_object_by_arm,
            object_replay=object_replay,
        )

        arm_metrics: Dict[str, Dict[str, object]] = {}
        for arm in arms:
            current_eval_pose = get_current_pose_for_error(renderer, arm, args.reach_error_pose_source)
            target_eval_pose = target_pose_for_error(renderer, arm, target_pose_world_wxyz_by_arm[arm], args.reach_error_pose_source)
            pos_err, rot_err = tcp_pose_errors(target_eval_pose, current_eval_pose)
            reached = (
                statuses.get(arm, "Missing") == "Success"
                and pos_err <= float(args.reach_pos_tol_m)
                and rot_err <= float(args.reach_rot_tol_deg)
            )
            arm_metrics[arm] = {
                "status": statuses.get(arm, "Missing"),
                "pos_err_m": float(pos_err),
                "rot_err_deg": float(rot_err),
                "reached": bool(reached),
            }

        stage_reached = all(bool(arm_metrics[arm]["reached"]) for arm in arms)
        attempt_history.append(
            {
                "attempt": attempt,
                "arms": arm_metrics,
                "reached": bool(stage_reached),
            }
        )

        overlay_lines = [
            f"stage={label}",
            f"attempt={attempt}/{max_attempts}",
            f"left_reached={int(arm_metrics.get('left', {}).get('reached', False))}",
            f"right_reached={int(arm_metrics.get('right', {}).get('reached', False))}",
            f"both_reached={int(stage_reached)}",
        ]
        record_frame(renderer, head_writer, third_writer, overlay_lines, use_overlay, debug_visuals, debug_execution_state)
        last_arm_metrics = arm_metrics
        if stage_reached:
            return {
                "attempts": attempt,
                "reached": True,
                "arms": arm_metrics,
                "attempt_history": attempt_history,
            }

    return {
        "attempts": max_attempts,
        "reached": False,
        "arms": last_arm_metrics,
        "attempt_history": attempt_history,
    }


def pause_after_keyframe1(
    renderer: ReplayRenderer,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    use_overlay: bool,
    args: argparse.Namespace,
    arm_label: str,
    goal_frame: int,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
) -> int:
    seconds = max(float(args.pause_after_keyframe1_seconds), 0.0)
    if seconds <= 0.0:
        return 0
    num_frames = max(int(round(seconds * float(args.fps))), 1)
    viewer_sleep = max(float(args.viewer_frame_delay), 0.0)
    print(f"[stage] reached keyframe_{int(goal_frame)} arm={arm_label}; pausing for {seconds:.2f}s before next target")
    for pause_idx in range(num_frames):
        record_frame(
            renderer,
            head_writer,
            third_writer,
            [
                "stage=pause_after_keyframe1",
                f"arm={arm_label}",
                f"goal=keyframe_{int(goal_frame)}",
                f"pause_frame={pause_idx + 1}/{num_frames}",
                f"pause_seconds={seconds:.2f}",
            ],
            use_overlay,
            debug_visuals,
            debug_execution_state,
        )
        if viewer_sleep > 0.0:
            time.sleep(viewer_sleep)
    return num_frames


def update_candidate_debug_visuals(
    debug_visuals: DebugVisualBundle,
    active_frame: Optional[int],
    common_candidates_per_frame: Dict[int, List[CandidatePose]],
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]],
) -> None:
    for frame, actors in debug_visuals.common_candidate_actors.items():
        candidates = common_candidates_per_frame.get(int(frame), [])
        for idx, actor in enumerate(actors):
            if active_frame is not None and int(frame) == int(active_frame) and idx < len(candidates):
                set_actor_pose(actor, candidates[idx].pose_world_wxyz)
            else:
                hide_actor(actor)
    for arm_name, per_frame in debug_visuals.arm_candidate_actors.items():
        candidates_map = arm_display_candidates.get(arm_name, {})
        axis_map = debug_visuals.arm_candidate_axis_actors.get(arm_name, {})
        for frame, actors in per_frame.items():
            candidates = candidates_map.get(int(frame), [])
            axis_actors = axis_map.get(frame, [])
            for idx, actor in enumerate(actors):
                if active_frame is not None and int(frame) == int(active_frame) and idx < len(candidates):
                    set_actor_pose(actor, candidates[idx].pose_world_wxyz)
                    if idx < len(axis_actors):
                        set_actor_pose(axis_actors[idx], candidates[idx].pose_world_wxyz)
                else:
                    hide_actor(actor)
                    if idx < len(axis_actors):
                        hide_actor(axis_actors[idx])




def export_rank_preview_images(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    keyframes: Sequence[int],
    object_states_per_frame: Dict[int, Dict[str, ObjectState]],
    arm_debugs: Dict[str, ArmDebugInfo],
    selected_keyframes: List[SelectedKeyframe],
    head_intrinsic: np.ndarray,
) -> List[RankPreviewRecord]:
    if not bool(args.save_rank_preview_images):
        return []

    preview_dir = args.output_dir / "rank_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    top_n = max(int(args.rank_preview_top_n), 0)
    if top_n <= 0:
        return []

    temp_actors = {
        "left": create_gripper_candidate_actor(
            renderer.scene,
            "rank_preview_left",
            color=[0.1, 0.45, 1.0],
            scale=1.35,
            opening_width_m=0.04,
        ),
        "right": create_gripper_candidate_actor(
            renderer.scene,
            "rank_preview_right",
            color=[1.0, 0.55, 0.0],
            scale=1.35,
            opening_width_m=0.04,
        ),
    }
    selected_map = {int(item.source_frame): (item.arm, int(item.candidate.candidate_idx)) for item in selected_keyframes}
    records: List[RankPreviewRecord] = []

    try:
        for frame in keyframes:
            frame = int(frame)
            object_states = object_states_per_frame[frame]
            for state in object_states.values():
                if state.actor is None:
                    continue
                if state.visible:
                    set_actor_pose(state.actor, state.pose_world_wxyz)
                else:
                    hide_actor(state.actor)

            left_ranked = arm_debugs.get("left", ArmDebugInfo("left", None, {}, {}, {})).ranked_candidates_per_frame.get(frame, [])
            right_ranked = arm_debugs.get("right", ArmDebugInfo("right", None, {}, {}, {})).ranked_candidates_per_frame.get(frame, [])

            for rank_idx in range(top_n):
                left_cand = left_ranked[rank_idx] if rank_idx < len(left_ranked) else None
                right_cand = right_ranked[rank_idx] if rank_idx < len(right_ranked) else None
                if left_cand is None and right_cand is None:
                    continue

                if left_cand is not None:
                    set_actor_pose(temp_actors["left"], left_cand.pose_world_wxyz)
                else:
                    hide_actor(temp_actors["left"])
                if right_cand is not None:
                    set_actor_pose(temp_actors["right"], right_cand.pose_world_wxyz)
                else:
                    hide_actor(temp_actors["right"])

                renderer.update_robot_link_cameras()
                renderer.scene.update_render()
                overlay_lines = [
                    "mode=rank_preview",
                    f"source_frame={frame}",
                    f"rank={rank_idx + 1}",
                    (
                        f"left_candidate={left_cand.candidate_idx}"
                        + (" selected" if selected_map.get(frame) == ("left", int(left_cand.candidate_idx)) else "")
                        if left_cand is not None
                        else "left_candidate=none"
                    ),
                    (
                        f"right_candidate={right_cand.candidate_idx}"
                        + (" selected" if selected_map.get(frame) == ("right", int(right_cand.candidate_idx)) else "")
                        if right_cand is not None
                        else "right_candidate=none"
                    ),
                    "color=blue:left orange:right",
                ]
                head_rgb, _ = renderer.capture_camera(renderer.zed_camera)
                image_bgr = base.overlay_text(head_rgb, overlay_lines)
                annotate_candidate_labels(
                    image_bgr,
                    renderer.zed_camera,
                    head_intrinsic,
                    frame,
                    {},
                    {
                        "left": {frame: [] if left_cand is None else [left_cand]},
                        "right": {frame: [] if right_cand is None else [right_cand]},
                    },
                    selected_keyframes,
                )
                out_path = preview_dir / f"keyframe_{frame:06d}_rank_{rank_idx + 1}.png"
                if not cv2.imwrite(str(out_path), image_bgr):
                    raise RuntimeError(f"Failed to write {out_path}")
                records.append(
                    RankPreviewRecord(
                        frame=frame,
                        rank=rank_idx + 1,
                        image_path=str(out_path),
                        left_candidate_idx=None if left_cand is None else int(left_cand.candidate_idx),
                        right_candidate_idx=None if right_cand is None else int(right_cand.candidate_idx),
                    )
                )
    finally:
        for actor in temp_actors.values():
            hide_actor(actor)
    return records


def generate_debug_preview(
    renderer: ReplayRenderer,
    args: argparse.Namespace,
    replay_frame_indices: np.ndarray,
    object_tracks: Dict[str, ObjectTrack],
    selected_keyframes: List[SelectedKeyframe],
    common_candidates_per_frame: Dict[int, List[CandidatePose]],
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]],
    head_intrinsic: np.ndarray,
    debug_visuals: DebugVisualBundle,
) -> Optional[Path]:
    if not bool(args.save_debug_preview):
        renderer.update_target_axis_visuals(None, None)
        return None

    debug_video_path = args.output_dir / "debug_selection_preview.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(debug_video_path), fourcc, int(args.debug_preview_fps), (args.image_width, args.image_height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open {debug_video_path}")

    selected_by_frame = {int(item.source_frame): item for item in selected_keyframes}
    selected_frames = sorted(selected_by_frame.keys())
    for item in selected_keyframes:
        actor = debug_visuals.keyframe_axis_actors.get((int(item.source_frame), item.arm))
        if actor is not None:
            set_actor_pose(actor, item.candidate.pose_world_wxyz)
    try:
        for idx, source_frame in enumerate(replay_frame_indices.tolist()):
            overlay_lines = [f"mode=debug_preview", f"source_frame={source_frame}"]
            for name, track in object_tracks.items():
                if track.actor is None:
                    continue
                is_visible = bool(track.visible[idx])
                if is_visible:
                    set_actor_pose(track.actor, track.pose_world_wxyz[idx])
                else:
                    hide_actor(track.actor)
                overlay_lines.append(f"{name}={int(is_visible)}")

            selected = None
            for keyframe in selected_frames:
                if int(source_frame) >= keyframe and int(source_frame) < keyframe + max(int(args.debug_keyframe_hold_frames), 1):
                    selected = selected_by_frame[keyframe]
                    break
            if selected is None:
                selected = selected_by_frame.get(int(source_frame))
            if selected is not None:
                update_candidate_debug_visuals(
                    debug_visuals,
                    int(selected.source_frame),
                    common_candidates_per_frame,
                    arm_display_candidates,
                )
                overlay_lines.extend(
                    [
                        f"selected_keyframe={selected.source_frame}",
                        f"selected_arm={selected.arm}",
                        f"expected_object={selected.candidate.nearest_object}",
                        f"selected_object={selected.candidate.nearest_object}",
                        f"candidate_idx={selected.candidate.candidate_idx}",
                        f"rot_err={selected.candidate.rotation_distance_deg:.2f}deg",
                        f"raw_candidates={len(common_candidates_per_frame.get(int(selected.source_frame), []))}",
                        f"per_arm_top_k={int(args.debug_candidate_top_k)}",
                        "color=green:raw blue:left orange:right red:selected",
                        "labels=tiny candidate_idx",
                    ]
                )
            else:
                update_candidate_debug_visuals(
                    debug_visuals,
                    None,
                    common_candidates_per_frame,
                    arm_display_candidates,
                )

            renderer.update_robot_link_cameras()
            renderer.scene.update_render()
            debug_rgb, _ = renderer.capture_camera(renderer.zed_camera)
            debug_bgr = base.overlay_text(debug_rgb, overlay_lines)
            annotate_candidate_labels(
                debug_bgr,
                renderer.zed_camera,
                head_intrinsic,
                int(selected.source_frame) if selected is not None else None,
                common_candidates_per_frame,
                arm_display_candidates,
                selected_keyframes,
            )
            writer.write(debug_bgr)
    finally:
        writer.release()
        update_candidate_debug_visuals(
            debug_visuals,
            None,
            common_candidates_per_frame,
            arm_display_candidates,
        )
    return debug_video_path


def main() -> None:
    args = parse_args()
    args.anygrasp_dir = args.anygrasp_dir.resolve()
    args.replay_dir = args.replay_dir.resolve()
    args.hand_npz = args.hand_npz.resolve()
    args.output_dir = args.output_dir.resolve()
    args.robot_config = args.robot_config.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.anygrasp_dir.is_dir():
        raise NotADirectoryError(f"anygrasp_dir not found: {args.anygrasp_dir}")
    if not args.replay_dir.is_dir():
        raise NotADirectoryError(f"replay_dir not found: {args.replay_dir}")
    if not args.hand_npz.is_file():
        raise FileNotFoundError(f"hand_npz not found: {args.hand_npz}")
    if bool(args.save_pose_debug):
        pose_debug_path = args.output_dir / "pose_debug.jsonl"
        if pose_debug_path.exists():
            pose_debug_path.unlink()

    args.manual_candidate_overrides = parse_manual_candidate_overrides(args.manual_candidate)
    args.object_mesh_overrides = parse_object_mesh_overrides(args.object_mesh_override)
    args.candidate_orientation_remap_label, args.candidate_orientation_remap_matrix = base.resolve_orientation_remap(args.candidate_orientation_remap_label)
    args.candidate_post_rot_xyz_deg = np.asarray(args.candidate_post_rot_xyz_deg, dtype=np.float64)
    args.candidate_post_rot_matrix = base.orthonormalize_rotation(
        R.from_euler("xyz", np.deg2rad(args.candidate_post_rot_xyz_deg)).as_matrix()
    )

    renderer = build_renderer(args)
    hand_data = load_hand_data(args.hand_npz)
    keyframes = [int(v) for v in args.keyframes]
    replay_frame_indices, object_tracks = load_object_tracks(args.replay_dir, args.object_mesh_overrides)
    replay_head_camera_pose_by_frame = load_replay_head_camera_poses(args.replay_dir)
    object_states_per_frame = {frame: load_object_states(args.replay_dir, frame, args.object_mesh_overrides) for frame in keyframes}
    selection_result, arm_debugs = choose_keyframes(
        renderer=renderer,
        args=args,
        anygrasp_dir=args.anygrasp_dir,
        hand_data=hand_data,
        keyframes=keyframes,
        arm_mode=args.arm,
        object_states_per_frame=object_states_per_frame,
    )
    selected_keyframes = selection_result.selected_keyframes
    ranked_candidates_per_frame = selection_result.ranked_candidates_per_frame
    all_candidates_per_frame = selection_result.all_candidates_per_frame
    common_candidates_per_frame = build_display_candidates_per_frame(
        args,
        keyframes,
        ranked_candidates_per_frame,
        all_candidates_per_frame,
    )
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]] = {}
    for arm_name, arm_info in arm_debugs.items():
        arm_display_candidates[arm_name] = {}
        for frame in keyframes:
            frame = int(frame)
            arm_display_candidates[arm_name][frame] = list(
                arm_info.ranked_candidates_per_frame.get(frame, [])[: max(int(args.debug_candidate_top_k), 0)]
            )

    arm = selected_keyframes[0].arm
    primary_object_name = selected_keyframes[0].candidate.nearest_object
    execution_sequences: List[Tuple[str, List[SelectedKeyframe]]] = [(arm, selected_keyframes)]
    if bool(args.execute_both_arms) and args.arm == "auto":
        for candidate_arm in ("left", "right"):
            if candidate_arm == arm:
                continue
            candidate_info = arm_debugs.get(candidate_arm)
            if candidate_info is None or candidate_info.selected_keyframes is None:
                continue
            execution_sequences.append((candidate_arm, candidate_info.selected_keyframes))
    if bool(args.execute_both_arms) and len(execution_sequences) < 2:
        print("[exec-mode] execute_both_arms=1 but only one arm has valid selected keyframes; falling back to single-arm execution")
    debug_visuals = create_debug_visual_bundle(
        renderer,
        args,
        keyframes,
        common_candidates_per_frame,
        arm_display_candidates,
        selected_keyframes,
    )

    object_states = object_states_per_frame[keyframes[0]]
    for state in object_states.values():
        state.actor = create_execution_object_actor(
            renderer.scene,
            state.mesh_file,
            f"planned_object_{state.name}",
            ignore_collision=bool(args.replay_objects_ignore_collision),
        )
        set_actor_pose(state.actor, state.pose_world_wxyz)
        if state.name in object_tracks:
            object_tracks[state.name].actor = state.actor

    for item in selected_keyframes:
        actor = debug_visuals.keyframe_axis_actors.get((int(item.source_frame), item.arm))
        if actor is not None:
            set_actor_pose(actor, item.candidate.pose_world_wxyz)

    renderer.update_robot_link_cameras()
    renderer.step_scene(steps=1)
    renderer.set_grippers(args.open_gripper if arm == "left" else None, args.open_gripper if arm == "right" else None)
    head_intrinsic = get_camera_intrinsic_matrix(
        renderer.zed_camera,
        image_width=args.image_width,
        image_height=args.image_height,
        fovy_deg=args.fovy_deg,
    )

    debug_video_path = generate_debug_preview(
        renderer,
        args,
        replay_frame_indices,
        object_tracks,
        selected_keyframes,
        common_candidates_per_frame,
        arm_display_candidates,
        head_intrinsic,
        debug_visuals,
    )
    rank_preview_records = export_rank_preview_images(
        renderer=renderer,
        args=args,
        keyframes=keyframes,
        object_states_per_frame=object_states_per_frame,
        arm_debugs=arm_debugs,
        selected_keyframes=selected_keyframes,
        head_intrinsic=head_intrinsic,
    )
    for state in object_states.values():
        if state.actor is not None:
            set_actor_pose(state.actor, state.pose_world_wxyz)
    update_candidate_debug_visuals(debug_visuals, None, common_candidates_per_frame, arm_display_candidates)
    renderer.step_scene(steps=1)

    head_video_path = args.output_dir / "head_cam_plan.mp4"
    third_video_path = args.output_dir / "third_cam_plan.mp4"
    debug_execution_video_path = args.output_dir / "debug_execution_preview.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    head_writer = cv2.VideoWriter(str(head_video_path), fourcc, args.fps, (args.image_width, args.image_height))
    if not head_writer.isOpened():
        raise RuntimeError(f"Failed to open {head_video_path}")
    third_writer = None
    if bool(args.third_person_view) and not bool(args.head_only):
        third_writer = cv2.VideoWriter(str(third_video_path), fourcc, args.fps, (args.image_width, args.image_height))
        if not third_writer.isOpened():
            raise RuntimeError(f"Failed to open {third_video_path}")
    debug_execution_writer = None
    if bool(args.save_debug_execution_preview):
        debug_execution_writer = cv2.VideoWriter(str(debug_execution_video_path), fourcc, int(args.debug_execution_fps), (args.image_width, args.image_height))
        if not debug_execution_writer.isOpened():
            raise RuntimeError(f"Failed to open {debug_execution_video_path}")
    debug_execution_state = DebugExecutionState(
        writer=debug_execution_writer,
        selected_keyframes=selected_keyframes,
        common_candidates_per_frame=common_candidates_per_frame,
        arm_display_candidates=arm_display_candidates,
        head_intrinsic=head_intrinsic,
        active_frame=None,
        pose_debug_path=(args.output_dir / "pose_debug.jsonl") if bool(args.save_pose_debug) else None,
        replay_head_camera_pose_by_frame=replay_head_camera_pose_by_frame,
        object_tracks=object_tracks,
    )
    use_overlay = bool(args.overlay_text)
    dual_sync_mode = bool(args.execute_both_arms) and args.arm == "auto" and len(execution_sequences) >= 2
    stages_by_executed_arm: Dict[str, Dict[str, object]] = {}
    supervision_targets_by_executed_arm: Dict[str, Dict[str, np.ndarray]] = {}
    supervision_only_arms_by_executed_arm: Dict[str, List[str]] = {}
    selected_candidates_by_executed_arm: Dict[str, List[SelectedKeyframe]] = {}
    selected_objects_by_executed_arm: Dict[str, str] = {}
    init_pose_info_by_executed_arm: Dict[str, Dict[str, object]] = {}
    init_prefix_frames_written_by_executed_arm: Dict[str, int] = {}

    try:
        if dual_sync_mode:
            exec_selected_by_arm = {arm_name: seq for arm_name, seq in execution_sequences[:2]}
            dual_arms = [arm_name for arm_name in ("left", "right") if arm_name in exec_selected_by_arm]
            for arm_name in dual_arms:
                seq = exec_selected_by_arm[arm_name]
                selected_objects_by_executed_arm[arm_name] = seq[0].candidate.nearest_object
                selected_candidates_by_executed_arm[arm_name] = list(seq)
                supervision_targets_by_executed_arm[arm_name] = {
                    "left": np.asarray(exec_selected_by_arm["left"][0].candidate.pose_world_wxyz, dtype=np.float64),
                    "right": np.asarray(exec_selected_by_arm["right"][0].candidate.pose_world_wxyz, dtype=np.float64),
                }
                supervision_only_arms_by_executed_arm[arm_name] = []

            dual_selected_keyframes = [item for arm_name in dual_arms for item in exec_selected_by_arm[arm_name]]
            debug_execution_state.selected_keyframes = dual_selected_keyframes
            print("[exec-mode] selected_arm=both supervision_only_arms=none")

            init_info = apply_robot_init_pose(renderer, open_gripper=args.open_gripper, settle_steps=args.settle_steps)
            init_pose_info_by_executed_arm["left"] = init_info
            init_pose_info_by_executed_arm["right"] = init_info
            init_prefix_written = emit_init_prefix_frames(
                renderer=renderer,
                head_writer=head_writer,
                third_writer=third_writer,
                use_overlay=use_overlay,
                init_info=init_info,
                fixed_frames=args.init_prefix_frames,
                arm_label="both",
                debug_visuals=debug_visuals,
                debug_execution_state=debug_execution_state,
            )
            init_prefix_frames_written_by_executed_arm["left"] = int(init_prefix_written)
            init_prefix_frames_written_by_executed_arm["right"] = int(init_prefix_written)
            renderer.set_grippers(args.open_gripper, args.open_gripper)

            record_frame(
                renderer,
                head_writer,
                third_writer,
                [
                    "stage=init",
                    "arm=both",
                    f"left_object={selected_objects_by_executed_arm['left']}",
                    f"right_object={selected_objects_by_executed_arm['right']}",
                    f"left_goal_frame={exec_selected_by_arm['left'][0].source_frame}",
                    f"right_goal_frame={exec_selected_by_arm['right'][0].source_frame}",
                ],
                use_overlay,
                debug_visuals,
                debug_execution_state,
            )

            pregrasp_targets = {
                arm_name: pose_with_offset_along_local_x(seq[0].candidate.pose_world_wxyz, args.approach_offset_m)
                for arm_name, seq in exec_selected_by_arm.items()
            }
            grasp_targets = {arm_name: seq[0].candidate.pose_world_wxyz for arm_name, seq in exec_selected_by_arm.items()}
            action_targets = {arm_name: seq[1].candidate.pose_world_wxyz for arm_name, seq in exec_selected_by_arm.items()}
            action_object_replay = None
            if bool(args.replay_objects_during_action):
                action_object_replay = ExecutionObjectReplayConfig(
                    replay_frame_indices=np.asarray(replay_frame_indices, dtype=np.int32),
                    object_tracks=object_tracks,
                    start_frame=int(exec_selected_by_arm["left"][0].source_frame),
                    end_frame=int(exec_selected_by_arm["left"][1].source_frame),
                )

            debug_execution_state.active_frame = int(exec_selected_by_arm["left"][0].source_frame)
            set_dual_arm_target_visuals(
                renderer,
                pregrasp_targets.get("left"),
                pregrasp_targets.get("right"),
            )
            pregrasp_dual = execute_dual_stage_until_reached(
                renderer=renderer,
                target_pose_world_wxyz_by_arm=pregrasp_targets,
                label="pregrasp",
                head_writer=head_writer,
                third_writer=third_writer,
                use_overlay=use_overlay,
                args=args,
                debug_visuals=debug_visuals,
                debug_execution_state=debug_execution_state,
            )

            debug_execution_state.active_frame = int(exec_selected_by_arm["left"][0].source_frame)
            set_dual_arm_target_visuals(
                renderer,
                grasp_targets.get("left"),
                grasp_targets.get("right"),
            )
            grasp_dual = execute_dual_stage_until_reached(
                renderer=renderer,
                target_pose_world_wxyz_by_arm=grasp_targets,
                label="grasp",
                head_writer=head_writer,
                third_writer=third_writer,
                use_overlay=use_overlay,
                args=args,
                debug_visuals=debug_visuals,
                debug_execution_state=debug_execution_state,
            )
            print(
                "[stage] keyframe_1 grasp finished "
                f"left_reached={int(bool(grasp_dual['arms'].get('left', {}).get('reached', False)))} "
                f"right_reached={int(bool(grasp_dual['arms'].get('right', {}).get('reached', False)))}"
            )

            renderer.set_grippers(args.close_gripper, args.close_gripper)
            record_frame(
                renderer,
                head_writer,
                third_writer,
                ["stage=close_gripper", "arm=both"],
                use_overlay,
                debug_visuals,
                debug_execution_state,
            )
            pause_after_keyframe1(
                renderer=renderer,
                head_writer=head_writer,
                third_writer=third_writer,
                use_overlay=use_overlay,
                args=args,
                arm_label="both",
                goal_frame=int(exec_selected_by_arm["left"][0].source_frame),
                debug_visuals=debug_visuals,
                debug_execution_state=debug_execution_state,
            )

            attached_actor_by_arm: Dict[str, Optional[sapien.Entity]] = {}
            tcp_to_object_by_arm: Dict[str, Optional[np.ndarray]] = {}
            if not bool(args.replay_objects_during_action):
                for arm_name in dual_arms:
                    obj_name = selected_objects_by_executed_arm[arm_name]
                    attached_actor_by_arm[arm_name] = object_states[obj_name].actor
                    tcp_pose = renderer.get_current_tcp_pose(arm_name)
                    tcp_to_object_by_arm[arm_name] = np.linalg.inv(pose_wxyz_to_matrix(tcp_pose)) @ object_states[obj_name].pose_world_matrix

            debug_execution_state.active_frame = int(exec_selected_by_arm["left"][1].source_frame)
            set_dual_arm_target_visuals(
                renderer,
                action_targets.get("left"),
                action_targets.get("right"),
            )
            action_dual = execute_dual_stage_until_reached(
                renderer=renderer,
                target_pose_world_wxyz_by_arm=action_targets,
                label="action",
                head_writer=head_writer,
                third_writer=third_writer,
                use_overlay=use_overlay,
                args=args,
                debug_visuals=debug_visuals,
                debug_execution_state=debug_execution_state,
                attached_actor_by_arm=attached_actor_by_arm,
                tcp_to_object_by_arm=tcp_to_object_by_arm,
                object_replay=action_object_replay,
            )

            for arm_name in dual_arms:
                stages_by_executed_arm[arm_name] = {
                    "pregrasp": {
                        "status": str(pregrasp_dual["arms"][arm_name]["status"]),
                        "attempts": int(pregrasp_dual["attempts"]),
                        "reached": bool(pregrasp_dual["arms"][arm_name]["reached"]),
                        "pos_err_m": float(pregrasp_dual["arms"][arm_name]["pos_err_m"]),
                        "rot_err_deg": float(pregrasp_dual["arms"][arm_name]["rot_err_deg"]),
                        "attempt_history": pregrasp_dual["attempt_history"],
                    },
                    "grasp": {
                        "status": str(grasp_dual["arms"][arm_name]["status"]),
                        "attempts": int(grasp_dual["attempts"]),
                        "reached": bool(grasp_dual["arms"][arm_name]["reached"]),
                        "pos_err_m": float(grasp_dual["arms"][arm_name]["pos_err_m"]),
                        "rot_err_deg": float(grasp_dual["arms"][arm_name]["rot_err_deg"]),
                        "attempt_history": grasp_dual["attempt_history"],
                    },
                    "action": {
                        "status": str(action_dual["arms"][arm_name]["status"]),
                        "attempts": int(action_dual["attempts"]),
                        "reached": bool(action_dual["arms"][arm_name]["reached"]),
                        "pos_err_m": float(action_dual["arms"][arm_name]["pos_err_m"]),
                        "rot_err_deg": float(action_dual["arms"][arm_name]["rot_err_deg"]),
                        "attempt_history": action_dual["attempt_history"],
                    },
                }
        else:
            for exec_arm, exec_selected_keyframes in execution_sequences:
                exec_object_name = exec_selected_keyframes[0].candidate.nearest_object
                selected_objects_by_executed_arm[exec_arm] = exec_object_name
                selected_candidates_by_executed_arm[exec_arm] = list(exec_selected_keyframes)
                debug_execution_state.selected_keyframes = exec_selected_keyframes

                init_info = apply_robot_init_pose(renderer, open_gripper=args.open_gripper, settle_steps=args.settle_steps)
                init_pose_info_by_executed_arm[exec_arm] = init_info
                init_prefix_frames_written_by_executed_arm[exec_arm] = int(
                    emit_init_prefix_frames(
                        renderer=renderer,
                        head_writer=head_writer,
                        third_writer=third_writer,
                        use_overlay=use_overlay,
                        init_info=init_info,
                        fixed_frames=args.init_prefix_frames,
                        arm_label=exec_arm,
                        debug_visuals=debug_visuals,
                        debug_execution_state=debug_execution_state,
                    )
                )

                supervision_targets = build_supervision_targets(arm_display_candidates, exec_selected_keyframes)
                supervision_targets_by_executed_arm[exec_arm] = supervision_targets
                supervision_only_arms = sorted([name for name in supervision_targets.keys() if name != exec_arm])
                supervision_only_arms_by_executed_arm[exec_arm] = supervision_only_arms
                print(
                    f"[exec-mode] selected_arm={exec_arm} "
                    f"supervision_only_arms={supervision_only_arms if supervision_only_arms else 'none'}"
                )

                renderer.set_grippers(
                    args.open_gripper if exec_arm == "left" else None,
                    args.open_gripper if exec_arm == "right" else None,
                )

                grasp_pose = exec_selected_keyframes[0].candidate.pose_world_wxyz
                pregrasp_pose = pose_with_offset_along_local_x(grasp_pose, args.approach_offset_m)
                action_pose = exec_selected_keyframes[1].candidate.pose_world_wxyz
                action_object_replay = None
                if bool(args.replay_objects_during_action):
                    action_object_replay = ExecutionObjectReplayConfig(
                        replay_frame_indices=np.asarray(replay_frame_indices, dtype=np.int32),
                        object_tracks=object_tracks,
                        start_frame=int(exec_selected_keyframes[0].source_frame),
                        end_frame=int(exec_selected_keyframes[1].source_frame),
                    )
                debug_execution_state.active_frame = int(exec_selected_keyframes[0].source_frame)
                set_single_arm_target_visual(renderer, exec_arm, grasp_pose)
                record_frame(
                    renderer,
                    head_writer,
                    third_writer,
                    [
                        "stage=init",
                        f"arm={exec_arm}",
                        f"object={exec_object_name}",
                        f"goal=keyframe_{exec_selected_keyframes[0].source_frame}",
                        f"selected_f1_candidate={exec_selected_keyframes[0].candidate.candidate_idx}",
                        f"selected_f22_candidate={exec_selected_keyframes[1].candidate.candidate_idx}",
                    ],
                    use_overlay,
                    debug_visuals,
                    debug_execution_state,
                )

                pregrasp_result = execute_stage_until_reached(
                    renderer=renderer,
                    arm=exec_arm,
                    target_pose_world_wxyz=pregrasp_pose,
                    label="pregrasp",
                    head_writer=head_writer,
                    third_writer=third_writer,
                    use_overlay=use_overlay,
                    args=args,
                    target_visual_pose=grasp_pose,
                    target_visual_label=f"keyframe_{exec_selected_keyframes[0].source_frame}",
                    debug_visuals=debug_visuals,
                    debug_execution_state=debug_execution_state,
                    supervision_targets=supervision_targets,
                )

                grasp_result = execute_stage_until_reached(
                    renderer=renderer,
                    arm=exec_arm,
                    target_pose_world_wxyz=grasp_pose,
                    label="grasp",
                    head_writer=head_writer,
                    third_writer=third_writer,
                    use_overlay=use_overlay,
                    args=args,
                    target_visual_pose=grasp_pose,
                    target_visual_label=f"keyframe_{exec_selected_keyframes[0].source_frame}",
                    debug_visuals=debug_visuals,
                    debug_execution_state=debug_execution_state,
                    supervision_targets=supervision_targets,
                )
                print(
                    "[stage] keyframe_1 grasp finished "
                    f"arm={exec_arm} reached={int(bool(grasp_result.get('reached', False)))}"
                )

                set_single_arm_target_visual(renderer, exec_arm, grasp_pose)
                renderer.set_grippers(args.close_gripper if exec_arm == "left" else None, args.close_gripper if exec_arm == "right" else None)
                record_frame(
                    renderer,
                    head_writer,
                    third_writer,
                    ["stage=close_gripper", f"arm={exec_arm}", f"goal=keyframe_{exec_selected_keyframes[0].source_frame}"],
                    use_overlay,
                    debug_visuals,
                    debug_execution_state,
                )
                pause_after_keyframe1(
                    renderer=renderer,
                    head_writer=head_writer,
                    third_writer=third_writer,
                    use_overlay=use_overlay,
                    args=args,
                    arm_label=exec_arm,
                    goal_frame=int(exec_selected_keyframes[0].source_frame),
                    debug_visuals=debug_visuals,
                    debug_execution_state=debug_execution_state,
                )

                attached_actor = None
                tcp_to_object = None
                if not bool(args.replay_objects_during_action):
                    attached_actor = object_states[exec_object_name].actor
                    tcp_pose = renderer.get_current_tcp_pose(exec_arm)
                    tcp_to_object = np.linalg.inv(pose_wxyz_to_matrix(tcp_pose)) @ object_states[exec_object_name].pose_world_matrix

                debug_execution_state.active_frame = int(exec_selected_keyframes[1].source_frame)
                action_result = execute_stage_until_reached(
                    renderer=renderer,
                    arm=exec_arm,
                    target_pose_world_wxyz=action_pose,
                    label="action",
                    head_writer=head_writer,
                    third_writer=third_writer,
                    use_overlay=use_overlay,
                    args=args,
                    attached_actor=attached_actor,
                    tcp_to_object=tcp_to_object,
                    target_visual_pose=action_pose,
                    target_visual_label=f"keyframe_{exec_selected_keyframes[1].source_frame}",
                    debug_visuals=debug_visuals,
                    debug_execution_state=debug_execution_state,
                    object_replay=action_object_replay,
                )
                stages_by_executed_arm[exec_arm] = {
                    "pregrasp": pregrasp_result,
                    "grasp": grasp_result,
                    "action": action_result,
                }
    finally:
        head_writer.release()
        if third_writer is not None:
            third_writer.release()
        if debug_execution_writer is not None:
            debug_execution_writer.release()
        renderer.update_target_axis_visuals(None, None)

    primary_exec_arm = execution_sequences[0][0]
    primary_exec_selected_keyframes = execution_sequences[0][1]
    primary_object_name = selected_objects_by_executed_arm[primary_exec_arm]
    primary_stages = stages_by_executed_arm[primary_exec_arm]
    primary_supervision_targets = supervision_targets_by_executed_arm[primary_exec_arm]
    primary_supervision_only_arms = supervision_only_arms_by_executed_arm[primary_exec_arm]

    summary = {
        "anygrasp_dir": str(args.anygrasp_dir),
        "replay_dir": str(args.replay_dir),
        "hand_npz": str(args.hand_npz),
        "keyframes": keyframes,
        "selected_arm": primary_exec_arm,
        "expected_object_for_selected_arm": selection_result.expected_object,
        "selection_diagnostics": selection_result.diagnostics,
        "candidate_orientation_remap_label": args.candidate_orientation_remap_label,
        "candidate_post_rot_xyz_deg": args.candidate_post_rot_xyz_deg.tolist(),
        "candidate_keep_camera_up": int(args.candidate_keep_camera_up),
        "candidate_camera_top_axis": str(args.candidate_camera_top_axis),
        "candidate_target_local_x_offset_m": float(args.candidate_target_local_x_offset_m),
        "reach_error_pose_source": args.reach_error_pose_source,
        "init_prefix_frames": int(args.init_prefix_frames),
        "execute_both_arms": int(args.execute_both_arms),
        "execution_mode": "dual_sync" if dual_sync_mode else "single_or_sequential",
        "manual_candidate_overrides": {arm: {str(frame): idx for frame, idx in frame_map.items()} for arm, frame_map in args.manual_candidate_overrides.items() if frame_map},
        "arm_debugs": {
            arm_name: {
                "expected_object": arm_info.expected_object,
                "diagnostics": arm_info.diagnostics,
                "selected_keyframes": (
                    [
                        {
                            "source_frame": item.source_frame,
                            "candidate_idx": item.candidate.candidate_idx,
                            "nearest_object": item.candidate.nearest_object,
                            "rotation_distance_deg": item.candidate.rotation_distance_deg,
                            "top_axis_up_dot": item.candidate.top_axis_up_dot,
                            "original_top_axis_up_dot": item.candidate.original_top_axis_up_dot,
                            "camera_up_flip_applied": item.candidate.camera_up_flip_applied,
                            "forward_axis_change_deg": item.candidate.forward_axis_change_deg,
                        }
                        for item in arm_info.selected_keyframes
                    ]
                    if arm_info.selected_keyframes is not None
                    else None
                ),
            }
            for arm_name, arm_info in arm_debugs.items()
        },
        "selected_object": primary_object_name,
        "selected_objects_by_executed_arm": selected_objects_by_executed_arm,
        "executed_arms": [item[0] for item in execution_sequences],
        "supervision_only_arms": primary_supervision_only_arms,
        "supervision_only_arms_by_executed_arm": supervision_only_arms_by_executed_arm,
        "supervision_targets": {arm_name: target.tolist() for arm_name, target in primary_supervision_targets.items()},
        "supervision_targets_by_executed_arm": {
            arm_name: {target_arm: target.tolist() for target_arm, target in targets.items()}
            for arm_name, targets in supervision_targets_by_executed_arm.items()
        },
        "init_pose_by_executed_arm": {
            arm_name: {
                "left_applied": bool(info["left_applied"]),
                "right_applied": bool(info["right_applied"]),
                "left_joints": None if info["left_joints"] is None else np.asarray(info["left_joints"], dtype=np.float64).reshape(6).tolist(),
                "right_joints": None if info["right_joints"] is None else np.asarray(info["right_joints"], dtype=np.float64).reshape(6).tolist(),
                "gripper_open": float(info["gripper_open"]),
            }
            for arm_name, info in init_pose_info_by_executed_arm.items()
        },
        "init_prefix_frames_written_by_executed_arm": init_prefix_frames_written_by_executed_arm,
        "stages": primary_stages,
        "stages_by_executed_arm": stages_by_executed_arm,
        "selected_candidates": [
            {
                "source_frame": item.source_frame,
                "arm": item.arm,
                "candidate_idx": item.candidate.candidate_idx,
                "score": item.candidate.score,
                "rotation_distance_deg": item.candidate.rotation_distance_deg,
                "nearest_object": item.candidate.nearest_object,
                "nearest_object_distance_m": item.candidate.nearest_object_distance_m,
                "width_m": item.candidate.width_m,
                "depth_m": item.candidate.depth_m,
                "top_axis_up_dot": item.candidate.top_axis_up_dot,
                "original_top_axis_up_dot": item.candidate.original_top_axis_up_dot,
                "camera_up_flip_applied": item.candidate.camera_up_flip_applied,
                "forward_axis_change_deg": item.candidate.forward_axis_change_deg,
                "raw_pose_world_wxyz": item.candidate.raw_pose_world_wxyz.tolist(),
                "pose_world_wxyz": item.candidate.pose_world_wxyz.tolist(),
                "translation_cam": item.candidate.translation_cam.tolist(),
                "rotation_cam": item.candidate.rotation_cam.tolist(),
                "hand_rotation_cam": item.hand_rotation_cam.tolist(),
            }
            for item in primary_exec_selected_keyframes
        ],
        "selected_candidates_by_executed_arm": {
            arm_name: [
                {
                    "source_frame": item.source_frame,
                    "arm": item.arm,
                    "candidate_idx": item.candidate.candidate_idx,
                    "score": item.candidate.score,
                    "rotation_distance_deg": item.candidate.rotation_distance_deg,
                    "nearest_object": item.candidate.nearest_object,
                    "nearest_object_distance_m": item.candidate.nearest_object_distance_m,
                    "width_m": item.candidate.width_m,
                    "depth_m": item.candidate.depth_m,
                    "top_axis_up_dot": item.candidate.top_axis_up_dot,
                    "original_top_axis_up_dot": item.candidate.original_top_axis_up_dot,
                    "camera_up_flip_applied": item.candidate.camera_up_flip_applied,
                    "forward_axis_change_deg": item.candidate.forward_axis_change_deg,
                    "raw_pose_world_wxyz": item.candidate.raw_pose_world_wxyz.tolist(),
                    "pose_world_wxyz": item.candidate.pose_world_wxyz.tolist(),
                    "translation_cam": item.candidate.translation_cam.tolist(),
                    "rotation_cam": item.candidate.rotation_cam.tolist(),
                    "hand_rotation_cam": item.hand_rotation_cam.tolist(),
                }
                for item in candidates
            ]
            for arm_name, candidates in selected_candidates_by_executed_arm.items()
        },
        "top_candidates_per_keyframe": {
            str(frame): [
                {
                    "candidate_idx": cand.candidate_idx,
                    "score": cand.score,
                    "nearest_object": cand.nearest_object,
                    "nearest_object_distance_m": cand.nearest_object_distance_m,
                    "rotation_distance_deg": cand.rotation_distance_deg,
                    "width_m": cand.width_m,
                    "depth_m": cand.depth_m,
                    "top_axis_up_dot": cand.top_axis_up_dot,
                    "original_top_axis_up_dot": cand.original_top_axis_up_dot,
                    "camera_up_flip_applied": cand.camera_up_flip_applied,
                    "forward_axis_change_deg": cand.forward_axis_change_deg,
                    "raw_pose_world_wxyz": cand.raw_pose_world_wxyz.tolist(),
                    "pose_world_wxyz": cand.pose_world_wxyz.tolist(),
                }
                for cand in ranked_candidates_per_frame.get(int(frame), [])[: max(int(args.debug_candidate_top_k), 0)]
            ]
            for frame in keyframes
        },
        "all_candidates_per_keyframe": {
            str(frame): [
                {
                    "candidate_idx": cand.candidate_idx,
                    "score": cand.score,
                    "nearest_object": cand.nearest_object,
                    "nearest_object_distance_m": cand.nearest_object_distance_m,
                    "rotation_distance_deg": cand.rotation_distance_deg,
                    "width_m": cand.width_m,
                    "depth_m": cand.depth_m,
                    "top_axis_up_dot": cand.top_axis_up_dot,
                    "original_top_axis_up_dot": cand.original_top_axis_up_dot,
                    "camera_up_flip_applied": cand.camera_up_flip_applied,
                    "forward_axis_change_deg": cand.forward_axis_change_deg,
                    "raw_pose_world_wxyz": cand.raw_pose_world_wxyz.tolist(),
                    "pose_world_wxyz": cand.pose_world_wxyz.tolist(),
                }
                for cand in all_candidates_per_frame.get(int(frame), [])
            ]
            for frame in keyframes
        },
        "debug_preview_video": str(debug_video_path) if debug_video_path is not None else None,
        "rank_preview_images": [
            {
                "frame": item.frame,
                "rank": item.rank,
                "image_path": item.image_path,
                "left_candidate_idx": item.left_candidate_idx,
                "right_candidate_idx": item.right_candidate_idx,
            }
            for item in rank_preview_records
        ],
        "debug_execution_video": str(debug_execution_video_path) if debug_execution_writer is not None else None,
        "head_video": str(head_video_path),
        "third_video": str(third_video_path) if third_writer is not None else None,
    }
    with (args.output_dir / "plan_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "[done] "
        f"executed_arms={summary['executed_arms']} primary_arm={primary_exec_arm} object={primary_object_name} "
        f"selected_f{primary_exec_selected_keyframes[0].source_frame}=candidate_{primary_exec_selected_keyframes[0].candidate.candidate_idx} "
        f"selected_f{primary_exec_selected_keyframes[1].source_frame}=candidate_{primary_exec_selected_keyframes[1].candidate.candidate_idx} "
        f"statuses_by_arm={summary['stages_by_executed_arm']} "
        f"head_video={head_video_path}"
    )
    renderer.hold_viewer()


if __name__ == "__main__":
    main()
