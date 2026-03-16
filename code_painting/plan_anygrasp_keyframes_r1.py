#!/usr/bin/env python3
"""Plan a two-keyframe RoboTwin demo from AnyGrasp candidates and hand gripper poses."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    pose_world_wxyz: np.ndarray
    pose_world_matrix: np.ndarray
    nearest_object: str
    nearest_object_distance_m: float
    rotation_distance_deg: float


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
    keyframe_axis_actors: Dict[int, sapien.Entity]
    common_candidate_actors: Dict[int, List[sapien.Entity]]
    arm_candidate_actors: Dict[str, Dict[int, List[sapien.Entity]]]


@dataclass
class DebugExecutionState:
    writer: Optional[cv2.VideoWriter]
    selected_keyframes: List[SelectedKeyframe]
    common_candidates_per_frame: Dict[int, List[CandidatePose]]
    arm_display_candidates: Dict[str, Dict[int, List[CandidatePose]]]
    head_intrinsic: np.ndarray
    active_frame: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use AnyGrasp keyframes + hand orientation to plan a two-keyframe RoboTwin demo.")
    parser.add_argument("--anygrasp_dir", type=Path, required=True, help="Per-video AnyGrasp result dir, e.g. anygrasp_batch_results/d_pour_blue_1.")
    parser.add_argument("--replay_dir", type=Path, required=True, help="Per-video replay dir containing multi_object_world_poses.npz.")
    parser.add_argument("--hand_npz", type=Path, required=True, help="Per-video hand_detections_*.npz with gripper pose fields.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output dir for planned demo video and metadata.")
    parser.add_argument("--keyframes", type=int, nargs=2, default=[1, 22], metavar=("GRASP_FRAME", "ACTION_FRAME"))
    parser.add_argument("--arm", choices=["auto", "left", "right"], default="auto")
    parser.add_argument("--planner_backend", choices=["urdfik", "curobo"], default="urdfik")
    parser.add_argument("--left_target_object", type=str, default="cup")
    parser.add_argument("--right_target_object", type=str, default="bottle")
    parser.add_argument("--candidate_object_max_distance_m", type=float, default=0.12)
    parser.add_argument("--enforce_target_object_constraint", type=int, default=1)
    parser.add_argument("--enforce_candidate_distance_constraint", type=int, default=1)
    parser.add_argument("--debug_candidate_top_k", type=int, default=4)
    parser.add_argument("--debug_show_all_candidates", type=int, default=1)
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
    parser.add_argument("--max_stage_replans", type=int, default=3)
    parser.add_argument("--hold_frames_after_stage", type=int, default=2)
    parser.add_argument("--save_debug_preview", type=int, default=1)
    parser.add_argument("--debug_preview_fps", type=int, default=10)
    parser.add_argument("--debug_keyframe_hold_frames", type=int, default=12)
    parser.add_argument("--save_debug_execution_preview", type=int, default=1)
    parser.add_argument("--debug_execution_fps", type=int, default=10)
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


def load_object_states(replay_dir: Path, source_frame: int) -> Dict[str, ObjectState]:
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
        states[object_name] = ObjectState(
            name=object_name,
            mesh_file=mesh_file,
            pose_world_wxyz=pose_world_wxyz,
            pose_world_matrix=pose_world_matrix,
            visible=visible,
        )
    return states


def load_object_tracks(replay_dir: Path) -> Tuple[np.ndarray, Dict[str, ObjectTrack]]:
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
        tracks[object_name] = ObjectTrack(
            name=object_name,
            mesh_file=mesh_file,
            frame_indices=frame_indices.copy(),
            pose_world_wxyz=np.asarray(data[f"{key}__pose_world_wxyz"], dtype=np.float64),
            pose_world_matrix=np.asarray(data[f"{key}__pose_world_matrix"], dtype=np.float64),
            visible=np.asarray(data[f"{key}__visible"], dtype=bool),
        )
    return frame_indices, tracks


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


def candidate_to_world_pose(renderer: ReplayRenderer, grasp: Dict) -> Tuple[np.ndarray, np.ndarray]:
    rot_cam = np.asarray(grasp["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    trans_cam = np.asarray(grasp["translation"], dtype=np.float64).reshape(3)
    pose_world_wxyz = renderer.camera_to_world_pose(trans_cam, rot_cam)
    pose_world_matrix = pose_wxyz_to_matrix(pose_world_wxyz)
    return pose_world_wxyz, pose_world_matrix


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
            pose_world_wxyz, pose_world_matrix = candidate_to_world_pose(renderer, grasp)
            nearest_name, nearest_dist = nearest_object_name(pose_world_matrix, object_states_per_frame[frame])
            candidate = CandidatePose(
                candidate_idx=candidate_idx,
                score=float(grasp["score"]),
                translation_cam=np.asarray(grasp["translation"], dtype=np.float64).reshape(3),
                rotation_cam=np.asarray(grasp["rotation_matrix"], dtype=np.float64).reshape(3, 3),
                width_m=float(grasp.get("width", 0.08)),
                depth_m=float(grasp.get("depth", 0.04)),
                pose_world_wxyz=pose_world_wxyz,
                pose_world_matrix=pose_world_matrix,
                nearest_object=nearest_name,
                nearest_object_distance_m=nearest_dist,
                rotation_distance_deg=rotation_distance_deg(ref_rotation, grasp["rotation_matrix"]),
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


def create_gripper_candidate_actor(scene: sapien.Scene, name: str, color: Sequence[float], marker_side: str = "none") -> sapien.Entity:
    builder = scene.create_actor_builder()
    body_half = [0.008, 0.026, 0.004]
    finger_half = [0.018, 0.0035, 0.0035]
    builder.add_box_visual(pose=sapien.Pose([-0.012, 0.0, 0.0]), half_size=body_half, material=list(color))
    builder.add_box_visual(pose=sapien.Pose([0.012, 0.020, 0.0]), half_size=finger_half, material=list(color))
    builder.add_box_visual(pose=sapien.Pose([0.012, -0.020, 0.0]), half_size=finger_half, material=list(color))
    marker_color = [0.05, 0.05, 0.05]
    if marker_side == "left":
        builder.add_box_visual(pose=sapien.Pose([0.0, 0.0, 0.012]), half_size=[0.004, 0.004, 0.002], material=marker_color)
    elif marker_side == "right":
        builder.add_box_visual(pose=sapien.Pose([0.0, 0.0, -0.012]), half_size=[0.004, 0.004, 0.002], material=marker_color)
    actor = builder.build_kinematic(name=name)
    hide_actor(actor)
    return actor


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


def draw_small_candidate_label(image_bgr: np.ndarray, text: str, pixel_xy: Tuple[int, int], color_bgr: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.28
    thickness = 1
    x, y = int(pixel_xy[0]), int(pixel_xy[1])
    cv2.putText(image_bgr, text, (x, y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(image_bgr, text, (x, y), font, font_scale, color_bgr, thickness, cv2.LINE_AA)


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

    selected_by_frame = {int(item.source_frame): (item.arm, int(item.candidate.candidate_idx)) for item in selected_keyframes}

    common_candidates = common_candidates_per_frame.get(frame, [])
    for cand in common_candidates:
        origin = np.asarray(cand.pose_world_wxyz[:3], dtype=np.float64)
        pixel = project_world_point_to_image(camera, intrinsic, origin, width, height)
        if pixel is None:
            continue
        draw_small_candidate_label(image_bgr, str(int(cand.candidate_idx)), pixel, (0, 150, 0))

    arm_styles = {
        "left": ((220, 120, 20), np.array([0.0, 0.0, 0.018], dtype=np.float64)),
        "right": ((0, 140, 255), np.array([0.0, 0.0, -0.018], dtype=np.float64)),
    }
    for arm_name, (label_color, local_offset) in arm_styles.items():
        frame_candidates = arm_display_candidates.get(arm_name, {}).get(frame, [])
        selected_arm, selected_idx = selected_by_frame.get(frame, (None, None))
        for cand in frame_candidates:
            pose_world = np.asarray(cand.pose_world_matrix, dtype=np.float64)
            label_world = pose_world[:3, 3] + pose_world[:3, :3] @ local_offset
            pixel = project_world_point_to_image(camera, intrinsic, label_world, width, height)
            if pixel is None:
                continue
            color = (0, 0, 255) if arm_name == selected_arm and int(cand.candidate_idx) == selected_idx else label_color
            draw_small_candidate_label(image_bgr, str(int(cand.candidate_idx)), pixel, color)


def build_display_candidates_per_frame(
    args: argparse.Namespace,
    keyframes: Sequence[int],
    ranked_candidates_per_frame: Dict[int, List[CandidatePose]],
    all_candidates_per_frame: Dict[int, List[CandidatePose]],
) -> Dict[int, List[CandidatePose]]:
    display: Dict[int, List[CandidatePose]] = {}
    for frame in keyframes:
        frame = int(frame)
        if bool(args.debug_show_all_candidates):
            display[frame] = list(all_candidates_per_frame.get(frame, []))
        else:
            display[frame] = list(ranked_candidates_per_frame.get(frame, [])[: max(int(args.debug_candidate_top_k), 0)])
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
        int(frame): create_colored_axis_actor(
            renderer.scene,
            f"debug_axis_keyframe_{int(frame)}",
            axis_length=float(args.debug_target_axis_length),
            thickness=float(args.debug_target_axis_thickness),
            colors=axis_colors[int(frame)],
        )
        for frame in keyframes
    }
    selected_by_frame = {int(item.source_frame): item.candidate.candidate_idx for item in selected_keyframes}
    common_candidate_actors = {
        int(frame): [
            create_gripper_candidate_actor(
                renderer.scene,
                f"debug_common_candidate_{int(frame)}_{rank}",
                color=[0.1, 0.85, 0.2],
                marker_side="none",
            )
            for rank, _cand in enumerate(common_candidates_per_frame.get(int(frame), []))
        ]
        for frame in keyframes
    }
    arm_candidate_actors: Dict[str, Dict[int, List[sapien.Entity]]] = {}
    for arm_name, frame_candidates in arm_display_candidates.items():
        arm_candidate_actors[arm_name] = {}
        for frame in keyframes:
            frame = int(frame)
            selected_idx = None
            if arm_name == selected_keyframes[0].arm:
                selected_idx = selected_by_frame.get(frame)
            actors = []
            for rank, cand in enumerate(frame_candidates.get(frame, [])):
                actors.append(
                    create_gripper_candidate_actor(
                        renderer.scene,
                        f"debug_{arm_name}_candidate_{frame}_{rank}",
                        color=[1.0, 0.1, 0.1] if cand.candidate_idx == selected_idx else ([0.1, 0.45, 1.0] if arm_name == "left" else [1.0, 0.55, 0.0]),
                        marker_side="none",
                    )
                )
            arm_candidate_actors[arm_name][frame] = actors
    return DebugVisualBundle(
        keyframe_axis_actors=keyframe_axis_actors,
        common_candidate_actors=common_candidate_actors,
        arm_candidate_actors=arm_candidate_actors,
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


def record_frame(
    renderer: ReplayRenderer,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    overlay_lines: Sequence[str],
    use_overlay: bool,
    debug_visuals: Optional[DebugVisualBundle] = None,
    debug_execution_state: Optional[DebugExecutionState] = None,
) -> None:
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
    renderer.update_robot_link_cameras()
    renderer.scene.update_render()
    head_rgb, _ = renderer.capture_camera(renderer.zed_camera)
    head_bgr = base.overlay_text(head_rgb, overlay_lines) if use_overlay else cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
    head_writer.write(head_bgr)
    if third_writer is not None:
        third_rgb, _ = renderer.capture_camera(renderer.third_camera)
        third_bgr = base.overlay_text(third_rgb, overlay_lines) if use_overlay else cv2.cvtColor(third_rgb, cv2.COLOR_RGB2BGR)
        third_writer.write(third_bgr)
    if debug_execution_state is not None and debug_execution_state.writer is not None and debug_visuals is not None:
        for item in debug_execution_state.selected_keyframes:
            actor = debug_visuals.keyframe_axis_actors.get(int(item.source_frame))
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
        debug_overlay = list(overlay_lines)
        debug_overlay.extend(
            [
                "mode=debug_execution",
                (
                    f"active_keyframe={int(debug_execution_state.active_frame)}"
                    if debug_execution_state.active_frame is not None
                    else "active_keyframe=none"
                ),
                "color=green:all blue:left orange:right red:selected",
                "labels=small colored candidate_idx",
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
        )
        debug_execution_state.writer.write(debug_bgr)
        for actors in debug_visuals.common_candidate_actors.values():
            for actor in actors:
                hide_actor(actor)
        for per_frame in debug_visuals.arm_candidate_actors.values():
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
        if attached_actor is not None and tcp_to_object is not None:
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
) -> Dict[str, object]:
    last_status = "Missing"
    last_pos_err = float("inf")
    last_rot_err = float("inf")
    attempts = 0
    for attempt in range(1, max(int(args.max_stage_replans), 1) + 1):
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
        )
        current_tcp = renderer.get_current_tcp_pose(arm)
        last_pos_err, last_rot_err = tcp_pose_errors(target_pose_world_wxyz, current_tcp)
        reached = (
            last_status == "Success"
            and last_pos_err <= float(args.reach_pos_tol_m)
            and last_rot_err <= float(args.reach_rot_tol_deg)
        )
        record_frame(
            renderer,
            head_writer,
            third_writer,
            [
                f"stage={label}",
                f"arm={arm}",
                f"attempt={attempt}/{args.max_stage_replans}",
                f"status={last_status}",
                f"pos_err={last_pos_err:.4f}m",
                f"rot_err={last_rot_err:.2f}deg",
                f"reached={int(reached)}",
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
            }

    return {
        "status": last_status,
        "attempts": attempts,
        "reached": False,
        "pos_err_m": last_pos_err,
        "rot_err_deg": last_rot_err,
    }


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
        for frame, actors in per_frame.items():
            candidates = candidates_map.get(int(frame), [])
            for idx, actor in enumerate(actors):
                if active_frame is not None and int(frame) == int(active_frame) and idx < len(candidates):
                    set_actor_pose(actor, candidates[idx].pose_world_wxyz)
                else:
                    hide_actor(actor)


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
        actor = debug_visuals.keyframe_axis_actors.get(int(item.source_frame))
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
                        f"all_candidates={len(common_candidates_per_frame.get(int(selected.source_frame), []))}",
                        "color=green:all blue:left orange:right red:selected",
                        "labels=small colored candidate_idx",
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

    renderer = build_renderer(args)
    hand_data = load_hand_data(args.hand_npz)
    keyframes = [int(v) for v in args.keyframes]
    replay_frame_indices, object_tracks = load_object_tracks(args.replay_dir)
    object_states_per_frame = {frame: load_object_states(args.replay_dir, frame) for frame in keyframes}
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
        state.actor = create_object_actor(renderer.scene, state.mesh_file, f"planned_object_{state.name}")
        set_actor_pose(state.actor, state.pose_world_wxyz)
        if state.name in object_tracks:
            object_tracks[state.name].actor = state.actor

    for item in selected_keyframes:
        actor = debug_visuals.keyframe_axis_actors.get(int(item.source_frame))
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
    )
    use_overlay = bool(args.overlay_text)

    try:
        grasp_pose = selected_keyframes[0].candidate.pose_world_wxyz
        pregrasp_pose = pose_with_offset_along_local_x(grasp_pose, args.approach_offset_m)
        action_pose = selected_keyframes[1].candidate.pose_world_wxyz
        debug_execution_state.active_frame = int(selected_keyframes[0].source_frame)
        set_single_arm_target_visual(renderer, arm, grasp_pose)
        record_frame(
            renderer,
            head_writer,
            third_writer,
            [
                "stage=init",
                f"arm={arm}",
                f"object={primary_object_name}",
                f"goal=keyframe_{selected_keyframes[0].source_frame}",
                f"selected_f1_candidate={selected_keyframes[0].candidate.candidate_idx}",
                f"selected_f22_candidate={selected_keyframes[1].candidate.candidate_idx}",
            ],
            use_overlay,
            debug_visuals,
            debug_execution_state,
        )

        pregrasp_result = execute_stage_until_reached(
            renderer=renderer,
            arm=arm,
            target_pose_world_wxyz=pregrasp_pose,
            label="pregrasp",
            head_writer=head_writer,
            third_writer=third_writer,
            use_overlay=use_overlay,
            args=args,
            target_visual_pose=grasp_pose,
            target_visual_label=f"keyframe_{selected_keyframes[0].source_frame}",
            debug_visuals=debug_visuals,
            debug_execution_state=debug_execution_state,
        )

        grasp_result = execute_stage_until_reached(
            renderer=renderer,
            arm=arm,
            target_pose_world_wxyz=grasp_pose,
            label="grasp",
            head_writer=head_writer,
            third_writer=third_writer,
            use_overlay=use_overlay,
            args=args,
            target_visual_pose=grasp_pose,
            target_visual_label=f"keyframe_{selected_keyframes[0].source_frame}",
            debug_visuals=debug_visuals,
            debug_execution_state=debug_execution_state,
        )

        set_single_arm_target_visual(renderer, arm, grasp_pose)
        renderer.set_grippers(args.close_gripper if arm == "left" else None, args.close_gripper if arm == "right" else None)
        record_frame(
            renderer,
            head_writer,
            third_writer,
            ["stage=close_gripper", f"arm={arm}", f"goal=keyframe_{selected_keyframes[0].source_frame}"],
            use_overlay,
            debug_visuals,
            debug_execution_state,
        )

        attached_actor = object_states[primary_object_name].actor
        tcp_pose = renderer.get_current_tcp_pose(arm)
        tcp_to_object = np.linalg.inv(pose_wxyz_to_matrix(tcp_pose)) @ object_states[primary_object_name].pose_world_matrix

        debug_execution_state.active_frame = int(selected_keyframes[1].source_frame)
        action_result = execute_stage_until_reached(
            renderer=renderer,
            arm=arm,
            target_pose_world_wxyz=action_pose,
            label="action",
            head_writer=head_writer,
            third_writer=third_writer,
            use_overlay=use_overlay,
            args=args,
            attached_actor=attached_actor,
            tcp_to_object=tcp_to_object,
            target_visual_pose=action_pose,
            target_visual_label=f"keyframe_{selected_keyframes[1].source_frame}",
            debug_visuals=debug_visuals,
            debug_execution_state=debug_execution_state,
        )
    finally:
        head_writer.release()
        if third_writer is not None:
            third_writer.release()
        if debug_execution_writer is not None:
            debug_execution_writer.release()
        renderer.update_target_axis_visuals(None, None)

    summary = {
        "anygrasp_dir": str(args.anygrasp_dir),
        "replay_dir": str(args.replay_dir),
        "hand_npz": str(args.hand_npz),
        "keyframes": keyframes,
        "selected_arm": arm,
        "expected_object_for_selected_arm": selection_result.expected_object,
        "selection_diagnostics": selection_result.diagnostics,
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
        "stages": {
            "pregrasp": pregrasp_result,
            "grasp": grasp_result,
            "action": action_result,
        },
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
                "pose_world_wxyz": item.candidate.pose_world_wxyz.tolist(),
                "translation_cam": item.candidate.translation_cam.tolist(),
                "rotation_cam": item.candidate.rotation_cam.tolist(),
                "hand_rotation_cam": item.hand_rotation_cam.tolist(),
            }
            for item in selected_keyframes
        ],
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
                    "pose_world_wxyz": cand.pose_world_wxyz.tolist(),
                }
                for cand in all_candidates_per_frame.get(int(frame), [])
            ]
            for frame in keyframes
        },
        "debug_preview_video": str(debug_video_path) if debug_video_path is not None else None,
        "debug_execution_video": str(debug_execution_video_path) if debug_execution_writer is not None else None,
        "head_video": str(head_video_path),
        "third_video": str(third_video_path) if third_writer is not None else None,
    }
    with (args.output_dir / "plan_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "[done] "
        f"arm={arm} object={primary_object_name} "
        f"selected_f{selected_keyframes[0].source_frame}=candidate_{selected_keyframes[0].candidate.candidate_idx} "
        f"selected_f{selected_keyframes[1].source_frame}=candidate_{selected_keyframes[1].candidate.candidate_idx} "
        f"statuses={summary['stages']} "
        f"head_video={head_video_path}"
    )
    renderer.hold_viewer()


if __name__ == "__main__":
    main()
