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
from render_object_pose_r1_npz import create_object_actor, pose_wxyz_to_matrix
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
class CandidatePose:
    candidate_idx: int
    score: float
    translation_cam: np.ndarray
    rotation_cam: np.ndarray
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use AnyGrasp keyframes + hand orientation to plan a two-keyframe RoboTwin demo.")
    parser.add_argument("--anygrasp_dir", type=Path, required=True, help="Per-video AnyGrasp result dir, e.g. anygrasp_batch_results/d_pour_blue_1.")
    parser.add_argument("--replay_dir", type=Path, required=True, help="Per-video replay dir containing multi_object_world_poses.npz.")
    parser.add_argument("--hand_npz", type=Path, required=True, help="Per-video hand_detections_*.npz with gripper pose fields.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output dir for planned demo video and metadata.")
    parser.add_argument("--keyframes", type=int, nargs=2, default=[1, 22], metavar=("GRASP_FRAME", "ACTION_FRAME"))
    parser.add_argument("--arm", choices=["auto", "left", "right"], default="auto")
    parser.add_argument("--planner_backend", choices=["urdfik", "curobo"], default="urdfik")
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
        debug_target_axis_length=0.08,
        debug_target_axis_thickness=0.004,
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


def select_keyframes_for_arm(
    renderer: ReplayRenderer,
    anygrasp_dir: Path,
    hand_data: Dict[str, np.ndarray],
    keyframes: Sequence[int],
    arm: str,
    object_states_per_frame: Dict[int, Dict[str, ObjectState]],
) -> Optional[List[SelectedKeyframe]]:
    hand_valid = np.asarray(hand_data[f"{arm}_gripper_valid"], dtype=bool)
    hand_rotations = np.asarray(hand_data[f"{arm}_gripper_rotation_matrix"], dtype=np.float64)
    if any(not bool(hand_valid[frame]) for frame in keyframes):
        return None

    candidates_per_frame: Dict[int, List[CandidatePose]] = {}
    for frame in keyframes:
        all_candidates = []
        for candidate_idx, grasp in enumerate(load_anygrasp_candidates(anygrasp_dir, frame)):
            pose_world_wxyz, pose_world_matrix = candidate_to_world_pose(renderer, grasp)
            nearest_name, nearest_dist = nearest_object_name(pose_world_matrix, object_states_per_frame[frame])
            all_candidates.append(
                CandidatePose(
                    candidate_idx=candidate_idx,
                    score=float(grasp["score"]),
                    translation_cam=np.asarray(grasp["translation"], dtype=np.float64).reshape(3),
                    rotation_cam=np.asarray(grasp["rotation_matrix"], dtype=np.float64).reshape(3, 3),
                    pose_world_wxyz=pose_world_wxyz,
                    pose_world_matrix=pose_world_matrix,
                    nearest_object=nearest_name,
                    nearest_object_distance_m=nearest_dist,
                    rotation_distance_deg=rotation_distance_deg(hand_rotations[frame], grasp["rotation_matrix"]),
                )
            )
        candidates_per_frame[frame] = all_candidates

    best_selection: Optional[List[SelectedKeyframe]] = None
    best_cost = float("inf")
    for first_candidate in candidates_per_frame[keyframes[0]]:
        object_name = first_candidate.nearest_object
        second_candidates = [cand for cand in candidates_per_frame[keyframes[1]] if cand.nearest_object == object_name]
        if not second_candidates:
            continue
        best_second = min(
            second_candidates,
            key=lambda cand: (cand.rotation_distance_deg, cand.nearest_object_distance_m, -cand.score),
        )
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
    return best_selection


def choose_keyframes(
    renderer: ReplayRenderer,
    anygrasp_dir: Path,
    hand_data: Dict[str, np.ndarray],
    keyframes: Sequence[int],
    arm_mode: str,
    object_states_per_frame: Dict[int, Dict[str, ObjectState]],
) -> List[SelectedKeyframe]:
    candidate_arms = [arm_mode] if arm_mode in ("left", "right") else ["left", "right"]
    best: Optional[List[SelectedKeyframe]] = None
    best_cost = float("inf")
    for arm in candidate_arms:
        selection = select_keyframes_for_arm(renderer, anygrasp_dir, hand_data, keyframes, arm, object_states_per_frame)
        if selection is None:
            continue
        total_cost = sum(item.candidate.rotation_distance_deg for item in selection)
        if total_cost < best_cost:
            best_cost = total_cost
            best = selection
    if best is None:
        raise RuntimeError("Failed to find valid AnyGrasp candidates that match the requested hand orientation.")
    return best


def set_actor_pose(actor: sapien.Entity, pose_world_wxyz: np.ndarray) -> None:
    pose_world_wxyz = np.asarray(pose_world_wxyz, dtype=np.float64).reshape(7)
    actor.set_pose(sapien.Pose(pose_world_wxyz[:3], base.normalize_quat_wxyz(pose_world_wxyz[3:])))


def record_frame(
    renderer: ReplayRenderer,
    head_writer: cv2.VideoWriter,
    third_writer: Optional[cv2.VideoWriter],
    overlay_lines: Sequence[str],
    use_overlay: bool,
) -> None:
    renderer.update_robot_link_cameras()
    renderer.scene.update_render()
    head_rgb, _ = renderer.capture_camera(renderer.zed_camera)
    head_bgr = base.overlay_text(head_rgb, overlay_lines) if use_overlay else cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
    head_writer.write(head_bgr)
    if third_writer is not None:
        third_rgb, _ = renderer.capture_camera(renderer.third_camera)
        third_bgr = base.overlay_text(third_rgb, overlay_lines) if use_overlay else cv2.cvtColor(third_rgb, cv2.COLOR_RGB2BGR)
        third_writer.write(third_bgr)


def pose_with_offset_along_local_x(pose_world_wxyz: np.ndarray, offset_m: float) -> np.ndarray:
    pose_world_wxyz = np.asarray(pose_world_wxyz, dtype=np.float64).reshape(7)
    pose_world_matrix = pose_wxyz_to_matrix(pose_world_wxyz)
    pose_world_matrix[:3, 3] -= pose_world_matrix[:3, 0] * float(offset_m)
    rot = pose_world_matrix[:3, :3]
    quat = base.quat_xyzw_to_wxyz(R.from_matrix(rot).as_quat())
    return np.concatenate([pose_world_matrix[:3, 3], quat]).astype(np.float64)


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
) -> str:
    status = renderer._plan_status(plan)
    overlay_lines = [f"stage={label}", f"arm={arm}", f"status={status}"]
    if status != "Success":
        record_frame(renderer, head_writer, third_writer, overlay_lines, use_overlay)
        return status

    position = np.asarray(plan["position"], dtype=np.float64)
    velocity = np.asarray(plan["velocity"], dtype=np.float64)
    for idx in range(position.shape[0]):
        renderer.robot.set_arm_joints(position[idx], velocity[idx], arm)
        if attached_actor is not None and tcp_to_object is not None:
            tcp_pose = renderer.get_current_tcp_pose(arm)
            object_world = pose_wxyz_to_matrix(tcp_pose) @ tcp_to_object
            quat = base.quat_xyzw_to_wxyz(R.from_matrix(object_world[:3, :3]).as_quat())
            set_actor_pose(attached_actor, np.concatenate([object_world[:3, 3], quat]))
        renderer.step_scene(steps=1)
        record_frame(renderer, head_writer, third_writer, overlay_lines + [f"plan_step={idx + 1}/{position.shape[0]}"], use_overlay)
    renderer.step_scene(steps=4)
    record_frame(renderer, head_writer, third_writer, overlay_lines + ["plan_step=done"], use_overlay)
    return status


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
    object_states_per_frame = {frame: load_object_states(args.replay_dir, frame) for frame in keyframes}
    selected_keyframes = choose_keyframes(
        renderer=renderer,
        anygrasp_dir=args.anygrasp_dir,
        hand_data=hand_data,
        keyframes=keyframes,
        arm_mode=args.arm,
        object_states_per_frame=object_states_per_frame,
    )

    arm = selected_keyframes[0].arm
    primary_object_name = selected_keyframes[0].candidate.nearest_object

    object_states = object_states_per_frame[keyframes[0]]
    for state in object_states.values():
        state.actor = create_object_actor(renderer.scene, state.mesh_file, f"planned_object_{state.name}")
        set_actor_pose(state.actor, state.pose_world_wxyz)

    renderer.update_robot_link_cameras()
    renderer.step_scene(steps=1)
    renderer.set_grippers(args.open_gripper if arm == "left" else None, args.open_gripper if arm == "right" else None)

    head_video_path = args.output_dir / "head_cam_plan.mp4"
    third_video_path = args.output_dir / "third_cam_plan.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    head_writer = cv2.VideoWriter(str(head_video_path), fourcc, args.fps, (args.image_width, args.image_height))
    if not head_writer.isOpened():
        raise RuntimeError(f"Failed to open {head_video_path}")
    third_writer = None
    if bool(args.third_person_view) and not bool(args.head_only):
        third_writer = cv2.VideoWriter(str(third_video_path), fourcc, args.fps, (args.image_width, args.image_height))
        if not third_writer.isOpened():
            raise RuntimeError(f"Failed to open {third_video_path}")
    use_overlay = bool(args.overlay_text)

    try:
        record_frame(renderer, head_writer, third_writer, ["stage=init", f"arm={arm}", f"object={primary_object_name}"], use_overlay)

        grasp_pose = selected_keyframes[0].candidate.pose_world_wxyz
        pregrasp_pose = pose_with_offset_along_local_x(grasp_pose, args.approach_offset_m)
        action_pose = selected_keyframes[1].candidate.pose_world_wxyz

        pregrasp_plan = renderer.plan_path(arm, pregrasp_pose)
        pregrasp_status = execute_single_arm_plan(renderer, arm, pregrasp_plan, "pregrasp", head_writer, third_writer, use_overlay)

        grasp_plan = renderer.plan_path(arm, grasp_pose)
        grasp_status = execute_single_arm_plan(renderer, arm, grasp_plan, "grasp", head_writer, third_writer, use_overlay)

        renderer.set_grippers(args.close_gripper if arm == "left" else None, args.close_gripper if arm == "right" else None)
        record_frame(renderer, head_writer, third_writer, ["stage=close_gripper", f"arm={arm}"], use_overlay)

        attached_actor = object_states[primary_object_name].actor
        tcp_pose = renderer.get_current_tcp_pose(arm)
        tcp_to_object = np.linalg.inv(pose_wxyz_to_matrix(tcp_pose)) @ object_states[primary_object_name].pose_world_matrix

        action_plan = renderer.plan_path(arm, action_pose)
        action_status = execute_single_arm_plan(
            renderer,
            arm,
            action_plan,
            "action",
            head_writer,
            third_writer,
            use_overlay,
            attached_actor=attached_actor,
            tcp_to_object=tcp_to_object,
        )
    finally:
        head_writer.release()
        if third_writer is not None:
            third_writer.release()

    summary = {
        "anygrasp_dir": str(args.anygrasp_dir),
        "replay_dir": str(args.replay_dir),
        "hand_npz": str(args.hand_npz),
        "keyframes": keyframes,
        "selected_arm": arm,
        "selected_object": primary_object_name,
        "stages": {
            "pregrasp": pregrasp_status,
            "grasp": grasp_status,
            "action": action_status,
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
                "pose_world_wxyz": item.candidate.pose_world_wxyz.tolist(),
                "translation_cam": item.candidate.translation_cam.tolist(),
                "rotation_cam": item.candidate.rotation_cam.tolist(),
                "hand_rotation_cam": item.hand_rotation_cam.tolist(),
            }
            for item in selected_keyframes
        ],
        "head_video": str(head_video_path),
        "third_video": str(third_video_path) if third_writer is not None else None,
    }
    with (args.output_dir / "plan_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "[done] "
        f"arm={arm} object={primary_object_name} "
        f"statuses={summary['stages']} "
        f"head_video={head_video_path}"
    )
    renderer.hold_viewer()


if __name__ == "__main__":
    main()
