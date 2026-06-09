#!/usr/bin/env python3
"""Mode K (Ablation): Foundation Pose Target — object position + hand orientation as IK target.

Reads per-arm keyframes from hand_keyframes_all.json, extracts the foundation object's
world position and the human hand's world orientation at each keyframe, combines them
into a target pose, then feeds them to the existing planner via a generated
plan_summary.json (--reuse_plan_summary_json).

An additional retreat (--foundation_pose_retreat_m, default 3cm) is applied along the
gripper approach axis (local +Z = blue axis) to compensate for the offset between
the object center (foundation) and the object surface (grasp point).

No AnyGrasp candidates are used. This isolates:
- Foundation model's object localization (WHERE the object is)
- Human demonstration's grasp strategy (HOW to grasp the object)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from render_hand_retarget_r1_npz import CV_TO_WORLD_CAMERA_PRESETS

PLANNER_SCRIPT = THIS_DIR / "plan_anygrasp_keyframes_piper.py"
PIPER_CONFIG = PROJECT_ROOT / "robot_config_PiperPika_agx_dual_table_0515.json"

TASK_OBJECT_MAP = {
    "pick_diverse_bottles": {"left": "left_bottle", "right": "right_bottle"},
    "place_bread_basket": {"left": "basket", "right": "bread"},
    "stack_cups": {"left": "left_light_pink_cup", "right": "right_dark_red_cup"},
    "handover_bottle": {"left": "right_bottle", "right": "right_bottle"},
    "pnp_bread": {"left": "left_bread", "right": "right_bread"},
    "pnp_tray": {"left": "left_dark_red_cup", "right": "right_bottle"},
}


# ──────────────────────────────────────────────
# Keyframe resolution (shared with Mode J)
# ──────────────────────────────────────────────

def classify_keyframe_mode(video_info: dict) -> str:
    global_kf = [int(v) for v in video_info.get("keyframes", [])]
    left_kf = [int(v) for v in video_info.get("left_keyframes", [])]
    right_kf = [int(v) for v in video_info.get("right_keyframes", [])]
    if len(global_kf) >= 2 and len(left_kf) == 0 and len(right_kf) == 0:
        return "A"
    if len(left_kf) >= 2 and len(right_kf) >= 2 and len(global_kf) == 0:
        return "B"
    if len(global_kf) >= 1 and len(left_kf) >= 1 and len(right_kf) >= 1:
        return "C" if (len(left_kf) == 1 and len(right_kf) == 1) else "D"
    if len(global_kf) >= 1 and (len(left_kf) >= 1 or len(right_kf) >= 1):
        return "D"
    if len(global_kf) >= 2:
        return "A"
    return "B"


def resolve_effective_keyframes(video_info: dict) -> Tuple[List[int], Dict[str, List[int]]]:
    global_kf = [int(v) for v in video_info.get("keyframes", [])]
    effective_by_arm: Dict[str, List[int]] = {}
    for arm in ("left", "right"):
        arm_kf = [int(v) for v in video_info.get(f"{arm}_keyframes", [])]
        if len(arm_kf) >= 2:
            effective_by_arm[arm] = arm_kf[:2]
        else:
            combined = arm_kf + global_kf
            seen = set()
            deduped = [x for x in combined if not (x in seen or seen.add(x))]
            effective_by_arm[arm] = deduped[:2]
    if len(global_kf) >= 2:
        effective_global = global_kf[:2]
    else:
        all_frames = effective_by_arm["left"] + effective_by_arm["right"] + global_kf
        effective_global = sorted(set(all_frames))
    return effective_global, effective_by_arm


# ──────────────────────────────────────────────
# World-space pose computation
# ──────────────────────────────────────────────

def compute_hand_world_rotation(
    hand_data: Dict[str, np.ndarray],
    obj_data: Dict[str, np.ndarray],
    arm: str,
    frame: int,
    cv_axis_mode: str = "legacy_r1",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compute world-space hand rotation matrix and quaternion [qw,qx,qy,qz] at a frame.

    Returns (rot_world_3x3, quat_world_wxyz) or None.
    """
    rot_key = f"{arm}_gripper_rotation_matrix"
    valid_key = f"{arm}_gripper_valid"

    if rot_key not in hand_data:
        return None
    rotations = np.asarray(hand_data[rot_key])
    if frame < 0 or frame >= len(rotations):
        return None
    if valid_key in hand_data:
        valid_arr = np.asarray(hand_data[valid_key])
        if frame < len(valid_arr) and not bool(valid_arr[frame]):
            return None

    rotation_cam = rotations[frame].astype(np.float64)
    if not np.isfinite(rotation_cam).all():
        return None

    # Get head camera world pose
    head_cam_key = "head_camera_pose_world_wxyz"
    if head_cam_key not in obj_data:
        return None
    head_cam_poses = np.asarray(obj_data[head_cam_key])
    if frame >= len(head_cam_poses):
        return None

    head_pose_wxyz = head_cam_poses[frame].astype(np.float64)
    qw, qx, qy, qz = head_pose_wxyz[3:7]

    cam_cv_to_local = CV_TO_WORLD_CAMERA_PRESETS.get(cv_axis_mode, CV_TO_WORLD_CAMERA_PRESETS["legacy_r1"])
    rot_world_from_head = R.from_quat([qx, qy, qz, qw]).as_matrix()
    rot_world = rot_world_from_head @ cam_cv_to_local @ rotation_cam
    quat_xyzw = R.from_matrix(rot_world).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

    return rot_world, quat_wxyz


def get_object_world_position(
    obj_data: Dict[str, np.ndarray],
    object_name: str,
    frame: int,
) -> Optional[np.ndarray]:
    """Get foundation object world position (xyz) at a given frame."""
    pos_key = f"{object_name}__pose_world_wxyz"
    if pos_key not in obj_data:
        return None
    poses = np.asarray(obj_data[pos_key])
    if frame < 0 or frame >= len(poses):
        return None
    pose_wxyz = poses[frame].astype(np.float64)
    if not np.isfinite(pose_wxyz).all():
        return None
    return pose_wxyz[:3]  # [px, py, pz]


# ──────────────────────────────────────────────
# Target computation
# ──────────────────────────────────────────────

def compute_foundation_pose_target(
    hand_data: Dict[str, np.ndarray],
    obj_data: Dict[str, np.ndarray],
    arm: str,
    frame: int,
    object_name: str,
    retreat_m: float,
    cv_axis_mode: str = "legacy_r1",
) -> Optional[np.ndarray]:
    """Compute world-space target pose combining object position + hand orientation.

    Returns planner-order [px, py, pz, qw, qx, qy, qz] world-space pose, with retreat applied along
    the approach axis (local +Z = blue axis, gripper forward direction).
    """
    # Get object world position
    obj_pos = get_object_world_position(obj_data, object_name, frame)
    if obj_pos is None:
        return None

    # Get hand world rotation
    rot_result = compute_hand_world_rotation(hand_data, obj_data, arm, frame, cv_axis_mode)
    if rot_result is None:
        return None
    rot_world, quat_wxyz = rot_result

    # Apply retreat along approach axis (local +Z = blue = third column of rotation matrix)
    # Retreat = move BACKWARDS along approach axis (away from object)
    approach_axis_world = rot_world[:, 2]  # local +Z in world frame
    retreated_pos = obj_pos - retreat_m * approach_axis_world

    target_pose = np.concatenate([retreated_pos, quat_wxyz]).astype(np.float64)
    return target_pose


# ──────────────────────────────────────────────
# Plan summary generation
# ──────────────────────────────────────────────

def build_candidate_entry(
    arm: str,
    frame: int,
    pose_world_wxyz: np.ndarray,
    hand_rotation_cam: np.ndarray,
    target_object: str,
    candidate_idx: int = 0,
) -> dict:
    px, py, pz = float(pose_world_wxyz[0]), float(pose_world_wxyz[1]), float(pose_world_wxyz[2])
    qw, qx, qy, qz = float(pose_world_wxyz[3]), float(pose_world_wxyz[4]), float(pose_world_wxyz[5]), float(pose_world_wxyz[6])
    return {
        "source_frame": int(frame),
        "arm": str(arm),
        "candidate_idx": int(candidate_idx),
        "score": 1.0,
        "rotation_distance_deg": 0.0,
        "nearest_object": str(target_object),
        "nearest_object_distance_m": 0.0,
        "width_m": 0.08,
        "depth_m": 0.04,
        "top_axis_up_dot": 0.0,
        "original_top_axis_up_dot": 0.0,
        "camera_up_flip_applied": 0,
        "forward_axis_change_deg": 0.0,
        "camera_up_selection_mode": "none",
        "raw_pose_world_wxyz": [px, py, pz, qw, qx, qy, qz],
        "pose_world_wxyz": [px, py, pz, qw, qx, qy, qz],
        "translation_cam": [0.0, 0.0, 0.0],
        "rotation_cam": np.eye(3, dtype=np.float64).tolist(),
        "hand_rotation_cam": np.asarray(hand_rotation_cam, dtype=np.float64).tolist(),
    }


def build_plan_summary(
    task: str,
    video_id: int,
    effective_by_arm: Dict[str, List[int]],
    hand_data: Dict[str, np.ndarray],
    obj_data: Dict[str, np.ndarray],
    mode: str,
    retreat_m: float,
    cv_axis_mode: str = "legacy_r1",
) -> dict:
    obj_map = TASK_OBJECT_MAP.get(task, {"left": "object", "right": "object"})
    candidates_by_arm: Dict[str, List[dict]] = {"left": [], "right": []}

    for arm in ("left", "right"):
        keyframes = effective_by_arm.get(arm, [])
        target_obj = obj_map.get(arm, "object")

        rot_key = f"{arm}_gripper_rotation_matrix"
        hand_rotations = np.asarray(hand_data[rot_key]) if rot_key in hand_data else None

        for idx, kf_frame in enumerate(keyframes):
            if kf_frame <= 0:
                continue

            pose_wxyz = compute_foundation_pose_target(
                hand_data, obj_data, arm, kf_frame, target_obj, retreat_m, cv_axis_mode
            )
            if pose_wxyz is None:
                print(f"  [WARNING] arm={arm} frame={kf_frame}: could not compute foundation pose target, skipping")
                continue

            hand_rot_cam = hand_rotations[int(kf_frame)] if hand_rotations is not None and int(kf_frame) < len(hand_rotations) else np.eye(3)

            entry = build_candidate_entry(
                arm=arm,
                frame=kf_frame,
                pose_world_wxyz=pose_wxyz,
                hand_rotation_cam=hand_rot_cam,
                target_object=target_obj,
                candidate_idx=idx,
            )
            candidates_by_arm[arm].append(entry)

            print(f"  [{arm}] keyframe[{idx}] frame={kf_frame} obj={target_obj} retreat={retreat_m:.3f}m pos=({pose_wxyz[0]:.3f},{pose_wxyz[1]:.3f},{pose_wxyz[2]:.3f})")

    primary_arm = "left"
    if not candidates_by_arm["left"] and candidates_by_arm["right"]:
        primary_arm = "right"

    all_candidates = candidates_by_arm["left"] + candidates_by_arm["right"]
    if not all_candidates:
        raise ValueError("No valid targets computed for any arm")

    return {
        "task": task,
        "video_id": int(video_id),
        "target_source": "foundation_pose",
        "foundation_pose_retreat_m": float(retreat_m),
        "mode": mode,
        "selected_arm": primary_arm,
        "selected_candidates": all_candidates,
        "selected_candidates_by_executed_arm": {arm: entries for arm, entries in candidates_by_arm.items() if entries},
        "executed_arms": [arm for arm, entries in candidates_by_arm.items() if entries],
    }


# ──────────────────────────────────────────────
# Planner invocation
# ──────────────────────────────────────────────

def run_planner_with_targets(plan_summary_path: Path, args: argparse.Namespace, env: dict) -> bool:
    view_mode = bool(args.enable_viewer)
    cmd = [
        sys.executable, str(PLANNER_SCRIPT),
        "--anygrasp_dir", str(args.anygrasp_dir),
        "--replay_dir", str(args.replay_dir),
        "--hand_npz", str(args.hand_npz),
        "--output_dir", str(args.output_dir),
        "--reuse_plan_summary_json", str(plan_summary_path),
        "--arm", "auto",
        "--execute_both_arms", "1",
        "--dual_stage_require_all_plans", "1",
        "--require_keyframe1_reached_before_close", str(args.require_keyframe1_reached_before_close),
        "--require_keyframe1_reached_before_action", str(args.require_keyframe1_reached_before_action),
        "--planner_backend", args.planner_backend,
        "--urdfik_trajectory_mode", args.urdfik_trajectory_mode,
        "--urdfik_joint_interp_waypoints", str(args.urdfik_joint_interp_waypoints),
        "--urdfik_cartesian_interp_auto_step_m", str(args.urdfik_cartesian_interp_auto_step_m),
        "--execute_partial_cartesian_plan", str(args.execute_partial_cartesian_plan),
        "--urdfik_max_position_threshold_m", str(args.urdfik_max_position_threshold_m),
        "--urdfik_max_rotation_threshold_rad", str(args.urdfik_max_rotation_threshold_rad),
        "--candidate_orientation_remap_label", args.candidate_orientation_remap_label,
        "--candidate_target_local_x_offset_m", "0.0",
        "--candidate_target_local_z_offset_m", "0.0",
        "--approach_axis", args.approach_axis,
        "--approach_offset_m", str(args.approach_offset_m),
        "--reach_error_pose_source", args.reach_error_pose_source,
        "--reach_pos_tol_m", str(args.reach_pos_tol_m),
        "--reach_rot_tol_deg", str(args.reach_rot_tol_deg),
        "--replan_until_reached", "1",
        "--replan_until_reached_max_attempts", str(args.replan_until_reached_max_attempts),
        "--execute_interp_steps", str(args.execute_interp_steps),
        "--joint_command_scene_steps", str(args.joint_command_scene_steps),
        "--settle_steps", str(args.settle_steps),
        "--joint_target_wait_steps", str(args.joint_target_wait_steps),
        "--joint_target_wait_tol_rad", str(args.joint_target_wait_tol_rad),
        "--hold_frames_after_stage", str(args.hold_frames_after_stage),
        "--save_pose_debug", "1",
        "--save_debug_preview", "1",
        "--debug_visualize_targets", str(args.debug_visualize_targets),
        "--debug_candidate_top_k", str(args.debug_candidate_top_k),
        "--debug_common_candidate_top_k", "0",
        "--debug_visualize_selected_keyframe_axes", str(args.debug_visualize_selected_keyframe_axes),
        "--debug_visualize_ik_waypoints", str(args.debug_visualize_ik_waypoints),
        "--debug_visualize_cameras", str(args.debug_visualize_cameras),
        "--viewer_show_camera_frustums", str(args.viewer_show_camera_frustums),
        "--debug_gripper_actor_forward_axis", args.debug_gripper_actor_forward_axis,
        "--enable_grasp_action_object_collision", str(args.enable_grasp_action_object_collision),
        "--replay_objects_ignore_collision", str(args.replay_objects_ignore_collision),
        "--pure_scene_output", str(args.pure_scene_output),
        "--overlay_text", "0",
        "--vscode_compatible_video", "1",
        "--lighting_mode", args.lighting_mode,
        "--camera_cv_axis_mode", args.camera_cv_axis_mode,
        "--head_camera_local_pos", *[str(v) for v in args.head_camera_local_pos],
        "--head_camera_local_quat_wxyz", *[str(v) for v in args.head_camera_local_quat_wxyz],
        "--third_person_view", str(args.third_person_view),
        "--head_only", str(args.head_only),
        "--image_width", str(args.image_width),
        "--image_height", str(args.image_height),
        "--fovy_deg", str(args.fovy_deg),
        "--fps", str(int(args.fps)),
        "--robot_config", str(args.robot_config),
    ]

    if view_mode:
        cmd += [
            "--enable_viewer", "1",
            "--viewer_wait_at_end", str(args.viewer_wait_at_end),
            "--viewer_frame_delay", str(args.viewer_frame_delay),
        ]

    obj_map = TASK_OBJECT_MAP.get(args.task, {})
    if obj_map.get("left"):
        cmd += ["--left_target_object", obj_map["left"]]
    if obj_map.get("right"):
        cmd += ["--right_target_object", obj_map["right"]]

    # Mesh/scale overrides (passed as single key=value strings matching planner convention)
    for override in args.object_mesh_override:
        cmd += ["--object_mesh_override", str(override)]
    for override in args.execution_object_visual_scale_override:
        cmd += ["--execution_object_visual_scale_override", str(override)]
    for override in args.execution_object_collision_scale_override:
        cmd += ["--execution_object_collision_scale_override", str(override)]

    print(f"\n[foundation-pose] Running planner...")
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mode K: Foundation Pose Target — object position + hand orientation.")
    parser.add_argument("--anygrasp_dir", type=Path, required=True)
    parser.add_argument("--replay_dir", type=Path, required=True)
    parser.add_argument("--hand_npz", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hand_keyframes_json", type=Path, required=True)
    parser.add_argument("--video_id", type=int, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--robot_config", type=Path, default=PIPER_CONFIG)
    parser.add_argument("--gpu", type=int, default=2)

    # Mode K specific
    parser.add_argument("--foundation_pose_retreat_m", type=float, default=0.03,
                        help="Retreat distance along gripper approach axis (local +Z, blue) from object center to grasp surface. Default 3cm.")

    # Planner args
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--fovy_deg", type=float, default=42.499880046655484)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--enable_viewer", type=int, default=0)
    parser.add_argument("--viewer_wait_at_end", type=int, default=0)
    parser.add_argument("--viewer_frame_delay", type=float, default=0.02)
    parser.add_argument("--lighting_mode", type=str, default="front_no_shadow")
    parser.add_argument("--camera_cv_axis_mode", type=str, default="legacy_r1")
    parser.add_argument("--head_camera_local_pos", type=float, nargs=3, default=[0.11210396690038413, -0.39189397826604927, 0.4753892624100325])
    parser.add_argument("--head_camera_local_quat_wxyz", type=float, nargs=4, default=[0.8524694864910365, -0.0011011947849308937, 0.5226654778798345, 0.010740586780925399])
    parser.add_argument("--third_person_view", type=int, default=1)
    parser.add_argument("--head_only", type=int, default=0)
    parser.add_argument("--planner_backend", choices=["urdfik", "curobo"], default="urdfik")
    parser.add_argument("--urdfik_trajectory_mode", type=str, default="cartesian_interp_ik")
    parser.add_argument("--urdfik_joint_interp_waypoints", type=int, default=40)
    parser.add_argument("--urdfik_cartesian_interp_steps", type=int, default=-1)
    parser.add_argument("--urdfik_cartesian_interp_auto_step_m", type=float, default=0.03)
    parser.add_argument("--execute_partial_cartesian_plan", type=int, default=1)
    parser.add_argument("--urdfik_max_position_threshold_m", type=float, default=0.02)
    parser.add_argument("--urdfik_max_rotation_threshold_rad", type=float, default=3.14)
    parser.add_argument("--urdfik_num_seeds", type=int, default=20)
    parser.add_argument("--candidate_orientation_remap_label", type=str, default="identity")
    parser.add_argument("--approach_axis", type=str, default="local_z")
    parser.add_argument("--approach_offset_m", type=float, default=0.12)
    parser.add_argument("--require_keyframe1_reached_before_close", type=int, default=1)
    parser.add_argument("--require_keyframe1_reached_before_action", type=int, default=1)
    parser.add_argument("--execute_interp_steps", type=int, default=24)
    parser.add_argument("--joint_command_scene_steps", type=int, default=10)
    parser.add_argument("--settle_steps", type=int, default=30)
    parser.add_argument("--joint_target_wait_steps", type=int, default=25)
    parser.add_argument("--joint_target_wait_tol_rad", type=float, default=0.01)
    parser.add_argument("--hold_frames_after_stage", type=int, default=8)
    parser.add_argument("--reach_pos_tol_m", type=float, default=0.03)
    parser.add_argument("--reach_rot_tol_deg", type=float, default=180)
    parser.add_argument("--replan_until_reached_max_attempts", type=int, default=3, help="Max replan attempts per stage before giving up (default 3, 0=unbounded)")
    parser.add_argument("--reach_error_pose_source", type=str, default="ee")
    parser.add_argument("--debug_visualize_targets", type=int, default=1)
    parser.add_argument("--debug_candidate_top_k", type=int, default=1)
    parser.add_argument("--debug_visualize_selected_keyframe_axes", type=int, default=1)
    parser.add_argument("--debug_visualize_ik_waypoints", type=int, default=1)
    parser.add_argument("--debug_visualize_cameras", type=int, default=0)
    parser.add_argument("--viewer_show_camera_frustums", type=int, default=0)
    parser.add_argument("--debug_gripper_actor_forward_axis", type=str, default="local_z")
    parser.add_argument("--pure_scene_output", type=int, default=1)
    parser.add_argument("--enable_grasp_action_object_collision", type=int, default=0)
    parser.add_argument("--replay_objects_ignore_collision", type=int, default=1)
    parser.add_argument("--object_mesh_override", action="append", default=[], help="Repeatable mesh override in the form NAME=/abs/path/to/mesh.obj")
    parser.add_argument("--execution_object_visual_scale_override", action="append", default=[], help="Repeatable visual scale override NAME=SCALE")
    parser.add_argument("--execution_object_collision_scale_override", action="append", default=[], help="Repeatable collision scale override NAME=SCALE")

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    args.anygrasp_dir = args.anygrasp_dir.resolve()
    args.replay_dir = args.replay_dir.resolve()
    args.hand_npz = args.hand_npz.resolve()
    args.output_dir = args.output_dir.resolve()
    args.hand_keyframes_json = args.hand_keyframes_json.resolve()
    args.robot_config = args.robot_config.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[foundation-pose] task={args.task} video_id={args.video_id} retreat={args.foundation_pose_retreat_m}m output={args.output_dir}", flush=True)

    with open(args.hand_keyframes_json, "r") as f:
        hand_annotations = json.load(f)

    video_name = f"hand_vis_{args.video_id}.mp4"
    videos = hand_annotations.get("videos", {})
    if video_name not in videos:
        raise KeyError(f"{video_name} not found in {args.hand_keyframes_json}")
    video_info = videos[video_name]
    status = video_info.get("status", "")
    if status in ("reject", "discard", "bad"):
        print(f"[foundation-pose] SKIP video_id={args.video_id} status={status}")
        return

    mode = classify_keyframe_mode(video_info)
    effective_global, effective_by_arm = resolve_effective_keyframes(video_info)
    print(f"[foundation-pose] mode={mode} effective_by_arm={effective_by_arm}")

    if not any(effective_by_arm.values()):
        print(f"[foundation-pose] SKIP: no effective keyframes")
        return

    hand_data = dict(np.load(str(args.hand_npz), allow_pickle=True))
    obj_npz_path = args.replay_dir / "multi_object_world_poses.npz"
    if not obj_npz_path.is_file():
        raise FileNotFoundError(f"Object NPZ not found: {obj_npz_path}")
    obj_data = dict(np.load(str(obj_npz_path), allow_pickle=True))

    print(f"[foundation-pose] Computing object+hand targets (retreat={args.foundation_pose_retreat_m}m)...")
    plan_summary = build_plan_summary(
        task=args.task,
        video_id=args.video_id,
        effective_by_arm=effective_by_arm,
        hand_data=hand_data,
        obj_data=obj_data,
        mode=mode,
        retreat_m=args.foundation_pose_retreat_m,
        cv_axis_mode=args.camera_cv_axis_mode,
    )

    plan_summary_path = args.output_dir / "plan_summary_foundation_pose.json"
    with open(plan_summary_path, "w") as f:
        json.dump(plan_summary, f, indent=2)
    print(f"[foundation-pose] Wrote plan summary: {plan_summary_path}")

    env = os.environ.copy()
    if bool(args.enable_viewer):
        env.pop("CUDA_VISIBLE_DEVICES", None)
    elif args.gpu >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    success = run_planner_with_targets(plan_summary_path, args, env)
    print(f"\n[foundation-pose] {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
