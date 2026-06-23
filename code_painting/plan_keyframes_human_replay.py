#!/usr/bin/env python3
"""Mode J (Ablation): Human Replay Target — use hand gripper pose directly as IK target.

Reads per-arm keyframes from hand_keyframes_all.json, computes world-space hand
gripper poses at each keyframe frame, then feeds them to the existing planner
via a generated plan_summary.json (--reuse_plan_summary_json).

No AnyGrasp candidates are used. The planner's pregrasp/grasp/action pipeline
and IK solver remain unchanged — only the target source differs.

This isolates the effect of AnyGrasp's grasp ranking from the IK planning pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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

# Reuse constants from the renderer
from render_hand_retarget_r1_npz import CV_TO_WORLD_CAMERA_PRESETS

PLANNER_SCRIPT = THIS_DIR / "plan_anygrasp_keyframes_piper.py"
PIPER_CONFIG = PROJECT_ROOT / "robot_config_PiperPika_agx_dual_table_0515.json"
IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

# Per-arm task object mapping
TASK_OBJECT_MAP = {
    "pick_diverse_bottles": {"left": "left_bottle", "right": "right_bottle"},
    "place_bread_basket": {"left": "bread", "right": "basket"},
    "stack_cups": {"left": "left_light_pink_cup", "right": "right_dark_red_cup"},
    "handover_bottle": {"left": "right_bottle", "right": "right_bottle"},
    "pnp_bread": {"left": "left_bread", "right": "right_bread"},
    "pnp_tray": {"left": "left_dark_red_cup", "right": "right_bottle"},
}

# ──────────────────────────────────────────────
# Keyframe resolution
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
    """Resolve effective keyframes per arm, matching render_anygrasp_ranked_preview logic."""
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
# World-space hand pose computation
# ──────────────────────────────────────────────

def compute_hand_world_pose(
    hand_data: Dict[str, np.ndarray],
    obj_data: Dict[str, np.ndarray],
    arm: str,
    frame: int,
    cv_axis_mode: str = "legacy_r1",
) -> Optional[np.ndarray]:
    """Compute planner-order world-space hand gripper pose [px,py,pz,qw,qx,qy,qz]."""
    pos_key = f"{arm}_gripper_position"
    rot_key = f"{arm}_gripper_rotation_matrix"
    valid_key = f"{arm}_gripper_valid"

    if pos_key not in hand_data or rot_key not in hand_data:
        return None

    positions = np.asarray(hand_data[pos_key])
    rotations = np.asarray(hand_data[rot_key])

    if frame < 0 or frame >= len(positions):
        return None

    # Check validity
    if valid_key in hand_data:
        valid_arr = np.asarray(hand_data[valid_key])
        if frame < len(valid_arr) and not bool(valid_arr[frame]):
            return None

    position_cam = positions[frame].astype(np.float64)
    rotation_cam = rotations[frame].astype(np.float64)

    if not np.isfinite(position_cam).all() or not np.isfinite(rotation_cam).all():
        return None

    # Get head camera world pose from object NPZ
    head_cam_key = "head_camera_pose_world_wxyz"
    if head_cam_key not in obj_data:
        return None

    head_cam_poses = np.asarray(obj_data[head_cam_key])
    if frame >= len(head_cam_poses):
        return None

    head_pose_wxyz = head_cam_poses[frame].astype(np.float64)
    head_pos_world = head_pose_wxyz[:3]
    qw, qx, qy, qz = head_pose_wxyz[3:7]

    # Camera CV axis remap
    cam_cv_to_local = CV_TO_WORLD_CAMERA_PRESETS.get(cv_axis_mode, CV_TO_WORLD_CAMERA_PRESETS["legacy_r1"])

    # World rotation from head camera
    rot_world_from_head = R.from_quat([qx, qy, qz, qw]).as_matrix()

    # Convert position to world space
    pos_world = head_pos_world + rot_world_from_head @ (cam_cv_to_local @ position_cam)

    # Convert rotation to world space
    rot_world = rot_world_from_head @ cam_cv_to_local @ rotation_cam
    quat_xyzw = R.from_matrix(rot_world).as_quat()  # scipy returns xyzw
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)

    return np.concatenate([pos_world, quat_wxyz]).astype(np.float64)


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
    """Build a single candidate entry in plan_summary.json format."""
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
    cv_axis_mode: str = "legacy_r1",
    action_orientation_source: str = "keyframe",
    target_retreat_m: float = 0.0,
) -> dict:
    """Build a complete plan_summary.json."""
    obj_map = TASK_OBJECT_MAP.get(task, {"left": "object", "right": "object"})
    candidates_by_arm: Dict[str, List[dict]] = {"left": [], "right": []}

    for arm in ("left", "right"):
        keyframes = effective_by_arm.get(arm, [])
        target_obj = obj_map.get(arm, "object")

        # Get hand rotation in camera space for the reference
        rot_key = f"{arm}_gripper_rotation_matrix"
        hand_rotations = np.asarray(hand_data[rot_key]) if rot_key in hand_data else None
        grasp_pose_wxyz: Optional[np.ndarray] = None

        for idx, kf_frame in enumerate(keyframes):
            if kf_frame <= 0:
                continue

            # Compute world-space hand pose
            pose_wxyz = compute_hand_world_pose(hand_data, obj_data, arm, kf_frame, cv_axis_mode)
            if pose_wxyz is None:
                print(f"  [WARNING] arm={arm} frame={kf_frame}: could not compute world pose, skipping")
                continue
            # Apply target retreat along gripper local Z (approach axis).
            # The hand keyframe gives the TCP pose; retreat converts to link6 target.
            # Set --target_retreat_m to gripper_bias (e.g. 0.12 for Piper) so that
            # link6 is placed behind the TCP and the actual TCP reaches the hand position.
            if abs(target_retreat_m) > 1e-12:
                quat_xyzw = [float(pose_wxyz[4]), float(pose_wxyz[5]), float(pose_wxyz[6]), float(pose_wxyz[3])]
                local_z = R.from_quat(quat_xyzw).as_matrix()[:, 2]
                pose_wxyz[:3] = pose_wxyz[:3] - local_z * float(target_retreat_m)
            orientation_source = f"hand_frame_{int(kf_frame)}"
            if idx > 0 and action_orientation_source == "grasp" and grasp_pose_wxyz is not None:
                pose_wxyz = np.concatenate([pose_wxyz[:3], grasp_pose_wxyz[3:]]).astype(np.float64)
                orientation_source = f"grasp_frame_{int(keyframes[0])}"
            elif idx == 0:
                grasp_pose_wxyz = pose_wxyz.copy()

            hand_rot_cam = hand_rotations[int(kf_frame)] if hand_rotations is not None and int(kf_frame) < len(hand_rotations) else np.eye(3)

            entry = build_candidate_entry(
                arm=arm,
                frame=kf_frame,
                pose_world_wxyz=pose_wxyz,
                hand_rotation_cam=hand_rot_cam,
                target_object=target_obj,
                candidate_idx=idx,
            )
            entry["human_replay_orientation_source"] = orientation_source
            candidates_by_arm[arm].append(entry)
            print(
                f"  [{arm}] keyframe[{idx}] frame={kf_frame} orientation={orientation_source} "
                f"pos=({pose_wxyz[0]:.3f},{pose_wxyz[1]:.3f},{pose_wxyz[2]:.3f})"
            )

    # ── Place strategy tasks ──
    # pnp_tray:   offset=2cm  (cup/bottle are fragile, place gently on tray)
    # pnp_bread:  offset=5cm  (bread can drop onto plate safely)
    # stack_cups: offset=1cm  (stack tightly on the cup below)
    # place_bread_basket: left arm places bread (offset=2cm, retreat=10cm),
    #                      right arm just lifts basket (no place strategy)
    place_raise_m = 0.0
    g1_tcp_z_by_arm: Dict[str, float] = {}
    place_lower_offset_m = 0.02
    place_strategy_per_arm: Dict[str, bool] = {}
    retreat_raise_m_by_arm: Dict[str, float] = {}
    place_lower_offset_by_arm: Dict[str, float] = {}
    if task == "pnp_tray":
        place_raise_m = 0.05
        place_lower_offset_m = 0.02
    elif task == "pnp_bread":
        place_raise_m = 0.05
        place_lower_offset_m = 0.05
    elif task == "stack_cups":
        place_raise_m = 0.05
        place_lower_offset_m = 0.01
    elif task == "place_bread_basket":
        place_raise_m = 0.05
        place_lower_offset_m = 0.02
        # Left: full place (approach+5cm, lower 2cm for bread, retreat 10cm).
        #   G2 Z +5cm because the gripper can't reach inside the basket like a human hand.
        # Right: no place strategy, just R1 pick→close→R2 lift 5cm above R1 TCP.
        place_strategy_per_arm = {"left": True, "right": False}
        retreat_raise_m_by_arm = {"left": 0.10, "right": 0.0}
        place_lower_offset_by_arm = {"left": 0.02, "right": 0.0}
    # handover_bottle uses a different strategy (handover transfer),
    # not the generic place strategy. Marked via handover_strategy flag.
    handover_strategy = (task == "handover_bottle")
    _place_tasks = ("pnp_tray", "pnp_bread", "stack_cups", "place_bread_basket")
    if task in _place_tasks:
        for arm in ("left", "right"):
            entries = candidates_by_arm.get(arm, [])
            if len(entries) < 2:
                continue
            g1_entry = entries[0]
            g2_entry = entries[1]
            g1_pose = np.array(g1_entry["pose_world_wxyz"], dtype=np.float64)
            g2_pose = np.array(g2_entry["pose_world_wxyz"], dtype=np.float64)

            # Recover G1 TCP Z
            g1_quat_xyzw = [float(g1_pose[4]), float(g1_pose[5]), float(g1_pose[6]), float(g1_pose[3])]
            g1_local_z = R.from_quat(g1_quat_xyzw).as_matrix()[:, 2]
            g1_tcp_z = float(g1_pose[2] + g1_local_z[2] * float(target_retreat_m))
            g1_tcp_z_by_arm[arm] = g1_tcp_z

            _use_place = place_strategy_per_arm.get(arm, True)
            # place_bread_basket adjustments:
            #   Left G2: +5cm Z (gripper can't reach inside basket like human hand).
            #   Right G2: lift to R1 TCP Z + 5cm (just lift the basket, keep gripper closed).
            if task == "place_bread_basket":
                if arm == "left":
                    g2_pose[2] += 0.05
                    g2_entry["pose_world_wxyz"] = g2_pose.tolist()
                elif arm == "right":
                    _r1_tcp_z = g1_tcp_z
                    _new_z = _r1_tcp_z + 0.05 - g1_local_z[2] * float(target_retreat_m)
                    g2_pose[2] = _new_z
                    g2_entry["pose_world_wxyz"] = g2_pose.tolist()
            print(
                f"  [{task}] {arm} G1 TCP Z={g1_tcp_z:.3f} "
                f"(place_strategy={_use_place}, lower-to={g1_tcp_z + place_lower_offset_m:.3f})"
            )

    # Determine primary arm
    primary_arm = "left"
    if not candidates_by_arm["left"] and candidates_by_arm["right"]:
        primary_arm = "right"

    # Determine execution arm order based on keyframe timing.
    # For stack_cups, right hand grabs first (R1 < L1), so right arm must
    # execute completely (including open_gripper) before left arm starts,
    # otherwise the two arms collide in the shared workspace.
    execution_arm_order = ["left", "right"]
    if task == "stack_cups":
        left_kf = effective_by_arm.get("left", [])
        right_kf = effective_by_arm.get("right", [])
        if left_kf and right_kf and int(right_kf[0]) < int(left_kf[0]):
            execution_arm_order = ["right", "left"]
            print(f"  [stack_cups] execution order: right first (R1={right_kf[0]} < L1={left_kf[0]})")

    all_candidates = candidates_by_arm["left"] + candidates_by_arm["right"]
    if not all_candidates:
        raise ValueError("No valid targets computed for any arm")

    return {
        "task": task,
        "video_id": int(video_id),
        "target_source": "human_replay",
        "human_replay_action_orientation_source": str(action_orientation_source),
        "human_replay_target_retreat_m": float(target_retreat_m),
        "mode": mode,
        "place_strategy": "raise_above_then_lower" if place_raise_m > 0 else "none",
        "place_raise_m": float(place_raise_m),
        "place_lower_offset_m": float(place_lower_offset_m),
        "g1_tcp_z_by_arm": {arm: float(z) for arm, z in g1_tcp_z_by_arm.items()},
        "place_strategy_per_arm": {str(a): bool(v) for a, v in place_strategy_per_arm.items()},
        "retreat_raise_m_by_arm": {str(a): float(v) for a, v in retreat_raise_m_by_arm.items()},
        "place_lower_offset_by_arm": {str(a): float(v) for a, v in place_lower_offset_by_arm.items()},
        "handover_strategy": bool(handover_strategy),
        "execution_arm_order": [str(a) for a in execution_arm_order],
        "selected_arm": primary_arm,
        "selected_candidates": all_candidates,
        "selected_candidates_by_executed_arm": {arm: entries for arm in execution_arm_order for entries in [candidates_by_arm.get(arm, [])] if entries},
        "executed_arms": [arm for arm, entries in candidates_by_arm.items() if entries],
    }


# ──────────────────────────────────────────────
# Planner invocation
# ──────────────────────────────────────────────

def run_planner_with_targets(
    plan_summary_path: Path,
    args: argparse.Namespace,
    env: dict,
) -> bool:
    """Run the existing planner with a generated plan_summary.json."""
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
        "--dual_stage_freeze_reached_arms_on_replan", str(args.dual_stage_freeze_reached_arms_on_replan),
        "--require_keyframe1_reached_before_close", str(args.require_keyframe1_reached_before_close),
        "--require_keyframe1_reached_before_action", str(args.require_keyframe1_reached_before_action),
        "--planner_backend", args.planner_backend,
        "--urdfik_trajectory_mode", args.urdfik_trajectory_mode,
        "--urdfik_joint_interp_waypoints", str(args.urdfik_joint_interp_waypoints),
        "--urdfik_cartesian_interp_auto_step_m", str(args.urdfik_cartesian_interp_auto_step_m),
        "--urdfik_num_seeds", str(args.urdfik_num_seeds),
        "--urdfik_solution_selection", args.urdfik_solution_selection,
        "--urdfik_seed_perturbations", str(args.urdfik_seed_perturbations),
        "--urdfik_seed_perturbation_scale", str(args.urdfik_seed_perturbation_scale),
        "--urdfik_max_joint_step_rad", str(args.urdfik_max_joint_step_rad),
        "--execute_partial_cartesian_plan", str(args.execute_partial_cartesian_plan),
        "--urdfik_max_position_threshold_m", str(args.urdfik_max_position_threshold_m),
        "--urdfik_max_rotation_threshold_rad", str(args.urdfik_max_rotation_threshold_rad),
        "--piper_urdfik_apply_global_trans_to_ik", str(args.piper_urdfik_apply_global_trans_to_ik),
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
        "--joint_trajectory_interpolation", args.joint_trajectory_interpolation,
        "--joint_command_scene_steps", str(args.joint_command_scene_steps),
        "--settle_steps", str(args.settle_steps),
        "--joint_target_wait_steps", str(args.joint_target_wait_steps),
        "--joint_target_wait_tol_rad", str(args.joint_target_wait_tol_rad),
        "--hold_frames_after_stage", str(args.hold_frames_after_stage),
        "--save_pose_debug", "1",
        "--save_debug_preview", "1",
        "--debug_visualize_targets", str(args.debug_visualize_targets),
        "--debug_candidate_top_k", "0",
        "--debug_common_candidate_top_k", "0",
        "--debug_visualize_selected_keyframe_axes", str(args.debug_visualize_selected_keyframe_axes),
        "--debug_visualize_ik_waypoints", str(args.debug_visualize_ik_waypoints),
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
        *(["--piper_calibration_bundle", str(args.piper_calibration_bundle)] if args.piper_calibration_bundle else []),
        "--fail_on_execution_failure", str(args.fail_on_execution_failure),
        "--wrist_preview", str(args.wrist_preview),
        "--viewer_show_camera_frustums", str(args.viewer_show_camera_frustums),
        "--debug_visualize_cameras", str(args.debug_visualize_cameras),
        "--debug_camera_axis_length", str(args.debug_camera_axis_length),
        # Wrist camera tuning (same convention as O.1)
        "--wrist_left_forward_offset_m", str(args.wrist_left_forward_offset_m),
        "--wrist_right_forward_offset_m", str(args.wrist_right_forward_offset_m),
        "--wrist_left_lateral_offset_m", str(args.wrist_left_lateral_offset_m),
        "--wrist_right_lateral_offset_m", str(args.wrist_right_lateral_offset_m),
        "--wrist_left_roll_deg", str(args.wrist_left_roll_deg),
        "--wrist_right_roll_deg", str(args.wrist_right_roll_deg),
        "--wrist_left_yaw_deg", str(args.wrist_left_yaw_deg),
        "--wrist_right_yaw_deg", str(args.wrist_right_yaw_deg),
        "--wrist_left_pitch_deg", str(args.wrist_left_pitch_deg),
        "--wrist_right_pitch_deg", str(args.wrist_right_pitch_deg),
    ]

    if view_mode:
        cmd += [
            "--enable_viewer", "1",
            "--viewer_wait_at_end", str(args.viewer_wait_at_end),
            "--viewer_frame_delay", str(args.viewer_frame_delay),
        ]

    # Task-specific object args
    obj_map = TASK_OBJECT_MAP.get(args.task, {})
    if obj_map.get("left"):
        cmd += ["--left_target_object", obj_map["left"]]
    if obj_map.get("right"):
        cmd += ["--right_target_object", obj_map["right"]]

    # Mesh/scale overrides (passed as single key=value strings)
    for override in args.object_mesh_override:
        cmd += ["--object_mesh_override", str(override)]
    for override in args.execution_object_visual_scale_override:
        cmd += ["--execution_object_visual_scale_override", str(override)]
    for override in args.execution_object_collision_scale_override:
        cmd += ["--execution_object_collision_scale_override", str(override)]

    print(f"\n[human-replay] Running planner...")
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mode J: Human Replay Target — use hand gripper pose directly as IK target.")
    parser.add_argument("--anygrasp_dir", type=Path, required=True, help="AnyGrasp result dir (used for object mesh paths only)")
    parser.add_argument("--replay_dir", type=Path, required=True, help="Per-video D435 replay dir (contains multi_object_world_poses.npz)")
    parser.add_argument("--hand_npz", type=Path, required=True, help="Hand detections NPZ")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hand_keyframes_json", type=Path, required=True, help="Path to hand_keyframes_all.json")
    parser.add_argument("--video_id", type=int, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--robot_config", type=Path, default=PIPER_CONFIG)
    parser.add_argument("--gpu", type=int, default=2)

    # Planner args (same defaults as L15.19.1)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--fovy_deg", type=float, default=42.499880046655484)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--enable_viewer", type=int, default=0)
    parser.add_argument("--viewer_wait_at_end", type=int, default=0)
    parser.add_argument("--viewer_frame_delay", type=float, default=0.02)
    parser.add_argument("--wrist_preview", type=int, default=0, help="If 1, show live left+right wrist camera preview window during viewer mode.")
    parser.add_argument("--viewer_show_camera_frustums", type=int, default=0, help="If 1, keep SAPIEN viewer camera frustum lines visible.")
    parser.add_argument("--debug_visualize_cameras", type=int, default=0, help="If 1, draw camera axis actors (RGB=XYZ) for head and wrist cameras.")
    parser.add_argument("--debug_camera_axis_length", type=float, default=0.10)
    parser.add_argument("--lighting_mode", type=str, default="front_no_shadow")
    parser.add_argument("--camera_cv_axis_mode", type=str, default="legacy_r1")
    parser.add_argument("--head_camera_local_pos", type=float, nargs=3, default=[0.11210396690038413, -0.39189397826604927, 0.4753892624100325])
    parser.add_argument("--head_camera_local_quat_wxyz", type=float, nargs=4, default=[0.8524694864910365, -0.0011011947849308937, 0.5226654778798345, 0.010740586780925399])
    parser.add_argument("--third_person_view", type=int, default=1)
    parser.add_argument("--head_only", type=int, default=0)
    parser.add_argument("--planner_backend", choices=["urdfik", "curobo"], default="urdfik")
    parser.add_argument("--urdfik_trajectory_mode", type=str, default="joint_interp")
    parser.add_argument("--urdfik_joint_interp_waypoints", type=int, default=40)
    parser.add_argument("--urdfik_cartesian_interp_steps", type=int, default=-1)
    parser.add_argument("--urdfik_cartesian_interp_auto_step_m", type=float, default=0.03)
    parser.add_argument("--execute_partial_cartesian_plan", type=int, default=0)
    parser.add_argument("--urdfik_max_position_threshold_m", type=float, default=0.02)
    parser.add_argument("--urdfik_max_rotation_threshold_rad", type=float, default=3.14)
    parser.add_argument("--urdfik_num_seeds", type=int, default=1)
    parser.add_argument("--urdfik_solution_selection", choices=["pose_error", "joint_continuity"], default="joint_continuity")
    parser.add_argument("--urdfik_seed_perturbations", type=int, default=6)
    parser.add_argument("--urdfik_seed_perturbation_scale", type=float, default=0.05)
    parser.add_argument("--urdfik_max_joint_step_rad", type=float, default=0.0)
    parser.add_argument("--piper_urdfik_apply_global_trans_to_ik", type=int, default=0)
    parser.add_argument("--candidate_orientation_remap_label", type=str, default="identity")
    parser.add_argument("--approach_axis", type=str, default="local_z")
    parser.add_argument("--approach_offset_m", type=float, default=0.12)
    parser.add_argument("--piper_calibration_bundle", type=Path, default=None, help="Optional self-contained Piper calibration bundle; overrides robot_config and wrist camera local poses.")
    parser.add_argument("--target_retreat_m", type=float, default=0.0,
                        help="Offset grasp target backward along approach axis (local Z) to convert hand TCP to link6 target. Set to gripper_bias (e.g. 0.12 for Piper) to compensate for wrist-to-tip distance.")
    parser.add_argument("--action_orientation_source", choices=["keyframe", "grasp"], default="grasp")
    parser.add_argument("--dual_stage_freeze_reached_arms_on_replan", type=int, default=1)
    parser.add_argument("--require_keyframe1_reached_before_close", type=int, default=1)
    parser.add_argument("--require_keyframe1_reached_before_action", type=int, default=1)
    parser.add_argument("--execute_interp_steps", type=int, default=24)
    parser.add_argument("--joint_trajectory_interpolation", choices=["linear", "cubic"], default="cubic")
    parser.add_argument("--joint_command_scene_steps", type=int, default=10)
    parser.add_argument("--settle_steps", type=int, default=30)
    parser.add_argument("--joint_target_wait_steps", type=int, default=25)
    parser.add_argument("--joint_target_wait_tol_rad", type=float, default=0.01)
    parser.add_argument("--hold_frames_after_stage", type=int, default=8)
    parser.add_argument("--reach_pos_tol_m", type=float, default=0.04)
    parser.add_argument("--reach_rot_tol_deg", type=float, default=180)
    parser.add_argument("--replan_until_reached_max_attempts", type=int, default=5, help="Max replan attempts per stage before giving up (default 5, 0=unbounded)")
    parser.add_argument("--reach_error_pose_source", type=str, default="ee")
    parser.add_argument("--debug_visualize_targets", type=int, default=1)
    parser.add_argument("--debug_visualize_selected_keyframe_axes", type=int, default=1)
    parser.add_argument("--debug_visualize_ik_waypoints", type=int, default=1)
    parser.add_argument("--debug_gripper_actor_forward_axis", type=str, default="local_z")
    parser.add_argument("--pure_scene_output", type=int, default=1)
    parser.add_argument("--enable_grasp_action_object_collision", type=int, default=0)
    parser.add_argument("--replay_objects_ignore_collision", type=int, default=1)
    parser.add_argument("--fail_on_execution_failure", type=int, default=1)
    # Wrist camera tuning (same convention as O.1 envs/camera/camera.py)
    parser.add_argument("--wrist_left_forward_offset_m", type=float, default=0.0)
    parser.add_argument("--wrist_right_forward_offset_m", type=float, default=0.0)
    parser.add_argument("--wrist_left_lateral_offset_m", type=float, default=0.0)
    parser.add_argument("--wrist_right_lateral_offset_m", type=float, default=0.0)
    parser.add_argument("--wrist_left_roll_deg", type=float, default=0.0)
    parser.add_argument("--wrist_right_roll_deg", type=float, default=0.0)
    parser.add_argument("--wrist_left_yaw_deg", type=float, default=0.0)
    parser.add_argument("--wrist_right_yaw_deg", type=float, default=0.0)
    parser.add_argument("--wrist_left_pitch_deg", type=float, default=0.0)
    parser.add_argument("--wrist_right_pitch_deg", type=float, default=0.0)
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

    print(f"[human-replay] task={args.task} video_id={args.video_id} output={args.output_dir}", flush=True)

    # Load data
    with open(args.hand_keyframes_json, "r") as f:
        hand_annotations = json.load(f)

    video_name = f"hand_vis_{args.video_id}.mp4"
    videos = hand_annotations.get("videos", {})
    if video_name not in videos:
        raise KeyError(f"{video_name} not found in {args.hand_keyframes_json}")
    video_info = videos[video_name]
    status = video_info.get("status", "")
    if status in ("reject", "discard", "bad"):
        print(f"[human-replay] SKIP video_id={args.video_id} status={status}")
        return

    mode = classify_keyframe_mode(video_info)
    effective_global, effective_by_arm = resolve_effective_keyframes(video_info)
    print(f"[human-replay] mode={mode} effective_by_arm={effective_by_arm}")

    if not any(effective_by_arm.values()):
        print(f"[human-replay] SKIP: no effective keyframes")
        return

    # Load hand and object NPZ
    hand_data = dict(np.load(str(args.hand_npz), allow_pickle=True))
    obj_npz_path = args.replay_dir / "multi_object_world_poses.npz"
    if not obj_npz_path.is_file():
        raise FileNotFoundError(f"Object NPZ not found: {obj_npz_path}")
    obj_data = dict(np.load(str(obj_npz_path), allow_pickle=True))

    # Build plan summary
    print(f"[human-replay] Computing world-space hand targets...", flush=True)
    plan_summary = build_plan_summary(
        task=args.task,
        video_id=args.video_id,
        effective_by_arm=effective_by_arm,
        hand_data=hand_data,
        obj_data=obj_data,
        mode=mode,
        cv_axis_mode=args.camera_cv_axis_mode,
        action_orientation_source=args.action_orientation_source,
        target_retreat_m=args.target_retreat_m,
    )

    plan_summary_path = args.output_dir / "plan_summary_human_replay.json"
    with open(plan_summary_path, "w") as f:
        json.dump(plan_summary, f, indent=2)
    print(f"[human-replay] Wrote plan summary: {plan_summary_path}", flush=True)

    # Run planner
    env = os.environ.copy()
    if bool(args.enable_viewer):
        env.pop("CUDA_VISIBLE_DEVICES", None)
    elif args.gpu >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    success = run_planner_with_targets(plan_summary_path, args, env)
    print(f"\n[human-replay] {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
