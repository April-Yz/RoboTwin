#!/usr/bin/env python3
"""Batch planner for AnyGrasp-driven two-keyframe RoboTwin demos."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


THIS_DIR = Path(__file__).resolve().parent
SINGLE_SCRIPT = THIS_DIR / "plan_anygrasp_keyframes_r1.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch plan RoboTwin demos from AnyGrasp replay outputs.")
    parser.add_argument("--anygrasp_root", type=Path, required=True, help="Root containing per-video AnyGrasp dirs like d_pour_blue_1.")
    parser.add_argument("--replay_root", type=Path, required=True, help="Root containing per-video replay dirs like d_pour_blue_1.")
    parser.add_argument("--hand_dir", type=Path, required=True, help="Directory containing hand_detections_<id>.npz files.")
    parser.add_argument("--output_root", type=Path, required=True, help="Root for per-video planned demo outputs.")
    parser.add_argument("--reuse_preview_summary_root", type=Path, default=None, help="Optional root containing per-video preview summary.json files, e.g. anygrasp_direct_preview/d_pour_blue_batch_fi60.")
    parser.add_argument("--reuse_preview_frame_mode", choices=["legacy_1_max22rel", "annotated_json_keyframes"], default="legacy_1_max22rel")
    parser.add_argument("--reuse_preview_candidate_group", choices=["orientation", "fused"], default="orientation")
    parser.add_argument("--reuse_preview_top_rank", type=int, default=1)
    parser.add_argument("--ids", type=str, nargs="*", default=None, help="Optional subset ids like 1 4 22.")
    parser.add_argument("--keyframes", type=int, nargs=2, default=[1, 22], metavar=("GRASP_FRAME", "ACTION_FRAME"))
    parser.add_argument("--arm", choices=["auto", "left", "right"], default="auto")
    parser.add_argument("--execute_both_arms", type=int, default=1)
    parser.add_argument("--planner_backend", choices=["urdfik", "curobo"], default="urdfik")
    parser.add_argument("--urdfik_trajectory_mode", choices=["joint_interp", "cartesian_interp_ik"], default="joint_interp")
    parser.add_argument("--urdfik_cartesian_interp_steps", type=int, default=8)
    parser.add_argument("--urdfik_cartesian_interp_auto_step_m", type=float, default=0.05)
    parser.add_argument("--candidate_selection_mode", choices=["planner", "top_score_auto"], default="planner")
    parser.add_argument("--candidate_selection_relative_frame", type=int, default=None)
    parser.add_argument("--candidate_max_rotation_distance_deg", type=float, default=-1.0)
    parser.add_argument("--left_target_object", type=str, default="cup")
    parser.add_argument("--right_target_object", type=str, default="bottle")
    parser.add_argument("--candidate_object_max_distance_m", type=float, default=0.12)
    parser.add_argument("--enforce_target_object_constraint", type=int, default=1)
    parser.add_argument("--enforce_candidate_distance_constraint", type=int, default=1)
    parser.add_argument("--debug_candidate_top_k", type=int, default=5)
    parser.add_argument("--debug_show_all_candidates", type=int, default=1)
    parser.add_argument("--debug_common_candidate_top_k", type=int, default=0)
    parser.add_argument("--candidate_orientation_remap_label", type=str, default="identity")
    parser.add_argument("--candidate_post_rot_xyz_deg", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--candidate_keep_camera_up", type=int, default=0)
    parser.add_argument("--candidate_camera_top_axis", choices=["y", "z"], default="z")
    parser.add_argument("--candidate_target_local_x_offset_m", type=float, default=0.0, help="Shift each AnyGrasp planning target along gripper local +X before planning/visualization. Use this to compensate wrist/endlink vs fingertip-TCP mismatch; e.g. -0.12 retreats the planner target backward along local +X.")
    parser.add_argument("--manual_candidate", type=str, nargs=3, action="append", default=[])
    parser.add_argument("--object_mesh_override", action="append", default=[])
    parser.add_argument("--skip_existing", type=int, default=1)
    parser.add_argument("--continue_on_error", type=int, default=1)
    parser.add_argument("--robot_config", type=Path, default=(THIS_DIR.parent / "robot_config_R1.json"))
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--fovy_deg", type=float, default=90.0)
    parser.add_argument("--torso_qpos", type=float, nargs=4, default=[0.25, -0.4, -0.85, 0.0])
    parser.add_argument("--robot_base_pose", type=float, nargs=7, default=None, metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"))
    parser.add_argument("--open_gripper", type=float, default=1.0)
    parser.add_argument("--close_gripper", type=float, default=0.0)
    parser.add_argument("--approach_offset_m", type=float, default=0.08, help="Pregrasp-only retreat distance. The final grasp target is unchanged; a separate pregrasp pose is generated behind it along the local forward axis.")
    parser.add_argument("--settle_steps", type=int, default=4)
    parser.add_argument("--execute_interp_steps", type=int, default=24)
    parser.add_argument("--joint_command_scene_steps", type=int, default=2)
    parser.add_argument("--joint_target_wait_steps", type=int, default=60)
    parser.add_argument("--joint_target_wait_tol_rad", type=float, default=0.01)
    parser.add_argument("--reach_pos_tol_m", type=float, default=0.03)
    parser.add_argument("--reach_rot_tol_deg", type=float, default=20.0)
    parser.add_argument("--reach_error_pose_source", choices=["tcp", "ee"], default="tcp")
    parser.add_argument("--max_stage_replans", type=int, default=3)
    parser.add_argument("--replan_until_reached", type=int, default=1)
    parser.add_argument("--replan_until_reached_max_attempts", type=int, default=0)
    parser.add_argument("--hold_frames_after_stage", type=int, default=2)
    parser.add_argument("--init_prefix_frames", type=int, default=0)
    parser.add_argument("--pause_after_keyframe1_seconds", type=float, default=0.0)
    parser.add_argument("--replay_objects_during_action", type=int, default=0)
    parser.add_argument("--replay_objects_ignore_collision", type=int, default=1)
    parser.add_argument("--enable_grasp_action_object_collision", type=int, default=0)
    parser.add_argument("--grasp_action_object_collision_start_stage", choices=["close_gripper", "grasp", "pregrasp"], default="close_gripper")
    parser.add_argument("--execution_object_collision_mode", choices=["convex", "solid_bbox"], default="convex")
    parser.add_argument("--execution_object_scale_override", action="append", default=[])
    parser.add_argument("--execution_object_visual_scale_override", action="append", default=[])
    parser.add_argument("--execution_object_collision_scale_override", action="append", default=[])
    parser.add_argument("--debug_collision_report", type=int, default=0)
    parser.add_argument("--debug_visualize_object_collision_bbox", type=int, default=0)
    parser.add_argument("--gripper_contact_monitor_mode", choices=["fingers", "fingers_and_base", "all_robot_links"], default="fingers")
    parser.add_argument("--save_debug_preview", type=int, default=1)
    parser.add_argument("--debug_preview_fps", type=int, default=10)
    parser.add_argument("--debug_keyframe_hold_frames", type=int, default=12)
    parser.add_argument("--save_debug_execution_preview", type=int, default=1)
    parser.add_argument("--debug_execution_fps", type=int, default=10)
    parser.add_argument("--save_pose_debug", type=int, default=0)
    parser.add_argument("--pure_scene_output", type=int, default=0)
    parser.add_argument("--debug_visualize_targets", type=int, default=1)
    parser.add_argument("--debug_visualize_ik_waypoints", type=int, default=0)
    parser.add_argument("--debug_visualize_cameras", type=int, default=0)
    parser.add_argument("--debug_camera_axis_length", type=float, default=0.16)
    parser.add_argument("--debug_camera_axis_thickness", type=float, default=0.006)
    parser.add_argument("--target_local_forward_retreat_m", type=float, default=0.0)
    parser.add_argument("--save_rank_preview_images", type=int, default=1)
    parser.add_argument("--rank_preview_top_n", type=int, default=3)
    parser.add_argument("--debug_target_axis_length", type=float, default=0.08)
    parser.add_argument("--debug_target_axis_thickness", type=float, default=0.004)
    parser.add_argument("--head_only", type=int, default=1)
    parser.add_argument("--overlay_text", type=int, default=1)
    parser.add_argument("--third_person_view", type=int, default=0)
    parser.add_argument("--enable_viewer", type=int, default=0)
    parser.add_argument("--viewer_show_camera_frustums", type=int, default=0)
    parser.add_argument("--viewer_frame_delay", type=float, default=0.0)
    parser.add_argument("--viewer_wait_at_end", type=int, default=0)
    parser.add_argument("--disable_table", type=int, default=1)
    parser.add_argument("--base_occluder_enable", type=int, default=0)
    parser.add_argument("--base_occluder_local_pos", type=float, nargs=3, default=[0.0, 0.0, 0.4])
    parser.add_argument("--base_occluder_half_size", type=float, nargs=3, default=[0.28, 0.32, 0.02])
    parser.add_argument("--base_occluder_color", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--lighting_mode", choices=["default", "front", "front_no_shadow"], default="front_no_shadow")
    parser.add_argument("--camera_cv_axis_mode", type=str, default="legacy_r1")
    parser.add_argument("--head_camera_local_pos", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--head_camera_local_quat_wxyz", type=float, nargs=4, default=[1.0, 1.0, -1.0, 1.0])
    return parser.parse_args()


def trailing_id(name: str) -> Optional[int]:
    match = re.search(r"(\d+)$", name)
    return None if match is None else int(match.group(1))


def discover_anygrasp_dirs(anygrasp_root: Path) -> List[Path]:
    root = anygrasp_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"anygrasp_root is not a directory: {root}")
    dirs = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if (child / "grasps").is_dir():
            dirs.append(child)
    return sorted(
        dirs,
        key=lambda p: (
            trailing_id(p.name) is None,
            trailing_id(p.name) if trailing_id(p.name) is not None else sys.maxsize,
            p.name,
        ),
    )


def filter_dirs(video_dirs: List[Path], ids: Optional[List[str]]) -> List[Path]:
    if not ids:
        return video_dirs
    wanted = {str(item) for item in ids}
    return [video_dir for video_dir in video_dirs if video_dir.name in wanted or str(trailing_id(video_dir.name)) in wanted]


def build_single_command(args: argparse.Namespace, anygrasp_dir: Path, replay_dir: Path, hand_npz: Path, output_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(SINGLE_SCRIPT),
        "--anygrasp_dir",
        str(anygrasp_dir.resolve()),
        "--replay_dir",
        str(replay_dir.resolve()),
        "--hand_npz",
        str(hand_npz.resolve()),
        "--output_dir",
        str(output_dir.resolve()),
        *(["--reuse_preview_summary_json", str((args.reuse_preview_summary_root / anygrasp_dir.name / "summary.json").resolve())] if args.reuse_preview_summary_root is not None else []),
        "--reuse_preview_frame_mode",
        str(args.reuse_preview_frame_mode),
        "--reuse_preview_candidate_group",
        str(args.reuse_preview_candidate_group),
        "--reuse_preview_top_rank",
        str(args.reuse_preview_top_rank),
        "--keyframes",
        *(str(v) for v in args.keyframes),
        "--arm",
        str(args.arm),
        "--execute_both_arms",
        str(args.execute_both_arms),
        "--planner_backend",
        str(args.planner_backend),
        "--urdfik_trajectory_mode",
        str(args.urdfik_trajectory_mode),
        "--urdfik_cartesian_interp_steps",
        str(args.urdfik_cartesian_interp_steps),
        "--urdfik_cartesian_interp_auto_step_m",
        str(args.urdfik_cartesian_interp_auto_step_m),
        "--candidate_selection_mode",
        str(args.candidate_selection_mode),
        "--left_target_object",
        str(args.left_target_object),
        "--right_target_object",
        str(args.right_target_object),
        "--candidate_max_rotation_distance_deg",
        str(args.candidate_max_rotation_distance_deg),
        "--candidate_object_max_distance_m",
        str(args.candidate_object_max_distance_m),
        "--enforce_target_object_constraint",
        str(args.enforce_target_object_constraint),
        "--enforce_candidate_distance_constraint",
        str(args.enforce_candidate_distance_constraint),
        "--debug_candidate_top_k",
        str(args.debug_candidate_top_k),
        "--debug_show_all_candidates",
        str(args.debug_show_all_candidates),
        "--debug_common_candidate_top_k",
        str(args.debug_common_candidate_top_k),
        "--candidate_orientation_remap_label",
        str(args.candidate_orientation_remap_label),
        "--candidate_post_rot_xyz_deg",
        *(str(v) for v in args.candidate_post_rot_xyz_deg),
        "--candidate_keep_camera_up",
        str(args.candidate_keep_camera_up),
        "--candidate_camera_top_axis",
        str(args.candidate_camera_top_axis),
        "--candidate_target_local_x_offset_m",
        str(args.candidate_target_local_x_offset_m),
        *(["--candidate_selection_relative_frame", str(args.candidate_selection_relative_frame)] if args.candidate_selection_relative_frame is not None else []),
        *(sum((["--manual_candidate", str(spec[0]), str(spec[1]), str(spec[2])] for spec in args.manual_candidate), [])),
        *(sum((["--object_mesh_override", str(spec)] for spec in args.object_mesh_override), [])),
        "--robot_config",
        str(args.robot_config.resolve()),
        "--image_width",
        str(args.image_width),
        "--image_height",
        str(args.image_height),
        "--fps",
        str(args.fps),
        "--fovy_deg",
        str(args.fovy_deg),
        "--torso_qpos",
        *(str(v) for v in args.torso_qpos),
        "--open_gripper",
        str(args.open_gripper),
        "--close_gripper",
        str(args.close_gripper),
        "--approach_offset_m",
        str(args.approach_offset_m),
        "--settle_steps",
        str(args.settle_steps),
        "--execute_interp_steps",
        str(args.execute_interp_steps),
        "--joint_command_scene_steps",
        str(args.joint_command_scene_steps),
        "--joint_target_wait_steps",
        str(args.joint_target_wait_steps),
        "--joint_target_wait_tol_rad",
        str(args.joint_target_wait_tol_rad),
        "--reach_pos_tol_m",
        str(args.reach_pos_tol_m),
        "--reach_rot_tol_deg",
        str(args.reach_rot_tol_deg),
        "--reach_error_pose_source",
        str(args.reach_error_pose_source),
        "--max_stage_replans",
        str(args.max_stage_replans),
        "--replan_until_reached",
        str(args.replan_until_reached),
        "--replan_until_reached_max_attempts",
        str(args.replan_until_reached_max_attempts),
        "--hold_frames_after_stage",
        str(args.hold_frames_after_stage),
        "--init_prefix_frames",
        str(args.init_prefix_frames),
        "--pause_after_keyframe1_seconds",
        str(args.pause_after_keyframe1_seconds),
        "--replay_objects_during_action",
        str(args.replay_objects_during_action),
        "--replay_objects_ignore_collision",
        str(args.replay_objects_ignore_collision),
        "--enable_grasp_action_object_collision",
        str(args.enable_grasp_action_object_collision),
        "--grasp_action_object_collision_start_stage",
        str(args.grasp_action_object_collision_start_stage),
        "--execution_object_collision_mode",
        str(args.execution_object_collision_mode),
        "--debug_collision_report",
        str(args.debug_collision_report),
        "--debug_visualize_object_collision_bbox",
        str(args.debug_visualize_object_collision_bbox),
        "--gripper_contact_monitor_mode",
        str(args.gripper_contact_monitor_mode),
        "--save_debug_preview",
        str(args.save_debug_preview),
        "--debug_preview_fps",
        str(args.debug_preview_fps),
        "--debug_keyframe_hold_frames",
        str(args.debug_keyframe_hold_frames),
        "--save_debug_execution_preview",
        str(args.save_debug_execution_preview),
        "--debug_execution_fps",
        str(args.debug_execution_fps),
        "--save_pose_debug",
        str(args.save_pose_debug),
        "--pure_scene_output",
        str(args.pure_scene_output),
        "--debug_visualize_targets",
        str(args.debug_visualize_targets),
        "--debug_visualize_ik_waypoints",
        str(args.debug_visualize_ik_waypoints),
        "--debug_visualize_cameras",
        str(args.debug_visualize_cameras),
        "--debug_camera_axis_length",
        str(args.debug_camera_axis_length),
        "--debug_camera_axis_thickness",
        str(args.debug_camera_axis_thickness),
        "--target_local_forward_retreat_m",
        str(args.target_local_forward_retreat_m),
        "--save_rank_preview_images",
        str(args.save_rank_preview_images),
        "--rank_preview_top_n",
        str(args.rank_preview_top_n),
        "--debug_target_axis_length",
        str(args.debug_target_axis_length),
        "--debug_target_axis_thickness",
        str(args.debug_target_axis_thickness),
        "--head_only",
        str(args.head_only),
        "--overlay_text",
        str(args.overlay_text),
        "--third_person_view",
        str(args.third_person_view),
        "--enable_viewer",
        str(args.enable_viewer),
        "--viewer_show_camera_frustums",
        str(args.viewer_show_camera_frustums),
        "--viewer_frame_delay",
        str(args.viewer_frame_delay),
        "--viewer_wait_at_end",
        str(args.viewer_wait_at_end),
        "--disable_table",
        str(args.disable_table),
        "--base_occluder_enable",
        str(args.base_occluder_enable),
        "--base_occluder_local_pos",
        *(str(v) for v in args.base_occluder_local_pos),
        "--base_occluder_half_size",
        *(str(v) for v in args.base_occluder_half_size),
        "--base_occluder_color",
        *(str(v) for v in args.base_occluder_color),
        "--lighting_mode",
        str(args.lighting_mode),
        "--camera_cv_axis_mode",
        str(args.camera_cv_axis_mode),
        "--head_camera_local_pos",
        *(str(v) for v in args.head_camera_local_pos),
        "--head_camera_local_quat_wxyz",
        *(str(v) for v in args.head_camera_local_quat_wxyz),
    ]
    if args.robot_base_pose is not None:
        cmd.extend(["--robot_base_pose", *(str(v) for v in args.robot_base_pose)])
    for spec in args.execution_object_scale_override:
        cmd.extend(["--execution_object_scale_override", str(spec)])
    for spec in args.execution_object_visual_scale_override:
        cmd.extend(["--execution_object_visual_scale_override", str(spec)])
    for spec in args.execution_object_collision_scale_override:
        cmd.extend(["--execution_object_collision_scale_override", str(spec)])
    return cmd


def load_plan_summary(summary_path: Path) -> Optional[dict]:
    if not summary_path.is_file():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    args.anygrasp_root = args.anygrasp_root.resolve()
    args.replay_root = args.replay_root.resolve()
    args.hand_dir = args.hand_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.robot_config = args.robot_config.resolve()
    args.reuse_preview_summary_root = None if args.reuse_preview_summary_root is None else args.reuse_preview_summary_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if not args.hand_dir.is_dir():
        raise NotADirectoryError(f"hand_dir not found: {args.hand_dir}")
    if not args.robot_config.is_file():
        raise FileNotFoundError(f"robot_config not found: {args.robot_config}")
    if args.reuse_preview_summary_root is not None and not args.reuse_preview_summary_root.is_dir():
        raise NotADirectoryError(f"reuse_preview_summary_root not found: {args.reuse_preview_summary_root}")

    all_dirs = discover_anygrasp_dirs(args.anygrasp_root)
    selected_dirs = filter_dirs(all_dirs, args.ids)
    if not selected_dirs:
        raise RuntimeError(f"No AnyGrasp directories matched under {args.anygrasp_root}")

    logging.info("Found %d AnyGrasp dirs, selected %d", len(all_dirs), len(selected_dirs))
    failures = []
    successes = []
    for anygrasp_dir in selected_dirs:
        video_name = anygrasp_dir.name
        video_id = trailing_id(video_name)
        if video_id is None:
            failures.append({"video_name": video_name, "video_id": None, "reason": "invalid_video_id"})
            logging.error("Skipping %s because trailing id is missing", video_name)
            if not bool(args.continue_on_error):
                break
            continue

        replay_dir = args.replay_root / video_name
        hand_npz = args.hand_dir / f"hand_detections_{video_id}.npz"
        output_dir = args.output_root / video_name
        summary_path = output_dir / "plan_summary.json"
        if bool(args.skip_existing) and summary_path.is_file():
            logging.info("Skipping %s because %s exists", video_name, summary_path)
            continue

        if not replay_dir.is_dir():
            failures.append({"video_name": video_name, "video_id": video_id, "reason": f"missing_replay_dir:{replay_dir}"})
            logging.error("Missing replay dir for %s: %s", video_name, replay_dir)
            if not bool(args.continue_on_error):
                break
            continue
        if not hand_npz.is_file():
            failures.append({"video_name": video_name, "video_id": video_id, "reason": f"missing_hand_npz:{hand_npz}"})
            logging.error("Missing hand npz for %s: %s", video_name, hand_npz)
            if not bool(args.continue_on_error):
                break
            continue
        if args.reuse_preview_summary_root is not None:
            preview_summary_path = args.reuse_preview_summary_root / video_name / "summary.json"
            if not preview_summary_path.is_file():
                failures.append({"video_name": video_name, "video_id": video_id, "reason": f"missing_preview_summary:{preview_summary_path}"})
                logging.error("Missing preview summary for %s: %s", video_name, preview_summary_path)
                if not bool(args.continue_on_error):
                    break
                continue

        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_single_command(args, anygrasp_dir, replay_dir, hand_npz, output_dir)
        logging.info("Running %s", video_name)
        logging.info("Command: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            plan_summary = load_plan_summary(summary_path)
            if isinstance(plan_summary, dict) and bool(plan_summary.get("execution_failed", False)):
                failures.append(
                    {
                        "video_name": video_name,
                        "video_id": video_id,
                        "reason": "execution_failed_in_summary",
                        "summary_path": str(summary_path),
                    }
                )
                logging.warning("Completed %s with stage failures recorded in summary", video_name)
            else:
                successes.append(video_name)
        except subprocess.CalledProcessError as exc:
            plan_summary = load_plan_summary(summary_path)
            reason = f"exit_code:{exc.returncode}"
            if isinstance(plan_summary, dict) and bool(plan_summary.get("execution_failed", False)):
                reason = "execution_failed"
            failures.append(
                {
                    "video_name": video_name,
                    "video_id": video_id,
                    "reason": reason,
                    "exit_code": int(exc.returncode),
                    "summary_path": str(summary_path) if summary_path.exists() else None,
                }
            )
            logging.error("Failed %s (exit_code=%s)", video_name, exc.returncode)
            if not bool(args.continue_on_error):
                break

    failed_ids = [item["video_id"] for item in failures if isinstance(item, dict) and item.get("video_id") is not None]
    batch_summary = {
        "anygrasp_root": str(args.anygrasp_root),
        "replay_root": str(args.replay_root),
        "hand_dir": str(args.hand_dir),
        "output_root": str(args.output_root),
        "reuse_preview_summary_root": None if args.reuse_preview_summary_root is None else str(args.reuse_preview_summary_root),
        "reuse_preview_frame_mode": str(args.reuse_preview_frame_mode),
        "reuse_preview_candidate_group": str(args.reuse_preview_candidate_group),
        "reuse_preview_top_rank": int(args.reuse_preview_top_rank),
        "keyframes": [int(v) for v in args.keyframes],
        "candidate_selection_mode": str(args.candidate_selection_mode),
        "candidate_selection_relative_frame": None if args.candidate_selection_relative_frame is None else int(args.candidate_selection_relative_frame),
        "candidate_max_rotation_distance_deg": float(args.candidate_max_rotation_distance_deg),
        "pure_scene_output": int(args.pure_scene_output),
        "selected_ids": list(args.ids) if args.ids else None,
        "successes": successes,
        "failures": failures,
        "failed_ids": failed_ids,
    }
    summary_path = args.output_root / "batch_plan_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2)
    failed_ids_path = args.output_root / "batch_failed_ids.json"
    with failed_ids_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_root": str(args.output_root),
                "failed_ids": failed_ids,
                "failures": failures,
            },
            f,
            indent=2,
        )

    if failures:
        logging.warning("Batch planning completed with failed_ids=%s failed_json=%s", failed_ids, failed_ids_path)
        return
    logging.info("Batch planning completed successfully.")


if __name__ == "__main__":
    main()
