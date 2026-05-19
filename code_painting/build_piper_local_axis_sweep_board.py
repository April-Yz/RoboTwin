#!/usr/bin/env python3
"""Build a fixed-camera local-axis sweep board for Piper gripper orientation debug."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import render_hand_retarget_r1_npz as base
from render_hand_retarget_piper_dual_npz_urdfik import HandRetargetPiperDualURDFIKRenderer


DEFAULT_ROBOT_CONFIG = PROJECT_ROOT / "robot_config_PiperPika_agx_dual_table.json"
DEFAULT_HEAD_CAMERA_LOCAL_POS = [0.107882, -0.2693875, 0.464396]
DEFAULT_HEAD_CAMERA_LOCAL_QUAT_WXYZ = [0.85401166, 0.01255256, 0.51885652, -0.0359783]
DEFAULT_INIT_ARM_JOINTS = [0.0, 0.8, 1.2, 0.0, -0.4, 0.0]


def axis_semantic(renderer: HandRetargetPiperDualURDFIKRenderer, vec_world: Sequence[float]) -> Dict[str, float | str]:
    vec = np.asarray(vec_world, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return {"label": "zero", "score": 0.0}
    vec = vec / norm
    base_rot = base.R.from_quat(base.quat_wxyz_to_xyzw(renderer._base_pose.q)).as_matrix()
    basis = {
        "forward": base_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "back": base_rot @ np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "left": base_rot @ np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "right": base_rot @ np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "up": base_rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "down": base_rot @ np.array([0.0, 0.0, -1.0], dtype=np.float64),
    }
    label = "unknown"
    score = -1.0
    for name, axis in basis.items():
        dot = float(np.dot(vec, axis))
        if dot > score:
            label = name
            score = dot
    return {"label": label, "score": score}


def axis_line(name: str, vec_world: Sequence[float], semantic: Dict[str, float | str]) -> str:
    vec = np.asarray(vec_world, dtype=np.float64).reshape(3)
    return (
        f"{name}=({vec[0]:+.2f},{vec[1]:+.2f},{vec[2]:+.2f}) "
        f"{semantic['label']} {float(semantic['score']):.2f}"
    )


def build_contact_sheet(
    image_paths: Sequence[Path],
    output_path: Path,
    thumb_size: Tuple[int, int],
    columns: int,
) -> None:
    if not image_paths:
        return
    columns = max(int(columns), 1)
    thumb_w, thumb_h = int(thumb_size[0]), int(thumb_size[1])
    rows = (len(image_paths) + columns - 1) // columns
    canvas = np.zeros((rows * thumb_h, columns * thumb_w, 3), dtype=np.uint8)
    for idx, path in enumerate(image_paths):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        row = idx // columns
        col = idx % columns
        y0 = row * thumb_h
        x0 = col * thumb_w
        canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb
    cv2.imwrite(str(output_path), canvas)


def compose_contact_sheet(
    images: Sequence[np.ndarray],
    thumb_size: Tuple[int, int],
    columns: int,
    min_slots: int = 0,
) -> np.ndarray:
    columns = max(int(columns), 1)
    thumb_w, thumb_h = int(thumb_size[0]), int(thumb_size[1])
    slots = max(len(images), int(min_slots), 1)
    rows = (slots + columns - 1) // columns
    canvas = np.zeros((rows * thumb_h, columns * thumb_w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        row = idx // columns
        col = idx % columns
        y0 = row * thumb_h
        x0 = col * thumb_w
        canvas[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb
    return canvas


def signed_axis_vector(token: str) -> np.ndarray:
    token = str(token).strip().lower()
    if len(token) != 2 or token[0] not in "xyz" or token[1] not in "pm":
        raise ValueError(f"Unsupported signed axis token: {token}")
    axis = "xyz".index(token[0])
    sign = 1.0 if token[1] == "p" else -1.0
    vec = np.zeros(3, dtype=np.float64)
    vec[axis] = sign
    return vec


def remap_from_output_axes(label: str, x_from: np.ndarray, y_from: np.ndarray) -> Tuple[str, np.ndarray]:
    x_from = np.asarray(x_from, dtype=np.float64).reshape(3)
    y_from = np.asarray(y_from, dtype=np.float64).reshape(3)
    if abs(float(np.dot(x_from, y_from))) > 1e-6:
        raise ValueError(f"Output x/y source axes must be orthogonal: {label}")
    z_from = np.cross(x_from, y_from)
    mat = np.column_stack([x_from, y_from, z_from]).astype(np.float64)
    if np.linalg.det(mat) < 0.5:
        raise ValueError(f"Invalid remap determinant for {label}: {np.linalg.det(mat)}")
    return label, mat


def preferred_secondary_axis(primary: np.ndarray, preferred: Sequence[str]) -> np.ndarray:
    primary = np.asarray(primary, dtype=np.float64).reshape(3)
    for token in preferred:
        candidate = signed_axis_vector(token)
        if abs(float(np.dot(primary, candidate))) < 1e-6:
            return candidate
    raise ValueError("No orthogonal secondary axis found.")


def build_semantic_axis_candidates() -> List[Tuple[str, np.ndarray]]:
    candidates: List[Tuple[str, np.ndarray]] = []
    signed_axes = ["xp", "xm", "yp", "ym", "zp", "zm"]

    for token in signed_axes:
        x_from = signed_axis_vector(token)
        y_from = preferred_secondary_axis(x_from, ["yp", "zp", "xp", "ym", "zm", "xm"])
        label, mat = remap_from_output_axes(f"forward_from_{token}", x_from, y_from)
        candidates.append((label, mat))

    for token in signed_axes:
        y_from = signed_axis_vector(token)
        x_from = preferred_secondary_axis(y_from, ["zp", "xp", "yp", "zm", "xm", "ym"])
        label, mat = remap_from_output_axes(f"open_from_{token}", x_from, y_from)
        candidates.append((label, mat))

    return candidates


def build_axis_candidates(mode: str) -> List[Tuple[str, np.ndarray]]:
    normalized = str(mode).strip().lower()
    if normalized == "remap":
        return base.build_orientation_remap_candidates()
    if normalized == "semantic":
        return build_semantic_axis_candidates()
    if normalized == "both":
        candidates = base.build_orientation_remap_candidates() + build_semantic_axis_candidates()
        out: List[Tuple[str, np.ndarray]] = []
        seen = set()
        for label, mat in candidates:
            key = tuple(np.asarray(mat, dtype=np.float64).reshape(-1).tolist())
            if key in seen:
                continue
            seen.add(key)
            out.append((label, mat))
        return out
    raise ValueError(f"Unsupported candidate_mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Piper local-axis sweep board from one hand frame.")
    parser.add_argument("--input_npz", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=-1)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=1)
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--pose_source", choices=["gripper", "retreat"], default="gripper")
    parser.add_argument("--require_stored_gripper_pose", type=int, default=0)
    parser.add_argument("--retreat_distance", type=float, default=0.11)
    parser.add_argument("--thumb_tip_idx", type=int, default=4)
    parser.add_argument("--index_tip_idx", type=int, default=8)
    parser.add_argument("--index_joint_idx", type=int, default=6)
    parser.add_argument("--robot_config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--camera_cv_axis_mode", choices=sorted(base.CV_TO_WORLD_CAMERA_PRESETS.keys()), default="legacy_r1")
    parser.add_argument("--head_camera_local_pos", type=float, nargs=3, default=DEFAULT_HEAD_CAMERA_LOCAL_POS)
    parser.add_argument("--head_camera_local_quat_wxyz", type=float, nargs=4, default=DEFAULT_HEAD_CAMERA_LOCAL_QUAT_WXYZ)
    parser.add_argument("--wrist_camera_local_pos", type=float, nargs=3, default=base.DEFAULT_WRIST_CAMERA_LOCAL_POS.tolist())
    parser.add_argument("--wrist_camera_local_quat_wxyz", type=float, nargs=4, default=base.DEFAULT_WRIST_CAMERA_LOCAL_QUAT_WXYZ.tolist())
    parser.add_argument("--target_world_offset_xyz", type=float, nargs=3, default=[0.0, 0.1, 0.1])
    parser.add_argument("--left_target_world_offset_xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--right_target_world_offset_xyz", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--disable_table", type=int, default=0)
    parser.add_argument("--lighting_mode", choices=["default", "front", "front_no_shadow"], default="front_no_shadow")
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--fovy_deg", type=float, default=90.0)
    parser.add_argument("--execute", type=int, default=0)
    parser.add_argument("--execute_waypoint_scene_steps", type=int, default=5)
    parser.add_argument("--execute_settle_scene_steps", type=int, default=20)
    parser.add_argument("--urdfik_joint_interp_waypoints", type=int, default=10)
    parser.add_argument("--init_left_arm_joints", type=float, nargs=6, default=DEFAULT_INIT_ARM_JOINTS)
    parser.add_argument("--init_right_arm_joints", type=float, nargs=6, default=DEFAULT_INIT_ARM_JOINTS)
    parser.add_argument("--init_gripper_open", type=float, default=1.0)
    parser.add_argument("--third_person_view", type=int, default=1)
    parser.add_argument("--save_wrist_views", type=int, default=0)
    parser.add_argument("--candidate_mode", choices=["remap", "semantic", "both"], default="remap")
    parser.add_argument("--video_mode", type=int, default=0)
    parser.add_argument("--video_fps", type=float, default=5.0)
    parser.add_argument("--contact_sheet_columns", type=int, default=4)
    parser.add_argument("--contact_sheet_width", type=int, default=320)
    parser.add_argument("--contact_sheet_height", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectory = base.HandRetargetTrajectory(
        npz_path=args.input_npz,
        retreat_distance=args.retreat_distance,
        thumb_tip_idx=args.thumb_tip_idx,
        index_tip_idx=args.index_tip_idx,
        index_joint_idx=args.index_joint_idx,
    )
    frame_start = args.frame_idx if args.frame_start is None else int(args.frame_start)
    frame_start = max(frame_start, 0)
    frame_end = trajectory.length - 1 if int(args.frame_end) < 0 else min(int(args.frame_end), trajectory.length - 1)
    if frame_start > frame_end:
        raise ValueError(f"Invalid frame range: start={frame_start}, end={frame_end}, length={trajectory.length}")
    indices = list(range(frame_start, frame_end + 1, max(int(args.frame_stride), 1)))
    if int(args.max_frames) > 0:
        indices = indices[: int(args.max_frames)]
    if not indices:
        raise ValueError("No frames selected.")

    if bool(args.require_stored_gripper_pose) and not trajectory.has_stored_gripper_pose(args.arm, args.pose_source):
        missing = trajectory.missing_stored_gripper_keys(args.arm, args.pose_source)
        raise ValueError(
            "Missing stored gripper pose fields required by --require_stored_gripper_pose=1: "
            + ", ".join(missing)
        )

    renderer = HandRetargetPiperDualURDFIKRenderer(
        robot_config_path=args.robot_config,
        image_width=args.image_width,
        image_height=args.image_height,
        fovy_deg=args.fovy_deg,
        torso_qpos=base.DEFAULT_TORSO_QPOS,
        robot_base_pose_override=None,
        third_person_view=bool(args.third_person_view),
        need_topp=False,
        link_cam_debug_enable=False,
        link_cam_axis_mode="none",
        link_cam_debug_rot_xyz_deg=[0.0, 0.0, 0.0],
        link_cam_debug_shift_fru=[0.0, 0.0, 0.0],
        camera_cv_axis_mode=args.camera_cv_axis_mode,
        head_camera_local_pos=args.head_camera_local_pos,
        head_camera_local_quat_wxyz=args.head_camera_local_quat_wxyz,
        wrist_camera_local_pos=args.wrist_camera_local_pos,
        wrist_camera_local_quat_wxyz=args.wrist_camera_local_quat_wxyz,
        camera_debug_target="head",
        enable_viewer=False,
        viewer_frame_delay=0.0,
        viewer_wait_at_end=False,
        debug_mode=False,
        debug_force_orientation="none",
        debug_visualize_targets=True,
        debug_target_axis_length=0.08,
        debug_target_axis_thickness=0.004,
        debug_visualize_cameras=False,
        debug_camera_axis_length=0.16,
        debug_camera_axis_thickness=0.006,
        orientation_remap_label="identity",
        orientation_remap_matrix=np.eye(3, dtype=np.float64),
        stored_orientation_post_rot_xyz_deg=[0.0, 0.0, 0.0],
        target_local_forward_retreat_m=0.0,
        target_world_offset_xyz=args.target_world_offset_xyz,
        left_target_world_offset_xyz=args.left_target_world_offset_xyz,
        right_target_world_offset_xyz=args.right_target_world_offset_xyz,
        target_world_z_offset=0.0,
        disable_table=bool(args.disable_table),
        base_occluder_enable=False,
        base_occluder_local_pos=[0.0, 0.0, 0.4],
        base_occluder_half_size=[0.28, 0.32, 0.02],
        base_occluder_color=[1.0, 1.0, 1.0],
        camera_sweep_enable=False,
        camera_sweep_steps_deg=[-180.0, -90.0, 0.0, 90.0],
        init_left_arm_joints=args.init_left_arm_joints,
        init_right_arm_joints=args.init_right_arm_joints,
        init_gripper_open=args.init_gripper_open,
        execute_waypoint_scene_steps=args.execute_waypoint_scene_steps,
        execute_settle_scene_steps=args.execute_settle_scene_steps,
        urdfik_joint_interp_waypoints=args.urdfik_joint_interp_waypoints,
        lighting_mode=args.lighting_mode,
    )

    initial_state = renderer.snapshot_robot_state()
    per_case_dir = args.output_dir / "per_case"
    board_frame_dir = args.output_dir / "board_frames"
    per_case_dir.mkdir(parents=True, exist_ok=True)
    board_frame_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict] = []
    zed_paths: List[Path] = []
    third_paths: List[Path] = []
    left_wrist_paths: List[Path] = []
    right_wrist_paths: List[Path] = []

    candidates = build_axis_candidates(args.candidate_mode)
    thumb_size = (args.contact_sheet_width, args.contact_sheet_height)
    all_writer = None
    success_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    try:
        for frame_idx in indices:
            target_cam = trajectory.get_side_target(args.arm, frame_idx, args.pose_source)
            if not target_cam.valid:
                print(f"[local-axis-sweep] frame={frame_idx:04d} skipped: invalid target")
                continue
            target_cam_axes = {
                "x": target_cam.rotation_cam[:, 0].copy(),
                "y": target_cam.rotation_cam[:, 1].copy(),
                "z": target_cam.rotation_cam[:, 2].copy(),
            }
            frame_all_images: List[np.ndarray] = []
            frame_success_images: List[np.ndarray] = []

            for case_idx, (remap_label, remap_matrix) in enumerate(candidates):
                renderer.restore_robot_state(initial_state)
                renderer.orientation_remap_label = remap_label
                renderer.orientation_remap_matrix = remap_matrix.copy()
                target_world = base.build_world_target_pose(renderer, args.arm, target_cam, apply_forced_orientation=False)
                if target_world is None:
                    continue

                if args.arm == "left":
                    renderer.update_target_axis_visuals(target_world, None)
                else:
                    renderer.update_target_axis_visuals(None, target_world)
                renderer.update_robot_link_cameras()
                renderer.step_scene(steps=1)

                plan = renderer.plan_path(args.arm, target_world)
                status = renderer._plan_status(plan)

                row: Dict[str, object] = {
                    "frame_idx": int(frame_idx),
                    "case_idx": int(case_idx),
                    "remap_label": remap_label,
                    "candidate_mode": args.candidate_mode,
                    "plan_status": status,
                    "input_position_cam": np.asarray(target_cam.position_cam, dtype=np.float64).round(6).tolist(),
                    "input_rotation_cam_axes": {k: np.asarray(v, dtype=np.float64).round(6).tolist() for k, v in target_cam_axes.items()},
                    "target_world_pose": np.asarray(target_world, dtype=np.float64).round(6).tolist(),
                }

                if bool(args.execute) and status == "Success":
                    if args.arm == "left":
                        exec_status, _ = renderer.execute_plans(plan, None)
                    else:
                        _, exec_status = renderer.execute_plans(None, plan)
                    actual_tcp = renderer.get_current_tcp_pose(args.arm)
                    row["executed_status"] = exec_status
                    row["actual_tcp_pose_world"] = np.asarray(actual_tcp, dtype=np.float64).round(6).tolist()
                    row["pos_err_after_execute_m"] = float(np.linalg.norm(actual_tcp[:3] - target_world[:3]))
                    row["rot_err_after_execute_deg"] = float(base.quat_angle_deg_wxyz(actual_tcp[3:], target_world[3:]))
                else:
                    row["executed_status"] = "Skipped"

                axes_world = base.rotation_axes_world_wxyz(target_world[3:])
                axis_desc: Dict[str, Dict[str, float | str | List[float]]] = {}
                for axis_name, vec_world in axes_world.items():
                    semantic = axis_semantic(renderer, vec_world)
                    axis_desc[axis_name] = {
                        "vector_world": np.asarray(vec_world, dtype=np.float64).round(6).tolist(),
                        "semantic": str(semantic["label"]),
                        "semantic_score": float(semantic["score"]),
                    }
                row["target_world_axes"] = axis_desc

                overlay_lines = [
                    f"frame={frame_idx} arm={args.arm}",
                    f"case={case_idx:02d} {remap_label}",
                    f"plan={status} exec={row['executed_status']}",
                    axis_line("x", axes_world["x"], axis_semantic(renderer, axes_world["x"])),
                    axis_line("y", axes_world["y"], axis_semantic(renderer, axes_world["y"])),
                    axis_line("z", axes_world["z"], axis_semantic(renderer, axes_world["z"])),
                ]
                if "pos_err_after_execute_m" in row:
                    overlay_lines.append(f"exec_pos_err={float(row['pos_err_after_execute_m']):.3f}m")
                    overlay_lines.append(f"exec_rot_err={float(row['rot_err_after_execute_deg']):.1f}deg")

                safe_label = remap_label.replace("/", "_")
                success_basis = str(row["executed_status"] if args.execute else status)
                is_success = success_basis == "Success"
                status_suffix = base.short_status(success_basis)
                zed_rgb, _ = renderer.capture_camera(renderer.zed_camera)
                zed_bgr = base.overlay_text(zed_rgb, overlay_lines)
                zed_path = per_case_dir / f"frame_{frame_idx:04d}_{case_idx:02d}_{safe_label}_{status_suffix}_zed.png"
                cv2.imwrite(str(zed_path), zed_bgr)
                zed_paths.append(zed_path)
                frame_all_images.append(zed_bgr)
                if is_success:
                    frame_success_images.append(zed_bgr)

                if renderer.third_person_view:
                    third_rgb, _ = renderer.capture_camera(renderer.third_camera)
                    third_bgr = base.overlay_text(third_rgb, overlay_lines)
                    third_path = per_case_dir / f"frame_{frame_idx:04d}_{case_idx:02d}_{safe_label}_{status_suffix}_third.png"
                    cv2.imwrite(str(third_path), third_bgr)
                    third_paths.append(third_path)

                if bool(args.save_wrist_views):
                    if renderer._left_wrist_camera_link is not None:
                        left_wrist_rgb, _ = renderer.capture_camera(renderer.left_wrist_camera)
                        left_bgr = base.overlay_text(left_wrist_rgb, overlay_lines)
                        left_path = per_case_dir / f"frame_{frame_idx:04d}_{case_idx:02d}_{safe_label}_{status_suffix}_left_wrist.png"
                        cv2.imwrite(str(left_path), left_bgr)
                        left_wrist_paths.append(left_path)
                    if renderer._right_wrist_camera_link is not None:
                        right_wrist_rgb, _ = renderer.capture_camera(renderer.right_wrist_camera)
                        right_bgr = base.overlay_text(right_wrist_rgb, overlay_lines)
                        right_path = per_case_dir / f"frame_{frame_idx:04d}_{case_idx:02d}_{safe_label}_{status_suffix}_right_wrist.png"
                        cv2.imwrite(str(right_path), right_bgr)
                        right_wrist_paths.append(right_path)

                summary_rows.append(row)
                print(
                    f"[local-axis-sweep] frame={frame_idx:04d} case={case_idx:02d} remap={remap_label} "
                    f"plan={status} exec={row['executed_status']}"
                )

            all_board = compose_contact_sheet(
                frame_all_images,
                thumb_size=thumb_size,
                columns=args.contact_sheet_columns,
                min_slots=len(candidates),
            )
            all_board_path = board_frame_dir / f"frame_{frame_idx:04d}_all_zed.png"
            cv2.imwrite(str(all_board_path), all_board)
            if bool(args.video_mode):
                if all_writer is None:
                    h, w = all_board.shape[:2]
                    all_writer = cv2.VideoWriter(str(args.output_dir / "board_all_zed.mp4"), fourcc, args.video_fps, (w, h))
                all_writer.write(all_board)

            success_board = compose_contact_sheet(
                frame_success_images,
                thumb_size=thumb_size,
                columns=args.contact_sheet_columns,
                min_slots=len(candidates),
            )
            success_board_path = board_frame_dir / f"frame_{frame_idx:04d}_success_zed.png"
            cv2.imwrite(str(success_board_path), success_board)
            if bool(args.video_mode):
                if success_writer is None:
                    h, w = success_board.shape[:2]
                    success_writer = cv2.VideoWriter(str(args.output_dir / "board_success_zed.mp4"), fourcc, args.video_fps, (w, h))
                success_writer.write(success_board)
    finally:
        if all_writer is not None:
            all_writer.release()
        if success_writer is not None:
            success_writer.release()

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(base.make_json_safe(summary_rows), f, ensure_ascii=False, indent=2)

    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_idx", "remap_label", "plan_status", "executed_status", "x_semantic", "y_semantic", "z_semantic"])
        for row in summary_rows:
            axes = row["target_world_axes"]
            writer.writerow([
                row["case_idx"],
                row["remap_label"],
                row["plan_status"],
                row["executed_status"],
                axes["x"]["semantic"],
                axes["y"]["semantic"],
                axes["z"]["semantic"],
            ])

    build_contact_sheet(
        zed_paths,
        args.output_dir / "board_zed.png",
        thumb_size=(args.contact_sheet_width, args.contact_sheet_height),
        columns=args.contact_sheet_columns,
    )
    if third_paths:
        build_contact_sheet(
            third_paths,
            args.output_dir / "board_third.png",
            thumb_size=(args.contact_sheet_width, args.contact_sheet_height),
            columns=args.contact_sheet_columns,
        )
    if left_wrist_paths:
        build_contact_sheet(
            left_wrist_paths,
            args.output_dir / "board_left_wrist.png",
            thumb_size=(args.contact_sheet_width, args.contact_sheet_height),
            columns=args.contact_sheet_columns,
        )
    if right_wrist_paths:
        build_contact_sheet(
            right_wrist_paths,
            args.output_dir / "board_right_wrist.png",
            thumb_size=(args.contact_sheet_width, args.contact_sheet_height),
            columns=args.contact_sheet_columns,
        )

    print(f"[local-axis-sweep] summary_json={summary_path}")
    print(f"[local-axis-sweep] summary_csv={csv_path}")
    print(f"[local-axis-sweep] board_zed={args.output_dir / 'board_zed.png'}")


if __name__ == "__main__":
    main()
