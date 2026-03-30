#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

import plan_anygrasp_keyframes_r1 as planner


DEFAULT_ROBOT_CONFIG = planner.R1_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a smoothed planner bundle (head/wrist videos + pose_debug.jsonl) from raw planner pose_debug records."
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Raw planner output dir containing pose_debug.jsonl and plan_summary.json")
    parser.add_argument("--output_dir", type=Path, required=True, help="Smoothed planner bundle output dir")
    parser.add_argument("--pose_debug_name", type=str, default="pose_debug.jsonl")
    parser.add_argument("--plan_summary_name", type=str, default="plan_summary.json")
    parser.add_argument("--output_pose_debug_name", type=str, default="pose_debug.jsonl")
    parser.add_argument("--output_head_name", type=str, default="head_cam_plan.mp4")
    parser.add_argument("--output_left_wrist_name", type=str, default="left_wrist_cam_plan.mp4")
    parser.add_argument("--output_right_wrist_name", type=str, default="right_wrist_cam_plan.mp4")
    parser.add_argument("--output_summary_name", type=str, default="smooth_summary.json")
    parser.add_argument("--interp_factor", type=int, default=2, help="Number of subframes per kept interval. 1 means no densification.")
    parser.add_argument("--fps", type=int, default=10, help="Output FPS for smoothed videos.")
    parser.add_argument("--keep_hover_frames_every", type=int, default=1, help="When near-duplicate frames repeat, keep every N-th duplicate inside the run. 1 keeps all duplicates before endpoint pruning; larger values thin hover frames.")
    parser.add_argument("--dedup_pos_thresh_m", type=float, default=0.002)
    parser.add_argument("--dedup_rot_thresh_deg", type=float, default=1.5)
    parser.add_argument("--dedup_joint_thresh_rad", type=float, default=0.01)
    parser.add_argument("--dedup_gripper_thresh", type=float, default=0.01)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--fovy_deg", type=float, default=90.0)
    parser.add_argument("--robot_config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--torso_qpos", type=float, nargs=4, default=planner.base.DEFAULT_TORSO_QPOS.tolist())
    parser.add_argument("--robot_base_pose", type=float, nargs=7, default=None)
    parser.add_argument("--disable_table", type=int, default=1)
    parser.add_argument("--base_occluder_enable", type=int, default=0)
    parser.add_argument("--base_occluder_local_pos", type=float, nargs=3, default=[0.0, 0.0, 0.4])
    parser.add_argument("--base_occluder_half_size", type=float, nargs=3, default=[0.28, 0.32, 0.02])
    parser.add_argument("--base_occluder_color", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--lighting_mode", choices=["default", "front", "front_no_shadow"], default="front_no_shadow")
    parser.add_argument("--camera_cv_axis_mode", choices=sorted(planner.base.CV_TO_WORLD_CAMERA_PRESETS.keys()), default="legacy_r1")
    parser.add_argument("--head_camera_local_pos", type=float, nargs=3, default=planner.base.DEFAULT_HEAD_CAMERA_LOCAL_POS.tolist())
    parser.add_argument("--head_camera_local_quat_wxyz", type=float, nargs=4, default=planner.base.DEFAULT_HEAD_CAMERA_LOCAL_QUAT_WXYZ.tolist())
    parser.add_argument("--overlay_text", type=int, default=0)
    parser.add_argument("--enable_viewer", type=int, default=0)
    parser.add_argument("--viewer_show_camera_frustums", type=int, default=0)
    parser.add_argument("--viewer_frame_delay", type=float, default=0.0)
    parser.add_argument("--viewer_wait_at_end", type=int, default=0)
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pose_debug_records(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No records found in {path}")
    return records


def _normalize_quat_wxyz(q: Sequence[float]) -> np.ndarray:
    q_arr = np.asarray(q, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(q_arr)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    n = np.linalg.norm(q_arr)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q_arr / n


def quat_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    qa = _normalize_quat_wxyz(a)
    qb = _normalize_quat_wxyz(b)
    dot = float(np.clip(np.abs(np.dot(qa, qb)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def interpolate_scalar(a: float, b: float, alpha: float) -> float:
    return float((1.0 - alpha) * float(a) + alpha * float(b))


def interpolate_vector(a: Sequence[float], b: Sequence[float], alpha: float) -> List[float]:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    return ((1.0 - alpha) * a_arr + alpha * b_arr).astype(np.float64).tolist()


def interpolate_pose_wxyz(a: Sequence[float], b: Sequence[float], alpha: float) -> List[float]:
    a_arr = np.asarray(a, dtype=np.float64).reshape(7)
    b_arr = np.asarray(b, dtype=np.float64).reshape(7)
    if not np.all(np.isfinite(a_arr)):
        return b_arr.astype(np.float64).tolist()
    if not np.all(np.isfinite(b_arr)):
        return a_arr.astype(np.float64).tolist()
    pos = (1.0 - alpha) * a_arr[:3] + alpha * b_arr[:3]
    qa = _normalize_quat_wxyz(a_arr[3:])
    qb = _normalize_quat_wxyz(b_arr[3:])
    if np.linalg.norm(qa - qb) < 1e-12:
        quat_wxyz = qa
    else:
        rots = R.from_quat([
            planner.base.quat_wxyz_to_xyzw(qa),
            planner.base.quat_wxyz_to_xyzw(qb),
        ])
        slerp = Slerp([0.0, 1.0], rots)
        quat_xyzw = slerp([float(alpha)]).as_quat()[0]
        quat_wxyz = planner.base.quat_xyzw_to_wxyz(quat_xyzw)
    return np.concatenate([pos, quat_wxyz], axis=0).astype(np.float64).tolist()


def interpolate_object_pose_maps(a_map: Dict[str, Dict], b_map: Dict[str, Dict], alpha: float) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    names = sorted(set(a_map.keys()) | set(b_map.keys()))
    for name in names:
        a_info = a_map.get(name)
        b_info = b_map.get(name)
        if a_info is None and b_info is None:
            continue
        if a_info is None:
            out[name] = dict(b_info)
            continue
        if b_info is None:
            out[name] = dict(a_info)
            continue
        out[name] = {}
        for key in sorted(set(a_info.keys()) | set(b_info.keys())):
            a_val = a_info.get(key)
            b_val = b_info.get(key)
            if key.endswith("pose_world_wxyz") and a_val is not None and b_val is not None:
                out[name][key] = interpolate_pose_wxyz(a_val, b_val, alpha)
            else:
                out[name][key] = a_val if alpha < 0.5 else b_val
    return out


def interpolate_record(a: Dict, b: Dict, alpha: float) -> Dict:
    out = {
        "record_index": int(round((1.0 - alpha) * int(a.get("record_index", 0)) + alpha * int(b.get("record_index", 0)))),
        "active_frame": a.get("active_frame") if alpha < 0.5 else b.get("active_frame"),
        "stage": a.get("stage") if alpha < 0.5 else b.get("stage"),
        "overlay_lines": list(a.get("overlay_lines", [])) if alpha < 0.5 else list(b.get("overlay_lines", [])),
        "current_left_arm_qpos_rad": interpolate_vector(a["current_left_arm_qpos_rad"], b["current_left_arm_qpos_rad"], alpha),
        "current_right_arm_qpos_rad": interpolate_vector(a["current_right_arm_qpos_rad"], b["current_right_arm_qpos_rad"], alpha),
        "current_left_gripper_joint_qpos_rad": interpolate_vector(a["current_left_gripper_joint_qpos_rad"], b["current_left_gripper_joint_qpos_rad"], alpha),
        "current_right_gripper_joint_qpos_rad": interpolate_vector(a["current_right_gripper_joint_qpos_rad"], b["current_right_gripper_joint_qpos_rad"], alpha),
        "current_left_gripper_command": interpolate_scalar(a.get("current_left_gripper_command", 0.0), b.get("current_left_gripper_command", 0.0), alpha),
        "current_right_gripper_command": interpolate_scalar(a.get("current_right_gripper_command", 0.0), b.get("current_right_gripper_command", 0.0), alpha),
        "object_actor_poses": interpolate_object_pose_maps(a.get("object_actor_poses", {}), b.get("object_actor_poses", {}), alpha),
        "frame_metrics": a.get("frame_metrics") if alpha < 0.5 else b.get("frame_metrics"),
    }
    for key in (
        "current_head_camera_pose_world_wxyz",
        "current_left_wrist_camera_pose_world_wxyz",
        "current_right_wrist_camera_pose_world_wxyz",
        "current_left_tcp_pose_world_wxyz",
        "current_right_tcp_pose_world_wxyz",
        "current_left_ee_pose_world_wxyz",
        "current_right_ee_pose_world_wxyz",
        "replay_head_camera_pose_world_wxyz",
    ):
        a_pose = a.get(key)
        b_pose = b.get(key)
        if a_pose is not None and b_pose is not None:
            out[key] = interpolate_pose_wxyz(a_pose, b_pose, alpha)
        else:
            out[key] = a_pose if alpha < 0.5 else b_pose
    return out


def record_distance_metrics(a: Dict, b: Dict) -> Dict[str, float]:
    left_pos_a = np.asarray(a["current_left_tcp_pose_world_wxyz"][:3], dtype=np.float64)
    left_pos_b = np.asarray(b["current_left_tcp_pose_world_wxyz"][:3], dtype=np.float64)
    right_pos_a = np.asarray(a["current_right_tcp_pose_world_wxyz"][:3], dtype=np.float64)
    right_pos_b = np.asarray(b["current_right_tcp_pose_world_wxyz"][:3], dtype=np.float64)
    left_q_a = np.asarray(a["current_left_arm_qpos_rad"], dtype=np.float64)
    left_q_b = np.asarray(b["current_left_arm_qpos_rad"], dtype=np.float64)
    right_q_a = np.asarray(a["current_right_arm_qpos_rad"], dtype=np.float64)
    right_q_b = np.asarray(b["current_right_arm_qpos_rad"], dtype=np.float64)
    return {
        "left_tcp_dist_m": float(np.linalg.norm(left_pos_b - left_pos_a)),
        "right_tcp_dist_m": float(np.linalg.norm(right_pos_b - right_pos_a)),
        "left_tcp_rot_deg": quat_angle_deg(a["current_left_tcp_pose_world_wxyz"][3:], b["current_left_tcp_pose_world_wxyz"][3:]),
        "right_tcp_rot_deg": quat_angle_deg(a["current_right_tcp_pose_world_wxyz"][3:], b["current_right_tcp_pose_world_wxyz"][3:]),
        "left_qpos_max_abs_rad": float(np.max(np.abs(left_q_b - left_q_a))),
        "right_qpos_max_abs_rad": float(np.max(np.abs(right_q_b - right_q_a))),
        "left_gripper_delta": float(abs(float(b.get("current_left_gripper_command", 0.0)) - float(a.get("current_left_gripper_command", 0.0)))),
        "right_gripper_delta": float(abs(float(b.get("current_right_gripper_command", 0.0)) - float(a.get("current_right_gripper_command", 0.0)))),
    }


def is_near_duplicate(a: Dict, b: Dict, args: argparse.Namespace) -> bool:
    m = record_distance_metrics(a, b)
    return (
        m["left_tcp_dist_m"] <= float(args.dedup_pos_thresh_m)
        and m["right_tcp_dist_m"] <= float(args.dedup_pos_thresh_m)
        and m["left_tcp_rot_deg"] <= float(args.dedup_rot_thresh_deg)
        and m["right_tcp_rot_deg"] <= float(args.dedup_rot_thresh_deg)
        and m["left_qpos_max_abs_rad"] <= float(args.dedup_joint_thresh_rad)
        and m["right_qpos_max_abs_rad"] <= float(args.dedup_joint_thresh_rad)
        and m["left_gripper_delta"] <= float(args.dedup_gripper_thresh)
        and m["right_gripper_delta"] <= float(args.dedup_gripper_thresh)
        and str(a.get("stage")) == str(b.get("stage"))
    )


def thin_hover_frames(records: Sequence[Dict], args: argparse.Namespace) -> List[Dict]:
    if len(records) <= 2:
        return list(records)
    keep_every = max(int(args.keep_hover_frames_every), 1)
    out: List[Dict] = [records[0]]
    dup_run_count = 0
    for idx in range(1, len(records) - 1):
        prev_rec = out[-1]
        cur_rec = records[idx]
        next_rec = records[idx + 1]
        prev_dup = is_near_duplicate(prev_rec, cur_rec, args)
        next_dup = is_near_duplicate(cur_rec, next_rec, args)
        if prev_dup and next_dup:
            dup_run_count += 1
            if dup_run_count % keep_every == 0:
                out.append(cur_rec)
            continue
        dup_run_count = 0
        out.append(cur_rec)
    out.append(records[-1])
    return out


def densify_records(records: Sequence[Dict], interp_factor: int) -> List[Dict]:
    factor = max(int(interp_factor), 1)
    if len(records) <= 1 or factor <= 1:
        return list(records)
    dense: List[Dict] = []
    for idx in range(len(records) - 1):
        a = records[idx]
        b = records[idx + 1]
        for sub in range(factor):
            alpha = float(sub) / float(factor)
            dense.append(interpolate_record(a, b, alpha) if sub > 0 else dict(a))
    dense.append(dict(records[-1]))
    return dense


def build_renderer_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        planner_backend="urdfik",
        urdfik_trajectory_mode="joint_interp",
        urdfik_cartesian_interp_steps=2,
        urdfik_cartesian_interp_auto_step_m=0.05,
        robot_config=args.robot_config,
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        fovy_deg=float(args.fovy_deg),
        torso_qpos=np.asarray(args.torso_qpos, dtype=np.float64),
        robot_base_pose=args.robot_base_pose,
        third_person_view=False,
        camera_cv_axis_mode=str(args.camera_cv_axis_mode),
        head_camera_local_pos=np.asarray(args.head_camera_local_pos, dtype=np.float64),
        head_camera_local_quat_wxyz=np.asarray(args.head_camera_local_quat_wxyz, dtype=np.float64),
        enable_viewer=bool(args.enable_viewer),
        viewer_show_camera_frustums=bool(args.viewer_show_camera_frustums),
        viewer_frame_delay=float(args.viewer_frame_delay),
        viewer_wait_at_end=bool(args.viewer_wait_at_end),
        debug_visualize_targets=False,
        debug_target_axis_length=0.08,
        debug_target_axis_thickness=0.004,
        disable_table=bool(args.disable_table),
        base_occluder_enable=bool(args.base_occluder_enable),
        base_occluder_local_pos=np.asarray(args.base_occluder_local_pos, dtype=np.float64),
        base_occluder_half_size=np.asarray(args.base_occluder_half_size, dtype=np.float64),
        base_occluder_color=np.asarray(args.base_occluder_color, dtype=np.float64),
        lighting_mode=str(args.lighting_mode),
    )


def _joint_indices(entity, joints: Sequence) -> List[int]:
    active = list(entity.get_active_joints())
    return [active.index(j) for j in joints]


def build_joint_index_cache(renderer) -> Dict[str, object]:
    robot = renderer.robot
    left_entity = robot.left_entity
    right_entity = robot.right_entity
    return {
        "shared_entity": left_entity is right_entity,
        "left_entity": left_entity,
        "right_entity": right_entity,
        "left_arm_idx": _joint_indices(left_entity, robot.left_arm_joints),
        "right_arm_idx": _joint_indices(right_entity, robot.right_arm_joints),
        "left_gripper_idx": _joint_indices(left_entity, [item[0] for item in robot.left_gripper]),
        "right_gripper_idx": _joint_indices(right_entity, [item[0] for item in robot.right_gripper]),
    }


def apply_record_state(renderer, cache: Dict[str, object], record: Dict) -> None:
    robot = renderer.robot
    left_entity = cache["left_entity"]
    right_entity = cache["right_entity"]
    shared = bool(cache["shared_entity"])
    if shared:
        qpos = np.asarray(left_entity.get_qpos(), dtype=np.float64).copy()
        qvel = np.zeros_like(qpos, dtype=np.float64)
        for idx, value in zip(cache["left_arm_idx"], np.asarray(record["current_left_arm_qpos_rad"], dtype=np.float64).reshape(-1)):
            qpos[int(idx)] = float(value)
        for idx, value in zip(cache["right_arm_idx"], np.asarray(record["current_right_arm_qpos_rad"], dtype=np.float64).reshape(-1)):
            qpos[int(idx)] = float(value)
        for idx, value in zip(cache["left_gripper_idx"], np.asarray(record["current_left_gripper_joint_qpos_rad"], dtype=np.float64).reshape(-1)):
            qpos[int(idx)] = float(value)
        for idx, value in zip(cache["right_gripper_idx"], np.asarray(record["current_right_gripper_joint_qpos_rad"], dtype=np.float64).reshape(-1)):
            qpos[int(idx)] = float(value)
        left_entity.set_qpos(qpos)
        left_entity.set_qvel(qvel)
    else:
        left_qpos = np.asarray(left_entity.get_qpos(), dtype=np.float64).copy()
        left_qvel = np.zeros_like(left_qpos, dtype=np.float64)
        right_qpos = np.asarray(right_entity.get_qpos(), dtype=np.float64).copy()
        right_qvel = np.zeros_like(right_qpos, dtype=np.float64)
        for idx, value in zip(cache["left_arm_idx"], np.asarray(record["current_left_arm_qpos_rad"], dtype=np.float64).reshape(-1)):
            left_qpos[int(idx)] = float(value)
        for idx, value in zip(cache["left_gripper_idx"], np.asarray(record["current_left_gripper_joint_qpos_rad"], dtype=np.float64).reshape(-1)):
            left_qpos[int(idx)] = float(value)
        for idx, value in zip(cache["right_arm_idx"], np.asarray(record["current_right_arm_qpos_rad"], dtype=np.float64).reshape(-1)):
            right_qpos[int(idx)] = float(value)
        for idx, value in zip(cache["right_gripper_idx"], np.asarray(record["current_right_gripper_joint_qpos_rad"], dtype=np.float64).reshape(-1)):
            right_qpos[int(idx)] = float(value)
        left_entity.set_qpos(left_qpos)
        left_entity.set_qvel(left_qvel)
        right_entity.set_qpos(right_qpos)
        right_entity.set_qvel(right_qvel)
    robot.left_gripper_val = float(record.get("current_left_gripper_command", robot.left_gripper_val))
    robot.right_gripper_val = float(record.get("current_right_gripper_command", robot.right_gripper_val))


def create_object_actors(renderer, replay_dir: Optional[Path]) -> Dict[str, object]:
    if replay_dir is None:
        return {}
    try:
        _, object_tracks = planner.load_object_tracks(replay_dir)
    except Exception as exc:
        print(f"[smooth-bundle] failed to load object tracks from {replay_dir}: {exc}")
        return {}
    actors: Dict[str, object] = {}
    for name, track in object_tracks.items():
        actor, _ = planner.create_execution_object_actor(
            renderer.scene,
            track.mesh_file,
            f"smooth_bundle_object_{name}",
            ignore_collision=True,
        )
        actors[str(name)] = actor
    return actors


def update_object_actors(object_actors: Dict[str, object], record: Dict) -> None:
    pose_map = record.get("object_actor_poses", {}) or {}
    for name, actor in object_actors.items():
        info = pose_map.get(name)
        if info is None:
            continue
        pose = info.get("actor_pose_world_wxyz")
        if pose is None:
            continue
        planner.set_actor_pose(actor, np.asarray(pose, dtype=np.float64).reshape(7))


def overlay_lines_for_record(record: Dict, frame_idx: int, total_frames: int) -> List[str]:
    lines = [
        f"smooth_bundle={frame_idx + 1}/{total_frames}",
        f"stage={record.get('stage', 'unknown')}",
    ]
    active_frame = record.get("active_frame")
    if active_frame is not None:
        lines.append(f"active_frame={active_frame}")
    return lines


def write_record_videos(renderer, records: Sequence[Dict], args: argparse.Namespace, output_dir: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    size = (int(args.image_width), int(args.image_height))
    head_writer = cv2.VideoWriter(str(output_dir / args.output_head_name), fourcc, int(args.fps), size)
    left_writer = cv2.VideoWriter(str(output_dir / args.output_left_wrist_name), fourcc, int(args.fps), size)
    right_writer = cv2.VideoWriter(str(output_dir / args.output_right_wrist_name), fourcc, int(args.fps), size)
    if not head_writer.isOpened() or not left_writer.isOpened() or not right_writer.isOpened():
        raise RuntimeError(f"Failed to open one or more output videos in {output_dir}")

    cache = build_joint_index_cache(renderer)
    plan_summary_path = args.input_dir / args.plan_summary_name
    replay_dir = None
    if plan_summary_path.is_file():
        summary = load_json(plan_summary_path)
        replay_dir_value = summary.get("replay_dir")
        if replay_dir_value:
            replay_dir = Path(str(replay_dir_value)).resolve()
    object_actors = create_object_actors(renderer, replay_dir)

    try:
        for frame_idx, record in enumerate(records):
            apply_record_state(renderer, cache, record)
            update_object_actors(object_actors, record)
            renderer.update_robot_link_cameras()
            renderer.scene.update_render()
            if getattr(renderer, "viewer", None) is not None and not renderer.viewer.closed:
                renderer.viewer.render()
            lines = overlay_lines_for_record(record, frame_idx, len(records))
            head_rgb, _ = renderer.capture_camera(renderer.zed_camera)
            left_rgb, _ = renderer.capture_camera(renderer.left_wrist_camera)
            right_rgb, _ = renderer.capture_camera(renderer.right_wrist_camera)
            if bool(args.overlay_text):
                head_bgr = planner.base.overlay_text(head_rgb, lines)
                left_bgr = planner.base.overlay_text(left_rgb, lines)
                right_bgr = planner.base.overlay_text(right_rgb, lines)
            else:
                head_bgr = cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
                left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
                right_bgr = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)
            head_writer.write(head_bgr)
            left_writer.write(left_bgr)
            right_writer.write(right_bgr)
    finally:
        head_writer.release()
        left_writer.release()
        right_writer.release()
        if getattr(renderer, "viewer", None) is not None and bool(args.viewer_wait_at_end):
            renderer.hold_viewer()


def save_pose_debug_jsonl(path: Path, records: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def main() -> None:
    args = parse_args()
    raw_pose_debug = args.input_dir / args.pose_debug_name
    raw_plan_summary = args.input_dir / args.plan_summary_name
    records = load_pose_debug_records(raw_pose_debug)
    filtered_records = thin_hover_frames(records, args)
    smooth_records = densify_records(filtered_records, args.interp_factor)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if raw_plan_summary.is_file():
        summary = load_json(raw_plan_summary)
    else:
        summary = {}

    render_args = build_renderer_args(args)
    renderer = planner.build_renderer(render_args)
    write_record_videos(renderer=renderer, records=smooth_records, args=args, output_dir=args.output_dir)
    save_pose_debug_jsonl(args.output_dir / args.output_pose_debug_name, smooth_records)

    smooth_summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "raw_pose_debug": str(raw_pose_debug),
        "raw_plan_summary": str(raw_plan_summary) if raw_plan_summary.is_file() else None,
        "raw_record_count": len(records),
        "filtered_record_count": len(filtered_records),
        "smoothed_record_count": len(smooth_records),
        "interp_factor": int(args.interp_factor),
        "fps": int(args.fps),
        "dedup_pos_thresh_m": float(args.dedup_pos_thresh_m),
        "dedup_rot_thresh_deg": float(args.dedup_rot_thresh_deg),
        "dedup_joint_thresh_rad": float(args.dedup_joint_thresh_rad),
        "dedup_gripper_thresh": float(args.dedup_gripper_thresh),
        "keep_hover_frames_every": int(args.keep_hover_frames_every),
        "source_plan_summary_selected_arm": summary.get("selected_arm"),
        "source_plan_summary_keyframes": summary.get("keyframes"),
        "outputs": {
            "head_video": str(args.output_dir / args.output_head_name),
            "left_wrist_video": str(args.output_dir / args.output_left_wrist_name),
            "right_wrist_video": str(args.output_dir / args.output_right_wrist_name),
            "pose_debug_jsonl": str(args.output_dir / args.output_pose_debug_name),
        },
    }
    with (args.output_dir / args.output_summary_name).open("w", encoding="utf-8") as f:
        json.dump(smooth_summary, f, indent=2, ensure_ascii=False)

    print(
        "[smooth-bundle] "
        f"raw={len(records)} filtered={len(filtered_records)} smoothed={len(smooth_records)} "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
