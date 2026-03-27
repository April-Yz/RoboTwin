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
        description="Replay AnyGrasp planner pose_debug.jsonl with extra interpolation so exported videos look smoother."
    )
    parser.add_argument("--plan_summary_json", type=Path, default=None, help="Optional plan_summary.json. If provided, infer replay_dir and pose_debug.jsonl from it.")
    parser.add_argument("--pose_debug_jsonl", type=Path, default=None, help="Recorded pose_debug.jsonl exported by plan_anygrasp_keyframes_r1.py.")
    parser.add_argument("--replay_dir", type=Path, default=None, help="Replay dir containing multi_object_world_poses.npz. If omitted and --plan_summary_json is set, infer from the summary.")
    parser.add_argument("--output_path", type=Path, required=True, help="Output MP4 path for the smoothed replay video.")
    parser.add_argument("--interp_factor", type=int, default=4, help="How many subframes to generate per recorded interval. 1 means no extra interpolation.")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS.")
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--fovy_deg", type=float, default=90.0)
    parser.add_argument("--robot_config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--torso_qpos", type=float, nargs=4, default=planner.base.DEFAULT_TORSO_QPOS.tolist())
    parser.add_argument("--robot_base_pose", type=float, nargs=7, default=None)
    parser.add_argument("--head_only", type=int, default=1)
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
    parser.add_argument("--camera_cv_axis_mode", choices=sorted(planner.base.CV_TO_WORLD_CAMERA_PRESETS.keys()), default="legacy_r1")
    parser.add_argument("--head_camera_local_pos", type=float, nargs=3, default=planner.base.DEFAULT_HEAD_CAMERA_LOCAL_POS.tolist())
    parser.add_argument("--head_camera_local_quat_wxyz", type=float, nargs=4, default=planner.base.DEFAULT_HEAD_CAMERA_LOCAL_QUAT_WXYZ.tolist())
    parser.add_argument("--overlay_text", type=int, default=1, help="If 1, overlay stage/index labels on the smoothed replay video.")
    parser.add_argument("--max_input_records", type=int, default=0, help="If > 0, replay only the first N pose_debug records. Useful for quick smoke tests.")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_inputs(args: argparse.Namespace) -> Tuple[Path, Optional[Path], Optional[Dict]]:
    summary = None
    pose_debug = args.pose_debug_jsonl
    replay_dir = args.replay_dir
    if args.plan_summary_json is not None:
        summary = load_json(args.plan_summary_json)
        if pose_debug is None:
            pose_debug_value = summary.get("pose_debug")
            if pose_debug_value is None:
                raise ValueError(f"plan_summary has no pose_debug field: {args.plan_summary_json}")
            pose_debug = Path(str(pose_debug_value))
        if replay_dir is None:
            replay_value = summary.get("replay_dir")
            if replay_value is not None:
                replay_dir = Path(str(replay_value))
    if pose_debug is None:
        raise ValueError("Specify --pose_debug_jsonl or --plan_summary_json")
    pose_debug = pose_debug.resolve()
    if replay_dir is not None:
        replay_dir = replay_dir.resolve()
    return pose_debug, replay_dir, summary


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
        a_pose = a_info.get("actor_pose_world_wxyz")
        b_pose = b_info.get("actor_pose_world_wxyz")
        if a_pose is not None and b_pose is not None:
            out[name]["actor_pose_world_wxyz"] = interpolate_pose_wxyz(a_pose, b_pose, alpha)
        elif b_pose is not None:
            out[name]["actor_pose_world_wxyz"] = list(b_pose)
        elif a_pose is not None:
            out[name]["actor_pose_world_wxyz"] = list(a_pose)
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


def build_replay_renderer_args(args: argparse.Namespace) -> SimpleNamespace:
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
        third_person_view=bool(args.third_person_view),
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
    cache: Dict[str, object] = {
        "shared_entity": left_entity is right_entity,
        "left_entity": left_entity,
        "right_entity": right_entity,
        "left_arm_idx": _joint_indices(left_entity, robot.left_arm_joints),
        "right_arm_idx": _joint_indices(right_entity, robot.right_arm_joints),
        "left_gripper_idx": _joint_indices(left_entity, [item[0] for item in robot.left_gripper]),
        "right_gripper_idx": _joint_indices(right_entity, [item[0] for item in robot.right_gripper]),
    }
    return cache


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
        print(f"[smooth-replay] failed to load object tracks from {replay_dir}: {exc}")
        return {}
    actors: Dict[str, object] = {}
    for name, track in object_tracks.items():
        actor, _ = planner.create_execution_object_actor(
            renderer.scene,
            track.mesh_file,
            f"smooth_replay_object_{name}",
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


def render_record_frame(renderer, head_writer: cv2.VideoWriter, third_writer: Optional[cv2.VideoWriter], record: Dict, use_overlay: bool, frame_idx: int, total_frames: int) -> None:
    renderer.update_robot_link_cameras()
    renderer.scene.update_render()
    if getattr(renderer, "viewer", None) is not None and not renderer.viewer.closed:
        renderer.viewer.render()
    overlay_lines = [
        f"smooth_replay={frame_idx + 1}/{total_frames}",
        f"stage={record.get('stage', 'unknown')}",
    ]
    active_frame = record.get("active_frame")
    if active_frame is not None:
        overlay_lines.append(f"active_frame={active_frame}")
    head_rgb, _ = renderer.capture_camera(renderer.zed_camera)
    head_bgr = planner.base.overlay_text(head_rgb, overlay_lines) if use_overlay else cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR)
    head_writer.write(head_bgr)
    if third_writer is not None:
        third_rgb, _ = renderer.capture_camera(renderer.third_camera)
        third_bgr = planner.base.overlay_text(third_rgb, overlay_lines) if use_overlay else cv2.cvtColor(third_rgb, cv2.COLOR_RGB2BGR)
        third_writer.write(third_bgr)


def main() -> None:
    args = parse_args()
    pose_debug_path, replay_dir, summary = resolve_inputs(args)
    records = load_pose_debug_records(pose_debug_path)
    if int(args.max_input_records) > 0:
        records = records[: max(int(args.max_input_records), 1)]
    dense_records = densify_records(records, args.interp_factor)

    render_args = build_replay_renderer_args(args)
    renderer = planner.build_renderer(render_args)
    cache = build_joint_index_cache(renderer)
    object_actors = create_object_actors(renderer, replay_dir)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    head_writer = cv2.VideoWriter(str(args.output_path), fourcc, int(args.fps), (int(args.image_width), int(args.image_height)))
    if not head_writer.isOpened():
        raise RuntimeError(f"Failed to open output video: {args.output_path}")
    third_writer = None
    if bool(args.third_person_view) and not bool(args.head_only):
        third_path = args.output_path.with_name(args.output_path.stem + "_third.mp4")
        third_writer = cv2.VideoWriter(str(third_path), fourcc, int(args.fps), (int(args.image_width), int(args.image_height)))
        if not third_writer.isOpened():
            raise RuntimeError(f"Failed to open third-person video: {third_path}")

    try:
        for frame_idx, record in enumerate(dense_records):
            apply_record_state(renderer, cache, record)
            update_object_actors(object_actors, record)
            render_record_frame(
                renderer=renderer,
                head_writer=head_writer,
                third_writer=third_writer,
                record=record,
                use_overlay=bool(args.overlay_text),
                frame_idx=frame_idx,
                total_frames=len(dense_records),
            )
        print(
            "[smooth-replay] "
            f"input_frames={len(records)} output_frames={len(dense_records)} interp_factor={int(args.interp_factor)} "
            f"output={args.output_path}"
        )
        if summary is not None:
            print(
                "[smooth-replay] "
                f"source_plan_summary={args.plan_summary_json} source_pose_debug={pose_debug_path}"
            )
    finally:
        head_writer.release()
        if third_writer is not None:
            third_writer.release()
        if getattr(renderer, "viewer", None) is not None and bool(args.viewer_wait_at_end):
            renderer.hold_viewer()


if __name__ == "__main__":
    main()
