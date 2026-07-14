#!/usr/bin/env python3
"""Read-only comparison of OursV2 and three AnyGrasp selection strategies.

This script never imports or invokes a planner.  It reads existing preview and
planner summaries, reconstructs the relevant poses, and writes independent V4
PNG/JSON audit artifacts.  Existing summaries and result directories are never
modified.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rotation


REPO = Path("/home/zaijia001/ssd/RoboTwin")
DATA_ROOT = Path("/home/zaijia001/ssd/data/piper/hand")
PREVIEW_ROOT = REPO / "code_painting/anygrasp_h2o_preview_d435_robot_frame"
TOP_SCORE_ROOT = REPO / "code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/S_graspnet_topscore_rightcam_m003_selected25"
OURS_ROOT = REPO / "code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_de_human_replay_clean_right_cam"
OUTPUT_ROOT = REPO / "code_painting/selection_strategy_compare_v4"
CALIBRATION_BUNDLE = REPO / "calibration_bundle_piper_new_table_0515.json"

TASKS = (
    "pick_diverse_bottles",
    "place_bread_basket",
    "stack_cups",
    "handover_bottle",
    "pnp_bread",
    "pnp_tray",
)

# OpenCV BGR colors.  Axis colors remain X=red, Y=green, Z=blue.
STRATEGY_COLORS = {
    "oursv2": (255, 255, 0),       # cyan
    "orientation": (255, 0, 255), # magenta
    "fused": (0, 255, 255),       # yellow
    "top_score": (20, 20, 20),    # near-black
    "top_raw": (0, 140, 255),     # orange
    "top_legacy": (255, 90, 20),  # cobalt blue
}
AXIS_COLORS = {
    "x": (0, 0, 255),
    "y": (0, 210, 0),
    "z": (255, 0, 0),
}

# camera CV -> replay head-camera local, matching legacy_r1.
CV_TO_HEAD = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)

# Canonical robot/replay frame expressed in raw AnyGrasp local coordinates:
# canonical +X = raw -Z, canonical +Y = raw +Y, canonical +Z = raw +X.
RAW_TO_CANONICAL = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-root", type=Path, default=PREVIEW_ROOT)
    parser.add_argument("--top-score-root", type=Path, default=TOP_SCORE_ROOT)
    parser.add_argument("--ours-root", type=Path, default=OURS_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--calibration-bundle", type=Path, default=CALIBRATION_BUNDLE)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--ids", nargs="*", type=int, default=None)
    parser.add_argument("--requested-frames", nargs="*", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--approach-offset-m", type=float, default=0.12)
    parser.add_argument("--axis-length-m", type=float, default=0.045)
    parser.add_argument("--cell-width", type=int, default=640)
    parser.add_argument("--cell-height", type=int, default=480)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return data


def orthonormalize(rot: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(rot, dtype=np.float64).reshape(3, 3))
    out = u @ vh
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1.0
        out = u @ vh
    return out


def pose_matrix(pose_wxyz: Sequence[float]) -> np.ndarray:
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = pose[:3]
    out[:3, :3] = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix()
    return out


def matrix_pose(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    quat_xyzw = Rotation.from_matrix(orthonormalize(matrix[:3, :3])).as_quat()
    return np.concatenate([matrix[:3, 3], [quat_xyzw[3], *quat_xyzw[:3]]]).astype(np.float64)


def pose_from_position_rotation(position: Sequence[float], rotation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    matrix[:3, :3] = orthonormalize(rotation)
    return matrix_pose(matrix)


def shift_pose_local(pose_wxyz: Sequence[float], axis: str, offset_m: float) -> np.ndarray:
    matrix = pose_matrix(pose_wxyz)
    axis_idx = {"local_x": 0, "local_y": 1, "local_z": 2}[axis]
    matrix[:3, 3] += matrix[:3, axis_idx] * float(offset_m)
    return matrix_pose(matrix)


def rotation_distance_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = orthonormalize(rot_a).T @ orthonormalize(rot_b)
    value = np.clip((float(np.trace(rel)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def pose_payload(
    pose_wxyz: Optional[Sequence[float]], *, frame: str = "world"
) -> Optional[Dict[str, Any]]:
    if pose_wxyz is None:
        return None
    pose = np.asarray(pose_wxyz, dtype=np.float64).reshape(7)
    matrix = pose_matrix(pose)
    return {
        "frame": frame,
        f"position_{frame}_m": pose[:3].tolist(),
        "quat_wxyz": pose[3:].tolist(),
        f"rotation_{frame}": matrix[:3, :3].tolist(),
        f"matrix_{frame}": matrix.tolist(),
    }


def finite_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(finite_json(dict(payload)), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_anygrasp_frame(anygrasp_dir: Path, frame: int) -> Tuple[Dict[str, Any], Path]:
    candidates = [
        anygrasp_dir / "grasps" / f"grasp_{int(frame):06d}.json",
        anygrasp_dir / "grasps" / f"grasp_{int(frame)}.json",
    ]
    for path in candidates:
        if path.is_file():
            return load_json(path), path
    raise FileNotFoundError(f"AnyGrasp JSON missing for frame={frame}: {anygrasp_dir}")


class ReplayData:
    def __init__(self, replay_dir: Path):
        self.replay_dir = replay_dir
        self.path = replay_dir / "multi_object_world_poses.npz"
        self.data = np.load(str(self.path), allow_pickle=True)
        if "selected_source_frame_indices" in self.data.files:
            frames = np.asarray(self.data["selected_source_frame_indices"], dtype=np.int64).reshape(-1)
            self.frame_to_index = {int(frame): idx for idx, frame in enumerate(frames)}
        else:
            self.frame_to_index = {}

    def index(self, frame: int) -> int:
        if int(frame) in self.frame_to_index:
            return int(self.frame_to_index[int(frame)])
        poses = np.asarray(self.data["head_camera_pose_world_wxyz"])
        if 0 <= int(frame) < len(poses):
            return int(frame)
        raise KeyError(f"frame={frame} absent from {self.path}")

    def head_pose(self, frame: int) -> np.ndarray:
        return np.asarray(self.data["head_camera_pose_world_wxyz"], dtype=np.float64)[self.index(frame)].reshape(7)


def camera_to_world_pose(
    position_cam: Sequence[float],
    rotation_cam: np.ndarray,
    head_pose_wxyz: Sequence[float],
) -> np.ndarray:
    head = pose_matrix(head_pose_wxyz)
    position_cam = np.asarray(position_cam, dtype=np.float64).reshape(3)
    rotation_cam = orthonormalize(rotation_cam)
    position_world = head[:3, 3] + head[:3, :3] @ (CV_TO_HEAD @ position_cam)
    rotation_world = head[:3, :3] @ CV_TO_HEAD @ rotation_cam
    return pose_from_position_rotation(position_world, rotation_world)


def compute_hand_world_pose(
    hand_data: Mapping[str, np.ndarray],
    replay: ReplayData,
    arm: str,
    frame: int,
) -> Optional[np.ndarray]:
    pos_key = f"{arm}_gripper_position"
    rot_key = f"{arm}_gripper_rotation_matrix"
    valid_key = f"{arm}_gripper_valid"
    if pos_key not in hand_data or rot_key not in hand_data:
        return None
    positions = np.asarray(hand_data[pos_key])
    rotations = np.asarray(hand_data[rot_key])
    if frame < 0 or frame >= len(positions):
        return None
    if valid_key in hand_data:
        valid = np.asarray(hand_data[valid_key], dtype=bool)
        if frame < len(valid) and not bool(valid[frame]):
            return None
    position = np.asarray(positions[frame], dtype=np.float64).reshape(3)
    rotation = np.asarray(rotations[frame], dtype=np.float64).reshape(3, 3)
    if not np.isfinite(position).all() or not np.isfinite(rotation).all():
        return None
    return camera_to_world_pose(position, rotation, replay.head_pose(frame))


def load_calibration(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    robot_config = payload.get("robot_config", payload)
    left_pose = payload.get("left_base_world_pose")
    right_pose = payload.get("right_base_world_pose")
    if left_pose is None or right_pose is None:
        config = robot_config["left_embodiment_config"]
        poses = config["robot_pose"]
        left_pose, right_pose = poses[0], poses[1]
    return {
        "path": str(path),
        "robot_config": robot_config,
        "base_world_pose": {"left": left_pose, "right": right_pose},
    }


def ik_target_trace(
    target_world_wxyz: Sequence[float],
    arm: str,
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    target_world = pose_matrix(target_world_wxyz)
    base_world = pose_matrix(calibration["base_world_pose"][arm])
    target_base = np.linalg.inv(base_world) @ target_world
    config_key = f"{arm}_embodiment_config"
    config = calibration["robot_config"][config_key]
    gripper_bias = float(config.get("gripper_bias", 0.12))
    delta = np.asarray(config.get("delta_matrix", np.eye(3)), dtype=np.float64).reshape(3, 3)
    link6_base = target_base.copy()
    link6_base[:3, 3] += link6_base[:3, :3] @ np.array([0.12 - gripper_bias, 0.0, 0.0])
    link6_base[:3, :3] = orthonormalize(link6_base[:3, :3] @ np.linalg.inv(delta))
    link6_world = base_world @ link6_base
    return {
        "tcp_world": pose_payload(target_world_wxyz),
        "world_to_base_matrix": np.linalg.inv(base_world).tolist(),
        "tcp_base": pose_payload(matrix_pose(target_base), frame="base"),
        "tcp_to_link6": {
            "translation_local_x_m": float(0.12 - gripper_bias),
            "delta_matrix": delta.tolist(),
            "apply_global_trans_to_ik": False,
        },
        "link6_base": pose_payload(matrix_pose(link6_base), frame="base"),
        "link6_world_for_projection": pose_payload(matrix_pose(link6_world)),
    }


def preview_frame_entry(summary: Mapping[str, Any], requested_frame: int) -> Optional[Dict[str, Any]]:
    frames = [entry for entry in summary.get("frames", []) if isinstance(entry, dict)]
    for entry in frames:
        if int(entry.get("requested_frame", entry.get("frame", -1))) == int(requested_frame):
            return entry
    for entry in frames:
        if int(entry.get("frame", -1)) == int(requested_frame):
            return entry
    return None


def selected_entries(summary: Mapping[str, Any], arm: str) -> List[Dict[str, Any]]:
    by_arm = summary.get("selected_candidates_by_executed_arm", {})
    entries = by_arm.get(arm, []) if isinstance(by_arm, dict) else []
    return [entry for entry in entries if isinstance(entry, dict)]


def base_record(
    *, task: str, episode: int, arm: str, strategy: str,
    requested_frame: int, resolved_frame: int, selection_source: str,
    candidate_idx: Optional[int], candidate_score: Optional[float],
    source_files: Sequence[Path], warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    return {
        "task": task,
        "episode": f"foundation_input_{episode}",
        "episode_id": int(episode),
        "arm": arm,
        "strategy": strategy,
        "requested_frame": int(requested_frame),
        "resolved_frame": int(resolved_frame),
        "delta_frame": int(resolved_frame) - int(requested_frame),
        "selection_source": selection_source,
        "candidate_idx": None if candidate_idx is None else int(candidate_idx),
        "candidate_score": None if candidate_score is None else float(candidate_score),
        "source_files": [str(path) for path in source_files],
        "warnings": list(warnings or []),
    }


def preview_strategy_record(
    *, task: str, episode: int, arm: str, strategy: str, event_index: int,
    requested_frame: int, preview_summary: Mapping[str, Any], anygrasp_dir: Path,
    replay: ReplayData, calibration: Mapping[str, Any], approach_offset_m: float,
    preview_summary_path: Path,
) -> Optional[Dict[str, Any]]:
    frame_entry = preview_frame_entry(preview_summary, requested_frame)
    if frame_entry is None:
        return None
    resolved_frame = int(frame_entry["frame"])
    ranked_key = f"{arm}_{strategy}"
    ranked = frame_entry.get("top_candidates", {}).get(ranked_key, [])
    if not ranked:
        return None
    candidate = ranked[0]
    candidate_idx = int(candidate["candidate_idx"])
    grasp_payload, grasp_path = load_anygrasp_frame(anygrasp_dir, resolved_frame)
    grasps = list(grasp_payload.get("grasps", []))
    if candidate_idx >= len(grasps):
        raise IndexError(f"candidate_idx={candidate_idx} out of range in {grasp_path}")
    raw = grasps[candidate_idx]
    raw_rotation_cam = np.asarray(raw["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    raw_position_cam = np.asarray(raw["translation"], dtype=np.float64).reshape(3)
    head_pose = np.asarray(frame_entry.get("camera_pose_world_wxyz", replay.head_pose(resolved_frame)), dtype=np.float64)
    raw_pose = camera_to_world_pose(raw_position_cam, raw_rotation_cam, head_pose)
    canonical_rotation_cam = orthonormalize(raw_rotation_cam @ RAW_TO_CANONICAL)
    selection_pose = camera_to_world_pose(raw_position_cam, canonical_rotation_cam, head_pose)
    local_z_offset = float(preview_summary.get("candidate_target_local_z_offset_m", -0.05))
    reconstructed_target = shift_pose_local(selection_pose, "local_z", local_z_offset)
    stored_target = pose_from_position_rotation(
        candidate["translation_world"], pose_matrix(selection_pose)[:3, :3]
    )
    residual_m = float(np.linalg.norm(stored_target[:3] - reconstructed_target[:3]))
    warnings: List[str] = []
    if residual_m > 1e-5:
        warnings.append(f"preview stored/reconstructed target residual={residual_m:.6g}m")
    record = base_record(
        task=task, episode=episode, arm=arm, strategy=strategy,
        requested_frame=requested_frame, resolved_frame=resolved_frame,
        selection_source=f"robot-frame preview {strategy} rank1",
        candidate_idx=candidate_idx, candidate_score=float(candidate["anygrasp_score"]),
        source_files=[preview_summary_path, grasp_path, replay.path], warnings=warnings,
    )
    pregrasp = shift_pose_local(stored_target, "local_z", -float(approach_offset_m)) if event_index == 0 else None
    record.update({
        "event_index": int(event_index),
        "raw_pose": pose_payload(raw_pose),
        "raw_frame_convention": "AnyGrasp raw: local +X approach, +Y opening, +Z side normal",
        "canonical_pose": pose_payload(selection_pose),
        "orientation_remap": {
            "name": "anygrasp_raw_to_robot_replay",
            "matrix_right_multiply": RAW_TO_CANONICAL.tolist(),
            "mapping": {"canonical_x": "-raw_z", "canonical_y": "+raw_y", "canonical_z": "+raw_x"},
            "rotation_change_deg": rotation_distance_deg(
                pose_matrix(raw_pose)[:3, :3], pose_matrix(selection_pose)[:3, :3]
            ),
        },
        "selection_pose": pose_payload(selection_pose),
        "selection_metrics": {
            "ranking_strategy": strategy,
            "anygrasp_score_raw": float(candidate["anygrasp_score"]),
            "rotation_distance_deg": float(candidate["rotation_distance_deg"]),
            "orientation_score": float(candidate["orientation_score"]),
            "fused_score": float(candidate["fused_score"]),
            "fused_formula": "0.25 * raw_anygrasp_score + 0.75 * orientation_score",
            "hard_rotation_filter_deg": 90.0,
        },
        "planner_target": {
            "status": "preview reconstruction" if strategy == "orientation" else "hypothetical; fused preview was not the executed group",
            "effective_world_pose": pose_payload(stored_target),
            "reconstructed_world_pose": pose_payload(reconstructed_target),
            "stored_reconstruction_residual_m": residual_m,
            "pregrasp_world_pose": pose_payload(pregrasp),
            "ik_chain": ik_target_trace(stored_target, arm, calibration),
        },
        "offsets": {"local_x_m": 0.0, "local_y_m": 0.0, "local_z_m": local_z_offset, "retreat_m": 0.0, "pregrasp_m": float(approach_offset_m) if event_index == 0 else 0.0},
        "tcp_semantics": "canonical robot/replay TCP target; current Piper0515 gripper_bias=0.12 makes TCP-to-link6 translation zero",
        "gripper_width_m": float(candidate.get("width", raw.get("width", 0.08))),
        "gripper_depth_m": float(candidate.get("depth", raw.get("depth", 0.04))),
    })
    return record


def top_score_record(
    *, task: str, episode: int, arm: str, event_index: int, requested_frame: int,
    top_summary: Mapping[str, Any], anygrasp_dir: Path, replay: ReplayData,
    calibration: Mapping[str, Any], approach_offset_m: float, top_summary_path: Path,
) -> Optional[Dict[str, Any]]:
    entries = selected_entries(top_summary, arm)
    if event_index >= len(entries):
        return None
    entry = entries[event_index]
    resolved_frame = int(entry["source_frame"])
    candidate_idx = int(entry["candidate_idx"])
    grasp_payload, grasp_path = load_anygrasp_frame(anygrasp_dir, resolved_frame)
    grasps = list(grasp_payload.get("grasps", []))
    if candidate_idx >= len(grasps):
        raise IndexError(f"candidate_idx={candidate_idx} out of range in {grasp_path}")
    raw = grasps[candidate_idx]
    raw_pose = np.asarray(entry["raw_pose_world_wxyz"], dtype=np.float64).reshape(7)
    raw_world_rot = pose_matrix(raw_pose)[:3, :3]
    canonical_pose = pose_from_position_rotation(raw_pose[:3], raw_world_rot @ RAW_TO_CANONICAL)
    legacy_target = np.asarray(entry["pose_world_wxyz"], dtype=np.float64).reshape(7)
    local_z_offset = float(top_summary.get("candidate_target_local_z_offset_m", -0.05))
    canonical_target = shift_pose_local(canonical_pose, "local_z", local_z_offset)
    legacy_pregrasp = shift_pose_local(legacy_target, "local_z", -float(approach_offset_m)) if event_index == 0 else None
    canonical_pregrasp = shift_pose_local(canonical_target, "local_z", -float(approach_offset_m)) if event_index == 0 else None
    raw_json_rot = np.asarray(raw["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    summary_rot = np.asarray(entry["rotation_cam"], dtype=np.float64).reshape(3, 3)
    warnings: List[str] = []
    raw_rotation_residual = float(np.max(np.abs(raw_json_rot - summary_rot)))
    if raw_rotation_residual > 1e-7:
        warnings.append(f"plan summary/raw JSON rotation residual={raw_rotation_residual:.6g}")
    record = base_record(
        task=task, episode=episode, arm=arm, strategy="top_score",
        requested_frame=requested_frame, resolved_frame=resolved_frame,
        selection_source="plan_summary.selected_candidates_by_executed_arm (actual top-score candidate)",
        candidate_idx=candidate_idx, candidate_score=float(entry["score"]),
        source_files=[top_summary_path, grasp_path, replay.path], warnings=warnings,
    )
    record.update({
        "event_index": int(event_index),
        "raw_pose": pose_payload(raw_pose),
        "raw_frame_convention": "legacy executed Top-score kept AnyGrasp raw local axes; local +X is approach",
        "canonical_pose": pose_payload(canonical_pose),
        "orientation_remap": {
            "name": "missing_in_legacy_top_score__audit_reconstruction_only",
            "matrix_right_multiply": RAW_TO_CANONICAL.tolist(),
            "mapping": {"canonical_x": "-raw_z", "canonical_y": "+raw_y", "canonical_z": "+raw_x"},
            "rotation_change_deg": rotation_distance_deg(raw_world_rot, pose_matrix(canonical_pose)[:3, :3]),
        },
        "selection_pose": pose_payload(canonical_pose),
        "selection_metrics": {
            "ranking_strategy": "top_score",
            "anygrasp_score_raw": float(entry["score"]),
            "human_orientation_used_for_ranking": False,
        },
        "planner_target": {
            "status": "legacy actual semantics plus canonical audit reconstruction",
            "legacy_actual_world_pose": pose_payload(legacy_target),
            "canonical_rebuild_world_pose": pose_payload(canonical_target),
            "legacy_vs_canonical_position_distance_m": float(np.linalg.norm(legacy_target[:3] - canonical_target[:3])),
            "legacy_vs_canonical_rotation_deg": rotation_distance_deg(
                pose_matrix(legacy_target)[:3, :3], pose_matrix(canonical_target)[:3, :3]
            ),
            "legacy_pregrasp_world_pose": pose_payload(legacy_pregrasp),
            "canonical_pregrasp_world_pose": pose_payload(canonical_pregrasp),
            "legacy_ik_chain": ik_target_trace(legacy_target, arm, calibration),
            "canonical_ik_chain": ik_target_trace(canonical_target, arm, calibration),
        },
        "offsets": {"local_x_m": 0.0, "local_y_m": 0.0, "local_z_m": local_z_offset, "retreat_m": 0.0, "pregrasp_m": float(approach_offset_m) if event_index == 0 else 0.0},
        "tcp_semantics": "legacy: raw AnyGrasp frame + local-Z offset; canonical rebuild: robot/replay frame + local-Z offset",
        "gripper_width_m": float(entry.get("width_m", raw.get("width", 0.08))),
        "gripper_depth_m": float(entry.get("depth_m", raw.get("depth", 0.04))),
        "raw_summary_rotation_max_abs_residual": raw_rotation_residual,
    })
    return record


def oursv2_record(
    *, task: str, episode: int, arm: str, event_index: int, requested_frame: int,
    ours_summary: Mapping[str, Any], ours_summary_path: Path, hand_data: Mapping[str, np.ndarray],
    hand_path: Path, replay: ReplayData, calibration: Mapping[str, Any], approach_offset_m: float,
    execution_complete: bool,
) -> Optional[Dict[str, Any]]:
    entries = selected_entries(ours_summary, arm)
    if event_index >= len(entries):
        return None
    planner_entry = entries[event_index]
    raw_hand_pose = compute_hand_world_pose(hand_data, replay, arm, requested_frame)
    if raw_hand_pose is None:
        return None
    selection_pose = raw_hand_pose.copy()
    orientation_source = str(ours_summary.get("human_replay_action_orientation_source", "grasp"))
    if event_index > 0 and orientation_source == "grasp":
        grasp_frames = [int(entry["source_frame"]) for entry in entries]
        if grasp_frames:
            grasp_pose = compute_hand_world_pose(hand_data, replay, arm, grasp_frames[0])
            if grasp_pose is not None:
                selection_pose[3:] = grasp_pose[3:]
    planner_target = np.asarray(planner_entry["pose_world_wxyz"], dtype=np.float64).reshape(7)
    retreat_m = float(ours_summary.get("human_replay_target_retreat_m", 0.0))
    expected_after_retreat = shift_pose_local(selection_pose, "local_z", -retreat_m)
    task_adjustment = planner_target[:3] - expected_after_retreat[:3]
    pregrasp = shift_pose_local(planner_target, "local_z", -float(approach_offset_m)) if event_index == 0 else None
    warnings: List[str] = []
    if not execution_complete:
        warnings.append("OursV2 final plan_summary.json missing; human replay target exists but execution was incomplete")
    record = base_record(
        task=task, episode=episode, arm=arm, strategy="oursv2",
        requested_frame=requested_frame, resolved_frame=requested_frame,
        selection_source="synthetic hand-retarget target; no AnyGrasp candidate ranking",
        candidate_idx=None, candidate_score=None,
        source_files=[ours_summary_path, hand_path, replay.path], warnings=warnings,
    )
    record.update({
        "event_index": int(event_index),
        "raw_pose": pose_payload(raw_hand_pose),
        "raw_frame_convention": "HaMeR/D435 hand gripper: local +Y opening, local +Z approach",
        "canonical_pose": pose_payload(selection_pose),
        "orientation_remap": {
            "name": "identity_hand_robot_replay_frame",
            "matrix_right_multiply": np.eye(3).tolist(),
            "action_orientation_source": planner_entry.get("human_replay_orientation_source"),
            "rotation_change_deg": rotation_distance_deg(
                pose_matrix(raw_hand_pose)[:3, :3], pose_matrix(selection_pose)[:3, :3]
            ),
        },
        "selection_pose": pose_payload(selection_pose),
        "planner_target": {
            "status": "target generated by plan_keyframes_human_replay.py",
            "effective_world_pose": pose_payload(planner_target),
            "after_retreat_before_task_adjustment_world_pose": pose_payload(expected_after_retreat),
            "task_adjustment_world_xyz_m": task_adjustment.tolist(),
            "pregrasp_world_pose": pose_payload(pregrasp),
            "ik_chain": ik_target_trace(planner_target, arm, calibration),
            "execution_complete": bool(execution_complete),
        },
        "offsets": {"local_x_m": 0.0, "local_y_m": 0.0, "local_z_m": 0.0, "retreat_m": retreat_m, "pregrasp_m": float(approach_offset_m) if event_index == 0 else 0.0, "task_adjustment_world_xyz_m": task_adjustment.tolist()},
        "tcp_semantics": "hand center is Selection Pose; target_retreat_m converts it to the legacy link6/planner target before IK",
        "gripper_width_m": float(planner_entry.get("width_m", 0.08)),
        "gripper_depth_m": float(planner_entry.get("depth_m", 0.04)),
    })
    return record


def project_point(point_cam: np.ndarray, camera: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    point = np.asarray(point_cam, dtype=np.float64).reshape(3)
    if not np.isfinite(point).all() or point[2] <= 1e-6:
        return None
    u = float(camera["fx"]) * point[0] / point[2] + float(camera["cx"])
    v = float(camera["fy"]) * point[1] / point[2] + float(camera["cy"])
    return int(round(u)), int(round(v))


def world_pose_to_camera(pose_wxyz: Sequence[float], head_pose_wxyz: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    world_pose = pose_matrix(pose_wxyz)
    head = pose_matrix(head_pose_wxyz)
    camera_from_world_rot = CV_TO_HEAD.T @ head[:3, :3].T
    position_cam = camera_from_world_rot @ (world_pose[:3, 3] - head[:3, 3])
    rotation_cam = camera_from_world_rot @ world_pose[:3, :3]
    return position_cam, orthonormalize(rotation_cam)


def draw_dashed_line(image: np.ndarray, a: Tuple[int, int], b: Tuple[int, int], color: Tuple[int, int, int], thickness: int = 1) -> None:
    start = np.asarray(a, dtype=np.float64)
    end = np.asarray(b, dtype=np.float64)
    length = float(np.linalg.norm(end - start))
    if length < 1.0:
        return
    direction = (end - start) / length
    for begin in np.arange(0.0, length, 9.0):
        finish = min(begin + 5.0, length)
        p0 = tuple(np.round(start + direction * begin).astype(int))
        p1 = tuple(np.round(start + direction * finish).astype(int))
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


def draw_pose(
    image: np.ndarray,
    pose_wxyz: Sequence[float],
    head_pose_wxyz: Sequence[float],
    camera: Mapping[str, Any],
    *, color: Tuple[int, int, int], label: str, width_m: float, depth_m: float,
    axis_length_m: float, forward_axis: str = "local_z", dashed: bool = False,
    draw_axes: bool = True, line_thickness: int = 2,
    marker: str = "circle", label_offset: Tuple[int, int] = (5, -5),
) -> None:
    position, rotation = world_pose_to_camera(pose_wxyz, head_pose_wxyz)
    x_axis, y_axis, z_axis = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    forward = z_axis if forward_axis == "local_z" else x_axis
    width = float(np.clip(width_m, 0.012, 0.12))
    depth = float(np.clip(depth_m, 0.015, 0.08))
    back = position - forward * min(0.018, depth * 0.7)
    lb = position + y_axis * width * 0.5
    rb = position - y_axis * width * 0.5
    points = [
        (back + y_axis * width * 0.5, back - y_axis * width * 0.5),
        (back + y_axis * width * 0.5, lb),
        (back - y_axis * width * 0.5, rb),
        (lb, lb + forward * max(0.018, depth)),
        (rb, rb + forward * max(0.018, depth)),
    ]
    for start, end in points:
        p0, p1 = project_point(start, camera), project_point(end, camera)
        if p0 is None or p1 is None:
            continue
        if dashed:
            draw_dashed_line(image, p0, p1, color, line_thickness)
        else:
            cv2.line(image, p0, p1, color, line_thickness, cv2.LINE_AA)
    origin = project_point(position, camera)
    if origin is None:
        return
    ox, oy = origin
    if marker == "square":
        cv2.rectangle(image, (ox - 5, oy - 5), (ox + 5, oy + 5), color, 2, cv2.LINE_AA)
    elif marker == "diamond":
        points_2d = np.asarray([[ox, oy - 7], [ox + 7, oy], [ox, oy + 7], [ox - 7, oy]], dtype=np.int32)
        cv2.polylines(image, [points_2d], True, color, 2, cv2.LINE_AA)
    elif marker == "triangle":
        points_2d = np.asarray([[ox, oy - 7], [ox + 7, oy + 6], [ox - 7, oy + 6]], dtype=np.int32)
        cv2.polylines(image, [points_2d], True, color, 2, cv2.LINE_AA)
    elif marker == "cross":
        cv2.line(image, (ox - 6, oy - 6), (ox + 6, oy + 6), color, 2, cv2.LINE_AA)
        cv2.line(image, (ox - 6, oy + 6), (ox + 6, oy - 6), color, 2, cv2.LINE_AA)
    else:
        cv2.circle(image, origin, 4, color, -1, cv2.LINE_AA)
    if draw_axes:
        for axis, axis_name in ((x_axis, "x"), (y_axis, "y"), (z_axis, "z")):
            endpoint = project_point(position + axis * float(axis_length_m), camera)
            if endpoint is not None:
                cv2.arrowedLine(image, origin, endpoint, AXIS_COLORS[axis_name], 1, cv2.LINE_AA, tipLength=0.18)
    cv2.putText(
        image, label, (origin[0] + label_offset[0], origin[1] + label_offset[1]),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
    )


def draw_pregrasp(
    image: np.ndarray, target_pose: Sequence[float], pregrasp_pose: Optional[Sequence[float]],
    head_pose: Sequence[float], camera: Mapping[str, Any], color: Tuple[int, int, int], label: str,
) -> None:
    if pregrasp_pose is None:
        return
    target_cam, _ = world_pose_to_camera(target_pose, head_pose)
    pre_cam, _ = world_pose_to_camera(pregrasp_pose, head_pose)
    p_target, p_pre = project_point(target_cam, camera), project_point(pre_cam, camera)
    if p_target is None or p_pre is None:
        return
    draw_dashed_line(image, p_pre, p_target, color, 1)
    cv2.circle(image, p_pre, 4, color, 1, cv2.LINE_AA)
    cv2.putText(image, label, (p_pre[0] + 4, p_pre[1] + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def camera_info_for_frame(anygrasp_dir: Path, frame: int, image: np.ndarray) -> Dict[str, float]:
    try:
        payload, _ = load_anygrasp_frame(anygrasp_dir, frame)
        camera = dict(payload.get("camera", {}))
    except Exception:
        camera = {}
    height, width = image.shape[:2]
    camera.setdefault("width", width)
    camera.setdefault("height", height)
    if not all(key in camera for key in ("fx", "fy", "cx", "cy")):
        fovy = math.radians(42.499880046655484)
        focal = height / (2.0 * math.tan(fovy * 0.5))
        camera.update({"fx": focal, "fy": focal, "cx": width * 0.5, "cy": height * 0.5})
    return {key: float(camera[key]) for key in ("fx", "fy", "cx", "cy", "width", "height")}


def base_image(replay_dir: Path, frame: int, cell_width: int, cell_height: int) -> Tuple[np.ndarray, Optional[Path]]:
    paths = [
        replay_dir / "head_anygrasp_frames" / f"color_{frame:06d}.png",
        replay_dir / "head_anygrasp_frames" / f"color_{frame}.png",
    ]
    image = None
    used = None
    for path in paths:
        if path.is_file():
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                used = path
                break
    if image is None:
        image = np.full((cell_height, cell_width, 3), 245, dtype=np.uint8)
        cv2.putText(image, f"missing Foundation frame {frame}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2, cv2.LINE_AA)
    if image.shape[1] != cell_width or image.shape[0] != cell_height:
        image = cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
    return image, used


def payload_pose(payload: Optional[Mapping[str, Any]]) -> Optional[np.ndarray]:
    if not payload:
        return None
    frame = str(payload.get("frame", "world"))
    return np.asarray([*payload[f"position_{frame}_m"], *payload["quat_wxyz"]], dtype=np.float64)


def draw_selection_record(image: np.ndarray, record: Mapping[str, Any], head_pose: np.ndarray, camera: Mapping[str, Any], axis_length_m: float) -> None:
    strategy = str(record["strategy"])
    arm = str(record["arm"])[0].upper()
    selection = payload_pose(record.get("selection_pose"))
    if selection is not None:
        style = {
            "oursv2": {"dashed": False, "line_thickness": 2, "marker": "circle", "label_offset": (6, 17)},
            "orientation": {"dashed": False, "line_thickness": 5, "marker": "square", "label_offset": (6, -10)},
            "fused": {"dashed": True, "line_thickness": 2, "marker": "diamond", "label_offset": (6, 18)},
            "top_score": {"dashed": False, "line_thickness": 3, "marker": "triangle", "label_offset": (6, -10)},
        }[strategy]
        draw_pose(
            image, selection, head_pose, camera,
            color=STRATEGY_COLORS[strategy], label=f"{strategy[:3].upper()}-{arm}",
            width_m=float(record.get("gripper_width_m", 0.08)), depth_m=float(record.get("gripper_depth_m", 0.04)),
            axis_length_m=axis_length_m, **style,
        )
    if strategy == "top_score":
        raw = payload_pose(record.get("raw_pose"))
        if raw is not None:
            draw_pose(
                image, raw, head_pose, camera,
                color=STRATEGY_COLORS["top_raw"], label=f"TOP-RAW-{arm}",
                width_m=float(record.get("gripper_width_m", 0.08)), depth_m=float(record.get("gripper_depth_m", 0.04)),
                axis_length_m=axis_length_m, forward_axis="local_x", dashed=True, draw_axes=False,
                line_thickness=2, marker="cross", label_offset=(6, 18),
            )


def draw_planner_record(image: np.ndarray, record: Mapping[str, Any], head_pose: np.ndarray, camera: Mapping[str, Any], axis_length_m: float) -> None:
    strategy = str(record["strategy"])
    arm = str(record["arm"])[0].upper()
    planner = record.get("planner_target", {})
    if strategy == "top_score":
        legacy = payload_pose(planner.get("legacy_actual_world_pose"))
        canonical = payload_pose(planner.get("canonical_rebuild_world_pose"))
        if legacy is not None:
            draw_pose(
                image, legacy, head_pose, camera,
                color=STRATEGY_COLORS["top_legacy"], label=f"TOP-OLD-{arm}",
                width_m=float(record.get("gripper_width_m", 0.08)), depth_m=float(record.get("gripper_depth_m", 0.04)),
                axis_length_m=axis_length_m, forward_axis="local_z", dashed=True, draw_axes=False,
                line_thickness=2, marker="cross", label_offset=(6, 18),
            )
            draw_pregrasp(image, legacy, payload_pose(planner.get("legacy_pregrasp_world_pose")), head_pose, camera, STRATEGY_COLORS["top_legacy"], "pre-old")
        if canonical is not None:
            draw_pose(
                image, canonical, head_pose, camera,
                color=STRATEGY_COLORS["top_score"], label=f"TOP-CAN-{arm}",
                width_m=float(record.get("gripper_width_m", 0.08)), depth_m=float(record.get("gripper_depth_m", 0.04)),
                axis_length_m=axis_length_m, line_thickness=3, marker="triangle", label_offset=(6, -10),
            )
            draw_pregrasp(image, canonical, payload_pose(planner.get("canonical_pregrasp_world_pose")), head_pose, camera, STRATEGY_COLORS["top_score"], "pre-can")
        return
    target = payload_pose(planner.get("effective_world_pose"))
    if target is None:
        return
    style = {
        "oursv2": {"dashed": False, "line_thickness": 2, "marker": "circle", "label_offset": (6, 17)},
        "orientation": {"dashed": False, "line_thickness": 5, "marker": "square", "label_offset": (6, -10)},
        "fused": {"dashed": True, "line_thickness": 2, "marker": "diamond", "label_offset": (6, 18)},
    }[strategy]
    draw_pose(
        image, target, head_pose, camera,
        color=STRATEGY_COLORS[strategy], label=f"{strategy[:3].upper()}-{arm}",
        width_m=float(record.get("gripper_width_m", 0.08)), depth_m=float(record.get("gripper_depth_m", 0.04)),
        axis_length_m=axis_length_m, **style,
    )
    draw_pregrasp(image, target, payload_pose(planner.get("pregrasp_world_pose")), head_pose, camera, STRATEGY_COLORS[strategy], "pre")


def render_overlay(
    *, records: Sequence[Mapping[str, Any]], requested_frame: int, replay: ReplayData,
    replay_dir: Path, anygrasp_dir: Path, task: str, episode: int,
    output_path: Path, cell_width: int, cell_height: int, axis_length_m: float,
) -> List[Dict[str, Any]]:
    resolved_frames = sorted(
        {int(record["resolved_frame"]) for record in records},
        key=lambda frame: (0 if frame == requested_frame else 1, abs(frame - requested_frame), frame),
    )
    header_h, row_title_h = 126, 30
    canvas = np.full((header_h + 2 * (row_title_h + cell_height), max(1, len(resolved_frames)) * cell_width, 3), 250, dtype=np.uint8)
    cv2.putText(canvas, f"Selection Strategy Audit V4 | {task} id={episode} requested={requested_frame}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "OURS cyan solid | ORI magenta thick-square | FUSED yellow dashed-diamond", (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "TOP-CAN black | TOP-RAW orange dashed | TOP-OLD blue dashed | axes X red Y green Z blue", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Panel A Selection: no retreat, offset, pregrasp, or TCP compensation", (12, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Panel B Planner: historical target + canonical TOP rebuild; pregrasp path dashed", (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (40, 40, 40), 1, cv2.LINE_AA)
    frame_sources: List[Dict[str, Any]] = []
    for col, frame in enumerate(resolved_frames):
        image, image_path = base_image(replay_dir, frame, cell_width, cell_height)
        selection_image, planner_image = image.copy(), image.copy()
        camera = camera_info_for_frame(anygrasp_dir, frame, image)
        head_pose = replay.head_pose(frame)
        frame_records = [record for record in records if int(record["resolved_frame"]) == frame]
        for record in frame_records:
            draw_selection_record(selection_image, record, head_pose, camera, axis_length_m)
            draw_planner_record(planner_image, record, head_pose, camera, axis_length_m)
        delta = frame - requested_frame
        labels = sorted({f"{record['strategy']}:{str(record['arm'])[0].upper()}" for record in frame_records})
        title = f"Foundation frame={frame} requested={requested_frame} delta={delta:+d} | {' '.join(labels)}"
        x0 = col * cell_width
        cv2.rectangle(canvas, (x0, header_h), (x0 + cell_width - 1, header_h + row_title_h - 1), (35, 35, 35), -1)
        cv2.putText(canvas, "A Selection | " + title, (x0 + 7, header_h + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 255, 255), 1, cv2.LINE_AA)
        y_selection = header_h + row_title_h
        canvas[y_selection:y_selection + cell_height, x0:x0 + cell_width] = selection_image
        y_title_b = y_selection + cell_height
        cv2.rectangle(canvas, (x0, y_title_b), (x0 + cell_width - 1, y_title_b + row_title_h - 1), (35, 35, 35), -1)
        cv2.putText(canvas, "B Planner | " + title, (x0 + 7, y_title_b + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 255, 255), 1, cv2.LINE_AA)
        y_planner = y_title_b + row_title_h
        canvas[y_planner:y_planner + cell_height, x0:x0 + cell_width] = planner_image
        frame_sources.append({
            "frame": int(frame), "requested_frame": int(requested_frame), "delta_frame": int(delta),
            "foundation_image": None if image_path is None else str(image_path),
            "head_camera_pose_world_wxyz": head_pose.tolist(), "camera_intrinsics": camera,
            "strategies_and_arms": labels,
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to write {output_path}")
    return frame_sources


def rank_preview_stats(top_summary: Mapping[str, Any]) -> Tuple[int, int, int]:
    actual: Dict[Tuple[int, str], int] = {}
    for arm in ("left", "right"):
        for entry in selected_entries(top_summary, arm):
            actual[(int(entry["source_frame"]), arm)] = int(entry["candidate_idx"])
    previews: Dict[Tuple[int, str], List[Tuple[int, int]]] = {}
    for item in top_summary.get("rank_preview_images", []):
        frame, rank = int(item["frame"]), int(item["rank"])
        for arm in ("left", "right"):
            idx = item.get(f"{arm}_candidate_idx")
            if idx is not None:
                previews.setdefault((frame, arm), []).append((rank, int(idx)))
    total = rank1 = topn = 0
    for key, candidate_idx in actual.items():
        total += 1
        values = previews.get(key, [])
        rank1 += int(any(rank == 1 and idx == candidate_idx for rank, idx in values))
        topn += int(any(idx == candidate_idx for _, idx in values))
    return total, rank1, topn


def episode_paths(args: argparse.Namespace) -> List[Tuple[str, int, Path]]:
    selected_tasks = set(args.tasks)
    selected_ids = None if args.ids is None else set(args.ids)
    paths: List[Tuple[str, int, Path]] = []
    for path in args.top_score_root.glob("*/*/plan_summary.json"):
        relative = path.relative_to(args.top_score_root)
        task = relative.parts[0]
        match = re.fullmatch(r"foundation_input_(\d+)", relative.parts[1])
        if task not in selected_tasks or match is None:
            continue
        episode = int(match.group(1))
        if selected_ids is not None and episode not in selected_ids:
            continue
        paths.append((task, episode, path))
    paths.sort(key=lambda item: (TASKS.index(item[0]) if item[0] in TASKS else 999, item[1]))
    if args.max_episodes > 0:
        paths = paths[: args.max_episodes]
    return paths


def summarize_numbers(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(len(array)), "min": float(array.min()), "mean": float(array.mean()), "max": float(array.max())}


def audit_incomplete_ours(ours_root: Path) -> List[Dict[str, Any]]:
    cases = [("handover_bottle", 6), ("pnp_tray", 35)]
    output = []
    for task, episode in cases:
        directory = ours_root / task / f"foundation_input_{episode}"
        stderr = directory / "stderr.log"
        tail = ""
        if stderr.is_file():
            lines = stderr.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-12:])
        output.append({
            "task": task,
            "episode_id": episode,
            "directory": str(directory),
            "human_summary_exists": (directory / "plan_summary_human_replay.json").is_file(),
            "final_plan_summary_exists": (directory / "plan_summary.json").is_file(),
            "head_video_exists": (directory / "head_cam_plan.mp4").is_file(),
            "stderr_tail": tail,
        })
    return output


def write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    rank = report["top_score_rank_preview_comparison"]
    remap = report["top_score_canonical_remap"]
    lines = [
        "# Selection Strategy Audit V4 批量报告",
        "",
        f"- tasks: `{report['coverage']['tasks']}`",
        f"- episodes audited: `{report['coverage']['episodes_audited']}`",
        f"- keyframe comparison images: `{report['coverage']['keyframe_outputs']}`",
        f"- arm-strategy records: `{report['coverage']['arm_strategy_records']}`",
        "",
        "## Top-score 旧图片错误",
        "",
        f"- actual arm-frame pairs: `{rank['actual_pairs']}`",
        f"- old rank-1 matches: `{rank['rank1_matches']}`",
        f"- selected candidate appears in exported top-N: `{rank['selected_in_exported_topn']}`",
        f"- selected candidate absent from exported top-N: `{rank['actual_pairs'] - rank['selected_in_exported_topn']}`",
        "",
        "## Top-score canonical 重建",
        "",
        f"- remap rotation change (deg): `{remap['rotation_change_deg']}`",
        f"- legacy/canonical target position gap (m): `{remap['planner_target_position_gap_m']}`",
        f"- legacy/canonical target rotation gap (deg): `{remap['planner_target_rotation_gap_deg']}`",
        "",
        "## requested/resolved 帧",
        "",
        f"- Top-score delta histogram: `{report['frame_resolution']['top_score_delta_histogram']}`",
        f"- all-strategy delta histogram: `{report['frame_resolution']['all_strategy_delta_histogram']}`",
        "",
        "## 已知不完整 OursV2 episode",
        "",
    ]
    for item in report["known_incomplete_oursv2"]:
        lines.append(
            f"- `{item['task']}/foundation_input_{item['episode_id']}`: "
            f"human_summary={item['human_summary_exists']}, final_plan_summary={item['final_plan_summary_exists']}, head_video={item['head_video_exists']}"
        )
    lines.extend(["", "## 输入/审计失败", ""])
    if report["failures"]:
        for item in report["failures"]:
            lines.append(f"- `{item['task']}/foundation_input_{item['episode_id']}`: `{item['error']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## 缺失的 arm-strategy record", ""])
    gaps = report["record_gaps"]
    lines.append(f"- total: `{gaps['count']}`")
    lines.append(f"- by strategy/reason: `{gaps['histogram']}`")
    for item in gaps["items"][:100]:
        lines.append(
            f"- `{item['task']}/foundation_input_{item['episode_id']}` "
            f"arm=`{item['arm']}` event=`{item['event_index']}` frame=`{item['requested_frame']}` "
            f"strategy=`{item['strategy']}` reason=`{item['reason']}`"
        )
    if len(gaps["items"]) > 100:
        lines.append(f"- ... remaining `{len(gaps['items']) - 100}` entries are in audit_report.json")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    calibration = load_calibration(args.calibration_bundle)
    paths = episode_paths(args)
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output root: {args.output_root}. "
            "Choose a new --output-root (preferred) or pass --overwrite explicitly for V4-only artifacts."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    episodes_audited = 0
    keyframe_outputs = 0
    arm_strategy_records = 0
    failures: List[Dict[str, Any]] = []
    record_gaps: List[Dict[str, Any]] = []
    all_delta = Counter()
    top_delta = Counter()
    remap_angles: List[float] = []
    target_position_gaps: List[float] = []
    target_rotation_gaps: List[float] = []
    rank_total = rank1_matches = rank_topn = 0
    task_counter = Counter()

    for task, episode, top_summary_path in paths:
        preview_summary_path = args.preview_root / task / f"foundation_input_{episode}" / "summary.json"
        ours_dir = args.ours_root / task / f"foundation_input_{episode}"
        ours_summary_path = ours_dir / "plan_summary_human_replay.json"
        final_ours_path = ours_dir / "plan_summary.json"
        if not preview_summary_path.is_file() or not ours_summary_path.is_file():
            failures.append({"task": task, "episode_id": episode, "error": "missing preview or OursV2 human summary"})
            continue
        try:
            top_summary = load_json(top_summary_path)
            preview_summary = load_json(preview_summary_path)
            ours_summary = load_json(ours_summary_path)
            replay_dir = Path(top_summary["replay_dir"])
            anygrasp_dir = Path(top_summary["anygrasp_dir"])
            hand_path = Path(top_summary["hand_npz"])
            replay = ReplayData(replay_dir)
            hand_data = dict(np.load(str(hand_path), allow_pickle=True))
            effective = preview_summary.get("frame_selection", {}).get("effective_keyframes_by_arm", {})
            episode_records: Dict[int, List[Dict[str, Any]]] = {}
            for arm in ("left", "right"):
                requested_frames = [int(frame) for frame in effective.get(arm, [])]
                for event_index, requested_frame in enumerate(requested_frames):
                    if args.requested_frames is not None and requested_frame not in set(args.requested_frames):
                        continue
                    for strategy in ("orientation", "fused"):
                        record = preview_strategy_record(
                            task=task, episode=episode, arm=arm, strategy=strategy, event_index=event_index,
                            requested_frame=requested_frame, preview_summary=preview_summary,
                            anygrasp_dir=anygrasp_dir, replay=replay, calibration=calibration,
                            approach_offset_m=args.approach_offset_m, preview_summary_path=preview_summary_path,
                        )
                        if record is not None:
                            episode_records.setdefault(requested_frame, []).append(record)
                        else:
                            record_gaps.append({
                                "task": task, "episode_id": episode, "arm": arm,
                                "event_index": event_index, "requested_frame": requested_frame,
                                "strategy": strategy, "reason": "preview frame or ranked candidate missing",
                            })
                    record = top_score_record(
                        task=task, episode=episode, arm=arm, event_index=event_index,
                        requested_frame=requested_frame, top_summary=top_summary, anygrasp_dir=anygrasp_dir,
                        replay=replay, calibration=calibration, approach_offset_m=args.approach_offset_m,
                        top_summary_path=top_summary_path,
                    )
                    if record is not None:
                        episode_records.setdefault(requested_frame, []).append(record)
                    else:
                        record_gaps.append({
                            "task": task, "episode_id": episode, "arm": arm,
                            "event_index": event_index, "requested_frame": requested_frame,
                            "strategy": "top_score", "reason": "executed-arm selected candidate missing",
                        })
                    record = oursv2_record(
                        task=task, episode=episode, arm=arm, event_index=event_index,
                        requested_frame=requested_frame, ours_summary=ours_summary,
                        ours_summary_path=ours_summary_path, hand_data=hand_data, hand_path=hand_path,
                        replay=replay, calibration=calibration, approach_offset_m=args.approach_offset_m,
                        execution_complete=final_ours_path.is_file(),
                    )
                    if record is not None:
                        episode_records.setdefault(requested_frame, []).append(record)
                    else:
                        record_gaps.append({
                            "task": task, "episode_id": episode, "arm": arm,
                            "event_index": event_index, "requested_frame": requested_frame,
                            "strategy": "oursv2", "reason": "human target or hand pose missing",
                        })
            task_dir = args.output_root / task
            for requested_frame, records in sorted(episode_records.items()):
                stem = f"id{episode}_keyframe_{requested_frame:06d}"
                png_path = task_dir / f"{stem}_overlay.png"
                metadata_path = task_dir / f"{stem}_metadata.json"
                if args.overwrite or not png_path.is_file() or not metadata_path.is_file():
                    frame_sources = render_overlay(
                        records=records, requested_frame=requested_frame, replay=replay,
                        replay_dir=replay_dir, anygrasp_dir=anygrasp_dir, task=task, episode=episode,
                        output_path=png_path, cell_width=args.cell_width, cell_height=args.cell_height,
                        axis_length_m=args.axis_length_m,
                    )
                    metadata = {
                        "schema": "selection_strategy_audit_v4.keyframe.v2",
                        "read_only_audit": True,
                        "output_layout": "flat_task_directory",
                        "task": task,
                        "episode": f"foundation_input_{episode}",
                        "episode_id": int(episode),
                        "requested_frame": int(requested_frame),
                        "frame_columns": frame_sources,
                        "records": records,
                        "output_image": str(png_path),
                    }
                    write_json(metadata_path, metadata)
                keyframe_outputs += 1
                arm_strategy_records += len(records)
                for record in records:
                    delta = int(record["delta_frame"])
                    all_delta[delta] += 1
                    if record["strategy"] == "top_score":
                        top_delta[delta] += 1
                        remap_angles.append(float(record["orientation_remap"]["rotation_change_deg"]))
                        planner = record["planner_target"]
                        target_position_gaps.append(float(planner["legacy_vs_canonical_position_distance_m"]))
                        target_rotation_gaps.append(float(planner["legacy_vs_canonical_rotation_deg"]))
            total, rank1, topn = rank_preview_stats(top_summary)
            rank_total += total
            rank1_matches += rank1
            rank_topn += topn
            episodes_audited += 1
            task_counter[task] += 1
        except Exception as exc:
            failures.append({"task": task, "episode_id": episode, "error": f"{type(exc).__name__}: {exc}"})

    report = {
        "schema": "selection_strategy_audit_v4.report.v1",
        "read_only_audit": True,
        "input_roots": {
            "preview": str(args.preview_root),
            "top_score": str(args.top_score_root),
            "oursv2": str(args.ours_root),
            "calibration_bundle": str(args.calibration_bundle),
        },
        "output_root": str(args.output_root),
        "coverage": {
            "tasks": dict(sorted(task_counter.items())),
            "episodes_discovered": len(paths),
            "episodes_audited": episodes_audited,
            "keyframe_outputs": keyframe_outputs,
            "arm_strategy_records": arm_strategy_records,
        },
        "top_score_rank_preview_comparison": {
            "actual_pairs": rank_total,
            "rank1_matches": rank1_matches,
            "selected_in_exported_topn": rank_topn,
            "selected_absent_from_exported_topn": rank_total - rank_topn,
        },
        "top_score_canonical_remap": {
            "mapping": {"canonical_x": "-raw_z", "canonical_y": "+raw_y", "canonical_z": "+raw_x"},
            "matrix_right_multiply": RAW_TO_CANONICAL.tolist(),
            "rotation_change_deg": summarize_numbers(remap_angles),
            "planner_target_position_gap_m": summarize_numbers(target_position_gaps),
            "planner_target_rotation_gap_deg": summarize_numbers(target_rotation_gaps),
        },
        "frame_resolution": {
            "top_score_delta_histogram": {str(k): int(v) for k, v in sorted(top_delta.items())},
            "all_strategy_delta_histogram": {str(k): int(v) for k, v in sorted(all_delta.items())},
            "max_abs_top_score_delta": max((abs(key) for key in top_delta), default=0),
        },
        "known_incomplete_oursv2": audit_incomplete_ours(args.ours_root),
        "record_gaps": {
            "count": len(record_gaps),
            "histogram": dict(sorted(Counter(
                f"{item['strategy']}: {item['reason']}" for item in record_gaps
            ).items())),
            "items": record_gaps,
        },
        "failures": failures,
        "notes": [
            "OursV2 is a synthetic hand-retarget target and is not a fourth AnyGrasp ranking.",
            "Fused planner targets are hypothetical reconstructions because the historical planner read orientation rank1.",
            "Top-score legacy targets remain unchanged; canonical targets are audit-only reconstructions.",
        ],
    }
    write_json(args.output_root / "audit_report.json", report)
    write_report_markdown(args.output_root / "audit_report.zh.md", report)
    return report


def main() -> None:
    args = parse_args()
    report = run(args)
    print(json.dumps(finite_json(report["coverage"]), ensure_ascii=False, indent=2))
    if report["failures"]:
        print(f"[audit-warning] failures={len(report['failures'])}; see {args.output_root / 'audit_report.json'}")
    print(f"[audit-output] {args.output_root}")


if __name__ == "__main__":
    main()
