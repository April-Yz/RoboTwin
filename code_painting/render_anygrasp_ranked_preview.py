#!/usr/bin/env python3
"""Render staged AnyGrasp candidate previews with object filtering and score fusion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FALLBACK_HEAD_CAMERA_POSE_WORLD_WXYZ = [
    -0.06000501662492752,
    -0.2077939808368683,
    1.383233666419983,
    0.6120354008062044,
    -0.3541386764550369,
    0.3541382592225186,
    0.6120331358296768,
]
CV_TO_WORLD_CAMERA_PRESETS = {
    "legacy_r1": np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64),
}


@dataclass
class CandidateRecord:
    candidate_idx: int
    anygrasp_score: float
    orientation_score: float
    fused_score: float
    rotation_distance_deg: float
    raw_rotation_distance_deg: float
    translation: np.ndarray
    rotation_matrix: np.ndarray
    width: float
    depth: float
    nearest_object: str
    nearest_object_distance_m: float
    translation_world: np.ndarray


@dataclass
class SharedCandidateRecord:
    candidate_idx: int
    anygrasp_score: float
    translation: np.ndarray
    rotation_matrix: np.ndarray
    width: float
    depth: float
    nearest_object: str
    nearest_object_distance_m: float
    translation_world: np.ndarray
    distances_to_objects_world_m: Dict[str, float]


@dataclass
class HandReference:
    arm_name: str
    ref_frame: int
    position: np.ndarray
    rotation_matrix: np.ndarray
    aligned_rotation_matrix: np.ndarray
    finger_distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate AnyGrasp result images with planner-style staged filtering.")
    parser.add_argument("--anygrasp_dir", type=Path, required=True, help="Per-video AnyGrasp result dir.")
    parser.add_argument("--replay_dir", type=Path, default=None, help="Optional per-video replay dir containing multi_object_world_poses.npz.")
    parser.add_argument("--hand_npz", type=Path, required=True, help="hand_detections_<id>.npz used to distinguish left/right orientation similarity.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save annotated preview images.")
    parser.add_argument("--frames", type=int, nargs="+", required=True, help="Source frame ids, e.g. 1 21.")
    parser.add_argument("--top_k", type=int, default=20, help="How many candidates to annotate for each arm in each stage.")
    parser.add_argument("--draw_grasp_boxes", type=int, default=1, help="If 1, draw a lightweight grasp wireframe for each displayed candidate.")
    parser.add_argument("--box_thickness", type=int, default=1)
    parser.add_argument("--left_target_object", type=str, default="cup")
    parser.add_argument("--right_target_object", type=str, default="bottle")
    parser.add_argument("--anygrasp_score_weight", type=float, default=0.5)
    parser.add_argument("--orientation_score_weight", type=float, default=0.5)
    parser.add_argument("--max_rotation_distance_deg", type=float, default=90.0, help="If >= 0, drop arm-specific candidates whose hand-orientation rotation distance exceeds this threshold.")
    parser.add_argument("--draw_object_overlay", type=int, default=0, help="If 1, overlay replay object center markers and labels in staged mode.")
    parser.add_argument("--draw_hand_reference", type=int, default=1, help="If 1, overlay the reference human-hand gripper pose using a distinct color.")
    parser.add_argument("--debug_dump_object_distances", type=int, default=0, help="If 1, dump per-candidate distances to every visible object in staged mode.")
    parser.add_argument("--camera_cv_axis_mode", choices=sorted(CV_TO_WORLD_CAMERA_PRESETS.keys()), default="legacy_r1")
    parser.add_argument("--candidate_target_local_x_offset_m", type=float, default=0.0)
    parser.add_argument("--candidate_post_rot_xyz_deg", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--candidate_orientation_remap_label", type=str, default="identity")
    parser.add_argument(
        "--fallback_head_camera_pose_world_wxyz",
        type=float,
        nargs=7,
        default=DEFAULT_FALLBACK_HEAD_CAMERA_POSE_WORLD_WXYZ,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="Used only when replay export does not contain head_camera_pose_world_wxyz.",
    )
    parser.add_argument(
        "--base_image_dir",
        type=Path,
        default=None,
        help="Optional raw color image directory. If provided, use color_<frame>.png instead of AnyGrasp vis/grasp_result_<frame>.png.",
    )
    parser.add_argument(
        "--base_image_mode",
        choices=["auto", "raw", "vis"],
        default="auto",
        help="Choose raw replay image, AnyGrasp vis image, or auto preference.",
    )
    parser.add_argument("--font_scale", type=float, default=0.35)
    parser.add_argument("--line_height", type=int, default=18)
    parser.add_argument("--panel_width", type=int, default=420)
    return parser.parse_args()


def load_hand_data(hand_npz: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(hand_npz), allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def load_grasp_json(anygrasp_dir: Path, frame: int) -> Dict:
    grasp_path = anygrasp_dir / "grasps" / f"grasp_{int(frame):06d}.json"
    if not grasp_path.is_file():
        raise FileNotFoundError(f"Missing AnyGrasp JSON: {grasp_path}")
    with grasp_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_available_grasp_frames(anygrasp_dir: Path) -> List[int]:
    grasp_dir = anygrasp_dir / "grasps"
    if not grasp_dir.is_dir():
        raise FileNotFoundError(f"Missing grasp directory: {grasp_dir}")
    pattern = re.compile(r"^grasp_(\d+)\.json$")
    frames: List[int] = []
    for path in grasp_dir.iterdir():
        match = pattern.match(path.name)
        if match is None:
            continue
        frames.append(int(match.group(1)))
    if not frames:
        raise RuntimeError(f"No grasp_*.json files found in {grasp_dir}")
    return sorted(frames)


def resolve_requested_frames(requested_frames: Sequence[int], available_frames: Sequence[int]) -> List[Tuple[int, int]]:
    available = [int(v) for v in available_frames]
    resolved: List[Tuple[int, int]] = []
    for requested in [int(v) for v in requested_frames]:
        if requested >= 0:
            resolved.append((requested, requested))
            continue
        if abs(int(requested)) > len(available):
            raise IndexError(
                f"Requested relative frame {requested} is out of range for {len(available)} available grasp frames."
            )
        resolved_frame = int(available[int(requested)])
        resolved.append((requested, resolved_frame))
    return resolved


def load_base_image(
    anygrasp_dir: Path,
    base_image_dir: Optional[Path],
    frame: int,
    width: int,
    height: int,
    image_mode: str,
) -> np.ndarray:
    if image_mode in ("auto", "raw") and base_image_dir is not None:
        color_path = base_image_dir / f"color_{int(frame):06d}.png"
        if color_path.is_file():
            image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            if image is not None:
                return image
        if image_mode == "raw":
            raise FileNotFoundError(f"Missing raw base image: {color_path}")
    vis_path = anygrasp_dir / "vis" / f"grasp_result_{int(frame):06d}.png"
    if image_mode in ("auto", "vis") and vis_path.is_file():
        image = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    return np.full((height, width, 3), 255, dtype=np.uint8)


def compute_fixed_head_camera_pose_world_wxyz(args: argparse.Namespace) -> np.ndarray:
    print("[head-pose-fallback] using fixed fallback head camera pose from CLI/default constant")
    pose_world_wxyz = np.asarray(args.fallback_head_camera_pose_world_wxyz, dtype=np.float64).reshape(7).copy()
    print(
        "[head-pose-fallback] fixed head pose world wxyz="
        f"{np.round(pose_world_wxyz, 6).tolist()}"
    )
    return pose_world_wxyz


def load_replay_frame_data(
    replay_dir: Path,
    frame: int,
    fixed_camera_pose_world_wxyz: Optional[np.ndarray],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, str]:
    pose_path = replay_dir / "multi_object_world_poses.npz"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing multi_object_world_poses.npz: {pose_path}")
    data = np.load(str(pose_path), allow_pickle=True)
    frame_indices = np.asarray(data["selected_source_frame_indices"], dtype=np.int32).reshape(-1)
    matches = np.where(frame_indices == int(frame))[0]
    if matches.size == 0:
        raise KeyError(f"source_frame={frame} not found in {pose_path}")
    idx = int(matches[0])
    object_names = [str(name) for name in np.asarray(data["object_names"], dtype=object).tolist()]
    object_world_positions: Dict[str, np.ndarray] = {}
    for object_name in object_names:
        visible = bool(np.asarray(data[f"{object_name}__visible"], dtype=bool)[idx])
        if not visible:
            continue
        pose_world_wxyz = np.asarray(data[f"{object_name}__pose_world_wxyz"], dtype=np.float64)[idx]
        object_world_positions[object_name] = np.asarray(pose_world_wxyz[:3], dtype=np.float64).reshape(3)
    if "head_camera_pose_world_wxyz" in data.files:
        camera_pose_world_wxyz = np.asarray(data["head_camera_pose_world_wxyz"], dtype=np.float64)[idx]
        camera_pose_source = "replay_npz"
    else:
        if fixed_camera_pose_world_wxyz is None:
            raise RuntimeError(
                "Replay export does not contain head_camera_pose_world_wxyz and no fixed fallback pose was provided."
            )
        camera_pose_world_wxyz = np.asarray(fixed_camera_pose_world_wxyz, dtype=np.float64).reshape(7)
        camera_pose_source = "fixed_fallback"
    return object_world_positions, np.asarray(camera_pose_world_wxyz, dtype=np.float64).reshape(7), str(camera_pose_source)


def nearest_valid_hand_frame(frame: int, hand_valid: np.ndarray) -> Optional[int]:
    valid_indices = np.where(np.asarray(hand_valid, dtype=bool))[0]
    if valid_indices.size == 0:
        return None
    return int(valid_indices[np.argmin(np.abs(valid_indices - int(frame)))])


def rotation_distance_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rot_a = np.asarray(rot_a, dtype=np.float64).reshape(3, 3)
    rot_b = np.asarray(rot_b, dtype=np.float64).reshape(3, 3)
    trace_value = float(np.trace(rot_a.T @ rot_b))
    cos_theta = np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def orientation_score_from_rotation_distance(rotation_distance: float) -> float:
    return float(np.clip(1.0 - float(rotation_distance) / 180.0, 0.0, 1.0))


def orthonormalize_rotation(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    u, _, vh = np.linalg.svd(rot)
    rot_ortho = u @ vh
    if np.linalg.det(rot_ortho) < 0:
        u[:, -1] *= -1.0
        rot_ortho = u @ vh
    return rot_ortho


def pose_wxyz_to_matrix(pose_world_wxyz: np.ndarray) -> np.ndarray:
    pose_world_wxyz = np.asarray(pose_world_wxyz, dtype=np.float64).reshape(7)
    rotation = R.from_quat(
        [
            float(pose_world_wxyz[4]),
            float(pose_world_wxyz[5]),
            float(pose_world_wxyz[6]),
            float(pose_world_wxyz[3]),
        ]
    ).as_matrix()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = pose_world_wxyz[:3]
    return matrix


def resolve_orientation_remap(label: str) -> np.ndarray:
    normalized = str(label).strip().lower()
    if normalized in ("", "identity", "default", "robot_default"):
        return np.eye(3, dtype=np.float64)
    raise ValueError(
        f"Unsupported candidate_orientation_remap_label '{label}' in render_anygrasp_ranked_preview.py. "
        "This lightweight preview currently supports identity only."
    )


def camera_cv_rotation(camera_cv_axis_mode: str) -> np.ndarray:
    return np.asarray(CV_TO_WORLD_CAMERA_PRESETS[str(camera_cv_axis_mode)], dtype=np.float64).reshape(3, 3)


def world_point_to_camera(point_world: np.ndarray, camera_pose_world_wxyz: np.ndarray, camera_cv_axis_mode: str) -> np.ndarray:
    camera_world = pose_wxyz_to_matrix(camera_pose_world_wxyz)
    rot_world_from_head = np.asarray(camera_world[:3, :3], dtype=np.float64)
    point_local = rot_world_from_head.T @ (np.asarray(point_world, dtype=np.float64).reshape(3) - camera_world[:3, 3])
    return camera_cv_rotation(camera_cv_axis_mode).T @ point_local


def camera_pose_to_world_pose(
    translation_cam: np.ndarray,
    rotation_cam: np.ndarray,
    camera_pose_world_wxyz: np.ndarray,
    camera_cv_axis_mode: str,
    candidate_post_rot_matrix: np.ndarray,
    candidate_orientation_remap_matrix: np.ndarray,
    candidate_target_local_x_offset_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    translation_cam = np.asarray(translation_cam, dtype=np.float64).reshape(3)
    rotation_cam = np.asarray(rotation_cam, dtype=np.float64).reshape(3, 3)
    camera_world = pose_wxyz_to_matrix(camera_pose_world_wxyz)
    rot_world_from_head = np.asarray(camera_world[:3, :3], dtype=np.float64)
    cam_cv_to_local = camera_cv_rotation(camera_cv_axis_mode)
    remapped_rotation = orthonormalize_rotation(
        orthonormalize_rotation(rotation_cam) @ np.asarray(candidate_post_rot_matrix, dtype=np.float64).reshape(3, 3) @ np.asarray(candidate_orientation_remap_matrix, dtype=np.float64).reshape(3, 3)
    )
    pos_world = np.asarray(camera_world[:3, 3], dtype=np.float64) + rot_world_from_head @ (cam_cv_to_local @ translation_cam)
    rot_world = orthonormalize_rotation(rot_world_from_head @ cam_cv_to_local @ remapped_rotation)
    if abs(float(candidate_target_local_x_offset_m)) > 1e-12:
        pos_world = pos_world + rot_world[:, 0] * float(candidate_target_local_x_offset_m)
    return pos_world.astype(np.float64), rot_world.astype(np.float64)


def object_distances_world(
    candidate_world_position: np.ndarray,
    object_world_positions: Dict[str, np.ndarray],
) -> Tuple[str, float, Dict[str, float]]:
    pos_world = np.asarray(candidate_world_position, dtype=np.float64).reshape(3)
    best_name = ""
    best_dist = float("inf")
    per_object: Dict[str, float] = {}
    for name, point_world in object_world_positions.items():
        dist = float(np.linalg.norm(pos_world - np.asarray(point_world, dtype=np.float64).reshape(3)))
        per_object[str(name)] = dist
        if dist < best_dist:
            best_dist = dist
            best_name = str(name)
    if not best_name:
        raise RuntimeError("No visible objects found for nearest-object matching.")
    return best_name, best_dist, per_object


def project_camera_point(point_cam: np.ndarray, camera_info: Dict) -> Optional[Tuple[int, int]]:
    point_cam = np.asarray(point_cam, dtype=np.float64).reshape(3)
    z = float(point_cam[2])
    if not np.isfinite(z) or z <= 1e-6:
        return None
    fx = float(camera_info["fx"])
    fy = float(camera_info["fy"])
    cx = float(camera_info["cx"])
    cy = float(camera_info["cy"])
    u = int(round(fx * float(point_cam[0]) / z + cx))
    v = int(round(fy * float(point_cam[1]) / z + cy))
    width = int(camera_info["width"])
    height = int(camera_info["height"])
    if u < -32 or u >= width + 32 or v < -32 or v >= height + 32:
        return None
    return u, v


def object_display_color(object_name: str) -> Tuple[int, int, int]:
    name = str(object_name).strip().lower()
    if name == "cup":
        return (60, 220, 60)
    if name == "bottle":
        return (255, 200, 0)
    palette = [
        (255, 120, 0),
        (180, 80, 255),
        (0, 180, 255),
        (120, 255, 120),
    ]
    return palette[sum(ord(ch) for ch in name) % len(palette)]


def draw_tiny_label(image: np.ndarray, text: str, xy: Tuple[int, int], color: Tuple[int, int, int], font_scale: float) -> None:
    cv2.putText(
        image,
        text,
        (int(xy[0]), int(xy[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(font_scale),
        color,
        1,
        cv2.LINE_AA,
    )


def draw_warning_banner(image: np.ndarray, text: str) -> None:
    h, w = image.shape[:2]
    cv2.rectangle(image, (0, 0), (w - 1, 40), (0, 0, 180), -1, cv2.LINE_AA)
    cv2.putText(
        image,
        text,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_object_overlay(
    image: np.ndarray,
    camera_info: Dict,
    object_world_positions: Dict[str, np.ndarray],
    camera_pose_world_wxyz: np.ndarray,
    camera_cv_axis_mode: str,
) -> None:
    for object_name, point_world in sorted(object_world_positions.items()):
        point_cam = world_point_to_camera(point_world, camera_pose_world_wxyz, camera_cv_axis_mode)
        pixel = project_camera_point(point_cam, camera_info)
        if pixel is None:
            continue
        color = object_display_color(object_name)
        center = (int(pixel[0]), int(pixel[1]))
        cv2.circle(image, center, 6, color, -1, cv2.LINE_AA)
        cv2.circle(image, center, 9, (255, 255, 255), 1, cv2.LINE_AA)
        draw_tiny_label(
            image,
            str(object_name),
            (center[0] + 10, center[1] - 6),
            color,
            0.38,
        )


def draw_grasp_wireframe(
    image: np.ndarray,
    camera_info: Dict,
    translation: np.ndarray,
    rotation_matrix: np.ndarray,
    width_m: float,
    depth_m: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    x_axis = rotation_matrix[:, 0]
    y_axis = rotation_matrix[:, 1]

    grip_width = float(np.clip(width_m, 0.01, 0.12))
    grip_depth = float(np.clip(depth_m, 0.01, 0.08))
    palm_back = float(min(0.018, grip_depth * 0.7))
    finger_len = float(max(0.018, grip_depth))

    center = translation
    back_center = center - x_axis * palm_back
    left_base = center + y_axis * (grip_width * 0.5)
    right_base = center - y_axis * (grip_width * 0.5)
    left_back = back_center + y_axis * (grip_width * 0.5)
    right_back = back_center - y_axis * (grip_width * 0.5)
    left_tip = left_base + x_axis * finger_len
    right_tip = right_base + x_axis * finger_len

    segments = [
        (left_back, right_back),
        (left_back, left_base),
        (right_back, right_base),
        (left_base, left_tip),
        (right_base, right_tip),
    ]
    for start, end in segments:
        start_px = project_camera_point(start, camera_info)
        end_px = project_camera_point(end, camera_info)
        if start_px is None or end_px is None:
            continue
        cv2.line(
            image,
            (int(start_px[0]), int(start_px[1])),
            (int(end_px[0]), int(end_px[1])),
            color,
            int(thickness),
            cv2.LINE_AA,
        )


def build_object_partitioned_candidates(
    grasps: Sequence[Dict],
    object_world_positions: Dict[str, np.ndarray],
    camera_pose_world_wxyz: np.ndarray,
    camera_cv_axis_mode: str,
    candidate_post_rot_matrix: np.ndarray,
    candidate_orientation_remap_matrix: np.ndarray,
    candidate_target_local_x_offset_m: float,
) -> Tuple[List[SharedCandidateRecord], List[Dict[str, object]], Dict[str, int]]:
    all_candidates: List[SharedCandidateRecord] = []
    debug_records: List[Dict[str, object]] = []
    object_partition_counts: Dict[str, int] = {}
    for candidate_idx, grasp in enumerate(grasps):
        translation = np.asarray(grasp["translation"], dtype=np.float64).reshape(3)
        rotation_matrix = np.asarray(grasp["rotation_matrix"], dtype=np.float64).reshape(3, 3)
        translation_world, rotation_world = camera_pose_to_world_pose(
            translation_cam=translation,
            rotation_cam=rotation_matrix,
            camera_pose_world_wxyz=camera_pose_world_wxyz,
            camera_cv_axis_mode=camera_cv_axis_mode,
            candidate_post_rot_matrix=candidate_post_rot_matrix,
            candidate_orientation_remap_matrix=candidate_orientation_remap_matrix,
            candidate_target_local_x_offset_m=candidate_target_local_x_offset_m,
        )
        nearest_object, nearest_dist, distances_world = object_distances_world(
            candidate_world_position=translation_world,
            object_world_positions=object_world_positions,
        )
        candidate = SharedCandidateRecord(
            candidate_idx=int(candidate_idx),
            anygrasp_score=float(grasp["score"]),
            translation=translation,
            rotation_matrix=rotation_matrix,
            width=float(grasp.get("width", 0.0)),
            depth=float(grasp.get("depth", 0.0)),
            nearest_object=str(nearest_object),
            nearest_object_distance_m=float(nearest_dist),
            translation_world=translation_world,
            distances_to_objects_world_m={name: float(dist) for name, dist in sorted(distances_world.items())},
        )
        all_candidates.append(candidate)
        object_partition_counts[str(nearest_object)] = int(object_partition_counts.get(str(nearest_object), 0)) + 1
        debug_records.append(
            {
                "candidate_idx": int(candidate_idx),
                "candidate_translation_cam": translation.tolist(),
                "candidate_translation_world": translation_world.tolist(),
                "nearest_object": str(nearest_object),
                "nearest_object_distance_m": float(nearest_dist),
                "distances_to_objects_world_m": {name: float(dist) for name, dist in sorted(distances_world.items())},
                "object_world_positions": {name: np.asarray(point, dtype=np.float64).reshape(3).tolist() for name, point in sorted(object_world_positions.items())},
            }
        )
    return all_candidates, debug_records, object_partition_counts


def build_candidates_for_arm(
    shared_candidates: Sequence[SharedCandidateRecord],
    hand_reference: HandReference,
    target_object: str,
    anygrasp_score_weight: float,
    orientation_score_weight: float,
    max_rotation_distance_deg: float,
) -> Tuple[List[CandidateRecord], int, int]:
    ref_frame = int(hand_reference.ref_frame)
    ref_rot_raw = np.asarray(hand_reference.rotation_matrix, dtype=np.float64).reshape(3, 3)
    ref_rot = np.asarray(hand_reference.aligned_rotation_matrix, dtype=np.float64).reshape(3, 3)

    filtered_candidates: List[CandidateRecord] = []
    target_object_candidates = 0
    for candidate in shared_candidates:
        if str(candidate.nearest_object) != str(target_object):
            continue
        target_object_candidates += 1
        raw_rotation_distance = rotation_distance_deg(ref_rot_raw, candidate.rotation_matrix)
        rotation_distance = rotation_distance_deg(ref_rot, candidate.rotation_matrix)
        if float(max_rotation_distance_deg) >= 0.0 and float(rotation_distance) > float(max_rotation_distance_deg):
            continue
        orientation_score = orientation_score_from_rotation_distance(rotation_distance)
        fused_score = float(anygrasp_score_weight) * float(candidate.anygrasp_score) + float(orientation_score_weight) * orientation_score
        filtered_candidates.append(
            CandidateRecord(
                candidate_idx=int(candidate.candidate_idx),
                anygrasp_score=float(candidate.anygrasp_score),
                orientation_score=orientation_score,
                fused_score=fused_score,
                rotation_distance_deg=rotation_distance,
                raw_rotation_distance_deg=raw_rotation_distance,
                translation=np.asarray(candidate.translation, dtype=np.float64).reshape(3),
                rotation_matrix=np.asarray(candidate.rotation_matrix, dtype=np.float64).reshape(3, 3),
                width=float(candidate.width),
                depth=float(candidate.depth),
                nearest_object=str(candidate.nearest_object),
                nearest_object_distance_m=float(candidate.nearest_object_distance_m),
                translation_world=np.asarray(candidate.translation_world, dtype=np.float64).reshape(3),
            )
        )
    return filtered_candidates, int(ref_frame), int(target_object_candidates)


def build_score_ranked_candidates_for_arm(
    grasps: Sequence[Dict],
    arm_name: str,
    hand_data: Dict[str, np.ndarray],
    frame: int,
) -> Tuple[List[CandidateRecord], int]:
    hand_valid = np.asarray(hand_data[f"{arm_name}_gripper_valid"], dtype=bool)
    hand_rotations = np.asarray(hand_data[f"{arm_name}_gripper_rotation_matrix"], dtype=np.float64)
    ref_frame = int(frame) if frame < hand_valid.shape[0] and bool(hand_valid[frame]) else nearest_valid_hand_frame(int(frame), hand_valid)
    if ref_frame is None:
        raise RuntimeError(f"No valid {arm_name} hand rotation exists for frame {frame}.")
    ref_rot_raw = np.asarray(hand_rotations[ref_frame], dtype=np.float64)
    ref_rot = align_hand_rotation_to_candidate_convention(ref_rot_raw)
    ranked: List[CandidateRecord] = []
    for candidate_idx, grasp in enumerate(grasps):
        translation = np.asarray(grasp["translation"], dtype=np.float64).reshape(3)
        rotation_matrix = np.asarray(grasp["rotation_matrix"], dtype=np.float64).reshape(3, 3)
        anygrasp_score = float(grasp["score"])
        raw_rotation_distance = rotation_distance_deg(ref_rot_raw, rotation_matrix)
        rotation_distance = rotation_distance_deg(ref_rot, rotation_matrix)
        orientation_score = orientation_score_from_rotation_distance(rotation_distance)
        ranked.append(
            CandidateRecord(
                candidate_idx=int(candidate_idx),
                anygrasp_score=anygrasp_score,
                orientation_score=orientation_score,
                fused_score=anygrasp_score,
                rotation_distance_deg=rotation_distance,
                raw_rotation_distance_deg=raw_rotation_distance,
                translation=translation,
                rotation_matrix=rotation_matrix,
                width=float(grasp.get("width", 0.0)),
                depth=float(grasp.get("depth", 0.0)),
                nearest_object="",
                nearest_object_distance_m=float("nan"),
                translation_world=np.full(3, np.nan, dtype=np.float64),
            )
        )
    ranked.sort(key=lambda item: (-float(item.anygrasp_score), float(item.rotation_distance_deg)))
    return ranked, int(ref_frame)


def align_hand_rotation_to_candidate_convention(rotation_matrix: np.ndarray) -> np.ndarray:
    hand_rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    return orthonormalize_rotation(
        np.column_stack(
            [
                hand_rotation[:, 2],
                hand_rotation[:, 1],
                -hand_rotation[:, 0],
            ]
        )
    )


def resolve_hand_reference(
    hand_data: Dict[str, np.ndarray],
    arm_name: str,
    frame: int,
) -> HandReference:
    hand_valid = np.asarray(hand_data[f"{arm_name}_gripper_valid"], dtype=bool)
    ref_frame = int(frame) if frame < hand_valid.shape[0] and bool(hand_valid[frame]) else nearest_valid_hand_frame(int(frame), hand_valid)
    if ref_frame is None:
        raise RuntimeError(f"No valid {arm_name} hand pose exists for frame {frame}.")
    position = np.asarray(hand_data[f"{arm_name}_gripper_position"], dtype=np.float64)[ref_frame].reshape(3)
    rotation_matrix = np.asarray(hand_data[f"{arm_name}_gripper_rotation_matrix"], dtype=np.float64)[ref_frame].reshape(3, 3)
    finger_distance = float(np.asarray(hand_data[f"{arm_name}_gripper_finger_distance"], dtype=np.float64)[ref_frame])
    return HandReference(
        arm_name=str(arm_name),
        ref_frame=int(ref_frame),
        position=position,
        rotation_matrix=rotation_matrix,
        aligned_rotation_matrix=align_hand_rotation_to_candidate_convention(rotation_matrix),
        finger_distance=finger_distance,
    )


def draw_hand_reference(
    image: np.ndarray,
    camera_info: Dict,
    hand_reference: HandReference,
    color: Tuple[int, int, int],
) -> None:
    draw_grasp_wireframe(
        image=image,
        camera_info=camera_info,
        translation=np.asarray(hand_reference.position, dtype=np.float64).reshape(3),
        rotation_matrix=np.asarray(hand_reference.aligned_rotation_matrix, dtype=np.float64).reshape(3, 3),
        width_m=float(np.clip(hand_reference.finger_distance, 0.01, 0.12)),
        depth_m=0.035,
        color=color,
        thickness=2,
    )
    center_px = project_camera_point(np.asarray(hand_reference.position, dtype=np.float64).reshape(3), camera_info)
    if center_px is None:
        return
    cv2.circle(image, (int(center_px[0]), int(center_px[1])), 5, color, -1, cv2.LINE_AA)
    draw_tiny_label(
        image,
        f"{hand_reference.arm_name[0].upper()}H",
        (int(center_px[0]) + 6, int(center_px[1]) - 6),
        color,
        0.42,
    )


def annotate_arm_preview(
    base_image: np.ndarray,
    arm_name: str,
    ranked: Sequence[CandidateRecord],
    hand_reference: Optional[HandReference],
    camera_info: Dict,
    object_world_positions: Optional[Dict[str, np.ndarray]],
    camera_pose_world_wxyz: Optional[np.ndarray],
    camera_cv_axis_mode: str,
    top_k: int,
    draw_grasp_boxes: bool,
    box_thickness: int,
    font_scale: float,
    panel_width: int,
    line_height: int,
    stage_title: str,
    formula_lines: Sequence[str],
    warning_text: Optional[str],
    line_builder,
) -> np.ndarray:
    title = "LEFT" if arm_name == "left" else "RIGHT"
    box_color = (255, 100, 0) if arm_name == "left" else (0, 140, 255)
    hand_color = (60, 255, 120) if arm_name == "left" else (255, 80, 180)
    image = base_image.copy()
    if hand_reference is not None:
        draw_hand_reference(
            image=image,
            camera_info=camera_info,
            hand_reference=hand_reference,
            color=hand_color,
        )
    if warning_text:
        draw_warning_banner(image, str(warning_text))
    if object_world_positions is not None and camera_pose_world_wxyz is not None:
        draw_object_overlay(
            image=image,
            camera_info=camera_info,
            object_world_positions=object_world_positions,
            camera_pose_world_wxyz=camera_pose_world_wxyz,
            camera_cv_axis_mode=camera_cv_axis_mode,
        )
    panel = np.full((image.shape[0], panel_width, 3), 255, dtype=np.uint8)
    draw_tiny_label(panel, f"{title} {stage_title}", (10, 24), (0, 0, 0), 0.55)
    if hand_reference is not None:
        draw_tiny_label(panel, f"hand ref idx={hand_reference.ref_frame}", (10, 40), hand_color, 0.36)
    if warning_text:
        draw_tiny_label(panel, str(warning_text), (10, 56), (0, 0, 200), 0.36)
    for formula_idx, formula_line in enumerate(list(formula_lines)[:3], start=0):
        draw_tiny_label(panel, formula_line, (10, 74 + formula_idx * 16), (0, 0, 0), 0.34)

    capped = list(ranked if int(top_k) < 0 else ranked[: max(int(top_k), 0)])
    for item in capped:
        if bool(draw_grasp_boxes):
            draw_grasp_wireframe(
                image=image,
                camera_info=camera_info,
                translation=item.translation,
                rotation_matrix=item.rotation_matrix,
                width_m=float(item.width),
                depth_m=float(item.depth),
                color=box_color,
                thickness=int(box_thickness),
            )
        pixel = project_camera_point(item.translation, camera_info)
        if pixel is None:
            continue
        offset = (4, -4) if arm_name == "left" else (4, 10)
        draw_tiny_label(
            image,
            str(int(item.candidate_idx)),
            (pixel[0] + offset[0], pixel[1] + offset[1]),
            (0, 0, 0),
            font_scale,
        )

    for row_idx, item in enumerate(capped, start=0):
        y = 128 + row_idx * int(line_height)
        if y >= panel.shape[0] - 8:
            break
        draw_tiny_label(panel, line_builder(row_idx, item), (10, y), (0, 0, 0), 0.36)

    return np.concatenate([image, panel], axis=1)


def build_combined_image(left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
    return np.concatenate([left_image, right_image], axis=0)


def orientation_line(row_idx: int, item: CandidateRecord) -> str:
    return (
        f"#{row_idx + 1:02d} "
        f"idx={item.candidate_idx:02d} "
        f"1-{item.rotation_distance_deg:.1f}/180="
        f"{item.orientation_score:.3f} "
        f"raw={item.raw_rotation_distance_deg:.1f}"
    )


def score_line(row_idx: int, item: CandidateRecord) -> str:
    return (
        f"#{row_idx + 1:02d} "
        f"idx={item.candidate_idx:02d} "
        f"s={item.anygrasp_score:.3f} "
        f"rot={item.rotation_distance_deg:.1f}"
    )


def make_fused_line_builder(anygrasp_weight: float, orientation_weight: float):
    def fused_line(row_idx: int, item: CandidateRecord) -> str:
        return (
            f"#{row_idx + 1:02d} "
            f"idx={item.candidate_idx:02d} "
            f"{item.anygrasp_score:.3f}*{anygrasp_weight:.2f} + "
            f"{item.orientation_score:.3f}*{orientation_weight:.2f} = "
            f"{item.fused_score:.3f} "
            f"(raw={item.raw_rotation_distance_deg:.1f})"
        )

    return fused_line


def orientation_formula_lines(max_rotation_distance_deg: float) -> List[str]:
    lines = [
        "ori = max(0, 1 - aligned_rot/180)",
        "raw_rot shown per row",
    ]
    if float(max_rotation_distance_deg) >= 0.0:
        lines.append(f"keep if rot <= {float(max_rotation_distance_deg):.1f} deg")
    return lines


def fused_formula_lines(anygrasp_weight: float, orientation_weight: float, max_rotation_distance_deg: float) -> List[str]:
    lines = [
        f"total = anygrasp*{float(anygrasp_weight):.2f} + ori*{float(orientation_weight):.2f}",
        "ori = max(0, 1 - aligned_rot/180)",
    ]
    if float(max_rotation_distance_deg) >= 0.0:
        lines.append(f"keep if rot <= {float(max_rotation_distance_deg):.1f} deg")
    return lines


def summarize_top_candidates(ranked: Sequence[CandidateRecord], top_n: int) -> List[Dict[str, object]]:
    capped = list(ranked if int(top_n) < 0 else ranked[: max(int(top_n), 0)])
    summary: List[Dict[str, object]] = []
    for item in capped:
        summary.append(
            {
                "candidate_idx": int(item.candidate_idx),
                "anygrasp_score": float(item.anygrasp_score),
                "orientation_score": float(item.orientation_score),
                "fused_score": float(item.fused_score),
                "rotation_distance_deg": float(item.rotation_distance_deg),
                "raw_rotation_distance_deg": float(item.raw_rotation_distance_deg),
                "nearest_object": str(item.nearest_object),
                "nearest_object_distance_m": float(item.nearest_object_distance_m),
                "translation_cam": np.asarray(item.translation, dtype=np.float64).reshape(3).tolist(),
                "translation_world": np.asarray(item.translation_world, dtype=np.float64).reshape(3).tolist(),
                "rotation_matrix": np.asarray(item.rotation_matrix, dtype=np.float64).reshape(3, 3).tolist(),
                "width": float(item.width),
                "depth": float(item.depth),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    args.anygrasp_dir = args.anygrasp_dir.resolve()
    args.replay_dir = None if args.replay_dir is None else args.replay_dir.resolve()
    args.hand_npz = args.hand_npz.resolve()
    args.output_dir = args.output_dir.resolve()
    args.base_image_dir = None if args.base_image_dir is None else args.base_image_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_post_rot_matrix = orthonormalize_rotation(
        R.from_euler("xyz", np.deg2rad(np.asarray(args.candidate_post_rot_xyz_deg, dtype=np.float64).reshape(3))).as_matrix()
    )
    candidate_orientation_remap_matrix = resolve_orientation_remap(args.candidate_orientation_remap_label)

    hand_data = load_hand_data(args.hand_npz)
    available_frames = list_available_grasp_frames(args.anygrasp_dir)
    resolved_frame_pairs = resolve_requested_frames(args.frames, available_frames)
    fixed_camera_pose_world_wxyz: Optional[np.ndarray] = None
    summary = []
    warnings_log: List[Dict[str, object]] = []
    for requested_frame, frame in resolved_frame_pairs:
        if int(requested_frame) != int(frame):
            print(f"[frame-resolve] requested={int(requested_frame)} resolved={int(frame)}")
            warnings_log.append(
                {
                    "type": "frame_resolve",
                    "requested_frame": int(requested_frame),
                    "resolved_frame": int(frame),
                    "message": f"requested frame {int(requested_frame)} resolved to available frame {int(frame)}",
                }
            )
        payload = load_grasp_json(args.anygrasp_dir, frame)
        camera_info = dict(payload["camera"])
        grasps = list(payload.get("grasps", []))
        base_image = load_base_image(
            anygrasp_dir=args.anygrasp_dir,
            base_image_dir=args.base_image_dir,
            frame=frame,
            width=int(camera_info["width"]),
            height=int(camera_info["height"]),
            image_mode=str(args.base_image_mode),
        )
        left_hand_reference = resolve_hand_reference(hand_data, "left", frame)
        right_hand_reference = resolve_hand_reference(hand_data, "right", frame)
        left_hand_reference_overlay = left_hand_reference if bool(args.draw_hand_reference) else None
        right_hand_reference_overlay = right_hand_reference if bool(args.draw_hand_reference) else None
        if args.replay_dir is None:
            left_score_ranked, left_ref_frame = build_score_ranked_candidates_for_arm(grasps, "left", hand_data, frame)
            right_score_ranked, right_ref_frame = build_score_ranked_candidates_for_arm(grasps, "right", hand_data, frame)
            score_image = build_combined_image(
                annotate_arm_preview(
                base_image=base_image,
                arm_name="left",
                ranked=left_score_ranked,
                    hand_reference=left_hand_reference_overlay,
                    camera_info=camera_info,
                    object_world_positions=None,
                    camera_pose_world_wxyz=None,
                    camera_cv_axis_mode=str(args.camera_cv_axis_mode),
                top_k=int(args.top_k),
                    draw_grasp_boxes=bool(args.draw_grasp_boxes),
                    box_thickness=int(args.box_thickness),
                    font_scale=float(args.font_scale),
                    panel_width=int(args.panel_width),
                    line_height=int(args.line_height),
                    stage_title="score rank",
                    formula_lines=["score rank only"],
                    warning_text=None,
                    line_builder=score_line,
                ),
                annotate_arm_preview(
                base_image=base_image,
                arm_name="right",
                ranked=right_score_ranked,
                    hand_reference=right_hand_reference_overlay,
                    camera_info=camera_info,
                    object_world_positions=None,
                    camera_pose_world_wxyz=None,
                    camera_cv_axis_mode=str(args.camera_cv_axis_mode),
                top_k=int(args.top_k),
                    draw_grasp_boxes=bool(args.draw_grasp_boxes),
                    box_thickness=int(args.box_thickness),
                    font_scale=float(args.font_scale),
                    panel_width=int(args.panel_width),
                    line_height=int(args.line_height),
                    stage_title="score rank",
                    formula_lines=["score rank only"],
                    warning_text=None,
                    line_builder=score_line,
                ),
            )
            score_path = args.output_dir / f"frame_{frame:06d}_left_right_score_rank.png"
            if not cv2.imwrite(str(score_path), score_image):
                raise RuntimeError(f"Failed to write {score_path}")
            summary.append(
                {
                    "frame": int(frame),
                    "requested_frame": int(requested_frame),
                    "mode": "score_only",
                    "left_reference_hand_frame": int(left_ref_frame),
                    "right_reference_hand_frame": int(right_ref_frame),
                    "base_image_mode": str(args.base_image_mode),
                    "base_image_dir": None if args.base_image_dir is None else str(args.base_image_dir),
                    "draw_grasp_boxes": int(args.draw_grasp_boxes),
                    "draw_hand_reference": int(args.draw_hand_reference),
                    "score_image": str(score_path),
                    "top_k": int(args.top_k),
                }
            )
            continue
        if fixed_camera_pose_world_wxyz is None:
            pose_path = args.replay_dir / "multi_object_world_poses.npz"
            if pose_path.is_file():
                data = np.load(str(pose_path), allow_pickle=True)
                if "head_camera_pose_world_wxyz" not in data.files:
                    fixed_camera_pose_world_wxyz = compute_fixed_head_camera_pose_world_wxyz(args)
        object_world_positions, camera_pose_world_wxyz, camera_pose_source = load_replay_frame_data(
            args.replay_dir,
            frame,
            fixed_camera_pose_world_wxyz,
        )

        shared_candidates, object_debug_records, object_partition_counts = build_object_partitioned_candidates(
            grasps=grasps,
            object_world_positions=object_world_positions,
            camera_pose_world_wxyz=camera_pose_world_wxyz,
            camera_cv_axis_mode=str(args.camera_cv_axis_mode),
            candidate_post_rot_matrix=candidate_post_rot_matrix,
            candidate_orientation_remap_matrix=candidate_orientation_remap_matrix,
            candidate_target_local_x_offset_m=float(args.candidate_target_local_x_offset_m),
        )
        left_object_filtered, left_ref_frame, left_target_count = build_candidates_for_arm(
            shared_candidates=shared_candidates,
            hand_reference=left_hand_reference,
            target_object=str(args.left_target_object),
            anygrasp_score_weight=float(args.anygrasp_score_weight),
            orientation_score_weight=float(args.orientation_score_weight),
            max_rotation_distance_deg=float(args.max_rotation_distance_deg),
        )
        right_object_filtered, right_ref_frame, right_target_count = build_candidates_for_arm(
            shared_candidates=shared_candidates,
            hand_reference=right_hand_reference,
            target_object=str(args.right_target_object),
            anygrasp_score_weight=float(args.anygrasp_score_weight),
            orientation_score_weight=float(args.orientation_score_weight),
            max_rotation_distance_deg=float(args.max_rotation_distance_deg),
        )
        object_count_text = " ".join(
            f"{name}={int(count)}"
            for name, count in sorted(object_partition_counts.items())
        )
        if not object_count_text:
            object_count_text = "none=0"

        print(
            f"[frame {frame:06d}][object-partition] "
            f"before_total={len(grasps)} "
            f"{object_count_text}"
        )
        print(
            f"[frame {frame:06d}][arm-map] "
            f"left<-{args.left_target_object}:{left_target_count} "
            f"right<-{args.right_target_object}:{right_target_count}"
        )
        print(
            f"[frame {frame:06d}][orientation-filter] "
            f"left_before={left_target_count} left_after={len(left_object_filtered)} "
            f"right_before={right_target_count} right_after={len(right_object_filtered)} "
            f"max_rot_deg={float(args.max_rotation_distance_deg):.1f}"
        )
        warning_messages: List[str] = []
        if len(left_object_filtered) == 0:
            warning_messages.append(f"LEFT WARNING: no candidate after select for {args.left_target_object}")
        if len(right_object_filtered) == 0:
            warning_messages.append(f"RIGHT WARNING: no candidate after select for {args.right_target_object}")
        for warning_message in warning_messages:
            print(f"[frame {frame:06d}][warning] {warning_message}")
            warnings_log.append(
                {
                    "type": "no_candidate_after_select",
                    "frame": int(frame),
                    "requested_frame": int(requested_frame),
                    "message": str(warning_message),
                }
            )
        if bool(args.debug_dump_object_distances):
            debug_payload = {
                "frame": int(frame),
                "requested_frame": int(requested_frame),
                "camera_pose_world_wxyz": np.asarray(camera_pose_world_wxyz, dtype=np.float64).tolist(),
                "camera_pose_source": str(camera_pose_source),
                "camera_cv_axis_mode": str(args.camera_cv_axis_mode),
                "candidate_target_local_x_offset_m": float(args.candidate_target_local_x_offset_m),
                "candidate_post_rot_xyz_deg": np.asarray(args.candidate_post_rot_xyz_deg, dtype=np.float64).reshape(3).tolist(),
                "candidate_orientation_remap_label": str(args.candidate_orientation_remap_label),
                "object_world_positions": {
                    name: np.asarray(point, dtype=np.float64).reshape(3).tolist()
                    for name, point in sorted(object_world_positions.items())
                },
                "object_partition_counts": {name: int(count) for name, count in sorted(object_partition_counts.items())},
                "arm_target_mapping": {
                    "left": str(args.left_target_object),
                    "right": str(args.right_target_object),
                },
                "orientation_filter": {
                    "max_rotation_distance_deg": float(args.max_rotation_distance_deg),
                    "left_before": int(left_target_count),
                    "left_after": int(len(left_object_filtered)),
                    "right_before": int(right_target_count),
                    "right_after": int(len(right_object_filtered)),
                },
                "warning_messages": [str(v) for v in warning_messages],
                "object_candidates": object_debug_records,
            }
            debug_path = args.output_dir / f"frame_{frame:06d}_object_distance_debug.json"
            with debug_path.open("w", encoding="utf-8") as f:
                json.dump(debug_payload, f, indent=2)
            object_pos_text = ", ".join(
                f"{name}:{np.asarray(point, dtype=np.float64).reshape(3).tolist()}"
                for name, point in sorted(object_world_positions.items())
            )
            print(
                f"[frame {frame:06d}][object-world] "
                f"camera_world={np.asarray(camera_pose_world_wxyz[:3], dtype=np.float64).reshape(3).tolist()} "
                f"objects={{ {object_pos_text} }}"
            )
            suspicious_object_world_warning: Optional[str] = None
            max_object_norm = max(
                float(np.linalg.norm(np.asarray(point, dtype=np.float64).reshape(3)))
                for point in object_world_positions.values()
            ) if object_world_positions else 0.0
            min_object_z = min(
                float(np.asarray(point, dtype=np.float64).reshape(3)[2])
                for point in object_world_positions.values()
            ) if object_world_positions else 0.0
            if max_object_norm > 5.0 or min_object_z < -2.0:
                suspicious_object_world_warning = (
                    f"detected suspicious object world poses: max_norm={max_object_norm:.4f}m min_z={min_object_z:.4f}m"
                )
                print(
                    f"[frame {frame:06d}][object-world-warning] "
                    f"{suspicious_object_world_warning}. "
                    f"Check whether --replay_dir matches the planner replay export."
                )
                warnings_log.append(
                    {
                        "type": "object_world_warning",
                        "frame": int(frame),
                        "requested_frame": int(requested_frame),
                        "message": str(suspicious_object_world_warning),
                    }
                )
            for record in object_debug_records:
                dist_text = ", ".join(
                    f"{name}:{dist:.4f}"
                    for name, dist in sorted(record["distances_to_objects_world_m"].items())
                )
                hand_tags: List[str] = []
                if str(record["nearest_object"]) == str(args.left_target_object):
                    hand_tags.append("left")
                if str(record["nearest_object"]) == str(args.right_target_object):
                    hand_tags.append("right")
                hand_text = ",".join(hand_tags) if hand_tags else "unassigned"
                print(
                    f"[frame {frame:06d}][cand {int(record['candidate_idx']):02d}] "
                    f"object={record['nearest_object']} "
                    f"arms={hand_text} "
                    f"gripper_world={np.asarray(record['candidate_translation_world'], dtype=np.float64).reshape(3).tolist()} "
                    f"nearest_dist={float(record['nearest_object_distance_m']):.4f} "
                    f"world_dists={{ {dist_text} }}"
                )

        left_orientation_ranked = sorted(left_object_filtered, key=lambda item: (-float(item.orientation_score), float(item.rotation_distance_deg), -float(item.anygrasp_score)))
        right_orientation_ranked = sorted(right_object_filtered, key=lambda item: (-float(item.orientation_score), float(item.rotation_distance_deg), -float(item.anygrasp_score)))

        left_fused_ranked = sorted(left_object_filtered, key=lambda item: (-float(item.fused_score), -float(item.anygrasp_score), -float(item.orientation_score)))
        right_fused_ranked = sorted(right_object_filtered, key=lambda item: (-float(item.fused_score), -float(item.anygrasp_score), -float(item.orientation_score)))

        orientation_image = build_combined_image(
            annotate_arm_preview(
                base_image=base_image,
                arm_name="left",
                ranked=left_orientation_ranked,
                hand_reference=left_hand_reference_overlay,
                camera_info=camera_info,
                object_world_positions=object_world_positions if bool(args.draw_object_overlay) else None,
                camera_pose_world_wxyz=camera_pose_world_wxyz if bool(args.draw_object_overlay) else None,
                camera_cv_axis_mode=str(args.camera_cv_axis_mode),
                top_k=int(args.top_k),
                draw_grasp_boxes=bool(args.draw_grasp_boxes),
                box_thickness=int(args.box_thickness),
                font_scale=float(args.font_scale),
                panel_width=int(args.panel_width),
                line_height=int(args.line_height),
                stage_title="orientation rank",
                formula_lines=orientation_formula_lines(float(args.max_rotation_distance_deg)),
                warning_text=None if len(left_object_filtered) > 0 else f"NO LEFT CANDIDATE ({args.left_target_object})",
                line_builder=orientation_line,
            ),
            annotate_arm_preview(
                base_image=base_image,
                arm_name="right",
                ranked=right_orientation_ranked,
                hand_reference=right_hand_reference_overlay,
                camera_info=camera_info,
                object_world_positions=object_world_positions if bool(args.draw_object_overlay) else None,
                camera_pose_world_wxyz=camera_pose_world_wxyz if bool(args.draw_object_overlay) else None,
                camera_cv_axis_mode=str(args.camera_cv_axis_mode),
                top_k=int(args.top_k),
                draw_grasp_boxes=bool(args.draw_grasp_boxes),
                box_thickness=int(args.box_thickness),
                font_scale=float(args.font_scale),
                panel_width=int(args.panel_width),
                line_height=int(args.line_height),
                stage_title="orientation rank",
                formula_lines=orientation_formula_lines(float(args.max_rotation_distance_deg)),
                warning_text=None if len(right_object_filtered) > 0 else f"NO RIGHT CANDIDATE ({args.right_target_object})",
                line_builder=orientation_line,
            ),
        )
        fused_line = make_fused_line_builder(
            anygrasp_weight=float(args.anygrasp_score_weight),
            orientation_weight=float(args.orientation_score_weight),
        )
        fused_image = build_combined_image(
            annotate_arm_preview(
                base_image=base_image,
                arm_name="left",
                ranked=left_fused_ranked,
                hand_reference=left_hand_reference_overlay,
                camera_info=camera_info,
                object_world_positions=object_world_positions if bool(args.draw_object_overlay) else None,
                camera_pose_world_wxyz=camera_pose_world_wxyz if bool(args.draw_object_overlay) else None,
                camera_cv_axis_mode=str(args.camera_cv_axis_mode),
                top_k=int(args.top_k),
                draw_grasp_boxes=bool(args.draw_grasp_boxes),
                box_thickness=int(args.box_thickness),
                font_scale=float(args.font_scale),
                panel_width=int(args.panel_width),
                line_height=int(args.line_height),
                stage_title="fused rank",
                formula_lines=fused_formula_lines(float(args.anygrasp_score_weight), float(args.orientation_score_weight), float(args.max_rotation_distance_deg)),
                warning_text=None if len(left_object_filtered) > 0 else f"NO LEFT CANDIDATE ({args.left_target_object})",
                line_builder=fused_line,
            ),
            annotate_arm_preview(
                base_image=base_image,
                arm_name="right",
                ranked=right_fused_ranked,
                hand_reference=right_hand_reference_overlay,
                camera_info=camera_info,
                object_world_positions=object_world_positions if bool(args.draw_object_overlay) else None,
                camera_pose_world_wxyz=camera_pose_world_wxyz if bool(args.draw_object_overlay) else None,
                camera_cv_axis_mode=str(args.camera_cv_axis_mode),
                top_k=int(args.top_k),
                draw_grasp_boxes=bool(args.draw_grasp_boxes),
                box_thickness=int(args.box_thickness),
                font_scale=float(args.font_scale),
                panel_width=int(args.panel_width),
                line_height=int(args.line_height),
                stage_title="fused rank",
                formula_lines=fused_formula_lines(float(args.anygrasp_score_weight), float(args.orientation_score_weight), float(args.max_rotation_distance_deg)),
                warning_text=None if len(right_object_filtered) > 0 else f"NO RIGHT CANDIDATE ({args.right_target_object})",
                line_builder=fused_line,
            ),
        )

        orientation_path = args.output_dir / f"frame_{frame:06d}_left_right_orientation_rank.png"
        fused_path = args.output_dir / f"frame_{frame:06d}_left_right_fused_rank.png"
        if not cv2.imwrite(str(orientation_path), orientation_image):
            raise RuntimeError(f"Failed to write {orientation_path}")
        if not cv2.imwrite(str(fused_path), fused_image):
            raise RuntimeError(f"Failed to write {fused_path}")

        summary.append(
            {
                "frame": int(frame),
                "requested_frame": int(requested_frame),
                "left_reference_hand_frame": int(left_ref_frame),
                "right_reference_hand_frame": int(right_ref_frame),
                "base_image_mode": str(args.base_image_mode),
                "base_image_dir": None if args.base_image_dir is None else str(args.base_image_dir),
                "draw_grasp_boxes": int(args.draw_grasp_boxes),
                "draw_hand_reference": int(args.draw_hand_reference),
                "draw_object_overlay": int(args.draw_object_overlay),
                "camera_pose_world_wxyz": np.asarray(camera_pose_world_wxyz, dtype=np.float64).tolist(),
                "camera_pose_source": str(camera_pose_source),
                "object_world_positions": {
                    name: np.asarray(point, dtype=np.float64).reshape(3).tolist()
                    for name, point in sorted(object_world_positions.items())
                },
                "warning_messages": [str(v) for v in warning_messages],
                "object_filter_counts": {
                    "before_total": int(len(grasps)),
                    "object_partition_counts": {name: int(count) for name, count in sorted(object_partition_counts.items())},
                    "max_rotation_distance_deg": float(args.max_rotation_distance_deg),
                    "left_before_orientation_filter": int(left_target_count),
                    "left_after": int(len(left_object_filtered)),
                    "right_before_orientation_filter": int(right_target_count),
                    "right_after": int(len(right_object_filtered)),
                    "left_target_object": str(args.left_target_object),
                    "right_target_object": str(args.right_target_object),
                },
                "orientation_image": str(orientation_path),
                "fused_image": str(fused_path),
                "top_k": int(args.top_k),
                "score_weights": {
                    "anygrasp": float(args.anygrasp_score_weight),
                    "orientation": float(args.orientation_score_weight),
                },
                "top_candidates": {
                    "left_orientation": summarize_top_candidates(left_orientation_ranked, int(args.top_k)),
                    "right_orientation": summarize_top_candidates(right_orientation_ranked, int(args.top_k)),
                    "left_fused": summarize_top_candidates(left_fused_ranked, int(args.top_k)),
                    "right_fused": summarize_top_candidates(right_fused_ranked, int(args.top_k)),
                },
                "object_distance_debug_path": None if not bool(args.debug_dump_object_distances) else str(args.output_dir / f"frame_{frame:06d}_object_distance_debug.json"),
            }
        )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "requested_frames": [int(v) for v in args.frames],
                "resolved_frames": [{"requested": int(req), "resolved": int(res)} for req, res in resolved_frame_pairs],
                "frames": summary,
            },
            f,
            indent=2,
        )
    warnings_path = args.output_dir / "warnings.json"
    with warnings_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "requested_frames": [int(v) for v in args.frames],
                "resolved_frames": [{"requested": int(req), "resolved": int(res)} for req, res in resolved_frame_pairs],
                "warnings": warnings_log,
            },
            f,
            indent=2,
        )
    print(f"[done] wrote {len(summary)} frame preview groups to {args.output_dir}")


if __name__ == "__main__":
    main()
