#!/usr/bin/env python3
"""Render 20260708 VR/HaMeR 3D diagnostic videos.

The scene is drawn in VR/world RUF coordinates. HaMeR 2D detections are not
treated as real world 3D points; they are visualized as rays from the best
eye-pose approximation found by the previous alignment sweep.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_VR_ROOT = "/home/zaijia001/ssd/data/piper/vr/data"
DEFAULT_HAMER_ROOT = "/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1"
DEFAULT_BESTFIT_JSON = (
    "/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/"
    "compare_bestfit_20260708/alignment_sweep_20260708.json"
)
DEFAULT_OUT_DIR = "/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708"

HAND_CONNECTION_NAMES = [
    ("wrist", "palm"),
    ("palm", "thumb_metacarpal"),
    ("thumb_metacarpal", "thumb_proximal"),
    ("thumb_proximal", "thumb_distal"),
    ("thumb_distal", "thumb_tip"),
    ("palm", "index_metacarpal"),
    ("index_metacarpal", "index_proximal"),
    ("index_proximal", "index_intermediate"),
    ("index_intermediate", "index_distal"),
    ("index_distal", "index_tip"),
    ("palm", "middle_metacarpal"),
    ("middle_metacarpal", "middle_proximal"),
    ("middle_proximal", "middle_intermediate"),
    ("middle_intermediate", "middle_distal"),
    ("middle_distal", "middle_tip"),
    ("palm", "ring_metacarpal"),
    ("ring_metacarpal", "ring_proximal"),
    ("ring_proximal", "ring_intermediate"),
    ("ring_intermediate", "ring_distal"),
    ("ring_distal", "ring_tip"),
    ("palm", "little_metacarpal"),
    ("little_metacarpal", "little_proximal"),
    ("little_proximal", "little_intermediate"),
    ("little_intermediate", "little_distal"),
    ("little_distal", "little_tip"),
]

HAMER21_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

CORE_JOINTS = {"wrist", "palm", "index_metacarpal", "middle_metacarpal", "ring_metacarpal", "little_metacarpal"}
LEFT_COLOR = (255, 220, 40)
RIGHT_COLOR = (50, 80, 255)
HAMER_LEFT = (80, 255, 160)
HAMER_RIGHT = (80, 180, 255)
PSEUDO_LEFT = (255, 255, 80)
PSEUDO_RIGHT = (120, 120, 255)
WHITE = (245, 245, 245)
GRAY = (155, 155, 155)
AXIS_X = (60, 60, 255)
AXIS_Y = (80, 220, 80)
AXIS_Z = (255, 120, 40)
POSE_COLORS = {
    "center_eye_pose": (245, 245, 245),
    "left_eye_pose": (255, 180, 60),
    "right_eye_pose": (180, 80, 255),
}


@dataclass
class ViewSpec:
    name: str
    cam_offset: np.ndarray
    up: np.ndarray


@dataclass
class SceneBounds:
    center: np.ndarray
    scale: float
    lo: np.ndarray
    hi: np.ndarray


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vr-root", default=DEFAULT_VR_ROOT)
    ap.add_argument("--hamer-root", default=DEFAULT_HAMER_ROOT)
    ap.add_argument("--bestfit-json", default=DEFAULT_BESTFIT_JSON)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--episode-substr", default="20260708")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit frames per video; 0 means full episode.")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ray-length", type=float, default=0.75)
    ap.add_argument("--trajectory-window", type=int, default=80)
    ap.add_argument("--panel-size", type=int, default=640)
    return ap.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else {}


def quat_xyzw_to_matrix(q: list[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.asarray(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-12:
        return vec * 0.0
    return vec / n


def hand_name_index(hand: dict[str, Any]) -> dict[str, int]:
    return {name: i for i, name in enumerate(hand.get("joint_names") or [])}


def hand_points(hand: dict[str, Any]) -> np.ndarray:
    pts = []
    for pose in hand.get("poses") or []:
        if len(pose) >= 3 and all(math.isfinite(float(v)) for v in pose[:3]):
            pts.append([float(v) for v in pose[:3]])
    return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 3), dtype=np.float64)


def hand_centroid(hand: dict[str, Any], joint_filter: set[str] | None = None) -> np.ndarray | None:
    pts = []
    for name, pose in zip(hand.get("joint_names") or [], hand.get("poses") or []):
        if joint_filter is not None and name not in joint_filter:
            continue
        if len(pose) >= 3 and all(math.isfinite(float(v)) for v in pose[:3]):
            pts.append([float(v) for v in pose[:3]])
    if not pts:
        return None
    return np.mean(np.asarray(pts, dtype=np.float64), axis=0)


def pose_key_from_best(pose_mode: str) -> str | None:
    return {
        "center_eye_xyz": "center_eye_pose",
        "left_eye_xyz": "left_eye_pose",
        "right_eye_xyz": "right_eye_pose",
    }.get(pose_mode)


def transform_point(point_world: np.ndarray, frame: dict[str, Any], pose_mode: str) -> np.ndarray | None:
    if pose_mode == "world_xyz":
        return point_world
    key = pose_key_from_best(pose_mode)
    pose = frame.get(key or "") or []
    if len(pose) < 7:
        return None
    t = np.asarray(pose[:3], dtype=np.float64)
    rot_eye_to_world = quat_xyzw_to_matrix(pose[3:7])
    return rot_eye_to_world.T @ (point_world - t)


def features_from_point(point: np.ndarray, model_mode: str) -> np.ndarray | None:
    if model_mode == "linear_xyz":
        return point.astype(np.float64)
    if abs(float(point[2])) < 1e-6:
        return None
    return np.asarray([float(point[0]) / float(point[2]), float(point[1]) / float(point[2])], dtype=np.float64)


def apply_bestfit(point_world: np.ndarray, frame: dict[str, Any], best: dict[str, Any]) -> np.ndarray | None:
    point = transform_point(point_world, frame, str(best["pose_mode"]))
    if point is None:
        return None
    feat = features_from_point(point, str(best["model_mode"]))
    if feat is None:
        return None
    coef = np.asarray(best["coef"], dtype=np.float64)
    return np.r_[feat, 1.0] @ coef


def camera_intrinsics(params_path: Path, width: int, height: int) -> tuple[float, float, float, float]:
    params = read_json(params_path)
    fx = float(params.get("fx", width / 2.0))
    fy = float(params.get("fy", height / 2.0))
    cx = float(params.get("cx", params.get("ppx", width / 2.0)))
    cy = float(params.get("cy", params.get("ppy", height / 2.0)))
    return fx, fy, cx, cy


def hamer_ray_from_uv(
    uv_px: np.ndarray,
    frame: dict[str, Any],
    pose_key: str,
    intrinsics: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray] | None:
    pose = frame.get(pose_key) or []
    if len(pose) < 7:
        return None
    fx, fy, cx, cy = intrinsics
    u = float(uv_px[0])
    v = float(uv_px[1])
    # RUF local camera convention: +x right, +y up, +z forward. Image y points down.
    ray_local = normalize(np.asarray([(u - cx) / fx, -(v - cy) / fy, 1.0], dtype=np.float64))
    origin = np.asarray(pose[:3], dtype=np.float64)
    rot_eye_to_world = quat_xyzw_to_matrix(pose[3:7])
    direction = normalize(rot_eye_to_world @ ray_local)
    return origin, direction


def collect_bounds(frames: list[dict[str, Any]]) -> SceneBounds:
    points: list[np.ndarray] = []
    for frame in frames:
        for side in ("left_hand", "right_hand"):
            hand = frame.get(side) or {}
            if hand.get("is_tracked"):
                pts = hand_points(hand)
                if pts.size:
                    points.append(pts)
        for key in ("center_eye_pose", "left_eye_pose", "right_eye_pose"):
            pose = frame.get(key) or []
            if len(pose) >= 3:
                points.append(np.asarray([pose[:3]], dtype=np.float64))
    if not points:
        lo = np.asarray([-0.5, -0.5, -0.5], dtype=np.float64)
        hi = np.asarray([0.5, 0.5, 0.5], dtype=np.float64)
    else:
        pts_all = np.concatenate(points, axis=0)
        lo = np.nanmin(pts_all, axis=0)
        hi = np.nanmax(pts_all, axis=0)
    center = (lo + hi) / 2.0
    span = hi - lo
    scale = float(max(np.max(span), 0.35))
    return SceneBounds(center=center, scale=scale, lo=lo, hi=hi)


def project_point(point: np.ndarray, view: ViewSpec, bounds: SceneBounds, width: int, height: int) -> tuple[int, int]:
    cam_pos = bounds.center + view.cam_offset * bounds.scale
    forward = normalize(bounds.center - cam_pos)
    right = normalize(np.cross(forward, view.up))
    if np.linalg.norm(right) < 1e-8:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    up = normalize(np.cross(right, forward))
    rel = point - bounds.center
    x = float(np.dot(rel, right))
    y = float(np.dot(rel, up))
    scale = bounds.scale * 1.25
    px = int(round(width * 0.5 + x / scale * width * 0.45))
    py = int(round(height * 0.5 - y / scale * height * 0.45))
    return px, py


def draw_line_3d(img: np.ndarray, a: np.ndarray, b: np.ndarray, view: ViewSpec, bounds: SceneBounds, color: tuple[int, int, int], thickness: int = 2, dashed: bool = False) -> None:
    pa = np.asarray(project_point(a, view, bounds, img.shape[1], img.shape[0]))
    pb = np.asarray(project_point(b, view, bounds, img.shape[1], img.shape[0]))
    if not dashed:
        cv2.line(img, tuple(pa), tuple(pb), color, thickness, cv2.LINE_AA)
        return
    segments = 14
    for i in range(segments):
        if i % 2:
            continue
        t0 = i / segments
        t1 = (i + 1) / segments
        q0 = (1 - t0) * pa + t0 * pb
        q1 = (1 - t1) * pa + t1 * pb
        cv2.line(img, tuple(np.round(q0).astype(int)), tuple(np.round(q1).astype(int)), color, thickness, cv2.LINE_AA)


def draw_text_box(img: np.ndarray, lines: list[str], scale: float = 0.52) -> None:
    if not lines:
        return
    h = int(22 * len(lines) + 12)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.56, img, 0.44, 0, img)
    y = 20
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1, cv2.LINE_AA)
        y += 22


def draw_global_axes(img: np.ndarray, view: ViewSpec, bounds: SceneBounds) -> None:
    origin = bounds.lo.copy()
    origin[1] = bounds.lo[1]
    length = bounds.scale * 0.18
    draw_line_3d(img, origin, origin + np.asarray([length, 0, 0]), view, bounds, AXIS_X, 2)
    draw_line_3d(img, origin, origin + np.asarray([0, length, 0]), view, bounds, AXIS_Y, 2)
    draw_line_3d(img, origin, origin + np.asarray([0, 0, length]), view, bounds, AXIS_Z, 2)
    for label, endpoint, color in [
        ("X right", origin + np.asarray([length, 0, 0]), AXIS_X),
        ("Y up", origin + np.asarray([0, length, 0]), AXIS_Y),
        ("Z fwd", origin + np.asarray([0, 0, length]), AXIS_Z),
    ]:
        px, py = project_point(endpoint, view, bounds, img.shape[1], img.shape[0])
        cv2.putText(img, label, (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_eye_pose(
    img: np.ndarray,
    frame: dict[str, Any],
    pose_key: str,
    view: ViewSpec,
    bounds: SceneBounds,
    intrinsics: tuple[float, float, float, float],
    width: int,
    height: int,
    highlight: bool,
) -> None:
    pose = frame.get(pose_key) or []
    if len(pose) < 7:
        return
    color = POSE_COLORS.get(pose_key, WHITE)
    origin = np.asarray(pose[:3], dtype=np.float64)
    rot = quat_xyzw_to_matrix(pose[3:7])
    axis_len = bounds.scale * (0.11 if highlight else 0.075)
    thickness = 3 if highlight else 1
    for axis_idx, axis_color in [(0, AXIS_X), (1, AXIS_Y), (2, AXIS_Z)]:
        draw_line_3d(img, origin, origin + rot[:, axis_idx] * axis_len, view, bounds, axis_color, thickness)
    fx, fy, cx, cy = intrinsics
    frustum_len = bounds.scale * (0.22 if highlight else 0.16)
    corners = []
    for u, v in [(0, 0), (width, 0), (width, height), (0, height)]:
        ray = normalize(np.asarray([(u - cx) / fx, -(v - cy) / fy, 1.0], dtype=np.float64))
        corners.append(origin + rot @ ray * frustum_len)
    for corner in corners:
        draw_line_3d(img, origin, corner, view, bounds, color, thickness)
    for a, b in zip(corners, corners[1:] + corners[:1]):
        draw_line_3d(img, a, b, view, bounds, color, thickness)
    px, py = project_point(origin, view, bounds, img.shape[1], img.shape[0])
    label = pose_key.replace("_eye_pose", "")
    cv2.putText(img, label, (px + 5, py + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_hand_3d(img: np.ndarray, hand: dict[str, Any], view: ViewSpec, bounds: SceneBounds, color: tuple[int, int, int]) -> None:
    if not hand.get("is_tracked"):
        return
    pts = hand_points(hand)
    if pts.size == 0:
        return
    idx = hand_name_index(hand)
    projected = [project_point(pt, view, bounds, img.shape[1], img.shape[0]) for pt in pts]
    for a, b in HAND_CONNECTION_NAMES:
        ia = idx.get(a)
        ib = idx.get(b)
        if ia is None or ib is None or ia >= len(projected) or ib >= len(projected):
            continue
        cv2.line(img, projected[ia], projected[ib], color, 2, cv2.LINE_AA)
    for i, point in enumerate(projected):
        cv2.circle(img, point, 3 if i not in {0, 1, 5, 10, 15, 20, 25} else 5, color, -1, cv2.LINE_AA)


def draw_trajectory(img: np.ndarray, points: list[np.ndarray], view: ViewSpec, bounds: SceneBounds, color: tuple[int, int, int]) -> None:
    if len(points) < 2:
        return
    projected = [project_point(pt, view, bounds, img.shape[1], img.shape[0]) for pt in points]
    for a, b in zip(projected, projected[1:]):
        cv2.line(img, a, b, color, 1, cv2.LINE_AA)


def draw_hamer_ray(
    img: np.ndarray,
    frame: dict[str, Any],
    uv_px: np.ndarray,
    pose_key: str,
    intrinsics: tuple[float, float, float, float],
    view: ViewSpec,
    bounds: SceneBounds,
    color: tuple[int, int, int],
    ray_length: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    ray = hamer_ray_from_uv(uv_px, frame, pose_key, intrinsics)
    if ray is None:
        return None
    origin, direction = ray
    end = origin + direction * ray_length
    draw_line_3d(img, origin, end, view, bounds, color, 2, dashed=True)
    return origin, direction


def draw_hamer_2d(img: np.ndarray, kpts: np.ndarray, color: tuple[int, int, int]) -> None:
    if kpts.ndim != 2 or kpts.shape[1] != 2 or np.allclose(kpts, 0):
        return
    pts = [(int(round(x)), int(round(y))) for x, y in kpts]
    for a, b in HAMER21_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, color, -1, cv2.LINE_AA)


def draw_bestfit_2d(img: np.ndarray, hand: dict[str, Any], frame: dict[str, Any], best: dict[str, Any], color: tuple[int, int, int]) -> None:
    if not hand.get("is_tracked"):
        return
    poses = hand.get("poses") or []
    if not poses:
        return
    pts: list[tuple[int, int]] = []
    for pose in poses:
        if len(pose) < 3:
            pts.append((-9999, -9999))
            continue
        uv = apply_bestfit(np.asarray(pose[:3], dtype=np.float64), frame, best)
        if uv is None or not np.isfinite(uv).all():
            pts.append((-9999, -9999))
            continue
        px = int(round(float(uv[0]) * img.shape[1]))
        py = int(round(float(uv[1]) * img.shape[0]))
        pts.append((px, py))
    idx = hand_name_index(hand)
    for a, b in HAND_CONNECTION_NAMES:
        ia = idx.get(a)
        ib = idx.get(b)
        if ia is None or ib is None or ia >= len(pts) or ib >= len(pts):
            continue
        pa, pb = pts[ia], pts[ib]
        if pa[0] < -img.shape[1] or pb[0] < -img.shape[1]:
            continue
        cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)
    for p in pts:
        if p[0] < -img.shape[1]:
            continue
        cv2.circle(img, p, 3, color, -1, cv2.LINE_AA)


def make_view_image(
    view: ViewSpec,
    frame: dict[str, Any],
    hamer_uvs: dict[str, np.ndarray],
    best: dict[str, Any],
    bounds: SceneBounds,
    intrinsics: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    traj_left: list[np.ndarray],
    traj_right: list[np.ndarray],
    ray_length: float,
    panel_size: int,
    text_lines: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    img = np.full((panel_size, panel_size, 3), (28, 30, 34), dtype=np.uint8)
    best_pose_key = pose_key_from_best(str(best["pose_mode"])) or "center_eye_pose"
    draw_global_axes(img, view, bounds)
    for pose_key in ("center_eye_pose", "left_eye_pose", "right_eye_pose"):
        draw_eye_pose(img, frame, pose_key, view, bounds, intrinsics, image_width, image_height, highlight=(pose_key == best_pose_key))
    draw_trajectory(img, traj_left, view, bounds, (120, 180, 30))
    draw_trajectory(img, traj_right, view, bounds, (40, 80, 180))
    draw_hand_3d(img, frame.get("left_hand") or {}, view, bounds, LEFT_COLOR)
    draw_hand_3d(img, frame.get("right_hand") or {}, view, bounds, RIGHT_COLOR)
    ray_metrics: dict[str, float] = {}
    for side, color in [("left", HAMER_LEFT), ("right", HAMER_RIGHT)]:
        if side not in hamer_uvs:
            continue
        ray = draw_hamer_ray(img, frame, hamer_uvs[side], best_pose_key, intrinsics, view, bounds, color, ray_length)
        centroid = hand_centroid(frame.get(f"{side}_hand") or {}, CORE_JOINTS)
        if ray is not None and centroid is not None:
            origin, direction = ray
            vec = centroid - origin
            t = float(np.dot(vec, direction))
            closest = origin + direction * t
            dist = float(np.linalg.norm(centroid - closest))
            angle = float(math.degrees(math.acos(np.clip(np.dot(normalize(vec), direction), -1.0, 1.0)))) if np.linalg.norm(vec) > 1e-8 else float("nan")
            ray_metrics[f"{side}_ray_dist_m"] = dist
            ray_metrics[f"{side}_ray_angle_deg"] = angle
    draw_text_box(img, [view.name] + text_lines)
    cv2.putText(img, "RUF world: X right, Y up, Z forward", (10, panel_size - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, WHITE, 1, cv2.LINE_AA)
    return img, ray_metrics


def read_frame_or_blank(cap: cv2.VideoCapture, width: int, height: int) -> tuple[bool, np.ndarray]:
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, np.zeros((height, width, 3), dtype=np.uint8)
    return True, frame


def transcode_to_vscode(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "baseline",
        "-level",
        "3.1",
        "-crf",
        "22",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def make_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter: {path}")
    return writer


def write_transcoded(tmp: Path, final: Path) -> None:
    transcode_to_vscode(tmp, final)
    tmp.unlink(missing_ok=True)


def render_episode(
    item: dict[str, Any],
    best_entry: dict[str, Any],
    vr_root: Path,
    hamer_root: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    episode = item["episode"]
    video_id = str(item["video_id"])
    best = best_entry.get("best")
    if not best:
        return {
            "episode": episode,
            "video_id": video_id,
            "status": "skipped",
            "failure_reason": best_entry.get("skip_reason") or "bestfit missing",
        }
    frames = read_json(vr_root / episode / f"{episode}.json").get("frames") or []
    npz_path = hamer_root / "output" / f"hand_detections_{video_id}.npz"
    rgb_path = Path(item["rgb_video"])
    params_path = hamer_root / "input" / f"params_{video_id}.json"
    if not frames:
        return {"episode": episode, "video_id": video_id, "status": "skipped", "failure_reason": "missing VR frames"}
    if not npz_path.exists():
        return {"episode": episode, "video_id": video_id, "status": "skipped", "failure_reason": f"missing HaMeR npz: {npz_path}"}
    if not rgb_path.exists():
        return {"episode": episode, "video_id": video_id, "status": "skipped", "failure_reason": f"missing RGB video: {rgb_path}"}
    npz = np.load(npz_path, allow_pickle=True)
    width = int(item["width"])
    height = int(item["height"])
    intr = camera_intrinsics(params_path, width, height)
    best_pose_key = pose_key_from_best(str(best["pose_mode"]))
    if best_pose_key is None:
        return {
            "episode": episode,
            "video_id": video_id,
            "status": "skipped",
            "failure_reason": f"best pose {best['pose_mode']} has no eye pose for ray back-projection",
        }
    pose_available = sum(1 for fr in frames if len(fr.get(best_pose_key) or []) >= 7)
    if pose_available == 0:
        return {"episode": episode, "video_id": video_id, "status": "skipped", "failure_reason": f"missing {best_pose_key}"}
    left_tracked = sum(bool((fr.get("left_hand") or {}).get("is_tracked")) for fr in frames)
    right_tracked = sum(bool((fr.get("right_hand") or {}).get("is_tracked")) for fr in frames)
    left_detected = int(np.asarray(npz["left_hand_detected"]).astype(bool).sum())
    right_detected = int(np.asarray(npz["right_hand_detected"]).astype(bool).sum())

    bounds = collect_bounds(frames)
    cap_rgb = cv2.VideoCapture(str(rgb_path))
    fps = float(args.fps or cap_rgb.get(cv2.CAP_PROP_FPS) or item.get("fps") or 30.0)
    n_video = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(n_video, int(item.get("frames_written") or n_video))
    if args.max_frames > 0:
        total = min(total, args.max_frames)
    if total <= 0:
        return {"episode": episode, "video_id": video_id, "status": "skipped", "failure_reason": "no renderable RGB frames"}

    ep_dir = out_dir / f"id_{video_id}_{episode}"
    outputs = {
        "high_back": ep_dir / "high_back_3d_vscode.mp4",
        "front": ep_dir / "front_3d_vscode.mp4",
        "top": ep_dir / "top_3d_vscode.mp4",
        "quad": ep_dir / "quadview_3d_vscode.mp4",
    }
    if all(path.exists() for path in outputs.values()) and not args.overwrite:
        return {
            "episode": episode,
            "video_id": video_id,
            "status": "exists",
            "outputs": {k: str(v) for k, v in outputs.items()},
            "best": best,
        }

    panel = int(args.panel_size)
    views = [
        ViewSpec("high_back third-person", np.asarray([0.0, 0.75, -1.35], dtype=np.float64), np.asarray([0.0, 1.0, 0.0], dtype=np.float64)),
        ViewSpec("front/opposite", np.asarray([0.0, 0.35, 1.35], dtype=np.float64), np.asarray([0.0, 1.0, 0.0], dtype=np.float64)),
        ViewSpec("top", np.asarray([0.0, 1.65, 0.001], dtype=np.float64), np.asarray([0.0, 0.0, 1.0], dtype=np.float64)),
    ]
    tmp_paths = {k: v.with_suffix(".tmp.mp4") for k, v in outputs.items()}
    writers = {
        "high_back": make_writer(tmp_paths["high_back"], fps, (panel, panel)),
        "front": make_writer(tmp_paths["front"], fps, (panel, panel)),
        "top": make_writer(tmp_paths["top"], fps, (panel, panel)),
        "quad": make_writer(tmp_paths["quad"], fps, (panel * 2, panel * 2)),
    }

    lag = int(best["lag"])
    traj_left: list[np.ndarray] = []
    traj_right: list[np.ndarray] = []
    ray_dists: list[float] = []
    ray_angles: list[float] = []
    hamer_ray_frames = 0
    for hamer_idx in range(total):
        ok, rgb = read_frame_or_blank(cap_rgb, width, height)
        if not ok:
            break
        vr_idx = hamer_idx - lag
        frame = frames[vr_idx] if 0 <= vr_idx < len(frames) else {}
        if frame:
            lc = hand_centroid(frame.get("left_hand") or {}, CORE_JOINTS)
            rc = hand_centroid(frame.get("right_hand") or {}, CORE_JOINTS)
            if lc is not None:
                traj_left.append(lc)
                traj_left = traj_left[-int(args.trajectory_window):]
            if rc is not None:
                traj_right.append(rc)
                traj_right = traj_right[-int(args.trajectory_window):]

        hamer_uvs: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            detected = np.asarray(npz[f"{side}_hand_detected"]).astype(bool)
            kpts = np.asarray(npz[f"{side}_kpts_2d"], dtype=np.float64)
            if 0 <= hamer_idx < len(detected) and detected[hamer_idx]:
                pts2 = kpts[hamer_idx]
                if pts2.ndim == 2 and pts2.shape[1] == 2 and not np.allclose(pts2, 0):
                    hamer_uvs[side] = np.mean(pts2, axis=0)

        text = [
            f"id={video_id} frame={hamer_idx} vr={vr_idx} lag={lag}",
            f"best={best['pose_mode']} {best['model_mode']} n={best['n']}",
            f"R2={best['r2_u']:.2f}/{best['r2_v']:.2f} RMSE={best['rmse_u_px']:.0f}/{best['rmse_v_px']:.0f}px",
        ]
        view_imgs = []
        frame_ray_metrics: list[float] = []
        frame_angle_metrics: list[float] = []
        for view in views:
            view_img, metrics = make_view_image(
                view,
                frame,
                hamer_uvs,
                best,
                bounds,
                intr,
                width,
                height,
                traj_left,
                traj_right,
                float(args.ray_length),
                panel,
                text,
            )
            for key, value in metrics.items():
                if key.endswith("_ray_dist_m") and math.isfinite(value):
                    frame_ray_metrics.append(value)
                if key.endswith("_ray_angle_deg") and math.isfinite(value):
                    frame_angle_metrics.append(value)
            view_imgs.append(view_img)
        if frame_ray_metrics:
            hamer_ray_frames += 1
            ray_dists.extend(frame_ray_metrics)
        if frame_angle_metrics:
            ray_angles.extend(frame_angle_metrics)

        for key, img in zip(("high_back", "front", "top"), view_imgs):
            writers[key].write(img)

        overlay = cv2.resize(rgb, (panel, panel), interpolation=cv2.INTER_AREA)
        if frame:
            draw_bestfit_2d(overlay, frame.get("left_hand") or {}, frame, best, PSEUDO_LEFT)
            draw_bestfit_2d(overlay, frame.get("right_hand") or {}, frame, best, PSEUDO_RIGHT)
        for side, color in [("left", HAMER_LEFT), ("right", HAMER_RIGHT)]:
            if side in hamer_uvs:
                kpts = np.asarray(npz[f"{side}_kpts_2d"], dtype=np.float64)[hamer_idx]
                kpts_scaled = kpts.copy()
                kpts_scaled[:, 0] *= panel / width
                kpts_scaled[:, 1] *= panel / height
                draw_hamer_2d(overlay, kpts_scaled, color)
        draw_text_box(
            overlay,
            [
                "RGB + HaMeR 2D + bestfit VR pseudo overlay",
                f"id={video_id} frame={hamer_idx} vr={vr_idx} {best['pose_mode']} lag={lag}",
                "HaMeR is image-space; VR pseudo is fitted, not true projection",
            ],
        )
        quad = np.vstack([np.hstack([overlay, view_imgs[0]]), np.hstack([view_imgs[1], view_imgs[2]])])
        writers["quad"].write(quad)

    cap_rgb.release()
    for writer in writers.values():
        writer.release()
    for key in outputs:
        write_transcoded(tmp_paths[key], outputs[key])

    mean_ray_dist = float(np.mean(ray_dists)) if ray_dists else None
    median_ray_dist = float(np.median(ray_dists)) if ray_dists else None
    mean_ray_angle = float(np.mean(ray_angles)) if ray_angles else None
    return {
        "episode": episode,
        "video_id": video_id,
        "status": "rendered",
        "rendered_frames": total,
        "json_frames": len(frames),
        "rgb_frames": n_video,
        "left_tracked": left_tracked,
        "right_tracked": right_tracked,
        "left_hamer_detected": left_detected,
        "right_hamer_detected": right_detected,
        "best": best,
        "best_pose_key": best_pose_key,
        "pose_available_frames": pose_available,
        "hamer_ray_frames": hamer_ray_frames,
        "mean_hamer_ray_to_vr_core_dist_m": mean_ray_dist,
        "median_hamer_ray_to_vr_core_dist_m": median_ray_dist,
        "mean_hamer_ray_angle_deg": mean_ray_angle,
        "outputs": {k: str(v) for k, v in outputs.items()},
        "notes": [
            "3D views are in VR/world RUF coordinates.",
            "HaMeR detections are drawn as rays from the best eye-pose approximation, not as true world 3D.",
            "Bestfit overlay is pseudo/fitted and should not be interpreted as calibrated projection.",
        ],
    }


def write_summary(results: list[dict[str, Any]], out_dir: Path) -> None:
    payload = {
        "description": "20260708 VR/HaMeR 3D diagnostics in VR/world RUF coordinates",
        "coordinate_system": "RUF: X right, Y up, Z forward",
        "outputs_root": str(out_dir),
        "episodes": results,
    }
    (out_dir / "summary_3d_20260708.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# 20260708 VR/HaMeR 3D Diagnostic Summary",
        "",
        "Coordinate system: VR/world RUF (`X right`, `Y up`, `Z forward`).",
        "HaMeR is rendered as image-space detections and best-eye back-projection rays; it is not treated as true world 3D.",
        "",
        "| id | episode | status | frames | best pose/model/lag | R2 u/v | RMSE px u/v | ray dist mean/median m | outputs / reason |",
        "|---:|---|---|---:|---|---:|---:|---:|---|",
    ]
    for res in results:
        best = res.get("best")
        if res.get("status") != "rendered":
            lines.append(
                f"| {res.get('video_id')} | {res.get('episode')} | {res.get('status')} | 0 | - | - | - | - | {res.get('failure_reason', '')} |"
            )
            continue
        ray_mean = res.get("mean_hamer_ray_to_vr_core_dist_m")
        ray_median = res.get("median_hamer_ray_to_vr_core_dist_m")
        ray_text = "" if ray_mean is None else f"{ray_mean:.3f}/{ray_median:.3f}"
        out = res["outputs"]
        out_text = f"`{out['quad']}`"
        lines.append(
            f"| {res['video_id']} | {res['episode']} | {res['status']} | {res['rendered_frames']} | "
            f"{best['pose_mode']} / {best['model_mode']} / {best['lag']} | "
            f"{best['r2_u']:.3f}/{best['r2_v']:.3f} | {best['rmse_u_px']:.1f}/{best['rmse_v_px']:.1f} | "
            f"{ray_text} | {out_text} |"
        )
    lines += [
        "",
        "## Interpretation Guide",
        "",
        "- Good VR world tracking: smooth hand skeletons and trajectories in high_back/front/top views.",
        "- HaMeR ray roughly points to the VR hand: small ray-to-hand distance and ray visually crosses the hand volume.",
        "- Eye-pose issue: one of left/right/center frustums consistently aligns with rays and hand motion.",
        "- Time-sync issue: best lag is not zero, especially if it clusters around -3 to -6 frames.",
        "- Screen-composite/warp issue: bestfit 2D overlay is good while 3D pinhole rays or perspective models remain inconsistent.",
        "",
        "Current expectation from the previous sweep: episode-local linear fits are more reliable than global perspective projection; this supports a user-view/screen-composited image rather than a raw calibrated passthrough camera.",
    ]
    (out_dir / "summary_3d_20260708.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    vr_root = Path(args.vr_root)
    hamer_root = Path(args.hamer_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(hamer_root / "vr_hamer_input_manifest.json")
    bestfit = read_json(Path(args.bestfit_json))
    best_by_id = {str(ep.get("video_id")): ep for ep in bestfit.get("episodes", [])}
    results: list[dict[str, Any]] = []
    for item in manifest:
        if item.get("skipped") or args.episode_substr not in str(item.get("episode", "")):
            continue
        video_id = str(item.get("video_id"))
        best_entry = best_by_id.get(video_id)
        if best_entry is None:
            res = {
                "episode": item.get("episode"),
                "video_id": video_id,
                "status": "skipped",
                "failure_reason": "bestfit entry missing",
            }
        else:
            print(f"[episode] id={video_id} {item['episode']}")
            res = render_episode(item, best_entry, vr_root, hamer_root, out_dir, args)
        results.append(res)
        print(f"  -> {res.get('status')} {res.get('failure_reason', '')}")
    write_summary(results, out_dir)
    print(f"[done] summary={out_dir / 'summary_3d_20260708.md'}")


if __name__ == "__main__":
    main()
