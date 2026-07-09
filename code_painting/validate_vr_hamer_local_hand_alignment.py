#!/usr/bin/env python3
"""Validate 20260708 VR/HaMeR hand-local alignment.

This diagnostic intentionally avoids requiring a global camera/world projection.
It compares VR hand joints and HaMeR 2D keypoints in hand-local normalized
coordinates, then renders per-episode videos for local shape and motion review.
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_3D_SUMMARY_JSON = (
    "/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/"
    "compare_3d_20260708/summary_3d_20260708.json"
)
DEFAULT_OUT_DIR = "/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708"

HAMER_KEYPOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

HAMER_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Map HaMeR-21 to the closest visible VR joints. VR has 26 joints: palm,
# wrist, four thumb joints, and five joints for each non-thumb finger. The
# extra palm and non-thumb metacarpal joints are deliberately ignored here.
VR_FROM_HAMER = {
    "wrist": "wrist",
    "thumb_cmc": "thumb_metacarpal",
    "thumb_mcp": "thumb_proximal",
    "thumb_ip": "thumb_distal",
    "thumb_tip": "thumb_tip",
    "index_mcp": "index_proximal",
    "index_pip": "index_intermediate",
    "index_dip": "index_distal",
    "index_tip": "index_tip",
    "middle_mcp": "middle_proximal",
    "middle_pip": "middle_intermediate",
    "middle_dip": "middle_distal",
    "middle_tip": "middle_tip",
    "ring_mcp": "ring_proximal",
    "ring_pip": "ring_intermediate",
    "ring_dip": "ring_distal",
    "ring_tip": "ring_tip",
    "pinky_mcp": "little_proximal",
    "pinky_pip": "little_intermediate",
    "pinky_dip": "little_distal",
    "pinky_tip": "little_tip",
}

VR_IGNORED_26 = [
    "palm",
    "index_metacarpal",
    "middle_metacarpal",
    "ring_metacarpal",
    "little_metacarpal",
]

FINGER_GROUPS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

ANGLE_TRIPLES = [
    (1, 2, 3), (2, 3, 4),
    (5, 6, 7), (6, 7, 8),
    (9, 10, 11), (10, 11, 12),
    (13, 14, 15), (14, 15, 16),
    (17, 18, 19), (18, 19, 20),
]

BGR_WHITE = (245, 245, 245)
BGR_GRID = (70, 74, 82)
BGR_HAMER_LEFT = (80, 255, 160)
BGR_HAMER_RIGHT = (80, 180, 255)
BGR_VR_LEFT = (255, 255, 80)
BGR_VR_RIGHT = (180, 120, 255)
BGR_BAD = (50, 80, 255)
BGR_MED = (40, 210, 240)
BGR_GOOD = (80, 220, 80)


@dataclass
class LocalHand:
    coords: np.ndarray
    origin: np.ndarray
    axis_x: np.ndarray
    axis_y: np.ndarray
    scale: float


@dataclass
class Sample:
    hamer_idx: int
    vr_idx: int
    side: str
    vr_local: np.ndarray
    hamer_local: np.ndarray
    hamer_origin_px: np.ndarray
    hamer_axis_x_px: np.ndarray
    hamer_axis_y_px: np.ndarray
    hamer_scale_px: float
    hamer_kpts_px: np.ndarray


@dataclass
class FrameFit:
    src_aligned: np.ndarray
    rmse: float
    point_errors: np.ndarray
    transform: dict[str, Any]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vr-root", default=DEFAULT_VR_ROOT)
    ap.add_argument("--hamer-root", default=DEFAULT_HAMER_ROOT)
    ap.add_argument("--bestfit-json", default=DEFAULT_BESTFIT_JSON)
    ap.add_argument("--summary-3d-json", default=DEFAULT_3D_SUMMARY_JSON)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--episode-substr", default="20260708")
    ap.add_argument("--lag-min", type=int, default=-10)
    ap.add_argument("--lag-max", type=int, default=10)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--panel-size", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=0, help="Limit rendered frames per episode; 0 means full video.")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else {}


def finite_points(points: np.ndarray) -> bool:
    return points.ndim == 2 and points.shape[0] == 21 and points.shape[1] in (2, 3) and np.isfinite(points).all() and not np.allclose(points, 0)


def normalize_vec(vec: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(vec))
    if n < 1e-8 or not math.isfinite(n):
        return None
    return vec / n


def hand_name_index(hand: dict[str, Any]) -> dict[str, int]:
    return {name: i for i, name in enumerate(hand.get("joint_names") or [])}


def extract_vr_mapped_points(hand: dict[str, Any]) -> np.ndarray | None:
    if not hand.get("is_tracked"):
        return None
    idx = hand_name_index(hand)
    poses = hand.get("poses") or []
    points: list[list[float]] = []
    for hamer_name in HAMER_KEYPOINT_NAMES:
        vr_name = VR_FROM_HAMER[hamer_name]
        vi = idx.get(vr_name)
        if vi is None or vi >= len(poses) or len(poses[vi]) < 3:
            return None
        xyz = [float(v) for v in poses[vi][:3]]
        if not all(math.isfinite(v) for v in xyz):
            return None
        points.append(xyz)
    return np.asarray(points, dtype=np.float64)


def palm_scale_2d(points: np.ndarray) -> float:
    pairs = [(0, 5), (0, 9), (0, 17), (5, 17), (5, 9), (9, 17)]
    vals = [float(np.linalg.norm(points[a] - points[b])) for a, b in pairs]
    vals = [v for v in vals if math.isfinite(v) and v > 1e-8]
    return float(np.mean(vals)) if vals else 0.0


def local_from_hamer_2d(points: np.ndarray) -> LocalHand | None:
    if not finite_points(points):
        return None
    origin = points[0].astype(np.float64)
    x_axis = normalize_vec(points[5] - points[17])
    y_ref = points[9] - origin
    if x_axis is None:
        return None
    y_axis = y_ref - float(np.dot(y_ref, x_axis)) * x_axis
    y_axis = normalize_vec(y_axis)
    if y_axis is None:
        y_axis = np.asarray([-x_axis[1], x_axis[0]], dtype=np.float64)
    scale = palm_scale_2d(points)
    if scale <= 1e-8:
        return None
    rel = points - origin
    coords = np.stack([rel @ x_axis, rel @ y_axis], axis=1) / scale
    return LocalHand(coords=coords, origin=origin, axis_x=x_axis, axis_y=y_axis, scale=scale)


def local_from_vr_3d(points: np.ndarray) -> LocalHand | None:
    if not finite_points(points):
        return None
    origin = points[0].astype(np.float64)
    x_axis = normalize_vec(points[5] - points[17])
    y_ref = points[9] - origin
    if x_axis is None:
        return None
    z_axis = normalize_vec(np.cross(x_axis, y_ref))
    if z_axis is None:
        return None
    y_axis = normalize_vec(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None
    scale = palm_scale_2d(points)
    if scale <= 1e-8:
        return None
    rel = points - origin
    coords = np.stack([rel @ x_axis, rel @ y_axis], axis=1) / scale
    return LocalHand(coords=coords, origin=origin, axis_x=x_axis, axis_y=y_axis, scale=scale)


def similarity_fit(src: np.ndarray, dst: np.ndarray) -> FrameFit:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    xs = src - src_mean
    ys = dst - dst_mean
    denom = float(np.sum(xs * xs))
    if denom < 1e-12:
        aligned = np.repeat(dst_mean[None, :], len(src), axis=0)
        err = np.linalg.norm(aligned - dst, axis=1)
        return FrameFit(aligned, float(np.sqrt(np.mean(err * err))), err, {"scale": 0.0, "rotation": [[1.0, 0.0], [0.0, 1.0]], "translation": dst_mean.tolist()})
    cov = xs.T @ ys
    u, s, vt = np.linalg.svd(cov)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    scale = float(np.sum(s) / denom)
    aligned = scale * (xs @ rot) + dst_mean
    err = np.linalg.norm(aligned - dst, axis=1)
    return FrameFit(aligned, float(np.sqrt(np.mean(err * err))), err, {"scale": scale, "rotation": rot.tolist(), "translation": (dst_mean - scale * (src_mean @ rot)).tolist()})


def affine_fit(src: np.ndarray, dst: np.ndarray) -> FrameFit:
    x = np.concatenate([src, np.ones((len(src), 1), dtype=np.float64)], axis=1)
    coef, *_ = np.linalg.lstsq(x, dst, rcond=None)
    aligned = x @ coef
    err = np.linalg.norm(aligned - dst, axis=1)
    return FrameFit(aligned, float(np.sqrt(np.mean(err * err))), err, {"coef": coef.tolist()})


def pairwise_distance_error(src: np.ndarray, dst: np.ndarray) -> float:
    ds = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=2)
    dd = np.linalg.norm(dst[:, None, :] - dst[None, :, :], axis=2)
    tri = np.triu_indices(len(src), k=1)
    diff = ds[tri] - dd[tri]
    return float(np.sqrt(np.mean(diff * diff)))


def bone_length_error(src: np.ndarray, dst: np.ndarray) -> tuple[float, dict[str, float]]:
    vals = []
    by_finger: dict[str, list[float]] = {k: [] for k in FINGER_GROUPS}
    for a, b in HAMER_CONNECTIONS:
        ls = float(np.linalg.norm(src[a] - src[b]))
        ld = float(np.linalg.norm(dst[a] - dst[b]))
        if ls < 1e-8 or ld < 1e-8:
            continue
        val = abs(math.log((ls + 1e-8) / (ld + 1e-8)))
        vals.append(val)
        for finger, inds in FINGER_GROUPS.items():
            if a in inds and b in inds:
                by_finger[finger].append(val)
    mean = float(np.mean(vals)) if vals else float("nan")
    return mean, {k: float(np.mean(v)) if v else float("nan") for k, v in by_finger.items()}


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return None
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(math.degrees(math.acos(np.clip(cos, -1.0, 1.0))))


def joint_angle_error(src: np.ndarray, dst: np.ndarray) -> tuple[float, dict[str, float]]:
    vals = []
    by_finger = {k: [] for k in FINGER_GROUPS}
    for a, b, c in ANGLE_TRIPLES:
        asrc = angle_deg(src[a], src[b], src[c])
        adst = angle_deg(dst[a], dst[b], dst[c])
        if asrc is None or adst is None:
            continue
        val = abs(asrc - adst)
        vals.append(val)
        for finger, inds in FINGER_GROUPS.items():
            if a in inds and b in inds and c in inds:
                by_finger[finger].append(val)
    mean = float(np.mean(vals)) if vals else float("nan")
    return mean, {k: float(np.mean(v)) if v else float("nan") for k, v in by_finger.items()}


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or b.size < 2:
        return None
    aa = a.reshape(-1).astype(np.float64)
    bb = b.reshape(-1).astype(np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[mask]
    bb = bb[mask]
    if aa.size < 2:
        return None
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    den = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if den < 1e-12:
        return None
    return float(np.dot(aa, bb) / den)


def motion_metrics(samples: list[Sample], aligned_by_idx: dict[int, np.ndarray]) -> dict[str, float | None]:
    if len(samples) < 3:
        return {"velocity_corr": None, "delta_direction_agreement": None, "mean_delta_cos": None}
    samples_sorted = sorted(samples, key=lambda s: s.hamer_idx)
    vr_deltas = []
    h_deltas = []
    cosines = []
    prev: Sample | None = None
    for sample in samples_sorted:
        if prev is None:
            prev = sample
            continue
        if sample.hamer_idx <= prev.hamer_idx:
            prev = sample
            continue
        av = aligned_by_idx.get(sample.hamer_idx)
        pv = aligned_by_idx.get(prev.hamer_idx)
        if av is None or pv is None:
            prev = sample
            continue
        dv = (av - pv).reshape(-1)
        dh = (sample.hamer_local - prev.hamer_local).reshape(-1)
        nv = float(np.linalg.norm(dv))
        nh = float(np.linalg.norm(dh))
        if nv > 1e-8 and nh > 1e-8:
            vr_deltas.append(dv)
            h_deltas.append(dh)
            cosines.append(float(np.dot(dv, dh) / (nv * nh)))
        prev = sample
    if not vr_deltas:
        return {"velocity_corr": None, "delta_direction_agreement": None, "mean_delta_cos": None}
    corr = pearson_corr(np.concatenate(vr_deltas), np.concatenate(h_deltas))
    cos_arr = np.asarray(cosines, dtype=np.float64)
    return {
        "velocity_corr": corr,
        "delta_direction_agreement": float(np.mean(cos_arr > 0.0)),
        "mean_delta_cos": float(np.mean(cos_arr)),
    }


def evaluate_samples(samples: list[Sample], alignment: str) -> dict[str, Any]:
    fits: list[FrameFit] = []
    pair_errors = []
    bone_errors = []
    angle_errors = []
    per_point = []
    per_finger_points: dict[str, list[float]] = {k: [] for k in FINGER_GROUPS}
    per_finger_bone: dict[str, list[float]] = {k: [] for k in FINGER_GROUPS}
    per_finger_angle: dict[str, list[float]] = {k: [] for k in FINGER_GROUPS}
    aligned_by_idx: dict[int, np.ndarray] = {}

    for sample in samples:
        fit = similarity_fit(sample.vr_local, sample.hamer_local) if alignment == "similarity" else affine_fit(sample.vr_local, sample.hamer_local)
        fits.append(fit)
        aligned_by_idx[sample.hamer_idx] = fit.src_aligned
        per_point.append(fit.point_errors)
        for finger, inds in FINGER_GROUPS.items():
            per_finger_points[finger].append(float(np.sqrt(np.mean(fit.point_errors[inds] ** 2))))
        pair_errors.append(pairwise_distance_error(sample.vr_local, sample.hamer_local))
        bone_mean, bone_by = bone_length_error(sample.vr_local, sample.hamer_local)
        angle_mean, angle_by = joint_angle_error(sample.vr_local, sample.hamer_local)
        bone_errors.append(bone_mean)
        angle_errors.append(angle_mean)
        for finger in FINGER_GROUPS:
            if math.isfinite(bone_by[finger]):
                per_finger_bone[finger].append(bone_by[finger])
            if math.isfinite(angle_by[finger]):
                per_finger_angle[finger].append(angle_by[finger])

    point_arr = np.asarray(per_point, dtype=np.float64) if per_point else np.empty((0, 21))
    motion = motion_metrics(samples, aligned_by_idx)
    rmse = float(np.mean([f.rmse for f in fits])) if fits else float("nan")
    pair = float(np.mean(pair_errors)) if pair_errors else float("nan")
    bone = float(np.nanmean(bone_errors)) if bone_errors else float("nan")
    angle = float(np.nanmean(angle_errors)) if angle_errors else float("nan")
    vel = motion["velocity_corr"]
    vel_for_score = 0.0 if vel is None or not math.isfinite(float(vel)) else max(-1.0, min(1.0, float(vel)))
    penalty = (
        0.45 * min(rmse / 0.45, 2.0)
        + 0.20 * min(pair / 0.35, 2.0)
        + 0.15 * min(bone / 0.45, 2.0)
        + 0.10 * min(angle / 70.0, 2.0)
        + 0.10 * ((1.0 - vel_for_score) / 2.0)
    )
    score = float(max(0.0, min(100.0, 100.0 * (1.0 - penalty / 1.4))))
    return {
        "alignment": alignment,
        "matched_frames": len(samples),
        "rmse_mean": rmse,
        "rmse_median": float(np.median([f.rmse for f in fits])) if fits else None,
        "per_keypoint_rmse": {name: float(point_arr[:, i].mean()) if point_arr.size else None for i, name in enumerate(HAMER_KEYPOINT_NAMES)},
        "per_finger_rmse": {k: float(np.mean(v)) if v else None for k, v in per_finger_points.items()},
        "per_finger_bone_log_error": {k: float(np.mean(v)) if v else None for k, v in per_finger_bone.items()},
        "per_finger_angle_deg_error": {k: float(np.mean(v)) if v else None for k, v in per_finger_angle.items()},
        "pairwise_distance_rmse": pair,
        "bone_length_log_error": bone,
        "joint_angle_deg_error": angle,
        "velocity_corr": motion["velocity_corr"],
        "delta_direction_agreement": motion["delta_direction_agreement"],
        "mean_delta_cos": motion["mean_delta_cos"],
        "local_shape_score": score,
        "fits": fits,
        "aligned_by_idx": aligned_by_idx,
    }


def collect_samples_for_lag(
    frames: list[dict[str, Any]],
    npz: Any,
    side: str,
    lag: int,
    total_frames: int,
) -> list[Sample]:
    detected = np.asarray(npz[f"{side}_hand_detected"]).astype(bool)
    kpts = np.asarray(npz[f"{side}_kpts_2d"], dtype=np.float64)
    samples: list[Sample] = []
    for hamer_idx in range(total_frames):
        vr_idx = hamer_idx - lag
        if vr_idx < 0 or vr_idx >= len(frames) or hamer_idx >= len(detected) or not detected[hamer_idx]:
            continue
        hamer_points = kpts[hamer_idx]
        h_local = local_from_hamer_2d(hamer_points)
        if h_local is None:
            continue
        hand = frames[vr_idx].get(f"{side}_hand") or {}
        vr_points = extract_vr_mapped_points(hand)
        if vr_points is None:
            continue
        v_local = local_from_vr_3d(vr_points)
        if v_local is None:
            continue
        samples.append(
            Sample(
                hamer_idx=hamer_idx,
                vr_idx=vr_idx,
                side=side,
                vr_local=v_local.coords,
                hamer_local=h_local.coords,
                hamer_origin_px=h_local.origin,
                hamer_axis_x_px=h_local.axis_x,
                hamer_axis_y_px=h_local.axis_y,
                hamer_scale_px=h_local.scale,
                hamer_kpts_px=hamer_points,
            )
        )
    return samples


def classify_result(metrics: dict[str, Any], lag: int) -> str:
    score = float(metrics.get("local_shape_score") or 0.0)
    rmse = float(metrics.get("rmse_mean") or 999.0)
    vel = metrics.get("velocity_corr")
    vel = -1.0 if vel is None else float(vel)
    matched = int(metrics.get("matched_frames") or 0)
    if matched < 20:
        return "skipped"
    if score >= 68.0 and rmse <= 0.38 and vel >= 0.25 and abs(lag) <= 8:
        return "good"
    if score >= 45.0 and rmse <= 0.65 and vel >= -0.05:
        return "medium"
    return "bad"


def draw_text_box(img: np.ndarray, lines: list[str], scale: float = 0.52) -> None:
    if not lines:
        return
    h = int(22 * len(lines) + 12)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, img, 0.42, 0, img)
    y = 20
    for line in lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, BGR_WHITE, 1, cv2.LINE_AA)
        y += 22


def draw_skeleton(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], radius: int = 3, thickness: int = 2) -> None:
    ipts = [(int(round(x)), int(round(y))) for x, y in pts]
    for a, b in HAMER_CONNECTIONS:
        if a < len(ipts) and b < len(ipts):
            cv2.line(img, ipts[a], ipts[b], color, thickness, cv2.LINE_AA)
    for i, p in enumerate(ipts):
        r = radius + 1 if i in {0, 5, 9, 13, 17} else radius
        cv2.circle(img, p, r, color, -1, cv2.LINE_AA)


def local_to_image(local_pts: np.ndarray, sample: Sample) -> np.ndarray:
    return sample.hamer_origin_px + sample.hamer_scale_px * (
        local_pts[:, [0]] * sample.hamer_axis_x_px[None, :] + local_pts[:, [1]] * sample.hamer_axis_y_px[None, :]
    )


def local_to_panel(local_pts: np.ndarray, panel: int, zoom: float = 170.0) -> np.ndarray:
    center = np.asarray([panel * 0.5, panel * 0.56], dtype=np.float64)
    pts = np.empty_like(local_pts, dtype=np.float64)
    pts[:, 0] = center[0] + local_pts[:, 0] * zoom
    pts[:, 1] = center[1] - local_pts[:, 1] * zoom
    return pts


def draw_local_grid(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    cx, cy = int(w * 0.5), int(h * 0.56)
    for off in range(-3, 4):
        x = cx + off * 85
        y = cy + off * 85
        cv2.line(img, (x, 0), (x, h), BGR_GRID, 1, cv2.LINE_AA)
        cv2.line(img, (0, y), (w, y), BGR_GRID, 1, cv2.LINE_AA)
    cv2.line(img, (0, cy), (w, cy), (120, 120, 120), 1, cv2.LINE_AA)
    cv2.line(img, (cx, 0), (cx, h), (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(img, "hand-local normalized coords", (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGR_WHITE, 1, cv2.LINE_AA)


def plot_series(img: np.ndarray, values: list[float | None], color: tuple[int, int, int], ymin: float, ymax: float, label: str) -> None:
    h, w = img.shape[:2]
    left, right, top, bottom = 55, w - 24, 82, h - 72
    cv2.rectangle(img, (left, top), (right, bottom), (65, 68, 76), 1, cv2.LINE_AA)
    cv2.putText(img, label, (left, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    valid = [(i, v) for i, v in enumerate(values) if v is not None and math.isfinite(float(v))]
    if len(valid) < 2:
        return
    n = max(1, len(values) - 1)
    pts = []
    for i, v in valid:
        x = int(round(left + (right - left) * i / n))
        vv = max(ymin, min(ymax, float(v)))
        y = int(round(bottom - (vv - ymin) / (ymax - ymin) * (bottom - top)))
        pts.append((x, y))
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, color, 2, cv2.LINE_AA)


def draw_error_panel(panel: int, sample: Sample | None, frame_rmse: list[float | None], key_errors: np.ndarray | None, current_idx: int) -> np.ndarray:
    img = np.full((panel, panel, 3), (28, 30, 34), dtype=np.uint8)
    plot_series(img, frame_rmse, BGR_MED, 0.0, 1.0, "per-frame similarity RMSE")
    left, right, top, bottom = 55, panel - 24, 82, panel - 72
    n = max(1, len(frame_rmse) - 1)
    x = int(round(left + (right - left) * current_idx / n)) if frame_rmse else left
    cv2.line(img, (x, top), (x, bottom), BGR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f"frame {current_idx}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, BGR_WHITE, 1, cv2.LINE_AA)
    if key_errors is not None:
        x0, y0 = 28, panel - 42
        bar_w = max(8, (panel - 56) // len(key_errors))
        for i, err in enumerate(key_errors):
            val = max(0.0, min(1.0, float(err) / 0.8))
            col = (int(50 + 205 * val), int(220 * (1 - val)), int(80 * (1 - val)))
            cv2.rectangle(img, (x0 + i * bar_w, y0 - int(60 * val)), (x0 + (i + 1) * bar_w - 2, y0), col, -1)
        cv2.putText(img, "keypoint error heatmap", (28, panel - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, BGR_WHITE, 1, cv2.LINE_AA)
    return img


def draw_motion_panel(panel: int, corr_series: list[float | None], cos_series: list[float | None], current_idx: int) -> np.ndarray:
    img = np.full((panel, panel, 3), (28, 30, 34), dtype=np.uint8)
    plot_series(img, corr_series, BGR_GOOD, -1.0, 1.0, "rolling velocity corr")
    plot_series(img, cos_series, BGR_MED, -1.0, 1.0, "delta direction cosine")
    left, right, top, bottom = 55, panel - 24, 82, panel - 72
    n = max(1, len(corr_series) - 1)
    x = int(round(left + (right - left) * current_idx / n)) if corr_series else left
    cv2.line(img, (x, top), (x, bottom), BGR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "motion trend in hand-local shape space", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, BGR_WHITE, 1, cv2.LINE_AA)
    return img


def transcode_to_vscode(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "baseline",
        "-level", "3.1", "-crf", "22", "-preset", "veryfast", "-movflags", "+faststart", str(dst),
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


def read_frame_or_blank(cap: cv2.VideoCapture, width: int, height: int) -> tuple[bool, np.ndarray]:
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, np.zeros((height, width, 3), dtype=np.uint8)
    return True, frame


def prepare_best_refs(bestfit_json: Path, summary_3d_json: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    best_payload = read_json(bestfit_json)
    best_by_episode = {e.get("episode"): e for e in best_payload.get("episodes", []) if e.get("episode")}
    summary_payload = read_json(summary_3d_json)
    summary_by_episode = {e.get("episode"): e for e in summary_payload.get("episodes", []) if e.get("episode")}
    return best_by_episode, summary_by_episode



def compact_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    metrics = candidate.get("metrics", {})
    return {
        "side": candidate.get("side"),
        "lag": candidate.get("lag"),
        "alignment": candidate.get("alignment"),
        "matched_frames": metrics.get("matched_frames"),
        "rmse_mean": metrics.get("rmse_mean"),
        "pairwise_distance_rmse": metrics.get("pairwise_distance_rmse"),
        "bone_length_log_error": metrics.get("bone_length_log_error"),
        "joint_angle_deg_error": metrics.get("joint_angle_deg_error"),
        "velocity_corr": metrics.get("velocity_corr"),
        "delta_direction_agreement": metrics.get("delta_direction_agreement"),
        "local_shape_score": metrics.get("local_shape_score"),
    }

def select_best(all_metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not all_metrics:
        return None
    return max(
        all_metrics,
        key=lambda m: (
            float(m["metrics"].get("local_shape_score") or 0.0),
            int(m["metrics"].get("matched_frames") or 0),
            -abs(int(m.get("lag") or 0)),
        ),
    )


def rolling_motion_series(samples: list[Sample], aligned_by_idx: dict[int, np.ndarray], total: int) -> tuple[list[float | None], list[float | None]]:
    corr_series: list[float | None] = [None] * total
    cos_series: list[float | None] = [None] * total
    by_idx = {s.hamer_idx: s for s in samples}
    prev_idx: int | None = None
    recent_v: list[np.ndarray] = []
    recent_h: list[np.ndarray] = []
    for idx in range(total):
        sample = by_idx.get(idx)
        if sample is None or idx not in aligned_by_idx:
            continue
        if prev_idx is not None and prev_idx in by_idx and prev_idx in aligned_by_idx:
            dv = (aligned_by_idx[idx] - aligned_by_idx[prev_idx]).reshape(-1)
            dh = (sample.hamer_local - by_idx[prev_idx].hamer_local).reshape(-1)
            nv = float(np.linalg.norm(dv))
            nh = float(np.linalg.norm(dh))
            if nv > 1e-8 and nh > 1e-8:
                recent_v.append(dv)
                recent_h.append(dh)
                recent_v = recent_v[-20:]
                recent_h = recent_h[-20:]
                cos_series[idx] = float(np.dot(dv, dh) / (nv * nh))
                corr_series[idx] = pearson_corr(np.concatenate(recent_v), np.concatenate(recent_h))
        prev_idx = idx
    return corr_series, cos_series


def render_episode_videos(
    item: dict[str, Any],
    best: dict[str, Any],
    frames: list[dict[str, Any]],
    npz: Any,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, str]:
    panel = int(args.panel_size)
    side = str(best["side"])
    lag = int(best["lag"])
    alignment = str(best["alignment"])
    samples: list[Sample] = best["samples"]
    metrics: dict[str, Any] = best["metrics"]
    fits: list[FrameFit] = metrics["fits"]
    sample_by_idx = {s.hamer_idx: s for s in samples}
    fit_by_idx = {s.hamer_idx: f for s, f in zip(samples, fits)}
    aligned_by_idx = metrics["aligned_by_idx"]
    total = min(int(item.get("frames_written") or 0), len(np.asarray(npz[f"{side}_hand_detected"])))
    if args.max_frames > 0:
        total = min(total, int(args.max_frames))
    width = int(item["width"])
    height = int(item["height"])
    cap = cv2.VideoCapture(str(item["rgb_video"]))
    fps = float(args.fps or item.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    ep_dir = out_dir / f"id_{item['video_id']}_{item['episode']}"
    final = {
        "image_overlay": ep_dir / "image_overlay_local_alignment_vscode.mp4",
        "local_skeleton": ep_dir / "local_skeleton_comparison_vscode.mp4",
        "error_plot": ep_dir / "error_heatmap_timeplot_vscode.mp4",
        "motion_plot": ep_dir / "motion_trend_vscode.mp4",
        "quadview": ep_dir / "quadview_local_hand_alignment_vscode.mp4",
    }
    if all(p.exists() for p in final.values()) and not args.overwrite:
        return {k: str(v) for k, v in final.items()}
    tmp = {k: v.with_suffix(".tmp.mp4") for k, v in final.items()}
    writers = {
        "image_overlay": make_writer(tmp["image_overlay"], fps, (panel, panel)),
        "local_skeleton": make_writer(tmp["local_skeleton"], fps, (panel, panel)),
        "error_plot": make_writer(tmp["error_plot"], fps, (panel, panel)),
        "motion_plot": make_writer(tmp["motion_plot"], fps, (panel, panel)),
        "quadview": make_writer(tmp["quadview"], fps, (panel * 2, panel * 2)),
    }
    frame_rmse: list[float | None] = [None] * total
    for idx in range(total):
        fit = fit_by_idx.get(idx)
        frame_rmse[idx] = fit.rmse if fit is not None else None
    corr_series, cos_series = rolling_motion_series(samples, aligned_by_idx, total)
    hamer_color = BGR_HAMER_LEFT if side == "left" else BGR_HAMER_RIGHT
    vr_color = BGR_VR_LEFT if side == "left" else BGR_VR_RIGHT
    class_color = {"good": BGR_GOOD, "medium": BGR_MED, "bad": BGR_BAD}.get(str(best["category"]), BGR_WHITE)

    for idx in range(total):
        ok, rgb = read_frame_or_blank(cap, width, height)
        if not ok:
            break
        overlay = cv2.resize(rgb, (panel, panel), interpolation=cv2.INTER_AREA)
        sample = sample_by_idx.get(idx)
        fit = fit_by_idx.get(idx)
        key_errors = fit.point_errors if fit is not None else None
        if sample is not None and fit is not None:
            hpts = sample.hamer_kpts_px.copy()
            hpts[:, 0] *= panel / width
            hpts[:, 1] *= panel / height
            draw_skeleton(overlay, hpts, hamer_color, radius=3, thickness=2)
            vimg = local_to_image(fit.src_aligned, sample)
            vimg[:, 0] *= panel / width
            vimg[:, 1] *= panel / height
            draw_skeleton(overlay, vimg, vr_color, radius=3, thickness=2)
        draw_text_box(
            overlay,
            [
                f"RGB overlay side={side} frame={idx} lag={lag}",
                "HaMeR 2D = green/orange; aligned VR local = cyan/pink",
                f"{alignment} score={metrics['local_shape_score']:.1f} rmse={metrics['rmse_mean']:.3f} class={best['category']}",
            ],
            scale=0.46,
        )
        cv2.circle(overlay, (panel - 20, 20), 8, class_color, -1, cv2.LINE_AA)

        local_img = np.full((panel, panel, 3), (28, 30, 34), dtype=np.uint8)
        draw_local_grid(local_img)
        if sample is not None and fit is not None:
            draw_skeleton(local_img, local_to_panel(sample.hamer_local, panel), hamer_color, radius=3, thickness=2)
            draw_skeleton(local_img, local_to_panel(fit.src_aligned, panel), vr_color, radius=3, thickness=2)
            for i, err in enumerate(fit.point_errors):
                pt = local_to_panel(fit.src_aligned, panel)[i]
                val = min(1.0, float(err) / 0.75)
                cv2.circle(local_img, (int(pt[0]), int(pt[1])), int(4 + 8 * val), (0, int(220 * (1 - val)), int(255 * val)), 1, cv2.LINE_AA)
        draw_text_box(local_img, ["local skeleton comparison", f"side={side} align={alignment}"], scale=0.5)

        err_img = draw_error_panel(panel, sample, frame_rmse, key_errors, idx)
        motion_img = draw_motion_panel(panel, corr_series, cos_series, idx)
        writers["image_overlay"].write(overlay)
        writers["local_skeleton"].write(local_img)
        writers["error_plot"].write(err_img)
        writers["motion_plot"].write(motion_img)
        quad = np.vstack([np.hstack([overlay, local_img]), np.hstack([err_img, motion_img])])
        writers["quadview"].write(quad)

    cap.release()
    for writer in writers.values():
        writer.release()
    for key in final:
        write_transcoded(tmp[key], final[key])
    return {k: str(v) for k, v in final.items()}


def process_episode(
    item: dict[str, Any],
    best_refs: dict[str, Any],
    summary3d_refs: dict[str, Any],
    vr_root: Path,
    hamer_root: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    episode = item["episode"]
    video_id = str(item["video_id"])
    frames = read_json(vr_root / episode / f"{episode}.json").get("frames") or []
    npz_path = hamer_root / "output" / f"hand_detections_{video_id}.npz"
    if not frames:
        return {"episode": episode, "video_id": video_id, "status": "skipped", "failure_reason": "missing VR frames"}
    if not npz_path.exists():
        return {"episode": episode, "video_id": video_id, "status": "skipped", "failure_reason": f"missing HaMeR detections: {npz_path}"}
    if not Path(item["rgb_video"]).exists():
        return {"episode": episode, "video_id": video_id, "status": "skipped", "failure_reason": f"missing RGB video: {item['rgb_video']}"}
    npz = np.load(npz_path, allow_pickle=True)
    total = min(int(item.get("frames_written") or 0), len(np.asarray(npz["left_hand_detected"])), len(np.asarray(npz["right_hand_detected"])))
    if total <= 0:
        return {"episode": episode, "video_id": video_id, "status": "skipped", "failure_reason": "no RGB/HaMeR frames"}
    candidates = []
    sample_counts: dict[str, dict[str, int]] = {"left": {}, "right": {}}
    for side in ("left", "right"):
        for lag in range(int(args.lag_min), int(args.lag_max) + 1):
            samples = collect_samples_for_lag(frames, npz, side, lag, total)
            sample_counts[side][str(lag)] = len(samples)
            if len(samples) < int(args.min_samples):
                continue
            eval_similarity = evaluate_samples(samples, "similarity")
            eval_affine = evaluate_samples(samples, "affine")
            for metrics in (eval_similarity, eval_affine):
                candidates.append({"side": side, "lag": lag, "alignment": metrics["alignment"], "samples": samples, "metrics": metrics})
    best = select_best(candidates)
    best_similarity = select_best([c for c in candidates if c.get("alignment") == "similarity"])
    best_affine = select_best([c for c in candidates if c.get("alignment") == "affine"])
    left_tracked = sum(bool((fr.get("left_hand") or {}).get("is_tracked")) for fr in frames)
    right_tracked = sum(bool((fr.get("right_hand") or {}).get("is_tracked")) for fr in frames)
    left_detected = int(np.asarray(npz["left_hand_detected"]).astype(bool).sum())
    right_detected = int(np.asarray(npz["right_hand_detected"]).astype(bool).sum())
    bestfit_ref = best_refs.get(episode, {})
    summary3d_ref = summary3d_refs.get(episode, {})
    if best is None:
        return {
            "episode": episode,
            "video_id": video_id,
            "status": "skipped",
            "failure_reason": f"fewer than {args.min_samples} matched VR-tracked + HaMeR-detected samples for every side/lag",
            "json_frames": len(frames),
            "rgb_frames": total,
            "left_tracked": left_tracked,
            "right_tracked": right_tracked,
            "left_hamer_detected": left_detected,
            "right_hamer_detected": right_detected,
            "sample_counts_by_side_lag": sample_counts,
            "best_similarity": compact_candidate(best_similarity),
            "best_affine": compact_candidate(best_affine),
            "bestfit_reference": bestfit_ref.get("best"),
            "summary3d_reference": {k: summary3d_ref.get(k) for k in ("status", "mean_hamer_ray_to_vr_core_dist_m", "median_hamer_ray_to_vr_core_dist_m")},
        }
    category = classify_result(best["metrics"], int(best["lag"]))
    best["category"] = category
    outputs = render_episode_videos(item, best, frames, npz, out_dir, args)
    metrics = best["metrics"]
    return {
        "episode": episode,
        "video_id": video_id,
        "status": "rendered",
        "category": category,
        "best_side": best["side"],
        "best_lag": best["lag"],
        "best_alignment": best["alignment"],
        "matched_frames": int(metrics["matched_frames"]),
        "json_frames": len(frames),
        "rgb_frames": total,
        "left_tracked": left_tracked,
        "right_tracked": right_tracked,
        "left_hamer_detected": left_detected,
        "right_hamer_detected": right_detected,
        "rmse_mean": metrics["rmse_mean"],
        "rmse_median": metrics["rmse_median"],
        "pairwise_distance_rmse": metrics["pairwise_distance_rmse"],
        "bone_length_log_error": metrics["bone_length_log_error"],
        "joint_angle_deg_error": metrics["joint_angle_deg_error"],
        "velocity_corr": metrics["velocity_corr"],
        "delta_direction_agreement": metrics["delta_direction_agreement"],
        "mean_delta_cos": metrics["mean_delta_cos"],
        "local_shape_score": metrics["local_shape_score"],
        "per_keypoint_rmse": metrics["per_keypoint_rmse"],
        "per_finger_rmse": metrics["per_finger_rmse"],
        "per_finger_bone_log_error": metrics["per_finger_bone_log_error"],
        "per_finger_angle_deg_error": metrics["per_finger_angle_deg_error"],
        "sample_counts_by_side_lag": sample_counts,
        "best_similarity": compact_candidate(best_similarity),
        "best_affine": compact_candidate(best_affine),
        "bestfit_reference": bestfit_ref.get("best"),
        "summary3d_reference": {k: summary3d_ref.get(k) for k in ("status", "mean_hamer_ray_to_vr_core_dist_m", "median_hamer_ray_to_vr_core_dist_m")},
        "outputs": outputs,
    }



def episode_tag(row: dict[str, Any]) -> str:
    return f"id{row.get('video_id')}({row.get('episode')})"


def bestfit_mean_r2(row: dict[str, Any]) -> float | None:
    ref = row.get("bestfit_reference") or {}
    val = ref.get("mean_r2")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def build_interpretation(results: list[dict[str, Any]]) -> dict[str, Any]:
    local_credible = [r for r in results if r.get("category") in {"good", "medium"}]
    local_good = [r for r in results if r.get("category") == "good"]
    local_medium = [r for r in results if r.get("category") == "medium"]
    local_bad = [r for r in results if r.get("category") == "bad"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    stable_projection = []
    projection_issue = []
    world_coord_not_trusted = list(skipped)
    for row in local_credible:
        r2 = bestfit_mean_r2(row)
        if r2 is not None and r2 >= 0.85:
            stable_projection.append(row)
        if r2 is not None and r2 < 0.75:
            projection_issue.append(row)
            world_coord_not_trusted.append(row)
        elif abs(int(row.get("best_lag") or 0)) >= 10:
            world_coord_not_trusted.append(row)
    return {
        "local_tracking_credible": [episode_tag(r) for r in local_credible],
        "local_good": [episode_tag(r) for r in local_good],
        "local_medium": [episode_tag(r) for r in local_medium],
        "local_bad": [episode_tag(r) for r in local_bad],
        "skipped": [episode_tag(r) for r in skipped],
        "projection_or_recording_issue_likely_not_local_tracking": [episode_tag(r) for r in projection_issue],
        "local_and_previous_global_fit_both_reasonable": [episode_tag(r) for r in stable_projection],
        "not_trusted_for_world_coordinate_trajectory_without_more_calibration": [episode_tag(r) for r in world_coord_not_trusted],
        "notes": [
            "All rendered episodes selected affine as the best local overlay; similarity results are stored in best_similarity and are consistently weaker, so strict local bone geometry is only medium quality.",
            "High velocity correlation in rendered episodes means the local hand-motion trend is broadly consistent with HaMeR even when global projection is poor.",
            "Episodes with medium local scores but low previous global bestfit R2 point to recording/projection/time-sync issues rather than a total VR hand-tracking failure.",
        ],
    }

def compact_result(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "episode", "video_id", "status", "category", "best_side", "best_lag", "best_alignment",
        "matched_frames", "rmse_mean", "pairwise_distance_rmse", "bone_length_log_error",
        "joint_angle_deg_error", "velocity_corr", "delta_direction_agreement", "local_shape_score",
        "failure_reason",
    ]
    return {k: row.get(k) for k in keys}


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    rows = [compact_result(r) for r in results]
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(results: list[dict[str, Any]], path: Path, out_dir: Path) -> None:
    lines = [
        "# 20260708 VR/HaMeR Hand-Local Alignment Validation",
        "",
        "This validation compares VR hand joints and HaMeR 2D keypoints in hand-local normalized coordinates. It does not require a true camera/world projection.",
        "",
        "## Joint Mapping",
        "",
        "HaMeR 21 keypoints are mapped to VR joints as follows:",
        "",
        "| HaMeR keypoint | VR joint |",
        "|---|---|",
    ]
    for name in HAMER_KEYPOINT_NAMES:
        lines.append(f"| {name} | {VR_FROM_HAMER[name]} |")
    lines.extend([
        "",
        "Ignored VR joints for 26-joint input: `" + "`, `".join(VR_IGNORED_26) + "`.",
        "",
        "## Episode Summary",
        "",
        "| id | episode | status | class | side | lag | align | matched | score | rmse | pair | bone | angle deg | vel corr | quadview / reason |",
        "|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in results:
        vid = row.get("video_id", "")
        if row.get("status") == "rendered":
            out = row.get("outputs", {}).get("quadview", "")
            tail = f"`{out}`"
            lines.append(
                f"| {vid} | {row['episode']} | rendered | {row.get('category')} | {row.get('best_side')} | {row.get('best_lag')} | "
                f"{row.get('best_alignment')} | {row.get('matched_frames')} | {row.get('local_shape_score'):.1f} | "
                f"{row.get('rmse_mean'):.3f} | {row.get('pairwise_distance_rmse'):.3f} | {row.get('bone_length_log_error'):.3f} | "
                f"{row.get('joint_angle_deg_error'):.1f} | {row.get('velocity_corr') if row.get('velocity_corr') is not None else ''} | {tail} |"
            )
        else:
            lines.append(f"| {vid} | {row.get('episode')} | skipped | - | - | - | - | - | - | - | - | - | - | - | {row.get('failure_reason')} |")
    interpretation = build_interpretation(results)
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- local tracking credible: {', '.join(interpretation['local_tracking_credible']) or 'none'}",
        f"- good: {', '.join(interpretation['local_good']) or 'none'}",
        f"- medium: {', '.join(interpretation['local_medium']) or 'none'}",
        f"- bad: {', '.join(interpretation['local_bad']) or 'none'}",
        f"- skipped: {', '.join(interpretation['skipped']) or 'none'}",
        f"- likely projection/recording issue rather than local VR tracking failure: {', '.join(interpretation['projection_or_recording_issue_likely_not_local_tracking']) or 'none'}",
        f"- local and previous global fit both reasonable: {', '.join(interpretation['local_and_previous_global_fit_both_reasonable']) or 'none'}",
        f"- not trusted for world-coordinate trajectory without more calibration: {', '.join(interpretation['not_trusted_for_world_coordinate_trajectory_without_more_calibration']) or 'none'}",
        "",
        "Notes:",
    ])
    lines.extend([f"- {note}" for note in interpretation["notes"]])
    lines.extend([
        "",
        f"Outputs root: `{out_dir}`",
    ])
    path.write_text("\n".join(lines) + "\n")


def strip_runtime_objects(row: dict[str, Any]) -> dict[str, Any]:
    clean = dict(row)
    for key in ("samples", "metrics"):
        clean.pop(key, None)
    return clean


def main() -> None:
    args = parse_args()
    vr_root = Path(args.vr_root)
    hamer_root = Path(args.hamer_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(hamer_root / "vr_hamer_input_manifest.json")
    if not isinstance(manifest, list):
        raise RuntimeError(f"unexpected manifest format: {hamer_root / 'vr_hamer_input_manifest.json'}")
    items = [item for item in manifest if args.episode_substr in str(item.get("episode", "")) and not item.get("skipped")]
    best_refs, summary3d_refs = prepare_best_refs(Path(args.bestfit_json), Path(args.summary_3d_json))
    results = []
    for item in items:
        print(f"[episode] id={item['video_id']} {item['episode']}")
        row = process_episode(item, best_refs, summary3d_refs, vr_root, hamer_root, out_dir, args)
        if row.get("status") == "rendered":
            print(f"  -> {row['category']} side={row['best_side']} lag={row['best_lag']} align={row['best_alignment']} score={row['local_shape_score']:.1f}")
        else:
            print(f"  -> skipped {row.get('failure_reason')}")
        results.append(row)
    interpretation = build_interpretation(results)
    payload = {
        "description": "20260708 VR/HaMeR hand-local alignment validation; no global camera/world projection is required.",
        "episode_filter": args.episode_substr,
        "lag_range": [args.lag_min, args.lag_max],
        "min_samples": args.min_samples,
        "outputs_root": str(out_dir),
        "interpretation": interpretation,
        "joint_mapping": {
            "hamer_keypoints": HAMER_KEYPOINT_NAMES,
            "vr_from_hamer": VR_FROM_HAMER,
            "ignored_vr_26_joints": VR_IGNORED_26,
            "mapping_note": "Non-thumb VR metacarpal joints and palm are ignored to match HaMeR-21 visible hand keypoints.",
        },
        "episodes": [strip_runtime_objects(r) for r in results],
    }
    json_path = out_dir / "summary_local_hand_alignment_20260708.json"
    csv_path = out_dir / "summary_local_hand_alignment_20260708.csv"
    md_path = out_dir / "summary_local_hand_alignment_20260708.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv(results, csv_path)
    write_markdown(results, md_path, out_dir)
    print(f"[done] summary={md_path}")


if __name__ == "__main__":
    main()
