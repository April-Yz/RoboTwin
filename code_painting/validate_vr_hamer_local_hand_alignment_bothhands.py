#!/usr/bin/env python3
"""Validate 20260708 VR/HaMeR hand-local alignment with both hands.

This is a v2 diagnostic built on validate_vr_hamer_local_hand_alignment.py. It
keeps left/right results separate and adds explicit both-hand mapping hypotheses,
including swapped-hand and mirrored hand-local variants.
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

import validate_vr_hamer_local_hand_alignment as base


DEFAULT_OUT_DIR = "/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands"
MIRROR_VARIANTS = {
    "none": np.asarray([1.0, 1.0], dtype=np.float64),
    "mirror_x": np.asarray([-1.0, 1.0], dtype=np.float64),
    "mirror_y": np.asarray([1.0, -1.0], dtype=np.float64),
    "mirror_xy": np.asarray([-1.0, -1.0], dtype=np.float64),
}
MAPPING_HYPOTHESES = {
    "identity": {"left": "left", "right": "right"},
    "swapped": {"left": "right", "right": "left"},
}
SIDES = ("left", "right")
ALIGNMENTS = ("similarity", "affine")

BGR = {
    "hamer_left": (80, 255, 160),
    "hamer_right": (80, 180, 255),
    "vr_left": (255, 255, 80),
    "vr_right": (180, 120, 255),
    "white": (245, 245, 245),
    "green": (80, 220, 80),
    "yellow": (40, 210, 240),
    "red": (50, 80, 255),
    "grid": (70, 74, 82),
}


@dataclass
class V2Sample:
    hamer_idx: int
    vr_idx: int
    side: str
    vr_side: str
    hamer_side: str
    mirror_variant: str
    vr_local: np.ndarray
    hamer_local: np.ndarray
    hamer_origin_px: np.ndarray
    hamer_axis_x_px: np.ndarray
    hamer_axis_y_px: np.ndarray
    hamer_scale_px: float
    hamer_kpts_px: np.ndarray
    vr_wrist_world: np.ndarray
    vr_scale_world: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vr-root", default=base.DEFAULT_VR_ROOT)
    ap.add_argument("--hamer-root", default=base.DEFAULT_HAMER_ROOT)
    ap.add_argument("--bestfit-json", default=base.DEFAULT_BESTFIT_JSON)
    ap.add_argument("--summary-3d-json", default=base.DEFAULT_3D_SUMMARY_JSON)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--episode-substr", default="20260708")
    ap.add_argument("--lag-min", type=int, default=-10)
    ap.add_argument("--lag-max", type=int, default=10)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--panel-size", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--top-candidates-per-side", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def apply_mirror(coords: np.ndarray, variant: str) -> np.ndarray:
    return coords * MIRROR_VARIANTS[variant][None, :]


def collect_v2_samples(
    frames: list[dict[str, Any]],
    npz: Any,
    vr_side: str,
    hamer_side: str,
    lag: int,
    mirror_variant: str,
    total_frames: int,
) -> list[V2Sample]:
    detected = np.asarray(npz[f"{hamer_side}_hand_detected"]).astype(bool)
    kpts = np.asarray(npz[f"{hamer_side}_kpts_2d"], dtype=np.float64)
    samples: list[V2Sample] = []
    for hamer_idx in range(total_frames):
        vr_idx = hamer_idx - lag
        if vr_idx < 0 or vr_idx >= len(frames) or hamer_idx >= len(detected) or not detected[hamer_idx]:
            continue
        hamer_points = kpts[hamer_idx]
        h_local = base.local_from_hamer_2d(hamer_points)
        if h_local is None:
            continue
        hand = frames[vr_idx].get(f"{vr_side}_hand") or {}
        vr_points = base.extract_vr_mapped_points(hand)
        if vr_points is None:
            continue
        v_local = base.local_from_vr_3d(vr_points)
        if v_local is None:
            continue
        samples.append(
            V2Sample(
                hamer_idx=hamer_idx,
                vr_idx=vr_idx,
                side=f"vr_{vr_side}_to_hamer_{hamer_side}",
                vr_side=vr_side,
                hamer_side=hamer_side,
                mirror_variant=mirror_variant,
                vr_local=apply_mirror(v_local.coords, mirror_variant),
                hamer_local=h_local.coords,
                hamer_origin_px=h_local.origin,
                hamer_axis_x_px=h_local.axis_x,
                hamer_axis_y_px=h_local.axis_y,
                hamer_scale_px=h_local.scale,
                hamer_kpts_px=hamer_points,
                vr_wrist_world=vr_points[0],
                vr_scale_world=max(float(base.palm_scale_2d(vr_points)), 1e-8),
            )
        )
    return samples


def compact_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    metrics = candidate.get("metrics", {})
    return sanitize(
        {
            "vr_side": candidate.get("vr_side"),
            "hamer_side": candidate.get("hamer_side"),
            "lag": candidate.get("lag"),
            "alignment": candidate.get("alignment"),
            "mirror_variant": candidate.get("mirror_variant"),
            "mapping_hypothesis": candidate.get("mapping_hypothesis"),
            "matched_frames": metrics.get("matched_frames"),
            "rmse_mean": metrics.get("rmse_mean"),
            "per_finger_rmse": metrics.get("per_finger_rmse"),
            "pairwise_distance_rmse": metrics.get("pairwise_distance_rmse"),
            "bone_length_log_error": metrics.get("bone_length_log_error"),
            "joint_angle_deg_error": metrics.get("joint_angle_deg_error"),
            "velocity_corr": metrics.get("velocity_corr"),
            "delta_direction_agreement": metrics.get("delta_direction_agreement"),
            "local_shape_score": metrics.get("local_shape_score"),
            "category": candidate.get("category"),
        }
    )


def candidate_key(candidate: dict[str, Any]) -> tuple[float, int, float, float]:
    metrics = candidate.get("metrics", {})
    return (
        float(metrics.get("local_shape_score") or 0.0),
        int(metrics.get("matched_frames") or 0),
        -abs(int(candidate.get("lag") or 0)),
        -float(metrics.get("rmse_mean") or 999.0),
    )


def select_best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    return max(candidates, key=candidate_key) if candidates else None


def evaluate_single_candidates(
    frames: list[dict[str, Any]],
    npz: Any,
    vr_side: str,
    args: argparse.Namespace,
    total_frames: int,
) -> list[dict[str, Any]]:
    candidates = []
    for hamer_side in SIDES:
        for mirror_variant in MIRROR_VARIANTS:
            for lag in range(int(args.lag_min), int(args.lag_max) + 1):
                samples = collect_v2_samples(frames, npz, vr_side, hamer_side, lag, mirror_variant, total_frames)
                if len(samples) < int(args.min_samples):
                    continue
                for alignment in ALIGNMENTS:
                    metrics = base.evaluate_samples(samples, alignment)
                    candidate = {
                        "vr_side": vr_side,
                        "hamer_side": hamer_side,
                        "lag": lag,
                        "alignment": alignment,
                        "mirror_variant": mirror_variant,
                        "mapping_hypothesis": f"single_vr_{vr_side}_to_hamer_{hamer_side}",
                        "samples": samples,
                        "metrics": metrics,
                    }
                    candidate["category"] = base.classify_result(metrics, lag)
                    candidates.append(candidate)
    return candidates


def interhand_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_samples: dict[int, V2Sample] = {s.hamer_idx: s for s in left["samples"]}
    right_samples: dict[int, V2Sample] = {s.hamer_idx: s for s in right["samples"]}
    common = sorted(set(left_samples) & set(right_samples))
    if len(common) < 2:
        return {
            "matched_both_frames": len(common),
            "interhand_distance_rmse": None,
            "interhand_distance_corr": None,
            "interhand_delta_corr": None,
            "interhand_score": 0.0,
        }
    vr_dist = []
    h_dist = []
    for idx in common:
        ls = left_samples[idx]
        rs = right_samples[idx]
        vr_scale = max((ls.vr_scale_world + rs.vr_scale_world) * 0.5, 1e-8)
        h_scale = max((ls.hamer_scale_px + rs.hamer_scale_px) * 0.5, 1e-8)
        vr_dist.append(float(np.linalg.norm(ls.vr_wrist_world - rs.vr_wrist_world) / vr_scale))
        h_dist.append(float(np.linalg.norm(ls.hamer_origin_px - rs.hamer_origin_px) / h_scale))
    vr_arr = np.asarray(vr_dist, dtype=np.float64)
    h_arr = np.asarray(h_dist, dtype=np.float64)
    scale = float(np.median(h_arr) / max(np.median(vr_arr), 1e-8))
    vr_scaled = vr_arr * scale
    rmse = float(np.sqrt(np.mean((vr_scaled - h_arr) ** 2)))
    corr = base.pearson_corr(vr_arr, h_arr)
    delta_corr = base.pearson_corr(np.diff(vr_arr), np.diff(h_arr)) if len(common) > 2 else None
    corr_val = 0.0 if corr is None else max(-1.0, min(1.0, float(corr)))
    delta_val = 0.0 if delta_corr is None else max(-1.0, min(1.0, float(delta_corr)))
    rmse_score = max(0.0, min(1.0, 1.0 - rmse / 1.6))
    inter_score = float(100.0 * (0.45 * rmse_score + 0.35 * ((corr_val + 1.0) / 2.0) + 0.20 * ((delta_val + 1.0) / 2.0)))
    return {
        "matched_both_frames": len(common),
        "interhand_distance_rmse": rmse,
        "interhand_distance_corr": corr,
        "interhand_delta_corr": delta_corr,
        "interhand_scale_vr_to_hamer": scale,
        "interhand_score": inter_score,
    }


def classify_both(score: float, left_category: str, right_category: str, inter: dict[str, Any]) -> str:
    if inter.get("matched_both_frames", 0) < 20:
        return "skipped"
    if left_category == "bad" or right_category == "bad":
        return "bad" if score < 45.0 else "medium"
    if score >= 68.0 and left_category == "good" and right_category == "good":
        return "good"
    if score >= 45.0:
        return "medium"
    return "bad"


def build_both_candidates(
    left_candidates: list[dict[str, Any]],
    right_candidates: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    both_candidates = []
    for mapping_name, mapping in MAPPING_HYPOTHESES.items():
        left_pool = [c for c in left_candidates if c["hamer_side"] == mapping["left"]]
        right_pool = [c for c in right_candidates if c["hamer_side"] == mapping["right"]]
        left_pool = sorted(left_pool, key=candidate_key, reverse=True)[:top_n]
        right_pool = sorted(right_pool, key=candidate_key, reverse=True)[:top_n]
        for lc in left_pool:
            for rc in right_pool:
                inter = interhand_metrics(lc, rc)
                if int(inter.get("matched_both_frames") or 0) < 20:
                    continue
                ls = float(lc["metrics"].get("local_shape_score") or 0.0)
                rs = float(rc["metrics"].get("local_shape_score") or 0.0)
                inter_score = float(inter.get("interhand_score") or 0.0)
                score = float(0.42 * ls + 0.42 * rs + 0.16 * inter_score)
                category = classify_both(score, str(lc.get("category")), str(rc.get("category")), inter)
                both_candidates.append(
                    {
                        "mapping_hypothesis": mapping_name,
                        "left_component": lc,
                        "right_component": rc,
                        "left_lag": lc["lag"],
                        "right_lag": rc["lag"],
                        "left_mirror_variant": lc["mirror_variant"],
                        "right_mirror_variant": rc["mirror_variant"],
                        "left_alignment": lc["alignment"],
                        "right_alignment": rc["alignment"],
                        "interhand_metrics": inter,
                        "local_shape_score": score,
                        "category": category,
                        "matched_frames": min(int(lc["metrics"]["matched_frames"]), int(rc["metrics"]["matched_frames"])),
                    }
                )
    return both_candidates


def select_best_both(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda c: (
            float(c.get("local_shape_score") or 0.0),
            int(c.get("matched_frames") or 0),
            int(c.get("interhand_metrics", {}).get("matched_both_frames") or 0),
            -abs(int(c.get("left_lag") or 0)) - abs(int(c.get("right_lag") or 0)),
        ),
    )


def draw_label(img: np.ndarray, lines: list[str], scale: float = 0.48) -> None:
    base.draw_text_box(img, lines, scale=scale)


def draw_local_grid(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    cx, cy = int(w * 0.5), int(h * 0.56)
    for off in range(-3, 4):
        x = cx + off * 85
        y = cy + off * 85
        cv2.line(img, (x, 0), (x, h), BGR["grid"], 1, cv2.LINE_AA)
        cv2.line(img, (0, y), (w, y), BGR["grid"], 1, cv2.LINE_AA)
    cv2.line(img, (0, cy), (w, cy), (120, 120, 120), 1, cv2.LINE_AA)
    cv2.line(img, (cx, 0), (cx, h), (120, 120, 120), 1, cv2.LINE_AA)


def local_to_panel(local_pts: np.ndarray, panel: int, center_x: float | None = None, zoom: float = 170.0) -> np.ndarray:
    center = np.asarray([panel * 0.5 if center_x is None else center_x, panel * 0.56], dtype=np.float64)
    pts = np.empty_like(local_pts, dtype=np.float64)
    pts[:, 0] = center[0] + local_pts[:, 0] * zoom
    pts[:, 1] = center[1] - local_pts[:, 1] * zoom
    return pts


def local_to_image(local_pts: np.ndarray, sample: V2Sample) -> np.ndarray:
    return sample.hamer_origin_px + sample.hamer_scale_px * (
        local_pts[:, [0]] * sample.hamer_axis_x_px[None, :] + local_pts[:, [1]] * sample.hamer_axis_y_px[None, :]
    )


def color_for_hamer(side: str) -> tuple[int, int, int]:
    return BGR["hamer_left"] if side == "left" else BGR["hamer_right"]


def color_for_vr(side: str) -> tuple[int, int, int]:
    return BGR["vr_left"] if side == "left" else BGR["vr_right"]


def make_series(samples: list[V2Sample], fits: list[base.FrameFit], total: int) -> tuple[list[float | None], list[float | None], list[float | None]]:
    fit_by_idx = {s.hamer_idx: f for s, f in zip(samples, fits)}
    rmse = [None] * total
    for idx, fit in fit_by_idx.items():
        if 0 <= idx < total:
            rmse[idx] = fit.rmse
    corr, cos = base.rolling_motion_series(samples, {s.hamer_idx: f.src_aligned for s, f in zip(samples, fits)}, total)
    return rmse, corr, cos


def plot_series(img: np.ndarray, values: list[float | None], color: tuple[int, int, int], ymin: float, ymax: float, label: str, current_idx: int | None = None) -> None:
    h, w = img.shape[:2]
    left, right, top, bottom = 55, w - 24, 82, h - 72
    cv2.rectangle(img, (left, top), (right, bottom), (65, 68, 76), 1, cv2.LINE_AA)
    cv2.putText(img, label, (left, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    valid = [(i, v) for i, v in enumerate(values) if v is not None and math.isfinite(float(v))]
    if len(valid) >= 2:
        n = max(1, len(values) - 1)
        pts = []
        for i, v in valid:
            x = int(round(left + (right - left) * i / n))
            vv = max(ymin, min(ymax, float(v)))
            y = int(round(bottom - (vv - ymin) / (ymax - ymin) * (bottom - top)))
            pts.append((x, y))
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, color, 2, cv2.LINE_AA)
    if current_idx is not None and values:
        n = max(1, len(values) - 1)
        x = int(round(left + (right - left) * current_idx / n))
        cv2.line(img, (x, top), (x, bottom), BGR["white"], 1, cv2.LINE_AA)


def render_single(
    item: dict[str, Any],
    result: dict[str, Any],
    frames: list[dict[str, Any]],
    npz: Any,
    out_dir: Path,
    args: argparse.Namespace,
    label: str,
) -> dict[str, str]:
    candidate = result.get("candidate")
    if candidate is None:
        return {}
    panel = int(args.panel_size)
    total = min(int(item.get("frames_written") or 0), len(np.asarray(npz["left_hand_detected"])), len(np.asarray(npz["right_hand_detected"])))
    if args.max_frames > 0:
        total = min(total, int(args.max_frames))
    width, height = int(item["width"]), int(item["height"])
    samples: list[V2Sample] = candidate["samples"]
    fits: list[base.FrameFit] = candidate["metrics"]["fits"]
    sample_by_idx = {s.hamer_idx: s for s in samples}
    fit_by_idx = {s.hamer_idx: f for s, f in zip(samples, fits)}
    rmse_series, corr_series, cos_series = make_series(samples, fits, total)
    ep_dir = out_dir / f"id_{item['video_id']}_{item['episode']}"
    outputs = {
        "image_overlay": ep_dir / f"{label}_image_overlay_local_alignment_vscode.mp4",
        "local_skeleton": ep_dir / f"{label}_local_skeleton_comparison_vscode.mp4",
        "error_plot": ep_dir / f"{label}_error_heatmap_timeplot_vscode.mp4",
        "motion_plot": ep_dir / f"{label}_motion_trend_vscode.mp4",
        "quadview": ep_dir / f"{label}_quadview_local_alignment_vscode.mp4",
    }
    if all(p.exists() for p in outputs.values()) and not args.overwrite:
        return {k: str(v) for k, v in outputs.items()}
    tmp = {k: v.with_suffix(".tmp.mp4") for k, v in outputs.items()}
    writers = {
        "image_overlay": base.make_writer(tmp["image_overlay"], args.fps, (panel, panel)),
        "local_skeleton": base.make_writer(tmp["local_skeleton"], args.fps, (panel, panel)),
        "error_plot": base.make_writer(tmp["error_plot"], args.fps, (panel, panel)),
        "motion_plot": base.make_writer(tmp["motion_plot"], args.fps, (panel, panel)),
        "quadview": base.make_writer(tmp["quadview"], args.fps, (panel * 2, panel * 2)),
    }
    cap = cv2.VideoCapture(str(item["rgb_video"]))
    for idx in range(total):
        ok, rgb = base.read_frame_or_blank(cap, width, height)
        if not ok:
            break
        sample = sample_by_idx.get(idx)
        fit = fit_by_idx.get(idx)
        overlay = cv2.resize(rgb, (panel, panel), interpolation=cv2.INTER_AREA)
        local_img = np.full((panel, panel, 3), (28, 30, 34), dtype=np.uint8)
        draw_local_grid(local_img)
        key_errors = None
        if sample is not None and fit is not None:
            hpts = sample.hamer_kpts_px.copy()
            hpts[:, 0] *= panel / width
            hpts[:, 1] *= panel / height
            base.draw_skeleton(overlay, hpts, color_for_hamer(sample.hamer_side), radius=3, thickness=2)
            vimg = local_to_image(fit.src_aligned, sample)
            vimg[:, 0] *= panel / width
            vimg[:, 1] *= panel / height
            base.draw_skeleton(overlay, vimg, color_for_vr(sample.vr_side), radius=3, thickness=2)
            base.draw_skeleton(local_img, local_to_panel(sample.hamer_local, panel), color_for_hamer(sample.hamer_side), radius=3, thickness=2)
            base.draw_skeleton(local_img, local_to_panel(fit.src_aligned, panel), color_for_vr(sample.vr_side), radius=3, thickness=2)
            key_errors = fit.point_errors
        draw_label(
            overlay,
            [
                f"{label} frame={idx} VR {candidate['vr_side']} -> HaMeR {candidate['hamer_side']}",
                f"lag={candidate['lag']} align={candidate['alignment']} mirror={candidate['mirror_variant']}",
                f"score={candidate['metrics']['local_shape_score']:.1f} class={candidate['category']}",
            ],
        )
        draw_label(local_img, [f"{label} local skeleton", "HaMeR=green/orange; VR=cyan/pink"], scale=0.5)
        err_img = base.draw_error_panel(panel, sample, rmse_series, key_errors, idx)
        motion_img = base.draw_motion_panel(panel, corr_series, cos_series, idx)
        writers["image_overlay"].write(overlay)
        writers["local_skeleton"].write(local_img)
        writers["error_plot"].write(err_img)
        writers["motion_plot"].write(motion_img)
        writers["quadview"].write(np.vstack([np.hstack([overlay, local_img]), np.hstack([err_img, motion_img])]))
    cap.release()
    for writer in writers.values():
        writer.release()
    for key in outputs:
        base.write_transcoded(tmp[key], outputs[key])
    return {k: str(v) for k, v in outputs.items()}


def render_both(
    item: dict[str, Any],
    result: dict[str, Any],
    frames: list[dict[str, Any]],
    npz: Any,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, str]:
    both = result.get("candidate")
    if both is None:
        return {}
    lc = both["left_component"]
    rc = both["right_component"]
    panel = int(args.panel_size)
    total = min(int(item.get("frames_written") or 0), len(np.asarray(npz["left_hand_detected"])), len(np.asarray(npz["right_hand_detected"])))
    if args.max_frames > 0:
        total = min(total, int(args.max_frames))
    width, height = int(item["width"]), int(item["height"])
    candidates = [lc, rc]
    sample_maps = [{s.hamer_idx: s for s in cand["samples"]} for cand in candidates]
    fit_maps = [{s.hamer_idx: f for s, f in zip(cand["samples"], cand["metrics"]["fits"])} for cand in candidates]
    series = [make_series(cand["samples"], cand["metrics"]["fits"], total) for cand in candidates]
    ep_dir = out_dir / f"id_{item['video_id']}_{item['episode']}"
    outputs = {
        "image_overlay": ep_dir / "bothhands_image_overlay_local_alignment_vscode.mp4",
        "local_skeleton": ep_dir / "bothhands_local_skeleton_comparison_vscode.mp4",
        "error_plot": ep_dir / "bothhands_error_heatmap_timeplot_vscode.mp4",
        "motion_plot": ep_dir / "bothhands_motion_trend_vscode.mp4",
        "quadview": ep_dir / "bothhands_quadview_local_alignment_vscode.mp4",
    }
    if all(p.exists() for p in outputs.values()) and not args.overwrite:
        return {k: str(v) for k, v in outputs.items()}
    tmp = {k: v.with_suffix(".tmp.mp4") for k, v in outputs.items()}
    writers = {
        "image_overlay": base.make_writer(tmp["image_overlay"], args.fps, (panel, panel)),
        "local_skeleton": base.make_writer(tmp["local_skeleton"], args.fps, (panel, panel)),
        "error_plot": base.make_writer(tmp["error_plot"], args.fps, (panel, panel)),
        "motion_plot": base.make_writer(tmp["motion_plot"], args.fps, (panel, panel)),
        "quadview": base.make_writer(tmp["quadview"], args.fps, (panel * 2, panel * 2)),
    }
    cap = cv2.VideoCapture(str(item["rgb_video"]))
    for idx in range(total):
        ok, rgb = base.read_frame_or_blank(cap, width, height)
        if not ok:
            break
        overlay = cv2.resize(rgb, (panel, panel), interpolation=cv2.INTER_AREA)
        local_img = np.full((panel, panel, 3), (28, 30, 34), dtype=np.uint8)
        cv2.line(local_img, (panel // 2, 0), (panel // 2, panel), (80, 80, 88), 1, cv2.LINE_AA)
        cv2.putText(local_img, "VR left component", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGR["white"], 1, cv2.LINE_AA)
        cv2.putText(local_img, "VR right component", (panel // 2 + 16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BGR["white"], 1, cv2.LINE_AA)
        current_errors = []
        for comp_i, cand in enumerate(candidates):
            sample = sample_maps[comp_i].get(idx)
            fit = fit_maps[comp_i].get(idx)
            if sample is None or fit is None:
                continue
            hpts = sample.hamer_kpts_px.copy()
            hpts[:, 0] *= panel / width
            hpts[:, 1] *= panel / height
            base.draw_skeleton(overlay, hpts, color_for_hamer(sample.hamer_side), radius=3, thickness=2)
            vimg = local_to_image(fit.src_aligned, sample)
            vimg[:, 0] *= panel / width
            vimg[:, 1] *= panel / height
            base.draw_skeleton(overlay, vimg, color_for_vr(sample.vr_side), radius=3, thickness=2)
            center_x = panel * (0.25 if comp_i == 0 else 0.75)
            base.draw_skeleton(local_img, local_to_panel(sample.hamer_local, panel, center_x=center_x, zoom=90.0), color_for_hamer(sample.hamer_side), radius=2, thickness=2)
            base.draw_skeleton(local_img, local_to_panel(fit.src_aligned, panel, center_x=center_x, zoom=90.0), color_for_vr(sample.vr_side), radius=2, thickness=2)
            current_errors.append(fit.point_errors)
        draw_label(
            overlay,
            [
                f"both_hands frame={idx} mapping={both['mapping_hypothesis']} class={both['category']}",
                f"L: HaMeR {lc['hamer_side']} lag={lc['lag']} {lc['alignment']} {lc['mirror_variant']}",
                f"R: HaMeR {rc['hamer_side']} lag={rc['lag']} {rc['alignment']} {rc['mirror_variant']}",
                f"score={both['local_shape_score']:.1f} inter={both['interhand_metrics'].get('interhand_score'):.1f}",
            ],
            scale=0.42,
        )
        draw_label(local_img, ["both-hands local skeleton comparison"], scale=0.5)
        err_img = np.full((panel, panel, 3), (28, 30, 34), dtype=np.uint8)
        plot_series(err_img, series[0][0], color_for_vr("left"), 0.0, 1.0, "left component RMSE", current_idx=idx)
        plot_series(err_img, series[1][0], color_for_vr("right"), 0.0, 1.0, "right component RMSE", current_idx=idx)
        if current_errors:
            vals = np.nanmean(np.vstack(current_errors), axis=0)
            x0, y0 = 28, panel - 42
            bar_w = max(8, (panel - 56) // len(vals))
            for i, err in enumerate(vals):
                val = max(0.0, min(1.0, float(err) / 0.8))
                col = (0, int(220 * (1 - val)), int(255 * val))
                cv2.rectangle(err_img, (x0 + i * bar_w, y0 - int(60 * val)), (x0 + (i + 1) * bar_w - 2, y0), col, -1)
            cv2.putText(err_img, "mean keypoint error heatmap", (28, panel - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, BGR["white"], 1, cv2.LINE_AA)
        motion_img = np.full((panel, panel, 3), (28, 30, 34), dtype=np.uint8)
        plot_series(motion_img, series[0][1], color_for_vr("left"), -1.0, 1.0, "left velocity corr", current_idx=idx)
        plot_series(motion_img, series[1][1], color_for_vr("right"), -1.0, 1.0, "right velocity corr", current_idx=idx)
        cv2.putText(motion_img, f"interhand matched={both['interhand_metrics'].get('matched_both_frames')} dist_corr={both['interhand_metrics'].get('interhand_distance_corr')}", (12, panel - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BGR["white"], 1, cv2.LINE_AA)
        writers["image_overlay"].write(overlay)
        writers["local_skeleton"].write(local_img)
        writers["error_plot"].write(err_img)
        writers["motion_plot"].write(motion_img)
        writers["quadview"].write(np.vstack([np.hstack([overlay, local_img]), np.hstack([err_img, motion_img])]))
    cap.release()
    for writer in writers.values():
        writer.release()
    for key in outputs:
        base.write_transcoded(tmp[key], outputs[key])
    return {k: str(v) for k, v in outputs.items()}


def result_from_candidate(label: str, candidate: dict[str, Any] | None, failure: str | None = None) -> dict[str, Any]:
    if candidate is None:
        return {"status": "skipped", "category": "skipped", "failure_reason": failure or "no valid candidate"}
    metrics = candidate["metrics"]
    return sanitize(
        {
            "status": "rendered",
            "category": candidate.get("category"),
            "vr_side": candidate.get("vr_side"),
            "hamer_side": candidate.get("hamer_side"),
            "best_mapping": candidate.get("mapping_hypothesis"),
            "best_lag": candidate.get("lag"),
            "best_alignment": candidate.get("alignment"),
            "best_mirror_variant": candidate.get("mirror_variant"),
            "matched_frames": metrics.get("matched_frames"),
            "local_shape_score": metrics.get("local_shape_score"),
            "motion_score": metrics.get("velocity_corr"),
            "rmse_mean": metrics.get("rmse_mean"),
            "per_keypoint_rmse": metrics.get("per_keypoint_rmse"),
            "per_finger_rmse": metrics.get("per_finger_rmse"),
            "pairwise_distance_rmse": metrics.get("pairwise_distance_rmse"),
            "bone_length_log_error": metrics.get("bone_length_log_error"),
            "joint_angle_deg_error": metrics.get("joint_angle_deg_error"),
            "velocity_corr": metrics.get("velocity_corr"),
            "delta_direction_agreement": metrics.get("delta_direction_agreement"),
            "candidate": candidate,
            "summary": compact_candidate(candidate),
        }
    )


def result_from_both(candidate: dict[str, Any] | None, failure: str | None = None) -> dict[str, Any]:
    if candidate is None:
        return {"status": "skipped", "category": "skipped", "failure_reason": failure or "no valid both-hands candidate"}
    left = candidate["left_component"]
    right = candidate["right_component"]
    inter = candidate["interhand_metrics"]
    motion_values = [left["metrics"].get("velocity_corr"), right["metrics"].get("velocity_corr"), inter.get("interhand_delta_corr")]
    motion_values = [float(v) for v in motion_values if v is not None and math.isfinite(float(v))]
    return sanitize(
        {
            "status": "rendered",
            "category": candidate.get("category"),
            "best_mapping_hypothesis": candidate.get("mapping_hypothesis"),
            "best_left_lag": candidate.get("left_lag"),
            "best_right_lag": candidate.get("right_lag"),
            "best_left_alignment": candidate.get("left_alignment"),
            "best_right_alignment": candidate.get("right_alignment"),
            "best_left_mirror_variant": candidate.get("left_mirror_variant"),
            "best_right_mirror_variant": candidate.get("right_mirror_variant"),
            "left_component": compact_candidate(left),
            "right_component": compact_candidate(right),
            "matched_frames": candidate.get("matched_frames"),
            "matched_both_frames": inter.get("matched_both_frames"),
            "local_shape_score": candidate.get("local_shape_score"),
            "motion_score": float(np.mean(motion_values)) if motion_values else None,
            "interhand_metrics": inter,
            "candidate": candidate,
        }
    )


def strip_runtime(result: dict[str, Any]) -> dict[str, Any]:
    clean = dict(result)
    candidate = clean.pop("candidate", None)
    if isinstance(candidate, dict):
        candidate_clean = {k: v for k, v in candidate.items() if k not in {"samples", "metrics", "left_component", "right_component"}}
        clean["candidate_runtime_summary"] = sanitize(candidate_clean)
    return sanitize(clean)


def process_episode(item: dict[str, Any], vr_root: Path, hamer_root: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    episode = item["episode"]
    video_id = str(item["video_id"])
    frames = base.read_json(vr_root / episode / f"{episode}.json").get("frames") or []
    npz_path = hamer_root / "output" / f"hand_detections_{video_id}.npz"
    if not frames:
        reason = "missing VR frames"
        return {"episode": episode, "video_id": video_id, "status": "skipped", "left_only": result_from_candidate("left", None, reason), "right_only": result_from_candidate("right", None, reason), "both_hands": result_from_both(None, reason)}
    if not npz_path.exists():
        reason = f"missing HaMeR detections: {npz_path}"
        return {"episode": episode, "video_id": video_id, "status": "skipped", "left_only": result_from_candidate("left", None, reason), "right_only": result_from_candidate("right", None, reason), "both_hands": result_from_both(None, reason)}
    npz = np.load(npz_path, allow_pickle=True)
    total = min(int(item.get("frames_written") or 0), len(np.asarray(npz["left_hand_detected"])), len(np.asarray(npz["right_hand_detected"])))
    if total <= 0:
        reason = "no RGB/HaMeR frames"
        return {"episode": episode, "video_id": video_id, "status": "skipped", "left_only": result_from_candidate("left", None, reason), "right_only": result_from_candidate("right", None, reason), "both_hands": result_from_both(None, reason)}
    left_candidates = evaluate_single_candidates(frames, npz, "left", args, total)
    right_candidates = evaluate_single_candidates(frames, npz, "right", args, total)
    left_best = select_best(left_candidates)
    right_best = select_best(right_candidates)
    both_candidates = build_both_candidates(left_candidates, right_candidates, int(args.top_candidates_per_side))
    both_best = select_best_both(both_candidates)
    row = {
        "episode": episode,
        "video_id": video_id,
        "status": "rendered" if (left_best or right_best or both_best) else "skipped",
        "json_frames": len(frames),
        "rgb_frames": total,
        "left_tracked": sum(bool((fr.get("left_hand") or {}).get("is_tracked")) for fr in frames),
        "right_tracked": sum(bool((fr.get("right_hand") or {}).get("is_tracked")) for fr in frames),
        "left_hamer_detected": int(np.asarray(npz["left_hand_detected"]).astype(bool).sum()),
        "right_hamer_detected": int(np.asarray(npz["right_hand_detected"]).astype(bool).sum()),
        "left_only": result_from_candidate("left", left_best, f"fewer than {args.min_samples} matched samples for every hypothesis"),
        "right_only": result_from_candidate("right", right_best, f"fewer than {args.min_samples} matched samples for every hypothesis"),
        "both_hands": result_from_both(both_best, f"fewer than {args.min_samples} matched both-hand samples for every mapping hypothesis"),
        "candidate_counts": {
            "left_only": len(left_candidates),
            "right_only": len(right_candidates),
            "both_hands": len(both_candidates),
        },
        "best_similarity": {
            "left_only": compact_candidate(select_best([c for c in left_candidates if c["alignment"] == "similarity"])),
            "right_only": compact_candidate(select_best([c for c in right_candidates if c["alignment"] == "similarity"])),
        },
        "best_affine": {
            "left_only": compact_candidate(select_best([c for c in left_candidates if c["alignment"] == "affine"])),
            "right_only": compact_candidate(select_best([c for c in right_candidates if c["alignment"] == "affine"])),
        },
    }
    if left_best:
        row["left_only"]["outputs"] = render_single(item, row["left_only"], frames, npz, out_dir, args, "left")
    if right_best:
        row["right_only"]["outputs"] = render_single(item, row["right_only"], frames, npz, out_dir, args, "right")
    if both_best:
        row["both_hands"]["outputs"] = render_both(item, row["both_hands"], frames, npz, out_dir, args)
    return row


def category_counts(results: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in results:
        cat = row.get(key, {}).get("category") or "skipped"
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "episode", "video_id", "part", "status", "category", "best_mapping", "best_lag", "best_lag_pair",
        "best_alignment", "best_mirror_variant", "matched_frames", "motion_score", "local_shape_score", "failure_reason",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            for part in ("left_only", "right_only", "both_hands"):
                sub = row.get(part, {})
                if part == "both_hands":
                    best_mapping = sub.get("best_mapping_hypothesis")
                    best_lag = ""
                    best_lag_pair = f"{sub.get('best_left_lag')},{sub.get('best_right_lag')}" if sub.get("status") == "rendered" else ""
                    best_alignment = f"{sub.get('best_left_alignment')},{sub.get('best_right_alignment')}" if sub.get("status") == "rendered" else ""
                    best_mirror = f"{sub.get('best_left_mirror_variant')},{sub.get('best_right_mirror_variant')}" if sub.get("status") == "rendered" else ""
                else:
                    best_mapping = sub.get("best_mapping")
                    best_lag = sub.get("best_lag")
                    best_lag_pair = ""
                    best_alignment = sub.get("best_alignment")
                    best_mirror = sub.get("best_mirror_variant")
                writer.writerow(
                    {
                        "episode": row.get("episode"),
                        "video_id": row.get("video_id"),
                        "part": part,
                        "status": sub.get("status"),
                        "category": sub.get("category"),
                        "best_mapping": best_mapping,
                        "best_lag": best_lag,
                        "best_lag_pair": best_lag_pair,
                        "best_alignment": best_alignment,
                        "best_mirror_variant": best_mirror,
                        "matched_frames": sub.get("matched_frames") or sub.get("matched_both_frames"),
                        "motion_score": sub.get("motion_score"),
                        "local_shape_score": sub.get("local_shape_score"),
                        "failure_reason": sub.get("failure_reason"),
                    }
                )


def write_markdown(results: list[dict[str, Any]], path: Path, out_dir: Path) -> None:
    lines = [
        "# 20260708 VR/HaMeR Both-Hands Local Alignment Validation",
        "",
        "This v2 validation keeps left/right hands separate and adds a both-hands mapping hypothesis test. It still avoids requiring a true raw-camera extrinsic.",
        "",
        "## Mapping Hypotheses",
        "",
        "- `identity`: VR_left -> HaMeR_left, VR_right -> HaMeR_right.",
        "- `swapped`: VR_left -> HaMeR_right, VR_right -> HaMeR_left.",
        "- mirror variants are applied to VR hand-local coordinates before alignment: `none`, `mirror_x`, `mirror_y`, `mirror_xy`.",
        "- Mirror variants test possible HaMeR right-hand MANO/canonicalized-hand conventions. Affine alignment can also absorb reflection, so `best_similarity` should be checked when strict local shape matters.",
        "",
        "## Episode Summary",
        "",
        "| id | episode | left class/mapping/lag | right class/mapping/lag | both class/mapping/lag_pair | both score | both matched | quadview |",
        "|---:|---|---|---|---|---:|---:|---|",
    ]
    for row in results:
        left = row.get("left_only", {})
        right = row.get("right_only", {})
        both = row.get("both_hands", {})
        ltxt = f"{left.get('category')} / {left.get('best_mapping')} / {left.get('best_lag')} / {left.get('best_mirror_variant')}" if left.get("status") == "rendered" else f"skipped: {left.get('failure_reason')}"
        rtxt = f"{right.get('category')} / {right.get('best_mapping')} / {right.get('best_lag')} / {right.get('best_mirror_variant')}" if right.get("status") == "rendered" else f"skipped: {right.get('failure_reason')}"
        if both.get("status") == "rendered":
            btxt = f"{both.get('category')} / {both.get('best_mapping_hypothesis')} / {both.get('best_left_lag')},{both.get('best_right_lag')} / {both.get('best_left_mirror_variant')},{both.get('best_right_mirror_variant')}"
            score = f"{both.get('local_shape_score'):.1f}"
            matched = str(both.get("matched_both_frames"))
            quad = f"`{both.get('outputs', {}).get('quadview', '')}`"
        else:
            btxt = f"skipped: {both.get('failure_reason')}"
            score = ""
            matched = ""
            quad = ""
        lines.append(f"| {row.get('video_id')} | {row.get('episode')} | {ltxt} | {rtxt} | {btxt} | {score} | {matched} | {quad} |")
    lines.extend(
        [
            "",
            "## Category Counts",
            "",
            f"- left_only: {category_counts(results, 'left_only')}",
            f"- right_only: {category_counts(results, 'right_only')}",
            f"- both_hands: {category_counts(results, 'both_hands')}",
            "",
            "## Notes",
            "",
            "- `left_only` and `right_only` report the best VR-side to HaMeR-side match independently; they may choose swapped or mirrored variants if HaMeR side labels/canonicalization disagree with VR side labels.",
            "- `both_hands` reports the best joint hypothesis over identity vs swapped mappings, with independent lag/mirror/alignment choices for each VR hand.",
            "- A strong both-hands result supports local hand-shape/motion consistency; it still does not prove a globally calibrated camera/world projection.",
            "",
            f"Outputs root: `{out_dir}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def strip_result(row: dict[str, Any]) -> dict[str, Any]:
    clean = dict(row)
    for part in ("left_only", "right_only", "both_hands"):
        if part in clean:
            clean[part] = strip_runtime(clean[part])
    return sanitize(clean)


def build_interpretation(results: list[dict[str, Any]]) -> dict[str, Any]:
    def tags(part: str, cat: str) -> list[str]:
        return [f"id{r.get('video_id')}({r.get('episode')})" for r in results if r.get(part, {}).get("category") == cat]
    mapping_counts: dict[str, int] = {}
    mirror_counts: dict[str, int] = {}
    for row in results:
        both = row.get("both_hands", {})
        if both.get("status") == "rendered":
            mh = str(both.get("best_mapping_hypothesis"))
            mapping_counts[mh] = mapping_counts.get(mh, 0) + 1
            for key in ("best_left_mirror_variant", "best_right_mirror_variant"):
                mv = str(both.get(key))
                mirror_counts[mv] = mirror_counts.get(mv, 0) + 1
    return {
        "left_only_counts": category_counts(results, "left_only"),
        "right_only_counts": category_counts(results, "right_only"),
        "both_hands_counts": category_counts(results, "both_hands"),
        "both_hands_good": tags("both_hands", "good"),
        "both_hands_medium": tags("both_hands", "medium"),
        "both_hands_bad": tags("both_hands", "bad"),
        "both_hands_skipped": tags("both_hands", "skipped"),
        "best_mapping_hypothesis_counts": mapping_counts,
        "best_mirror_variant_counts": mirror_counts,
        "notes": [
            "Mirrored variants are applied before alignment to test canonicalized/right-hand MANO-style hand-local conventions.",
            "Affine can absorb reflection; use best_similarity fields in JSON for stricter non-reflective shape checks.",
            "Both-hands validation checks local shape/motion plus normalized inter-hand wrist distance trend; it still does not require or prove global camera calibration.",
        ],
    }


def main() -> None:
    args = parse_args()
    vr_root = Path(args.vr_root)
    hamer_root = Path(args.hamer_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = base.read_json(hamer_root / "vr_hamer_input_manifest.json")
    if not isinstance(manifest, list):
        raise RuntimeError("unexpected VR HaMeR manifest format")
    items = [item for item in manifest if args.episode_substr in str(item.get("episode", "")) and not item.get("skipped")]
    results = []
    for item in items:
        print(f"[episode] id={item['video_id']} {item['episode']}")
        row = process_episode(item, vr_root, hamer_root, out_dir, args)
        left = row.get("left_only", {})
        right = row.get("right_only", {})
        both = row.get("both_hands", {})
        print(
            f"  -> left={left.get('category')} right={right.get('category')} both={both.get('category')} "
            f"mapping={both.get('best_mapping_hypothesis')} score={both.get('local_shape_score')}"
        )
        results.append(row)
    interpretation = build_interpretation(results)
    payload = {
        "description": "20260708 VR/HaMeR both-hands hand-local alignment validation.",
        "episode_filter": args.episode_substr,
        "lag_range": [args.lag_min, args.lag_max],
        "min_samples": args.min_samples,
        "outputs_root": str(out_dir),
        "mapping_hypotheses": MAPPING_HYPOTHESES,
        "mirror_variants": list(MIRROR_VARIANTS),
        "mirror_note": "Mirror variants are applied to VR hand-local coordinates before similarity/affine alignment to test HaMeR canonicalized/right-hand MANO conventions.",
        "joint_mapping": {
            "hamer_keypoints": base.HAMER_KEYPOINT_NAMES,
            "vr_from_hamer": base.VR_FROM_HAMER,
            "ignored_vr_26_joints": base.VR_IGNORED_26,
        },
        "interpretation": interpretation,
        "episodes": [strip_result(r) for r in results],
    }
    json_path = out_dir / "summary_bothhands_local_alignment_20260708.json"
    csv_path = out_dir / "summary_bothhands_local_alignment_20260708.csv"
    md_path = out_dir / "summary_bothhands_local_alignment_20260708.md"
    json_path.write_text(json.dumps(sanitize(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    write_csv(results, csv_path)
    write_markdown(results, md_path, out_dir)
    print(f"[done] summary={md_path}")


if __name__ == "__main__":
    main()
