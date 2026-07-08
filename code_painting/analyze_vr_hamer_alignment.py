#!/usr/bin/env python3
"""Analyze rough alignment between VR hand joints and HaMeR 2D detections.

This is a diagnostic script. It estimates episode-local mappings from VR hand
joint centroids to HaMeR image-space centroids, sweeps eye-pose candidates and
frame lags, and renders best-fit VR skeleton overlays beside HaMeR videos.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_VR_ROOT = "/home/zaijia001/ssd/data/piper/vr/data"
DEFAULT_HAMER_ROOT = "/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1"
DEFAULT_OUT_DIR = "/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_bestfit_20260708"

POSE_MODES = ("world_xyz", "center_eye_xyz", "left_eye_xyz", "right_eye_xyz")
MODEL_MODES = ("linear_xyz", "perspective_xy_over_z")
CORE_JOINTS = {"wrist", "palm", "index_metacarpal", "middle_metacarpal", "ring_metacarpal", "little_metacarpal"}
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


@dataclass
class Sample:
    episode: str
    video_id: str
    side: str
    vr_frame: int
    hamer_frame: int
    uv: np.ndarray
    world_core: np.ndarray


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vr-root", default=DEFAULT_VR_ROOT)
    ap.add_argument("--hamer-root", default=DEFAULT_HAMER_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--episode-substr", default="20260708")
    ap.add_argument("--lag-min", type=int, default=-10)
    ap.add_argument("--lag-max", type=int, default=10)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--render-videos", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-render-frames", type=int, default=0)
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


def hand_name_index(hand: dict[str, Any]) -> dict[str, int]:
    return {name: i for i, name in enumerate(hand.get("joint_names") or [])}


def hand_centroid(hand: dict[str, Any], joint_filter: set[str] | None = None) -> np.ndarray | None:
    points: list[list[float]] = []
    names = hand.get("joint_names") or []
    poses = hand.get("poses") or []
    for name, pose in zip(names, poses):
        if joint_filter is not None and name not in joint_filter:
            continue
        if len(pose) >= 3 and all(math.isfinite(float(v)) for v in pose[:3]):
            points.append([float(v) for v in pose[:3]])
    if not points:
        return None
    return np.mean(np.asarray(points, dtype=np.float64), axis=0)


def transform_point(point_world: np.ndarray, frame: dict[str, Any], pose_mode: str) -> np.ndarray | None:
    if pose_mode == "world_xyz":
        return point_world
    key = {
        "center_eye_xyz": "center_eye_pose",
        "left_eye_xyz": "left_eye_pose",
        "right_eye_xyz": "right_eye_pose",
    }[pose_mode]
    pose = frame.get(key) or []
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


def build_samples(
    frames: list[dict[str, Any]],
    npz: Any,
    width: float,
    height: float,
    episode: str,
    video_id: str,
    lag: int,
    max_frames: int,
) -> list[Sample]:
    samples: list[Sample] = []
    limit = min(len(frames), max_frames)
    for side in ("left", "right"):
        detected = np.asarray(npz[f"{side}_hand_detected"]).astype(bool)
        kpts_2d = np.asarray(npz[f"{side}_kpts_2d"], dtype=np.float64)
        for vr_idx in range(limit):
            hamer_idx = vr_idx + lag
            if hamer_idx < 0 or hamer_idx >= len(detected):
                continue
            hand = frames[vr_idx].get(f"{side}_hand") or {}
            if not hand.get("is_tracked") or not detected[hamer_idx]:
                continue
            world_core = hand_centroid(hand, CORE_JOINTS)
            if world_core is None:
                continue
            points_2d = kpts_2d[hamer_idx]
            if points_2d.ndim != 2 or points_2d.shape[1] != 2 or np.allclose(points_2d, 0):
                continue
            uv = np.mean(points_2d, axis=0) / np.asarray([width, height], dtype=np.float64)
            samples.append(Sample(episode, video_id, side, vr_idx, hamer_idx, uv, world_core))
    return samples


def fit_linear(X: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
    X1 = np.c_[X, np.ones(len(X), dtype=np.float64)]
    coef, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    pred = X1 @ coef
    err = pred - Y
    rmse = np.sqrt(np.mean(err * err, axis=0))
    ss_res = np.sum(err * err, axis=0)
    ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    corr_u = []
    corr_v = []
    for i in range(X.shape[1]):
        corr_u.append(float(np.corrcoef(X[:, i], Y[:, 0])[0, 1]) if np.std(X[:, i]) > 1e-12 and np.std(Y[:, 0]) > 1e-12 else float("nan"))
        corr_v.append(float(np.corrcoef(X[:, i], Y[:, 1])[0, 1]) if np.std(X[:, i]) > 1e-12 and np.std(Y[:, 1]) > 1e-12 else float("nan"))
    return {
        "coef": coef,
        "rmse_norm": rmse,
        "rmse_px_1280": rmse * 1280.0,
        "r2": r2,
        "corr_u": corr_u,
        "corr_v": corr_v,
        "mean_rmse_px_1280": float(np.mean(rmse * 1280.0)),
        "mean_r2": float(np.mean(r2)),
    }


def evaluate_config(
    samples: list[Sample],
    frames: list[dict[str, Any]],
    pose_mode: str,
    model_mode: str,
    min_samples: int,
) -> dict[str, Any] | None:
    Xs: list[np.ndarray] = []
    Ys: list[np.ndarray] = []
    for sample in samples:
        point = transform_point(sample.world_core, frames[sample.vr_frame], pose_mode)
        if point is None:
            continue
        feat = features_from_point(point, model_mode)
        if feat is None or not np.isfinite(feat).all():
            continue
        Xs.append(feat)
        Ys.append(sample.uv)
    if len(Xs) < min_samples:
        return None
    X = np.stack(Xs)
    Y = np.stack(Ys)
    fit = fit_linear(X, Y)
    return {
        "pose_mode": pose_mode,
        "model_mode": model_mode,
        "n": int(len(Xs)),
        "r2_u": float(fit["r2"][0]),
        "r2_v": float(fit["r2"][1]),
        "mean_r2": fit["mean_r2"],
        "rmse_u_px": float(fit["rmse_px_1280"][0]),
        "rmse_v_px": float(fit["rmse_px_1280"][1]),
        "mean_rmse_px": fit["mean_rmse_px_1280"],
        "corr_u": fit["corr_u"],
        "corr_v": fit["corr_v"],
        "coef": fit["coef"].tolist(),
    }


def pick_best(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    return min(results, key=lambda r: (r["mean_rmse_px"], -r["mean_r2"], abs(r["lag"])))


def apply_model(point_world: np.ndarray, frame: dict[str, Any], config: dict[str, Any]) -> np.ndarray | None:
    point = transform_point(point_world, frame, config["pose_mode"])
    if point is None:
        return None
    feat = features_from_point(point, config["model_mode"])
    if feat is None:
        return None
    coef = np.asarray(config["coef"], dtype=np.float64)
    x1 = np.r_[feat, 1.0]
    return x1 @ coef


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


def read_frame_or_blank(cap: cv2.VideoCapture, width: int, height: int) -> tuple[bool, np.ndarray]:
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, np.zeros((height, width, 3), dtype=np.uint8)
    return True, frame


def draw_text(img: np.ndarray, lines: list[str]) -> None:
    overlay = img.copy()
    h = 26 * len(lines) + 12
    cv2.rectangle(overlay, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    y = 24
    for line in lines:
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26


def draw_projected_hand(img: np.ndarray, hand: dict[str, Any], frame: dict[str, Any], config: dict[str, Any], color: tuple[int, int, int]) -> None:
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
        uv = apply_model(np.asarray(pose[:3], dtype=np.float64), frame, config)
        if uv is None or not np.isfinite(uv).all():
            pts.append((-9999, -9999))
            continue
        px = int(round(float(uv[0]) * img.shape[1]))
        py = int(round(float(uv[1]) * img.shape[0]))
        if px < -img.shape[1] or px > 2 * img.shape[1] or py < -img.shape[0] or py > 2 * img.shape[0]:
            pts.append((-9999, -9999))
        else:
            pts.append((px, py))
    idx = hand_name_index(hand)
    for a, b in HAND_CONNECTION_NAMES:
        ia = idx.get(a)
        ib = idx.get(b)
        if ia is None or ib is None or ia >= len(pts) or ib >= len(pts):
            continue
        pa, pb = pts[ia], pts[ib]
        if pa[0] < 0 or pb[0] < 0:
            continue
        cv2.line(img, pa, pb, color, 3, cv2.LINE_AA)
    for i, point in enumerate(pts):
        if point[0] < 0:
            continue
        cv2.circle(img, point, 5 if i in {0, 1, 5, 10, 15, 20, 25} else 3, color, -1, cv2.LINE_AA)


def render_best_video(
    item: dict[str, Any],
    frames: list[dict[str, Any]],
    config: dict[str, Any],
    hamer_root: Path,
    out_dir: Path,
    overwrite: bool,
    max_render_frames: int,
) -> str | None:
    video_id = str(item["video_id"])
    episode = item["episode"]
    rgb_video = Path(item["rgb_video"])
    hamer_video = hamer_root / "output" / f"hand_vis_gripper_{video_id}.mp4"
    if not rgb_video.exists() or not hamer_video.exists():
        return None
    out_path = out_dir / "videos" / f"id_{video_id}_{episode}_bestfit_vr_vs_hamer_vscode.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return str(out_path)
    cap_rgb = cv2.VideoCapture(str(rgb_video))
    cap_hamer = cv2.VideoCapture(str(hamer_video))
    fps = cap_rgb.get(cv2.CAP_PROP_FPS) or cap_hamer.get(cv2.CAP_PROP_FPS) or 30.0
    n_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    n_hamer = int(cap_hamer.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(n_rgb, n_hamer)
    if max_render_frames > 0:
        total = min(total, max_render_frames)
    panel_w = 640
    panel_h = 640
    tmp_path = out_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel_w * 2, panel_h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter: {tmp_path}")
    lag = int(config["lag"])
    for hamer_idx in range(total):
        ok_rgb, rgb = read_frame_or_blank(cap_rgb, panel_w, panel_h)
        ok_hamer, hamer = read_frame_or_blank(cap_hamer, panel_w, panel_h)
        if not ok_rgb and not ok_hamer:
            break
        rgb = cv2.resize(rgb, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        hamer = cv2.resize(hamer, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        vr_idx = hamer_idx - lag
        if 0 <= vr_idx < len(frames):
            frame = frames[vr_idx]
            draw_projected_hand(rgb, frame.get("left_hand") or {}, frame, config, (255, 220, 40))
            draw_projected_hand(rgb, frame.get("right_hand") or {}, frame, config, (50, 80, 255))
        draw_text(
            rgb,
            [
                f"best-fit VR overlay id={video_id} frame={hamer_idx} vr={vr_idx}",
                f"{config['pose_mode']} {config['model_mode']} lag={lag} R2={config['r2_u']:.2f}/{config['r2_v']:.2f} RMSE={config['rmse_u_px']:.0f}/{config['rmse_v_px']:.0f}px",
            ],
        )
        draw_text(hamer, ["HaMeR gripper detection", episode])
        writer.write(np.hstack([rgb, hamer]))
    writer.release()
    cap_rgb.release()
    cap_hamer.release()
    transcode_to_vscode(tmp_path, out_path)
    tmp_path.unlink(missing_ok=True)
    return str(out_path)


def summarize_global(all_rows: list[dict[str, Any]], min_samples: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pose_mode in POSE_MODES:
        for model_mode in MODEL_MODES:
            rows = [row for row in all_rows if row["pose_mode"] == pose_mode and row["model_mode"] == model_mode]
            if not rows:
                continue
            # For each pose/model, first pick the best lag within each episode,
            # then aggregate those episode-local best-lag fits.
            best_by_episode: list[dict[str, Any]] = []
            for episode in sorted({row["episode"] for row in rows}):
                ep_rows = [row for row in rows if row["episode"] == episode]
                best_by_episode.append(min(ep_rows, key=lambda row: (row["mean_rmse_px"], -row["mean_r2"], abs(row["lag"]))))
            n_total = sum(int(row["n"]) for row in best_by_episode)
            if n_total < min_samples:
                continue
            out.append(
                {
                    "pose_mode": pose_mode,
                    "model_mode": model_mode,
                    "episode_count": len(best_by_episode),
                    "sample_count": n_total,
                    "mean_r2_u": float(np.average([row["r2_u"] for row in best_by_episode], weights=[row["n"] for row in best_by_episode])),
                    "mean_r2_v": float(np.average([row["r2_v"] for row in best_by_episode], weights=[row["n"] for row in best_by_episode])),
                    "mean_rmse_u_px": float(np.average([row["rmse_u_px"] for row in best_by_episode], weights=[row["n"] for row in best_by_episode])),
                    "mean_rmse_v_px": float(np.average([row["rmse_v_px"] for row in best_by_episode], weights=[row["n"] for row in best_by_episode])),
                    "best_lags": {str(row["lag"]): sum(1 for r in best_by_episode if r["lag"] == row["lag"]) for row in best_by_episode},
                }
            )
    return sorted(out, key=lambda row: (row["mean_rmse_u_px"] + row["mean_rmse_v_px"]) / 2.0)


def write_markdown(payload: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# VR-HaMeR Alignment Sweep",
        "",
        f"Episode filter: `{payload['episode_filter']}`",
        f"Lag convention: `{payload['lag_convention']}`",
        f"Episodes considered: {payload['episode_count']}",
        "",
        "## Best Per Episode",
        "",
        "| id | episode | n | best pose | model | lag | R2 u/v | RMSE px u/v | compare video | notes |",
        "|---:|---|---:|---|---|---:|---:|---:|---|---|",
    ]
    for item in payload["episodes"]:
        best = item.get("best")
        if not best:
            lines.append(f"| {item['video_id']} | {item['episode']} | 0 | - | - | - | - | - | - | {item.get('skip_reason', '')} |")
            continue
        note = "low_n" if int(best["n"]) < 50 else ""
        lines.append(
            f"| {item['video_id']} | {item['episode']} | {best['n']} | {best['pose_mode']} | {best['model_mode']} | {best['lag']} | "
            f"{best['r2_u']:.3f}/{best['r2_v']:.3f} | {best['rmse_u_px']:.1f}/{best['rmse_v_px']:.1f} | "
            f"`{item.get('compare_video') or ''}` | {note} |"
        )
    lines += ["", "## Global Weighted Overview: Best Lag Per Episode For Each Pose/Model", ""]
    lines += ["| pose | model | episodes | samples | mean R2 u/v | mean RMSE px u/v |", "|---|---|---:|---:|---:|---:|"]
    for row in payload["global_weighted_overview"]:
        lines.append(
            f"| {row['pose_mode']} | {row['model_mode']} | {row['episode_count']} | {row['sample_count']} | "
            f"{row['mean_r2_u']:.3f}/{row['mean_r2_v']:.3f} | {row['mean_rmse_u_px']:.1f}/{row['mean_rmse_v_px']:.1f} |"
        )
    lines += ["", "## Interpretation", ""]
    lines += payload["interpretation"]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    vr_root = Path(args.vr_root)
    hamer_root = Path(args.hamer_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(hamer_root / "vr_hamer_input_manifest.json")
    episode_payloads: list[dict[str, Any]] = []
    all_episode_config_rows: list[dict[str, Any]] = []

    for item in manifest:
        if item.get("skipped") or args.episode_substr not in item.get("episode", ""):
            continue
        episode = item["episode"]
        video_id = str(item["video_id"])
        frames = read_json(vr_root / episode / f"{episode}.json").get("frames") or []
        npz_path = hamer_root / "output" / f"hand_detections_{video_id}.npz"
        if not npz_path.exists():
            episode_payloads.append({"episode": episode, "video_id": video_id, "skip_reason": "missing HaMeR npz"})
            continue
        npz = np.load(npz_path, allow_pickle=True)
        max_frames = min(len(frames), int(item.get("frames_written") or len(frames)))
        config_results: list[dict[str, Any]] = []
        for lag in range(args.lag_min, args.lag_max + 1):
            samples = build_samples(frames, npz, float(item["width"]), float(item["height"]), episode, video_id, lag, max_frames)
            for pose_mode in POSE_MODES:
                for model_mode in MODEL_MODES:
                    result = evaluate_config(samples, frames, pose_mode, model_mode, args.min_samples)
                    if result is None:
                        continue
                    result["lag"] = lag
                    result["episode"] = episode
                    result["video_id"] = video_id
                    config_results.append(result)
                    all_episode_config_rows.append(result)
        best = pick_best(config_results)
        payload = {
            "episode": episode,
            "video_id": video_id,
            "config_count": len(config_results),
            "best": best,
            "top10": sorted(config_results, key=lambda r: (r["mean_rmse_px"], -r["mean_r2"], abs(r["lag"])))[:10],
        }
        if best and args.render_videos:
            payload["compare_video"] = render_best_video(item, frames, best, hamer_root, out_dir, args.overwrite, args.max_render_frames)
        elif not best:
            payload["skip_reason"] = f"fewer than {args.min_samples} matched tracked+detected samples for every config/lag"
        episode_payloads.append(payload)
        print(
            f"[episode] id={video_id} {episode} best="
            f"{best['pose_mode'] + '/' + best['model_mode'] + '/lag=' + str(best['lag']) if best else 'NONE'}"
        )

    global_overview = summarize_global(all_episode_config_rows, args.min_samples)
    best_counts: dict[str, int] = {}
    lag_counts: dict[str, int] = {}
    for ep in episode_payloads:
        best = ep.get("best")
        if not best:
            continue
        best_counts[best["pose_mode"]] = best_counts.get(best["pose_mode"], 0) + 1
        lag_counts[str(best["lag"])] = lag_counts.get(str(best["lag"]), 0) + 1
    interpretation = [
        f"- Best pose counts: `{best_counts}`.",
        f"- Best lag counts: `{lag_counts}`. Positive lag means `hamer_frame = vr_frame + lag`.",
        "- If a specific eye pose dominates with low RMSE, the recording is likely closer to that eye view.",
        "- If per-episode fits are good but best pose/lag varies, the issue is likely local screen recording/cropping/warp/timestamp rather than a single fixed axis transform.",
        "- If perspective models are not consistently better than linear models, the image is more like a screen-space/composited view than a raw pinhole camera.",
    ]
    payload = {
        "episode_filter": args.episode_substr,
        "lag_convention": "hamer_frame = vr_frame + lag; positive lag means RGB/HaMeR is later than VR hand frame",
        "pose_modes": list(POSE_MODES),
        "model_modes": list(MODEL_MODES),
        "episode_count": len(episode_payloads),
        "best_pose_counts": best_counts,
        "best_lag_counts": lag_counts,
        "episodes": episode_payloads,
        "global_weighted_overview": global_overview,
        "interpretation": interpretation,
    }
    (out_dir / "alignment_sweep_20260708.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    write_markdown(payload, out_dir / "alignment_sweep_20260708.md")
    print(f"[done] json={out_dir / 'alignment_sweep_20260708.json'}")
    print(f"[done] md={out_dir / 'alignment_sweep_20260708.md'}")


if __name__ == "__main__":
    main()
