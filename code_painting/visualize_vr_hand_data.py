#!/usr/bin/env python3
"""Visualize Meta Quest VR hand-tracking data over captured JPG frames.

The VR dataset currently has no usable camera intrinsics/extrinsics in the
camera_real metadata. The overlay therefore uses an episode-normalized x/z
projection of the 3D joints. It is intended for temporal/data-quality review,
not calibrated image-space projection.
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

LEFT_COLOR = (255, 220, 40)
RIGHT_COLOR = (50, 80, 255)
UNTRACKED_COLOR = (160, 160, 160)
TEXT_COLOR = (255, 255, 255)
WARN_COLOR = (0, 220, 255)


@dataclass
class EpisodeStats:
    episode: str
    json_frames: int
    jpg_frames: int
    video_json_frames: int | None
    fps: float | None
    width: int | None
    height: int | None
    left_tracked: int
    right_tracked: int
    left_valid: int
    right_valid: int
    metadata_keys: list[str]
    video_keys: list[str]
    has_intrinsics: bool
    has_camera_extrinsics: bool
    integrity_flags: int
    output_video: str | None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="/home/zaijia001/ssd/data/piper/vr/data")
    ap.add_argument("--output-root", default="/home/zaijia001/ssd/RoboTwin/code_painting/vr_hand_visualization")
    ap.add_argument("--episodes", nargs="*", default=None, help="Episode directory names. Default: all.")
    ap.add_argument("--fps", type=float, default=None, help="Override output FPS. Default: video_json fps or 30.")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit frames per episode for quick debug; 0 means all.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-temp", action="store_true")
    ap.add_argument("--draw-untracked", action="store_true", help="Always draw untracked hand poses in gray.")
    return ap.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def frame_number(path: Path) -> int:
    m = re.search(r"(\d+)$", path.stem)
    return int(m.group(1)) if m else 0


def load_episode(ep_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    main_path = ep_dir / f"{ep_dir.name}.json"
    meta_path = ep_dir / f"{ep_dir.name}_metadata.json"
    video_path = ep_dir / "camera_real" / f"{ep_dir.name}_video.json"
    main = read_json(main_path) if main_path.exists() else {}
    meta = read_json(meta_path) if meta_path.exists() else {}
    video = read_json(video_path) if video_path.exists() else {}
    return main, meta, video


def hand_name_index(hand: dict[str, Any]) -> dict[str, int]:
    return {name: i for i, name in enumerate(hand.get("joint_names", []))}


def collect_points(frames: list[dict[str, Any]], prefer_tracked: bool = True) -> np.ndarray:
    points: list[list[float]] = []
    for fr in frames:
        for side in ("left_hand", "right_hand"):
            hand = fr.get(side) or {}
            if prefer_tracked and not hand.get("is_tracked"):
                continue
            for pose in hand.get("poses") or []:
                if len(pose) >= 3 and all(math.isfinite(float(x)) for x in pose[:3]):
                    points.append([float(pose[0]), float(pose[1]), float(pose[2])])
    return np.asarray(points, dtype=np.float32) if points else np.empty((0, 3), dtype=np.float32)


def bounds_for_episode(frames: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, bool]:
    points = collect_points(frames, prefer_tracked=True)
    used_untracked_fallback = False
    if points.size == 0:
        points = collect_points(frames, prefer_tracked=False)
        used_untracked_fallback = True
    if points.size == 0:
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32), used_untracked_fallback
    lo = np.nanmin(points, axis=0)
    hi = np.nanmax(points, axis=0)
    span = hi - lo
    span[span < 1e-4] = 1.0
    return lo, lo + span, used_untracked_fallback


def project_points(poses: list[list[float]], lo: np.ndarray, hi: np.ndarray, width: int, height: int) -> list[tuple[int, int]]:
    margin_x = int(width * 0.08)
    margin_y = int(height * 0.10)
    usable_w = max(1, width - 2 * margin_x)
    usable_h = max(1, height - 2 * margin_y)
    span = hi - lo
    out = []
    for pose in poses:
        if len(pose) < 3:
            out.append((-9999, -9999))
            continue
        x = (float(pose[0]) - float(lo[0])) / float(span[0])
        z = (float(pose[2]) - float(lo[2])) / float(span[2])
        u = margin_x + int(np.clip(x, 0.0, 1.0) * usable_w)
        v = margin_y + int((1.0 - np.clip(z, 0.0, 1.0)) * usable_h)
        out.append((u, v))
    return out


def draw_hand(img: np.ndarray, hand: dict[str, Any], lo: np.ndarray, hi: np.ndarray, color: tuple[int, int, int], draw_untracked: bool) -> None:
    poses = hand.get("poses") or []
    if not poses:
        return
    tracked = bool(hand.get("is_tracked"))
    if not tracked and not draw_untracked:
        return
    color = color if tracked else UNTRACKED_COLOR
    pts = project_points(poses, lo, hi, img.shape[1], img.shape[0])
    name_to_idx = hand_name_index(hand)
    for a, b in HAND_CONNECTION_NAMES:
        ia = name_to_idx.get(a)
        ib = name_to_idx.get(b)
        if ia is None or ib is None or ia >= len(pts) or ib >= len(pts):
            continue
        pa, pb = pts[ia], pts[ib]
        if pa[0] < 0 or pb[0] < 0:
            continue
        cv2.line(img, pa, pb, color, 3 if tracked else 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if p[0] < 0:
            continue
        radius = 5 if i in {0, 1, 5, 10, 15, 20, 25} else 3
        cv2.circle(img, p, radius, color, -1, cv2.LINE_AA)


def draw_text_box(img: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    line_h = 25
    pad = 10
    box_w = min(img.shape[1] - 20, 900)
    box_h = pad * 2 + line_h * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    y = 10 + pad + 18
    for line in lines:
        cv2.putText(img, line, (20, y), font, scale, TEXT_COLOR, thickness, cv2.LINE_AA)
        y += line_h


def transcode_to_vscode(src: Path, dst: Path) -> bool:
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-profile:v", "baseline", "-level", "3.1", "-crf", "22",
        "-preset", "veryfast", "-movflags", "+faststart", str(dst),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as exc:
        print(f"[warn] ffmpeg transcode failed for {src}: {exc}")
        return False


def render_episode(ep_dir: Path, out_root: Path, fps_override: float | None, max_frames: int, overwrite: bool, keep_temp: bool, always_draw_untracked: bool) -> EpisodeStats:
    main, meta, video = load_episode(ep_dir)
    frames = main.get("frames") or []
    cam_dir = ep_dir / "camera_real"
    image_paths = sorted(cam_dir.glob("real_frame_*.jpg"), key=frame_number)
    out_dir = out_root / ep_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{ep_dir.name}_hand_overlay_vscode.mp4"
    tmp_video = out_dir / f"{ep_dir.name}_hand_overlay_tmp.mp4"

    left_tracked = sum(bool(f.get("left_hand", {}).get("is_tracked")) for f in frames)
    right_tracked = sum(bool(f.get("right_hand", {}).get("is_tracked")) for f in frames)
    left_valid = sum(bool(f.get("left_validation", {}).get("is_valid")) for f in frames)
    right_valid = sum(bool(f.get("right_validation", {}).get("is_valid")) for f in frames)
    intr = video.get("camera_intrinsics") or {}
    has_intrinsics = bool(intr.get("focal_length") and intr.get("principal_point")) if isinstance(intr, dict) else False
    has_extrinsics = bool(video.get("cameras"))
    width = video.get("width")
    height = video.get("height")
    fps = float(fps_override or video.get("video_fps") or 30.0)

    if image_paths and (overwrite or not out_video.exists()):
        first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"failed to read image: {image_paths[0]}")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open VideoWriter: {tmp_video}")
        lo, hi, fallback_untracked = bounds_for_episode(frames)
        draw_untracked_when_empty = fallback_untracked or always_draw_untracked
        n = len(image_paths) if max_frames <= 0 else min(len(image_paths), max_frames)
        for out_idx, img_path in enumerate(image_paths[:n]):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            frame_idx = frame_number(img_path)
            if frame_idx >= len(frames):
                frame_idx = min(out_idx, len(frames) - 1)
            fr = frames[frame_idx] if frames else {}
            draw_hand(img, fr.get("left_hand") or {}, lo, hi, LEFT_COLOR, draw_untracked_when_empty)
            draw_hand(img, fr.get("right_hand") or {}, lo, hi, RIGHT_COLOR, draw_untracked_when_empty)
            lines = [
                f"{ep_dir.name} frame={frame_idx} img={img_path.name} fps={fps:g}",
                f"L tracked={bool((fr.get('left_hand') or {}).get('is_tracked'))} valid={bool((fr.get('left_validation') or {}).get('is_valid'))}  R tracked={bool((fr.get('right_hand') or {}).get('is_tracked'))} valid={bool((fr.get('right_validation') or {}).get('is_valid'))}",
                "overlay: episode-normalized VR joint x/z; camera intrinsics/extrinsics missing, not calibrated projection",
            ]
            if draw_untracked_when_empty:
                lines.append("gray/untracked poses are shown because no tracked hand was found or --draw-untracked is enabled")
            draw_text_box(img, lines)
            writer.write(img)
        writer.release()
        if transcode_to_vscode(tmp_video, out_video):
            if not keep_temp:
                tmp_video.unlink(missing_ok=True)
        elif tmp_video.exists():
            tmp_video.rename(out_video)
    elif not image_paths:
        out_video = None

    return EpisodeStats(
        episode=ep_dir.name,
        json_frames=len(frames),
        jpg_frames=len(image_paths),
        video_json_frames=video.get("num_frames"),
        fps=video.get("video_fps"),
        width=width,
        height=height,
        left_tracked=left_tracked,
        right_tracked=right_tracked,
        left_valid=left_valid,
        right_valid=right_valid,
        metadata_keys=list((main.get("metadata") or {}).keys()),
        video_keys=list(video.keys()),
        has_intrinsics=has_intrinsics,
        has_camera_extrinsics=has_extrinsics,
        integrity_flags=len(video.get("frame_integrity") or []),
        output_video=str(out_video) if out_video else None,
    )


def write_stats(stats: list[EpisodeStats], out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    data = [s.__dict__ for s in stats]
    (out_root / "vr_data_stats.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    total_json = sum(s.json_frames for s in stats)
    total_jpg = sum(s.jpg_frames for s in stats)
    lines = [
        "# VR Hand Data Stats",
        "",
        f"Episodes: {len(stats)}",
        f"Total JSON frames: {total_json}",
        f"Total JPG frames: {total_jpg}",
        "",
        "Note: camera_intrinsics.focal_length/principal_point and cameras/extrinsics are empty in the inspected camera_real metadata, so overlay videos use episode-normalized VR joint x/z, not calibrated image projection.",
        "",
        "| episode | json_frames | jpg_frames | video_frames | fps | left_tracked | right_tracked | left_valid | right_valid | intrinsics | extrinsics | integrity_flags | output |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for s in stats:
        lines.append(
            f"| {s.episode} | {s.json_frames} | {s.jpg_frames} | {s.video_json_frames or 0} | {s.fps or 0:g} | "
            f"{s.left_tracked} | {s.right_tracked} | {s.left_valid} | {s.right_valid} | "
            f"{s.has_intrinsics} | {s.has_camera_extrinsics} | {s.integrity_flags} | {s.output_video or ''} |"
        )
    (out_root / "vr_data_stats.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if args.episodes:
        ep_dirs = [input_root / ep for ep in args.episodes]
    else:
        ep_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())
    stats = []
    for ep_dir in ep_dirs:
        if not ep_dir.exists():
            print(f"[skip] missing episode: {ep_dir}")
            continue
        print(f"[episode] {ep_dir.name}")
        stats.append(render_episode(ep_dir, output_root, args.fps, args.max_frames, args.overwrite, args.keep_temp, args.draw_untracked))
    write_stats(stats, output_root)
    print(f"[done] output_root={output_root}")
    print(f"[done] stats={output_root / 'vr_data_stats.md'}")


if __name__ == "__main__":
    main()
