#!/usr/bin/env python3
"""Convert VR JPG episodes to the flat RealR1-style input used by HaMeR."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import cv2


DEFAULT_INPUT_ROOT = "/home/zaijia001/ssd/data/piper/vr/data"
DEFAULT_OUTPUT_ROOT = "/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--fps", type=float, default=None, help="Override output fps. Default: video JSON fps or 30.")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit frames per episode for smoke tests; 0 means all.")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def camera_intrinsics(video: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float, bool]:
    intr = video.get("camera_intrinsics") or {}
    focal = intr.get("focal_length") if isinstance(intr, dict) else None
    principal = intr.get("principal_point") if isinstance(intr, dict) else None
    has_intrinsics = bool(focal and principal)
    fx = float(focal[0]) if focal and len(focal) >= 1 else width / 2.0
    fy = float(focal[1]) if focal and len(focal) >= 2 else height / 2.0
    cx = float(principal[0]) if principal and len(principal) >= 1 else width / 2.0
    cy = float(principal[1]) if principal and len(principal) >= 2 else height / 2.0
    return fx, fy, cx, cy, has_intrinsics


def transcode_to_h264(src: Path, dst: Path) -> None:
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


def write_video(image_paths: list[Path], out_video: Path, fps: float, overwrite: bool, max_frames: int) -> tuple[int, int, int]:
    if out_video.exists() and not overwrite:
        cap = cv2.VideoCapture(str(out_video))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return frames, width, height

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"failed to read first image: {image_paths[0]}")
    height, width = first.shape[:2]
    tmp = out_video.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter: {tmp}")
    selected = image_paths if max_frames <= 0 else image_paths[:max_frames]
    written = 0
    for image_path in selected:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(image)
        written += 1
    writer.release()
    transcode_to_h264(tmp, out_video)
    tmp.unlink(missing_ok=True)
    return written, width, height


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    hamer_input = output_root / "input"
    hamer_input.mkdir(parents=True, exist_ok=True)

    if args.episodes:
        episode_dirs = [input_root / ep for ep in args.episodes]
    else:
        episode_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())

    manifest: list[dict[str, Any]] = []
    next_id = 0
    for ep_dir in episode_dirs:
        if not ep_dir.exists():
            print(f"[skip] missing episode: {ep_dir}")
            continue
        video_json = ep_dir / "camera_real" / f"{ep_dir.name}_video.json"
        main_json = ep_dir / f"{ep_dir.name}.json"
        meta_json = ep_dir / f"{ep_dir.name}_metadata.json"
        video = read_json(video_json)
        main = read_json(main_json)
        meta = read_json(meta_json)
        image_paths = sorted((ep_dir / "camera_real").glob("real_frame_*.jpg"), key=frame_number)
        item: dict[str, Any] = {
            "episode": ep_dir.name,
            "source_dir": str(ep_dir),
            "video_id": None,
            "json_frames": len(main.get("frames") or []),
            "jpg_frames": len(image_paths),
            "skipped": False,
        }
        if not image_paths:
            item["skipped"] = True
            item["skip_reason"] = "no JPG frames under camera_real"
            manifest.append(item)
            print(f"[skip] {ep_dir.name}: no JPG frames")
            continue

        video_id = str(next_id)
        next_id += 1
        fps = float(args.fps or video.get("video_fps") or 30.0)
        rgb_video = hamer_input / f"rgb_{video_id}.mp4"
        frames_written, width, height = write_video(image_paths, rgb_video, fps, args.overwrite, args.max_frames)
        fx, fy, cx, cy, has_intrinsics = camera_intrinsics(video, width, height)
        params = {
            "fx": fx,
            "fy": fy,
            "ppx": cx,
            "ppy": cy,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
            "fps": fps,
            "source_episode": ep_dir.name,
            "source_video_json": str(video_json),
            "intrinsics_source": "camera_real/video_json" if has_intrinsics else "fallback_center_focal_width_half",
            "note": "VR camera extrinsics are not available; these params are only the image intrinsics used by HaMeR.",
        }
        param_path = hamer_input / f"params_{video_id}.json"
        param_path.write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n")

        item.update(
            {
                "video_id": video_id,
                "rgb_video": str(rgb_video),
                "params_json": str(param_path),
                "frames_written": frames_written,
                "fps": fps,
                "width": width,
                "height": height,
                "has_intrinsics": has_intrinsics,
                "has_camera_extrinsics": bool(video.get("cameras")),
                "device_name_main": (main.get("metadata") or {}).get("device_name"),
                "source_desc_metadata": meta.get("source_desc"),
                "coordinate_frame": ((meta.get("env_info") or {}).get("env_kwargs") or {}).get("coordinate_frame"),
            }
        )
        manifest.append(item)
        print(f"[ok] id={video_id} episode={ep_dir.name} frames={frames_written} out={rgb_video}")

    manifest_path = output_root / "vr_hamer_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# VR to HaMeR Input Manifest",
        "",
        f"Input root: `{input_root}`",
        f"HaMeR input: `{hamer_input}`",
        "",
        "| id | episode | frames | fps | size | intrinsics | extrinsics | rgb |",
        "|---:|---|---:|---:|---|---|---|---|",
    ]
    for item in manifest:
        if item.get("skipped"):
            lines.append(f"| - | {item['episode']} | 0 | 0 | - | - | - | skipped: {item.get('skip_reason', '')} |")
            continue
        lines.append(
            f"| {item['video_id']} | {item['episode']} | {item['frames_written']} | {item['fps']:g} | "
            f"{item['width']}x{item['height']} | {item['has_intrinsics']} | {item['has_camera_extrinsics']} | `{item['rgb_video']}` |"
        )
    (output_root / "vr_hamer_input_manifest.md").write_text("\n".join(lines) + "\n")
    print(f"[done] hamer_input={hamer_input}")
    print(f"[done] manifest={manifest_path}")


if __name__ == "__main__":
    main()
