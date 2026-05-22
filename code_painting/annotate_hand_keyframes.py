#!/usr/bin/env python3
"""Interactive hand keyframe annotator.

This is the repo-local version of the old d_pour_blue hand annotation tool.
It keeps one merged hand_keyframes_all.json and can mark bad videos as
discarded so downstream preview/planner batches can skip them.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import cv2
except ImportError as exc:
    print("Missing dependency: cv2. Install opencv-python in the active env.", file=sys.stderr)
    raise SystemExit(1) from exc


WINDOW_NAME = "Hand Keyframe Annotator"
SCHEMA_VERSION = 3
ARROW_LEFT = {81, 2424832, 65361}
ARROW_RIGHT = {83, 2555904, 65363}
DISCARD_STATUSES = {"reject", "discard", "bad"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate two hand keyframes per video and optionally discard bad ids.")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory containing hand_vis*.mp4 videos.")
    parser.add_argument("--output-json", type=Path, default=None, help="Merged hand_keyframes_all.json path.")
    parser.add_argument("--pattern", default="hand_vis_gripper_*.mp4", help="Video glob pattern.")
    parser.add_argument("--delay-ms", type=int, default=120, help="Autoplay delay in milliseconds.")
    parser.add_argument(
        "--json-video-name-mode",
        choices=["same", "hand_vis"],
        default="hand_vis",
        help="Use input filename as JSON key, or normalize hand_vis_gripper_<id>.mp4 to hand_vis_<id>.mp4.",
    )
    return parser.parse_args()


def natural_key(path: Path) -> Tuple[object, ...]:
    parts: List[object] = []
    for chunk in re.split(r"(\d+)", path.name):
        if not chunk:
            continue
        parts.append(int(chunk) if chunk.isdigit() else chunk)
    return tuple(parts)


def json_video_name(video_path: Path, mode: str) -> str:
    if mode == "same":
        return video_path.name
    match = re.search(r"(\d+)", video_path.stem)
    if match is None:
        return video_path.name
    return f"hand_vis_{int(match.group(1))}.mp4"


def load_annotations(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"_meta": {}, "videos": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON in {path}")
    if "videos" in data and isinstance(data["videos"], dict):
        videos = data["videos"]
        meta = data.get("_meta", {})
    else:
        videos = data
        meta = {"schema_version": 1}

    normalized = {"_meta": dict(meta), "videos": {}}
    for name, info in videos.items():
        if not isinstance(info, dict):
            continue
        normalized["videos"][name] = {
            "video_path": str(info.get("video_path", "")),
            "keyframes": sorted({int(v) for v in info.get("keyframes", [])}),
            "total_frames": int(info.get("total_frames", 0)),
            "fps": float(info.get("fps", 0.0)),
            "updated_at": str(info.get("updated_at", "")),
            "status": str(info.get("status", "in_progress")),
            "notes": str(info.get("notes", "")),
        }
    return normalized


def update_meta(annotations: Dict[str, object], output_json: Path, current_video: str, video_dir: Path, total_videos: int) -> None:
    videos = annotations.setdefault("videos", {})
    assert isinstance(videos, dict)
    annotations["_meta"] = {
        "schema_version": SCHEMA_VERSION,
        "video_dir": str(video_dir),
        "output_json": str(output_json),
        "current_video": current_video,
        "total_videos": int(total_videos),
        "annotated_videos": int(len(videos)),
        "completed_videos": int(sum(1 for info in videos.values() if str(info.get("status")) == "done")),
        "discarded_videos": int(sum(1 for info in videos.values() if str(info.get("status")) in DISCARD_STATUSES)),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Merged hand keyframe annotations. status=reject/discard/bad means downstream preview should skip the video.",
    }


def save_annotations(path: Path, annotations: Dict[str, object], current_video: str, video_dir: Path, total_videos: int) -> None:
    update_meta(annotations, path, current_video, video_dir, total_videos)
    videos = annotations.get("videos", {})
    assert isinstance(videos, dict)
    payload = {
        "_meta": annotations.get("_meta", {}),
        "videos": dict(sorted(videos.items(), key=lambda item: natural_key(Path(item[0])))),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def clamp_frame(frame_idx: int, total_frames: int) -> int:
    if total_frames <= 0:
        return 0
    return max(0, min(int(frame_idx), int(total_frames) - 1))


def read_frame(cap: "cv2.VideoCapture", frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    return frame if ok else None


def draw_overlay(frame, video_name: str, json_name: str, index: int, count: int, frame_idx: int, total_frames: int, paused: bool, keyframes, status: str, delay_ms: int):
    frame_canvas = frame.copy()
    h, w = frame_canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    status_color = (0, 0, 255) if status in DISCARD_STATUSES else (0, 220, 0)
    lines = [
        f"Video: {video_name} ({index + 1}/{count})  JSON key: {json_name}",
        f"Frame: {frame_idx}/{max(total_frames - 1, 0)}  Mode: {'PAUSE' if paused else f'PLAY {delay_ms}ms'}  Status: {status}",
        "Keys: space toggle keyframe | d discard/restore | left/right step | s pause | r replay | n next | p prev | q quit",
        f"Selected keyframes: {', '.join(map(str, keyframes)) if keyframes else 'None'}",
    ]
    line_h = 28
    info_h = 16 + 12 + line_h * len(lines)
    info_canvas = frame_canvas[:info_h, :, :].copy()
    info_canvas[:] = (18, 18, 18)
    cv2.line(info_canvas, (0, info_h - 2), (w, info_h - 2), status_color, 2)
    y = 38
    for line in lines:
        cv2.putText(info_canvas, line, (24, y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += line_h
    if frame_idx in keyframes:
        cv2.putText(frame_canvas, "KEYFRAME", (24, h - 32), font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    if status in DISCARD_STATUSES:
        cv2.putText(frame_canvas, "DISCARDED", (24, h - 72), font, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    return cv2.vconcat([info_canvas, frame_canvas])


def update_record(annotations: Dict[str, object], json_name: str, video_path: Path, keyframes, total_frames: int, fps: float, status: str) -> None:
    videos = annotations.setdefault("videos", {})
    assert isinstance(videos, dict)
    videos[json_name] = {
        "video_path": str(video_path),
        "keyframes": sorted({int(v) for v in keyframes}),
        "total_frames": int(total_frames),
        "fps": float(fps),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": str(status),
        "notes": "discarded_by_operator" if str(status) in DISCARD_STATUSES else "",
    }


def annotate_video(video_path: Path, json_name: str, index: int, count: int, annotations: Dict[str, object], video_dir: Path, output_json: Path, delay_ms: int) -> str:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] failed to open video: {video_path}", file=sys.stderr)
        return "next"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    existing = annotations.get("videos", {}).get(json_name, {})
    keyframes = set(int(v) for v in existing.get("keyframes", []))
    status = str(existing.get("status", "in_progress"))
    frame_idx = 0
    paused = False

    while True:
        frame = read_frame(cap, frame_idx)
        if frame is None:
            frame_idx = clamp_frame(frame_idx, total_frames)
            paused = True
            frame = read_frame(cap, frame_idx)
            if frame is None:
                break
        cv2.imshow(WINDOW_NAME, draw_overlay(frame, video_path.name, json_name, index, count, frame_idx, total_frames, paused, sorted(keyframes), status, delay_ms))
        key = cv2.waitKeyEx(0 if paused else delay_ms)

        if key == -1:
            if not paused:
                frame_idx += 1
                if frame_idx >= total_frames:
                    frame_idx = clamp_frame(total_frames - 1, total_frames)
                    paused = True
            continue
        if key in (ord("q"), 27):
            update_record(annotations, json_name, video_path, keyframes, total_frames, fps, status)
            save_annotations(output_json, annotations, json_name, video_dir, count)
            cap.release()
            return "quit"
        if key == ord(" "):
            if frame_idx in keyframes:
                keyframes.remove(frame_idx)
            else:
                keyframes.add(frame_idx)
            if status in DISCARD_STATUSES:
                status = "in_progress"
            update_record(annotations, json_name, video_path, keyframes, total_frames, fps, status)
            save_annotations(output_json, annotations, json_name, video_dir, count)
            continue
        if key in (ord("d"), ord("D")):
            status = "in_progress" if status in DISCARD_STATUSES else "reject"
            update_record(annotations, json_name, video_path, keyframes, total_frames, fps, status)
            save_annotations(output_json, annotations, json_name, video_dir, count)
            continue
        if key in (ord("n"), ord("N")):
            if status not in DISCARD_STATUSES:
                status = "done"
            update_record(annotations, json_name, video_path, keyframes, total_frames, fps, status)
            save_annotations(output_json, annotations, json_name, video_dir, count)
            cap.release()
            return "next"
        if key in (ord("p"), ord("P")):
            update_record(annotations, json_name, video_path, keyframes, total_frames, fps, status)
            save_annotations(output_json, annotations, json_name, video_dir, count)
            cap.release()
            return "prev"
        if key in (ord("s"), ord("S")):
            paused = not paused
            continue
        if key in (ord("r"), ord("R")):
            frame_idx = 0
            paused = False
            continue
        if key in ARROW_LEFT:
            paused = True
            frame_idx = clamp_frame(frame_idx - 1, total_frames)
            continue
        if key in ARROW_RIGHT:
            paused = True
            frame_idx = clamp_frame(frame_idx + 1, total_frames)
            continue

    update_record(annotations, json_name, video_path, keyframes, total_frames, fps, status)
    save_annotations(output_json, annotations, json_name, video_dir, count)
    cap.release()
    return "next"


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.resolve()
    output_json = (args.output_json or (video_dir / "hand_keyframes_all.json")).resolve()
    videos = sorted(video_dir.glob(args.pattern), key=natural_key)
    if not videos:
        raise SystemExit(f"No videos found in {video_dir} with pattern {args.pattern}")

    annotations = load_annotations(output_json)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1400, 900)

    index = 0
    while 0 <= index < len(videos):
        video = videos[index]
        action = annotate_video(
            video,
            json_video_name(video, args.json_video_name_mode),
            index,
            len(videos),
            annotations,
            video_dir,
            output_json,
            args.delay_ms,
        )
        if action == "quit":
            break
        index = max(0, index - 1) if action == "prev" else index + 1

    cv2.destroyAllWindows()
    current = json_video_name(videos[min(index, len(videos) - 1)], args.json_video_name_mode)
    save_annotations(output_json, annotations, current, video_dir, len(videos))
    print(f"Saved annotations to: {output_json}")


if __name__ == "__main__":
    main()
