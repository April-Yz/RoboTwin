#!/usr/bin/env python3
"""Compare VR hand-tracking metadata with HaMeR detections."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_VR_ROOT = "/home/zaijia001/ssd/data/piper/vr/data"
DEFAULT_HAMER_ROOT = "/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1"
DEFAULT_VR_VIS_ROOT = "/home/zaijia001/ssd/data/piper/vr/0vis/datav1"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vr-root", default=DEFAULT_VR_ROOT)
    ap.add_argument("--hamer-root", default=DEFAULT_HAMER_ROOT)
    ap.add_argument("--vr-vis-root", default=DEFAULT_VR_VIS_ROOT)
    ap.add_argument("--make-videos", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else {}


def frame_counts_from_vr(ep_dir: Path, aligned_frames: int | None = None) -> dict[str, int]:
    main = read_json(ep_dir / f"{ep_dir.name}.json")
    frames = main.get("frames") or []
    aligned = frames[:aligned_frames] if aligned_frames is not None else frames
    return {
        "json_frames": len(frames),
        "vr_aligned_frames": len(aligned),
        "vr_left_tracked_full": sum(bool((fr.get("left_hand") or {}).get("is_tracked")) for fr in frames),
        "vr_right_tracked_full": sum(bool((fr.get("right_hand") or {}).get("is_tracked")) for fr in frames),
        "vr_left_valid_full": sum(bool((fr.get("left_validation") or {}).get("is_valid")) for fr in frames),
        "vr_right_valid_full": sum(bool((fr.get("right_validation") or {}).get("is_valid")) for fr in frames),
        "vr_left_tracked": sum(bool((fr.get("left_hand") or {}).get("is_tracked")) for fr in aligned),
        "vr_right_tracked": sum(bool((fr.get("right_hand") or {}).get("is_tracked")) for fr in aligned),
        "vr_left_valid": sum(bool((fr.get("left_validation") or {}).get("is_valid")) for fr in aligned),
        "vr_right_valid": sum(bool((fr.get("right_validation") or {}).get("is_valid")) for fr in aligned),
    }


def hamer_counts(npz_path: Path) -> dict[str, int | bool]:
    if not npz_path.exists():
        return {
            "hamer_exists": False,
            "hamer_frames": 0,
            "hamer_left_detected": 0,
            "hamer_right_detected": 0,
            "hamer_left_gripper_valid": 0,
            "hamer_right_gripper_valid": 0,
        }
    data = np.load(npz_path, allow_pickle=True)
    left = np.asarray(data["left_hand_detected"]).astype(bool)
    right = np.asarray(data["right_hand_detected"]).astype(bool)
    left_gripper = np.asarray(data.get("left_gripper_valid", np.zeros_like(left))).astype(bool)
    right_gripper = np.asarray(data.get("right_gripper_valid", np.zeros_like(right))).astype(bool)
    return {
        "hamer_exists": True,
        "hamer_frames": int(max(len(left), len(right))),
        "hamer_left_detected": int(left.sum()),
        "hamer_right_detected": int(right.sum()),
        "hamer_left_gripper_valid": int(left_gripper.sum()),
        "hamer_right_gripper_valid": int(right_gripper.sum()),
    }


def transcode_to_vscode(src: Path, dst: Path) -> bool:
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
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as exc:
        print(f"[warn] ffmpeg transcode failed for {src}: {exc}")
        return False


def read_frame_or_blank(cap: cv2.VideoCapture, width: int, height: int) -> tuple[bool, np.ndarray]:
    ok, frame = cap.read()
    if not ok or frame is None:
        return False, np.zeros((height, width, 3), dtype=np.uint8)
    return True, frame


def draw_label(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], 42), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_compare_video(vr_video: Path, hamer_video: Path, out_video: Path, overwrite: bool) -> bool:
    if out_video.exists() and not overwrite:
        return True
    if not vr_video.exists() or not hamer_video.exists():
        return False
    out_video.parent.mkdir(parents=True, exist_ok=True)
    cap_a = cv2.VideoCapture(str(vr_video))
    cap_b = cv2.VideoCapture(str(hamer_video))
    fps = cap_b.get(cv2.CAP_PROP_FPS) or cap_a.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(max(cap_a.get(cv2.CAP_PROP_FRAME_COUNT), cap_b.get(cv2.CAP_PROP_FRAME_COUNT)))
    panel_w, panel_h = 640, 640
    tmp = out_video.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel_w * 2, panel_h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter: {tmp}")
    for _ in range(frames):
        ok_a, frame_a = read_frame_or_blank(cap_a, panel_w, panel_h)
        ok_b, frame_b = read_frame_or_blank(cap_b, panel_w, panel_h)
        if not ok_a and not ok_b:
            break
        frame_a = cv2.resize(frame_a, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        frame_b = cv2.resize(frame_b, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        frame_a = draw_label(frame_a, "VR JSON diagnostic overlay")
        frame_b = draw_label(frame_b, "HaMeR image-space detection")
        writer.write(np.hstack([frame_a, frame_b]))
    writer.release()
    cap_a.release()
    cap_b.release()
    if transcode_to_vscode(tmp, out_video):
        tmp.unlink(missing_ok=True)
    elif tmp.exists():
        tmp.rename(out_video)
    return True


def main() -> None:
    args = parse_args()
    vr_root = Path(args.vr_root)
    hamer_root = Path(args.hamer_root)
    vr_vis_root = Path(args.vr_vis_root)
    manifest = read_json(hamer_root / "vr_hamer_input_manifest.json")
    output_root = hamer_root / "compare"
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for item in manifest:
        if item.get("skipped"):
            continue
        episode = item["episode"]
        video_id = str(item["video_id"])
        ep_dir = vr_root / episode
        row: dict[str, Any] = {
            "episode": episode,
            "video_id": video_id,
            "rgb_frames": item.get("frames_written"),
            "fps": item.get("fps"),
            "has_intrinsics": item.get("has_intrinsics"),
            "has_camera_extrinsics": item.get("has_camera_extrinsics"),
        }
        row.update(frame_counts_from_vr(ep_dir, int(item.get("frames_written") or 0)))
        npz_path = hamer_root / "output" / f"hand_detections_{video_id}.npz"
        row.update(hamer_counts(npz_path))
        if args.make_videos:
            vr_video = vr_vis_root / episode / f"{episode}_hand_overlay_vscode.mp4"
            hamer_video = hamer_root / "output" / f"hand_vis_gripper_{video_id}.mp4"
            compare_video = output_root / f"id_{video_id}_{episode}_vr_vs_hamer_vscode.mp4"
            if make_compare_video(vr_video, hamer_video, compare_video, args.overwrite):
                row["compare_video"] = str(compare_video)
            else:
                row["compare_video"] = None
        rows.append(row)
        print(
            f"[compare] id={video_id} episode={episode} "
            f"VR L/R tracked={row['vr_left_tracked']}/{row['vr_right_tracked']} "
            f"HaMeR L/R detected={row['hamer_left_detected']}/{row['hamer_right_detected']}"
        )

    (output_root / "compare_vr_hamer_stats.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# VR JSON vs HaMeR Detection Comparison",
        "",
        "VR JSON joints are in the recorded tracking/world coordinate frame. HaMeR detections are image-space detections from RGB frames, then lifted to camera-space keypoints by the HaMeR pipeline. VR tracked/valid counts below are aligned to the RGB video frame count; full JSON counts are retained in the JSON file.",
        "",
        "| id | episode | frames | VR L tracked | VR R tracked | VR L valid | VR R valid | HaMeR L det | HaMeR R det | HaMeR L grip | HaMeR R grip | intrinsics | extrinsics | compare |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['video_id']} | {row['episode']} | {row.get('rgb_frames') or 0} | "
            f"{row['vr_left_tracked']} | {row['vr_right_tracked']} | {row['vr_left_valid']} | {row['vr_right_valid']} | "
            f"{row['hamer_left_detected']} | {row['hamer_right_detected']} | "
            f"{row['hamer_left_gripper_valid']} | {row['hamer_right_gripper_valid']} | "
            f"{row.get('has_intrinsics')} | {row.get('has_camera_extrinsics')} | `{row.get('compare_video') or ''}` |"
        )
    (output_root / "compare_vr_hamer_stats.md").write_text("\n".join(lines) + "\n")
    print(f"[done] stats={output_root / 'compare_vr_hamer_stats.md'}")


if __name__ == "__main__":
    main()
