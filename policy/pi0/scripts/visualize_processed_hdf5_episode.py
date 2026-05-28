#!/usr/bin/env python3
"""Visualize one pi0 processed_data episode as a three-camera review video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a quick review video from one processed_data episode HDF5.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="processed_data/<dataset> directory.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize.")
    parser.add_argument("--output-video", type=Path, default=None, help="Output mp4 path.")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    return parser.parse_args()


def decode_jpeg(item) -> np.ndarray:
    if isinstance(item, bytes):
        payload = item
    else:
        payload = bytes(item)
    arr = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode JPEG frame from HDF5.")
    return image


def draw_label(image: np.ndarray, label: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 260), 30), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def load_instruction(dataset_dir: Path, episode: int) -> str:
    path = dataset_dir / f"episode_{episode}" / "instructions.json"
    if not path.is_file():
        return ""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    instructions = payload.get("instructions", [])
    return str(instructions[0]) if instructions else ""


def main() -> None:
    args = parse_args()
    hdf5_path = args.dataset_dir / f"episode_{args.episode}" / f"episode_{args.episode}.hdf5"
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"Missing HDF5 episode: {hdf5_path}")

    output_video = args.output_video or args.dataset_dir / f"episode_{args.episode}" / "episode_review.mp4"
    output_video.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        action = f["action"]
        state = f["observations/state"]
        images = f["observations/images"]
        missing = [cam for cam in CAMERAS if cam not in images]
        if missing:
            raise KeyError(f"Missing camera datasets: {missing}")
        frame_count = min(len(images[cam]) for cam in CAMERAS)
        frame_count = min(frame_count, len(action), len(state))
        if args.max_frames > 0:
            frame_count = min(frame_count, args.max_frames)
        if frame_count <= 0:
            raise RuntimeError(f"No frames to visualize in {hdf5_path}")

        width = args.camera_width
        height = args.camera_height
        canvas_size = (width * len(CAMERAS), height)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(args.fps),
            canvas_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open output video writer: {output_video}")

        instruction = load_instruction(args.dataset_dir, args.episode)
        try:
            for idx in range(frame_count):
                panels = []
                for cam in CAMERAS:
                    img = decode_jpeg(images[cam][idx])
                    img = cv2.resize(img, (width, height))
                    panels.append(draw_label(img, f"{cam}  frame={idx}"))
                canvas = cv2.hconcat(panels)
                status = f"episode={args.episode} frames={frame_count} state_dim={state.shape[1]} action_dim={action.shape[1]}"
                cv2.rectangle(canvas, (0, height - 28), (canvas.shape[1], height), (0, 0, 0), -1)
                cv2.putText(canvas, status, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if instruction:
                    cv2.putText(canvas, instruction[:80], (canvas.shape[1] // 2, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)
                writer.write(canvas)
        finally:
            writer.release()

        print(f"[ok] hdf5={hdf5_path}")
        print(f"[ok] frames={frame_count} state_shape={state.shape} action_shape={action.shape}")
        print(f"[ok] output_video={output_video}")


if __name__ == "__main__":
    main()
