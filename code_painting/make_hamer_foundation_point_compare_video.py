#!/usr/bin/env python3
"""Create side-by-side HaMeR/FoundationPose videos with key point overlays."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np

from render_object_pose_r1_npz import load_pose_sequence

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8


def parse_object_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid object spec '{spec}', expected NAME=/path/to/dir")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid object spec '{spec}', empty name")
    return name, Path(path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand_npz", type=Path, required=True)
    parser.add_argument("--hand_video", type=Path, required=True, help="HaMeR hand_vis_gripper_<id>.mp4")
    parser.add_argument("--object", action="append", required=True, help="Repeat NAME=/foundation/video/object_dir")
    parser.add_argument("--left_object", type=str, required=True)
    parser.add_argument("--right_object", type=str, required=True)
    parser.add_argument("--output_video", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--output_plot", type=Path, default=None)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--fps", type=float, default=0.0, help="Output FPS; <=0 uses hand video FPS.")
    parser.add_argument("--panel_width", type=int, default=640)
    parser.add_argument("--panel_height", type=int, default=480)
    parser.add_argument("--point_radius", type=int, default=7)
    parser.add_argument("--plot_clip_abs_m", type=float, default=0.5, help="Clip plot display to +/- this value; <=0 disables clipping.")
    return parser.parse_args()


def load_camera_params(hand_npz: Path) -> Dict[str, float]:
    data = np.load(str(hand_npz), allow_pickle=True)
    params = data["camera_params"].item()
    return {k: float(v) for k, v in params.items() if k in {"fx", "fy", "cx", "cy", "width", "height"}}


def project_point(point_cam: Sequence[float], camera: Dict[str, float]) -> Optional[Tuple[int, int]]:
    x, y, z = [float(v) for v in point_cam]
    if not np.isfinite([x, y, z]).all() or z <= 1e-6:
        return None
    u = camera["fx"] * x / z + camera["cx"]
    v = camera["fy"] * y / z + camera["cy"]
    return int(round(u)), int(round(v))


def draw_marker(frame: np.ndarray, xy: Optional[Tuple[int, int]], color: Tuple[int, int, int], label: str, radius: int) -> None:
    if xy is None:
        return
    x, y = xy
    cv2.circle(frame, (x, y), radius + 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.putText(frame, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_text(frame: np.ndarray, text: str, xy: Tuple[int, int], color: Tuple[int, int, int] = (80, 255, 80)) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def resize_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def load_hand_data(hand_npz: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(hand_npz), allow_pickle=True)
    return {k: np.asarray(data[k]) for k in data.files if k.endswith("_kpts_2d") or k.endswith("_gripper_position")}


def load_object_tracks(object_specs: Sequence[str]) -> Tuple[Dict[str, Path], Dict[str, np.ndarray], Dict[str, Dict[int, int]]]:
    object_dirs: Dict[str, Path] = {}
    poses: Dict[str, np.ndarray] = {}
    frame_maps: Dict[str, Dict[int, int]] = {}
    for spec in object_specs:
        name, obj_dir = parse_object_spec(spec)
        object_dirs[name] = obj_dir
        pose_npz = obj_dir / "poses.npz"
        if pose_npz.is_file():
            pose_arr, source_frames = load_pose_sequence(pose_npz)
            poses[name] = pose_arr
            frame_maps[name] = {int(src): i for i, src in enumerate(source_frames.tolist())}
        else:
            print(f"[warn] missing object poses: {pose_npz}")
            poses[name] = np.zeros((0, 4, 4), dtype=np.float64)
            frame_maps[name] = {}
    return object_dirs, poses, frame_maps


def read_or_blank(cap: Optional[cv2.VideoCapture], width: int, height: int, title: str) -> np.ndarray:
    if cap is None:
        frame = np.full((height, width, 3), 235, dtype=np.uint8)
        draw_text(frame, f"missing: {title}", (18, 32), (0, 0, 255))
        return frame
    ok, frame = cap.read()
    if not ok:
        frame = np.full((height, width, 3), 235, dtype=np.uint8)
        draw_text(frame, f"no frame: {title}", (18, 32), (0, 0, 255))
        return frame
    return frame


def object_center_for_frame(name: str, frame_idx: int, poses: Dict[str, np.ndarray], frame_maps: Dict[str, Dict[int, int]]) -> Optional[np.ndarray]:
    idx = frame_maps.get(name, {}).get(int(frame_idx))
    if idx is None:
        return None
    return np.asarray(poses[name][idx], dtype=np.float64).reshape(4, 4)[:3, 3].copy()


def side_hand_midpoint(hand_data: Dict[str, np.ndarray], side: str, frame_idx: int) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    pos_key = f"{side}_gripper_position"
    kpt_key = f"{side}_kpts_2d"
    if pos_key not in hand_data or frame_idx >= len(hand_data[pos_key]):
        return None, None, None
    midpoint_3d = np.asarray(hand_data[pos_key][frame_idx], dtype=np.float64).reshape(3)
    if kpt_key in hand_data and frame_idx < len(hand_data[kpt_key]):
        kpts = np.asarray(hand_data[kpt_key][frame_idx], dtype=np.float64)
        thumb = tuple(np.rint(kpts[THUMB_TIP_IDX]).astype(int).tolist())
        index = tuple(np.rint(kpts[INDEX_TIP_IDX]).astype(int).tolist())
    else:
        thumb = None
        index = None
    return midpoint_3d, thumb, index


def plot_distance_curves(rows: Sequence[Dict[str, object]], side_to_object: Dict[str, str], output_plot: Path, clip_abs_m: float) -> None:
    if not rows:
        print(f"[warn] no rows to plot: {output_plot}")
        return
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray([int(row["frame"]) for row in rows], dtype=np.int64)
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    colors = {"x": "#d62728", "y": "#2ca02c", "z": "#1f77b4"}
    clipped_count = 0
    for ax, side in zip(axes, ("left", "right")):
        obj = side_to_object[side]
        for axis in ("x", "y", "z"):
            key = f"{side}_{obj}_hand_minus_object_d{axis}_m"
            values = []
            for row in rows:
                value = row.get(key, "")
                values.append(float(value) if value not in ("", None) else np.nan)
            arr = np.asarray(values, dtype=np.float64)
            plot_arr = arr.copy()
            if clip_abs_m and clip_abs_m > 0:
                finite = np.isfinite(plot_arr)
                clipped_count += int(np.count_nonzero(finite & (np.abs(plot_arr) > clip_abs_m)))
                plot_arr[finite] = np.clip(plot_arr[finite], -clip_abs_m, clip_abs_m)
            ax.plot(frames, plot_arr, label=f"d{axis}", color=colors[axis], linewidth=2.0)
        ax.axhline(0.0, color="0.45", linewidth=1.0)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("camera-axis distance (m)")
        ax.set_title(f"{side} hand midpoint vs {obj}: hand - object in camera frame")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("frame index")
    title = "HaMeR thumb/index midpoint to FoundationPose object center distance by camera axis"
    if clip_abs_m and clip_abs_m > 0:
        title += f" (plot clipped to +/-{clip_abs_m:.2f}m, clipped_values={clipped_count})"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(str(output_plot), dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv is None:
        args.output_csv = args.output_video.with_suffix(".csv")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_plot is None:
        args.output_plot = args.output_video.with_name(f"{args.output_video.stem}_distance.png")

    camera = load_camera_params(args.hand_npz)
    hand_data = load_hand_data(args.hand_npz)
    object_dirs, object_poses, object_frame_maps = load_object_tracks(args.object)
    side_to_object = {"left": args.left_object, "right": args.right_object}

    hand_cap = cv2.VideoCapture(str(args.hand_video))
    if not hand_cap.isOpened():
        raise FileNotFoundError(f"Could not open hand video: {args.hand_video}")
    input_fps = hand_cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(args.fps) if float(args.fps) > 0.0 else float(input_fps)
    frame_count = int(hand_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.max_frames > 0:
        frame_count = min(frame_count, int(args.max_frames))

    obj_caps: Dict[str, Optional[cv2.VideoCapture]] = {}
    for name, obj_dir in object_dirs.items():
        video_path = obj_dir / "mesh_overlay.mp4"
        cap = cv2.VideoCapture(str(video_path)) if video_path.is_file() else None
        if cap is not None and not cap.isOpened():
            cap = None
        obj_caps[name] = cap
        if cap is None:
            print(f"[warn] missing object video: {video_path}")

    object_names = list(object_dirs.keys())
    panel_w = int(args.panel_width)
    panel_h = int(args.panel_height)
    total_w = panel_w * (1 + len(object_names))
    writer = cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (total_w, panel_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer: {args.output_video}")

    headers = ["frame"]
    for side in ("left", "right"):
        obj = side_to_object[side]
        for axis in ("x", "y", "z"):
            headers.append(f"{side}_{obj}_hand_minus_object_d{axis}_m")

    rows = []
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=headers)
        csv_writer.writeheader()
        for frame_idx in range(frame_count):
            hand_frame = read_or_blank(hand_cap, panel_w, panel_h, "hand")
            hand_frame = resize_panel(hand_frame, panel_w, panel_h)
            row = {"frame": frame_idx}

            for side, color in (("left", (255, 80, 80)), ("right", (80, 220, 255))):
                hand_mid_3d, thumb_xy, index_xy = side_hand_midpoint(hand_data, side, frame_idx)
                mid_xy = None
                if thumb_xy is not None and index_xy is not None:
                    mid_xy = (int(round((thumb_xy[0] + index_xy[0]) * 0.5)), int(round((thumb_xy[1] + index_xy[1]) * 0.5)))
                draw_marker(hand_frame, thumb_xy, color, f"{side} thumb", args.point_radius - 2)
                draw_marker(hand_frame, index_xy, color, f"{side} index", args.point_radius - 2)
                draw_marker(hand_frame, mid_xy, (0, 255, 255), f"{side} mid", args.point_radius)

                obj_name = side_to_object[side]
                obj_center = object_center_for_frame(obj_name, frame_idx, object_poses, object_frame_maps)
                if hand_mid_3d is None or obj_center is None:
                    delta = np.full(3, np.nan, dtype=np.float64)
                else:
                    delta = hand_mid_3d - obj_center
                for axis, value in zip(("x", "y", "z"), delta.tolist()):
                    row[f"{side}_{obj_name}_hand_minus_object_d{axis}_m"] = "" if not np.isfinite(value) else f"{float(value):.8f}"

            draw_text(hand_frame, f"HaMeR hand points frame={frame_idx}", (14, 28))
            panels = [hand_frame]

            for obj_name in object_names:
                frame = read_or_blank(obj_caps.get(obj_name), panel_w, panel_h, obj_name)
                frame = resize_panel(frame, panel_w, panel_h)
                obj_center = object_center_for_frame(obj_name, frame_idx, object_poses, object_frame_maps)
                obj_xy = project_point(obj_center, camera) if obj_center is not None else None
                draw_marker(frame, obj_xy, (0, 255, 255), f"{obj_name} center", args.point_radius)
                draw_text(frame, f"FoundationPose {obj_name}", (14, 28))
                panels.append(frame)

            csv_writer.writerow(row)
            rows.append(row)
            writer.write(np.hstack(panels))

    writer.release()
    hand_cap.release()
    for cap in obj_caps.values():
        if cap is not None:
            cap.release()
    plot_distance_curves(rows, side_to_object, args.output_plot, args.plot_clip_abs_m)
    print(f"[done] video={args.output_video}")
    print(f"[done] csv={args.output_csv}")
    print(f"[done] plot={args.output_plot}")


if __name__ == "__main__":
    main()
