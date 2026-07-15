#!/usr/bin/env python3
"""Compose OursV2/Canonical/Piper-real control comparison videos."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


PANEL_W = 480
PANEL_H = 360
VIDEO_W = 1920
INFO_H = 72
PLOT_H = 648
VIDEO_H = PANEL_H + INFO_H + PLOT_H
METHODS = (
    ("Piper real", "#202020"),
    ("OursV2", "#00a6c8"),
    ("Canonical", "#d149b8"),
)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("piper_pos_vis_compose", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def resize_panel(image: np.ndarray, header: str, vis) -> np.ndarray:
    panel = cv2.resize(image, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    return vis.add_panel_header(panel, header)


def draw_series_axes(
    image: np.ndarray,
    frame_idx: int,
    positions: np.ndarray,
    rotations: np.ndarray,
    label: str,
    kind: str,
    vis,
    left_base_t_camera: np.ndarray,
    left_base_t_right_base: np.ndarray,
    success: np.ndarray | None = None,
) -> None:
    for arm_idx, arm_letter in enumerate(("L", "R")):
        if success is not None and not bool(success[frame_idx, arm_idx]):
            continue
        position_left, rotation_left = vis.arm_pose_in_left_base(
            arm_idx,
            positions[frame_idx, arm_idx],
            rotations[frame_idx, arm_idx],
            left_base_t_right_base,
        )
        vis.draw_pose_axes(
            image,
            position_left,
            rotation_left,
            left_base_t_camera,
            f"{label}-{arm_letter}",
            kind,
        )


def plot_background(
    real_world: np.ndarray,
    ours_world: np.ndarray,
    canonical_world: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    figure = Figure(figsize=(VIDEO_W / 100.0, PLOT_H / 100.0), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2, sharex="col", squeeze=False)
    frames = np.arange(len(real_world))
    for axis_idx, axis_name in enumerate("XYZ"):
        for arm_idx, arm_name in enumerate(("Left", "Right")):
            axis = axes[axis_idx, arm_idx]
            for values, (label, color) in zip(
                (real_world, ours_world, canonical_world), METHODS
            ):
                axis.plot(
                    frames,
                    values[:, arm_idx, axis_idx],
                    color=color,
                    linewidth=1.25,
                    label=label,
                )
            axis.grid(True, alpha=0.22, linewidth=0.6)
            axis.set_ylabel(f"world {axis_name} (m)", fontsize=8)
            axis.tick_params(labelsize=7)
            if axis_idx == 0:
                axis.set_title(
                    f"{arm_name} arm | {'same q input' if mode == 'joint' else 'same Real-TCP target'}",
                    fontsize=10,
                )
                axis.legend(loc="best", fontsize=7, ncol=3)
            if axis_idx == 2:
                axis.set_xlabel("synchronized D435 frame", fontsize=8)
    figure.subplots_adjust(left=0.055, right=0.985, top=0.94, bottom=0.085, hspace=0.22, wspace=0.13)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    background = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    cursor_boxes = []
    for arm_idx in range(2):
        x0 = int(round(axes[0, arm_idx].bbox.x0))
        x1 = int(round(axes[0, arm_idx].bbox.x1))
        y0 = int(round(PLOT_H - axes[0, arm_idx].bbox.y1))
        y1 = int(round(PLOT_H - axes[2, arm_idx].bbox.y0))
        cursor_boxes.append((x0, x1, y0, y1))
    figure.clear()
    return background, cursor_boxes


def add_cursor(
    background: np.ndarray,
    cursor_boxes: list[tuple[int, int, int, int]],
    frame_idx: int,
    count: int,
) -> np.ndarray:
    image = background.copy()
    ratio = 0.0 if count <= 1 else frame_idx / float(count - 1)
    for x0, x1, y0, y1 in cursor_boxes:
        x = int(round(x0 + ratio * (x1 - x0)))
        cv2.line(image, (x, y0), (x, y1), (35, 35, 235), 2, cv2.LINE_AA)
    return image


def info_banner(
    task: str,
    episode: str,
    mode: str,
    frame_idx: int,
    count: int,
    ours_success: np.ndarray,
    canonical_success: np.ndarray,
) -> np.ndarray:
    banner = np.full((INFO_H, VIDEO_W, 3), (31, 37, 45), dtype=np.uint8)
    title = (
        "JOINT CONTROL | common input: Piper real q1-q6"
        if mode == "joint"
        else "EE-POSE CONTROL | common input: Piper real T_B_RTCP"
    )
    cv2.putText(
        banner,
        f"{task}/{episode}  {title}  frame {frame_idx + 1}/{count}",
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.61,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    if mode == "joint":
        line = (
            "Plots: shared Piper0515 WORLD XYZ | colored pose axes: local TCP "
            "+X RED, +Y GREEN, +Z BLUE"
        )
    else:
        ours = "/".join(
            arm if ok else f"{arm}-FAIL"
            for arm, ok in zip(("L", "R"), ours_success[frame_idx])
        )
        canonical = "/".join(
            arm if ok else f"{arm}-FAIL"
            for arm, ok in zip(("L", "R"), canonical_success[frame_idx])
        )
        line = (
            f"IK status OursV2={ours}, Canonical={canonical} | failed arms use "
            "direct-q visual fallback and are excluded from curves"
        )
    cv2.putText(
        banner,
        line,
        (14, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (205, 214, 224),
        1,
        cv2.LINE_AA,
    )
    return banner


def read_paths(vis, path: Path, count: int) -> list[Path]:
    paths = vis.numeric_files(path, ".jpg")
    if len(paths) < count:
        raise RuntimeError(f"{path} has {len(paths)} frames; expected at least {count}")
    return paths[:count]


def render(args: argparse.Namespace) -> dict:
    vis = load_module(args.vis_script)
    plan = load_npz(args.plan)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    embodiment = calibration["robot_config"]["left_embodiment_config"]
    fk = vis.PiperFk(args.urdf)
    data = vis.load_episode(
        args.episode_dir,
        fk,
        np.asarray(embodiment["global_trans_matrix"], dtype=np.float64),
        float(embodiment["gripper_bias"]),
    )
    count = int(plan["real_q"].shape[0])
    if count > len(data["main_times"]):
        raise RuntimeError("plan frame count exceeds synchronized source")
    direct_paths = read_paths(vis, args.direct_sim_dir, count)
    ours_paths = read_paths(vis, args.ours_sim_dir, count)
    canonical_paths = read_paths(vis, args.canonical_sim_dir, count)
    left_base_t_camera = np.asarray(
        calibration["head_camera"]["raw_left_base_T_head_camera"]["matrix"],
        dtype=np.float64,
    )
    left_base_t_right_base = np.asarray(
        json.loads(Path(calibration["source_files"]["base"]).read_text(encoding="utf-8"))[
            "left_base_T_right_base"
        ]["matrix"],
        dtype=np.float64,
    )
    real_positions = plan["real_rtcp_positions"]
    real_rotations = plan["real_rtcp_rotations"]
    if args.mode == "joint":
        ours_positions = plan["joint_oursv2_tcp_positions"]
        ours_rotations = plan["joint_oursv2_tcp_rotations"]
        canonical_positions = plan["joint_canonical_rtcp_positions"]
        canonical_rotations = plan["joint_canonical_rtcp_rotations"]
    else:
        ours_positions = plan["oursv2_achieved_rtcp_positions"].copy()
        ours_rotations = plan["oursv2_achieved_rtcp_rotations"]
        canonical_positions = plan["canonical_achieved_rtcp_positions"].copy()
        canonical_rotations = plan["canonical_achieved_rtcp_rotations"]
        ours_positions[~plan["oursv2_ik_success"]] = np.nan
        canonical_positions[~plan["canonical_ik_success"]] = np.nan
    real_world = vis.positions_to_world(real_positions, calibration)
    ours_world = vis.positions_to_world(ours_positions, calibration)
    canonical_world = vis.positions_to_world(canonical_positions, calibration)
    plot, cursor_boxes = plot_background(
        real_world, ours_world, canonical_world, args.mode
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{VIDEO_W}x{VIDEO_H}",
        "-r", f"{float(data['fps']):.6f}", "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame_idx in range(count):
            main_native, _ = vis.read_and_resize(data["main_paths"][frame_idx])
            draw_series_axes(
                main_native, frame_idx, real_positions, real_rotations, "REAL", "real",
                vis, left_base_t_camera, left_base_t_right_base,
            )
            if args.mode == "joint":
                draw_series_axes(
                    main_native, frame_idx, ours_positions, ours_rotations, "OURS", "ours",
                    vis, left_base_t_camera, left_base_t_right_base,
                )
                draw_series_axes(
                    main_native, frame_idx, canonical_positions, canonical_rotations,
                    "CAN", "corrected", vis, left_base_t_camera, left_base_t_right_base,
                )
            direct_native, _ = vis.read_and_resize(direct_paths[frame_idx])
            if args.mode == "joint":
                left_path = data["left_cam_paths"][int(data["left_cam_indices"][frame_idx])]
                right_path = data["right_cam_paths"][int(data["right_cam_indices"][frame_idx])]
                _, second_native = vis.read_and_resize(left_path)
                _, third_native = vis.read_and_resize(right_path)
                panels = [
                    resize_panel(main_native, "Real D435 + 3 TCP definitions", vis),
                    resize_panel(second_native, "Real left wrist camera", vis),
                    resize_panel(third_native, "Real right wrist camera", vis),
                    resize_panel(direct_native, "Same q in calibrated RoboTwin", vis),
                ]
            else:
                ours_native, _ = vis.read_and_resize(ours_paths[frame_idx])
                canonical_native, _ = vis.read_and_resize(canonical_paths[frame_idx])
                draw_series_axes(
                    direct_native, frame_idx, real_positions, real_rotations, "TARGET", "real",
                    vis, left_base_t_camera, left_base_t_right_base,
                )
                draw_series_axes(
                    ours_native, frame_idx, plan["oursv2_achieved_rtcp_positions"],
                    ours_rotations, "OURS", "ours", vis, left_base_t_camera,
                    left_base_t_right_base, plan["oursv2_ik_success"],
                )
                draw_series_axes(
                    canonical_native, frame_idx, plan["canonical_achieved_rtcp_positions"],
                    canonical_rotations, "CAN", "corrected", vis, left_base_t_camera,
                    left_base_t_right_base, plan["canonical_ik_success"],
                )
                panels = [
                    resize_panel(main_native, "Real D435 + Real-TCP target", vis),
                    resize_panel(direct_native, "Piper real q reference", vis),
                    resize_panel(ours_native, "OursV2 legacy EE-pose IK", vis),
                    resize_panel(canonical_native, "Canonical server-semantic IK", vis),
                ]
            frame = np.concatenate(
                [
                    np.concatenate(panels, axis=1),
                    info_banner(
                        args.task, args.episode, args.mode, frame_idx, count,
                        plan["oursv2_ik_success"], plan["canonical_ik_success"],
                    ),
                    add_cursor(plot, cursor_boxes, frame_idx, count),
                ],
                axis=0,
            )
            if frame.shape != (VIDEO_H, VIDEO_W, 3):
                raise RuntimeError(f"unexpected frame shape {frame.shape}")
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    status = process.wait()
    if status:
        raise RuntimeError(f"ffmpeg failed with code {status}: {stderr}")
    return {
        "task": args.task,
        "episode": args.episode,
        "mode": args.mode,
        "output": str(args.output),
        "frames": count,
        "fps": float(data["fps"]),
        "resolution": [VIDEO_W, VIDEO_H],
        "plot_frame": "shared Piper0515 world",
        "plot_methods": [item[0] for item in METHODS],
        "local_axis_colors": {"+X": "red", "+Y": "green", "+Z": "blue"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--mode", choices=("joint", "eepose"), required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--vis-script", type=Path, required=True)
    parser.add_argument("--direct-sim-dir", type=Path, required=True)
    parser.add_argument("--ours-sim-dir", type=Path, required=True)
    parser.add_argument("--canonical-sim-dir", type=Path, required=True)
    args = parser.parse_args()
    result = render(args)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
