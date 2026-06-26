#!/usr/bin/env python3
"""Color-threshold L16 Stage-2 debug compositor.

This bypasses GroundingDINO/SAM.  It treats border-connected white pixels in
the robot replay as background, inverts that mask to obtain a foreground alpha,
and composites the robot/object foreground onto the Stage-1 inpainted video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np


DEFAULT_L16_ROOT = Path(
    "/home/zaijia001/ssd/RoboTwin/code_painting/"
    "anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean"
)
DEFAULT_STAGE1_ROOT = Path(
    "/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/"
    "stage1_human_object"
)
DEFAULT_OUT_ROOT = Path(
    "/home/zaijia001/ssd/inpainting_sam3_robot/"
    "results_repaint_piper_h2_l16_whitebg_invert/stage2_debug_color/e0_robot_object"
)


def parse_ids(raw: str | None, task_root: Path) -> list[int]:
    if raw:
        ids: list[int] = []
        for part in raw.replace(",", " ").split():
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                ids.extend(range(int(start_s), int(end_s) + 1))
            else:
                ids.append(int(part))
        return sorted(dict.fromkeys(ids))
    found = []
    for p in sorted(task_root.glob("foundation_input_*/head_cam_plan.mp4")):
        found.append(int(p.parent.name.rsplit("_", 1)[-1]))
    return found


def read_video(path: Path) -> list[np.ndarray]:
    frames = iio.mimread(path, memtest=False)
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    return [np.asarray(f[:, :, :3], dtype=np.uint8) for f in frames]


def border_connected(mask: np.ndarray, stride: int) -> np.ndarray:
    """Keep only white candidate regions connected to the image border."""
    h, w = mask.shape
    candidate = mask.astype(np.uint8)
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    step = max(1, int(stride))

    for x in range(0, w, step):
        for y in (0, h - 1):
            if candidate[y, x] == 1:
                cv2.floodFill(candidate, flood_mask, (x, y), 2)
    for y in range(0, h, step):
        for x in (0, w - 1):
            if candidate[y, x] == 1:
                cv2.floodFill(candidate, flood_mask, (x, y), 2)
    return candidate == 2


def white_mask(
    frame_rgb: np.ndarray,
    value_min: int,
    sat_max: int,
    rgb_min: int,
    rgb_delta_max: int,
    border_only: bool,
    border_stride: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    rgb_min_chan = frame_rgb.min(axis=2)
    rgb_delta = frame_rgb.max(axis=2) - frame_rgb.min(axis=2)
    candidate = (
        (val >= value_min)
        & (sat <= sat_max)
        & (rgb_min_chan >= rgb_min)
        & (rgb_delta <= rgb_delta_max)
    )
    if border_only:
        return border_connected(candidate, max(1, int(border_stride)))
    return candidate


def refine_mask(mask: np.ndarray, open_kernel: int, close_kernel: int, erode: int, dilate: int) -> np.ndarray:
    out = mask.astype(np.uint8)
    if open_kernel > 0:
        k = np.ones((open_kernel, open_kernel), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_kernel > 0:
        k = np.ones((close_kernel, close_kernel), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if erode > 0:
        out = cv2.erode(out, np.ones((erode, erode), np.uint8), iterations=1)
    if dilate > 0:
        out = cv2.dilate(out, np.ones((dilate, dilate), np.uint8), iterations=1)
    return out.astype(bool)


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = frame.copy()
    tint = np.zeros_like(out)
    tint[:, :] = color
    alpha = mask.astype(np.float32)[:, :, None] * 0.45
    return (out.astype(np.float32) * (1.0 - alpha) + tint.astype(np.float32) * alpha).astype(np.uint8)


def compose_episode(args: argparse.Namespace, task: str, episode_id: int) -> dict:
    robot_video = args.l16_root / task / f"foundation_input_{episode_id}" / "head_cam_plan.mp4"
    bg_video = (
        args.stage1_root
        / task
        / f"id_{episode_id}"
        / "stage1_human_inpaint"
        / f"removed_w_mask_rgb_{episode_id}.mp4"
    )
    out = args.out_root / task / f"id_{episode_id}_l16_white_color_human_object"
    final = out / "final_repainted.mp4"
    if final.exists() and not args.overwrite:
        print(f"[skip] task={task} id={episode_id} final={final}")
        return {"task": task, "id": episode_id, "skipped": True, "final": str(final)}
    if not robot_video.is_file():
        raise FileNotFoundError(f"missing robot video: {robot_video}")
    if not bg_video.is_file():
        raise FileNotFoundError(f"missing Stage-1 BG video: {bg_video}")

    out.mkdir(parents=True, exist_ok=True)
    mask_dir = out / "mask_head_cam_plan"
    white_dir = out / "white_bg_mask_head_cam_plan"
    mask_dir.mkdir(exist_ok=True)
    white_dir.mkdir(exist_ok=True)

    robot_frames = read_video(robot_video)
    if args.max_frames > 0:
        robot_frames = robot_frames[: args.max_frames]
    bg_frames = read_video(bg_video)
    out_len = len(robot_frames)
    bg_len = len(bg_frames)

    finals: list[np.ndarray] = []
    fg_overlays: list[np.ndarray] = []
    white_overlays: list[np.ndarray] = []
    fg_ratios: list[float] = []
    white_ratios: list[float] = []

    for idx, src in enumerate(robot_frames):
        bg_idx = 0 if bg_len == 1 or out_len == 1 else int(round(idx * (bg_len - 1) / (out_len - 1)))
        bg = bg_frames[bg_idx]
        h, w = bg.shape[:2]
        src_resized = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)

        white = white_mask(
            src_resized,
            value_min=args.white_value_min,
            sat_max=args.white_sat_max,
            rgb_min=args.white_rgb_min,
            rgb_delta_max=args.white_rgb_delta_max,
            border_only=bool(args.border_only),
            border_stride=args.border_stride,
        )
        white = refine_mask(
            white,
            open_kernel=args.white_open_kernel,
            close_kernel=args.white_close_kernel,
            erode=args.white_erode_kernel,
            dilate=args.white_dilate_kernel,
        )
        fg = ~white
        fg = refine_mask(
            fg,
            open_kernel=args.fg_open_kernel,
            close_kernel=args.fg_close_kernel,
            erode=args.fg_erode_kernel,
            dilate=args.fg_dilate_kernel,
        )
        alpha = fg.astype(np.float32)
        if args.blend_alpha_sigma > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), args.blend_alpha_sigma)
            alpha *= fg.astype(np.float32)
            alpha = np.clip(alpha, 0.0, 1.0)
        final = src_resized.astype(np.float32) * alpha[:, :, None] + bg.astype(np.float32) * (1.0 - alpha[:, :, None])

        finals.append(final.astype(np.uint8))
        fg_overlays.append(overlay_mask(src_resized, fg, (255, 0, 0)))
        white_overlays.append(overlay_mask(src_resized, white, (0, 255, 255)))
        fg_ratios.append(float(fg.mean()))
        white_ratios.append(float(white.mean()))
        if args.save_mask_frames:
            iio.imwrite(mask_dir / f"{idx:06d}.jpg", (fg.astype(np.uint8) * 255))
            iio.imwrite(white_dir / f"{idx:06d}.jpg", (white.astype(np.uint8) * 255))

    iio.mimwrite(out / "final_repainted.mp4", finals, fps=args.fps, macro_block_size=1)
    iio.mimwrite(out / "target_with_original_head_cam_plan.mp4", finals, fps=args.fps, macro_block_size=1)
    iio.mimwrite(out / "w_mask_head_cam_plan.mp4", fg_overlays, fps=args.fps, macro_block_size=1)
    iio.mimwrite(out / "w_white_bg_mask_head_cam_plan.mp4", white_overlays, fps=args.fps, macro_block_size=1)

    manifest = {
        "task": task,
        "id": episode_id,
        "robot_video": str(robot_video),
        "stage1_bg_video": str(bg_video),
        "output_dir": str(out),
        "fps": args.fps,
        "params": {
            "white_value_min": args.white_value_min,
            "white_sat_max": args.white_sat_max,
            "white_rgb_min": args.white_rgb_min,
            "white_rgb_delta_max": args.white_rgb_delta_max,
            "border_only": bool(args.border_only),
            "border_stride": args.border_stride,
            "white_open_kernel": args.white_open_kernel,
            "white_close_kernel": args.white_close_kernel,
            "white_erode_kernel": args.white_erode_kernel,
            "white_dilate_kernel": args.white_dilate_kernel,
            "fg_open_kernel": args.fg_open_kernel,
            "fg_close_kernel": args.fg_close_kernel,
            "fg_erode_kernel": args.fg_erode_kernel,
            "fg_dilate_kernel": args.fg_dilate_kernel,
            "blend_alpha_sigma": args.blend_alpha_sigma,
        },
        "frames": {
            "robot": len(robot_frames),
            "stage1_bg": len(bg_frames),
            "output": len(finals),
        },
        "foreground_ratio": {
            "min": min(fg_ratios),
            "max": max(fg_ratios),
            "mean": float(np.mean(fg_ratios)),
        },
        "white_bg_ratio": {
            "min": min(white_ratios),
            "max": max(white_ratios),
            "mean": float(np.mean(white_ratios)),
        },
    }
    (out / "color_white_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[ok] task={task} id={episode_id} final={out / 'final_repainted.mp4'} "
        f"fg_mean={manifest['foreground_ratio']['mean']:.3f} white_mean={manifest['white_bg_ratio']['mean']:.3f}"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L16 Stage-2 debug by color-threshold white-background removal.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--ids", default="", help="Episode ids, e.g. '0 1 2 3 4' or '0-4'.")
    parser.add_argument("--l16-root", type=Path, default=DEFAULT_L16_ROOT)
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="Only process the first N robot frames; 0 means all frames.")
    parser.add_argument("--white-value-min", type=int, default=185)
    parser.add_argument("--white-sat-max", type=int, default=55)
    parser.add_argument("--white-rgb-min", type=int, default=165)
    parser.add_argument("--white-rgb-delta-max", type=int, default=70)
    parser.add_argument("--border-only", type=int, default=1, choices=[0, 1])
    parser.add_argument("--border-stride", type=int, default=2)
    parser.add_argument("--white-open-kernel", type=int, default=0)
    parser.add_argument("--white-close-kernel", type=int, default=5)
    parser.add_argument("--white-erode-kernel", type=int, default=0)
    parser.add_argument("--white-dilate-kernel", type=int, default=3)
    parser.add_argument("--fg-open-kernel", type=int, default=0)
    parser.add_argument("--fg-close-kernel", type=int, default=3)
    parser.add_argument("--fg-erode-kernel", type=int, default=0)
    parser.add_argument("--fg-dilate-kernel", type=int, default=0)
    parser.add_argument("--blend-alpha-sigma", type=float, default=1.0)
    parser.add_argument("--save-mask-frames", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids = parse_ids(args.ids, args.l16_root / args.task)
    if not ids:
        raise RuntimeError(f"no ids found for task={args.task}")
    for episode_id in ids:
        compose_episode(args, args.task, episode_id)


if __name__ == "__main__":
    main()
