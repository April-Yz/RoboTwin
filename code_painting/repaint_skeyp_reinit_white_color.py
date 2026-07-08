#!/usr/bin/env python3
"""SKEYP/reinit Stage-2 compositor using white-background inversion.

This mirrors the L16 color-white route, but reads reinit-style replay files:
`<retarget_root>/<task>/id<ID>_d435_z005/zed_replay_d435.mp4`.
White background pixels are detected by color thresholding; the inverse mask is
used as foreground alpha and composited onto the Stage-1 hands-only background.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np

from repaint_l16_white_color_debug import read_video, white_mask, refine_mask, overlay_mask


DEFAULT_RETARGET_ROOT = Path(
    "/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/"
    "skeyp_v2_reinit_gripperonly/h2_pure_d435_selected25"
)
DEFAULT_STAGE1_ROOT = Path(
    "/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_skeyp/"
    "v2_reinit_gripperonly/stage1"
)
DEFAULT_OUT_ROOT = Path(
    "/home/zaijia001/ssd/inpainting_sam3_robot/"
    "results_repaint_piper_h2_skeyp_visible_reinit/v2_reinit_whitebg/e0_robot_color"
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
    for p in sorted(task_root.glob("id*_d435_z005/zed_replay_d435.mp4")):
        name = p.parent.name
        found.append(int(name.split("_", 1)[0][2:]))
    return found


def compose_episode(args: argparse.Namespace, task: str, episode_id: int) -> dict:
    robot_video = args.retarget_root / task / f"id{episode_id}_d435_z005" / "zed_replay_d435.mp4"
    bg_video = (
        args.stage1_root
        / task
        / f"id_{episode_id}"
        / "stage1_human_inpaint"
        / f"removed_w_mask_rgb_{episode_id}.mp4"
    )
    out = args.out_root / task / f"id_{episode_id}_skeyp_whitebg"
    final = out / "final_repainted.mp4"
    if final.exists() and not args.overwrite:
        print(f"[skip] task={task} id={episode_id} final={final}")
        return {"task": task, "id": episode_id, "skipped": True, "final": str(final)}
    if not robot_video.is_file():
        raise FileNotFoundError(f"missing robot replay video: {robot_video}")
    if not bg_video.is_file():
        raise FileNotFoundError(f"missing Stage-1 BG video: {bg_video}")

    out.mkdir(parents=True, exist_ok=True)
    mask_dir = out / "mask_zed_replay_d435"
    white_dir = out / "white_bg_mask_zed_replay_d435"
    if args.save_mask_frames:
        mask_dir.mkdir(exist_ok=True)
        white_dir.mkdir(exist_ok=True)

    robot_frames_full = read_video(robot_video)
    original_robot_len = len(robot_frames_full)
    if args.max_frames > 0:
        robot_frames = robot_frames_full[: args.max_frames]
        frame_limit_note = f"preview_first_{args.max_frames}_robot_frames"
    else:
        robot_frames = robot_frames_full
        frame_limit_note = "full_robot_video"
    bg_frames = read_video(bg_video)
    out_len = len(robot_frames)
    bg_len = len(bg_frames)
    print(
        f"[align] task={task} id={episode_id} output_frames={out_len} "
        f"original_robot_frames={original_robot_len} stage1_bg_frames={bg_len}; "
        "Stage-1 BG is sampled proportionally"
    )

    finals: list[np.ndarray] = []
    fg_masks: list[np.ndarray] = []
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
        final_frame = src_resized.astype(np.float32) * alpha[:, :, None] + bg.astype(np.float32) * (1.0 - alpha[:, :, None])

        finals.append(final_frame.astype(np.uint8))
        fg_masks.append((fg.astype(np.uint8) * 255))
        fg_overlays.append(overlay_mask(src_resized, fg, (255, 0, 0)))
        white_overlays.append(overlay_mask(src_resized, white, (0, 255, 255)))
        fg_ratios.append(float(fg.mean()))
        white_ratios.append(float(white.mean()))
        if args.save_mask_frames:
            iio.imwrite(mask_dir / f"{idx:06d}.jpg", (fg.astype(np.uint8) * 255))
            iio.imwrite(white_dir / f"{idx:06d}.jpg", (white.astype(np.uint8) * 255))

    iio.mimwrite(out / "final_repainted.mp4", finals, fps=args.fps, macro_block_size=1)
    iio.mimwrite(out / "target_with_original_zed_replay_d435.mp4", finals, fps=args.fps, macro_block_size=1)
    iio.mimwrite(out / "mask_zed_replay_d435.mp4", fg_masks, fps=args.fps, macro_block_size=1)
    iio.mimwrite(out / "w_mask_zed_replay_d435.mp4", fg_overlays, fps=args.fps, macro_block_size=1)
    iio.mimwrite(out / "w_white_bg_mask_zed_replay_d435.mp4", white_overlays, fps=args.fps, macro_block_size=1)

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
        "alignment": {
            "mode": "robot_frames_with_proportional_stage1_sampling",
            "frame_limit": frame_limit_note,
            "note": "Output follows robot frames (or --max-frames preview); Stage-1 background is sampled by round(i * (bg_len - 1) / (out_len - 1)).",
        },
        "frames": {
            "original_robot": original_robot_len,
            "processed_robot": len(robot_frames),
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
        f"frames={len(finals)}/{original_robot_len} fg_mean={manifest['foreground_ratio']['mean']:.3f} "
        f"white_mean={manifest['white_bg_ratio']['mean']:.3f}"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SKEYP reinit Stage-2 by white-background color inversion.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--ids", default="", help="Episode ids, e.g. '0 1 2 3 4' or '0-4'.")
    parser.add_argument("--retarget-root", type=Path, default=DEFAULT_RETARGET_ROOT)
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--white-value-min", type=int, default=210)
    parser.add_argument("--white-sat-max", type=int, default=45)
    parser.add_argument("--white-rgb-min", type=int, default=200)
    parser.add_argument("--white-rgb-delta-max", type=int, default=55)
    parser.add_argument("--border-only", type=int, default=1, choices=[0, 1])
    parser.add_argument("--border-stride", type=int, default=2)
    parser.add_argument("--white-open-kernel", type=int, default=0)
    parser.add_argument("--white-close-kernel", type=int, default=5)
    parser.add_argument("--white-erode-kernel", type=int, default=0)
    parser.add_argument("--white-dilate-kernel", type=int, default=2)
    parser.add_argument("--fg-open-kernel", type=int, default=0)
    parser.add_argument("--fg-close-kernel", type=int, default=3)
    parser.add_argument("--fg-erode-kernel", type=int, default=0)
    parser.add_argument("--fg-dilate-kernel", type=int, default=0)
    parser.add_argument("--blend-alpha-sigma", type=float, default=1.0)
    parser.add_argument("--save-mask-frames", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids = parse_ids(args.ids, args.retarget_root / args.task)
    if not ids:
        raise RuntimeError(f"no ids found for task={args.task}")
    for episode_id in ids:
        compose_episode(args, args.task, episode_id)


if __name__ == "__main__":
    main()
