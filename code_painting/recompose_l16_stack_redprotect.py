#!/usr/bin/env python3
"""Recompose stack_cups L16 whitebg outputs with red/pink cup protection.

This is a post-processing debug path. It reads the existing white-background
inverted foreground alpha from the stack B output, ORs in red/pink cup pixels
from the L16 source video, and writes a separate redprotect result tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import imageio.v2 as iio
import numpy as np

REPO_ROOT = Path('/home/zaijia001/ssd/RoboTwin')
L16_ROOT = REPO_ROOT / 'code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean'
STAGE1_ROOT = Path('/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/B_points_negative')
BASE_MASK_ROOT = Path('/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative')
OUT_ROOT = Path('/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative_redprotect')
TASK = 'stack_cups'


def parse_ids(spec: str | None, l16_root: Path) -> list[int]:
    if spec:
        ids: list[int] = []
        for part in spec.split(','):
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                a, b = part.split('-', 1)
                start, end = int(a), int(b)
                step = 1 if end >= start else -1
                ids.extend(range(start, end + step, step))
            else:
                ids.append(int(part))
        return sorted(dict.fromkeys(ids))
    ids = []
    for path in (l16_root / TASK).glob('foundation_input_*/head_cam_plan.mp4'):
        try:
            ids.append(int(path.parent.name.replace('foundation_input_', '')))
        except ValueError:
            pass
    return sorted(set(ids))


def open_cap(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f'[error] failed to open video: {path}')
    return cap


def read_cap_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f'failed to read frame {index}')
    return frame


def red_pink_mask_bgr(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(frame)
    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)

    red_hsv = (((h <= args.red_hue_low) | (h >= args.red_hue_high)) & (s >= args.min_sat) & (v >= args.min_val))
    red_dominant = (r16 >= args.min_red) & (r16 >= g16 + args.rg_margin) & (r16 >= b16 + args.rb_margin)
    # Light pink cups can have high blue as well; keep red distinctly above green.
    pink = (r16 >= args.min_pink_red) & (r16 >= g16 + args.pink_rg_margin) & (b16 >= g16 + args.pink_bg_margin)
    mask = (red_hsv | red_dominant | pink).astype(np.uint8) * 255

    if args.protect_open_kernel > 0:
        k = np.ones((args.protect_open_kernel, args.protect_open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if args.protect_close_kernel > 0:
        k = np.ones((args.protect_close_kernel, args.protect_close_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if args.protect_dilate_kernel > 0:
        k = np.ones((args.protect_dilate_kernel, args.protect_dilate_kernel), np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    out = frame.copy()
    on = mask > 127
    color_arr = np.array(color, dtype=np.float32)
    out[on] = np.clip(out[on].astype(np.float32) * (1.0 - alpha) + color_arr * alpha, 0, 255).astype(np.uint8)
    return out


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def write_video_cv2(path: Path, frames: Iterable[np.ndarray], fps: float, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        raise SystemExit(f'[error] failed to create {path}')
    for frame in frames:
        writer.write(frame)
    writer.release()


def compose_one(video_id: int, args: argparse.Namespace) -> dict[str, object] | None:
    robot_p = args.l16_root / TASK / f'foundation_input_{video_id}' / 'head_cam_plan.mp4'
    bg_p = args.stage1_root / TASK / f'id_{video_id}' / 'stage1_human_inpaint' / f'removed_w_mask_rgb_{video_id}.mp4'
    base_dir = args.base_mask_root / TASK / f'id_{video_id}_l16_whitebg_human_object'
    base_mask_dir = base_dir / 'mask_head_cam_plan'
    out_dir = args.output_root / TASK / f'id_{video_id}_l16_whitebg_human_object'
    final_p = out_dir / 'final_repainted.mp4'

    if final_p.is_file() and not args.overwrite:
        print(f'[skip existing] id={video_id} final={final_p}')
        return None
    missing = [p for p in [robot_p, bg_p, base_mask_dir] if not p.exists()]
    if missing:
        print(f'[skip missing] id={video_id}: ' + ', '.join(str(p) for p in missing))
        return None
    mask_frames = sorted(base_mask_dir.glob('*.jpg'))
    if not mask_frames:
        print(f'[skip missing] id={video_id}: no mask frames under {base_mask_dir}')
        return None

    robot_cap = open_cap(robot_p)
    bg_cap = open_cap(bg_p)
    robot_count = int(robot_cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(mask_frames))
    bg_count = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    fps = float(args.fps)
    frame_indices = list(range(len(mask_frames)))
    if args.max_frames > 0:
        frame_indices = frame_indices[:args.max_frames]

    out_mask_dir = out_dir / 'mask_head_cam_plan'
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    final_frames: list[np.ndarray] = []
    mask_video_frames: list[np.ndarray] = []
    wmask_frames: list[np.ndarray] = []
    protect_overlay_frames: list[np.ndarray] = []
    debug_frames: list[np.ndarray] = []
    old_alpha_ratios = []
    protect_ratios = []
    new_alpha_ratios = []
    red_recovered_ratios = []

    for idx in frame_indices:
        src = read_cap_frame(robot_cap, min(idx, max(0, robot_count - 1)))
        bg_idx = 0 if bg_count <= 1 or len(mask_frames) <= 1 else int(round(idx * (bg_count - 1) / (len(mask_frames) - 1)))
        bg = read_cap_frame(bg_cap, bg_idx)
        if src.shape[:2] != bg.shape[:2]:
            src = cv2.resize(src, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_LINEAR)

        old_mask = cv2.imread(str(mask_frames[idx]), cv2.IMREAD_GRAYSCALE)
        if old_mask is None:
            raise SystemExit(f'[error] failed to read mask {mask_frames[idx]}')
        if old_mask.shape[:2] != bg.shape[:2]:
            old_mask = cv2.resize(old_mask, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_NEAREST)
        old_mask = ((old_mask > 127).astype(np.uint8)) * 255
        protect = red_pink_mask_bgr(src, args)
        if protect.shape[:2] != bg.shape[:2]:
            protect = cv2.resize(protect, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_NEAREST)
        new_mask = np.maximum(old_mask, protect)

        alpha = (new_mask > 127).astype(np.float32)[:, :, None]
        final = (src.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        final_frames.append(final)
        mask_video_frames.append(cv2.cvtColor(new_mask, cv2.COLOR_GRAY2BGR))
        wmask_frames.append(overlay_mask(src, new_mask, (0, 255, 255)))
        protect_overlay_frames.append(overlay_mask(src, protect, (0, 0, 255)))

        red_region = protect > 127
        red_pixels = int(red_region.sum())
        recovered = int(((new_mask > 127) & red_region).sum())
        red_recovered_ratios.append(recovered / red_pixels if red_pixels else 0.0)
        old_alpha_ratios.append(float((old_mask > 127).mean()))
        protect_ratios.append(float((protect > 127).mean()))
        new_alpha_ratios.append(float((new_mask > 127).mean()))

        if args.save_debug_video and len(debug_frames) < args.debug_frames:
            size = (args.panel_width, args.panel_height)
            panels = [
                label(cv2.resize(src, size), '1 L16 source'),
                label(cv2.resize(bg, size), '2 Stage1 BG'),
                label(cv2.resize(overlay_mask(src, old_mask, (0, 255, 255)), size), '3 old alpha'),
                label(cv2.resize(overlay_mask(src, protect, (0, 0, 255)), size), '4 red protect'),
                label(cv2.resize(overlay_mask(src, new_mask, (255, 0, 255)), size), '5 new alpha'),
                label(cv2.resize(final, size), '6 redprotect final'),
            ]
            top = np.hstack(panels[:3])
            bottom = np.hstack(panels[3:])
            debug_frames.append(np.vstack([top, bottom]))

        cv2.imwrite(str(out_mask_dir / f'{idx:06d}.jpg'), new_mask)

    robot_cap.release()
    bg_cap.release()
    if not final_frames:
        print(f'[skip empty] id={video_id}')
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    write_video_cv2(out_dir / 'mask_head_cam_plan.mp4', mask_video_frames, fps, (final_frames[0].shape[1], final_frames[0].shape[0]))
    write_video_cv2(out_dir / 'w_mask_head_cam_plan.mp4', wmask_frames, fps, (final_frames[0].shape[1], final_frames[0].shape[0]))
    write_video_cv2(out_dir / 'red_protect_overlay.mp4', protect_overlay_frames, fps, (final_frames[0].shape[1], final_frames[0].shape[0]))
    write_video_cv2(out_dir / 'target_with_original_head_cam_plan.mp4', final_frames, fps, (final_frames[0].shape[1], final_frames[0].shape[0]))
    write_video_cv2(final_p, final_frames, fps, (final_frames[0].shape[1], final_frames[0].shape[0]))
    debug_p = None
    if args.save_debug_video and debug_frames:
        debug_p = out_dir / f'redprotect_debug_stack_cups_id{video_id}.mp4'
        write_video_cv2(debug_p, debug_frames, fps, (args.panel_width * 3, args.panel_height * 2))

    manifest = {
        'task': TASK,
        'id': video_id,
        'mode': 'whitebg_inverted_mask_plus_red_pink_color_protect',
        'robot': str(robot_p),
        'stage1_bg': str(bg_p),
        'base_mask_dir': str(base_mask_dir),
        'output_dir': str(out_dir),
        'final_repainted': str(final_p),
        'debug_video': str(debug_p) if debug_p else None,
        'frames': len(final_frames),
        'old_alpha_mean_ratio': float(np.mean(old_alpha_ratios)),
        'red_protect_mean_ratio': float(np.mean(protect_ratios)),
        'new_alpha_mean_ratio': float(np.mean(new_alpha_ratios)),
        'red_recovered_mean_ratio': float(np.mean(red_recovered_ratios)),
        'red_mask_params': {
            'red_hue_low': args.red_hue_low,
            'red_hue_high': args.red_hue_high,
            'min_sat': args.min_sat,
            'min_val': args.min_val,
            'min_red': args.min_red,
            'rg_margin': args.rg_margin,
            'rb_margin': args.rb_margin,
            'min_pink_red': args.min_pink_red,
            'pink_rg_margin': args.pink_rg_margin,
            'pink_bg_margin': args.pink_bg_margin,
            'protect_open_kernel': args.protect_open_kernel,
            'protect_close_kernel': args.protect_close_kernel,
            'protect_dilate_kernel': args.protect_dilate_kernel,
        },
    }
    (out_dir / 'redprotect_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(
        f"[ok] id={video_id} final={final_p} old_alpha={manifest['old_alpha_mean_ratio']:.4f} "
        f"red_protect={manifest['red_protect_mean_ratio']:.4f} new_alpha={manifest['new_alpha_mean_ratio']:.4f} "
        f"red_recovered={manifest['red_recovered_mean_ratio']:.4f}"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', default='0', help='ID spec such as 0,1,2 or 0-4. Use empty string with --all to process all.')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--l16_root', type=Path, default=L16_ROOT)
    parser.add_argument('--stage1_root', type=Path, default=STAGE1_ROOT)
    parser.add_argument('--base_mask_root', type=Path, default=BASE_MASK_ROOT)
    parser.add_argument('--output_root', type=Path, default=OUT_ROOT)
    parser.add_argument('--fps', type=float, default=5.0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max_frames', type=int, default=0, help='0 means all frames.')
    parser.add_argument('--save_debug_video', type=int, choices=(0, 1), default=1)
    parser.add_argument('--debug_frames', type=int, default=120)
    parser.add_argument('--panel_width', type=int, default=426)
    parser.add_argument('--panel_height', type=int, default=320)

    parser.add_argument('--red_hue_low', type=int, default=14)
    parser.add_argument('--red_hue_high', type=int, default=166)
    parser.add_argument('--min_sat', type=int, default=45)
    parser.add_argument('--min_val', type=int, default=35)
    parser.add_argument('--min_red', type=int, default=45)
    parser.add_argument('--rg_margin', type=int, default=18)
    parser.add_argument('--rb_margin', type=int, default=18)
    parser.add_argument('--min_pink_red', type=int, default=115)
    parser.add_argument('--pink_rg_margin', type=int, default=18)
    parser.add_argument('--pink_bg_margin', type=int, default=-8)
    parser.add_argument('--protect_open_kernel', type=int, default=0)
    parser.add_argument('--protect_close_kernel', type=int, default=5)
    parser.add_argument('--protect_dilate_kernel', type=int, default=7)
    args = parser.parse_args()

    args.l16_root = args.l16_root.expanduser().resolve()
    args.stage1_root = args.stage1_root.expanduser().resolve()
    args.base_mask_root = args.base_mask_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    ids = parse_ids(None if args.all else args.ids, args.l16_root)
    if not ids:
        raise SystemExit('[error] no ids selected')
    made = 0
    for video_id in ids:
        if compose_one(video_id, args) is not None:
            made += 1
    print(f'[summary] processed={made} selected={len(ids)} output_root={args.output_root}')
    return 0 if made > 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
