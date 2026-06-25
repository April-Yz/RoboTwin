#!/usr/bin/env python3
"""Build a debug montage for L16 white-background inverted-mask repaint.

Panels show the saved inverted foreground alpha mask and its inverse.
With run_l16_whitebg_repaint_task.sh, --invert_mask means mask_head_cam_plan/*.jpg
is already the robot/object foreground alpha used for compositing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path('/home/zaijia001/ssd/RoboTwin')
L16_ROOT = REPO_ROOT / 'code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean'
DEFAULT_STAGE1_ROOT = Path('/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object')
STACK_B_STAGE1_ROOT = Path('/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/B_points_negative')
DEFAULT_FINAL_ROOT = Path('/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object')
STACK_B_FINAL_ROOT = Path('/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative')
DEFAULT_OUTPUT_ROOT = REPO_ROOT / 'code_painting/l16_whitebg_mask_debug'


def auto_stage1_root(task: str) -> Path:
    if task == 'stack_cups' and (STACK_B_STAGE1_ROOT / task).is_dir():
        return STACK_B_STAGE1_ROOT
    return DEFAULT_STAGE1_ROOT


def auto_final_root(task: str) -> Path:
    if task == 'stack_cups' and (STACK_B_FINAL_ROOT / task).is_dir():
        return STACK_B_FINAL_ROOT
    return DEFAULT_FINAL_ROOT


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f'[error] failed to open video: {path}')
    return cap


def read_frame(cap: cv2.VideoCapture, frame_idx: int, size: tuple[int, int]) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def mask_to_bgr(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    m = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    on = mask > 127
    out = frame.copy()
    color_arr = np.array(color, dtype=np.float32)
    out[on] = np.clip(out[on].astype(np.float32) * (1 - alpha) + color_arr * alpha, 0, 255).astype(np.uint8)
    return out


def stack_panels(panels: list[np.ndarray], columns: int) -> np.ndarray:
    h, w = panels[0].shape[:2]
    rows = int(np.ceil(len(panels) / columns))
    canvas = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for i, panel in enumerate(panels):
        r = i // columns
        c = i % columns
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = panel
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='stack_cups')
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--stage1_root', type=Path, default=None)
    parser.add_argument('--final_root', type=Path, default=None)
    parser.add_argument('--l16_root', type=Path, default=L16_ROOT)
    parser.add_argument('--output_root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--fps', type=float, default=5.0)
    parser.add_argument('--panel_width', type=int, default=426)
    parser.add_argument('--panel_height', type=int, default=320)
    parser.add_argument('--max_frames', type=int, default=0, help='0 means all mask frames.')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--saved_mask_is_foreground', type=int, choices=(0, 1), default=1, help='Current L16 whitebg pipeline saves inverted foreground masks because --invert_mask is enabled.')
    args = parser.parse_args()

    stage1_root = (args.stage1_root or auto_stage1_root(args.task)).expanduser().resolve()
    final_root = (args.final_root or auto_final_root(args.task)).expanduser().resolve()
    l16_root = args.l16_root.expanduser().resolve()
    out_dir = args.output_root.expanduser().resolve() / args.task / f'id_{args.id}'
    out_dir.mkdir(parents=True, exist_ok=True)

    robot_p = l16_root / args.task / f'foundation_input_{args.id}' / 'head_cam_plan.mp4'
    bg_p = stage1_root / args.task / f'id_{args.id}' / 'stage1_human_inpaint' / f'removed_w_mask_rgb_{args.id}.mp4'
    result_dir = final_root / args.task / f'id_{args.id}_l16_whitebg_human_object'
    final_p = result_dir / 'final_repainted.mp4'
    wmask_p = result_dir / 'w_mask_head_cam_plan.mp4'
    mask_video_p = result_dir / 'mask_head_cam_plan.mp4'
    mask_dir = result_dir / 'mask_head_cam_plan'
    mask_frames = sorted(mask_dir.glob('*.jpg'))

    required = [robot_p, bg_p, final_p]
    missing = [str(p) for p in required if not p.is_file()]
    if not mask_frames:
        missing.append(str(mask_dir) + '/*.jpg')
    if missing:
        raise SystemExit('[error] missing inputs:\n  ' + '\n  '.join(missing))

    size = (args.panel_width, args.panel_height)
    caps = {
        'robot': open_video(robot_p),
        'bg': open_video(bg_p),
        'final': open_video(final_p),
    }
    if wmask_p.is_file():
        caps['wmask'] = open_video(wmask_p)
    if mask_video_p.is_file():
        caps['mask_video'] = open_video(mask_video_p)

    frame_indices = list(range(0, len(mask_frames), max(1, args.stride)))
    if args.max_frames > 0:
        frame_indices = frame_indices[:args.max_frames]
    if not frame_indices:
        raise SystemExit('[error] no frames selected')

    output_video = out_dir / f'whitebg_invert_debug_{args.task}_id{args.id}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video), fourcc, args.fps, (args.panel_width * 3, args.panel_height * 2))
    if not writer.isOpened():
        raise SystemExit(f'[error] failed to create {output_video}')

    white_ratios = []
    foreground_ratios = []
    for frame_idx in frame_indices:
        robot = read_frame(caps['robot'], frame_idx, size)
        bg_idx = int(round(frame_idx * max(0, int(caps['bg'].get(cv2.CAP_PROP_FRAME_COUNT)) - 1) / max(1, len(mask_frames) - 1)))
        bg = read_frame(caps['bg'], bg_idx, size)
        final = read_frame(caps['final'], frame_idx, size)

        mask = cv2.imread(str(mask_frames[frame_idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise SystemExit(f'[error] failed to read mask frame {mask_frames[frame_idx]}')
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        saved_mask = (mask > 127).astype(np.uint8) * 255
        inverse_mask = 255 - saved_mask
        if args.saved_mask_is_foreground:
            fg_mask = saved_mask
            bg_check_mask = inverse_mask
            saved_label = '3 saved alpha overlay'
            binary_label = '4 saved alpha binary'
            inverse_label = '5 inverse bg check'
        else:
            bg_check_mask = saved_mask
            fg_mask = inverse_mask
            saved_label = '3 raw WHITE overlay'
            binary_label = '4 raw WHITE binary'
            inverse_label = '5 inverted foreground'
        white_ratios.append(float((bg_check_mask > 0).mean()))
        foreground_ratios.append(float((fg_mask > 0).mean()))

        if 'wmask' in caps:
            saved_vis = read_frame(caps['wmask'], frame_idx, size)
        else:
            saved_vis = overlay_mask(robot, saved_mask, (0, 255, 255))
        if 'mask_video' in caps:
            saved_binary = read_frame(caps['mask_video'], frame_idx, size)
        else:
            saved_binary = mask_to_bgr(saved_mask, size)

        foreground_vis = overlay_mask(robot, fg_mask, (0, 0, 255))
        inverse_vis = overlay_mask(robot, bg_check_mask, (255, 128, 0))
        panels = [
            label(robot, '1 L16 head source'),
            label(bg, '2 Stage1 BG'),
            label(saved_vis, saved_label),
            label(saved_binary, binary_label),
            label(foreground_vis if args.saved_mask_is_foreground else inverse_vis, inverse_label),
            label(final, '6 final repaint'),
        ]
        writer.write(stack_panels(panels, columns=3))

    writer.release()
    for cap in caps.values():
        cap.release()

    manifest = {
        'task': args.task,
        'id': args.id,
        'output_video': str(output_video),
        'note': 'Current run_l16_whitebg_repaint_task.sh uses --invert_mask, so mask_head_cam_plan/*.jpg is saved foreground alpha. Panel 3/4 show the saved alpha; panel 5 shows the inverse/background check or inverted foreground depending on --saved_mask_is_foreground.',
        'paths': {
            'robot': str(robot_p),
            'stage1_bg': str(bg_p),
            'result_dir': str(result_dir),
            'final': str(final_p),
            'mask_dir': str(mask_dir),
            'w_mask_video': str(wmask_p),
        },
        'frames_written': len(frame_indices),
        'mean_background_check_ratio': float(np.mean(white_ratios)) if white_ratios else 0.0,
        'mean_foreground_alpha_ratio': float(np.mean(foreground_ratios)) if foreground_ratios else 0.0,
        'saved_mask_is_foreground': bool(args.saved_mask_is_foreground),
    }
    manifest_p = output_video.with_suffix('.json')
    manifest_p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'[ok] {output_video}')
    print(f'[ok] {manifest_p}')
    print(f"[ratios] foreground_alpha={manifest['mean_foreground_alpha_ratio']:.4f} background_check={manifest['mean_background_check_ratio']:.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
