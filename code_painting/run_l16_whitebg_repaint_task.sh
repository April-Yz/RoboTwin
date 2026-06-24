#!/usr/bin/env bash
set -eo pipefail

TASK=${TASK:?set TASK, e.g. pick_diverse_bottles}
GPU=${GPU:-0}
FPS=${FPS:-5}
OVERWRITE=${OVERWRITE:-0}
MASK_IDX=${MASK_IDX:-0}
COMPOSITE_ERODE=${COMPOSITE_ERODE:-0}
BLEND_ALPHA_SIGMA=${BLEND_ALPHA_SIGMA:-1.0}
WHITE_PROMPT=${WHITE_PROMPT:-white background, white floor, white table, blank white area.}

L16=${L16:-/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean}
SAM2=${SAM2:-/home/zaijia001/ssd/inpainting_sam2_robot}
SAM3=${SAM3:-/home/zaijia001/ssd/inpainting_sam3_robot}
STAGE1=${STAGE1:-/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object}
OUTROOT=${OUTROOT:-/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object}

ids() {
  if [[ -n "${IDS:-}" ]]; then
    echo "$IDS"
  else
    find "$L16/$TASK" -path '*/head_cam_plan.mp4' 2>/dev/null \
      | sed 's#.*/foundation_input_\([0-9]*\)/head_cam_plan.mp4#\1#' \
      | sort -n
  fi
}

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate inpainting-sam3-dino3
cd "$SAM3"

for ID in $(ids); do
  BG=$STAGE1/${TASK}/id_${ID}/stage1_human_inpaint/removed_w_mask_rgb_${ID}.mp4
  ROBOT=$L16/${TASK}/foundation_input_${ID}/head_cam_plan.mp4
  OUT=$OUTROOT/${TASK}/id_${ID}_l16_whitebg_human_object
  FINAL=$OUT/final_repainted.mp4
  [[ "$OVERWRITE" == "1" || ! -f "$FINAL" ]] || { echo "[repaint skip] task=$TASK id=$ID final=$FINAL"; continue; }
  [[ -f "$BG" ]] || { echo "[repaint skip] task=$TASK id=$ID missing BG=$BG"; continue; }
  [[ -f "$ROBOT" ]] || { echo "[repaint skip] task=$TASK id=$ID missing ROBOT=$ROBOT"; continue; }
  mkdir -p "$OUT"
  echo "[repaint run] task=$TASK id=$ID mask_idx=$MASK_IDX"
  CUDA_VISIBLE_DEVICES=$GPU python remove_anything_video_sam3_robot.py \
    --input_video "$ROBOT" --target_video "$BG" --output_dir "$OUT" \
    --coords_type key_in --point_coords 10 80 --point_labels 1 \
    --dilate_kernel_size 0 \
    --text_prompt "$WHITE_PROMPT" \
    --box_threshold 0.20 --text_threshold 0.20 \
    --max_mask_area_ratio 1.0 --exclude_bottom_ratio 0.0 \
    --erode_kernel_size 0 --composite_erode_kernel_size 0 --blend_alpha_sigma 0.0 \
    --invert_mask --mask_idx "$MASK_IDX" --fps "$FPS" --device cuda \
    --sam_ckpt "$SAM2/pretrained_models/sam_vit_h_4b8939.pth" \
    --lama_config "$SAM2/lama/configs/prediction/default.yaml" \
    --lama_ckpt "$SAM2/pretrained_models/big-lama" \
    --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
    --vi_ckpt "$SAM2/pretrained_models/sttn.pth" \
    --save_removed_video 0 --save_mask_frames 1 --save_mask_video 1 \
    --save_vis_mask_video 1 --save_vis_box_video 1 --save_target_composite_video 0

  python - "$ROBOT" "$BG" "$OUT" "$FPS" "$COMPOSITE_ERODE" "$BLEND_ALPHA_SIGMA" <<'PYCODE'
import shutil
import sys
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np

source_p, bg_p, out_p, fps_s, erode_s, sigma_s = sys.argv[1:7]
out = Path(out_p)
mask_dir = out / "mask_head_cam_plan"
mask_frames = sorted(mask_dir.glob("*.jpg"))
if not mask_frames:
    raise SystemExit(f"[compose error] no inverted mask frames under {mask_dir}")
fps = int(float(fps_s))
erode = int(erode_s)
sigma = float(sigma_s)
source = iio.get_reader(source_p)
bg_frames = iio.mimread(bg_p, memtest=False)
if not bg_frames:
    raise SystemExit(f"[compose error] no background frames in {bg_p}")
out_len = len(mask_frames)
bg_len = len(bg_frames)
print(f"[compose info] output_frames={out_len} bg_frames={bg_len}; background is sampled proportionally")
frames = []
for idx, mask_p in enumerate(mask_frames):
    src = source.get_data(idx)
    bg_idx = 0 if bg_len == 1 or out_len == 1 else int(round(idx * (bg_len - 1) / (out_len - 1)))
    bg = bg_frames[bg_idx]
    if src.ndim == 2:
        src = np.repeat(src[:, :, None], 3, axis=2)
    if bg.ndim == 2:
        bg = np.repeat(bg[:, :, None], 3, axis=2)
    src = src[:, :, :3]
    bg = bg[:, :, :3]
    h, w = bg.shape[:2]
    src = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"[compose error] failed to read mask {mask_p}")
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)
    if erode > 0:
        kernel = np.ones((erode, erode), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    alpha = mask.astype(np.float32)
    if sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
        alpha *= mask.astype(np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)
    alpha = alpha[:, :, None]
    frames.append((src.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8))
source.close()
if len(frames) != out_len:
    raise SystemExit(f"[compose error] composed {len(frames)} frames, expected {out_len}")
out_video = out / "target_with_original_head_cam_plan.mp4"
iio.mimwrite(out_video, frames, fps=fps, macro_block_size=1)
shutil.copyfile(out_video, out / "final_repainted.mp4")
print(f"[compose ok] {out_video}")
print(f"[final ok] {out / 'final_repainted.mp4'}")
PYCODE
done
