#!/usr/bin/env bash
set -eo pipefail

GPU=${GPU:-1}
IDS=${IDS:-"0 1 2 3 4"}
MAX_FRAMES=${MAX_FRAMES:-300}
VARIANTS=${VARIANTS:-"A_protect_dino B_points_negative C_hsv_green_protect D_tight_dino"}

ROBOWTIN=${ROBOWTIN:-/home/zaijia001/ssd/RoboTwin}
SAM2=${SAM2:-/home/zaijia001/ssd/inpainting_sam2_robot}
INPUT_ROOT=${INPUT_ROOT:-/home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_input}
OUTROOT=${OUTROOT:-/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants}

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate inpainting-sam2-r1
cd "$SAM2"

CUDA_VISIBLE_DEVICES=$GPU python "$ROBOWTIN/code_painting/l16_stack_cups_debug_variants.py" \
  --ids "$IDS" \
  --input_root "$INPUT_ROOT" \
  --output_root "$OUTROOT" \
  --sam2_root "$SAM2" \
  --sttn_ckpt "$SAM2/pretrained_models/sttn.pth" \
  --device cuda \
  --max_frames "$MAX_FRAMES" \
  --variants "$VARIANTS"
