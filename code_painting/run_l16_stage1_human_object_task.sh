#!/usr/bin/env bash
set -eo pipefail

TASK=${TASK:?set TASK, e.g. pick_diverse_bottles}
GPU=${GPU:-0}
FPS=${FPS:-5}
OVERWRITE=${OVERWRITE:-0}
DILATE=${DILATE:-100}

L16=${L16:-/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean}
SAM2=${SAM2:-/home/zaijia001/ssd/inpainting_sam2_robot}
OUTROOT=${OUTROOT:-/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object}

ids() {
  if [[ -n "${IDS:-}" ]]; then
    echo "$IDS"
  else
    find "$L16/$TASK" -path '*/head_cam_plan.mp4' 2>/dev/null \
      | sed 's#.*/foundation_input_\([0-9]*\)/head_cam_plan.mp4#\1#' \
      | sort -n
  fi
}

human_prompt() {
  case "$TASK" in
    pick_diverse_bottles) echo "arms, hands, wrists, watch, left bottle, right bottle, bottles." ;;
    place_bread_basket) echo "arms, hands, wrists, watch, bread, basket." ;;
    stack_cups) echo "arms, hands, wrists, watch, left light pink cup, right dark red cup." ;;
    handover_bottle) echo "arms, hands, wrists, watch, right bottle, bottle." ;;
    pnp_bread) echo "arms, hands, wrists, watch, bread." ;;
    pnp_tray) echo "arms, hands, wrists, watch, left dark red cup, right bottle." ;;
    *) echo "[error] unknown TASK=$TASK" >&2; return 1 ;;
  esac
}

source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate inpainting-sam2-r1
cd "$SAM2"

for ID in $(ids); do
  HUMAN=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_input/rgb_${ID}.mp4
  S1OUT=$OUTROOT/${TASK}/id_${ID}/stage1_human_inpaint
  BG=$S1OUT/removed_w_mask_rgb_${ID}.mp4
  [[ "$OVERWRITE" == "1" || ! -f "$BG" ]] || { echo "[stage1 skip] task=$TASK id=$ID bg=$BG"; continue; }
  [[ -f "$HUMAN" ]] || { echo "[stage1 skip] task=$TASK id=$ID missing HUMAN=$HUMAN"; continue; }
  mkdir -p "$S1OUT"
  echo "[stage1 run] task=$TASK id=$ID dilate=$DILATE"
  CUDA_VISIBLE_DEVICES=$GPU python remove_anything_video_sam2.py \
    --input_video "$HUMAN" \
    --coords_type key_in --point_coords 10 80 --point_labels 1 \
    --dilate_kernel_size "$DILATE" \
    --text_prompt "$(human_prompt)" \
    --box_threshold 0.35 --text_threshold 0.25 \
    --output_dir "$S1OUT" \
    --sam_ckpt "$SAM2/pretrained_models/sam_vit_h_4b8939.pth" \
    --lama_config "$SAM2/lama/configs/prediction/default.yaml" \
    --lama_ckpt "$SAM2/pretrained_models/big-lama" \
    --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
    --vi_ckpt "$SAM2/pretrained_models/sttn.pth" \
    --mask_idx 2 --fps "$FPS" --device cuda \
    --save_mask_frames 0 --save_mask_video 1 --save_vis_mask_video 1 --save_vis_box_video 1
done
