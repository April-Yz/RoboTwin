#!/usr/bin/env bash
set -euo pipefail

# Step 2: 用原始 planner v7 输出做 inpainting / repaint（不经过 smooth）
cd /home/zaijia001/ssd/inpainting_sam2_robot
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate inpainting-sam2-r1

GPU=0 \
DEVICE=cuda \
TASK_NAME=d_pour_blue \
ROBOT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v7 \
STAGE1_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint \
OUTPUT_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_v7 \
FORCE_REPAD=0 \
DILATE_KERNEL_SIZE=8 \
ERODE_KERNEL_SIZE=14 \
TEXT_PROMPT='robotic manipulator arm, forearm, wrist, gripper, end effector, cup, bottle.' \
BOX_THRESHOLD=0.25 \
TEXT_THRESHOLD=0.22 \
MAX_MASK_AREA_RATIO=0.60 \
MAX_SELECTED_BOXES=0 \
ARM_SPLIT_RATIO=0.5 \
EXCLUDE_BOTTOM_RATIO=0.10 \
COMPOSITE_ERODE_KERNEL_SIZE=2 \
BLEND_ALPHA_SIGMA=1.8 \
SAM_MODEL_TYPE=vit_h \
SAM_CKPT=./pretrained_models/sam_vit_h_4b8939.pth \
LAMA_CONFIG=./lama/configs/prediction/default.yaml \
LAMA_CKPT=./pretrained_models/big-lama \
TRACKER_CKPT=vitb_384_mae_ce_32x4_ep300 \
VI_CKPT=./pretrained_models/sttn.pth \
MASK_IDX=2 \
bash /home/zaijia001/ssd/inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad.sh

# Step 3: 手动可视化筛选 repaint 结果，按 y/n/m 写入 review json
cd /home/zaijia001/ssd/inpainting_sam2_robot
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate inpainting-sam2-r1

python /home/zaijia001/ssd/inpainting_sam2_robot/script/review_repaint_videos.py \
  --task-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_v7 \
  --dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' \
  --video-name target_with_original_head_cam_plan.mp4 \
  --json-path /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_v7/video_review.json

# Step 4: 把 review 中可行的原始 planner v7 repaint 数据处理成 robotwin / pi0 processed_data
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

python /home/zaijia001/ssd/RoboTwin/policy/pi0/scripts/process_repainted_planner_outputs.py \
  d_pour_blue \
  'pour water' \
  27 \
  --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_v7 \
  --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' \
  --head-video-name target_with_original_head_cam_plan.mp4 \
  --planner-root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v7 \
  --planner-dir-template 'd_pour_blue_{id}' \
  --left-wrist-video-name left_wrist_cam_plan.mp4 \
  --right-wrist-video-name right_wrist_cam_plan.mp4 \
  --pose-debug-name pose_debug.jsonl \
  --review-json /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_v7/video_review.json \
  --review-mode strict \
  --ignore-ids \
  --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-27-planner-v7
