#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-${1:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

unset LD_LIBRARY_PATH
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
set +u
conda activate RoboTwin_openvla
set -u

cd /home/zaijia001/ssd/RoboTwin/policy/openvla-oft

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_ENTITY="${WANDB_ENTITY:-yangzaijia}"
export WANDB_PROJECT="${WANDB_PROJECT:-openvla-oft}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-/home/zaijia001/ssd/RoboTwin/data/beat_block_hammer/runs_openvla_v1}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/home/zaijia001/ssd/RoboTwin/data/beat_block_hammer/tfds}"
DATASET_NAME="${DATASET_NAME:-aloha_beat_block_hammer_builder}"
VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"

mkdir -p "${RUN_ROOT_DIR}"

echo "Launching OpenVLA-OFT V1 training"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  WANDB_ENTITY=${WANDB_ENTITY}"
echo "  WANDB_PROJECT=${WANDB_PROJECT}"
echo "  DATA_ROOT_DIR=${DATA_ROOT_DIR}"
echo "  DATASET_NAME=${DATASET_NAME}"
echo "  RUN_ROOT_DIR=${RUN_ROOT_DIR}"

COMMON_ARGS=(
  --vla_path "${VLA_PATH}"
  --data_root_dir "${DATA_ROOT_DIR}"
  --dataset_name "${DATASET_NAME}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --use_l1_regression True
  --use_diffusion False
  --use_film True
  --num_images_in_input 3
  --use_proprio True
  --batch_size 2
  --learning_rate 5e-4
  --num_steps_before_decay 50000
  --max_steps 100000
  --use_val_set True
  --val_freq 1000
  --save_freq 5000
  --image_aug True
  --lora_rank 32
  --use_privileged_distill True
  --future_horizon 4
  --future_mode image_latent
  --distill_target action
  --distill_loss_type mse
  --distill_weight 0.5
  --bc_weight 1.0
  --teacher_detach True
  --wandb_entity "${WANDB_ENTITY}"
  --wandb_project "${WANDB_PROJECT}"
  --run_id_note "beat_block_hammer_v1_gpu${GPU_ID}"
)

if [[ "${NPROC_PER_NODE}" == "1" ]]; then
  echo "  LAUNCH_MODE=python"
  exec python vla-scripts/finetune.py "${COMMON_ARGS[@]}"
fi

echo "  LAUNCH_MODE=torchrun"
exec torchrun --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" vla-scripts/finetune.py "${COMMON_ARGS[@]}"
