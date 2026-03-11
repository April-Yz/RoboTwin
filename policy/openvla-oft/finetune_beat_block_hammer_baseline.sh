#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-${1:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

unset LD_LIBRARY_PATH
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
set +u
conda activate RoboTwin_openvla
set -u

cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export WANDB_ENTITY="${WANDB_ENTITY:-yangzaijia}"
export WANDB_PROJECT="${WANDB_PROJECT:-openvla-oft}"
export WANDB_MODE="${WANDB_MODE:-online}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-${SCRIPT_DIR}/runs/beat_block_hammer_baseline}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-/home/zaijia001/ssd/RoboTwin/data/beat_block_hammer/tfds}"
DATASET_NAME="${DATASET_NAME:-aloha_beat_block_hammer_builder}"
VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"
RESUME="${RESUME:-False}"
RESUME_STEP="${RESUME_STEP:-}"
RESUME_BASE_MODEL_PATH="${RESUME_BASE_MODEL_PATH:-openvla/openvla-7b}"
RUN_ID_NOTE="${RUN_ID_NOTE:-beat_block_hammer_baseline_gpu${GPU_ID}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-50000}"
MAX_STEPS="${MAX_STEPS:-100000}"
VAL_FREQ="${VAL_FREQ:-1000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
WANDB_LOG_FREQ="${WANDB_LOG_FREQ:-10}"
MERGE_LORA_DURING_TRAINING="${MERGE_LORA_DURING_TRAINING:-False}"
SAVE_LATEST_CHECKPOINT_ONLY="${SAVE_LATEST_CHECKPOINT_ONLY:-False}"
LOG_ATTENTION_DIAGNOSTICS="${LOG_ATTENTION_DIAGNOSTICS:-False}"
ATTENTION_DIAGNOSTICS_LOG_FREQ="${ATTENTION_DIAGNOSTICS_LOG_FREQ:-1000}"
ATTENTION_DIAGNOSTICS_NUM_SAMPLES="${ATTENTION_DIAGNOSTICS_NUM_SAMPLES:-1}"

mkdir -p "${RUN_ROOT_DIR}"

echo "Launching OpenVLA-OFT baseline training"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  WANDB_ENTITY=${WANDB_ENTITY}"
echo "  WANDB_PROJECT=${WANDB_PROJECT}"
echo "  DATA_ROOT_DIR=${DATA_ROOT_DIR}"
echo "  DATASET_NAME=${DATASET_NAME}"
echo "  RUN_ROOT_DIR=${RUN_ROOT_DIR}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  GRAD_ACCUMULATION_STEPS=${GRAD_ACCUMULATION_STEPS}"
echo "  LEARNING_RATE=${LEARNING_RATE}"
echo "  SAVE_FREQ=${SAVE_FREQ}"
echo "  RESUME=${RESUME}"
echo "  LOG_ATTENTION_DIAGNOSTICS=${LOG_ATTENTION_DIAGNOSTICS}"
if [[ "${RESUME}" == "True" ]]; then
  echo "  RESUME_STEP=${RESUME_STEP}"
  echo "  CHECKPOINT_DIR=${VLA_PATH}"
  echo "  RESUME_BASE_MODEL_PATH=${RESUME_BASE_MODEL_PATH}"
fi

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
  --use_privileged_distill False
  --batch_size "${BATCH_SIZE}"
  --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --num_steps_before_decay "${NUM_STEPS_BEFORE_DECAY}"
  --max_steps "${MAX_STEPS}"
  --use_val_set True
  --val_freq "${VAL_FREQ}"
  --save_freq "${SAVE_FREQ}"
  --save_latest_checkpoint_only "${SAVE_LATEST_CHECKPOINT_ONLY}"
  --image_aug True
  --lora_rank 32
  --merge_lora_during_training "${MERGE_LORA_DURING_TRAINING}"
  --wandb_entity "${WANDB_ENTITY}"
  --wandb_project "${WANDB_PROJECT}"
  --run_id_note "${RUN_ID_NOTE}"
  --wandb_log_freq "${WANDB_LOG_FREQ}"
  --log_attention_diagnostics "${LOG_ATTENTION_DIAGNOSTICS}"
  --attention_diagnostics_log_freq "${ATTENTION_DIAGNOSTICS_LOG_FREQ}"
  --attention_diagnostics_num_samples "${ATTENTION_DIAGNOSTICS_NUM_SAMPLES}"
)

if [[ "${RESUME}" == "True" ]]; then
  if [[ -z "${RESUME_STEP}" ]]; then
    echo "RESUME=True requires RESUME_STEP"
    exit 1
  fi
  COMMON_ARGS+=(
    --resume True
    --resume_step "${RESUME_STEP}"
    --resume_base_model_path "${RESUME_BASE_MODEL_PATH}"
  )
fi

if [[ "${NPROC_PER_NODE}" == "1" ]]; then
  echo "  LAUNCH_MODE=python"
  exec python vla-scripts/finetune.py "${COMMON_ARGS[@]}"
fi

echo "  LAUNCH_MODE=torchrun"
exec torchrun --standalone --nnodes 1 --nproc-per-node "${NPROC_PER_NODE}" vla-scripts/finetune.py "${COMMON_ARGS[@]}"
