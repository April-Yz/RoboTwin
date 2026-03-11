#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=pi0
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
attn_vis_enable=${7:-False}
attn_vis_every_n_steps=${8:-20}
attn_vis_max_images_per_episode=${9:-6}
attn_vis_overlay_alpha=${10:-0.45}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

source .venv/bin/activate
cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --attn_vis_enable ${attn_vis_enable} \
    --attn_vis_every_n_steps ${attn_vis_every_n_steps} \
    --attn_vis_max_images_per_episode ${attn_vis_max_images_per_episode} \
    --attn_vis_overlay_alpha ${attn_vis_overlay_alpha}
