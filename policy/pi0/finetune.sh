train_config_name=$1
model_name=$2
gpu_use=$3


# 在 finetune.sh 里的 python 命令前加上：
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

export CUDA_VISIBLE_DEVICES=$gpu_use
echo $CUDA_VISIBLE_DEVICES
# 降低显存使用比例，避免OOM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
uv run scripts/train.py $train_config_name --exp-name=$model_name --overwrite