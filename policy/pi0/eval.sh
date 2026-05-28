#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

resolve_python() {
  if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    echo "${SCRIPT_DIR}/.venv/bin/python"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "python or python3 is required but was not found in PATH." >&2
  exit 1
}

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G
for cuda_bin in /usr/local/cuda-12.8/bin /usr/local/cuda/bin; do
  if [[ -x "${cuda_bin}/ptxas" ]]; then
    export PATH="${cuda_bin}:${PATH}"
    break
  fi
done

policy_name=pi0
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
checkpoint_id=${7:-30000}
pi0_step=${8:-50}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" && -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.venv/bin/activate"
fi

PYTHON_BIN="$(resolve_python)"
export PYTHONPATH="${ROOT_DIR}/policy/pi0/src:${ROOT_DIR}/policy/pi0/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

PYTHONWARNINGS=ignore::UserWarning \
"${PYTHON_BIN}" script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --checkpoint_id ${checkpoint_id} \
    --pi0_step ${pi0_step} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} 
