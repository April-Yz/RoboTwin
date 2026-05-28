#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

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

GPU_ID="${1:-${GPU_ID:-0}}"
CONFIG_NAME="${CONFIG_NAME:-pi0_v1_1_aloha_robotwin_lora_distill}"
EXP_NAME="${EXP_NAME:-pi0_v1_1_run}"
DATA_REPO_ID="${DATA_REPO_ID:-}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
FSDP_DEVICES="${FSDP_DEVICES:-}"
DISTILL_LOSS_TYPE="${DISTILL_LOSS_TYPE:-}"
DISTILL_WEIGHT="${DISTILL_WEIGHT:-}"
BC_WEIGHT="${BC_WEIGHT:-}"
FUTURE_HORIZON="${FUTURE_HORIZON:-}"
OVERWRITE="${OVERWRITE:-True}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"
for cuda_bin in /usr/local/cuda-12.8/bin /usr/local/cuda/bin; do
  if [[ -x "${cuda_bin}/ptxas" ]]; then
    export PATH="${cuda_bin}:${PATH}"
    break
  fi
done
TRITON_WORKAROUND_FLAG="--xla_gpu_enable_triton_gemm=false"
if [[ -z "${XLA_FLAGS:-}" ]]; then
  export XLA_FLAGS="${TRITON_WORKAROUND_FLAG}"
elif [[ "${XLA_FLAGS}" != *"${TRITON_WORKAROUND_FLAG}"* ]]; then
  export XLA_FLAGS="${XLA_FLAGS} ${TRITON_WORKAROUND_FLAG}"
fi

if command -v uv >/dev/null 2>&1; then
  RUNNER_LABEL="uv"
  cmd=(uv run scripts/train.py "${CONFIG_NAME}" "--exp-name=${EXP_NAME}")
else
  PYTHON_BIN="$(resolve_python)"
  export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
  RUNNER_LABEL="${PYTHON_BIN}"
  cmd=("${PYTHON_BIN}" scripts/train.py "${CONFIG_NAME}" "--exp-name=${EXP_NAME}")
fi

if [[ "${OVERWRITE}" == "True" ]]; then
  cmd+=("--overwrite")
fi
if [[ -n "${DATA_REPO_ID}" ]]; then
  cmd+=("--data.repo-id=${DATA_REPO_ID}")
fi
if [[ -n "${NUM_TRAIN_STEPS}" ]]; then
  cmd+=("--num-train-steps=${NUM_TRAIN_STEPS}")
fi
if [[ -n "${SAVE_INTERVAL}" ]]; then
  cmd+=("--save-interval=${SAVE_INTERVAL}")
fi
if [[ -n "${KEEP_PERIOD}" ]]; then
  cmd+=("--keep-period=${KEEP_PERIOD}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  cmd+=("--batch-size=${BATCH_SIZE}")
fi
if [[ -n "${FSDP_DEVICES}" ]]; then
  cmd+=("--fsdp-devices=${FSDP_DEVICES}")
fi
if [[ -n "${DISTILL_LOSS_TYPE}" ]]; then
  cmd+=("--distill-loss-type=${DISTILL_LOSS_TYPE}")
fi
if [[ -n "${DISTILL_WEIGHT}" ]]; then
  cmd+=("--distill-weight=${DISTILL_WEIGHT}")
fi
if [[ -n "${BC_WEIGHT}" ]]; then
  cmd+=("--bc-weight=${BC_WEIGHT}")
fi
if [[ -n "${FUTURE_HORIZON}" ]]; then
  cmd+=("--future-horizon=${FUTURE_HORIZON}")
fi

printf 'Launching pi0 v1.1 training\n'
printf '  GPU_ID=%s\n' "${GPU_ID}"
printf '  CONFIG_NAME=%s\n' "${CONFIG_NAME}"
printf '  EXP_NAME=%s\n' "${EXP_NAME}"
printf '  BATCH_SIZE=%s\n' "${BATCH_SIZE}"
printf '  RUNNER=%s\n' "${RUNNER_LABEL}"
if [[ -n "${DATA_REPO_ID}" ]]; then
  printf '  DATA_REPO_ID=%s\n' "${DATA_REPO_ID}"
fi

"${cmd[@]}"
