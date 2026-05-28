#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TRAIN_CONFIG_NAME="${1:?train_config_name is required}"
MODEL_NAME="${2:?model_name is required}"
GPU_ID="${3:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

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
  cmd=(uv run scripts/train.py "${TRAIN_CONFIG_NAME}" "--exp-name=${MODEL_NAME}" "--overwrite" "--batch-size=${BATCH_SIZE}")
else
  PYTHON_BIN="$(resolve_python)"
  export PYTHONPATH="${SCRIPT_DIR}/src:${SCRIPT_DIR}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
  RUNNER_LABEL="${PYTHON_BIN}"
  cmd=("${PYTHON_BIN}" scripts/train.py "${TRAIN_CONFIG_NAME}" "--exp-name=${MODEL_NAME}" "--overwrite" "--batch-size=${BATCH_SIZE}")
fi

printf 'Launching pi0 training\n'
printf '  GPU_ID=%s\n' "${GPU_ID}"
printf '  CONFIG_NAME=%s\n' "${TRAIN_CONFIG_NAME}"
printf '  EXP_NAME=%s\n' "${MODEL_NAME}"
printf '  BATCH_SIZE=%s\n' "${BATCH_SIZE}"
printf '  RUNNER=%s\n' "${RUNNER_LABEL}"

"${cmd[@]}"
