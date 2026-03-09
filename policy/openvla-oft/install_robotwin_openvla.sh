#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "RoboTwin_openvla" ]]; then
  echo "Please activate the RoboTwin_openvla conda environment first."
  echo "  source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh"
  echo "  conda activate RoboTwin_openvla"
  exit 1
fi

cd "${SCRIPT_DIR}"

python -m pip install --upgrade pip "setuptools<82" wheel packaging ninja

# Install the local package without touching the existing torch/cu128 stack.
python -m pip install -e . --no-deps

python -m pip install \
  "accelerate>=0.25.0" \
  "draccus==0.8.0" \
  "einops" \
  "huggingface_hub" \
  "json-numpy" \
  "jsonlines" \
  "matplotlib" \
  "peft==0.11.1" \
  "protobuf" \
  "rich" \
  "sentencepiece==0.1.99" \
  "timm==0.9.10" \
  "tokenizers==0.19.1" \
  "wandb" \
  "tensorflow==2.15.0" \
  "tensorflow_datasets==4.9.3" \
  "tensorflow_graphics==2021.12.3" \
  "diffusers==0.33.1" \
  "imageio" \
  "uvicorn" \
  "fastapi"

python -m pip install \
  "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git" \
  "dlimp @ git+https://github.com/moojink/dlimp_openvla"

NVCC_BIN="$(command -v nvcc)"
export CUDA_HOME="$(cd "$(dirname "${NVCC_BIN}")/.." && pwd)"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}"
export MAX_JOBS="${MAX_JOBS:-4}"

echo "Using nvcc: ${NVCC_BIN}"
echo "Using CUDA_HOME: ${CUDA_HOME}"
echo "Using TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "Using FLASH_ATTN_CUDA_ARCHS: ${FLASH_ATTN_CUDA_ARCHS}"

# On Blackwell / sm120, the original 2.5.5 recipe is too old for the newer CUDA stack.
# Keep the version overridable, but default to a 2.8.x build and surface the failure directly.
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
python -m pip cache remove flash_attn || true
python -m pip install -v "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation

python - <<'PY'
import importlib
from diffusers import DDPMScheduler

mods = [
    "prismatic",
    "experiments.robot.openvla_utils",
    "transformers",
    "peft",
    "tensorflow",
    "tensorflow_datasets",
    "flash_attn",
]

for name in mods:
    try:
        mod = importlib.import_module(name)
        print(name, "OK", getattr(mod, "__version__", "no_version"))
    except Exception as exc:
        print(name, "FAILED", repr(exc))
        raise

print("diffusers.DDPMScheduler", "OK", DDPMScheduler.__name__)
PY
