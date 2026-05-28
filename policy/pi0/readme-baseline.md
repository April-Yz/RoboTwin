# pi0 Baseline Notes

This document is a compact map of the baseline `policy/pi0` implementation: model structure, training path, and the exact input/output format used inside RoboTwin.

For actual runs, treat `pi0_base_aloha_robotwin_lora` as the default baseline train config and [`finetune_pi0_baseline.sh`](/home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_baseline.sh) / [`eval_pi0_baseline.sh`](/home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_baseline.sh) as the standard helper scripts.

## 1. Code layout

- `scripts/train.py`
  - JAX/Flax-NNX training entrypoint.
  - Builds config, data loader, model state, optimizer, checkpointing, and the train loop.
- `src/openpi/training/config.py`
  - Defines `TrainConfig`, data configs, and named configs such as `pi0_base_aloha_robotwin_lora`.
- `src/openpi/training/data_loader.py`
  - Builds a LeRobot dataset, applies transforms, and returns `(Observation, Actions)`.
- `src/openpi/models/pi0.py`
  - Core pi0 model.
- `src/openpi/models/model.py`
  - Shared data structures such as `Observation`, action tensors, and image preprocessing.
- `src/openpi/policies/policy_config.py`
  - Loads a trained checkpoint and reconstructs an inference policy.
- `deploy_policy.py` and `pi_model.py`
  - RoboTwin integration layer.

## 2. Baseline architecture

The baseline pi0 model is a flow-matching policy with three main pieces:

1. Vision encoder
   - `SigLIP` in `src/openpi/models/pi0.py`
   - Encodes `base_0_rgb`, `left_wrist_0_rgb`, and `right_wrist_0_rgb`

2. Language + multimodal trunk
   - `PaliGemma/Gemma` stack in `src/openpi/models/pi0.py`
   - Prefix tokens are image tokens plus tokenized language prompt

3. Action expert / diffusion-style action head
   - State token + noisy action tokens + time embedding form the suffix
   - Model predicts a velocity field `v_t`
   - Training uses flow matching
   - Inference integrates from Gaussian noise to action chunks

The baseline model does not directly regress a single action. It predicts an action chunk:

- `action_horizon = 50`
- `action_dim = 32`

For ALOHA-style execution, only the first 14 dimensions are used by the output transform.

## 3. Baseline inputs

After transforms, the model consumes:

- `image["base_0_rgb"]`: head camera, shape `[B, 224, 224, 3]`
- `image["left_wrist_0_rgb"]`: left wrist camera, shape `[B, 224, 224, 3]`
- `image["right_wrist_0_rgb"]`: right wrist camera, shape `[B, 224, 224, 3]`
- `image_mask[...]`: per-camera validity mask
- `state`: robot state, shape `[B, action_dim]`
- `tokenized_prompt`: prompt tokens, shape `[B, max_token_len]`
- `tokenized_prompt_mask`

For ALOHA / RoboTwin data, the raw state/action are 14-D and are padded to the model action dimension.

## 4. Baseline outputs

Training target:

- `actions`: shape `[B, action_horizon, action_dim]`

Inference output:

- `sample_actions(...) -> [B, action_horizon, action_dim]`
- ALOHA output transform then slices back to the first 14 dims

Inside RoboTwin, `deploy_policy.py` pulls the returned chunk and executes only the requested prefix of that chunk.

## 5. Training flow

`scripts/train.py` does:

1. Load a named `TrainConfig`
2. Build the LeRobot dataset and transforms
3. Initialize the pi0 model and optionally load base weights
4. Run `train_step`
5. Save checkpoints with Orbax

The core baseline loss is in `src/openpi/models/pi0.py`:

- sample noise `n`
- sample time `t`
- construct noisy action `x_t = t * n + (1 - t) * a`
- target velocity `u_t = n - a`
- predict `v_t`
- minimize `mean((v_t - u_t)^2)`

## 6. Inference flow

At inference time:

1. `policy_config.create_trained_policy(...)` loads checkpoint params and norm stats
2. Input transforms repack images/state/prompt
3. `sample_actions(...)` starts from Gaussian noise
4. The model iteratively denoises for `num_steps` Euler updates
5. Output transforms unnormalize and map back to ALOHA action space

## 7. Baseline runbook

### 7.1 Environment and prerequisites

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv sync
```

Before training on a real dataset, make sure normalization stats exist under:

```text
policy/pi0/assets/pi0_base_aloha_robotwin_lora/<repo_id>/
```

The training loader will fail if stats are missing. The stock stats script reads the repo id from the named config, so if you train with a different `DATA_REPO_ID` you also need a matching asset directory for that repo.

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv run scripts/compute_norm_stats.py --config-name=pi0_base_aloha_robotwin_lora
```

### 7.2 Training

Train a baseline config:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv run scripts/train.py pi0_base_aloha_robotwin_lora --exp-name=my_baseline --batch-size=32 --overwrite
```

Use the existing helper:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
bash finetune.sh pi0_base_aloha_robotwin_lora my_baseline 0
```

Current default for the helper scripts is global `batch_size=32`. If you need a different value, override it with `BATCH_SIZE=<n>`.

Use the dedicated baseline helper with a concrete example:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
DATA_REPO_ID=aloha_beat_block_hammer_builder \
EXP_NAME=pi0_baseline_b32 \
BATCH_SIZE=32 \
SAVE_INTERVAL=1000 \
KEEP_PERIOD=5000 \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_baseline.sh 0
```

### 7.3 RoboTwin evaluation

Run RoboTwin eval directly:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
bash eval.sh beat_block_hammer demo_clean pi0_base_aloha_robotwin_lora my_baseline 0 0
```

Evaluate the latest baseline checkpoint with the helper:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
MODEL_NAME=pi0_baseline_b32 \
TASK_NAME=beat_block_hammer \
TASK_CONFIG=demo_clean \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_baseline.sh latest 0 0
```

Evaluate a specific saved step:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
MODEL_NAME=pi0_baseline_b32 \
TASK_NAME=beat_block_hammer \
TASK_CONFIG=demo_clean \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_baseline.sh 1000 0 0
```

The helper resolves checkpoints from:

```text
policy/pi0/checkpoints/<TRAIN_CONFIG_NAME>/<MODEL_NAME>/<STEP>/
```

The dedicated baseline helper also defaults to global `batch_size=32` when `BATCH_SIZE` is not set.

### 7.4 Quick tests and sanity checks

Check that the baseline config loads and that distillation is disabled:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv run python - <<'PY'
from openpi.training import config as cfg
train_cfg = cfg.get_config("pi0_base_aloha_robotwin_lora")
print(train_cfg.name, train_cfg.use_privileged_distill, train_cfg.data.repo_id)
PY
```

Run lightweight unit tests for the core pi0 path:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv run pytest src/openpi/models/pi0_test.py src/openpi/models/model_test.py -q
```

If RoboTwin eval fails immediately, check these first:

- checkpoint exists under `policy/pi0/checkpoints/pi0_base_aloha_robotwin_lora/<exp_name>/`
- the selected step contains `params/` and `assets/<repo_id>/`
- `MODEL_NAME`, `TRAIN_CONFIG_NAME`, `TASK_NAME`, and `TASK_CONFIG` all match the training run

## 8. What to read first

If you want to understand the baseline quickly, read in this order:

1. `src/openpi/models/pi0.py`
2. `src/openpi/models/model.py`
3. `src/openpi/training/config.py`
4. `scripts/train.py`
5. `src/openpi/policies/policy_config.py`
6. `deploy_policy.py`
