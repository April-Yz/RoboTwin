# pi0 OPSD v1 Notes

This document records the current pi0 offline privileged self-distillation implementation.

In the codebase the train config and helper scripts still use the older `pi0_v1.1` naming:

- config: `pi0_v1_1_aloha_robotwin_lora_distill`
- train helper: [`finetune_pi0_v1_1.sh`](/home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_v1_1.sh)
- eval helper: [`eval_pi0_v1_1.sh`](/home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_v1_1.sh)

In this note, `opsd_v1` refers to that current implementation.

## 1. Goal

Replicate the OpenVLA-OFT teacher/student idea in a lighter pi0 setup:

- student sees the current observation
- teacher sees privileged future images from the same trajectory
- student is trained with the original flow-matching loss plus an action-space distillation loss

The inference-time policy stays baseline-compatible. Distillation exists only during training.

## 1.1 Alignment with OpenVLA-OFT

At the requirement level, `opsd_v1` matches the OpenVLA-OFT teacher/student direction:

- offline teacher/student training
- student sees only deployable current observations
- teacher sees extra future images from the same trajectory
- teacher provides action-level supervision only
- inference stays baseline-compatible

But the implementation is not structurally identical to OpenVLA-OFT.

OpenVLA-OFT:

- encodes future frames into one `future_token`
- injects that token into the teacher branch as an extra conditioning token
- exposes explicit `forward_student()` / `forward_teacher()` semantics

Current pi0 `opsd_v1`:

- encodes each future frame with the same image encoder
- pools token sequences across the future horizon
- substitutes the teacher-side image prefix tokens with pooled future image tokens instead of adding a separate token
- uses `predict_velocity(..., use_future_images=...)` instead of a separate distillation wrapper class
- uses `jax.lax.stop_gradient()` on the teacher output, which is equivalent for parameter updates but is not literally a PyTorch-style `torch.no_grad()` block

So the current pi0 path is aligned with the earlier teacher/student requirements, but it is not the same OFT conditioning architecture one-to-one.

## 2. What changed

### 2.1 Future observations added to the training batch

Files:

- `src/openpi/models/model.py`
- `src/openpi/training/data_loader.py`

The training `Observation` now optionally carries:

- `future_images`
- `future_image_masks`
- `future_valid_mask`

`data_loader.py` now has a `PrivilegedDistillationDataset` wrapper:

- for sample `t`, it collects transformed future frames `t+1 ... t+k`
- it clamps at episode boundaries
- when a future index crosses into the next episode, it repeats the last valid frame
- `future_valid_mask` marks which future steps are still real

### 2.2 Teacher-side future image latent pooling

File:

- `src/openpi/models/pi0.py`

The model now has a teacher-side path that:

- encodes each future image in the horizon with the same image encoder
- averages image tokens across the future horizon using `future_valid_mask`
- substitutes these pooled future image tokens into the prefix

This is the `future_mode="image_latent"` path.

### 2.3 Distillation loss in training

File:

- `scripts/train.py`

Baseline pi0 already predicts a velocity field `v_t`. We recover a denoised action estimate using:

```text
a_hat = x_t - t * v_t
```

`pi0_v1.1` uses:

- `loss_bc`: original flow-matching loss
- `loss_distill`: student action estimate vs teacher action estimate

Total loss:

```text
loss = bc_weight * loss_bc + distill_weight * loss_distill
```

Supported distillation losses:

- `mse`
- `smooth_l1`

### 2.4 New config fields

File:

- `src/openpi/training/config.py`

Added:

- `use_privileged_distill`
- `future_horizon`
- `future_mode`
- `distill_target`
- `distill_loss_type`
- `distill_weight`
- `bc_weight`
- `teacher_detach`

Current supported values:

- `future_mode="image_latent"`
- `distill_target="action"`
- `distill_loss_type in {"mse", "smooth_l1"}`

### 2.5 New named training config

Added config:

- `pi0_v1_1_aloha_robotwin_lora_distill`

This is the closest pi0-side equivalent of the OpenVLA teacher/student experiment.

## 3. Baseline vs v1.1

Baseline:

- only current observation
- only flow-matching loss
- no future privileged signal

pi0_v1.1:

- current observation for student
- future image latent pooling for teacher
- flow-matching + privileged distillation

## 4. Runbook

### 4.1 Environment and prerequisites

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv sync
```

As with baseline training, make sure normalization stats exist under:

```text
policy/pi0/assets/pi0_v1_1_aloha_robotwin_lora_distill/<repo_id>/
```

If you override `DATA_REPO_ID`, the training pipeline and checkpoint loader both expect a matching asset subdirectory for that repo id.

### 4.2 Start `opsd_v1` from scratch

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv run scripts/train.py pi0_v1_1_aloha_robotwin_lora_distill --exp-name=pi0_v1_1_run --batch-size=32 --overwrite
```

Or with the helper:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
bash finetune.sh pi0_v1_1_aloha_robotwin_lora_distill pi0_v1_1_run 0
```

Current default for the helper scripts is global `batch_size=32`. If you need a different value, override it with `BATCH_SIZE=<n>`.

Dedicated helper with a concrete example:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
DATA_REPO_ID=aloha_beat_block_hammer_builder \
EXP_NAME=pi0_v1_1_b32 \
BATCH_SIZE=32 \
SAVE_INTERVAL=1000 \
KEEP_PERIOD=5000 \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_v1_1.sh 0
```

### 4.3 Override the data repo at launch time

If you want to keep the named config but swap the dataset:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv run scripts/train.py \
  pi0_v1_1_aloha_robotwin_lora_distill \
  --exp-name=pi0_v1_1_run \
  --batch-size=32 \
  --data.repo-id=your_repo_id \
  --overwrite
```

### 4.4 Match the current default settings

These are the current defaults in `pi0_v1_1_aloha_robotwin_lora_distill`:

- `batch_size=32`
- `use_privileged_distill=True`
- `future_horizon=4`
- `future_mode=image_latent`
- `distill_target=action`
- `distill_loss_type=smooth_l1`
- `distill_weight=1.0`
- `bc_weight=1.0`
- `teacher_detach=True`

### 4.5 RoboTwin evaluation

`opsd_v1` uses the same inference path as baseline. Only the training graph differs.

Evaluate the latest checkpoint:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
MODEL_NAME=pi0_v1_1_b32 \
TASK_NAME=beat_block_hammer \
TASK_CONFIG=demo_clean \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_v1_1.sh latest 0 0
```

Evaluate a specific checkpoint step:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
MODEL_NAME=pi0_v1_1_b32 \
TASK_NAME=beat_block_hammer \
TASK_CONFIG=demo_clean \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_v1_1.sh 1000 0 0
```

Run evaluation without the helper if you want the full underlying command:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
bash eval.sh beat_block_hammer demo_clean pi0_v1_1_aloha_robotwin_lora_distill pi0_v1_1_run 0 0 1000 50
```

Checkpoint layout is the same as baseline:

```text
policy/pi0/checkpoints/<TRAIN_CONFIG_NAME>/<MODEL_NAME>/<STEP>/
```

### 4.6 Quick tests and sanity checks

Check that the config loads and that privileged distillation is enabled:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv run python - <<'PY'
from openpi.training import config as cfg
train_cfg = cfg.get_config("pi0_v1_1_aloha_robotwin_lora_distill")
print(train_cfg.name)
print("use_privileged_distill =", train_cfg.use_privileged_distill)
print("future_horizon =", train_cfg.future_horizon)
print("distill_loss_type =", train_cfg.distill_loss_type)
PY
```

Run the shared unit tests for the pi0 model path:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv run pytest src/openpi/models/pi0_test.py src/openpi/models/model_test.py -q
```

After training starts, verify that these metrics appear in the log:

- `bc_loss`
- `distill_loss`
- `student_teacher_l1`
- `future_valid_ratio`

If `distill_loss` is missing or always zero, you probably launched the baseline config by mistake or disabled `use_privileged_distill`.

Before RoboTwin eval, check these paths:

- `policy/pi0/checkpoints/pi0_v1_1_aloha_robotwin_lora_distill/<exp_name>/<step>/params/`
- `policy/pi0/checkpoints/pi0_v1_1_aloha_robotwin_lora_distill/<exp_name>/<step>/assets/<repo_id>/`

## 5. Logged metrics

When distillation is enabled, training now logs:

- `loss`
- `bc_loss`
- `distill_loss`
- `student_action_abs_mean`
- `teacher_action_abs_mean`
- `student_teacher_l1`
- `future_valid_ratio`
- `grad_norm`
- `param_norm`

## 6. Current limitations

- only implemented for `pi0`, not `pi0_fast`
- teacher uses future image latents only
- no separate teacher weights; this is self-distillation with privileged teacher inputs
- teacher path is training-only; checkpoint format for inference remains baseline-compatible

## 7. Practical note

The main memory cost of `pi0_v1.1` vs baseline is the second forward pass. It should still be materially lighter than the OpenVLA-OFT version because pi0 itself is smaller.

## 8. Sync to another server

Push the current branch:

```bash
cd /home/zaijia001/ssd/RoboTwin
git push origin main
```

On the other server, update the same branch:

```bash
cd /path/to/RoboTwin
git fetch origin
git switch main
git pull --ff-only origin main
```

If you also want a separate local worktree there:

```bash
cd /path/to/RoboTwin
git worktree add /path/to/RoboTwin_main origin/main
```
