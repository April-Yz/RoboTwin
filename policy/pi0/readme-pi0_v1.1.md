# pi0_v1.1 Privileged Distillation

This document records the `pi0_v1.1` changes added on top of the baseline pi0 training pipeline.

## 1. Goal

Replicate the OpenVLA-OFT teacher/student idea in a lighter pi0 setup:

- student sees the current observation
- teacher sees privileged future images from the same trajectory
- student is trained with the original flow-matching loss plus an action-space distillation loss

The inference-time policy stays baseline-compatible. Distillation exists only during training.

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

## 4. Recommended training commands

### 4.1 Start pi0_v1.1 from scratch

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv run scripts/train.py pi0_v1_1_aloha_robotwin_lora_distill --exp-name=pi0_v1_1_run --overwrite
```

Or with the helper:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
bash finetune.sh pi0_v1_1_aloha_robotwin_lora_distill pi0_v1_1_run 0
```

### 4.2 Override the data repo at launch time

If you want to keep the named config but swap the dataset:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv run scripts/train.py \
  pi0_v1_1_aloha_robotwin_lora_distill \
  --exp-name=pi0_v1_1_run \
  --data.repo-id=your_repo_id \
  --overwrite
```

### 4.3 Match the OpenVLA-style default settings

These are the current defaults in `pi0_v1_1_aloha_robotwin_lora_distill`:

- `use_privileged_distill=True`
- `future_horizon=4`
- `future_mode=image_latent`
- `distill_target=action`
- `distill_loss_type=smooth_l1`
- `distill_weight=1.0`
- `bc_weight=1.0`
- `teacher_detach=True`

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
