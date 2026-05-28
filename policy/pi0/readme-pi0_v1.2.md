# pi0 v1.2 Full-FT Privileged Distillation

This document records the `v1.2` pi0 training variant for RoboTwin.

- config: `pi0_v1_2_aloha_robotwin_full_distill`
- train helper: [`finetune_pi0_v1_2.sh`](/home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_v1_2.sh)
- eval helper: [`eval_pi0_v1_2.sh`](/home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_v1_2.sh)

## What Changed From v1.1

`v1.2` keeps the same privileged self-distillation objective as `v1.1`:

- student sees current images only
- teacher sees current images plus pooled future-image tokens
- training loss is `bc_loss + distill_loss`
- inference stays on the student path only

The structural change is the trainable parameter set:

- `v1.1`: LoRA fine-tuning
  - `paligemma_variant="gemma_2b_lora"`
  - `action_expert_variant="gemma_300m_lora"`
  - `freeze_filter` freezes the base LLM weights and trains LoRA adapters
- `v1.2`: full fine-tuning
  - `paligemma_variant="gemma_2b"`
  - `action_expert_variant="gemma_300m"`
  - `freeze_filter=nnx.Nothing`, so all model parameters are trainable

The full-ft config is defined in [`config.py`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/training/config.py).

## Defaults

Current defaults in `pi0_v1_2_aloha_robotwin_full_distill`:

- `ema_decay=None`
- `use_privileged_distill=True`
- `future_horizon=4`
- `distill_target=action`
- `distill_loss_type=smooth_l1`
- `distill_weight=1.0`
- `bc_weight=1.0`
- `batch_size=4`
- `fsdp_devices=4`
- `num_train_steps=30000`

`ema_decay=None` is intentional. Full fine-tuning already carries the full parameter set plus Adam optimizer state, so disabling EMA reduces the training-state memory footprint.

`batch_size=4` and `fsdp_devices=4` are the recommended startup path on this machine. Full fine-tuning OOMed during `init_train_state` on 1 GPU and 2 GPUs in verification.

## Norm Stats

Norm stats must exist under:

```text
policy/pi0/assets/pi0_v1_2_aloha_robotwin_full_distill/<repo_id>/
```

For the current `beat_block_hammer` dataset, the matching asset path is:

```text
policy/pi0/assets/pi0_v1_2_aloha_robotwin_full_distill/aloha_beat_block_hammer_builder/
```

If you switch to another dataset, recompute norm stats:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv run scripts/compute_norm_stats.py --config-name=pi0_v1_2_aloha_robotwin_full_distill --data.repo-id=<repo_id>
```

## Training

Helper script:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0

DATA_REPO_ID=aloha_beat_block_hammer_builder \
EXP_NAME=pi0_opsd_v1_2_beat_block_hammer \
SAVE_INTERVAL=2500 \
KEEP_PERIOD=5000 \
EMA_DECAY=None \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_v1_2.sh '0,1,2,3'
```

Direct command:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv run scripts/train.py \
  pi0_v1_2_aloha_robotwin_full_distill \
  --exp-name=pi0_opsd_v1_2_beat_block_hammer \
  --data.repo-id=aloha_beat_block_hammer_builder \
  --batch-size=4 \
  --fsdp-devices=4 \
  --ema-decay=None \
  --overwrite
```

To try a larger batch:

```bash
BATCH_SIZE=8 bash /home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_v1_2.sh '0,1,2,3'
```

If you only expose 1 or 2 GPUs, expect `init_train_state` to OOM for full-ft on the current hardware/software stack.

## Evaluation

Evaluate the latest checkpoint:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0

MODEL_NAME=pi0_opsd_v1_2_beat_block_hammer \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_v1_2.sh latest 0 1
```

Evaluate a fixed checkpoint:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0

MODEL_NAME=pi0_opsd_v1_2_beat_block_hammer \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_v1_2.sh 2500 0 1
```

## Change Notes

Version `v1.2` changes relative to `v1.1`:

1. Switched from LoRA fine-tuning to full fine-tuning.
2. Kept the same teacher-student privileged distillation logic.
3. Added dedicated helper scripts so `v1.1` and `v1.2` can coexist.
4. Disabled EMA by default to reduce full-ft memory.
5. Switched the recommended launch path to 4-GPU FSDP instead of single-GPU startup.
