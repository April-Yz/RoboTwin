# CHANGELOG

## 2026-03-16

### pi0 v1.2

- Added `pi0_v1_2_aloha_robotwin_full_distill`.
- Switched the `v1.2` training config from LoRA to full fine-tuning by using `pi0.Pi0Config()` with `freeze_filter=nnx.Nothing`.
- Kept the same privileged self-distillation objective as `v1.1`.
- Disabled EMA by default in `v1.2` to reduce full-ft training-state memory.
- Added versioned helper scripts:
  - `finetune_pi0_v1_2.sh`
  - `eval_pi0_v1_2.sh`
- Added versioned docs:
  - `readme-pi0_v1.2.md`
  - `readme-pi0_v1.2_ZH.md`
- Switched the helper defaults to `GPU_ID=0,1,2,3`, `BATCH_SIZE=4`, `FSDP_DEVICES=4` for the full-ft path on this workstation.
