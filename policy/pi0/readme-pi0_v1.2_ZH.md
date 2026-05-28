# pi0 v1.2 Full-FT 特权蒸馏说明

这份文档记录 RoboTwin 上的 `pi0 v1.2` 版本。

- 配置名：`pi0_v1_2_aloha_robotwin_full_distill`
- 训练脚本：[`finetune_pi0_v1_2.sh`](/home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_v1_2.sh)
- 评测脚本：[`eval_pi0_v1_2.sh`](/home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_v1_2.sh)

## 和 v1.1 的区别

`v1.2` 保留了 `v1.1` 的 teacher-student 特权蒸馏逻辑：

- student 只看当前图像
- teacher 看当前图像加 future image 聚合 token
- 训练损失还是 `bc_loss + distill_loss`
- 推理阶段仍然只走 student 分支

真正变化的是可训练参数集合：

- `v1.1` 是 LoRA 微调
  - `paligemma_variant="gemma_2b_lora"`
  - `action_expert_variant="gemma_300m_lora"`
  - `freeze_filter` 会冻结主干，只训练 LoRA 和少量任务头
- `v1.2` 是 full fine-tune
  - `paligemma_variant="gemma_2b"`
  - `action_expert_variant="gemma_300m"`
  - `freeze_filter=nnx.Nothing`，也就是所有参数都参与训练

所以 `v1.2` 不是 `pi0_fast`，也不是 LoRA 版 `pi0`，而是标准 `pi0` 的 full-ft 蒸馏版。

## 当前默认值

`pi0_v1_2_aloha_robotwin_full_distill` 当前默认：

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

这里把 `ema_decay` 默认关掉，是为了减少 full-ft 的训练态内存占用。full-ft 已经包含完整参数、Adam 的 `mu/nu` 状态，如果再保留 EMA，很容易在初始化阶段直接爆显存。

这里把默认启动路径设成了 `batch_size=4 + fsdp_devices=4`。我在这台机器上验证时，1 卡和 2 卡都会在 `init_train_state` 阶段 OOM。

## Norm Stats

norm stats 需要放在：

```text
policy/pi0/assets/pi0_v1_2_aloha_robotwin_full_distill/<repo_id>/
```

对于当前 `beat_block_hammer` 数据集，对应目录是：

```text
policy/pi0/assets/pi0_v1_2_aloha_robotwin_full_distill/aloha_beat_block_hammer_builder/
```

如果你换了数据集，需要重新计算：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv run scripts/compute_norm_stats.py --config-name=pi0_v1_2_aloha_robotwin_full_distill --data.repo-id=<repo_id>
```

## 训练命令

直接用 helper：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0

DATA_REPO_ID=aloha_beat_block_hammer_builder \
EXP_NAME=pi0_opsd_v1_2_beat_block_hammer \
SAVE_INTERVAL=2500 \
KEEP_PERIOD=5000 \
EMA_DECAY=None \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_v1_2.sh '0,1,2,3'
```

直接调用 `train.py`：

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

如果你确认单卡还能继续加：

```bash
BATCH_SIZE=8 bash /home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_v1_2.sh '0,1,2,3'
```

如果你只暴露 1 张或 2 张 GPU，当前这套 full-ft 配置大概率会在 `init_train_state` 直接 OOM。

## 测试命令

评测最新 checkpoint：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0

MODEL_NAME=pi0_opsd_v1_2_beat_block_hammer \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_v1_2.sh latest 0 1
```

评测固定 step：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0

MODEL_NAME=pi0_opsd_v1_2_beat_block_hammer \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_v1_2.sh 2500 0 1
```

## 变更记录

`v1.2` 相比 `v1.1` 的改动：

1. 把 LoRA 微调改成了 full fine-tune。
2. 保留了原来的 future-image privileged self-distillation 逻辑。
3. 新增独立的 `v1.2` 训练和评测脚本，不覆盖 `v1.1`。
4. 默认关闭 EMA，降低 full-ft 训练态显存占用。
5. 推荐启动方式改成 4 卡 FSDP，而不是单卡直接起跑。
