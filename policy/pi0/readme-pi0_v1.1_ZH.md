# pi0 OPSD v1 说明（中文）

本文档记录当前 `pi0` 离线特权自蒸馏实现，也就是代码里仍然沿用旧命名的 `pi0_v1.1`。

对应关系如下：

- 训练配置名：`pi0_v1_1_aloha_robotwin_lora_distill`
- 训练脚本：[`finetune_pi0_v1_1.sh`](/home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_v1_1.sh)
- 测试脚本：[`eval_pi0_v1_1.sh`](/home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_v1_1.sh)

本文里的 `opsd_v1` 就是当前这套实现。

## 1. 目标

目标是把 OpenVLA-OFT 的 teacher-student 思路迁移到更轻量的 `pi0` 上：

- student 只看部署时可用的当前观测
- teacher 额外看同一轨迹里的 future images
- student 同时学习原始 flow-matching 目标和 action-level 蒸馏目标

推理阶段仍然走 baseline 的路径，蒸馏只在训练时存在。

## 2. 和 OpenVLA-OFT 的关系

从需求层面，当前 `opsd_v1` 和 OpenVLA-OFT 是对齐的：

- 都是离线 teacher-student 训练
- student 只看当前可部署观测
- teacher 额外看 future image
- teacher 只提供 action 监督
- 推理阶段保持 baseline 兼容

但实现结构不是一模一样。

OpenVLA-OFT 的做法：

- 先把 future frame 编码成一个 `future_token`
- 再把这个 token 注入 teacher 分支
- 显式区分 `forward_student()` 和 `forward_teacher()`

当前 `pi0 opsd_v1` 的做法：

- 每个 future frame 先过同一个图像编码器
- 再沿 future horizon 对 image token 做池化
- 用池化后的 future image tokens 替换 teacher 侧图像 prefix token
- 用 `predict_velocity(..., use_future_images=...)` 区分 student 和 teacher 路径
- 用 `jax.lax.stop_gradient()` 截断 teacher 梯度

所以它和你之前提的 teacher-student 要求是一致的，但不是 OpenVLA-OFT 的 OFT token 结构一比一复刻。

## 3. 当前版本到底是什么

当前 `v1.1` 不是 `pi0_fast`，也不是 full fine-tune。

它是标准 `pi0` 的 LoRA 版本：

- 配置里用的是 `pi0.Pi0Config(...)`，不是 `pi0_fast.Pi0FASTConfig(...)`
- `paligemma_variant="gemma_2b_lora"`
- `action_expert_variant="gemma_300m_lora"`

对应代码见：

- [`config.py:665`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/training/config.py#L665)
- [`pi0.py:67`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/models/pi0.py#L67)
- [`config.py:708`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/training/config.py#L708)
- [`pi0_fast.py:76`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/models/pi0_fast.py#L76)

如果你说的 “fft” 指的是 `pi0_fast`，答案是否定的。

如果你说的 “fft” 指的是 full fine-tune，答案也是否定的。当前训练会冻结主干的大部分参数，只训练 LoRA 和少量任务相关参数。冻结逻辑见：

- [`pi0.py:111`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/models/pi0.py#L111)

## 4. bs=32 是怎么生效的

这里的 `batch_size=32` 是训练配置里的全局 batch size，不是文档里随便写了一个数字。

生效路径有两层：

1. 命名配置本身就是 32

- [`config.py:691`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/training/config.py#L691)

2. 我把训练脚本默认值也补成了 32

- [`finetune_pi0_v1_1.sh:31`](/home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_v1_1.sh#L31)
- [`finetune_pi0_baseline.sh:31`](/home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_baseline.sh#L31)
- [`finetune.sh:10`](/home/zaijia001/ssd/RoboTwin/policy/pi0/finetune.sh#L10)

训练时真正进入 data loader 的逻辑是：

- `local_batch_size = config.batch_size // jax.process_count()`

对应代码：

- [`data_loader.py:206`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/training/data_loader.py#L206)

你当前是单进程训练，所以：

- `jax.process_count() = 1`
- `local_batch_size = 32`

训练入口还会检查这个 batch size 是否能被设备数整除：

- [`train.py:298`](/home/zaijia001/ssd/RoboTwin/policy/pi0/scripts/train.py#L298)

因此，当前这套脚本里 `bs=32` 是真生效的，不是只改了文档没改代码。

最直接的验证方式有两个：

- 启动日志里会打印 `BATCH_SIZE=32`
- data loader 初始化时张量第一维会显示 `(32, ...)`

## 5. 为什么显存占用还是低

这不代表 `bs=32` 没生效。更可能是下面几个原因叠加：

- 当前是 LoRA 训练，不是 full fine-tune
- 模型是标准 `pi0`，不是 OpenVLA-OFT 那种更重的视觉语言训练路径
- 当前配置 `fsdp_devices=1`，没有额外的 full-shard 开销
- 训练脚本里设置了 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，JAX 不会一上来把大块显存预占满
- `pi0` 的 trainable parameter 数量本来就比 full-ft 或更重的 teacher-student VLM 小很多

相关配置在这里：

- [`config.py:696`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/training/config.py#L696)
- [`finetune_pi0_v1_1.sh:40`](/home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_v1_1.sh#L40)

所以“显存占用低”和“batch size 没到 32”不是一回事。

如果你只是想进一步把显存吃满，可以继续往上试：

- `BATCH_SIZE=48`
- `BATCH_SIZE=64`

前提是先以 32 稳定跑通，再逐步往上加。真正要明显增加显存占用，影响最大的通常不是从 16 到 32，而是：

- LoRA 改成 full fine-tune
- 模型改成更重的 `openvla-oft` 或更大的并行配置

## 6. 这版改了什么

### 6.1 训练 batch 增加了 future observation

文件：

- [`model.py`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/models/model.py)
- [`data_loader.py`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/training/data_loader.py)

训练时的 `Observation` 现在可以额外携带：

- `future_images`
- `future_image_masks`
- `future_valid_mask`

`data_loader.py` 里新增了 `PrivilegedDistillationDataset`：

- 对于时间步 `t`，会额外取 `t+1 ... t+k` 的 future frame
- 遇到 episode 边界时，会重复最后一个有效 frame
- `future_valid_mask` 用来标记哪些 future step 还是真实有效的

### 6.2 teacher 侧使用 future image latent pooling

文件：

- [`pi0.py`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/models/pi0.py)

teacher 路径会：

- 先编码 future image
- 再沿 future horizon 对 token 做池化
- 最后把池化后的 token 替换 teacher 侧图像 prefix

当前实现只支持：

- `future_mode="image_latent"`

### 6.3 训练损失加入蒸馏项

文件：

- [`train.py`](/home/zaijia001/ssd/RoboTwin/policy/pi0/scripts/train.py)

baseline `pi0` 本来就预测速度场 `v_t`。这里先恢复 denoised action：

```text
a_hat = x_t - t * v_t
```

`opsd_v1` 的总损失是：

```text
loss = bc_weight * loss_bc + distill_weight * loss_distill
```

支持的 distill loss：

- `mse`
- `smooth_l1`

## 7. 当前默认配置

`pi0_v1_1_aloha_robotwin_lora_distill` 当前默认值：

- `batch_size=32`
- `fsdp_devices=1`
- `use_privileged_distill=True`
- `future_horizon=4`
- `future_mode=image_latent`
- `distill_target=action`
- `distill_loss_type=smooth_l1`
- `distill_weight=1.0`
- `bc_weight=1.0`
- `teacher_detach=True`

## 8. 训练方法

### 8.1 环境

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv sync
```

如果你用的是当前已经修好的启动脚本，也可以不手动激活 `.venv`，脚本会优先使用 `policy/pi0/.venv/bin/python`。

### 8.2 归一化统计

训练前需要确保 norm stats 已存在：

```text
policy/pi0/assets/pi0_v1_1_aloha_robotwin_lora_distill/<repo_id>/
```

### 8.3 启动训练

直接跑训练脚本：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0

DATA_REPO_ID=aloha_beat_block_hammer_builder \
EXP_NAME=pi0_opsd_v1_beat_block_hammer \
SAVE_INTERVAL=2500 \
KEEP_PERIOD=5000 \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_v1_1.sh 1
```

如果你想显式指定 batch size，也可以这样：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0

DATA_REPO_ID=aloha_beat_block_hammer_builder \
EXP_NAME=pi0_opsd_v1_beat_block_hammer \
BATCH_SIZE=32 \
SAVE_INTERVAL=2500 \
KEEP_PERIOD=5000 \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/finetune_pi0_v1_1.sh 1
```

底层等价命令：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
uv run scripts/train.py \
  pi0_v1_1_aloha_robotwin_lora_distill \
  --exp-name=pi0_opsd_v1_beat_block_hammer \
  --batch-size=32 \
  --data.repo-id=aloha_beat_block_hammer_builder \
  --overwrite
```

## 9. RoboTwin 测试方法

测试最新 checkpoint：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
MODEL_NAME=pi0_opsd_v1_beat_block_hammer \
TASK_NAME=beat_block_hammer \
TASK_CONFIG=demo_clean \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_v1_1.sh latest 0 1
```

测试指定 step：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
MODEL_NAME=pi0_opsd_v1_beat_block_hammer \
TASK_NAME=beat_block_hammer \
TASK_CONFIG=demo_clean \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/eval_pi0_v1_1.sh 2500 0 1
```

如果你要看底层 eval 命令：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
bash eval.sh beat_block_hammer demo_clean pi0_v1_1_aloha_robotwin_lora_distill pi0_opsd_v1_beat_block_hammer 0 1 2500 50
```

## 10. 快速排查

如果你怀疑 batch size 没生效，先看：

- 启动日志里是否打印 `BATCH_SIZE=32`
- data loader 初始化时第一维是不是 `(32, ...)`

如果训练启动失败，优先检查：

- `policy/pi0/.venv` 是否存在
- `assets/pi0_v1_1_aloha_robotwin_lora_distill/<repo_id>/norm_stats.json` 是否存在
- 数据集 repo id 是否和 `DATA_REPO_ID` 一致
- checkpoint 目录是否写在 `policy/pi0/checkpoints/pi0_v1_1_aloha_robotwin_lora_distill/<exp_name>/`

## 11. 建议阅读顺序

如果你要从代码角度快速理解这版实现，建议按下面顺序看：

1. [`src/openpi/models/pi0.py`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/models/pi0.py)
2. [`scripts/train.py`](/home/zaijia001/ssd/RoboTwin/policy/pi0/scripts/train.py)
3. [`src/openpi/training/data_loader.py`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/training/data_loader.py)
4. [`src/openpi/models/model.py`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/models/model.py)
5. [`src/openpi/training/config.py`](/home/zaijia001/ssd/RoboTwin/policy/pi0/src/openpi/training/config.py)
