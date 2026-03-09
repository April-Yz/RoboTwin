# OpenVLA-OFT Privileged Distillation V1

本文件说明 `policy/openvla-oft` 中第一版 `offline privileged self-distillation` 的实现范围、代码结构、配置项和使用方式。

## 1. 目标

这一版实现的是：

- `student` 只看当前可部署输入
- `teacher` 复用同一套主干，但额外看到 demo 的未来图像
- 训练仍然是 `offline`
- 不接 simulator
- 不做 on-policy rollout
- `teacher` 只提供监督，默认不参与梯度更新

第一版只实现：

- future 信息类型：`image_latent`
- distill target：`action`
- distill loss：`mse`

## 2. 总体结构

### 2.1 数据流

训练 batch 在开启蒸馏后会新增两项：

- `future_pixel_values`: `(B, K, C, H, W)`
- `future_mask`: `(B, K)`

其中 `K = future_horizon`。

未来窗口按 `[t+1, ..., t+K]` 取：

- 如果越过轨迹末尾，会 repeat 最后一帧
- `future_mask` 标出哪些 future slot 是真实存在的

相关实现：

- [`traj_transforms.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/prismatic/vla/datasets/rlds/traj_transforms.py)
- [`dataset.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/prismatic/vla/datasets/rlds/dataset.py)
- [`datasets.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/prismatic/vla/datasets/datasets.py)
- [`data_utils.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/prismatic/util/data_utils.py)

### 2.2 模型流

student 分支：

- 当前图像
- 当前 wrist 图像（如果启用）
- 当前 proprio（如果启用）
- instruction

teacher 分支：

- student 的全部输入
- `future_pixel_values`

future 图像先经过共享视觉 backbone 和 projector，得到 projected patch tokens。
然后通过一个轻量 `FutureObservationProjector` 做：

- patch 维平均
- 时间维 masked average pooling
- 两层 MLP 投到 `llm_dim`

输出一个 `(B, 1, D)` 的 `future_latent token`。

teacher 将这个 token 通过 `additional_patch_embeddings` 注入到 VLA 主干。

相关实现：

- [`modeling_prismatic.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/prismatic/extern/hf/modeling_prismatic.py)
- [`future_projector.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/prismatic/models/future_projector.py)
- [`distill_wrapper.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/prismatic/training/distill_wrapper.py)

### 2.3 损失

当前 V1 中：

- `loss_bc = L1(student_action, ground_truth_action)`
- `loss_distill = MSE(student_action, teacher_action)`
- `loss_total = bc_weight * loss_bc + distill_weight * loss_distill`

相关实现：

- [`distill_losses.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/prismatic/training/distill_losses.py)
- [`finetune.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/vla-scripts/finetune.py)

## 3. 关键文件

- [`finetune.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/vla-scripts/finetune.py)
  训练入口，负责配置、模块初始化、loss 组合、日志和 checkpoint。

- [`distill_wrapper.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/prismatic/training/distill_wrapper.py)
  封装 `forward_student()` / `forward_teacher()` 语义。

- [`future_projector.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/prismatic/models/future_projector.py)
  把未来多帧 projected patch embeddings 压成一个 teacher 条件 token。

- [`modeling_prismatic.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/prismatic/extern/hf/modeling_prismatic.py)
  新增 `additional_patch_embeddings` 注入口，尽量不改原 backbone 主体。

- [`datasets.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/prismatic/vla/datasets/datasets.py)
  把 RLDS batch 转成训练样本，并在开关打开时带上 `future_pixel_values/future_mask`。

## 4. 新增配置项

`vla-scripts/finetune.py` 中的 `FinetuneConfig` 新增了以下字段：

```yaml
use_privileged_distill: false
future_horizon: 4
future_mode: image_latent
distill_target: action
distill_loss_type: mse
distill_weight: 0.5
bc_weight: 1.0
teacher_detach: true
```

说明：

- `use_privileged_distill`: 总开关
- `future_horizon`: teacher 额外看的未来步数
- `future_mode`: V1 只支持 `image_latent`
- `distill_target`: V1 只支持 `action`
- `distill_loss_type`: V1 只支持 `mse`
- `distill_weight`: 蒸馏损失权重
- `bc_weight`: BC 损失权重
- `teacher_detach`: teacher 是否默认 `no_grad`

## 5. 使用方式

### 5.1 原训练路径

不启用蒸馏时，保持原逻辑：

```bash
unset LD_LIBRARY_PATH
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_openvla
cd /home/zaijia001/ssd/RoboTwin/policy/openvla-oft

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path /path/to/base/openvla \
  --data_root_dir /path/to/rlds \
  --dataset_name your_dataset \
  --use_l1_regression True \
  --use_diffusion False
```

### 5.2 开启 privileged distillation

```bash
unset LD_LIBRARY_PATH
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_openvla
cd /home/zaijia001/ssd/RoboTwin/policy/openvla-oft

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path /path/to/base/openvla \
  --data_root_dir /path/to/rlds \
  --dataset_name your_dataset \
  --use_l1_regression True \
  --use_diffusion False \
  --use_privileged_distill True \
  --future_horizon 4 \
  --future_mode image_latent \
  --distill_target action \
  --distill_loss_type mse \
  --distill_weight 0.5 \
  --bc_weight 1.0 \
  --teacher_detach True
```

### 5.3 `beat_block_hammer` 直接开训

`beat_block_hammer/demo_clean` 已经转换完成，当前可直接使用的数据集名是：

- `aloha_beat_block_hammer_builder`

对应 TFDS 根目录：

- `/home/zaijia001/ssd/RoboTwin/data/beat_block_hammer/tfds`

如果你要训练这次实现的 V1 privileged distillation，推荐直接使用下面这份启动脚本，而不是手动拼整条命令：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/openvla-oft
bash finetune_beat_block_hammer_v1.sh 1
```

这份脚本默认会：

- 清掉容易污染环境的 `LD_LIBRARY_PATH`
- 激活 `RoboTwin_openvla`
- 使用 `CUDA_VISIBLE_DEVICES=1`
- 采用数据集 `aloha_beat_block_hammer_builder`
- 输出到 `/home/zaijia001/ssd/RoboTwin/data/beat_block_hammer/runs_openvla_v1`
- 默认使用 `WANDB_ENTITY=yangzaijia`
- 默认使用 `WANDB_PROJECT=openvla-oft`

如果你想让训练脱离当前终端持续运行，直接用：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/openvla-oft
bash finetune_beat_block_hammer_v1_tmux.sh 1
```

这会创建一个 `tmux` session，并把日志写到：

- `/home/zaijia001/ssd/RoboTwin/data/beat_block_hammer/tmux_logs`

如果你想改卡号，可以把最后那个位置参数换成别的 GPU id，例如：

```bash
bash finetune_beat_block_hammer_v1.sh 0
```

如果你想显式看到脚本内部实际执行的训练参数，对应命令如下：

```bash
unset LD_LIBRARY_PATH
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_openvla
cd /home/zaijia001/ssd/RoboTwin/policy/openvla-oft

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /home/zaijia001/ssd/RoboTwin/data/beat_block_hammer/tfds \
  --dataset_name aloha_beat_block_hammer_builder \
  --run_root_dir /home/zaijia001/ssd/RoboTwin/data/beat_block_hammer/runs_openvla_v1 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100000 \
  --use_val_set True \
  --val_freq 1000 \
  --save_freq 5000 \
  --image_aug True \
  --lora_rank 32 \
  --use_privileged_distill True \
  --future_horizon 4 \
  --future_mode image_latent \
  --distill_target action \
  --distill_loss_type mse \
  --distill_weight 0.5 \
  --bc_weight 1.0 \
  --teacher_detach True \
  --wandb_entity YOUR_WANDB_ENTITY \
  --wandb_project YOUR_WANDB_PROJECT \
  --run_id_note beat_block_hammer_v1
```

说明：

- `num_images_in_input=3` 对应当前训练代码实际使用的 `head + left_wrist + right_wrist`
- 当前 batch transform 没有把 `low_cam_image` 送入模型，所以这里不是 4
- 当前 `finetune_beat_block_hammer_v1.sh` 会在单卡时直接调用 `python vla-scripts/finetune.py ...`
- 只有 `NPROC_PER_NODE > 1` 时才会自动切到 `torchrun`
- 如果你只想跑原始非蒸馏版本，把 `use_privileged_distill` 相关参数去掉即可

如果你现在只运行下面这条：

```bash
python vla-scripts/finetune.py \
  --data_root_dir /home/zaijia001/ssd/RoboTwin/data/beat_block_hammer/tfds \
  --dataset_name aloha_beat_block_hammer_builder
```

它会尽量使用默认值启动，但不建议这么做，因为：

- 你没有显式指定 `vla_path`
- 你没有显式指定 `wandb_entity / wandb_project`
- 你没有显式打开 `use_privileged_distill`
- 你没有显式指定 `num_images_in_input / use_proprio / use_film`

所以这条更像“能不能启动”的最简命令，不是推荐训练命令。

## 6. 最小可运行检查

项目里提供了一个 CPU smoke test：

- [`smoke_test_privileged_distill.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/vla-scripts/smoke_test_privileged_distill.py)

运行方式：

```bash
unset LD_LIBRARY_PATH
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_openvla
cd /home/zaijia001/ssd/RoboTwin/policy/openvla-oft
python vla-scripts/smoke_test_privileged_distill.py
```

这个脚本会验证：

- future trajectory chunking
- `future_mask` 行为
- `FutureObservationProjector` 的输入输出 shape
- `loss_distill`
- collator 对 `future_pixel_values/future_mask` 的拼 batch

如果你当前 shell 里继承了外部的 CUDA 或本地库路径，`torch` 可能会错误加载到别的 `libcudart/libtinfo`。当前机器上建议先执行 `unset LD_LIBRARY_PATH` 再激活 `RoboTwin_openvla`。

## 7. W&B 记录

训练记录会写到 W&B，但前提是：

- 你传了有效的 `--wandb_entity` 和 `--wandb_project`
- 当前环境已经执行过 `wandb login`
- 训练机器能访问 W&B

当前 `finetune.py` 会在训练开始时直接调用 `wandb.init(...)`，所以如果这几个条件不满足，训练可能会在初始化日志阶段失败。

除了在线 W&B 记录外，模型 checkpoint 和数据统计会保存在你指定的 `--run_root_dir` 下。

当前这份脚本默认把本地训练产物写到：

- `/home/zaijia001/ssd/RoboTwin/data/beat_block_hammer/runs_openvla_v1`

如果你想临时不上传云端，可以先设置：

```bash
export WANDB_MODE=offline
```

这样依然会保留本地 W&B 运行记录，后续可以再同步。

## 8. 当前限制

- 目前只支持 `use_l1_regression=True`
- 暂不支持和 `use_diffusion=True` 同时启用
- 暂不支持特征蒸馏、attention 蒸馏、EMA teacher
- 当前实现已经支持 `use_film=True` 时的 future teacher 分支
- 已经在当前会话里完成了多 step GPU 启动验证，训练可以进入实际 step 迭代

## 9. 训练时可观测指标

启用蒸馏后，日志中至少会出现：

- `loss_total`
- `loss_bc`
- `loss_distill`
- `curr_action_l1_loss`
- `next_actions_l1_loss`

## 10. 恢复训练和 checkpoint

启用蒸馏后，checkpoint 会额外保存：

- `future_projector--XXX_checkpoint.pt`

和原有模块一起由 [`finetune.py`](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/vla-scripts/finetune.py) 管理。

当前 `beat_block_hammer` 的专用辅助脚本有：

- [finetune_beat_block_hammer_v1.sh](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/finetune_beat_block_hammer_v1.sh)
- [finetune_beat_block_hammer_v1_tmux.sh](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/finetune_beat_block_hammer_v1_tmux.sh)
- [merge_lora_beat_block_hammer_v1.sh](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/merge_lora_beat_block_hammer_v1.sh)
- [eval_beat_block_hammer_v1.sh](/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/eval_beat_block_hammer_v1.sh)

典型流程：

```bash
# 1. 训练
bash finetune_beat_block_hammer_v1_tmux.sh 1

# 2. merge 最近一次 checkpoint
bash merge_lora_beat_block_hammer_v1.sh

# 3. eval 最近一次 checkpoint
bash eval_beat_block_hammer_v1.sh
```

说明：

- `merge_lora_beat_block_hammer_v1.sh` 不传参数时，会自动选 `runs_openvla_v1` 下最新的 `*chkpt` 目录
- `eval_beat_block_hammer_v1.sh` 不传参数时，也会默认评估最新的 `*chkpt` 目录
- 如果你要指定 checkpoint，可以把目录路径作为第一个参数传进去
