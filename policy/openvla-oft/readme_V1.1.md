# OpenVLA-OFT Privileged Distillation V1.1

本文件记录当前 `policy/openvla-oft` 这一版在 `beat_block_hammer` 上的实际训练和评估用法，以及这轮排查后确认的几个关键结论。

## 1. 当前推荐目录

- 训练输出目录默认放在：
  `/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/runs/beat_block_hammer_v1`
- `tmux` 日志默认放在：
  `/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/tmux_logs`

这样训练产物不会再默认散落到外部 `data/...` 路径下。

## 2. `batch_size * grad_accumulation_steps` 是什么

例如：

- `batch_size = 4`
- `grad_accumulation_steps = 4`

表示每次真正执行一次 `optimizer.step()` 前，会连续做 4 次前向和反向，每次吃 4 条样本。

所以：

- 单次前向实际显存压力主要由 `batch_size=4` 决定
- 有效 batch size 是 `4 * 4 = 16`

这也是训练 run 名里会出现 `b16` 的原因。这里的 `16` 不是单次前向 batch，而是每次参数更新累计等效看到的样本数。

结论：

- 想降单步显存，先降 `batch_size`
- 只改 `grad_accumulation_steps`，主要影响优化稳定性和吞吐，不会像直接改 `batch_size` 那样明显降低激活显存

## 3. 为什么训练会被直接 `killed`

如果日志最后是这种：

```text
[1] 2811328 killed python vla-scripts/finetune.py ...
```

而不是 Python traceback，例如 `CUDA out of memory`，那通常不是脚本逻辑异常，而是进程被系统层面直接杀掉。

这一类更像：

- 系统 RAM / cgroup 内存压力过大
- CPU 侧数据管线、checkpoint 保存、其他并发任务叠加后触发 OOM killer

不是：

- 单纯的 PyTorch GPU OOM

因为 GPU OOM 一般会留下明确的 Python traceback。

当前经验判断：

- `batch_size=4` 比 `batch_size=3` 更容易触发这种系统级 kill
- 同时跑 `train + eval` 会增加 CPU、RAM、磁盘 IO 和渲染资源竞争
- `save_freq` 改小到 `1000` 只能减少中途被杀后损失的进度，不能解决根因

更稳的建议：

- 优先用 `batch_size=3`
- 避免训练时并发跑评估
- 保持 `grad_accumulation_steps=4` 不变时，有效 batch 仍然是 `12`

## 4. 为什么 eval 看起来比 train 更吃显存

问题不在纯推理前向，而是在：

- 未 merge 的 LoRA checkpoint 评估前，需要先做一次 `merge`

这一步会同时涉及：

- base model
- LoRA adapter
- merge 过程中的权重搬运

所以显存峰值可能短时间高于你预期的普通 eval。

V1.1 已经把自动 merge 默认改成：

- `MERGE_DEVICE=cpu`

也就是：

- 评估前若发现 checkpoint 还没 merge，优先在 CPU 上完成 merge
- 避免 merge 过程把 GPU 打满，导致还没进入 eval 就 OOM

## 5. 为什么 `nvidia-smi` 看到显存占用但没有进程

像下面这种情况：

```text
GPU Memory-Usage 很高
Processes: No running processes found
```

一般不应简单理解为“当前这次 eval 留下了显存残留没清”。

更常见的是：

- 其他 namespace / 容器里的进程你当前看不到
- 驱动层上下文还没完全释放
- 不是你这个 shell 里已经退出的 Python 进程还能长期正常占着 80GB+

因此：

- GPU 2/3 上那种 `87GB~89GB` 且 `100% util`，更像别的任务
- 你自己这边真正需要盯的是你要用的目标卡，例如 GPU 1

## 6. 当前默认训练脚本

默认脚本：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/openvla-oft
bash finetune_beat_block_hammer_v1.sh 1
```

现在默认值里已经包含：

- `RUN_ROOT_DIR=/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/runs/beat_block_hammer_v1`
- `SAVE_FREQ=1000`

你也可以显式覆盖，例如：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/openvla-oft
BATCH_SIZE=3 \
GRAD_ACCUMULATION_STEPS=4 \
LEARNING_RATE=1e-4 \
DISTILL_WEIGHT=1.0 \
SAVE_FREQ=1000 \
bash finetune_beat_block_hammer_v1.sh 1
```

注意：如果你直接调用 `vla-scripts/finetune.py`，要自己把参数完整传进去；如果用 helper 脚本，优先通过环境变量覆盖。

## 7. 当前推荐训练命令

更稳的版本建议是：

```bash
unset LD_LIBRARY_PATH
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_openvla
cd /home/zaijia001/ssd/RoboTwin/policy/openvla-oft

export CUDA_VISIBLE_DEVICES=1

python vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /home/zaijia001/ssd/RoboTwin/data/beat_block_hammer/tfds \
  --dataset_name aloha_beat_block_hammer_builder \
  --run_root_dir /home/zaijia001/ssd/RoboTwin/policy/openvla-oft/runs/beat_block_hammer_v1 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --use_proprio True \
  --use_privileged_distill True \
  --future_horizon 4 \
  --future_mode image_latent \
  --distill_target action \
  --distill_loss_type mse \
  --distill_weight 1.0 \
  --bc_weight 1.0 \
  --teacher_detach True \
  --batch_size 3 \
  --grad_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100000 \
  --use_val_set True \
  --val_freq 1000 \
  --save_freq 1000 \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity yangzaijia \
  --wandb_project openvla-oft \
  --run_id_note beat_block_hammer_v1_1_bs3_ga4_lr1e4_dw1
```

如果你坚持 `batch_size=4`，建议先单独跑训练，不要同时开 eval。

## 8. 续训命令

从 `10000` step checkpoint 续训：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/openvla-oft
RESUME=True \
RESUME_STEP=10000 \
VLA_PATH=/home/zaijia001/ssd/RoboTwin/policy/openvla-oft/runs/beat_block_hammer_v1/<run_dir>--10000_chkpt \
RESUME_BASE_MODEL_PATH=openvla/openvla-7b \
BATCH_SIZE=3 \
SAVE_FREQ=1000 \
bash finetune_beat_block_hammer_v1_tmux.sh 1
```

续训能恢复：

- 模型权重
- LoRA adapter
- optimizer
- scheduler

但重新启动时传入的超参数仍以新命令为准。

## 9. 当前推荐 eval 方式

```bash
unset LD_LIBRARY_PATH
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_openvla
cd /home/zaijia001/ssd/RoboTwin/policy/openvla-oft

export SAPIEN_RT_DENOISER=none
CUDA_VISIBLE_DEVICES=1 bash eval_beat_block_hammer_v1.sh \
  /home/zaijia001/ssd/RoboTwin/policy/openvla-oft/runs/beat_block_hammer_v1/<run_dir>--5000_chkpt
```

现在这条链路会：

- 先检查 checkpoint 是否已 merge
- 如果还没 merge，默认先在 CPU 上 merge
- 然后再进入真正的 eval

## 10. 结果怎么看

评估成功跑起来后，看到这种：

```text
Fail! 400 / 400
Success rate: 0/x => 0.0%
```

说明的是：

- eval 流程本身是通的
- 模型在 400 步步长限制内没完成任务

这不是脚本崩了，而是当前 checkpoint 任务成功率还不行。

## 11. 这版新增的诊断

训练侧会额外记录：

- `student / teacher / gt` 动作统计
- `student-teacher` 差异
- `student/teacher -> gt` 误差
- `curr_action / next_actions` 拆分误差
- `future_latent_*`

评估侧会额外打印：

- `raw_action`
- `denorm_action`
- `executed_action`
- episode 级 `joint_delta_norm`
- `gripper_open_close_changes`

这些主要用于判断策略到底是：

- 预测幅度太小
- 反归一化异常
- 还是执行层被裁剪后几乎不动
