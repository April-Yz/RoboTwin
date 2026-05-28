# Task: 为 OpenVLA-OFT Privileged Distillation 增加动作链路诊断与 Eval 调试日志

## 1. 背景

当前 offline privileged self-distillation 版本已经可以训练，train loss 现象如下：

- `loss_bc / loss_total` 前期有明显下降
- 中后期震荡较大
- `loss_distill` 数值很小，说明蒸馏项较弱

但当前 eval 现象更关键：

- 成功率几乎为 0%
- episode 常表现为 `Fail! 400 / 400`
- 机械臂几乎没有明显位置变化

这说明当前问题可能不只是“train loss 没降”，而更可能是：

1. 模型输出动作幅值过小，策略接近静止
2. 动作反归一化 / action scaling / action clip 存在问题
3. eval 加载的 checkpoint / adapter / action head 配置不正确
4. teacher 信号过弱，模型退化成保守均值动作
5. train/eval 输入配置不一致

因此本轮任务重点不是继续改模型结构，而是增加：
- 动作统计
- teacher/student/GT 对比
- 反归一化后动作统计
- eval 时真实执行动作日志
- future teacher 有效性诊断

---

## 2. 本轮要回答的关键问题

### Q1. 模型预测动作是不是几乎接近 0？
### Q2. 反归一化后的动作是不是被压缩得很小？
### Q3. 最终真正送到控制器的动作是不是几乎没变化？
### Q4. teacher 是否真的比 student 更接近 GT？
### Q5. future 信息是否真的影响了 teacher 输出？
### Q6. eval 加载的 checkpoint 和训练配置是否一致？

---

## 3. 本轮实现范围

### 必做
1. 训练时记录 student / teacher / GT 的动作统计
2. eval 时记录 raw action / denorm action / executed action 的统计
3. 记录 student / teacher 相对 GT 的误差
4. 记录 teacher 和 student 之间的差异
5. 低频检查 future 是否真的影响 teacher 输出
6. 记录 train/eval 的关键配置摘要
7. 记录 checkpoint / adapter 加载信息

### 本轮不做
1. 不改 backbone 主体
2. 不改 teacher 结构
3. 不引入 simulator 训练
4. 不做 on-policy
5. 不默认增加重型 debug forward 到每 step

---

## 4. 训练阶段新增日志

## 4.1 student / teacher / gt 动作整体统计
每 N step 记录：

- `train/student_action_mean`
- `train/student_action_std`
- `train/student_action_abs_mean`
- `train/student_action_abs_max`

- `train/teacher_action_mean`
- `train/teacher_action_std`
- `train/teacher_action_abs_mean`
- `train/teacher_action_abs_max`

- `train/gt_action_mean`
- `train/gt_action_std`
- `train/gt_action_abs_mean`
- `train/gt_action_abs_max`

目的：
判断 student 输出是否塌缩到接近零动作。

---

## 4.2 动作逐维统计
对于 14 维动作，记录每一维的平均绝对值：

- `train/student_action_dim_abs_mean[i]`
- `train/teacher_action_dim_abs_mean[i]`
- `train/gt_action_dim_abs_mean[i]`

目的：
检查是否某些关键维度（例如 gripper 或某个 arm joint）始终接近 0。

---

## 4.3 teacher / student 对 GT 的误差
记录：

- `train/student_action_l1_to_gt`
- `train/teacher_action_l1_to_gt`
- `train/student_action_mse_to_gt`
- `train/teacher_action_mse_to_gt`

目的：
判断 teacher 看 future 后是否真的更强。

---

## 4.4 teacher 和 student 之间的差异
记录：

- `train/student_teacher_action_l1`
- `train/student_teacher_action_mse`
- `train/student_teacher_action_cosine_sim`（如果容易实现）

目的：
判断 distill loss 很小是否只是因为 teacher/student 本来几乎一样。

---

## 4.5 curr action 与 next actions 拆分统计
记录：

- `train/curr_action_l1_loss`
- `train/next_actions_l1_loss`
- `train/curr_action_pred_abs_mean`
- `train/next_actions_pred_abs_mean`
- `train/curr_action_gt_abs_mean`
- `train/next_actions_gt_abs_mean`

目的：
判断是否只学到很弱的当前动作，或 chunk 后续动作全部塌缩。

---

## 4.6 distill 相对 bc 的占比
记录：

- `train/loss_bc`
- `train/loss_distill`
- `train/loss_total`
- `train/distill_to_bc_ratio`
- `train/weighted_distill_to_total_ratio`

目的：
判断蒸馏项是否几乎没有实际作用。

---

## 4.7 future latent 统计
记录：

- `train/future_latent_norm`
- `train/future_latent_mean`
- `train/future_latent_std`

目的：
判断 future projector 是否输出异常小或异常恒定的表示。

---

## 4.8 低频 future corruption debug
不要每 step 都做，只在 debug 模式下每 N step 做一次：

1. 用真实 future 得到 `teacher_action`
2. 将 future_images 置零或打乱，得到 `teacher_action_corrupt`
3. 记录：

- `debug/teacher_action_delta_when_corrupt_future_l1`
- `debug/teacher_action_delta_when_corrupt_future_mse`

目的：
判断 teacher 是否真的利用了 future。

---

## 5. Eval 阶段新增日志

## 5.1 raw / denorm / executed action 统计
每个 episode 或每若干 step 记录：

- `eval/raw_action_mean`
- `eval/raw_action_std`
- `eval/raw_action_abs_mean`
- `eval/raw_action_abs_max`

- `eval/denorm_action_mean`
- `eval/denorm_action_std`
- `eval/denorm_action_abs_mean`
- `eval/denorm_action_abs_max`

- `eval/executed_action_mean`
- `eval/executed_action_std`
- `eval/executed_action_abs_mean`
- `eval/executed_action_abs_max`

目的：
检查动作在哪一层被压小。

---

## 5.2 eval 动作逐维统计
记录：

- `eval/raw_action_dim_abs_mean[i]`
- `eval/denorm_action_dim_abs_mean[i]`
- `eval/executed_action_dim_abs_mean[i]`

目的：
判断是否某些维度完全不动。

---

## 5.3 episode 级行为统计
记录：

- `eval/episode_len`
- `eval/success`
- `eval/end_effector_delta_norm`
- `eval/joint_delta_norm`
- `eval/gripper_open_close_changes`

目的：
如果机械臂几乎不动，这些量会非常小。

---

## 5.4 checkpoint / adapter / config 摘要
在 eval 启动时打印并记录：

- 实际加载的 checkpoint 路径
- 是否加载 LoRA adapter
- 是否 merge 成功
- `num_images_in_input`
- `use_proprio`
- `use_film`
- action normalization type
- action dim
- action chunk size

目的：
排查 train/eval 配置不一致。

---

## 6. 配置开关

新增配置项（命名可微调）：

```yaml
log_action_diagnostics: true
action_diagnostics_log_freq: 20

enable_future_corruption_debug: false
future_corruption_debug_freq: 1000

log_eval_action_stats: true
eval_action_stats_freq: 10

说明：

log_action_diagnostics：训练时动作统计总开关

action_diagnostics_log_freq：训练日志频率

enable_future_corruption_debug：是否开启低频 future 打乱检查

future_corruption_debug_freq：debug 检查间隔

log_eval_action_stats：是否记录 eval 动作链路统计

eval_action_stats_freq：eval 中的记录频率

7. 推荐改动位置

重点修改以下文件：

vla-scripts/finetune.py

prismatic/training/distill_wrapper.py

prismatic/training/distill_losses.py

eval 相关脚本 / policy 执行接口文件

若需要可新增：

prismatic/training/action_diagnostics.py

8. 验收标准

满足以下条件视为完成：

train 阶段能看到 student / teacher / gt 动作统计

能看到逐维动作统计

能看到 teacher 是否比 student 更接近 GT

能看到 distill 项占比

eval 阶段能看到 raw -> denorm -> executed 三层动作统计

能确认机械臂“不动”到底发生在哪一层

能看到当前加载 checkpoint 与关键配置摘要

新增日志不会导致明显 OOM

9. 本轮最终目的

本轮不是追求更高成功率，而是先定位：

模型是不是输出了接近零动作

动作是不是在反归一化或执行链路中被压小

teacher 是否真的有效

future 是否真的被 teacher 使用

当前 0% success 的主因究竟是训练无效还是执行链路问题


---

# 六、我再给你一个更直接的排查优先级

你现在别同时改很多东西，按这个顺序查最省时间：

### 第一步
**在 eval 里打印前 20 step 的 executed action。**

如果几乎全是非常小的数，你马上就知道核心问题在哪了。

### 第二步
同时打印：
- raw action
- denorm action
- executed action

这样能一眼看出是哪一层把动作弄小了。

### 第三步
在 train 里加：
- `teacher_l1_to_gt`
- `student_l1_to_gt`

如果 teacher 根本不比 student 强，那蒸馏先别急着继续做复杂化。

### 第四步
做一次 future corruption debug。
如果 teacher action 几乎不变，说明 future 分支还没真正起作用。

---

# 七、最后给你一句现在的判断

基于你现在的 **loss 曲线 + 0% eval + 几乎不动**，我最怀疑的是：

> **动作输出链路存在“幅值塌缩/反归一化/执行接口”问题，其次才是 teacher 蒸馏效果不足。**

也就是说，**先查动作，不要先调大 distill_weight。**

如果你愿意，我下一条可以继续帮你写一版更具体的：
**“让 Codex 直接去改 eval 和 train 日志的文件级 checklist”**。