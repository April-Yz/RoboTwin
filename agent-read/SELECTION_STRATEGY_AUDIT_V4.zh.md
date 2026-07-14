# Selection Strategy Audit V4

## 目的与边界

V4 是只读审计工具，用于把 OursV2、Orientation、Fused 和 Top-score 放到可追溯的同一张图中。它不导入或调用 planner，不重新运行 AnyGrasp，不修改旧算法、summary、JSON、NPZ、PNG、MP4 或执行结果。

代码：

```text
code_painting/render_selection_strategy_compare_v4.py
code_painting/analyze_selection_strategy_agreement_v4.py
```

本轮输出：

```text
code_painting/selection_strategy_compare_v4/
  audit_report.json
  audit_report.zh.md
  strategy_agreement_stats.json
  strategy_agreement_stats.zh.md
  strategy_agreement_stats.en.md
  <TASK>/
    id<ID>_keyframe_<FRAME>_overlay.png
    id<ID>_keyframe_<FRAME>_metadata.json
```

生成目录仍由 `code_painting/*` 忽略；`.gitignore` 只反忽略两个 V4 脚本。

## 四种记录的真实含义

设经过旧链路已有几何/物体/手臂过滤后保留下来的 AnyGrasp 集合为 \(\mathcal C\)，候选 \(i\) 的原始 AnyGrasp score 为 \(s_i\)，与人手目标完整旋转矩阵的 SO(3) 距离为：

\[
\theta_i=\cos^{-1}\!\left(\operatorname{clip}\left(
\frac{\operatorname{tr}(R_h^\top R_i)-1}{2},-1,1
\right)\right).
\]

旧 preview 的 Orientation/Fused 都先执行独立硬过滤：

\[
\mathcal C_{90}=\{i\in\mathcal C\mid\theta_i\leq90^\circ\}.
\]

Orientation 按完整旋转距离选择最接近人手朝向的候选：

\[
i_{\mathrm{ori}}=\arg\min_{i\in\mathcal C_{90}}\theta_i.
\]

旧代码的 orientation score 为：

\[
o_i=\operatorname{clip}\left(1-\frac{\theta_i}{180^\circ},0,1\right).
\]

六任务 wrapper 的 Fused 使用原始、未归一化的 AnyGrasp score，实际权重为：

\[
F_i=0.25s_i+0.75o_i,\qquad
i_{\mathrm{fused}}=\arg\max_{i\in\mathcal C_{90}}F_i.
\]

V4 不重新排序 Orientation/Fused，而是忠实读取已经保存的 robot-frame preview rank-1 记录。这样审计的是历史产物，而不是新的实现。

Top-score 也不再把旧 `rank_previews/...rank_1.png` 当作真实选择。唯一权威来源是：

```text
plan_summary.json
  -> selected_candidates_by_executed_arm
```

因此 V4 记录的是 planner 当时实际消费的 candidate index、score 和 resolved frame。

OursV2 不是第四种 AnyGrasp 排序。它从 HaMeR/D435 人手关键帧直接生成 synthetic hand-retarget target，`candidate_idx` 和 `candidate_score` 均为 `null`。历史 synthetic score 为 1 也不能解释成 AnyGrasp rank。

## 坐标轴重映射

AnyGrasp raw 与 canonical robot/replay frame 的关系为：

\[
\begin{aligned}
x_{robot}&=-z_{raw},\\
y_{robot}&= y_{raw},\\
z_{robot}&= x_{raw}.
\end{aligned}
\]

右乘矩阵为：

\[
R_{canonical}=R_{raw}
\begin{bmatrix}
0&0&1\\
0&1&0\\
-1&0&0
\end{bmatrix}.
\]

canonical local Y 是夹爪开合轴，local Z 是 approach/前进轴。图中坐标轴颜色固定为 X 红、Y 绿、Z 蓝。

旧 Top-score 没有执行这个 remap，却沿 raw local Z 应用了 \(-0.05\,\mathrm m\) target offset。V4 同时保留：

1. raw AnyGrasp pose；
2. canonical Selection Pose；
3. 旧 Top-score legacy actual target；
4. 按正确 remap 得到的 audit-only canonical target。

V4 不把第 4 项冒充历史执行结果。

## 双面板与多 Foundation 帧

Panel A `Selection Pose` 展示策略选中或人手重定向后的位置和朝向，不加入 \(-5\) cm offset、OursV2 retreat、pregrasp、world-to-base 或 TCP-to-link6 补偿。

Panel B `Planner Target` 展示历史链路准备送给 IK 的 target，并在 metadata 中逐项记录：

- camera-to-world；
- raw-to-canonical remap；
- local offset/retreat；
- 12 cm pregrasp，仅在第一事件显示；
- world-to-base；
- TCP-to-link6 translation 和 rotation delta；
- OursV2 task-specific world adjustment。

当前 0515 bundle 的 `gripper_bias=0.12`，因此脚本记录的 TCP-to-link6 local-X translation 是：

\[
0.12-\texttt{gripper\_bias}=0.
\]

如果某个策略把 requested frame 移到最近的非空 resolved frame，V4 不会把该 pose 投影到 requested frame 的画面。每个不同 resolved frame 使用自己的：

- `color_<resolved_frame>.png` Foundation replay 图；
- head camera world pose；
- 相机内参；
- `requested_frame`、`resolved_frame` 和 `delta_frame` 标题。

这些画面按列横向拼接；两行分别是 Selection 和 Planner。

## 颜色和线型

- OursV2：青色实线和圆点；
- Orientation：粗品红实线和方框；
- Fused：黄色虚线和菱形，因此与 Orientation 完全重合时仍能同时看到；
- Top-score canonical：黑色实线和三角形；
- Top-score raw：橙色虚线和叉号；
- Top-score legacy：蓝色虚线和叉号；
- pregrasp 路径：同策略颜色虚线；
- `L` / `R`：执行手臂。

## 2026-07-14 全量结果

- 任务：6；每任务 25 个 episode；共 150/150 episode 审计成功。
- 关键帧对比图：461。
- arm-strategy records：2192。
- 审计异常：0。
- 生成物：461 PNG、463 JSON、3 Markdown，约 169 MB。
- 旧输入 summary 的组合 SHA-256 在全量运行前后相同：`345226256cadb99935a0af49e7a95fdc7f72889d21bcda354819e9def0002bd1`。

真实 Top-score 与旧 rank preview：

- actual arm-frame pairs：600；
- 旧 rank-1 图片与实际候选一致：78/600；
- 实际候选出现在导出 top-N：204/600；
- 实际候选不在导出 top-N：396/600。

`handover_bottle/foundation_input_1/frame 39/right` 的旧 rank-1 是 candidate 13，V4 从实际 plan summary 读取 candidate 0。

扁平输出示例：

```text
code_painting/selection_strategy_compare_v4/pnp_tray/id2_keyframe_000052_overlay.png
```

Top-score raw-to-canonical 的 600 条记录全部旋转 90°。旧 legacy target 和 canonical rebuild 的平均/最小/最大位置差都为：

\[
0.0707106781\,\mathrm m=\sqrt{0.05^2+0.05^2}.
\]

这是同一个 \(-5\) cm offset 分别沿两个互相垂直的 local-Z 方向应用造成的几何差，不是新的 TCP 拟合误差。两者旋转差同样是 90°。

Top-score requested/resolved frame delta 分布：

```text
-13:2, -2:2, -1:10, 0:557, +1:11, +2:4, +3:1,
+4:2, +5:3, +6:3, +7:1, +8:2, +9:1, +18:1
```

最大绝对偏移为 18 帧。

## Fused 一致率与位置差统计

统计单位是一个 arm-event，左右手分别计一次。只有 `resolved_frame` 和 `candidate_idx` 同时相同才记为同 candidate。位置使用 canonical Selection Pose 的 world xyz。

Fused 与 Orientation：

- 配对 496；同 candidate 465/496，93.75%；
- 左手 229/245，93.47%；右手 236/251，94.02%；
- xyz 欧氏距离：平均 2.979 mm，中位数 0，p95 28.050 mm，最大 107.584 mm；
- 31 个非零样本的平均距离为 47.669 mm。

Fused 与 Top canonical：

- 配对 496；同 candidate 44/496，8.87%；
- 左手 26/245，10.61%；右手 18/251，7.17%；
- xyz 欧氏距离：平均 42.718 mm，中位数 23.626 mm，p95 133.854 mm；
- 最大 868.614 mm 来自 `stack_cups/id17/right/frame36`。该旧 Top target 的 world Z 约为 0.00075 m，是被忠实保留的历史异常值，因此不能只看 mean；JSON 同时保留 median、p95 和最大差样本。

六任务的真实 Fused 公式仍是：

\[
F=0.25s_{AG}+0.75o.
\]

在本批 496 个 Fused 候选上，平均加权 raw-score contribution 为 0.050557，平均 orientation contribution 为 0.559351；orientation 在最终 Fused score 中平均占 91.75%，而且 496/496 个样本都是 orientation contribution 更大。结合 93.75% 与 Orientation 相同、只有 8.87% 与 Top 相同，当前 Fused 明显偏向 rotation，而且由于 raw AnyGrasp score 未归一化，实际偏向比名义 75% 更强。

共有 208 条 record gap，全部来自 requested frame 上旧 Orientation/Fused preview 候选为空：各 104 条。任务分布为：

```text
handover_bottle 2
pick_diverse_bottles 14
place_bread_basket 140
pnp_bread 10
pnp_tray 32
stack_cups 10
```

例如 `place_bread_basket/foundation_input_0/requested 64/left` 中，OursV2 使用 frame 64，Top-score 使用 resolved frame 63；frame 64 上左臂 Orientation/Fused 候选为空。V4 因此生成 frame 64 和 frame 63 两个 Foundation 列，并在报告记录两个 gap。

## 已知不完整 OursV2 episode

- `handover_bottle/foundation_input_6`：`plan_summary_human_replay.json` 和 `head_cam_plan.mp4` 存在，最终 `plan_summary.json` 缺失；旧执行报 `IndexError: list index out of range`。
- `pnp_tray/foundation_input_35`：human summary 和视频存在，最终 plan summary 缺失；旧执行报 `KeyError: 'left_dark_red_cup'`。

这两个 episode 都不在本轮 Top-score selected-25 输入根目录中，因此 150-episode 全量批次不会为它们生成 overlay。V4 在总报告中单独读取 human summary、最终 summary、视频和 stderr 的存在状态并保留错误尾部。如果以后使用包含它们的 Top-score 输入根目录，V4 会把 `execution_complete=false` 和 warning 写入 metadata；不会伪称旧执行成功。

## 限制

- Fused 的 Planner Target 是基于已有 preview pose 的 hypothetical reconstruction，因为历史 planner 执行组不是 Fused。
- Orientation/Fused 在 requested frame 候选为空时保持缺失；V4 不擅自加入最近帧搜索来改变旧策略。
- V4 只审计选择与 pose 变换，不证明 IK、碰撞或物理抓取成功。
- 图片拥挤时，应以 metadata 矩阵、candidate index 和 frame provenance 为准。

复现命令见 `agent-read/COMMANDS/selection_strategy_audit_v4.zh.md`。
