# Smooth 处理记录（AnyGrasp keyframe planner）

## 1. 背景

你当前常用的是这条链路：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3 \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps -1 \
  --urdfik_cartesian_interp_auto_step_m 0.05 \
  --candidate_selection_mode planner \
  --left_target_object cup \
  --right_target_object bottle \
  --candidate_target_local_x_offset_m -0.03 \
  --approach_offset_m 0.08 \
  --reach_error_pose_source tcp \
  --replan_until_reached 1 \
  --replan_until_reached_max_attempts 3 \
  --save_debug_preview 1 \
  --save_debug_execution_preview 0 \
  --reach_pos_tol_m 0.03 \
  --reach_rot_tol_deg 20 \
  --enable_grasp_action_object_collision 1 \
  --pure_scene_output 1 \
  --overlay_text 0 \
  --debug_visualize_targets 0 \
  --viewer_show_camera_frustums 0 \
  --settle_steps 20 \
  --joint_target_wait_steps 70 \
  --enable_viewer 0
```

你当前遇到的问题是：

- `joint_target_wait_steps` 设大：真实执行更容易收敛到目标，精度更好
- 但导出视频里，阶段末端容易出现明显“跳变 / 瞬移”
- `joint_target_wait_steps` 设小：视频更干净，但末端收敛不够，抓取和倒水精度更差

这个矛盾本质上是：

- **执行收敛需要额外 physics step**
- **但当前主导出视频不会逐帧记录 settle loop 里的每个 step**

所以 viewer 或内部 physics 过程可能是连续的，但保存的视频只看到 settle 前一帧和 settle 后一帧，主观上就像瞬移。

---

## 2. 当前 smooth 相关处理方式总结

### 2.1 任务级路径结构

基于现有 md 文档和参数，当前执行链可以总结为：

1. 已知 `init` 位置
2. 已知抓杯子的 `grasp` 位置
3. 已知倒水结束时的 `action/end` 位置
4. 组织成：
   - `init`
   - `pregrasp`
   - `grasp`
   - `action`

其中：

- `pregrasp` 不是额外标注出来的帧
- 它是由 `grasp` 位姿反推得到
- 当前理解里，可近似记成：
  - **`init -> pregrasp(沿 grasp 朝向后退约 10cm) -> grasp -> action`**
- 更精确地说，当前常用参数是：
  - `--candidate_target_local_x_offset_m -0.03`
  - `--approach_offset_m 0.08`
- 合在一起可以理解成：
  - 先把最终抓取目标在夹爪局部 `+X` 反方向后退 `3cm`
  - 再从该抓取位姿生成一个更靠后的 `pregrasp`，约再退 `8cm`
  - 从宏观上看，相当于你说的“沿 grasp 朝向后退约 10cm 再进入抓取”这一类结构

### 2.2 当前 smooth / 轨迹生成语义

当前不是直接只做 joint-space 两点直连，而是：

1. 先定义阶段目标 EE/TCP pose
   - `init`
   - `pregrasp`
   - `grasp`
   - `action`
2. 在相邻阶段之间使用 **EE pose / TCP pose 插值**
3. 对插值后的每个 waypoint **分别求 IK**
4. 再按 joint trajectory 执行
5. 阶段完成后，如果误差仍超阈值，则进入额外等待 / 重规划

对应到当前参数：

- `--planner_backend urdfik`
- `--urdfik_trajectory_mode cartesian_interp_ik`
- `--urdfik_cartesian_interp_steps -1`
- `--urdfik_cartesian_interp_auto_step_m 0.05`

可以理解为：

- 采用 EE/TCP 空间插值
- 按长度自适应决定 waypoint 密度
- 每个 waypoint 单独求 IK
- 这是一种 **先笛卡尔平滑，再映射到 joint** 的方式

### 2.3 现有“smooth”工具与它们的角色

项目里现在至少有两类 smooth：

#### A. 执行前/执行时的 planner smooth

主要由下面参数决定：

- `--urdfik_trajectory_mode cartesian_interp_ik`
- `--urdfik_cartesian_interp_steps`
- `--urdfik_cartesian_interp_auto_step_m`
- 以及执行阶段的：
  - `--settle_steps`
  - `--joint_target_wait_steps`

它影响的是：

- 机器人怎么走到目标
- waypoint 密度
- 是否有额外收敛等待

#### B. 执行后的视频 replay / bundle smooth

已有脚本：

- `code_painting/replay_pose_debug_smooth.py`
- `code_painting/smooth_planner_outputs_from_pose_debug.py`
- `code_painting/batch_smooth_planner_outputs.sh`

它们做的是：

- 基于已保存的 `pose_debug.jsonl`
- 去掉近重复 hover 帧
- 在记录状态之间做插值
- 重新导出更平滑的视频/pose bundle

注意：

- 这类 smooth **改善的是导出观感**
- 不会提升原始那次执行的到位精度
- 也不会改变那次规划本身的成功率

---

## 3. 为什么 `joint_target_wait_steps` 会导致“越大越容易跳”

核心原因：

1. 规划轨迹 waypoint 已经执行完
2. 机器人还没有完全到位
3. 系统进入 settle / wait loop
4. 在这个 loop 里 physics 还在继续推进
5. 但主视频通常**不会把这个 loop 的每一个小步都录下来**
6. 最后只录到一个“已经收敛后的结果帧”

于是就出现：

- 真实执行：连续慢慢靠近
- 导出视频：像最后一下直接贴到目标上

因此：

- `joint_target_wait_steps` 越大
- 越允许“很多可见运动”发生在未被逐帧录制的 settle 段里
- 视频跳变感通常越明显

---

## 4. 在“不改代码”前提下，优先建议的办法

下面的方法都只涉及：

- 调参
- 改运行流程
- 改导出/后处理方式

不要求改 planner 源码。

### 方法 1：降低 `joint_target_wait_steps`，同时提高路径密度

思路：

- 不要把大量位姿变化留到 settle loop 里完成
- 尽量让更多运动发生在“已规划、已录制”的 waypoint 段中

建议尝试：

- 适度减小 `--joint_target_wait_steps`
- 同时减小 `--urdfik_cartesian_interp_auto_step_m`
  - 例如从 `0.05` 试到 `0.03` 或 `0.02`
- 或改成固定更密的 `--urdfik_cartesian_interp_steps`

优点：

- 最符合当前系统结构
- 不改代码
- 往往能直接减少“阶段末尾突然一下补齐”的观感

缺点：

- 如果 waypoint 太密、但单步跟随不足，运行时间会变长
- 仍然可能在最后留下一小段 settle 跳变
- 过密 IK waypoint 可能增加局部抖动或求解耗时

适用场景：

- 当前主要问题是视频观感，而不是根本到不了位
- 愿意用更长执行时长换更均匀的运动呈现

### 方法 2：保留较小/中等 `joint_target_wait_steps`，依赖 `replan_until_reached`

思路：

- 不让单次 stage 在末端等待太久
- 如果没到位，就重新从当前位置规划下一次短路径
- 用多次短修正代替一次长 settle

你当前已经在用：

- `--replan_until_reached 1`
- `--replan_until_reached_max_attempts 3`

建议方向：

- 把 `joint_target_wait_steps` 控制在中等值
- 让误差修正更多通过“下一次 replan”完成，而不是在一次未录制 settle 里慢慢磨

优点：

- 更容易避免单次结尾的大跳变
- 对最终精度通常比“完全不等”更稳
- 与当前 pipeline 一致

缺点：

- 如果 replan 很频繁，阶段边界可能变多
- 总时长可能上升
- 某些 case 会出现“多次短暂停-再走”的节奏感

适用场景：

- 当前数据里 reach 误差偶尔超阈值，但并不是完全失败
- 希望用更可控的短步修正替代长尾等待

### 方法 3：把 `settle_steps` 与 `joint_target_wait_steps` 区分使用

思路：

- `settle_steps` 可以理解为较粗的额外稳定时间
- `joint_target_wait_steps` 是更面向最终关节误差收敛的等待
- 两者不要一起都设得很大

建议：

- 只保留一个相对主要的等待来源
- 另一个压低
- 逐组实验：
  - 低 `settle_steps` + 中 `joint_target_wait_steps`
  - 中 `settle_steps` + 低 `joint_target_wait_steps`

优点：

- 能帮助区分究竟是“刚执行完轨迹时不稳定”，还是“最后 joint 误差收敛慢”
- 便于定位哪一段导致视频跳变最严重

缺点：

- 仍然属于经验调参
- 不能从根本上解决“settle 段未逐帧录制”的机制问题

适用场景：

- 你想先摸清“跳变到底来自哪种等待”
- 需要最小成本做参数扫描

### 方法 4：优先输出 pure scene，再单独做 post-hoc smooth replay

思路：

- 主执行仍然优先保证成功率和到位精度
- 视频观感交给执行后的 smooth 工具处理

现有工具：

- `code_painting/replay_pose_debug_smooth.py`
- `code_painting/smooth_planner_outputs_from_pose_debug.py`

推荐使用方式：

1. 原始 planner 运行时保留你认为更可靠的精度参数
2. 导出 `pose_debug.jsonl`
3. 对结果再做 replay smooth / bundle smooth

优点：

- 不牺牲原始执行精度
- 对演示视频最稳妥
- 已有现成脚本，工作量最低

缺点：

- 平滑的是“展示结果”，不是原始控制本身
- 如果后续任务依赖原始 head/wrist 视频逐帧时间语义，需要注意新旧数据源区别

适用场景：

- 主要需求是让导出视频看起来平滑
- 原始 planner 执行成功率比视频观感更重要

### 方法 5：人为增加中间语义关键点，拆成更多短段

思路：

- 当前是 `init -> pregrasp -> grasp -> action`
- 如果 `grasp -> action` 这段很长，或者姿态变化很大
- 即使中间有笛卡尔插值，也可能在局部 IK / 动力学跟随上把大量误差留到末尾
- 可以通过“流程层”把大段拆成更短段

例如概念上拆成：

- `init`
- `pregrasp`
- `grasp`
- `lift / stabilize`
- `pre-pour`
- `pour-end`

即便仍然使用同一套 planner，只要上游给出更多中间目标，也常常能降低单段末端的误差积累。

优点：

- 往往是同时兼顾精度与观感的有效办法
- 每段更短，IK 和执行都更稳
- 更接近复杂双臂任务的真实操作结构

缺点：

- 需要额外准备中间关键点/关键帧
- 不是“只改一个参数”就能完成
- 工作流会更复杂

适用场景：

- 当前 `grasp -> 倒水结束` 是长距离/大姿态变化运动
- 用户愿意增加标注或规则生成的中间 pose

### 方法 6：分离“评估精度参数”和“导出展示参数”

思路：

- 用一套偏保守的参数评估是否真实到位
- 再用另一套偏平滑的参数生成 presentation run

例如：

- 评估 run：较大 `joint_target_wait_steps`
- 展示 run：较小 `joint_target_wait_steps` + 更密 waypoint

优点：

- 最容易同时满足“实验判断”与“展示观感”两种目标
- 不必强迫同一条 run 同时达到两个最优

缺点：

- 两套结果需要管理一致性
- presentation run 可能与评估 run 在末端细节上不完全一致

适用场景：

- 你既要做方法验证，也要给人看视频
- 能接受双轨产物管理

---

## 5. 我认为当前最实用的组合建议

如果你现在**完全不改代码**，我最推荐按下面优先级试：

### 方案 A：先稳住执行，再靠后处理拿观感

适合你现在最现实的情况。

建议：

1. 保留当前较稳的到位策略：
   - `replan_until_reached=1`
   - `replan_until_reached_max_attempts=3`
2. 不要把 `joint_target_wait_steps` 拉得过大
   - 先试中等值，而不是非常大
3. 如果视频仍跳，用：
   - `replay_pose_debug_smooth.py`
   - 或 `smooth_planner_outputs_from_pose_debug.py`
   做结果后处理

原因：

- 这是对现有工程最兼容、风险最小的方案
- 能把“成功率/精度”和“视频观感”解耦

### 方案 B：减少 wait，增加路径密度

如果你希望主导出视频本身就更自然，可以优先试：

- `joint_target_wait_steps` 下降
- `urdfik_cartesian_interp_auto_step_m` 下降

目标是：

- 把主要运动从 settle 段挪回到可录制的 waypoint 段

这是我认为最值得先扫的一组参数。

### 方案 C：为长段动作补中间目标

如果 `grasp -> action` 特别长、特别复杂，那么单靠调 wait 和插值密度未必够。
此时最有效的往往不是继续加 wait，而是：

- 给动作补 1~2 个中间语义 pose
- 把长段拆短

这通常最能同时改善：

- 到位精度
- 姿态稳定性
- 视频观感

---

## 6. 建议的实验顺序（不改代码版）

建议按下面顺序小规模对比：

### 组 1：只扫 wait

固定：

- `urdfik_cartesian_interp_auto_step_m=0.05`
- 其余参数不变

对比：

- `joint_target_wait_steps = 10 / 20 / 40 / 70`

观察：

- 最终 reach 误差
- 视频末端是否明显跳变
- 哪个 stage 最明显

### 组 2：在较小 wait 下扫路径密度

固定一个较小或中等的：

- `joint_target_wait_steps`

再对比：

- `urdfik_cartesian_interp_auto_step_m = 0.05 / 0.03 / 0.02`

观察：

- 视频是否更连续
- 执行时长是否明显增加
- 是否出现局部抖动 / 反复修正

### 组 3：必要时改为后处理 smooth

如果组 1 和组 2 都说明：

- 想要精度就必须保留较多 wait
- 而较多 wait 又必然造成导出视频跳变

那么结论通常就是：

- **原始 run 负责精度**
- **smooth replay 负责观感**

这比继续硬抠单个参数更高效。

---

## 7. 你提出的“每 1cm 采样 + 前一解作种子 + jump threshold 拒绝整段”方案分析

你提出的方法可以概括为：

```python
Pose1 ---------------------------------> Pose3
每隔 1cm 采样一个 EE pose
IK(p1, seed=q0) -> IK(p2, seed=q1) -> ... -> IK(pN, seed=qN-1)
若相邻两点 joint 变化 > threshold，则整段拒绝并重规划
```

### 7.1 它和当前 `urdfik_cartesian_interp_ik` 的关系

这个想法**和当前实现不是完全不同路线**，而是：

- **有一部分已经存在**
- **有一部分还没有显式做强约束**

当前已有的部分：

1. 已经是 EE/TCP 笛卡尔插值，而不是只做 joint 两点插值
2. 已经会对插值 waypoint **逐点求 IK**
3. 已经会用**前一个 waypoint 的解作为下一个 waypoint 的 seed**

也就是说，你这个方案的核心思想里，当前系统其实已经具备：

- `Cartesian waypoint interpolation`
- `IK waypoint-by-waypoint`
- `previous solution as seed`

你提出的新东西里，真正和当前实现拉开差异的，主要是：

1. **采样更密**
   - 例如明确要求 `eef_step = 0.01m`
2. **连续性约束更强**
   - 不只是“尽量用上一个 seed”
   - 而是显式检测 `|q_i - q_{i-1}|`
3. **跳变即拒绝整段**
   - 当前更偏向 waypoint 求不过就失败
   - 你提出的是：即便每个点都能求出来，但如果 joint branch 发生突变，也判为坏轨迹

所以它本质上可以理解为：

- **当前 A2（cartesian_interp_ik）的强化版 / 更严格版**

### 7.2 这种方法的优点

#### 优点 1：最贴近你现在已有的执行链

因为当前已经是：

- 笛卡尔插值
- waypoint IK
- previous seed

所以如果以后真要实现，这不是推翻现有结构，而是：

- 在现有 A2 路径上继续加约束和更细采样

这是它最大的现实优势。

#### 优点 2：能明显减少 IK branch 跳变

如果每个 waypoint 都用上一个解作 seed，而且采样足够密：

- solver 更容易沿同一个局部连续分支走
- 不容易突然切到另一个完全不同的 joint configuration

这对减少“明明 EE 路径很平滑，但 joint 解突然翻腕/翻肘”的问题很有帮助。

#### 优点 3：可以在规划阶段更早发现坏轨迹

如果加入：

- 相邻 waypoint 关节角变化阈值
- 或最大单步速度/加速度近似阈值

那么很多“最终能到，但中间非常不自然”的轨迹，可以在执行前直接拒绝，而不是等到视频导出后才发现跳变。

#### 优点 4：对减少末端 wait 的依赖有帮助

如果轨迹本身更连续、更符合机械臂局部可跟随性：

- 执行器通常更容易一路跟上
- 留到 `joint_target_wait_steps` 里补收敛的那部分运动会变少

这正是你关心的视频跳变问题的关键。

### 7.3 这种方法的缺点

#### 缺点 1：运行成本会上升

例如：

- 20cm 路径
- 每 1cm 一个 waypoint

那就是约 20 个 IK 点；如果双臂同步、多个 stage 累加，计算量会上升明显。

相比当前较粗的 auto step：

- planning 时间更长
- batch 吞吐更低

#### 缺点 2：更容易在局部困难段“过早失败”

密采样 + jump threshold 的副作用是：

- 它会更严格
- 一些“虽然有一点关节切换，但最终仍可执行”的轨迹，也可能被直接拒绝

这会带来：

- 成功率下降
- 需要更多 replan / candidate fallback

#### 缺点 3：jump threshold 不容易统一设

不同阶段的姿态变化幅度不一样：

- `init -> pregrasp`
- `pregrasp -> grasp`
- `grasp -> action`

所允许的正常 joint 变化范围可能并不相同。

如果阈值太严：

- 合法轨迹也被拒绝

如果阈值太松：

- 又拦不住真正的 branch jump

#### 缺点 4：它仍然不是完整的轨迹优化器

即使做了密采样 + 连续 seed + jump check，它本质上仍然是：

- **逐点局部 IK**

而不是：

- 带平滑、速度、加速度、碰撞等全局约束的 trajectory optimization

所以它能显著改善连续性，但不是理论上最强的方案。

### 7.4 相对当前方法的实现难度评估

#### 实现难度：中等

原因：

- 不是从零新建 planner
- 现有 `cartesian_interp_ik` 已经很接近这个框架
- 主要新增的是：
  1. 更明确的笛卡尔采样策略（如固定 `eef_step=1cm`）
  2. joint 连续性指标
  3. 超阈值后的轨迹拒绝/重试逻辑
  4. summary/debug 输出里记录“为什么被拒绝”

换句话说：

- **比“重写成全局轨迹优化器”容易很多**
- **但比单纯调参数明显更复杂**

如果按实现难度分级，我会给：

- 低：只调 `auto_step_m / wait_steps / replan`
- **中：在当前 A2 上加入 dense sampling + jump threshold**
- 高：加入真正的轨迹优化/平滑约束求解

---

## 8. 其他可考虑的新方法（保留现有 run_plan_anygrasp_keyframes_r1_batch.sh 主流程前提下）

### 方法 A：固定步长密采样（1cm）替代当前较粗的 auto-step

本质：

- 仍然是当前 `cartesian_interp_ik`
- 但把 waypoint 采样策略改成更稳定、可控的固定步长

优点：

- 最直接
- 最符合你现在的想法
- 最容易解释和调试

缺点：

- 对长路径会明显变慢
- 如果姿态变化大但平移小，只按位置 1cm 采样仍可能不够

实现难度：**低到中**

### 方法 B：位置 + 旋转双阈值采样

本质：

- 不只按平移步长采样
- 还按旋转角度上限采样
- 例如：
  - 每 1cm 或每 5° 至少插一个 waypoint

优点：

- 比纯 `1cm` 更合理
- 对倒水这种姿态变化大的任务更适合

缺点：

- 采样规则更复杂
- 参数更多

实现难度：**中等**

### 方法 C：在 waypoint IK 之后增加 joint-space 连续性过滤

本质：

- 先照常求完整段 waypoint IK
- 再检测：
  - 相邻 `q` 差值
  - 单关节最大变化
  - 总体范数变化
- 若过大，则拒绝此段

优点：

- 和你提出的 jump-threshold 思路一致
- 对 branch jump 最直接
- 对当前结构侵入较小

缺点：

- 只能“检测并拒绝”，不能自动修复
- 如果候选和 seed 本身不好，可能频繁失败

实现难度：**中等**

### 方法 D：在 waypoint IK 中加入软平滑约束 / 代价

本质：

- 求每个 waypoint IK 时，不只追求 pose 误差最小
- 还偏好：
  - 离上一个 `q` 更近
  - 远离 joint limit
  - 更接近某个舒适姿态

优点：

- 比硬阈值拒绝更柔和
- 更可能“自动选出连续解”，而不是先跳再报错

缺点：

- 取决于底层 IK solver 暴露多少控制接口
- 若 solver 接口不够开放，实现会麻烦

实现难度：**中到高**

### 方法 E：先 waypoint IK，再做 joint trajectory smoothing / shortcut

本质：

- 先得到一串 `q1...qN`
- 再在 joint space 做后处理：
  - 平滑滤波
  - shortcut
  - velocity/acceleration clipping

优点：

- 思路清晰
- 可以和当前 pipeline 解耦
- 易于做 debug 对比

缺点：

- 平滑后会偏离原始 TCP 路径
- 若不过一遍 FK/误差回查，可能损伤抓取精度

实现难度：**中等**

### 方法 F：把长动作拆成更多语义中间 pose

本质：

- 不是优化求解器
- 而是减少每一段本身的跨度

优点：

- 对当前系统非常有效
- 经常是收益最高的方法之一
- 对双臂长程动作尤其友好

缺点：

- 需要额外中间点来源
- 工作流复杂度上升

实现难度：**低到中**（算法上低，数据/流程上中）

### 方法 G：切到真正的 trajectory optimization / motion generation

本质：

- 不再只是 waypoint IK
- 而是直接做全局约束轨迹规划

优点：

- 理论上最完整
- 最容易统一纳入平滑、限速、碰撞等要求

缺点：

- 改动最大
- 对当前 `run_plan_anygrasp_keyframes_r1_batch.sh` 这条 urdfik 路线不是小修小补
- 调试成本高

实现难度：**高**

---

## 9. 基于当前 IK 方法的难度与推荐顺序

结合当前事实：

- 当前 A2 已经是 Cartesian interpolation + waypoint IK + previous-seed
- 当前缺的是“更严格的连续性控制”和“更明确的拒绝机制”

所以我会这样排序：

### 最值得优先尝试

1. **固定步长/双阈值密采样**
2. **相邻 waypoint joint jump threshold 检测与拒绝**
3. **必要时再加更多中间语义 pose**

原因：

- 这三项都和当前结构最兼容
- 收益和实现成本比最好

### 第二梯队

4. **IK 目标里加入软连续性偏好**
5. **waypoint IK 后做 joint smoothing，再加 FK 误差回查**

原因：

- 有潜力
- 但实现比前一组更不确定

### 最后才考虑

6. **换成真正的全局 trajectory optimization**

原因：

- 成本最高
- 不适合先用来解决当前这个“视频跳变 + 连续性”的局部问题

---

## 10. 针对你这条 V7 debug 命令的额外分析

你现在这条命令的关键特征是：

- `urdfik_cartesian_interp_auto_step_m=0.01`
- `execute_interp_steps=24`
- `joint_command_scene_steps=4`
- `settle_steps=30`
- `joint_target_wait_steps=15`
- `joint_target_wait_tol_rad=0.01`
- `replan_until_reached=1`
- `replan_until_reached_max_attempts=3`
- `execute_both_arms=1`
- `debug_visualize_ik_waypoints=1`

这已经是一个“偏密采样、偏强调执行跟随”的配置，但你仍觉得视频不够平滑，而且不使用 try 时精度明显不够。

### 10.1 为什么“不使用 try”时更难到精确位置

这通常不是因为目标 pose 本身突然不对，而是因为当前系统里“规划到目标”和“真实执行后到目标”不是同一件事。

当前链路里至少有这几层误差来源：

1. **waypoint IK 是局部求解**
   - 每个 waypoint 解出来，不等于整段执行后就一定精确贴到最终 pose

2. **执行层存在跟随滞后**
   - `joint_command_scene_steps=4` 依然只是给每个 joint command 一小段 physics 时间
   - 尤其双臂、带碰撞、带物体时，drive 可能没完全跟上

3. **阶段结束判定是基于 reach tolerance，不是基于“理论规划终点”**
   - 即使规划时的 FK 终点看起来接近 target
   - 真正执行后的 `attempt` 仍可能超出 `reach_pos_tol_m / reach_rot_tol_deg`

4. **双臂同步和碰撞会加大误差积累**
   - 你现在启用了：
     - `execute_both_arms=1`
     - `grasp_action_object_collision_start_stage pregrasp`
     - `execution_object_collision_mode convex`
     - `gripper_contact_monitor_mode all_robot_links`
   - 这会让本来单臂可达的路径，在实际执行时变得更保守、更容易留残差

所以“不用 try 就不够精确”的根本原因通常是：

- **单次 plan + 单次 execute 只能把系统带到“接近目标”**
- **最后几厘米 / 几度的误差，需要靠额外收敛或下一轮从当前位置修正**

### 10.2 为什么 try / replan 会让结果更准，但观感反而更差

因为 try 机制本质上是在做：

1. 先执行一段
2. 看执行后的真实误差
3. 若未达阈值，再从“当前真实位置”重规划下一段

它提升精度的原因是：

- 第二次、第三次规划时，起点已经更接近目标
- 需要修正的只是残余误差
- 这比一次长路径硬压到完全到位更稳

但它也更容易让视频看起来不平滑，原因有两个：

#### 原因 A：阶段内部会出现“多次短修正”的节奏

也就是：

- 先走一段
- 停
- 判定没到
- 再走一段
- 再停

即使每一小段本身是平滑的，整体看也像 segmented motion。

#### 原因 B：每次 try 末尾仍可能带有少量未录制 settle

即使 `joint_target_wait_steps=15` 不算大：

- 只要末端还有可见运动发生在 settle loop
- 主导出视频仍可能把那一小段表现成“末尾跳一下”

所以 try 机制不是“错了”，而是：

- **它更偏向提高最终 reach 精度**
- **但天然不一定最适合生成最连续、最好看的单段视频**

### 10.3 你的感觉“try 机制可能有问题”，更准确地说是什么

更准确的说法不是：

- `try` 本身有 bug

而是：

- `try` 在当前系统里承担了“执行误差补偿器”的角色
- 这说明前一轮 plan+execute 还不足以一次就把真实系统稳定带到阈值内

也就是说，如果你观察到：

- 不用 try：经常差一点到不了
- 用 try：可以逐步磨到位

这通常说明主要矛盾在：

1. **执行跟随不足**
2. **waypoint 轨迹虽然理论可行，但对真实执行器来说不够“容易跟”**
3. **双臂/碰撞/物体约束让最后那一点误差更难一次消掉**

所以 try 的存在本身并不奇怪，反而说明：

- 当前系统更接近“plan-execute-correct”闭环
- 而不是“一次 open-loop 轨迹就足够精确”

### 10.4 如何兼顾“执行准确性”和“视频里不跳变”（不改代码分析版）

如果坚持不改代码，最现实的平衡思路是：

#### 方案 A：把 try 留着，但减少每次 try 里留给 settle 的可见运动

思路：

- 保留 `replan_until_reached`
- 但尽量让每次 try 的主运动更多发生在已录制 waypoint 段
- 少把运动留到 wait/settle 末尾

对应方向：

- 保持中等 `joint_target_wait_steps`
- 继续保持较密的 waypoint
- 避免把 `joint_target_wait_steps` 拉大来“硬磨精度”

这是最现实的折中方案。

#### 方案 B：把长动作拆成更多中间 pose，而不是依赖 try 在长段末尾修正

如果当前主要误差集中在：

- `grasp -> action`

那么 try 看起来像“有问题”，往往只是因为：

- 这段本身太长
- 姿态变化太大
- 剩余误差都堆到了末尾

所以最自然的改善方向其实是：

- 给动作加中间语义 pose
- 减少单段跨度

这样通常比单纯增大 `joint_target_wait_steps` 更能同时兼顾：

- 到位精度
- 视频连续性

#### 方案 C：原始执行为精度，视频用 replay/bundle smooth 补观感

如果你最终确认：

- 没有 try，精度就是不够
- 有 try，精度够但视频不够自然

那最工程化的结论通常就是：

- **原始 run 为 reach 精度服务**
- **导出展示视频由 smooth replay 服务**

这不是“逃避问题”，而是承认：

- 当前主视频的录制机制天然不擅长表达 settle / correction 段的连续细节

---

## 11. 你想新增的诊断量：点到目标前进轴的距离

你现在已经有：

- `dx/dy/dz/dist`
- `rot`
- `fwd_rot`
- `fwd_cm`

其中：

- `fwd_cm` 表示当前位置到目标位置的误差，在**目标夹爪前进轴（local +X）上的投影**
- 它回答的是：
  - 你是“沿着目标前进轴过前/过后”了多少

但它**不能回答**：

- 当前位置相对于目标前进轴，这个“轴线本身”偏离了多少
- 也就是“横向偏了多少”

你要的这个量，非常合理。可以定义成：

### 11.1 建议新增指标

给定：

- 目标位置：`p_target`
- 当前位置：`p_current`
- 目标夹爪前进轴单位向量：`a_target`（当前约定 local `+X`）

先算位置误差：

- `e = p_current - p_target`

然后分解成两部分：

1. **轴向误差（已有）**
   - `e_parallel = dot(e, a_target)`
   - 这就是当前类似 `fwd_cm` 的含义

2. **到目标前进轴线的横向距离（你要的新量）**
   - `e_perp = || e - dot(e, a_target) * a_target ||`

也可以记成：

- `lateral_to_target_forward_axis_cm`
- 或中文：
  - `到目标前进轴距离_cm`
  - `前进轴横向偏差_cm`

### 11.2 这个指标能回答什么问题

它能把“没到位”拆成两类：

1. **沿轴方向错位**
   - 太靠前 / 太靠后
   - 看 `fwd_cm`

2. **横向偏离轴线**
   - 没有沿着应该接近的那条轴线去对准
   - 看 `e_perp`

这对抓取很有价值，因为很多 case 里：

- `dist` 看起来不大
- `fwd_cm` 也不大
- 但其实夹爪在目标前进轴旁边“擦着过去”了

这时新增的 `e_perp` 会直接暴露问题。

### 11.3 它和现有指标的关系

假设位置误差向量是 `e`，那么：

- `dist = ||e||`
- `fwd_cm = dot(e, a_target)`
- `axis_lateral_cm = ||e - dot(e, a_target)a_target||`

三者关系是：

- `dist^2 = fwd^2 + lateral^2`

也就是说：

- `dist` 是总误差
- `fwd_cm` 是轴向分量
- `axis_lateral_cm` 是横向分量

这三个量一起看，会比单独 `dist` 或单独 `fwd_cm` 更有解释力。

### 11.4 对诊断最有帮助的打印方式

现在已经实现并输出：

- `dist`
- `fwd_cm`
- `lat_cm`
- `fwd_rot`

其中：

- `lat_cm`
  - 表示当前点到“目标前进轴直线”的横向距离，单位厘米
  - 越大代表越偏离应当沿着接近的那条轴线

目前这个量已经接入到：

- 单臂 `plan-request`
- 单臂 `plan-solution`
- 单臂 `attempt`
- 单臂 `attempt-supervision`
- 双臂 `plan-request`
- 双臂 `plan-solution`
- 双臂 `attempt`
- 以及写入 `attempt_history` / supervision error 结构

这样一眼就能区分：

- 是前后没到
- 还是侧向偏了
- 还是朝向没对准

### 11.5 实现难度评估

这个新增指标的实现难度我评估为：

- **低**

因为它不需要改规划逻辑，只是：

- 基于当前已经有的 `current pose`
- 和当前已经有的 `target forward axis`
- 再做一次误差向量分解

也就是说，它更像：

- 一个高价值、低侵入的诊断指标

---

## 12. 当前结论

如果你想在**保留现在 `run_plan_anygrasp_keyframes_r1_batch.sh` 主处理链**的前提下尝试新方法，我认为最自然的新方向就是：

1. 保留现在的：
   - `init -> pregrasp -> grasp -> action`
   - EE/TCP pose 插值
   - waypoint-by-waypoint IK
2. 在这个基础上加强：
   - **更密采样**
   - **更明确的 joint 连续性约束**
   - **超阈值拒绝整段并重规划**

也就是说，你提出的方案不是脱离当前体系的“另起炉灶”，而是：

- **对当前 `cartesian_interp_ik` 的最合理增强方向之一**

另外，针对你这次 V7 debug 观察，我的判断是：

- **不用 try 就不够准，说明当前系统确实需要执行后修正闭环；**
- **增大 `joint_target_wait_steps` 能提精度，但视频跳变，是因为更多可见运动被留在未逐帧录制的 settle 段；**
- **要兼顾两者，最现实的方向不是一味加 wait，而是减少单段末尾残差、减少长段跨度、并补充更能解释误差结构的诊断量（尤其是前进轴横向偏差）。**

如果只给一句建议：

- **从“固定更密采样 + jump threshold 拒绝 + 新增 axis lateral 诊断量”开始最合适；**
- **它比全局轨迹优化容易落地得多，又比单纯调 wait 参数更直接触及问题本身。**

---

## 13. 关于“保持手-物体相对关系不变，再整体施加一个坐标变换”的构造思路

你的目标可以概括成：

1. 保留现有主流程和已有能力
2. 不推翻现有 `run_plan_anygrasp_keyframes_r1_batch.sh`
3. 后续如果实现，希望通过**新增函数或新文件名**的方式接入
4. 由于人手示范和机器人操作空间异构，希望先在“目标 pose 构造层”做一个**对象相对坐标变换**，让机器人面对的是更可执行、更平滑的目标

这是一个很合理的方向。它和“直接改 IK 求解器”不同，核心不是改 solver，而是：

- **先把人手给出的目标，投影/修正到更适合机器人的一族目标上**

### 13.1 最推荐的建模视角：在物体坐标系里做手位姿修正

当前你已经有：

- 物体在世界系中的 pose：`T_world_obj`
- 人手/抓取目标在世界系中的 pose：`T_world_hand`

最自然的第一步不是直接在世界系里胡乱平移/旋转，而是先转到物体坐标系：

- `T_obj_hand = inv(T_world_obj) @ T_world_hand`

这个量表示：

- **手相对于物体的位姿关系**

如果你想“左右手和物体相对位置不变”，真正该保的就是这个结构，而不是直接保世界系坐标。

### 13.2 核心构造：对象相对 pose + 机器人可执行修正

我建议把新的目标构造分成两层：

#### 第 1 层：保留人手语义

先从示范里取：

- 左手相对杯子：`T_cup_left_hand_demo`
- 右手相对瓶子：`T_bottle_right_hand_demo`

这一步保留的是：

- 人类操作意图
- 抓取接触区域
- 倒水时相对物体的功能关系

#### 第 2 层：加一个机器人专用修正变换

然后不要直接把 demo 相对位姿拿去给机器人，而是引入一个额外修正：

- `Δ_left_robot`
- `Δ_right_robot`

得到：

- `T_cup_left_robot_target = T_cup_left_hand_demo @ Δ_left_robot`
- `T_bottle_right_robot_target = T_bottle_right_hand_demo @ Δ_right_robot`

最后再回到世界系：

- `T_world_left_target = T_world_cup @ T_cup_left_robot_target`
- `T_world_right_target = T_world_bottle @ T_bottle_right_robot_target`

这里的 `Δ_robot` 就是你要设计的“整体坐标变换”的真正合适位置。

### 13.3 为什么这个构造比直接在世界系改 target 更好

因为你真正面对的问题不是：

- 世界系绝对位置差一点

而是：

- 人手末端和机器人 gripper 的 kinematic morphology 不同
- 人手能舒服实现的 wrist pose，机器人未必能舒服实现
- 但“相对物体的功能关系”往往是你真正想保留的

所以最合理的是：

- **保物体相对关系**
- **在物体系中引入机器人专用修正**
- **而不是在世界系瞎平移一个固定偏置**

---

## 14. 这个 `Δ_robot` 应该怎么构造

我建议按从简单到复杂的顺序来设计。

### 14.1 最简单可落地版：常量刚体修正

也就是每只手各有一个固定 SE(3) 变换：

- 左手：`Δ_left_robot = [R_left, t_left]`
- 右手：`Δ_right_robot = [R_right, t_right]`

其中：

- `t`：在物体局部坐标系里的小平移
- `R`：在物体局部坐标系里的小旋转

解释：

- 人手抓杯时，手掌可能更贴近杯壁、更内扣
- 机器人夹爪需要更退后一点、更正一点、更接近某个固定抓取姿态
- 那么就让机器人永远对同一类物体应用同一套修正

优点：

- 最稳定
- 最容易先验证
- 完全可以通过“新增函数/新文件”接入，不破坏现有主链

缺点：

- 泛化能力有限
- 不同物体尺寸、不同关键帧、不同动作阶段可能不够统一

实现难度：**低**

### 14.2 更合理的版本：分阶段修正

不同阶段其实需要的修正不一定一样：

- `pregrasp`
- `grasp`
- `action`

所以可以定义：

- `Δ_left_pregrasp`
- `Δ_left_grasp`
- `Δ_left_action`
- `Δ_right_pregrasp`
- `Δ_right_grasp`
- `Δ_right_action`

这会比“一套变换打天下”更合理。

例如：

- `pregrasp` 希望离物体远一点、姿态更保守
- `grasp` 希望对准接触面
- `action` 希望优先保证可达性和倾倒轨迹稳定

优点：

- 更符合实际任务结构
- 比单常量偏置更容易兼顾成功率和流畅度

缺点：

- 参数数量增加
- 需要更多调试

实现难度：**低到中**

### 14.3 更进一步：按物体类别或尺寸自适应

如果后续你发现：

- cup / bottle / bowl
- 大杯子 / 小杯子
- 高瓶 / 矮瓶

需要不同修正，那么可以让 `Δ_robot` 依赖：

- 物体类别
- 包围盒尺寸
- 关键帧阶段

例如：

- `Δ = f(object_type, bbox_size, stage, arm)`

但这个 `f` 不一定非要是学习模型，也可以是规则函数。

优点：

- 适应性更强

缺点：

- 规则会越来越多
- 文档和维护成本上升

实现难度：**中等**

### 14.4 最强但最复杂的版本：局部可执行性优化后的投影

更高级的想法是：

- 在物体局部坐标系里，先取人手给出的 `T_obj_hand_demo`
- 然后在它附近搜索一个“小修正” `Δ`
- 使得新目标：
  - 更容易 IK 成功
  - 更连续
  - 更远离 joint limit
  - 更少发生 branch jump

可写成一个局部优化问题：

- 最小化：
  - `pose_deviation_cost(Δ)`
  - `ik_difficulty_cost`
  - `joint_jump_cost`
  - `distance_to_nominal_cost`
- 约束：
  - 修正幅度不能太大
  - 仍保持抓取功能语义

这个本质上是：

- **把“人手目标”投影到“机器人可执行目标流形”上**

优点：

- 最符合你真正的目标
- 可能同时提升成功率与平滑度

缺点：

- 复杂度最高
- 更像一个新的局部优化模块

实现难度：**高**

---

## 15. 我建议优先保哪些相对关系

你说“使得机器人左右手和物体相对位置不变”，这里要非常小心：

- **不建议把 6D 相对位姿完全硬保持不变**
- 更建议“保功能相关的部分，放松机器人难以实现的部分”

### 15.1 对抓取最值得保留的量

优先保：

1. 接近方向 / 抓取法向
2. 指向物体关键接触区的位置关系
3. 沿物体主轴的相对高度
4. 倒水任务中杯口/瓶口的相对朝向关系

### 15.2 最适合放松的量

更适合放松的是：

1. 围绕夹爪前进轴的 roll
2. 少量侧向平移
3. 少量沿前进轴的后退量

原因是：

- 这些通常对“功能是否成立”影响较小
- 但对机器人可达性/平滑性影响很大

换句话说，你不该追求：

- “人手相对物体 pose 原封不动复制给机器人”

而该追求：

- “保留任务功能语义的前提下，做最小机器人化修正”

---

## 16. 最推荐的新增模块形式（不破坏现有功能）

你要求“保留前面所有代码和功能能够正常实现，通过新增函数或者文件名的方式实现”。

按这个要求，最安全的结构是：

### 16.1 新增一个独立的 target adapter 层

例如以后实现时可以加：

- 新文件：
  - `code_painting/object_relative_target_adapter.py`
- 新函数：
  - `adapt_demo_hand_pose_in_object_frame(...)`
  - `build_robot_target_from_object_relative_pose(...)`
  - `apply_robot_morphology_delta(...)`

然后在现有 planner 真正吃 target pose 之前，额外走一层：

- 原始示范/候选 target
- -> 物体系相对 pose
- -> 机器人修正后的相对 pose
- -> 世界系机器人 target pose
- -> 现有 `plan_path(...)`

这样做的好处是：

- 原有逻辑还能完整保留
- 老参数还能继续用
- 新功能可以用新 flag 单独开关

### 16.2 不建议直接改老的 IK solver 核心逻辑

因为你真正想加的不是：

- 求解器数值细节

而是：

- **目标构造策略**

如果直接把这些语义塞进 `urdfik.py`，会导致：

- solver 和任务逻辑耦合
- 更难 debug
- 更难比较“开/不开修正”差异

所以我强烈建议：

- **在 solver 前面加 target adapter**
- **不要先动 solver 核心**

---

## 17. 一个最实用的起步版本

如果只讨论“先怎么构造最靠谱”，我建议第一版就做这个：

### 版本 S1：对象相对常量修正

对每只手、每个阶段：

1. 先算：
   - `T_obj_hand_demo`
2. 在物体系里施加一个小的常量修正：
   - 平移：
     - 沿目标前进轴后退一点
     - 沿法向/切向微调一点
   - 旋转：
     - 减少不必要的 wrist roll
     - 让夹爪更接近机器人偏好的“自然抓取姿态”
3. 再变回世界系
4. 走现有 `cartesian_interp_ik`

这是最适合你现在工程状态的版本，因为：

- 对现有功能侵入最小
- 与当前 `candidate_target_local_x_offset_m` / `approach_offset_m` 的思路一致，但更系统化
- 后续也最容易升级到“分阶段修正”或“局部优化投影”

---

## 18. 我对这个方向的最终建议

如果现在先不改代码，只分析构造方式，我的建议是：

1. **不要直接在 IK 里硬加“左右手和物体相对位置完全不变”这个死约束**
   - 太刚，会让机器人更难解
2. **改成：在物体坐标系里先表达人手相对位姿，再做机器人专用修正**
3. **优先保留功能相关几何关系，放松 roll 和小平移自由度**
4. **新增一个独立 target adapter 层，而不是一开始就改 solver**
5. **第一版最值得做的是“对象相对常量/分阶段修正”**

如果只给一句结论：

- **最合理的构造不是“把人手 pose 直接喂给 IK 并硬约束不变”，而是“先把人手 pose 转到物体系，再用一个机器人化修正 `Δ_robot` 把它投影到更可执行的目标 pose 上，然后仍走现有 planner/IK 主链”。**
