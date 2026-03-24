# 2026-03-25 IK / 执行回退问题排查

## 1. 背景

在如下命令下，`pregrasp` 会出现长时间 try：

- `planner_backend=urdfik`
- `urdfik_trajectory_mode=cartesian_interp_ik`
- `reach_error_pose_source=ee`
- `approach_offset_m=0.0`
- `candidate_target_local_x_offset_m=0.0`

现象是：

- `fwd_cm` 先从负值逼近 0
- 然后变成正值
- 再长期稳定在大约 `+11cm ~ +13cm`

## 2. 关键日志语义

当前代码已经有三类日志：

1. `plan-request`
   - 当前测量 pose 相对于目标 pose 的误差
   - `theory` 表示理论上沿目标前进轴应该前进还是后退

2. `plan-solution`
   - 用 `target_joints` 做 FK 后得到的理论规划终点
   - 再比较：
     - 规划终点 vs 目标
     - 规划终点 vs 当前

3. `attempt`
   - 真正执行后测到的误差

## 3. 这次重新校正后的理解

之前一度把 `plan_vs_current_fwd_cm` 读反了。

正确解释是：

- `plan_request_diagnostics(current, planned)` 内部会把 `planned` 当 target，把 `current` 当 current
- 所以 `plan_vs_current_fwd_cm > 0`
  - 表示“当前姿态在规划终点的前方”
  - 也就是“规划终点比当前姿态更靠后”

因此，在长时间 try 的后半段里：

- `plan-request` 已经显示应该后退
- `plan-solution` 也已经显示规划终点在当前姿态后方
- 但 `attempt` 仍然没有退回去

这说明问题的主矛盾更偏向执行层，而不是后期 IK 完全给错方向。

## 4. 本轮代码改动

### 4.1 执行阶段增加 scene step

新增：

- `--joint_command_scene_steps`

作用：

- 每发出一个 arm joint waypoint，不再只做 `step_scene(1)`
- 而是做更多 physics step，让 drive target 有时间跟上

### 4.2 轨迹结束后等待关节收敛

新增：

- `--joint_target_wait_steps`
- `--joint_target_wait_tol_rad`

作用：

- 在整段轨迹执行完之后，继续检查当前关节和最终目标关节之间的误差
- 只要误差还大于阈值，就继续 `step_scene(1)`，直到收敛或达到上限

### 4.3 输出最终 joint 误差

执行视频 overlay 现在会额外记录：

- 单臂：`joint_max_err=...rad`
- 双臂：`left_joint_max_err=...rad` / `right_joint_max_err=...rad`

目的是把“规划好了但没跟到”和“规划本身不对”分开看。

## 5. 当前判断

这轮改动优先验证的是：

- 如果问题主要是执行层没有充分跟上最终 joint target
- 那么只要给更多 scene step 和最终收敛等待
- `attempt` 的 `fwd_cm` 应该显著下降

如果下降不明显，下一轮应继续检查：

- `urdfik.py` 里的单 seed + 阈值放宽策略
- waypoint IK 是否在早期就进入了错误 basin

## 6. 2026-03-25 第二轮修正

进一步跑 `d_pour_blue_0` 后发现：

- 在 `approach_offset_m=0.0` 的测试里
- `pregrasp` 和 `grasp` 的目标其实完全一样

这意味着：

- 如果 `pregrasp` 已经到位
- 再对同一个 pose 做一次 `grasp` 重规划，本身就是冗余操作
- 日志也确实显示，`pregrasp` 改好后，后续重复的 `grasp` 反而会把末端再次拉走

因此新增了：

- `grasp_skipped_same_target`

行为是：

- 若 `pregrasp_pose` 和 `grasp_pose` 在位置和旋转上都等价
- 则跳过 `grasp` 阶段的再次规划/执行
- 直接复用 `pregrasp` 的结果

这条规则是为了精确覆盖你当前这条命令的语义，而不是盲目依赖更多 try。

## 7. 2026-03-25 第三轮修正

继续分析 `action` 阶段日志后，定位到一个更底层的问题：

- `action` 的 `plan-request` 已经明确要求“沿目标前进轴后退”
- 但 `plan-solution` 显示规划终点只比当前姿态后退了几毫米
- 同时离最终目标仍然差 5 到 11 厘米

这说明不是简单的“执行层没跟上”，而是：

- `cartesian_interp_ik` 把轨迹切得过细
- 每个 waypoint 的单步平移已经小于 IK 的成功阈值
- 于是求解器在单个 waypoint 上“几乎不动”也会被判成 success
- 30 个这样的小 success 串起来，最终整条轨迹仍然到不了目标

在这组命令里，这个矛盾非常直接：

- `--urdfik_cartesian_interp_steps 30`
- `action` 总位移大约只有数厘米到 9 厘米
- 单步平移大约只有 `3mm`
- 旧的 `urdfik.py` 位置成功阈值是 `5mm`

所以第三轮修改是：

1. 收紧 IK 成功阈值
   - 位置阈值：`0.005m -> 0.001m`
   - 旋转阈值：`0.05rad -> 0.02rad`

2. 收紧阈值放宽策略
   - 不再一路放宽到接近 `0.1m`
   - 只允许在小范围内放宽

3. 自动压缩过细的 cartesian waypoint
   - 如果请求的 waypoint 太多，导致单步平移/旋转低于 IK 阈值的安全倍数
   - 就自动把 `requested steps` 缩到更合理的 `effective steps`

这一轮的核心判断是：

- 对这个 bug，“继续加大插值步数”不是修复方向
- 反而可能让每个 waypoint 更容易被“原地成功”
- 所以先把 waypoint 分辨率和 IK 成功阈值的量级关系修正过来

## 8. 2026-03-25 第四轮修正

第三轮修正后，`pregrasp` 已经能重新达到阈值，但 `action` 仍出现不对称问题：

- 右手显著改善
- 左手仍稳定卡在大约 `5cm`

而这时日志显示：

- waypoint 数已经被自动压缩到很小
- 所以“单步太细”已经不是左手 `action` 的主因
- 更像是当前 seed 把 IK 固定在一个局部 basin 里

因此第四轮修正不再做全局评分，而是只在“当前单个 waypoint”上做有限候选比较：

1. 先用当前 seed 解一次
2. 再额外做一次无 seed 求解
3. 用 FK 后验比较两者对该 waypoint `ee` 目标的误差
4. 只保留更接近 waypoint 的那一支

这样做的目的，是：

- 尽量保留原有“沿轨迹连续前进”的 seeded 优点
- 但在 seed 明显把解锁死时，给 waypoint 一次跳出局部解的机会
