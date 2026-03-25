# 2026-03-25 某次整体分析

## 背景

本分析针对 `plan_anygrasp_keyframes_r1.py` 在以下配置附近的批量运行结果：

- `--planner_backend urdfik`
- `--urdfik_trajectory_mode cartesian_interp_ik`
- `--urdfik_cartesian_interp_steps -1`
- `--reach_error_pose_source tcp`
- `--enable_grasp_action_object_collision 1`

用户反馈：

- 轨迹形状相比旧版更合理
- 但仍存在：
  - `pregrasp / grasp / action` 经常到不了位
  - `close_gripper` 时经常没有接触到物体
  - 想确认这是否是夹爪接触检测 link 选错导致

## 总体判断

当前版本的主要问题已经不再是“执行轨迹形状明显错误”，而更像是：

1. `URDF IK / planner` 经常输出一个仍然离目标较远的终点
2. `grasp` 阶段经常未能把 TCP 带到物体附近
3. 因为 `grasp` 本身没有到位，后续 `close_gripper` 经常根本碰不到物体

换句话说：

- 轨迹“怎么走”的问题，相比之前已经改善
- 轨迹“最终走到哪里”的问题，仍主要是 planner / IK 解质量问题

## 从日志看出的结论

### 1. `action` 失败主要不是执行跟不上，而是 `plan` 本身就错

典型现象：

- `plan-request` 已显示 `theory=backward`
- 但 `plan-solution` 仍持续显示：
  - `plan_vs_target_fwd_cm` 很大且为正
  - `plan_vs_current_fwd_cm` 接近 0 或小正值

这说明：

- 从当前状态到目标，理论上应该后退
- 但 planner/IK 给出的理论终点本身仍停在目标前方
- 执行层只是把这个错误终点执行出来了

因此，这类失败不能主要归因为“执行没按轨迹走到位”。

### 2. `close_gripper` 日志表明当前更多是“没抓到”而不是“穿过去还继续关”

典型日志：

```text
[warn] grasp_not_reached_before_close ...
[gripper-close] left:reason=target_reached,cmd=0.000,contact=0 right:reason=target_reached,cmd=0.000,contact=0
```

含义：

- 进入闭夹爪前，系统已经知道 `grasp` 未到位
- 夹爪渐进闭合过程中，没有检测到与目标物体发生接触
- 最终是“关到命令目标值 0.0”而不是“接触停住”

因此，这里更像：

- 夹爪没有碰到物体
- 而不是已经碰到物体但接触检测漏掉

### 3. 当前 `execute_plans(...)` 的改动已经使“轨迹形状问题”退到次要位置

之前 `cartesian_interp_ik` 只是在规划阶段使用中间 waypoint，执行时并没有逐段消费 `joint_waypoints`。

当前版本已经改成：

- 规划阶段生成 `tcp_waypoints_world`
- 每个 waypoint 解 IK，形成 `joint_waypoints`
- 执行阶段真正逐段消费 `plan["position"]`

因此：

- 现在如果视频里轨迹形状更合理，这是符合预期的
- 剩下更值得怀疑的是 waypoint 本身或 IK 解，而不是执行器完全忽略中间轨迹

## 当前夹爪接触检测到底看哪些 link

### 代码路径

接触检测相关代码：

- `plan_anygrasp_keyframes_r1.py`
  - `_get_gripper_link_entities(...)`
  - `_contact_involves_entities(...)`
  - `close_grippers_progressively_with_collision_stop(...)`

机器人 gripper joint 的定义来源：

- `envs/robot/robot.py`
  - `init_joints()`
  - `get_gripper_joints(...)`

配置来源：

- `robot_config_R1.json`

### 当前实际检测对象

当前 `_get_gripper_link_entities(...)` 的逻辑是：

- 取 `renderer.robot.left_gripper` / `renderer.robot.right_gripper`
- 对每个 gripper joint，取 `joint_info[0].child_link`

所以它监控的不是：

- `left_gripper_link`
- `right_gripper_link`

而是 gripper finger joints 的 `child_link`。

根据 `robot_config_R1.json`，当前 gripper joints 配置为：

- 左手：
  - `left_gripper_finger_joint1`
  - `left_gripper_finger_joint2`
- 右手：
  - `right_gripper_finger_joint1`
  - `right_gripper_finger_joint2`

因此，当前接触检测监控的 link 实际上是：

- `left_gripper_finger_joint1.child_link`
- `left_gripper_finger_joint2.child_link`
- `right_gripper_finger_joint1.child_link`
- `right_gripper_finger_joint2.child_link`

也就是“两根手指”的 link，而不是整个 gripper 基座 link。

### 这意味着什么

这套检测逻辑有两个特点：

1. 如果物体真正夹在两指之间，这种检测思路是合理的
2. 如果物体主要接触的是：
   - palm / gripper base
   - `left_gripper_link` / `right_gripper_link`
   - 其他非 finger child links
   那么当前检测不会把这种接触计为 “gripper-object contact”

因此，用户“怀疑接触检测 link 也有问题”是合理的，但从当前日志看：

- 更大概率仍然是 `grasp` 没有到位，导致根本没碰到物体
- 不是接触检测逻辑本身必然错

更保守的表述是：

- 当前接触检测只覆盖 finger child links
- 这可能漏掉 palm/base 接触
- 但目前日志中更强的证据仍指向“抓取位姿未到位”

## 当前阶段性结论

### 可以确认的事

- `execute_plans(...)` 逐段执行 `joint_waypoints` 后，轨迹形状问题已缓解
- `close_gripper` 渐进闭合逻辑本身没有明显自相矛盾
- 当前很多 case 的失败主因是：
  - planner/IK 终点解错误
  - `grasp` 未到位
  - 因而 `close_gripper` 无接触

### 仍需继续确认的点

- `cartesian_interp_ik` 中间 waypoint 本身是否合理
- IK 是否在某些尝试里掉进错误 basin
- finger-only 接触检测是否遗漏了对 palm/base 接触的判定

## 后续 debug 建议

不改代码时，优先看三类信号：

1. `plan-request`
   - 当前状态到目标，理论上应该前进还是后退
2. `plan-solution`
   - planner 理论终点是否仍明显停在目标前方
3. `gripper-close`
   - 是 `contact_stall` 还是 `target_reached, contact=0`

如果出现：

- `grasp_not_reached_before_close`
- `gripper-close ... contact=0`

那么优先判断为：

- 抓取位姿没到，而不是闭合接触检测漏判。
