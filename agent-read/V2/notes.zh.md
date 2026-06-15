# 说明

- lift 保持 grasp x/y 和姿态，仅增加 z。
- 每段从上一段轨迹末端关节状态开始。
- third 是右侧视角；opposite_top 是对向俯视。
- 旧 pickle 会被拒绝，不能与 v2 回放混用。

## Foundation O.1.1 / O.1.2

- O.1 不再在 close 前重置物体 pose。close 后状态门控检查物体是否仍稳定且位于双指夹持区域，通过后才在当前 pose 建立 drive。
- 默认底部 `support_proxy` 避免细瓶身在 pregrasp/grasp 时被张开夹爪推倒。
- O.1.1 使用第一标注关键帧建场；O.1.2 再用第二关键帧 EE xyz 的单一 action 代替 lift/place，朝向保留 grasp 设定。
- O.1.2 使用 0515 左右独立 wrist 外参，以 raw `urdf_end_link` 为父坐标，并在 optical/render 转换后应用逐侧仿真 tuning；批采集每 ID 一个 episode、最多尝试三个 seed。

## Mode N-7

- Mode N-7 将 O.1.2 的“action 保持 grasp 朝向”移植到 Foundation Pose + 人手朝向路径：action 位置来自第二关键帧 Foundation 物体 xyz，朝向和 retreat 方向来自第一关键帧 grasp。
- dual-stage replan 可以启用已达标手冻结，防止一只手已经满足阈值后被下一次 replan 重新带离目标。
- R1/AnyGrasp 的 camera-up roll 约束以 local X 为 forward；Mode N 以 local +Z 为 forward，不能直接复用。

## Mode M-0611

- Human Replay 默认改为关节空间 cubic smoothstep，并借鉴 O.1 V4 对当前关节 seed 做显式小扰动搜索，按最小关节变化选择 IK 解。
- 第二关键帧 action 使用第二帧位置和第一帧 grasp quaternion，dual replan 冻结已 reached 手。
- `pick_diverse_bottles` ID 1、2 完整成功；ID 0 只在 action 超过 4cm 阈值。
- 仍待统一 Piper IK target 与 EE report 坐标变换，并在统一后实现 local +Z 前进轴的严格 roll 约束。

## O.1.2.1 Wrist Debug Recorder

- Viewer 可用 `--wrist_debug_record 1 --wrist_debug_tag <TAG>` 保存左右原始视频、带标签拼接视频和上下文 JSON。
- 坐标链缺失的是 `link6_T_real_tcp`；0515 提供的是 `real_tcp_T_camera`。当前 tuning 用于估计缺失机械外参，不等价于物理重标定。
