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
- Debug recorder 使用 VS Code 兼容的 H.264/yuv420p/faststart；正式 wrapper 支持四个 `WRIST_*` 环境变量进行无 viewer 参数覆盖。
- Viewer 新增 `--show_camera_frustums 1`，显式显示并校验 `left_camera`、`right_camera`、`head_camera`；修复 `--hold 1` 过去未生效的问题。
- 修复 Piper IK move/settle/gripper 循环只更新 wrist 图像但不调用 `viewer.render()` 的问题；实时 SAPIEN 与双腕预览可独立或同时运行。
- 新增 `script/diagnose_piper_wrist_camera_axes.py`，用于区分 Pika 物理 `+X` 与旧 debug `+Z` 前向约定；当前 wrist forward 与物理 `+X` 基本对齐，并可把微小开合平面误差作为 viewer yaw 参数输出。


## 2026-06-16：Foundation 抓取深度默认值

O.1/O.1.2 Foundation Piper IK 的 `foundation_grasp_standoff` 默认值现在是 `0.105m`。旧值 `0.085m` 容易让瓶子看起来进入夹爪根部；新值让 EE/gripper base 目标后退 2cm，使瓶子更靠近夹爪指尖闭合区域。调试入口：viewer `--foundation_grasp_standoff_m`，采集 wrapper `FOUNDATION_GRASP_STANDOFF_M`。


## 2026-06-16：Wrist 相机俯视与右手偏心

0515 wrist 标定与 `piper_pika_agx` adapter 的结果是 camera forward 基本沿 gripper `+X` 平视，而不是俯视夹爪；右手相机中心 `Y=-2.74cm`，左手 `Y=+2.07cm`。Viewer 已增加 pitch/lateral 调参入口，先用左右 pitch `15deg`、右手 lateral `+0.0067m` 观察 wrist 画面。

## 2026-07-10 R/S 实验
- R：oursv2 49ep 只改变数据配比；入口 code_painting/run_oursv2_49ep_pipeline.sh。
- S：graspnet 使用同一 25 ID，在关键帧选择 AnyGrasp top score 且不使用人手朝向限制。
- 当前 oursv2_piper0515 已是双臂 base frame；旧 300 条合并 repo 的 ours 半段仍是 world frame。
