# 版本摘要

## BASELINE

原始 RoboTwin/ALOHA 工作流及早期 Piper joint-space motion baseline。用于场景、关节和渲染诊断，不是当前推荐的 Piper Cartesian 抓放方案。

## V1

Piper Cartesian IK 基础版。单次 IK 后使用线性关节插值。当前作为速度和稳定性优先的默认版本。

## V2

在统一的连续分段轨迹协议上包含三个变体：V2 使用三次插值；V3 使用 MotionGen 并带回退；V4 使用多种子 IK 和三次插值。四个 IK 变体共用轨迹 schema v2、动作顺序、连续 endpoint、lift/place 修正和相机输出。

## 当前推荐

默认使用 `demo_piper_ik_seq_v1`。需要更平滑轨迹时使用 V2；研究 MotionGen 时使用 V3；需要多种子 IK 时使用 V4。旧 `demo_piper_ik_v*` pickle 与当前接口不兼容。

## Piper TCP/EE IK V3（独立坐标转换线）

2026-07-13 新增隔离的 Piper TCP/EE IK V3，用于修复 current OursV2 中“字段名是 EE、内容实际是带 12 cm 的 TCP”所导致的 pose->link6 IK 不可逆问题。该 V3 与上面的 MotionGen V3 不是同一条版本线；它使用新 renderer/planner/runner，默认 `ours_tcp` 语义和 `human_replay_v3/` 输出，不修改现有 OursV2。详见 `agent-read/PIPER_IK_V3.zh.md`。

## Dense Replay URDF-match v2（隔离 baseline 修复线）

2026-07-14 新增隔离的 Dense Replay v2。它不改变 Dense 的逐帧人手 retargeting 定义，只修复旧视频中的规划/执行对应错误：统一 `piper_pika_agx`、加入 Curobo→SAPIEN link6 固定 `Ry(-90 deg)` adapter、严格反演 0.12 m TCP、恢复 10 个插值 waypoint，并按实测关节收敛。旧代码和旧输出保留。详见 `agent-read/COMMANDS/dense_replay_urdfmatch_v2.zh.md`。

同日增加六任务顺序批处理和独立论文扩展图。424 个输入写入 `h2_pure_d435_urdfmatch_v2`，V1 保持不变；当前仅 raw replay 是 V2，已有 Dense Stage-2 repaint/HDF5 仍属于 V1，不能作为匹配的 V2 下游结果。

## Selection Strategy Audit V4（只读审计线）

2026-07-14 新增独立 V4 审计工具。它只读取现有 OursV2、人手关键帧 preview、Top-score plan summary、AnyGrasp JSON、Foundation replay 和 0515 标定，不调用 planner。V4 从 `selected_candidates_by_executed_arm` 获取真实 Top-score 候选，分别保留旧 raw/legacy 语义和 audit-only canonical 重建；Selection/Planner 双面板会把不同 resolved frame 的 Foundation 画面分栏显示。后续同日更新加入扁平 `id<ID>_keyframe_*` 文件、抗遮挡线型和独立 agreement/position/contribution 统计脚本。输出位于 `code_painting/selection_strategy_compare_v4/` 且不进入 Git。详见 `agent-read/SELECTION_STRATEGY_AUDIT_V4.zh.md`。

## O.1 Foundation 变体

`demo_piper_ik_foundation_v1..v4` 保留相同 IK 版本语义，但把随机 RoboTwin bottle 替换为 Foundation NPZ 的位置和原始 OBJ。O.1 使用显式 frame；O.1.1 用第一标注关键帧建场；O.1.2 使用第二关键帧 EE xyz 替代 lift/place；O.1.2.1 增加不改原始 0515 文件的逐侧 wrist 前移/roll tuning。推荐从 V1 开始；默认使用底部 `support_proxy` 和无瞬移抓取状态门控。pickle 要求 Foundation mode/source/keyframes/action/几何上下文完全匹配。批采集使用 run tag 隔离输出、每 ID 一个 episode 和有限 seed 重试；视频可再按 Foundation ID 索引为 `episode<ID>`。

### O.1.2.1 Wrist 调试补充

O.1.2.1 将可确定的父帧拼接错误与尚未测量的 `link6_T_real_tcp` 分开，并提供 viewer 同帧左右/拼接视频录制和参数 JSON。该功能不改变 V1-V4 IK 语义或正式采集轨迹。
Debug 视频现使用 H.264/faststart；正式采集 wrapper 可通过四个 `WRIST_*` 环境变量无 viewer 覆盖相机参数。
Viewer 可用 `--show_camera_frustums 1` 校验并显示 wrist/head camera linesets；`--hold 1` 已改为保持最终窗口直到用户退出。
2026-06-16 修复 Piper IK 自定义执行器遗漏逐步 `viewer.render()` 的问题；现在支持“仅实时 SAPIEN”以及“实时 SAPIEN + wrist RGB”两个模式。
同日新增 wrist 前向轴诊断脚本，当前结果显示相机 forward 与 Pika 物理 `+X` 基本对齐，开合平面误差小于 1 度；旧 debug `+Z` 的约 90 度差异不应直接作为外参修正。Viewer 现在可用 `--wrist_left_yaw_deg` / `--wrist_right_yaw_deg` 应用这个小的父坐标系 yaw。


## 2026-06-16：Foundation O.1/O.1.2 gripper standoff

Foundation Piper IK V1-V4 的默认 `foundation_grasp_standoff` 从 `0.085m` 改为 `0.105m`。这属于 O.1/O.1.2 抓取深度默认值调整：gripper base/EE grasp 目标离瓶子中心更远 2cm，使物体更靠近夹爪指尖/剪刀口；接口仍兼容，可用 viewer `--foundation_grasp_standoff_m` 或采集环境变量 `FOUNDATION_GRASP_STANDOFF_M` 覆盖。


## 2026-06-16：Wrist pitch/lateral 调试接口

Foundation Piper IK viewer 新增 wrist 相机 `parent_pitch_deg` 与 `parent_lateral_offset_m` 临时覆盖入口，用于验证 0515 wrist 标定“共面但不俯视”的问题。当前推荐起步值是左右 pitch `15deg`、右手 lateral `+0.0067m`；这些是 viewer/debug 参数，不改变 gripper 抓取规划。


## 2026-06-16：O.1.2 verified grasp/wrist v2

当前推荐的 O.1.2 viewer baseline 使用 `foundation_grasp_standoff_m=0.14`、wrist forward `0.145/0.13`、pitch `15deg`、lateral `-0.0207/0.0274`。同时新增真实抓取 debug 参数，可在 viewer 中切换 collision proxy、要求两指接触、关闭 grasp-assist 做纯物理观察。

补充：verified grasp/wrist v2 推荐命令显式包含 `foundation_capture_radial_tolerance_m=0.08` 与 `foundation_grasp_assist_max_distance_m=0.16`，否则默认门控对 `standoff=0.14` 偏严格。


## 2026-06-16：O.2 pnp_tray Foundation IK

O.2 是 O.1.2 Foundation IK 的任务扩展，不改变 V1-V4 IK 语义。新增 `pnp_tray_piper_ik_foundation`，将 Foundation NPZ 对象映射为左 `left_dark_red_cup`、右 `right_bottle`，使用 pnp_tray 的手工关键帧。默认 action target 来源为 Foundation 第二关键帧 OBJ center，而不是 `h2_pure_d435` EE target。动作顺序为 `pregrasp -> grasp -> close -> object-keyframe action -> open_gripper`。

推荐 O.2 从 V1 开始验证；pnp_tray 使用 `foundation_grasp_standoff=0.105`，因为 pick_diverse verified v2 的 `0.14` 会在 ID0 上推偏左杯。可选 `foundation_pregrasp_clearance=0.06` 可用于抬高 pregrasp 的避障试验，默认不启用。正式采集使用 `collect_foundation_piper_ik_verified.sh pnp_tray ...`，仍保存 head 和左右 wrist 视频。

## 2026-07-15：PiperCanonicalTCP-v1（独立 Real-TCP 链）

新增独立链路 `code_painting/piper_canonical_tcp_v1/`，不修改 OursV2 或 Piper IK V3。它把 `L6_SIM`、`L6_URDF`、`RTCP` 和 `CGRASP` 分开命名；运行时验证 `T_L6SIM_L6URDF=Ry(+pi/2)`，服务器工具严格保持 `T_L6URDF_RTCP=Ry(-1.57)@Tx(0.19)`。支持 corrected same-q joint 对比及 Orientation/Fused/Top-score EE-pose 对比。详见 `PIPER_CANONICAL_TCP_V1.zh.md`。

### 2026-07-16：Real control compare v1.x 增量

在同一隔离目录中新增 `real_control_compare.v1`，不提升 major version、不修改 OursV2。它用 Piper raw episode 的同步 q/endPose 做 Joint 与 EE-pose 两类三链路对比，输出与 2026-07-15 候选策略批次分离。Joint/EE-pose 两支视频均以 0515 world XYZ 作图，局部 TCP 轴继续用红/绿/蓝表示 +X/+Y/+Z。
