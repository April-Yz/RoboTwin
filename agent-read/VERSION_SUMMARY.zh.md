# 版本摘要

## BASELINE

原始 RoboTwin/ALOHA 工作流及早期 Piper joint-space motion baseline。用于场景、关节和渲染诊断，不是当前推荐的 Piper Cartesian 抓放方案。

## V1

Piper Cartesian IK 基础版。单次 IK 后使用线性关节插值。当前作为速度和稳定性优先的默认版本。

## V2

在统一的连续分段轨迹协议上包含三个变体：V2 使用三次插值；V3 使用 MotionGen 并带回退；V4 使用多种子 IK 和三次插值。四个 IK 变体共用轨迹 schema v2、动作顺序、连续 endpoint、lift/place 修正和相机输出。

## 当前推荐

默认使用 `demo_piper_ik_seq_v1`。需要更平滑轨迹时使用 V2；研究 MotionGen 时使用 V3；需要多种子 IK 时使用 V4。旧 `demo_piper_ik_v*` pickle 与当前接口不兼容。

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

O.2 是 O.1.2 Foundation IK 的任务扩展，不改变 V1-V4 IK 语义。新增 `pnp_tray_piper_ik_foundation`，将 Foundation NPZ 对象映射为左 `left_dark_red_cup`、右 `right_bottle`，使用 pnp_tray 的手工关键帧和 `h2_pure_d435` EE target。动作顺序为 `pregrasp -> grasp -> close -> second-keyframe action -> open_gripper`。

推荐 O.2 从 V1 开始验证；pnp_tray 使用 `foundation_grasp_standoff=0.105`，因为 pick_diverse verified v2 的 `0.14` 会在 ID0 上推偏左杯。正式采集使用 `collect_foundation_piper_ik_verified.sh pnp_tray ...`，仍保存 head 和左右 wrist 视频。
