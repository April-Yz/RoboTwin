# Piper Legacy / Canonical IK 2×4 对比

## 目的

该对比用于回答一个单一问题：在候选目标完全相同的情况下，Legacy/OursV2 与 PiperCanonicalTCP-v1 的 `T_W_RTCP -> URDF link6` 转换是否导致不同的 IK 和执行结果。

输出是 2 行 × 4 列的 D435 标定仿真视角：

| | Orientation | Fused | Top-score | Human Replay |
|---|---|---|---|---|
| 第一行 | Legacy IK | Legacy IK | Legacy IK | Legacy IK |
| 第二行 | Canonical IK | Canonical IK | Canonical IK | Canonical IK |

旧的 `canonical_vs_legacy_five_method_d435.mp4` 不是这个完整消融。旧视频包含四个 Canonical 方法和一个 Legacy Human Replay，没有 Legacy Orientation/Fused/Top-score。

## IK 的实际输入

八个格子都使用直接 `Selection Pose`，并统一表达为世界坐标中的 Real TCP：`T_W_RTCP`。

- Orientation：直接使用 robot-frame preview 选出的 Orientation candidate，做一次 `CGRASP -> RTCP` 轴映射。
- Fused：直接使用 `0.25 × AnyGrasp score + 0.75 × orientation score` 选出的 candidate，做一次相同轴映射。
- Top-score：直接使用 AnyGrasp 原生最高分 candidate，经 D435 camera -> world；原生轴按 RTCP 解释，不做 CGRASP 映射。
- Human Replay：直接使用人手夹爪姿态，经 D435 camera -> world，再做一次 `CGRASP_HUMAN -> RTCP` 轴映射。

所有方法都强制：

- 最终目标 retreat：`0 m`；
- candidate local X/Z offset：`0 m`；
- pregrasp：目标选定后单独生成，沿局部 RTCP `+X` 后退 `0.12 m`；
- reach/readback：物理 TCP；
- 位置坐标系：0515 `WORLD`；姿态轴：局部 `RTCP`。

`selection_strategy_compare_v4` 上方的 `Selection Pose` 是这里的直接输入来源。下方 `Planner Target` 已包含历史 offset、retreat、pregrasp、world/base 和 TCP/link6 处理，不作为本 2×4 的输入，否则会发生重复补偿。

## 两行唯一的语义差异

- Legacy 行：`HandRetargetPiperDualURDFIKRenderer`，使用 `robot._trans_from_gripper_to_endlink(...)`。
- Canonical 行：`PiperCanonicalTCPRenderer`，严格反演 Piper 服务器 `T_L6URDF_RTCP = Ry(-1.57) @ Tx(0.19)`。

候选选择、数值目标、pregrasp、IK 阈值、种子、轨迹和到达门控在同一列上下两格保持一致。合成前，`audit_ik_logic_inputs.py` 会比较候选 arm、frame、index 和 `pose_world_wxyz`；任何差异都会阻止视频合成。

## 旧 Human Replay 的 12/14 cm 说明

旧正式 Human Replay 的 `plan_summary_human_replay.json` 记录 `target_retreat=0.14 m @ local human +Z`。旧五路比较视频为了做显式 ablation 使用了 `0.12 m`，因此它不是旧正式结果的逐参数复现。新的 2×4 两行都使用 `target_retreat=0`，避免把历史 retreat 与 renderer 的 TCP/link6 转换叠加。

## 输出

默认根目录：

`code_painting/piper_canonical_tcp_v1/outputs_ik_logic_grid_20260716/`

单集重要文件：

- `legacy_vs_canonical_ik_logic_2x4_d435.mp4`：2×4 主视频；
- `input_equality_audit.json`：上下行输入一致性审计；
- `legacy_vs_canonical_ik_logic_2x4_d435.manifest.json`：源视频、执行状态、媒体属性与输入契约；
- `_sources/legacy/...`、`_sources/canonical/...`：八个独立方法目录；
- 每个方法的 `input_target_contract.json`：该格的输入与行特有转换。

IK/到达失败并不代表 runner 出错。只要生成了诊断视频，grid runner 会保留该格并继续；真正的链路错误会表现为没有视频、输入审计失败或合成失败。

## handover_bottle id1 单测结果

- 四列上下行各有 2 个目标，arm/frame/candidate index 完全一致，`pose_world_wxyz` 最大绝对差均为 `0.0`。
- Orientation 与 Fused 在该 id 选中相同目标；两种 IK 均无可执行严格解，左右臂 q/TCP 变化均为 0。
- Legacy Top-score 有实际运动，左右 TCP 最大位移约 `434/367 mm`，但 grasp 阶段最小位置误差仍约 `169/168 mm`；Canonical Top-score 无可执行解且不动。
- Legacy Human Replay 有实际运动，左右 TCP 最大位移约 `516/613 mm`，但目标残差长期约 `120 mm`，直接暴露旧 12 cm 末端定义并未到达同一个物理 RTCP。
- Canonical Human Replay 仅约 `19.4 mm` 小幅变化，目标位置误差仍为数百毫米；直接 human RTCP 姿态在严格 `0.12 rad` 旋转阈值和双臂 all-plan gate 下不可执行。
- 八格最终都 `execution_failed=true`，因此该视频仍是失败方式/坐标语义诊断，而不是成功效果展示。
- 主视频为 `1920×648`、H.264、`yuv420p`、5 fps、644 帧/128.8 s；全视频解码和中间帧视觉检查通过。
