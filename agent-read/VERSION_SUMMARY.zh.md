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

`demo_piper_ik_foundation_v1..v4` 保留相同 IK 版本语义，但把随机 RoboTwin bottle 替换为 Foundation NPZ 的位置和原始 OBJ。O.1 使用显式 frame；O.1.1 用第一标注关键帧建场；O.1.2 使用第二关键帧 EE xyz 替代 lift/place。推荐从 V1 开始；默认使用底部 `support_proxy` 和无瞬移抓取状态门控。pickle 要求 Foundation mode/source/keyframes/action/几何上下文完全匹配。
