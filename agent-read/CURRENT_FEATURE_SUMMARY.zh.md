# 当前功能摘要

## 当前主线

- 仓库默认版本仍按最新 v1.x 迭代管理；Foundation、AnyGrasp、OursV2、Dense Replay、Piper IK V3 和 PiperCanonicalTCP-v1 是独立实验线。
- 当前 Real-Piper-TCP 对比入口是 `PiperCanonicalTCP-v1`；它不修改 OursV2。

## 本轮新增

- planner target、current readback、reach check 和可视化统一为 `T_W_RTCP`。
- SAPIEN `L6_SIM` 与 CuRobo/server `L6_URDF` 原点一致、局部轴固定差精确 `Ry(+pi/2)`；适配后同-q FK 误差小于 `7.5e-8 m / 0.000016 deg`。
- 服务器工具保持字面量 `T_L6URDF_RTCP = Ry(-1.57) @ Tx(0.19)`。preview 的 `CGRASP -> RTCP` remap 是另一层独立变换。
- corrected same-q OursV2 TCP 与 Real TCP 左右 mean/max distance 都约 `70.0001 mm`，对应统一前进轴上的 12 cm 与 19 cm 差；旧 224.6 mm 结论无效。
- EE-pose 支持 Orientation、Fused、Top-score。`pnp_bread/id8/left` 三策略与 corrected joint 对比全部通过媒体/ffprobe/视觉 QA。
- 6 tasks × 5 episodes manifest、独立 batch orchestrator 和 tmux 命令已准备。策略 IK miss 保留视频与 failures TSV，不伪造 SUCCESS。
- canonical MP4 现在有统一的 H.264/`yuv420p`/faststart 后处理与严格原子转码审计；2026-07-15 batch 的 186 个 `mpeg4` 已转换，终态 258/258 完整解码通过。joint summary 也显式记录 OursV2 human-replay 输入和仿真 head-camera 来源语义。
- Selection Strategy Audit V4 与 Dense Replay URDF-match v2 仍为独立历史线。

## 读取顺序

1. `README.zh.md`
2. `CURRENT_FEATURE_SUMMARY.zh.md`
3. `VERSION_SUMMARY.zh.md`
4. `PIPER_CANONICAL_TCP_V1.zh.md`
5. `COMMANDS/piper_canonical_tcp_v1.zh.md`
6. `SELECTION_STRATEGY_AUDIT_V4.zh.md`
