# Piper Legacy-original / Canonical-RTCP 2×4 语义对比

## 结论

旧目录 `outputs_ik_logic_grid_20260716/` 的 V1 对比无效：它把同一数值 `T_W_RTCP` 同时送给两行，却把 Legacy 行标成“原 OursV2”。原 OursV2 实际还包含候选局部 `+Z` 的 `-0.05 m` target offset；Human Replay 则用 `0.14 m @ local human +Z` retreat。V1 丢掉这些输入适配，因此出现“Legacy EE/link6 原点直接对到物体”的现象。

V1 还有一个独立错误：Human 使用 `reuse_plan_summary_json`，该路径不会执行 `candidate_orientation_remap_label`。所以即使参数写了 `CGRASP_HUMAN -> RTCP`，存储位姿也没有真正变轴。

V2 不改 OursV2 源码。它从同一个语义候选/人手中心出发，再分别走两行的正确原生适配：

| | Orientation | Fused | Top-score | Human Replay |
|---|---|---|---|---|
| 上行 | Legacy original | Legacy original | Legacy original | Legacy original |
| 下行 | Canonical RTCP | Canonical RTCP | Canonical RTCP | Canonical RTCP |

## 输入到底是什么

共同输入不是“相同数值 planner target”，而是相同的语义源：同一个 AnyGrasp candidate center，或同一个 Human hand/gripper center。两行的数值 planner target 本来就应该不同。

### Legacy original（上行）

- Orientation/Fused：保留原 CGRASP 局部轴，target 沿局部 `+Z` 加 `-0.05 m`；pregrasp 沿局部 `+Z` 为 `0.12 m`。
- Top-score：保留 AnyGrasp 原生轴，同样使用 `-0.05 m @ local +Z` target offset 和 `0.12 m @ local +Z` pregrasp。
- Human Replay：完整复现原 `0.14 m @ local human +Z` retreat 规则；这个参数也参与 handover 关键帧修正，不是可在最后随意删掉的一次平移。
- renderer：`HandRetargetPiperDualURDFIKRenderer` / `robot._trans_from_gripper_to_endlink(...)`。
- 原生求解设置：宽松旋转阈值 `3.14 rad`、单 seed、EE reach/readback。

### Canonical RTCP（下行）

- Orientation/Fused：candidate center 原点不变，做一次 `R_W_RTCP = R_W_CGRASP @ R_CGRASP_RTCP`。
- Top-score：candidate center 原点不变，原生轴直接按 RTCP 解释。
- Human Replay：先使用与 Legacy 相同的 `0.14 m` Human recipe 生成同一语义源，再撤销 retreat，并把 CGRASP_HUMAN 旋转真正物化为 RTCP；最终 Canonical target retreat 为 `0`。
- pregrasp：`0.12 m @ local_RTCP +X`。
- renderer：`PiperCanonicalTCPRenderer`，严格反演服务器 `T_L6URDF_RTCP = Ry(-1.57) @ Tx(0.19)`。
- RTCP target 对应的 link6 原点满足 `p_L6 = p_RTCP - 0.19 * local_RTCP_X`。
- 原生求解设置：严格旋转阈值 `0.12 rad`、20 seeds、TCP reach/readback。

因此 V2 是“同一语义源经过两套完整原生链路”的效果比较，不是只改变一个 link6 转换的单变量消融。

## 审计

`semantic_source_audit.json` 同时检查：

- arm/frame/candidate 身份；
- 两行语义源的 world xyz；
- Orientation/Fused/Human 的 `CGRASP -> RTCP` 旋转关系；
- Top-score 的原生轴关系；
- 每格 target contract；
- Canonical `link6 - RTCP = [-0.19, 0, 0]`（局部 RTCP）。

任何检查失败都会阻止 2×4 合成。

## 相机配置

旧 V2 样本曾混用两套渲染参数：前三个 AnyGrasp 策略为 `640×360`、`fovy=90°`、10 fps 的广角诊断视图，Human Replay 为 `640×480`、`fovy=42.499880046655484°`、5 fps 的 D435 标定视图。虽然两边使用同一个 head pose，这种 FOV/分辨率差异会让前三格看起来视野更大、机器人更小，不能直接比较投影。

当前 runner 显式要求 `--camera-profile d435|wide`：

- `d435`：`640×480`、`fovy=42.499880046655484°`、5 fps；
- `wide`：`640×360`、`fovy=90°`、10 fps。

合成器会检查 8 个源视频的 profile、宽、高和 fps；只要有一格不同就拒绝合成。wide 是保留的诊断视角，不代表实体 D435 内参。

## 输出

默认根目录：

`code_painting/piper_canonical_tcp_v1/outputs_ik_semantic_grid_v2_20260716/`

输出按相机 profile 隔离源文件和审计元数据，最终视频直接扁平化保存到 `vis/`：

- `vis/<task>_id<id>_vd435.mp4`：8 格统一 D435；
- `vis/<task>_id<id>_vwide.mp4`：8 格统一广角；
- `_grid_meta/<profile>/<task>/foundation_input_<id>/semantic_source_audit.json`：语义源、轴变换、相机 profile、target contract 与 19 cm link6 审计；
- `_grid_meta/<profile>/<task>/foundation_input_<id>/legacy_original_vs_canonical_rtcp_2x4_<profile>.manifest.json`：源视频、执行状态和媒体属性；
- `_sources/<profile>/legacy_original/...`、`_sources/<profile>/canonical_rtcp/...`：8 个单元；
- `_superseded/canonical_human_before_shared_source_fix_20260716/`：保留的中间错误样本，不参与最终视频。

当前 6×1×2 批次固定为：`pick_diverse_bottles/id0`、`place_bread_basket/id0`、`stack_cups/id0`、`handover_bottle/id1`、`pnp_bread/id7`、`pnp_tray/id0`。`pnp_bread/id1` 不具备完整输入，因此选用首个完整交集 `id7`。

## handover_bottle / foundation_input_1 验证

- 4 种策略全部通过语义审计；所有源点位置差为 `0.0 m`，旋转矩阵误差为 `0` 到 `4.2e-16`。
- V2 Legacy Orientation 与历史 `viewer_gripper`、V2 Legacy Top-score 与历史 `S_graspnet_topscore...` 的 candidate identity、raw pose、planner target 最大差均为 `0.0`，证明上行已恢复原输入逻辑。
- Canonical Human 完成日志内部完整 handover（`[handover] SUCCESS`）；通用 summary 仍因左臂早期 action miss 返回失败状态，因此不能只看最终 `execution_success` 判断 handover 流程。
- 最终 MP4 为 `1920×648`、H.264 High、`yuv420p`、5 fps、265 帧/53 s；完整解码与中间帧视觉检查通过。

D435 profile 重跑后，8 个源均验证为 `640×480 @ 5 fps`，最终扁平视频为 `vis/handover_bottle_id1_vd435.mp4`，H.264 High、`yuv420p`、完整解码和视觉检查均通过。

## 为什么 Canonical AnyGrasp 看起来不动

在 `handover_bottle/id1` 的 Orientation/Fused 中，Canonical 不是“两边都没有 IK”：右臂计划成功，左臂计划失败；但 Canonical 启用了 `dual_stage_require_all_plans=1`，日志随后明确跳过整个 stage，因此画面不动。Canonical 还使用 `0.12 rad` 旋转阈值、20 seeds、物理 RTCP reach 与 10° reach 姿态容差。

Legacy 会动并不表示它到达同一个物理 RTCP。Legacy AnyGrasp 使用 `-0.05 m @ local +Z` 的旧 target、旧 gripper/endlink 语义、`3.14 rad` IK 姿态阈值、180° reach 姿态容差和 EE reach；Orientation/Fused 还允许 `dual_stage_require_all_plans=0`。因此它可以执行单臂/宽松解，但目标语义和验收标准均与 Canonical 不同。Canonical Human 与 Legacy Human 视觉上较接近，是因为二者共享 Human 语义源；Canonical Human 的 handover 状态机实际上已完成。

IK miss 仍可能由候选姿态不可达、严格旋转阈值或双臂 gate 导致。它与本次已修复的“输入语义错接”是两个独立问题。
