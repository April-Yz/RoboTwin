# Canonical 四方法与 Legacy Human Replay 对比

每个样本生成两支 D435 标定仿真视角视频：

- `canonical_four_method_d435.mp4`：Canonical Orientation、Fused、Top-score、Human Replay。
- `canonical_vs_legacy_five_method_d435.mp4`：上述四种 + Legacy OursV2 Human Replay retreat 基线。

前四路统一使用 `T_W_RTCP`、服务器 `Ry(-1.57) @ Tx(0.19)` 和 Canonical IK。第五路只用于暴露旧局部轴/retreat 语义差异，不属于 Canonical。

| Canonical 方法 | 位置来源 | 朝向来源 |
|---|---|---|
| Orientation | AnyGrasp 候选原点 | 最接近人手朝向；`CGRASP -> RTCP`。 |
| Fused | AnyGrasp 候选原点 | 0.25 原生分数 + 0.75 朝向分数；`CGRASP -> RTCP`。 |
| Top-score | AnyGrasp 候选原点 | 原生分数最高，按 RTCP 轴解释。 |
| Human Replay | 人手夹爪原点 | 人手/CGRASP 局部轴显式映射到 RTCP。 |

所有 Canonical 方法最终目标偏移均为 0；`approach_offset_m=0.12` 只生成 pregrasp，不改变最终抓取点。

## AnyGrasp translation

原始 translation 是 D435 相机坐标，不是 world 坐标，不能直接当 world XYZ。先经过 D435 外参变换到 world 后，它才表示所选抓取候选原点。Orientation/Fused 只改变局部朝向轴，不移动原点；Top-score 也不额外移动原点。

## Retreat

- Canonical Human Replay：固定 `target_retreat_m=0`；Human pose 先转为 RTCP，再由 Canonical 反解 link6。
- Legacy 当前 wrapper 默认：`target_retreat_m=0`。
- 历史 link6 补偿实验：显式 `target_retreat_m=0.12`，沿旧人手局部 `+Z` 反向移动最终目标。

因此不要取消 Canonical 的 `approach_offset_m=0.12`；需要取消的是旧链路可能显式加在最终目标上的 `target_retreat_m`。

输出位于 `code_painting/piper_canonical_tcp_v1/outputs_replay_method_compare_20260716/<task>/foundation_input_<id>/`。旁边的 manifest 记录源视频、方法语义、retreat、帧率和时长；`_sources/` 不覆盖 2026-07-15 Canonical 或旧 OursV2 输出。

“D435 视角”是使用 D435 replay 和 0515 标定重建的仿真 head view，不是把原始 D435 RGB 重复贴入五个面板。
