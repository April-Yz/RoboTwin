# 当前功能摘要

## 当前主线

- 仓库默认版本仍按最新 v1.x 迭代管理；已有 Foundation、AnyGrasp、Ours v2、Dense Replay、Piper IK V3 等独立实验线。
- Piper/0515 定性链路和近期数据状态见 `README.zh.md`、`VERSION_SUMMARY.zh.md` 与 `ACTIVE_MEMORY.zh.md`。

## 本轮新增

- 新增只读 `Selection Strategy Audit V4`；不调用 planner，不修改 OursV2、Orientation、Fused、Top-score 或旧输出。
- 从 `selected_candidates_by_executed_arm` 恢复真实 Top-score 候选，同时显示旧 raw/legacy 语义和 canonical 重建。
- 每个关键帧输出 Selection Pose 与 Planner Target 双面板；不同 resolved frame 使用各自 Foundation 图并横向拼接。
- 输出改为 `<TASK>/id<ID>_keyframe_<FRAME>_*` 扁平文件；粗品红 Orientation、黄虚线 Fused 与黑/橙/蓝 Top 语义避免重叠遮挡。
- 新增 `analyze_selection_strategy_agreement_v4.py`：左右手分开统计同 candidate 次数、canonical xyz 距离和 Fused 加权 contribution；当前 orientation 平均占 Fused score 的 91.75%。
- 全量覆盖 6 个任务、150 个 episode、461 张关键帧图和 2192 条 arm-strategy 记录；详见 `SELECTION_STRATEGY_AUDIT_V4.zh.md`。
- 上一条隔离修复线仍为 `Dense Replay URDF-match v2`，详见 `COMMANDS/dense_replay_urdfmatch_v2.zh.md`。
- Dense Replay V2 六任务批处理已在 tmux `dense_replay_urdfmatch_v2` 中启动；424 个输入按 episode 顺序写入独立 `h2_pure_d435_urdfmatch_v2`，旧 V1 保留。论文素材新增独立 `pipeline_grid_expanded_dense_urdfmatch_v2.mp4`，其中 V2 raw 与现有 V1 repaint 被明确标为不匹配版本。

## 读取顺序

1. `README.zh.md`
2. `CURRENT_FEATURE_SUMMARY.zh.md`
3. `VERSION_SUMMARY.zh.md`
4. `SELECTION_STRATEGY_AUDIT_V4.zh.md`
5. 与任务对应的 `COMMANDS/*.zh.md`
