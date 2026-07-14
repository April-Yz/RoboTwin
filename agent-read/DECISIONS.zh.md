# 长期决策

## 2026-07-14：Dense Replay 修复采用隔离版本

- 保留旧 renderer、runner 和论文素材，方便复现实验历史。
- 新实现命名为 `Dense Replay URDF-match v2`，写入独立 `h2_pure_d435_urdfmatch_v2` 输出根目录。
- 关节顺序保持 `joint1..joint6`；固定误差通过显式坐标 adapter 修复，不通过交换或手工偏置关节修复。
- HaMeR 指尖中点统一解释为 TCP；link6 仅是 IK 的内部目标帧。
- Dense 仍是 dense retargeting baseline。机器人不可达的人手姿态不由该修复伪装为 Ours v2 能力。

## 2026-07-14：Selection Strategy V4 仅作为只读审计

- 保留 OursV2、Orientation、Fused、Top-score 的旧算法、summary 和可视化，不用 V4 回写或“修正”历史结果。
- Top-score 的真实选择以 `plan_summary.json -> selected_candidates_by_executed_arm` 为准；旧 rank preview 仅作为历史错误证据。
- raw/legacy Top-score 和 canonical 重建同时保留，canonical 结果明确标为 audit-only，不冒充历史执行结果。
- resolved frame 不同则使用各自 Foundation 背景分栏，不跨帧静默投影。
- 同 pose 的 Orientation/Fused 不做人为位置偏移；使用粗实线/细虚线和不同 marker 同址显示，避免改变数据语义。
- 任务输出采用 `<TASK>/id<ID>_keyframe_<FRAME>_*`，不再创建 episode 中间目录；旧嵌套版保存在独立可回滚备份。
- 批量 PNG/JSON/报告保持 Git ignore；只版本化两个脚本和双语说明。
