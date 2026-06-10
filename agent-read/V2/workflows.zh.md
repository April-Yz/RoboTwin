# 工作流

Phase 1 依次规划并执行 pregrasp、grasp、lift、place，在 close 后修正 place 偏移，通过真实成功判定后保存轨迹。Phase 2 重建相同动作序列，校验 v2 pickle，回放保存数据并生成 instruction。
