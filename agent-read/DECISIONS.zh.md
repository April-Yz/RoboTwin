# 长期决策

## 2026-07-14：Dense Replay 修复采用隔离版本

- 保留旧 renderer、runner 和论文素材，方便复现实验历史。
- 新实现命名为 `Dense Replay URDF-match v2`，写入独立 `h2_pure_d435_urdfmatch_v2` 输出根目录。
- 关节顺序保持 `joint1..joint6`；固定误差通过显式坐标 adapter 修复，不通过交换或手工偏置关节修复。
- HaMeR 指尖中点统一解释为 TCP；link6 仅是 IK 的内部目标帧。
- Dense 仍是 dense retargeting baseline。机器人不可达的人手姿态不由该修复伪装为 Ours v2 能力。
