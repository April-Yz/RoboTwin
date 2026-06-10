# RoboTwin 项目概览

本仓库在 RoboTwin 仿真、数据采集和策略工作流基础上，维护 Piper/Pika 双臂场景、Cartesian IK 抓放、手部/物体 replay 与 AnyGrasp 规划工具。

## 当前推荐工作流

- Piper IK 双瓶抓放：使用 `pick_diverse_bottles_piper_ik` 与 `demo_piper_ik_seq_v1..v4`。
- Foundation OBJ 对比：使用 `pick_diverse_bottles_piper_ik_foundation` 与 `demo_piper_ik_foundation_v1..v4`，从 NPZ 读取物体位置和原始网格。
- 默认 IK：V1；V2 使用三次插值，V3 使用 MotionGen 并带 IK 插值回退，V4 使用多种子 IK。
- 数据流程：Phase 1 找稳定且真实成功的 seed 并保存版本化轨迹；Phase 2 在相同 seed 场景中校验、回放并保存 HDF5/视频/instruction。
- 相机：head、front、side、右侧 `third_camera`、对向俯视 `opposite_top_camera`，以及 top-level `third_view`。

## 环境与入口

- Conda：`RoboTwin_bw`
- 采集：`collect_data.sh`、`script/collect_data.py`
- Piper IK viewer：`view_pick_diverse_bottles_piper_ik_motion.py`
- 任务：`envs/pick_diverse_bottles_piper_ik.py`
- IK：`envs/robot/piper_ik.py`

命令详见 `agent-read/COMMANDS/piper_ik_cartesian.zh.md` 和 `piper_ik_foundation.zh.md`，版本关系详见 `agent-read/VERSION_SUMMARY.zh.md`。
