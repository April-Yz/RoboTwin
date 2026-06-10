# 项目理解

## 主要结构

- `envs/`：任务、机器人控制、观测与 HDF5 转换。
- `assets/embodiments/`：机器人 URDF、关节和静态相机配置。
- `task_config/`：任务采集配置。
- `description/`：任务 prompt 与 episode instruction 生成。
- `code_painting/`：手部/物体 replay、FoundationPose 和 AnyGrasp 相关工作流。

## Piper IK 数据流

输入是带 seed 的双瓶场景、左右瓶功能点、放置目标和 IK 配置。任务生成 pregrasp、grasp、lift、place 四个 Cartesian move 目标。每段 IK 输出 `position[N,6]` 和 `velocity[N,6]`，随后按上一段末端关节状态继续规划。

轨迹 pickle 输出 schema、版本、IK 版本、动作名、Cartesian 目标和左右关节轨迹。Phase 2 只接受当前 v2 schema，并在相同 seed 下回放。观测最终写入 HDF5，各 RGB camera 写入独立 MP4。

## 模型/后端职责

- V1/V2/V4：基于 Piper URDF 的 IK 求解，分别使用线性、三次和多种子策略。
- V3：先求有效 IK 终点，再尝试 MotionGen；MotionGen 不可用或失败时回退到该终点的三次轨迹。
- SAPIEN：执行关节命令、接触仿真、相机渲染和成功判定。

## 已知边界

- V3 的 MotionGen 优化在当前场景可能失败，但回退路径已实测成功。
- viewer 与采集轨迹逻辑一致；采集多一层 pickle schema 校验和观测保存。
- `save_all_episodes` 仅用于调试，不应用于正式成功数据筛选。
