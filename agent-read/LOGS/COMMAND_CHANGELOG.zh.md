## 2026-03-25 12:08:00 +08

- 新增参数：`--enable_grasp_action_object_collision 0|1`
  - 入口：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/run_plan_anygrasp_keyframes_r1_batch.sh`
  - 用途：
    - 为被执行臂选中的物体在 `close_gripper` 和 `action` 阶段启用碰撞阻挡
    - 默认 `0`，保留原来的无碰撞执行模式
  - 用法：
    - 单条命令：追加 `--enable_grasp_action_object_collision 1`
    - batch：同样在 batch 命令末尾追加 `--enable_grasp_action_object_collision 1`
  - 说明：
    - 该参数不会改变 `pregrasp/grasp/action` 的目标位姿构造
    - 也不会改变物体附着到 TCP 的相对变换
## 2026-03-25 13:05:00 +08

- 为 `plan_anygrasp_keyframes_r1.py` 增加可视化模式相关参数：
  - 新参数：
    - `--debug_visualize_targets 0|1`
    - `--viewer_show_camera_frustums 0|1`
  - 用途：
    - `debug_visualize_targets=0` 可全局关闭 target axis actor
    - `viewer_show_camera_frustums=0` 可关闭 viewer 中 SAPIEN 相机线框
  - 相关代码：
    - `code_painting/plan_anygrasp_keyframes_r1.py`

- 为 `plan_anygrasp_keyframes_r1_batch.py` 同步透传可视化模式参数：
  - 新增透传：
    - `--debug_visualize_targets`
    - `--viewer_show_camera_frustums`
  - 相关代码：
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-25 13:35:00 +08

- `--enable_grasp_action_object_collision 1`
  - 行为增强：
    - `close_gripper` 阶段不再总是一次性闭合到底
    - 现在会渐进闭合，并在检测到所选物体接触且夹爪关节运动停滞时提前停止
  - 相关代码：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 使用说明：
    - 参数形式不变，仍然只需追加 `--enable_grasp_action_object_collision 1`
    - 默认 `0` 时，仍保留原来的无碰撞快速闭合模式
