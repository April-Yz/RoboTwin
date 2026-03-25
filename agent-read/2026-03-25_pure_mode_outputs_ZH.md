# 2026-03-25 Pure 模式输出说明

## 背景

本轮修改面向 `plan_anygrasp_keyframes_r1.py` / `run_plan_anygrasp_keyframes_r1_batch.sh` 的 `pure_scene_output=1` 工作流，目标是让 pure 模式既保持主视频干净，又能同步保存后处理所需的视频和时序数据。

## pure 模式当前行为

当命令中设置：

```bash
--pure_scene_output 1
```

当前行为为：

- 主输出视频保持纯净：
  - `head_cam_plan.mp4` 不叠加左上角文字
  - 不显示 candidate gripper
  - 不显示 target axis
- 不再生成 `debug_selection_preview.mp4`
- 仍可按参数决定是否生成 `debug_execution_preview.mp4`
- 会额外保存 wrist 相机主视频：
  - `left_wrist_cam_plan.mp4`
  - `right_wrist_cam_plan.mp4`
- 会自动保存 `pose_debug.jsonl`，即使命令中没有显式传 `--save_pose_debug 1`

## 新增/保留的主要输出文件

典型输出目录中与 pure 模式直接相关的文件为：

- `head_cam_plan.mp4`
  - head camera 主规划视频
- `left_wrist_cam_plan.mp4`
  - 左手 wrist camera 规划视频
- `right_wrist_cam_plan.mp4`
  - 右手 wrist camera 规划视频
  - planner 导出前会统一做顺时针 90 度图像旋转修正，以匹配预期观看方向
- `pose_debug.jsonl`
  - 每帧一行 JSON 的时序状态记录，便于后处理和视频对齐
- `plan_summary.json`
  - 汇总本次运行的参数、候选、阶段结果、输出路径

若命令仍保留：

```bash
--save_debug_execution_preview 1
```

则还会保存：

- `debug_execution_preview.mp4`
- `debug_execution_metrics.jsonl`

## pose_debug.jsonl 数据说明

`pose_debug.jsonl` 为 JSON Lines 文件，每一行对应一次 `record_frame(...)` 写视频时刻。当前字段包括：

- `record_index`
  - 当前 planner 记录帧序号，从 0 递增
- `active_frame`
  - 当前关联的 replay/object 帧号；无活动关键帧时可能为 `null`
- `stage`
  - 当前阶段，例如 `init`、`pregrasp`、`grasp`、`close_gripper`、`action`
- `overlay_lines`
  - 当帧原本用于画面叠字的文本列表；pure 模式下视频虽然不显示，但这里仍保留，方便对齐逻辑状态

### 相机位姿

- `current_head_camera_pose_world_wxyz`
- `current_left_wrist_camera_pose_world_wxyz`
- `current_right_wrist_camera_pose_world_wxyz`
- `replay_head_camera_pose_world_wxyz`

以上均为世界坐标系下的 7 维 pose：

```text
[x, y, z, qw, qx, qy, qz]
```

### 机器人末端位姿

- `current_left_tcp_pose_world_wxyz`
- `current_right_tcp_pose_world_wxyz`
- `current_left_ee_pose_world_wxyz`
- `current_right_ee_pose_world_wxyz`

说明：

- `tcp` 对应 planner / gripper 参考点语义
- `ee` 对应 wrist/endlink 语义
- 当前实现里：
  - `current_*_tcp_pose_world_wxyz` 直接来自 planner 侧 7 维 pose
  - `current_*_ee_pose_world_wxyz` 来自 `robot.get_*_ee_pose()` 返回的 7 维列表，而不是 `sapien.Pose`
  - 导出层现已兼容这两种 pose 表示，后处理时都可统一按 `[x, y, z, qw, qx, qy, qz]` 读取

### 机器人关节状态

- `current_left_arm_qpos_rad`
- `current_right_arm_qpos_rad`
  - 左右臂 6 维关节角，单位弧度
- `current_left_gripper_joint_qpos_rad`
- `current_right_gripper_joint_qpos_rad`
  - 当前夹爪 finger joints 的实际 joint qpos
- `current_left_gripper_command`
- `current_right_gripper_command`
  - 当前夹爪开合命令值，来自机器人内部 gripper 标量状态

### 物体状态

- `object_actor_poses`
  - 字典，key 为物体名，例如 `cup` / `bottle`
  - 每个物体当前至少包含：
    - `actor_pose_world_wxyz`
  - 若当前 `active_frame` 能对齐到 replay 物体轨迹，还会包含：
    - `replay_pose_world_wxyz`

### 其它调试数据

- `frame_metrics`
  - 当前阶段的目标/当前/物体相对误差快照

## 代码位置

- 主逻辑：
  - `code_painting/plan_anygrasp_keyframes_r1.py`
- pure 模式下关闭 selection preview：
  - `generate_debug_preview(...)`
- head / wrist 视频逐帧写出：
  - `record_frame(...)`
- `pose_debug.jsonl` 写出：
  - `record_frame(...)`
- 输出路径写入汇总：
  - `plan_summary.json` 生成段

## 使用说明

如果你想要“纯净主视频 + wrist 视频 + 对应时序数据”，推荐命令组合为：

```bash
--pure_scene_output 1 \
--overlay_text 0 \
--debug_visualize_targets 0 \
--save_debug_execution_preview 0
```

其中：

- `pure_scene_output=1` 会自动启用 `pose_debug.jsonl`
- 若还想保留执行调试视频，可把 `--save_debug_execution_preview` 改回 `1`

## 处理建议

后续做视频-状态对齐时，建议优先使用：

- `record_index`
- `stage`
- `active_frame`

作为索引键，再联动：

- `head_cam_plan.mp4`
- `left_wrist_cam_plan.mp4`
- `right_wrist_cam_plan.mp4`
- `pose_debug.jsonl`

这样可以同时对齐：

- 画面
- TCP/EE 轨迹
- 关节角
- gripper 开合
- 物体 pose
