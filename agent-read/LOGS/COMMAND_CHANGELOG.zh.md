## 2026-03-25 18:55:00 +08

- R1 planner wrist 相机导出语义调整
  - 行为：
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - 不再依赖导出后图片旋转修正
    - 改为在 `plan_anygrasp_keyframes_r1.py` 内使用与 `galaxea_sim/robots/r1.py` 一致的 wrist 本地姿态
    - 输出尺寸恢复为原始横版 `image_width x image_height`
  - 相关代码：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 说明：
    - 这是只针对 R1 planner 链路的覆写，不改 `render_hand_retarget_r1_npz.py` 的全局默认值
    - 目的是让 wrist 视频通过相机真实挂载姿态得到正确视角，而不是靠导出后旋转图片补丁

## 2026-03-25 18:35:00 +08

- planner wrist 视频导出方向再次微调
  - 行为：
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - 改为导出前统一做 `90°` 逆时针旋转
    - writer 尺寸与旋转后的帧保持一致
  - 相关代码：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 说明：
    - 该修正基于用户实际结果：上一轮 `180°` 仍然相当于正确视角逆时针转了 `90°`
    - 不新增命令行参数

## 2026-03-25 18:20:00 +08

- planner wrist 视频导出方向再次修正
  - 行为：
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - 不再做 `90°` 旋转，改为导出前统一做 `180°` 旋转
    - 输出尺寸恢复为横版 `image_width x image_height`
  - 相关代码：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 说明：
    - 不新增命令行参数
    - 这次修正基于用户实际导出结果：原方案会把 wrist 视频写成竖版且仍然上下颠倒
    - 当前方案只修正图像平面方向，不修改相机挂载或规划坐标系

## 2026-03-25 16:45:00 +08

- `--debug_visualize_ik_waypoints 1`
  - 可视化增强：
    - 现在除了中间 TCP waypoint 外，也显示起点和终点 marker
    - 起点/终点统一使用红色 point+forward-axis marker
    - 中间 waypoint marker 缩小，便于观察手、目标轴和路径关系
  - 相关代码：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 使用说明：
    - 参数形式不变，仍然只需追加 `--debug_visualize_ik_waypoints 1`
    - 该参数只影响 viewer/debug 可视化，不改变规划与执行逻辑

## 2026-03-25 17:10:00 +08

- planner wrist 视频导出方向修正
  - 行为：
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - 现在在写出前统一顺时针旋转 90 度
  - 相关代码：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 说明：
    - 该修正不新增命令行参数
    - 只影响 planner wrist 视频文件的朝向
    - 不改变相机世界位姿或 planner 坐标系定义

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

## 2026-03-25 14:05:00 +08

- `--urdfik_cartesian_interp_steps`
  - 新增约定：
    - `-1` 表示自动 waypoint 模式
  - 自动模式规则：
    - 位移 `<= 0.05m` 时，不加中间 waypoint
    - 位移每超过一个 `0.05m` 档位，增加一个中间 waypoint
  - 示例：
    - `--urdfik_cartesian_interp_steps -1`
  - 相关代码：
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`

- 新增参数：`--urdfik_cartesian_interp_auto_step_m`
  - 入口：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - 用途：
    - 仅在 `--urdfik_cartesian_interp_steps=-1` 时生效，控制自动 waypoint 模式的平移阈值。
  - 默认值：
    - `0.05`
  - 示例：
    - `--urdfik_cartesian_interp_steps -1 --urdfik_cartesian_interp_auto_step_m 0.03`
  - 说明：
    - 值越小，中间 waypoint 越密。

## 2026-03-25 14:25:00 +08

- `planner_backend=urdfik` + `urdfik_trajectory_mode=cartesian_interp_ik`
  - 行为修正：
    - 现在执行层会真正消费 `plan["position"]` 中的整条 `joint_waypoints`
    - 不再只执行 `current_joints -> target_joints` 的端点直线
  - 相关代码：
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`
# 2026-03-25

- `--pure_scene_output 1`
  - 行为更新：
    - 不再生成 `debug_selection_preview.mp4`
    - 自动保留并输出：
      - `head_cam_plan.mp4`
      - `left_wrist_cam_plan.mp4`
      - `right_wrist_cam_plan.mp4`
    - 自动启用 `pose_debug.jsonl`
  - `pose_debug.jsonl` 当前关键字段：
    - `record_index`
    - `stage`
    - `active_frame`
    - `current_*_camera_pose_world_wxyz`
    - `current_*_tcp_pose_world_wxyz`
    - `current_*_ee_pose_world_wxyz`
    - `current_*_arm_qpos_rad`
    - `current_*_gripper_joint_qpos_rad`
    - `object_actor_poses`
  - 相关代码：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 说明文档：
    - `agent-read/2026-03-25_pure_mode_outputs_ZH.md`
    - `agent-read/2026-03-25_pure_mode_outputs.md`

- 新增命令参数：`--debug_visualize_ik_waypoints`
  - 入口：
    - `/home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `/home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - 用途：
    - 在 debug/viewer 中显示 `cartesian_interp_ik` 的中间 TCP/EE 平滑 waypoint，帮助判断是 waypoint 本身有问题，还是 IK/执行阶段出了问题。
  - 显示内容：
    - 中间 waypoint 的位置点和局部前进轴。
  - 默认值：
    - `0`
