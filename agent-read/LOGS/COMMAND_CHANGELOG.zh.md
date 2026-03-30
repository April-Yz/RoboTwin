## 2026-03-27 00:00:00 +08

- 新增 pi0 数据转换命令：`process_repainted_headcam_with_wrist.py`
  - 入口：
    - `policy/pi0/scripts/process_repainted_headcam_with_wrist.py`
  - 用途：
    - 把新的 head-cam repaint 结果与 retarget wrist 回放统一转成 `processed_data` HDF5
  - 关键参数：
    - `--head-root`
    - `--head-dir-template`
    - `--head-video-name`
    - `--retarget-root`
    - `--retarget-dir-template`
    - `--review-json`
    - `--ids`
    - `--ignore-ids`
  - 典型命令：
    - `python scripts/process_repainted_headcam_with_wrist.py d_pour_blue "pour water" 48 --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' --head-video-name target_with_original_head_cam_plan.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue --retarget-dir-template 'hand_detections_{id}' --ignore-ids`

## 2026-03-25 19:15:00 +08

- 新增底盘遮挡板参数（visual-only）
  - 参数：
    - `--base_occluder_enable 0|1`
    - `--base_occluder_local_pos X Y Z`
    - `--base_occluder_half_size HX HY HZ`
    - `--base_occluder_color R G B`
  - 入口：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/render_hand_retarget_r1_npz.py`
  - 用途：
    - 在机器人 base 上方添加一个随 base pose 移动的白色挡板，遮住底盘/底座
    - 当前只加 visual，不加 collision
  - 使用说明：
    - 适合 pure/debug 视频清理画面
    - `local_pos` 用来控制高度和前后左右偏移
    - `half_size` 用来控制挡板长宽厚

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

- 新增命令参数：`--debug_collision_report`
  - 入口：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - 用途：
    - 在 `close_gripper` 渐进闭合阶段打印更强的碰撞调试信息。
  - 主要输出：
    - `[collision-debug-init]`
    - `[collision-debug-step]`
    - 常规 `[gripper-close]` 新增 `base_contact=...`
  - 调试重点：
    - 区分 `finger_contact` 和 `base_contact`
    - 打印 `finger_pairs` / `base_pairs`
    - 查看目标物体、`left/right_gripper_link`、finger links 的 collision-shape 摘要
  - 默认值：
    - `0`

- 新增命令参数：`--execution_object_collision_mode {convex,solid_bbox}`
  - 入口：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - 用途：
    - 控制 execution object 在执行阶段使用的 collision 几何。
  - 模式：
    - `convex`
      - 保持原来的 `add_convex_collision_from_file`
    - `solid_bbox`
      - 读取 mesh bounds
      - 用单个 axis-aligned box 创建“实心” collision
  - 说明：
    - 只影响 execution collision，不改视觉 mesh
    - 当 `--replay_objects_ignore_collision 1` 且对象未被纳入抓取/动作碰撞时，仍然不会创建 collision
  - 默认值：
    - `convex`

- 新增命令参数：`--gripper_contact_monitor_mode {fingers,fingers_and_base,all_robot_links}`
  - 入口：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - 用途：
    - 控制 `close_gripper` 阶段哪些 robot links 可以触发接触监控。
  - 模式：
    - `fingers`
      - 仅 finger links
    - `fingers_and_base`
      - finger links + `left/right_gripper_link`
    - `all_robot_links`
      - 当前 arm 对应 articulation 的全部 links
  - 说明：
    - 这是停机判据使用的监控集合，不只是打印
    - 当前非常适合用来排查“finger/base shapes=0，但其他 link 是否有 collision”这一类问题

## 2026-03-25 23:10:00 +08

- 新增最小碰撞探针命令：
  - 入口脚本：
    - `code_painting/minimal_gripper_collision_probe.py`
  - 用途：
    - 不经过 AnyGrasp/IK/stage 主流程，直接验证 R1 gripper 与简单 box 或 mesh(solid_bbox/convex) 物体之间的 raw scene contact
  - 代表命令：
    - box：
      - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/minimal_gripper_collision_probe.py --arm left --object_kind box --probe_local_offset 0.04 0.0 0.0 --max_iters 20 --settle_steps_per_iter 8`
    - mesh：
      - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/minimal_gripper_collision_probe.py --arm left --object_kind mesh --mesh_path /home/zaijia001/ssd/data/R1/hand/obj_mesh/blue_cup/blue_cup.obj --mesh_collision_mode solid_bbox --probe_local_offset 0.04 0.0 0.0 --max_iters 20 --settle_steps_per_iter 8`
  - 关键输出：
    - 终端逐步打印：
      - `qpos`
      - `raw_contact_total`
      - `target_contact_total`
      - `target_contacts`
    - JSON：
      - `code_painting/minimal_gripper_collision_probe/probe_left_box.json`
      - `code_painting/minimal_gripper_collision_probe/probe_left_mesh.json`

- `--debug_collision_report 1` 调试输出增强：
  - 新增 close 阶段 raw target contact 打印：
    - `raw_target_contacts`
    - `raw_target_contact_total`
    - `[gripper-close] ... raw_target_contact=0|1`
  - 相关代码：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 用途：
    - 区分“monitor/helper 没匹配到接触”和“raw physics contact 本身就不存在”

- `--debug_collision_report 1` 输出继续增强：
  - 新增：
    - `target_pose=...`
    - `target_collision_debug=...`
  - 用途：
    - 直接查看 close 阶段目标物体 actor pose 是否稳定
    - 直接查看 `solid_bbox` 的 `center/half_size`

- 新增参数：`--debug_visualize_object_collision_bbox 0|1`
  - 入口：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - 用途：
    - 在 `execution_object_collision_mode=solid_bbox` 下，额外显示物体 collision bbox
  - 典型用法：
    - 在原命令末尾追加：
      - `--debug_visualize_object_collision_bbox 1`

- 新增参数：`--grasp_action_object_collision_start_stage {close_gripper,grasp,pregrasp}`
  - 用途：
    - 控制 selected execution objects 从哪个 stage 开始参与碰撞
  - 典型实验：
    - `--enable_grasp_action_object_collision 1 --grasp_action_object_collision_start_stage pregrasp --execution_object_collision_mode convex`

- 新增 close 前 pose 导出：
  - 输出文件：
    - `close_stage_snapshot_dual_before_close.json`
    - `close_stage_snapshot_<arm>_before_close.json`

- 新增参数：`--execution_object_scale_override NAME=S|SX,SY,SZ`
  - 用途：
    - 单独缩放 execution object 的 visual mesh 与 collision geometry
  - 典型示例：
    - `--execution_object_scale_override cup=0.9`
    - `--execution_object_scale_override bottle=0.9`

- 新增参数：
  - `--execution_object_visual_scale_override NAME=S|SX,SY,SZ`
  - `--execution_object_collision_scale_override NAME=S|SX,SY,SZ`
  - 用途：分别控制 execution object 的 visual mesh 与 collision shape 缩放
