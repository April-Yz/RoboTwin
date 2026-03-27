# AnyGrasp Keyframe Planner

这套脚本用于把 AnyGrasp 候选抓取结果和手部 `hand_detections_*.npz` 结合起来，在 RoboTwin 里生成一个两关键帧的规划 demo。

核心流程：

1. 从 AnyGrasp 结果中读取关键帧 `1` 和 `22` 的所有抓取候选
2. 读取人手 `npz` 中对应帧的 `left/right_gripper_rotation_matrix`
3. 按手别绑定目标物体
   - 左手默认只允许 `cup`
   - 右手默认只允许 `bottle`
4. 过滤掉距离目标物体太远的候选
5. 在剩余候选里按 gripper 朝向接近程度排序
6. 要求两个关键帧都落在同一个物体上
5. 在 RoboTwin 中执行：
   - `pregrasp`
   - `grasp`
   - `close_gripper`
   - `action`
6. 每个阶段执行后检查 TCP 是否到位，不到位则重规划


## 文件

- 单视频脚本: [plan_anygrasp_keyframes_r1.py](/home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py)
- 批处理脚本: [plan_anygrasp_keyframes_r1_batch.py](/home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py)
- 启动脚本: [run_plan_anygrasp_keyframes_r1_batch.sh](/home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh)


## 输入目录约定

### AnyGrasp

根目录例如：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results
```

其中每个视频一个目录：

```text
d_pour_blue_1/
  grasps/grasp_000001.json
  grasps/grasp_000022.json
  ...
```

### Replay

根目录例如：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue
```

其中每个视频一个目录，至少包含：

```text
d_pour_blue_1/
  multi_object_world_poses.npz
```

### Hand NPZ

目录例如：

```text
/home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis
```

其中每个视频一个文件：

```text
hand_detections_1.npz
hand_detections_2.npz
...
```


## 最常用命令

### 直接复用 preview `summary.json` 的 top1，并使用 `frame 1` 和 `max(22, -10)`

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_from_preview_fi60 \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview/d_pour_blue_batch_fi60 \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --keyframes 1 22 \
  --candidate_selection_mode top_score_auto \
  --candidate_selection_relative_frame -10 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps 30 \
  --candidate_max_rotation_distance_deg 60 \
  --left_target_object cup \
  --right_target_object bottle
```

这条命令的含义：

- 直接读取 preview 批处理输出根目录里每个视频的 `summary.json`
- 使用 preview 的 `resolved_frames`
- 实际执行关键帧是：
  - 第 1 帧
  - `max(22, resolved(-10))`
- 候选直接取 preview JSON 里的 top1：
  - 左手 `top_candidates.left_orientation[0]`
  - 右手 `top_candidates.right_orientation[0]`
- 中间轨迹平滑使用：
  - `--planner_backend urdfik`
  - `--urdfik_trajectory_mode cartesian_interp_ik`

如果要开有头界面，把下面三项加到命令末尾：

```bash
--enable_viewer 1 \
--viewer_frame_delay 0.02 \
--viewer_wait_at_end 1
```

如果你想把执行用的目标物体碰撞/视觉模型一起缩小，可以额外加：

```bash
--execution_object_scale_override cup=0.9 \
--execution_object_scale_override bottle=0.9
```

如果你想分别控制 visual 和 collision 的缩放比例，可以改用：

```bash
--execution_object_visual_scale_override cup=0.7 \
--execution_object_collision_scale_override cup=0.5 \
--execution_object_visual_scale_override bottle=0.7 \
--execution_object_collision_scale_override bottle=0.5
```

其中：

- `--execution_object_scale_override` 是旧的统一缩放参数
- `--execution_object_visual_scale_override` 只改 visual mesh
- `--execution_object_collision_scale_override` 只改 collision shape
- 如果同一物体同时给了统一缩放和专用缩放，则专用缩放优先

如果你要保留原来的“只在 `close_gripper` 阶段才让目标物体开始参与碰撞”的逻辑，再显式加：

```bash
--grasp_action_object_collision_start_stage close_gripper
```

关于 preview-top1 路径下的阶段顺序，以及“物体是否附着到夹爪”的关系，见：
- [V1.14_preview_top1_execution_relation_ZH.md](/home/zaijia001/ssd/RoboTwin/agent-read/V1.14_preview_top1_execution_relation_ZH.md)

完整参数说明也整理在：
- [V1.14_preview_top1_execution_relation_ZH.md](/home/zaijia001/ssd/RoboTwin/agent-read/V1.14_preview_top1_execution_relation_ZH.md)
- 里面包含：
  - 完整执行命令模板
  - viewer 版命令
  - 候选来源、关键帧、平滑、执行模式、物体 replay 等参数说明

### 直接使用 hand keyframes JSON 指定的两个关键帧

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_from_annotated_keyframes \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps 30 \
  --approach_offset_m 0.08 \
  --left_target_object cup \
  --right_target_object bottle
```

这条命令的含义：

- preview 输出来自 `anygrasp_direct_preview_keyframes_batch`
- 每个视频的 `summary.json` 里已经有三帧：
  - `frame 0`
  - 标注关键帧1
  - 标注关键帧2
- planner 会忽略 `frame 0` 的执行作用，只把它当 preview 上下文
- 真正执行的两帧改为：
  - `frame_selection.annotated_keyframes[0]` 作为 `grasp`
  - `frame_selection.annotated_keyframes[1]` 作为 `action`
- 路径保持不变：
  - `init -> pregrasp -> grasp -> close_gripper -> keyframe_2`
- 如果你想让主输出视频只有真实场景，再加：
  - `--pure_scene_output 1`
- 如果你想限制每个 stage 最多尝试 20 次，超过后把该视频记为失败，再加：
- 如果你想限制每个 stage 最多尝试 20 次，超过后继续下一阶段，但把未达标 stage 和最终误差记进 JSON，再加：
  - `--replan_until_reached 1`
  - `--replan_until_reached_max_attempts 20`
  - batch 结束后会在输出根目录写：
    - `batch_plan_summary.json`
    - `batch_failed_ids.json`

### 先跑 `id=1` demo

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes \
  --ids 1 \
  --keyframes 1 22 \
  --lighting_mode front_no_shadow \
  --planner_backend urdfik \
  --save_debug_preview 1 \
  --debug_preview_fps 10 \
  --debug_keyframe_hold_frames 20 \
  --execute_interp_steps 40 \
  --settle_steps 10 \
  --hold_frames_after_stage 6 \
  --reach_pos_tol_m 0.02 \
  --reach_rot_tol_deg 12 \
  --max_stage_replans 5
```

### batch 跑所有视频

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes \
  --keyframes 1 22 \
  --lighting_mode front_no_shadow \
  --planner_backend urdfik
```

### batch 跑所有视频，stage 超过 20 次仍未到位时继续执行并记录失败 id

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_from_annotated_keyframes \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps 30 \
  --approach_offset_m 0.08 \
  --candidate_target_local_x_offset_m 0.0 \
  --left_target_object cup \
  --right_target_object bottle \
  --arm auto \
  --execute_both_arms 1 \
  --replay_objects_during_action 0 \
  --save_debug_preview 1 \
  --save_debug_execution_preview 1 \
  --enable_viewer 1 \
  --reach_pos_tol_m 0.2 \
  --reach_rot_tol_deg 20 \
  --replan_until_reached 1 \
  --replan_until_reached_max_attempts 20
```

行为说明：
- 单个视频里，只要任一执行 arm 的 `pregrasp / grasp / action` 在 20 次内仍未 `reached`
- 单视频脚本会写出 `plan_summary.json`
- 并打印 `[warn] ...`
- 但不会中断后续 stage，也不会因为这个原因中止单视频进程
- `plan_summary.json` 会额外记录：
  - `execution_failed`
  - `execution_success`
  - `failed_stage_records`
- batch 会把失败视频汇总到：
  - `batch_plan_summary.json`
  - `batch_failed_ids.json`

### 打开 viewer 调试

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes \
  --ids 1 \
  --keyframes 1 22 \
  --lighting_mode front_no_shadow \
  --planner_backend urdfik \
  --save_debug_preview 1 \
  --debug_preview_fps 10 \
  --debug_keyframe_hold_frames 20 \
  --execute_interp_steps 40 \
  --settle_steps 10 \
  --hold_frames_after_stage 6 \
  --reach_pos_tol_m 0.02 \
  --reach_rot_tol_deg 12 \
  --max_stage_replans 5 \
  --enable_viewer 1 \
  --viewer_frame_delay 0.02 \
  --viewer_wait_at_end 1
```


## 输出

每个视频目录下会生成：

- `debug_selection_preview.mp4`
- `debug_execution_preview.mp4`
- `head_cam_plan.mp4`
- `plan_summary.json`

如果开启了：

- `--save_pose_debug 1`
- 或 `--pure_scene_output 1`

还会额外生成：

- `pose_debug.jsonl`

它记录每一帧的：

- 左右臂 joint qpos
- gripper command / gripper joint qpos
- 头相机 / TCP / EE pose
- 当前执行物体 actor pose

这个文件可以被新的后处理平滑重放脚本直接复用：

- `code_painting/replay_pose_debug_smooth.py`
- `code_painting/run_replay_pose_debug_smooth.sh`

batch 根目录下会生成：

- `batch_plan_summary.json`


## Debug Preview

`debug_selection_preview.mp4` 用来调试关键帧选择逻辑。

它会做三件事：

1. 按 `multi_object_world_poses.npz` 里的轨迹，把该视频里的物体完整回放
2. 持续显示关键帧 `1` 和 `22` 的两个不同颜色目标轴
3. 在命中关键帧窗口时，额外显示该关键帧 top-k 候选的简化夹爪预览
4. 在画面左上角写出：
   - `source_frame`
   - `selected_arm`
   - `selected_object`
   - `candidate_idx`
   - `rot_err`

颜色约定：

- 绿色：所有候选
- 蓝色：按左右手各自排序后保留下来的 top-k 候选
- 红色：最终被选中的候选

左右手区分：

- 左右手和最终选中改用颜色区分：

- 绿色：原始候选里按分数排序后保留下来的少量候选
- 蓝色：左手 top 候选
- 橙色：右手 top 候选
- 红色：最终选中的候选
- 每个候选都会在图像里投影出更小的纯数字 `candidate_idx`
- 数字不带背景，也不显示 `L/R` 前缀
- 红色选中候选会画得更大一些，方便直接定位

`debug_execution_preview.mp4` 用来调试慢速执行。

它会在机器人执行 `pregrasp -> grasp -> action` 的过程中持续保留：

- 当前阶段对应关键帧的候选抓取
- 关键帧 `1` 和 `22` 的目标轴
- 左上角的阶段信息
- 左右手和最终候选的颜色区分以及小号编号

相关参数：

- `--save_debug_preview`
  - `1` 保存 debug 预览视频
  - `0` 不保存
- `--debug_preview_fps`
  - debug 预览视频帧率
- `--save_debug_execution_preview`
  - 是否保存慢速执行的 debug 视频
- `--debug_execution_fps`
  - 第三个 debug 执行视频帧率
- `--debug_keyframe_hold_frames`
  - 关键帧命中的抓取坐标轴在 debug 预览里持续显示的帧数
- `--debug_candidate_top_k`
  - 左手和右手各自最多显示多少个 top 候选
  - 只想看 top1 时设成 `1`
- `--debug_common_candidate_top_k`
  - 绿色原始候选最多显示多少个
  - `0` 表示不显示绿色原始候选
  - 推荐调试值是 `0`
- `--candidate_orientation_remap_label`
  - AnyGrasp 候选姿态的固定轴重映射标签
- `--candidate_post_rot_xyz_deg`
  - AnyGrasp 候选姿态在相机系里的固定后置欧拉角旋转
- `--debug_target_axis_length`
  - 抓取位姿坐标轴长度
- `--debug_target_axis_thickness`
  - 抓取位姿坐标轴粗细


## 参数说明

### 输入相关

- `--anygrasp_root`
  - AnyGrasp 批处理结果根目录
- `--replay_root`
  - 物体 replay 根目录
- `--hand_dir`
  - `hand_detections_<id>.npz` 所在目录
- `--output_root`
  - 输出目录
- `--ids`
  - 只跑指定视频 id，例如 `--ids 1 4 7`
- `--keyframes`
  - 两个关键帧，默认 `1 22`

### 规划相关

- `--planner_backend`
  - `urdfik` 或 `curobo`
  - 当前推荐 `urdfik`
- `--arm`
  - `auto` / `left` / `right`
  - `auto` 默认配合双臂模式使用
- `--left_target_object`
  - 左手允许抓取的物体名，默认 `cup`
- `--right_target_object`
  - 右手允许抓取的物体名，默认 `bottle`
- `--execute_both_arms`
  - 当前默认 `1`
  - 当 `--arm auto` 时，默认双臂同步执行
  - 只有某一只手没有有效关键帧候选时，才自动回退到单臂
- `--candidate_object_max_distance_m`
  - 候选抓取点到目标物体中心的最大允许距离
- `--approach_offset_m`
  - `pregrasp` 相对 `grasp` 的后退距离
- `--open_gripper`
  - 抓取前 gripper 开度
- `--close_gripper`
  - 抓取时 gripper 开度

### 到位控制

- `--execute_interp_steps`
  - 执行前对 `plan["position"]` 再做一次 joint 轨迹插值
  - 如果 planner 产出的点数本来就少，它会明显增加中间 command waypoint 数
  - 如果 planner 已经有很多点，它只会轻度加密
- `--joint_command_scene_steps`
  - 每下发一个 joint waypoint 后，实际推进多少个 physics step
  - `head_cam_plan.mp4` 的主执行段通常是“每个 command waypoint 录 1 帧”，不是“每个 physics step 录 1 帧”
- `--settle_steps`
  - command 轨迹全部下发后，先额外推进的粗粒度稳定步数
- `--joint_target_wait_steps`
  - command 轨迹结束后，最多再给多少个 physics step，让机械臂继续收敛到最终 joint target
  - 这一步在 viewer 中仍然可见，但主视频不会把每一个 wait step 都逐帧录下来，只会在结束后再录 `plan_step=done`
  - 所以如果主要运动发生在这里，viewer 看起来会连续，`head_cam_plan.mp4` 可能看起来像突然跳到终点
- `--joint_target_wait_tol_rad`
  - 最终 joint 收敛等待的每关节误差阈值
- `--hold_frames_after_stage`
  - 每段结束后额外停留几帧，方便看视频
- `--reach_pos_tol_m`
  - 位置到位阈值
  - 当前默认 `0.03` 米
- `--reach_rot_tol_deg`
  - 姿态到位阈值
  - 当前默认 `20` 度
- `--max_stage_replans`
  - 旧模式下每个阶段最多重规划次数
- `--replan_until_reached`
  - 是否在 `reach=False` 时继续从当前位置重规划
  - 当前默认 `1`
- `--replan_until_reached_max_attempts`
  - 继续重规划的最大次数
  - 当前默认 `0`
  - `0` 或负数表示不限次数，直到真正到位

当前推荐行为：
- 保持 `--replan_until_reached 1`
- 若你希望同时保留纯净主视频和可视化 debug 视频，使用：
  - `--pure_scene_output 1`
  - `--save_debug_preview 1`
  - `--save_debug_execution_preview 1`

新的终端日志：
- 每次 `try` 结束后都会输出当前 target 误差
- 单臂格式：
  - `dx dy dz dist rot reached`
- 双臂格式：
  - 同一行分别打印 `left` 和 `right` 的 `dx dy dz dist rot reached`
- 单臂模式下如果存在监督手，还会额外打印 `attempt-supervision`

### 渲染相关

- `--lighting_mode`
  - `default` / `front` / `front_no_shadow`
- `--head_only`
  - `1` 只录 head 相机
- `--third_person_view`
  - `1` 时录制第三人称视频
- `--overlay_text`
  - 是否在视频左上角显示状态文字
- `--image_width`
- `--image_height`
- `--fps`
- `--fovy_deg`


## `plan_summary.json`

关键字段：

- `selected_arm`
- `expected_object_for_selected_arm`
- `selected_object`
- `arm_debugs`
  - 左右手各自的筛选诊断和最终是否被选中
- `selected_candidates` 里的 `candidate_idx`
  - 对应视频中红色候选旁边的编号
- `selected_candidates`
  - 每个关键帧最终选中的候选
- `top_candidates_per_keyframe`
  - 每个关键帧按规则排序后的前几个候选
- `all_candidates_per_keyframe`
  - 每个关键帧的全部候选
- `stages.pregrasp`
- `stages.grasp`
- `stages.action`

每个 stage 会包含：

- `status`
- `attempts`
- `reached`
- `pos_err_m`
- `rot_err_deg`


## 基于 `pose_debug.jsonl` 的平滑重放

如果你已经先跑完了一次 pure 模式或开启了 `--save_pose_debug 1`，可以再做一遍**后处理平滑重放**，把原来记录下来的离散执行帧补插值后重新导出更平滑的视频。

最常用命令：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_replay_pose_debug_smooth.sh \
  --plan_summary_json /path/to/d_pour_blue_35/plan_summary.json \
  --output_path /path/to/d_pour_blue_35/head_cam_plan_smooth.mp4 \
  --interp_factor 4 \
  --fps 20 \
  --overlay_text 0 \
  --base_occluder_enable 1 \
  --base_occluder_local_pos 0.0 0.0 0.4 \
  --base_occluder_half_size 0.45 0.45 0.02 \
  --base_occluder_color 1.0 1.0 1.0
```

说明：

- 它会自动从 `plan_summary.json` 找到：
  - `pose_debug.jsonl`
  - `replay_dir`
- 在相邻记录帧之间按 `--interp_factor` 做更密的插值
- 然后重新渲染为一个更平滑的 replay 视频

建议起步参数：

- `--interp_factor 4`
- `--fps 20`

重要限制：

- 这个脚本只改善**回放视频观感**
- 不会改变原始 run 的 `reached / pos_err / rot_err`
- 不会让那次真实执行本身变得更准确

也就是说：

- 真实执行质量，仍然要优先通过：
  - `joint_command_scene_steps`
  - `settle_steps`
  - `joint_target_wait_steps`
  去调
- `replay_pose_debug_smooth.py` 更适合做“跑完后导出更平滑展示视频”

## 当前实现边界

- 抓取后物体是通过 TCP 刚性附着，不是物理夹持
- 当前只把选中的主操作物体附着到 gripper，其他物体保持 replay 中的位置
- 如果关键帧候选本身不可达，到位检查会失败并触发重规划，但不会自动换下一个候选


## 建议的 debug 顺序

1. 先看 `debug_selection_preview.mp4`
   - 确认物体轨迹是否对
   - 确认关键帧 1 和 22 选中的抓取位姿是否合理
2. 再看 `head_cam_plan.mp4`
   - 确认 `pregrasp/grasp/action` 是否真的到位
3. 最后看 `plan_summary.json`
   - 检查 `reached`, `pos_err_m`, `rot_err_deg`


## Orientation Check

关于 AnyGrasp 朝向链路和 hand replay 朝向链路的对比结论，见：
- [README_anygrasp_orientation_check.md](/home/zaijia001/ssd/RoboTwin/code_painting/README_anygrasp_orientation_check.md)

### Viewer fallback and rank previews

- If the log shows `Renderer does not support display`, SAPIEN failed to open an interactive window in the current shell/VNC context and will automatically fall back to offscreen rendering only. The planner still runs and writes videos and PNGs.
- For manual candidate picking, the planner now writes `rank_previews/keyframe_<frame>_rank_<k>.png`. Each PNG shows the left-hand rank-`k` candidate in blue and the right-hand rank-`k` candidate in orange for that keyframe.
- The selected automatic candidate, if it matches that rank, is tagged with `selected` in the overlay text.
- Main controls: `--save_rank_preview_images 1` and `--rank_preview_top_n 3`.

### Manual candidate override

You can pin specific AnyGrasp candidates before orientation debugging with repeated `--manual_candidate FRAME ARM CANDIDATE_IDX` arguments.

Example:
```bash
--manual_candidate 1 left 5 \
--manual_candidate 1 right 11
```

If only part of the keyframes are specified, the override is used to reorder and highlight debug candidates. If both keyframes are specified for one arm, the planner will use those candidates directly for that arm.
