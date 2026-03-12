# AnyGrasp Keyframe Planner

这套脚本用于把 AnyGrasp 候选抓取结果和手部 `hand_detections_*.npz` 结合起来，在 RoboTwin 里生成一个两关键帧的规划 demo。

核心流程：

1. 从 AnyGrasp 结果中读取关键帧 `1` 和 `22` 的所有抓取候选
2. 读取人手 `npz` 中对应帧的 `left/right_gripper_rotation_matrix`
3. 选择和人手 gripper 朝向最接近的候选
4. 要求两个关键帧都落在同一个物体上
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
- `head_cam_plan.mp4`
- `plan_summary.json`

batch 根目录下会生成：

- `batch_plan_summary.json`


## Debug Preview

`debug_selection_preview.mp4` 用来调试关键帧选择逻辑。

它会做三件事：

1. 按 `multi_object_world_poses.npz` 里的轨迹，把该视频里的物体完整回放
2. 在关键帧 `1` 和 `22` 上显示被选中的抓取位姿坐标轴
3. 在画面左上角写出：
   - `source_frame`
   - `selected_arm`
   - `selected_object`
   - `candidate_idx`
   - `rot_err`

相关参数：

- `--save_debug_preview`
  - `1` 保存 debug 预览视频
  - `0` 不保存
- `--debug_preview_fps`
  - debug 预览视频帧率
- `--debug_keyframe_hold_frames`
  - 关键帧命中的抓取坐标轴在 debug 预览里持续显示的帧数
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
  - `auto` 会自动在左右手里选更接近人手朝向的一侧
- `--approach_offset_m`
  - `pregrasp` 相对 `grasp` 的后退距离
- `--open_gripper`
  - 抓取前 gripper 开度
- `--close_gripper`
  - 抓取时 gripper 开度

### 到位控制

- `--execute_interp_steps`
  - 执行时的轨迹插值步数，越大动作越慢
- `--settle_steps`
  - 每段动作执行后额外稳定的仿真步数
- `--hold_frames_after_stage`
  - 每段结束后额外停留几帧，方便看视频
- `--reach_pos_tol_m`
  - 位置到位阈值
- `--reach_rot_tol_deg`
  - 姿态到位阈值
- `--max_stage_replans`
  - 每个阶段最多重规划次数

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
- `selected_object`
- `selected_candidates`
  - 每个关键帧最终选中的候选
- `stages.pregrasp`
- `stages.grasp`
- `stages.action`

每个 stage 会包含：

- `status`
- `attempts`
- `reached`
- `pos_err_m`
- `rot_err_deg`


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
