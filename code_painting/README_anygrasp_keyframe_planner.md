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
- `debug_execution_preview.mp4`
- `head_cam_plan.mp4`
- `plan_summary.json`

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
  - `auto` 会自动在左右手里选更接近人手朝向的一侧
- `--left_target_object`
  - 左手允许抓取的物体名，默认 `cup`
- `--right_target_object`
  - 右手允许抓取的物体名，默认 `bottle`
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
