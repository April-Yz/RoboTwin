# 2026-03-27：planner -> SAM repaint -> pi0 处理 的前后关系说明

## 1. 这条链路现在推荐怎么理解

当前更推荐把这条链路理解成连续三步：

### 第一步：planner / AnyGrasp 生成执行视频与状态
入口命令：
- `code_painting/run_plan_anygrasp_keyframes_r1_batch.sh`

它输出：
- `head_cam_plan.mp4`
- `left_wrist_cam_plan.mp4`
- `right_wrist_cam_plan.mp4`
- `pose_debug.jsonl`
- `plan_summary.json`

这些文件都在：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_<id>/
```

### 第二步：SAM / repaint 把 planner head 合成到背景上
入口命令：
- `inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad.sh`

它读取第一步的：
- `head_cam_plan.mp4`

然后输出：
- `target_with_original_head_cam_plan.mp4`

输出目录：

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/id_<id>_head_cam_arm_gripper_cup_bottle_pad_target/
```

### 第三步：pi0 数据处理
新的推荐入口：
- `policy/pi0/scripts/process_repainted_planner_outputs.py`

它读取：
- 第二步的 repaint head：
  - `target_with_original_head_cam_plan.mp4`
- 第一步 planner 目录里的 wrist：
  - `left_wrist_cam_plan.mp4`
  - `right_wrist_cam_plan.mp4`
- 第一步 planner 目录里的状态：
  - `pose_debug.jsonl`

然后转成：

```text
processed_data/<task>-<num>/episode_x/
  instructions.json
  episode_x.hdf5
```

所以这三个阶段的前后关系是：

```text
run_plan_anygrasp_keyframes_r1_batch.sh
    ↓
  生成 planner 的 head / wrist / pose_debug
    ↓
batch_head_cam_repaint_with_auto_pad.sh
    ↓
  把 planner 的 head 做 SAM/repaint，生成 target_with_original_head_cam_plan.mp4
    ↓
process_repainted_planner_outputs.py
    ↓
  用 repaint 后的 head + planner wrist + planner pose_debug 生成 pi0 HDF5
```

---

## 2. 为什么现在推荐用这个新脚本

之前的问题在于：
- head 用的是 planner 的 `head_cam_plan.mp4`
- wrist / pose 用的是 hand-retarget 目录下的 `left/right_wrist_replay.mp4` + `world_targets_and_status.npz`

这会混用两套不同来源：
- planner 流
- hand-retarget 流

从而造成：
- head 帧数通常是 `255~263`
- wrist / pose 通常只有 `35~60`
- 最终处理长度被 wrist / pose 卡住

新脚本 `process_repainted_planner_outputs.py` 改成：
- head：仍然用 repaint 后的 planner head
- wrist：改用 planner 输出的 wrist cam
- state：改用 planner 输出的 `pose_debug.jsonl`

这样三路数据都来自同一条 planner 执行链路，长度是一致的。

---

## 3. 第一步命令：planner / AnyGrasp 批处理

下面是一个完整示例，等价于你当前常用链路中的 planner 阶段。

```bash
cd /home/zaijia001/ssd/RoboTwin
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3 \
  --ids 0 \
  --skip_existing 0 \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps 20 \
  --urdfik_cartesian_interp_auto_step_m 0.05 \
  --candidate_selection_mode planner \
  --left_target_object cup \
  --right_target_object bottle \
  --candidate_target_local_x_offset_m -0.05 \
  --approach_offset_m 0.20 \
  --reach_error_pose_source tcp \
  --replan_until_reached 1 \
  --replan_until_reached_max_attempts 3 \
  --save_debug_preview 1 \
  --save_debug_execution_preview 1 \
  --reach_pos_tol_m 0.03 \
  --reach_rot_tol_deg 20 \
  --enable_grasp_action_object_collision 1 \
  --grasp_action_object_collision_start_stage pregrasp \
  --execution_object_collision_mode convex \
  --execution_object_visual_scale_override cup=0.7 \
  --execution_object_collision_scale_override cup=0.3 \
  --execution_object_visual_scale_override bottle=0.7 \
  --execution_object_collision_scale_override bottle=0.3 \
  --debug_visualize_targets 1 \
  --debug_visualize_ik_waypoints 1 \
  --debug_collision_report 1 \
  --debug_visualize_object_collision_bbox 1 \
  --gripper_contact_monitor_mode all_robot_links \
  --pure_scene_output 1 \
  --overlay_text 0 \
  --viewer_show_camera_frustums 0 \
  --viewer_frame_delay 0.02 \
  --viewer_wait_at_end 1 \
  --settle_steps 10 \
  --joint_target_wait_steps 70 \
  --disable_table 1 \
  --base_occluder_enable 0 \
  --enable_viewer 1
```

### 这一步的关键输入

- `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results`
  - AnyGrasp 候选抓取结果
- `/home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot`
  - 物体 replay 轨迹
- `/home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis`
  - 手 keyframes / hand 数据
- `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch`
  - preview 关键帧和候选选择结果

### 这一步的关键输出

以 `id=0` 为例：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_0/
  head_cam_plan.mp4
  left_wrist_cam_plan.mp4
  right_wrist_cam_plan.mp4
  pose_debug.jsonl
  plan_summary.json
```

其中：
- `head_cam_plan.mp4`：后续 SAM/repaint 的输入 head
- `left/right_wrist_cam_plan.mp4`：后续 pi0 处理使用的 wrist
- `pose_debug.jsonl`：后续 pi0 处理使用的状态序列

---

## 4. 第二步命令：SAM / repaint planner head

```bash
cd /home/zaijia001/ssd/inpainting_sam2_robot
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate inpainting-sam2-r1

bash /home/zaijia001/ssd/inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad.sh 0
```

### 这一步的关键输入

默认从脚本里读取：

- planner head：
  ```text
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_0/head_cam_plan.mp4
  ```
- stage1 背景：
  ```text
  /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/stage1_no-human-obj_d_pour_blue_0/removed_w_mask_rgb_0.mp4
  ```

### 这一步的关键输出

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/id_0_head_cam_arm_gripper_cup_bottle_pad_target/
  target_with_original_head_cam_plan.mp4
  w_mask_head_cam_plan.mp4
  w_box_head_cam_plan.mp4
  mask_head_cam_plan.mp4
```

其中：
- `target_with_original_head_cam_plan.mp4`
  - 是后续 pi0 处理使用的 head 视频

---

## 5. 第三步命令：pi0 处理，全部改用 planner 同源数据

新的推荐命令：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

python scripts/process_repainted_planner_outputs.py \
  d_pour_blue \
  "pour water" \
  27 \
  --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue \
  --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' \
  --head-video-name target_with_original_head_cam_plan.mp4 \
  --planner-root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3 \
  --planner-dir-template 'd_pour_blue_{id}' \
  --left-wrist-video-name left_wrist_cam_plan.mp4 \
  --right-wrist-video-name right_wrist_cam_plan.mp4 \
  --pose-debug-name pose_debug.jsonl \
  --review-json /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json \
  --review-mode strict \
  --ignore-ids \
  --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-27-planner
```

### 这一步各参数对应什么数据

- `--head-root`
  - repaint 结果根目录
- `--head-dir-template`
  - repaint episode 目录模板
- `--head-video-name`
  - repaint 后的 head 视频文件名
- `--planner-root`
  - 第一步 planner 输出根目录
- `--planner-dir-template`
  - planner episode 目录模板
- `--left-wrist-video-name`
  - planner 左 wrist 视频
- `--right-wrist-video-name`
  - planner 右 wrist 视频
- `--pose-debug-name`
  - planner 状态日志
- `--review-json`
  - 人工筛选结果，只处理 `y` 的视频
- `--review-mode strict`
  - 严格模式，只处理 `y`
- `--output-dir`
  - 最终 pi0 HDF5 输出目录

### 这一步实际读取的文件

以 `id=0` 为例：

#### head（repaint 后）

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/id_0_head_cam_arm_gripper_cup_bottle_pad_target/target_with_original_head_cam_plan.mp4
```

#### planner wrist

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_0/left_wrist_cam_plan.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_0/right_wrist_cam_plan.mp4
```

#### planner state

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_0/pose_debug.jsonl
```

---

## 6. 新脚本到底怎么从 planner 目录里取 state

新脚本：
- `policy/pi0/scripts/process_repainted_planner_outputs.py`

状态来自 `pose_debug.jsonl` 的这些字段：

- `current_left_tcp_pose_world_wxyz`
- `current_right_tcp_pose_world_wxyz`
- `current_left_gripper_command`
- `current_right_gripper_command`

每一帧会构成 14 维状态：

- 左臂：
  - `x y z roll pitch yaw gripper`
- 右臂：
  - `x y z roll pitch yaw gripper`

也就是说，新的 state 已经不再来自：
- `world_targets_and_status.npz`

而改成来自：
- planner 的 `pose_debug.jsonl`

---

## 7. 为什么这个新方案更合理

因为现在三路数据终于是同源的：

- head：planner head 做完 repaint
- left wrist：planner left wrist
- right wrist：planner right wrist
- state：planner pose_debug

它们都来自同一条 planner 执行记录。

实际抽样已经验证：
- `head_cam_plan.mp4`
- `left_wrist_cam_plan.mp4`
- `right_wrist_cam_plan.mp4`
- `pose_debug.jsonl`

在 planner 目录下帧数一致，例如 `id=0`：

```text
head_cam_plan.mp4         263 frames
left_wrist_cam_plan.mp4   263 frames
right_wrist_cam_plan.mp4  263 frames
pose_debug.jsonl          263 lines
```

所以新的脚本输出长度会与 planner 执行长度一致，而不会再被 hand-retarget wrist 的低帧数卡住。

---

## 8. 推荐理解方式

如果你后面要迁移服务器或者向别人说明流程，建议直接用这句话概括：

```text
run_plan_anygrasp_keyframes_r1_batch.sh
  先生成 planner 的 head / wrist / pose_debug
batch_head_cam_repaint_with_auto_pad.sh
  再把 planner 的 head 做 SAM/repaint
process_repainted_planner_outputs.py
  最后用 repaint 后的 head + planner wrist + planner pose_debug 生成 pi0 训练数据
```

这样前后关系最清晰，也最不容易再把 retarget 流和 planner 流混在一起。
