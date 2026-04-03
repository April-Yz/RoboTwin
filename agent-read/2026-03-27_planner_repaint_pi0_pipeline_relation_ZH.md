# 2026-03-27：planner -> SAM repaint -> pi0 处理 的前后关系说明

## 1. 完整链路怎么理解（把当前 Step1~4 前面的步骤也补齐）

如果把你现在这一套流程从更上游开始完整串起来，可以按下面这条 **10 段链路** 理解。

其中：
- 前 6 段是当前 Step1~4 之前的准备链路
- 后 4 段就是你现在正在实际使用的 `planner -> smooth -> repaint -> pi0`

### Stage A：上游做人手检测 / HaMeR，生成 hand_vis 与 hand_detections

这一段不在 RoboTwin 里实现，但当前链路依赖它的输出。

你在 `usage.sh` 里记录的典型命令是：

```bash
python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py \
  --data_dir /home/zaijia001/ssd/data/R1/hand/gt_depth/d_pour_blue \
  --output_dir /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis
```

它的作用：
- 输入真人采集序列：
  - `/home/zaijia001/ssd/data/R1/hand/gt_depth/d_pour_blue`
- 输出手部检测与可视化：
  - `/home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis`

这一步的关键下游产物通常包括：
- `hand_vis_<id>.mp4`
- `hand_vis_gripper_<id>.mp4`
- `hand_detections_<id>.npz`

也就是说，后面所有“手关键帧、planner 关键帧选择、手 replay / retarget”都以这里的结果为起点。

### Stage B：上游做 FoundationPose，生成物体轨迹

这一步同样是上游步骤，不在 RoboTwin repo 内部实现，但 `usage.sh` 里记录了当前常用命令。

单物体示例：

```bash
CUDA_VISIBLE_DEVICES=1 python /home/zaijia001/FoundationPose/run_realr1_dino_sam_batch.py \
  --data_dir /home/zaijia001/ssd/data/R1/hand/gt_depth/d_pour_blue \
  --mesh_file /home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle.obj \
  --output_root /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/obj_vis \
  --prompt bottle \
  --save_video 1 \
  --save_mesh_overlay_video 1 \
  --save_bbox_overlay_video 1 \
  --mesh_overlay_alpha 0.45
```

多物体示例：

```bash
source /home/zaijia001/FoundationPose/source_foundationpose_env.sh
cd /home/zaijia001/FoundationPose

CUDA_VISIBLE_DEVICES=2 python run_realr1_dino_sam_batch.py \
  --data_dir /home/zaijia001/ssd/data/R1/hand/gt_depth/d_pour_blue \
  --output_root /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/obj_vis_multi \
  --object bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle.obj \
  --object cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/cup.obj \
  --save_video 1 \
  --save_mesh_overlay_video 1 \
  --save_bbox_overlay_video 1 \
  --mesh_overlay_alpha 0.45
```

这一步的关键下游产物是：
- `obj_vis/.../poses.npz`

后面的物体 replay 会消费这些 `poses.npz`。

### Stage C：在 RoboTwin 中 replay FoundationPose 物体，同时隐藏机器人并导出 AnyGrasp 用的 RGB-D

这一步就是你提到的：
- 借助 foundation 的结果
- 在 **不显示机器人** 的情况下
- 在 RoboTwin 中 replay 物体
- 并导出给 AnyGrasp 用的 `head_anygrasp_frames`

当前常用全参数命令是：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue_multi_clean/obj_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  5 \
  --lighting_mode front_no_shadow \
  --hide_robot 1 \
  --save_head_depth 1 \
  --save_anygrasp_frames 1 \
  --object bottle=/home/zaijia001/ssd/data/R1/hand/obj_mesh/bottle/bottle.obj \
  --object cup=/home/zaijia001/ssd/data/R1/hand/obj_mesh/blue_cup/blue_cup.obj
```

这一步的作用：
- 消费 FoundationPose 的 `poses.npz`
- 在 RoboTwin 中把物体重放到统一世界系
- 隐藏机器人，只保留物体与相机
- 导出 AnyGrasp 需要的 RGB / depth / camera 数据

关键输出目录：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot/d_pour_blue_<id>/
```

关键输出文件通常包括：
- `head_cam_replay.mp4`
- `head_depth_frames/`
- `head_anygrasp_frames/`
- `multi_object_world_poses.npz`

### Stage D：在 RoboTwin 中只 replay 人手逐帧 retarget 得到的 R1 末端执行器轨迹

这一步是你说的“只 replay 人手，逐帧 retarget 得到 R1 机器人 ee pose”。

`usage.sh` 里记录的代表性命令是：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz_urdfik.sh \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/hand_detections_1.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green \
  5 \
  --require_stored_gripper_pose 1 \
  --pose_source gripper \
  --orientation_remap_label swap_red_blue_keep_green \
  --stored_orientation_post_rot_xyz_deg 0 0 0 \
  --debug_force_orientation none \
  --enable_viewer 1 \
  --viewer_wait_at_end 1
```

无窗口批处理版本（这一类是历史上用于生成 retarget replay 的来源）：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz_urdfik_pool.sh \
  --variant clean \
  --workers 2,2,2,3,3,3 \
  --lighting_mode front_no_shadow \
  --smooth_mode 1 \
  --smooth_interp_frames 1
```

这一步的关键意义：
- 它给了你一条“人手 -> R1 ee pose / wrist / zed replay”的参考支线
- 历史上旧版 repaint/pi0 流程大量使用过这条支线
- 但当前推荐主线已经切换为 planner 同源数据，不再把它当最终 Step4 的状态来源

### Stage E：把 Stage C 产出的无机器人物体 replay 包送到 Luka 服务器，在 AnyGrasp 上生成抓取候选

你在 `usage.sh` 里记录的流程是：

本机侧先打包：

```bash
cd /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot
tar -czvf blue_pour_norobot.tar.gz replay_m_obj_pose_d_pour_blue_norobot
```

Luka 服务器侧解包并运行 AnyGrasp：

```bash
cd /home/luka/yzj/
tar -xzvf blue_pour_norobot.tar.gz

cd /home/luka/yzj/anygrasp_sdk/grasp_detection
conda activate a1_anygrasp
python run_robotwin_replay_batch.py \
  --input_root /home/luka/yzj/replay_m_obj_pose_d_pour_blue_norobot \
  --checkpoint_path log/checkpoint_detection.tar

tar -czvf blue_pour_bg_anygrasp_norobot.tar.gz anygrasp_batch_results
```

再把 AnyGrasp 结果带回本机后，解包到 RoboTwin 侧使用：

```bash
cd /home/zaijia001/ssd/RoboTwin/code_painting
tar -xzvf blue_pour_bg_anygrasp_norobot.tar.gz
```

最终你在本机上会得到：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results/d_pour_blue_<id>/
  grasps/grasp_000000.json
  grasps/grasp_000001.json
  ...
```

### Stage F：先可视化 AnyGrasp top 候选，再对 hand_vis 标关键帧

你提到这一段顺序上包含两件事：
1. 先可视化 top 候选（你说过常看 top3）
2. 再标注 `hand_vis` 的关键帧

这两个动作本质上都是在为后面的 planner 选关键帧 / 选 candidate 做准备。

#### F1. 先看 AnyGrasp top3 候选预览

推荐单样本命令：

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/render_anygrasp_ranked_preview.py \
  --anygrasp_dir /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results/d_pour_blue_1 \
  --replay_dir /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot/d_pour_blue_1 \
  --hand_npz /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/hand_detections_1.npz \
  --base_image_dir /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot/d_pour_blue_1/head_anygrasp_frames \
  --base_image_mode raw \
  --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_top3/d_pour_blue_1 \
  --frames 1 22 -10 \
  --top_k 3 \
  --left_target_object cup \
  --right_target_object bottle \
  --draw_grasp_boxes 1
```

如果你已经做完关键帧标注，想按 `frame 0 + 标注关键帧` 批量预览，也可以：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch_top3 \
  --top_k 3 \
  --left_target_object cup \
  --right_target_object bottle \
  --draw_grasp_boxes 1
```

#### F2. 标注 hand_vis 关键帧

你提到的脚本就是：

```bash
python3 /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/annotate_hand_keyframes.py
```

如果要调播放速度：

```bash
python3 /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/annotate_hand_keyframes.py --delay-ms 150
```

关键输出：

```text
/home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/hand_keyframes_all.json
```

这个 JSON 后面会被 preview / planner 复用。

### Stage G：生成 planner 的 pure-v3 原始执行 bundle（也就是现在 smooth 之前那一步）

你特别提醒了这一点：
- `smooth` 的前一步，不是泛指 planner
- 而是你现在正在用的 **v3 planner 输出**

当前推荐的 full command 是：

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

到这里为止，才进入你当前正在使用的后四步。

## 2. 当前关心的主干怎么理解（保留原来的 Step1~4）

保留原来“planner -> repaint -> pi0”的主干结构，但在 Step1 和 Step2 之间插入一个 **smooth 中间步骤**。因此当前更推荐按连续四步理解：

### Step 1：planner / AnyGrasp 生成原始执行视频与状态
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

### Step 2：对 Step1 的 planner 结果做 smooth
新的推荐入口：
- `code_painting/batch_smooth_planner_outputs.sh`
- 核心脚本：
  - `code_painting/smooth_planner_outputs_from_pose_debug.py`

它读取 Step1 的：
- `head_cam_plan.mp4`
- `left_wrist_cam_plan.mp4`
- `right_wrist_cam_plan.mp4`
- `pose_debug.jsonl`
- `plan_summary.json`

但真正驱动 smooth 的主状态来源是：
- `pose_debug.jsonl`

它会做两类处理：
- **去掉徘徊/近重复帧**
- **对保留下来的关键状态做插值平滑**

然后输出新的 smooth bundle：
- `head_cam_plan.mp4`
- `left_wrist_cam_plan.mp4`
- `right_wrist_cam_plan.mp4`
- `pose_debug.jsonl`
- `smooth_summary.json`

输出目录：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth/d_pour_blue_<id>/
```

### Step 3：SAM / repaint 把 smooth 后的 planner head 合成到背景上
新的推荐入口：
- `inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad_from_smooth.sh`

它读取 Step2 的：
- `head_cam_plan.mp4`

然后根据 smooth 后 head 的 fps / duration 对原来的人手背景视频重新补长，再输出：
- `target_with_original_head_cam_plan.mp4`

输出目录：

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_smooth/id_<id>_head_cam_arm_gripper_cup_bottle_pad_target/
```

### Step 4：pi0 数据处理
新的推荐入口：
- `policy/pi0/scripts/process_repainted_planner_outputs.py`
- 包装脚本：
  - `policy/pi0/run_process_repainted_smoothed_planner_outputs.sh`

它读取：
- Step3 的 repaint head：
  - `target_with_original_head_cam_plan.mp4`
- Step2 smooth planner 目录里的 wrist：
  - `left_wrist_cam_plan.mp4`
  - `right_wrist_cam_plan.mp4`
- Step2 smooth planner 目录里的状态：
  - `pose_debug.jsonl`

然后转成：

```text
processed_data/<task>-<num>/episode_x/
  instructions.json
  episode_x.hdf5
```

所以这四个阶段的前后关系是：

```text
run_plan_anygrasp_keyframes_r1_batch.sh
    ↓
  生成原始 planner 的 head / wrist / pose_debug
    ↓
batch_smooth_planner_outputs.sh
    ↓
  去徘徊帧 + 插值平滑，生成 smooth planner bundle
    ↓
batch_head_cam_repaint_with_auto_pad_from_smooth.sh
    ↓
  对 smooth 后的 planner head 做 SAM/repaint，并按 smooth 后时长补齐背景
    ↓
process_repainted_planner_outputs.py
    ↓
  用 smooth repaint head + smooth planner wrist + smooth planner pose_debug 生成 pi0 HDF5
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

## 3. Step 1 命令：planner / AnyGrasp 批处理

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

## 4. Step 2 命令：对 Step1 的 planner 结果做 smooth

单条示例：

```bash
cd /home/zaijia001/ssd/RoboTwin
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

bash /home/zaijia001/ssd/RoboTwin/code_painting/batch_smooth_planner_outputs.sh 0
```

批处理示例（0~60 全部尝试，缺失样本自动 skip）：

```bash
cd /home/zaijia001/ssd/RoboTwin
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

bash /home/zaijia001/ssd/RoboTwin/code_painting/batch_smooth_planner_outputs.sh
```

如果你要显式写全所有关键参数，可以这样：

```bash
cd /home/zaijia001/ssd/RoboTwin
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

TASK_NAME=d_pour_blue \
INPUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3 \
OUTPUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth \
INTERP_FACTOR=2 \
FPS=10 \
KEEP_HOVER_FRAMES_EVERY=3 \
DEDUP_POS_THRESH_M=0.002 \
DEDUP_ROT_THRESH_DEG=1.5 \
DEDUP_JOINT_THRESH_RAD=0.01 \
DEDUP_GRIPPER_THRESH=0.01 \
OVERLAY_TEXT=0 \
DISABLE_TABLE=1 \
BASE_OCCLUDER_ENABLE=0 \
LIGHTING_MODE=front_no_shadow \
bash /home/zaijia001/ssd/RoboTwin/code_painting/batch_smooth_planner_outputs.sh 0
```

### Step 2 的关键输入

以 `id=0` 为例：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_0/
  head_cam_plan.mp4
  left_wrist_cam_plan.mp4
  right_wrist_cam_plan.mp4
  pose_debug.jsonl
  plan_summary.json
```

### Step 2 的关键输出

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth/d_pour_blue_0/
  head_cam_plan.mp4
  left_wrist_cam_plan.mp4
  right_wrist_cam_plan.mp4
  pose_debug.jsonl
  smooth_summary.json
```

### Step 2 的关键语义

- `KEEP_HOVER_FRAMES_EVERY`
  - 用于去掉徘徊/近重复帧
- `INTERP_FACTOR`
  - 在保留关键帧之间插值，平滑跳变
- `pose_debug.jsonl`
  - 是 smooth 之后的状态来源
- 输出目录里的 head / wrist / pose_debug 长度保持一致

## 5. Step 3 命令：SAM / repaint smooth 后的 planner head

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

单条示例：

```bash
cd /home/zaijia001/ssd/inpainting_sam2_robot
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate inpainting-sam2-r1

bash /home/zaijia001/ssd/inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad_from_smooth.sh 0
```

批处理示例（0~60 全部尝试，缺失样本自动 skip）：

```bash
cd /home/zaijia001/ssd/inpainting_sam2_robot
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate inpainting-sam2-r1

bash /home/zaijia001/ssd/inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad_from_smooth.sh
```

这就是 Step 3 的总控命令。

如果你要显式写全所有关键参数，可以这样：

```bash
cd /home/zaijia001/ssd/inpainting_sam2_robot
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate inpainting-sam2-r1

GPU=0 \
DEVICE=cuda \
TASK_NAME=d_pour_blue \
ROBOT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth \
STAGE1_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint \
OUTPUT_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_smooth \
FORCE_REPAD=0 \
ROBOT_VIDEO_NAME=head_cam_plan.mp4 \
DILATE_KERNEL_SIZE=8 \
ERODE_KERNEL_SIZE=14 \
TEXT_PROMPT='robotic manipulator arm, forearm, wrist, gripper, end effector, cup, bottle.' \
BOX_THRESHOLD=0.25 \
TEXT_THRESHOLD=0.22 \
MAX_MASK_AREA_RATIO=0.60 \
MAX_SELECTED_BOXES=0 \
ARM_SPLIT_RATIO=0.5 \
EXCLUDE_BOTTOM_RATIO=0.10 \
COMPOSITE_ERODE_KERNEL_SIZE=2 \
BLEND_ALPHA_SIGMA=1.8 \
SAM_MODEL_TYPE=vit_h \
SAM_CKPT=./pretrained_models/sam_vit_h_4b8939.pth \
LAMA_CONFIG=./lama/configs/prediction/default.yaml \
LAMA_CKPT=./pretrained_models/big-lama \
TRACKER_CKPT=vitb_384_mae_ce_32x4_ep300 \
VI_CKPT=./pretrained_models/sttn.pth \
MASK_IDX=2 \
bash /home/zaijia001/ssd/inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad_from_smooth.sh 0
```

### Step 3 的关键输入

- smooth planner head：
  ```text
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth/d_pour_blue_0/head_cam_plan.mp4
  ```
- 原始 stage1 背景：
  ```text
  /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/stage1_no-human-obj_d_pour_blue_0/removed_w_mask_rgb_0.mp4
  ```

### Step 3 的对齐方式

这一步与旧的 step2 不同，必须按 **smooth 后的 head** 的 fps 和 duration 对背景重新补长：

- 不再沿用原 planner head 的时长
- 也不沿用 hand-retarget 的时长
- 而是使用：
  - `ROBOT_ROOT/.../head_cam_plan.mp4`（smooth 之后）

这样 Step 3 输出的 repaint head，才会与 Step 2 smooth bundle 的 wrist / pose_debug 对齐。

### Step 3 的关键输出

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_smooth/id_0_head_cam_arm_gripper_cup_bottle_pad_target/
  target_with_original_head_cam_plan.mp4
```

## 6. Step 4 命令：pi0 处理，全部改用 smooth planner 同源数据

单条示例：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

python scripts/process_repainted_planner_outputs.py \
  d_pour_blue \
  "pour water" \
  27 \
  --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_smooth \
  --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' \
  --head-video-name target_with_original_head_cam_plan.mp4 \
  --planner-root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth \
  --planner-dir-template 'd_pour_blue_{id}' \
  --left-wrist-video-name left_wrist_cam_plan.mp4 \
  --right-wrist-video-name right_wrist_cam_plan.mp4 \
  --pose-debug-name pose_debug.jsonl \
  --review-json /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json \
  --review-mode strict \
  --ignore-ids \
  --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-27-planner-smooth
```

批处理示例（按 review-json 批量处理所有 `y` 样本）：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

TASK_NAME=d_pour_blue \
INSTRUCTION='pour water' \
EXPERT_DATA_NUM=27 \
HEAD_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_smooth \
PLANNER_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth \
REVIEW_JSON=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json \
REVIEW_MODE=strict \
OUTPUT_DIR=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-27-planner-smooth \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/run_process_repainted_smoothed_planner_outputs.sh
```

这就是 Step 4 的总控命令。

如果你想把 Step 2 + Step 3 + Step 4 串起来，并且只处理 review 里标记为 `y` 的样本，也可以直接用新的总控批处理脚本：

```bash
cd /home/zaijia001/ssd/RoboTwin
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh

TASK_NAME=d_pour_blue \
REVIEW_JSON=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json \
REVIEW_MODE=strict \
INSTRUCTION='pour water' \
EXPERT_DATA_NUM=auto \
PI0_OUTPUT_DIR=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-reviewed-y-smooth \
bash /home/zaijia001/ssd/RoboTwin/run_reviewed_smooth_repaint_pi0_pipeline.sh
```

上面这个脚本会自动：
- 从 `video_review.json` 里找出所有 `label=y` / `usable=true` 的 id
- 对这些 id 跑 Step 2 smooth
- 对这些 id 跑 Step 3 smooth repaint
- 最后用同一个 review-json 在 Step 4 生成 pi0 数据

如果你只想先看会选到哪些 id，不真正执行，可以这样 dry-run：

```bash
cd /home/zaijia001/ssd/RoboTwin
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh

TASK_NAME=d_pour_blue \
REVIEW_JSON=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json \
REVIEW_MODE=strict \
DRY_RUN=1 \
bash /home/zaijia001/ssd/RoboTwin/run_reviewed_smooth_repaint_pi0_pipeline.sh
```

如果你想只跑其中某几步，也可以：
- 只跑 Step 2 + Step 3，不跑 Step 4：`RUN_PI0=0`
- 跳过 Step 2，只重跑 Step 3 + Step 4：`RUN_SMOOTH=0`
- 只重跑 Step 4：`RUN_SMOOTH=0 RUN_REPAINT=0`

如果你只想用 Step 4 的包装脚本，也可以：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

TASK_NAME=d_pour_blue \
INSTRUCTION='pour water' \
EXPERT_DATA_NUM=27 \
HEAD_ROOT=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_smooth \
HEAD_DIR_TEMPLATE='id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' \
HEAD_VIDEO_NAME=target_with_original_head_cam_plan.mp4 \
PLANNER_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth \
PLANNER_DIR_TEMPLATE='d_pour_blue_{id}' \
LEFT_WRIST_VIDEO_NAME=left_wrist_cam_plan.mp4 \
RIGHT_WRIST_VIDEO_NAME=right_wrist_cam_plan.mp4 \
POSE_DEBUG_NAME=pose_debug.jsonl \
REVIEW_JSON=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json \
REVIEW_MODE=strict \
OUTPUT_DIR=/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-27-planner-smooth \
bash /home/zaijia001/ssd/RoboTwin/policy/pi0/run_process_repainted_smoothed_planner_outputs.sh
```

### 这一步各参数对应什么数据

- `--head-root`
  - Step 3 smooth repaint 结果根目录
- `--head-dir-template`
  - repaint episode 目录模板
- `--head-video-name`
  - repaint 后的 head 视频文件名
- `--planner-root`
  - Step 2 smooth planner bundle 根目录
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

#### head（smooth repaint 后）

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_smooth/id_0_head_cam_arm_gripper_cup_bottle_pad_target/target_with_original_head_cam_plan.mp4
```

#### smooth planner wrist

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth/d_pour_blue_0/left_wrist_cam_plan.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth/d_pour_blue_0/right_wrist_cam_plan.mp4
```

#### smooth planner state

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth/d_pour_blue_0/pose_debug.jsonl
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
  先生成原始 planner 的 head / wrist / pose_debug
batch_smooth_planner_outputs.sh
  再对 planner 输出去徘徊帧并插值平滑，得到 smooth bundle
batch_head_cam_repaint_with_auto_pad_from_smooth.sh
  然后对 smooth 后的 planner head 做 SAM/repaint，并按 smooth 后时长补齐背景
process_repainted_planner_outputs.py
  最后用 smooth repaint head + smooth planner wrist + smooth planner pose_debug 生成 pi0 训练数据
```

这样四个 step 的前后关系最清晰，也不容易再把 retarget 流和 planner 流混在一起。
