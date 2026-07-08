# SKEYP 关键帧消融流水线

## 目的

`skeyp` 是一个只使用人手关键帧生成机器人 replay 的消融版本。它复用 `ours` 的 planner-output 数据转换格式，不改变原有 `ours`、`reinit` 或 Mode N 脚本逻辑。

## 核心逻辑

- 选取 id：复用 `/home/zaijia001/ssd/RoboTwin/code_painting/l16_ours_review_first25/selections/<TASK>/ours_review_selection.json` 中每个任务的 25 条。
- Stage 1：只 inpaint 人手、手腕、手表、手臂，保留真实操作物体。
- Planner：使用 `run_plan_keyframes_human_replay_piper_d435.sh`，按人手关键帧插值生成 `head_cam_plan.mp4`、左右腕相机视频和 `pose_debug.jsonl`。
- Stage 2：只把机器人 mask/repaint 到 Stage-1 背景上，不重新贴物体。
- 数据转换：复用 `run_l16_ours_selected_pipeline.sh` 的 `process lerobot subset piper0515` 步骤，最终生成真机 Piper 0515 坐标对齐的 LeRobot 数据。

## 一键运行

```bash
tmux new-session -d -s skeyp_selected25_pipeline \
  'bash /home/zaijia001/ssd/RoboTwin/code_painting/run_skeyp_selected25_pipeline.sh'
```

## 监控

```bash
tmux attach -t skeyp_selected25_pipeline
tail -f /home/zaijia001/tmp/skeyp_selected25_rightcam_m003_20260708_logs/task_stack_cups.log
```

每个任务各自有一个日志：

```text
/home/zaijia001/tmp/skeyp_selected25_rightcam_m003_20260708_logs/task_<TASK>.log
```

## 输出位置

- Stage-1 hands-only 背景：`/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_skeyp/stage1/<TASK>/id_<ID>/stage1_human_inpaint/removed_w_mask_rgb_<ID>.mp4`
- 关键帧 replay：`/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/skeyp_selected25_rightcam_m003_20260708/<TASK>/foundation_input_<ID>/`
- Stage-2 robot-only repaint：`/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_skeyp_visible_reinit/e0_robot/<TASK>/id_<ID>_skeyp/final_repainted.mp4`
- 中间 HDF5：`/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_<TASK>_skeyp-120`
- 本地 LeRobot：`/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_skeyp_piper0515_25ep`
- 本地压缩包：`/home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_piper0515_6task_25ep.zip`

## 手动上传

```bash
rclone copy /home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_piper0515_6task_25ep.zip \
  gdrive:piper/multi/6task/robot_skeyp_piper0515 \
  -P --drive-chunk-size 64M --transfers 4
```

## 相关代码

- 总控入口：`code_painting/run_skeyp_selected25_pipeline.sh`
- 关键帧 planner：`code_painting/run_plan_keyframes_human_replay_piper_d435.sh`
- Planner 数据转换：`policy/pi0/scripts/process_repainted_planner_outputs.py`
- piper0515 坐标对齐：`code_painting/convert_lerobot_piper0515_world_to_base.py`

## 2026-07-08 运行结果

- tmux session：`skeyp_selected25_pipeline`，已结束。
- Stage-2 `final_repainted.mp4`：6 个任务均为 25/25。
- 本地 zip：`/home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_piper0515_6task_25ep.zip`，约 191 MB。
- zip 校验：150 个 parquet，6 个 `piper0515_world_to_base_conversion.json`。
- 每个 `h2o_<TASK>_skeyp_piper0515_25ep` repo 均为 25 个 parquet，并包含 piper0515 坐标转换标记。

## SKEYP v2: reinit gripper-only

### 目的

`skeyp_reinit_gripperonly` 是为了和 reinit/pinpointing 链路对齐的第二版消融。它不走 planner-output，不贴回 object/Foundation replay；真实物体保留在 Stage-1 背景里，Stage-2 只把 reinit 风格 replay 里的夹爪/末端执行器贴回去。

### 核心区别

- v1：`plan_keyframes_human_replay.py` 输出 `pose_debug.jsonl`、`head_cam_plan.mp4`，再走 `process_repainted_planner_outputs.py`。
- v2：复用 reinit 风格 `h2_pure_d435` 的 `world_targets_and_status.npz`、`zed_replay_d435.mp4` 和左右腕相机视频，再走 `process_repainted_headcam_with_wrist.py`。
- v2 的 Stage-1 prompt 仍为 `arms, hands, wrists, watch.`，保留真实操作物体。
- v2 的 Stage-2 prompt 默认为 `robotic gripper, gripper fingers, end effector, robot hand.`，目标是只检测并贴回夹爪区域。
- 当前实现复用已有 `h2_pure_d435` 轨迹，因此是 reinit-compatible gripper-only 版本；如果以后要严格做到“只从关键帧插值生成 `world_targets_and_status.npz`”，需要新增一个 reinit-compatible keyframe 轨迹生成器。

### 一键运行

```bash
tmux new-session -d -s skeyp_v2_reinit_gripperonly_pipeline \
  'bash /home/zaijia001/ssd/RoboTwin/code_painting/run_skeyp_v2_reinit_gripperonly_pipeline.sh'
```

### 监控

```bash
tmux attach -t skeyp_v2_reinit_gripperonly_pipeline
tail -f /home/zaijia001/tmp/skeyp_v2_reinit_gripperonly_20260708_logs/task_stack_cups.log
```

每个任务有独立日志：

```text
/home/zaijia001/tmp/skeyp_v2_reinit_gripperonly_20260708_logs/task_<TASK>.log
```

### 输出位置

- Stage-1 hands-only 背景：`/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_skeyp/v2_reinit_gripperonly/stage1/<TASK>/id_<ID>/stage1_human_inpaint/removed_w_mask_rgb_<ID>.mp4`
- reinit 风格轨迹软链接：`/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/skeyp_v2_reinit_gripperonly/h2_pure_d435_selected25/<TASK>/id<ID>_d435_z005/`
- Stage-2 gripper-only repaint：`/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_skeyp_visible_reinit/v2_reinit_gripperonly/e0_gripper/<TASK>/id_<ID>_skeyp_gripper/final_repainted.mp4`
- Stage-2 调试视频：同目录下的 `w_box_zed_replay_d435.mp4`、`w_mask_zed_replay_d435.mp4`、`mask_zed_replay_d435.mp4` 和 `visible_reinit_meta.json`
- 中间 HDF5：`/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_<TASK>_skeyp_reinit_gripperonly-120`
- 本地 LeRobot：`/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_skeyp_reinit_gripperonly_piper0515_25ep`
- 本地压缩包：`/home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_reinit_gripperonly_piper0515_6task_25ep.zip`

### 手动上传

```bash
rclone copy /home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_reinit_gripperonly_piper0515_6task_25ep.zip \
  gdrive:piper/multi/6task/robot_skeyp_reinit_gripperonly_piper0515 \
  -P --drive-chunk-size 64M --transfers 4
```

### 相关代码

- 总控入口：`code_painting/run_skeyp_v2_reinit_gripperonly_pipeline.sh`
- Stage-1 hands-only inpainting：`/home/zaijia001/ssd/inpainting_sam2_robot/remove_anything_video_sam2.py`
- Stage-2 gripper-only repaint：`/home/zaijia001/ssd/inpainting_sam3_robot/remove_anything_video_sam3_robot_visible_reinit.py`
- reinit/D435 数据转换：`policy/pi0/scripts/process_repainted_headcam_with_wrist.py`
- piper0515 坐标对齐：`code_painting/convert_lerobot_piper0515_world_to_base.py`

### 2026-07-08 v2 运行结果

- tmux session：`skeyp_v2_reinit_gripperonly_pipeline`，已结束。
- Stage-2 `final_repainted.mp4`：6 个任务均为 25/25。
- 中间 HDF5：6 个任务均为 25/25。
- Piper0515 LeRobot repo：6 个任务均为 25 个 parquet，并包含 `meta/piper0515_world_to_base_conversion.json`。
- 本地 zip：`/home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_reinit_gripperonly_piper0515_6task_25ep.zip`，约 130 MB。
- zip 校验：150 个 parquet，6 个 `piper0515_world_to_base_conversion.json`。
