# 当前记忆

更新时间：2026-06-30

用途：记录当前会话和近期数据处理的高优先级上下文。这里不是完整命令库；完整可复现命令仍以 `COMMAND_LIBRARY.zh.md` 和 `agent-read/COMMANDS/` 为准。

## 当前主要数据版本

- 当前推荐的人手重绘训练数据标识是 `oursv2_piper0515_25ep`。
- 上传任务使用 tmux session：`upload_oursv2_piper0515_6task`。
- 本地 zip 路径：`/home/zaijia001/.cache/huggingface/lerobot/local/robot_oursv2_piper0515_6task_25ep.zip`。
- rclone 目标：`gdrive:piper/multi/6task/robot_oursv2_piper0515`。
- zip 内目录形如：`h2o_<TASK>_oursv2_piper0515_25ep`。

## 坐标系注意事项

- 不要直接把中间版本 `h2o_<TASK>_ours_rightcam_m003_color-120` 或 `h2o_<TASK>_ours_rightcam_m003_color_25ep` 当成最终真机一致数据使用。
- 真机 Piper 0515 坐标修正版应使用：`/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_ours_rightcam_m003_color_piper0515_25ep`。
- 该版本应包含 `meta/piper0515_world_to_base_conversion.json`，并按真机 world-to-base 约定转换 state/action，夹爪尺度使用 `0.0967`。

## L16 Stage-2 重绘状态

- 当前 right-cam m003 color Stage-2 输出根目录：`/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/stage2_color_rightcam_m003_full_0_120/e0_robot_object`。
- 主要脚本：`/home/zaijia001/ssd/RoboTwin/code_painting/repaint_l16_white_color_debug.py`。
- 当前推荐路线是颜色去白底，而不是 SAM 文字提示去白底；原因是 SAM 对白底/物体边界容易误删，颜色阈值更稳定。
- 帧对齐要参考机器人 replay 长度，避免向较短 Stage-1 背景长度截断。

## Mode N 消融实验

- Mode N 是 `COMMAND_LIBRARY.zh.md` 中 `N. 消融实验：Foundation Pose 物体位置 + 人手朝向`。
- 逻辑：使用 FoundationPose 物体世界位置 + 人手方向，不使用 AnyGrasp 候选排序。
- 关键帧逻辑：每只手最多两个有效关键帧；grasp 关键帧使用该帧 Foundation 物体位置和该帧人手朝向，action 关键帧使用第二帧 Foundation 物体位置。当前 N-7 通过 `--foundation_pose_action_orientation_source grasp` 保持 grasp 朝向，只更新 action 位置。
- 当前 debug 使用 N-7 参数：`--foundation_pose_retreat_m 0.10`、`--approach_offset_m 0.07`、`--foundation_pose_action_orientation_source grasp`、`--dual_stage_freeze_reached_arms_on_replan 1`、`--debug_viewer_overlay`。
- 本次 6 任务各 5 个 id 的输出根目录：`/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-7_foundation_pose_humanrot_keyframe_debug_6task5_20260630`。
- 重跑脚本：`/home/zaijia001/tmp/run_mode_n_n7_keyframe_debug_6task5_20260630.sh`。
- 重跑时会创建 tmux sessions：`mode_n_n7_kf_debug_pick_diverse_bottles`、`mode_n_n7_kf_debug_place_bread_basket`、`mode_n_n7_kf_debug_stack_cups`、`mode_n_n7_kf_debug_handover_bottle`、`mode_n_n7_kf_debug_pnp_bread`、`mode_n_n7_kf_debug_pnp_tray`。
- 本次 id：`pick_diverse_bottles 0-4`、`place_bread_basket 0-4`、`stack_cups 0-4`、`handover_bottle 1-5`、`pnp_bread 7-11`、`pnp_tray 0-4`。
- 每个结果优先查看：`foundation_input_<ID>/head_cam_plan.mp4`、`foundation_input_<ID>/debug_execution_preview.mp4`、`foundation_input_<ID>/plan_summary_foundation_pose.json`、`foundation_input_<ID>/pose_debug.jsonl`。


## Mode N no-axis selected25 pipeline

- 总控 tmux：`mode_n_n7_noaxes_selected25_pipeline`。
- 重跑脚本：`/home/zaijia001/tmp/run_mode_n_n7_noaxes_selected25_pipeline_20260630.sh`。
- Stage1 no-axis planner root：`/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-7_foundation_pose_humanrot_noaxes_selected25_20260630`。
- Stage2 repaint root：`/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/stage2_color_mode_n_n7_fpose_hrot_noaxes_selected25/e0_robot_object`。
- 数据后缀：`mode_n_n7_fpose_hrot_noaxes`；复用 `l16_ours_review_first25` 中同一批 25 个 id。
- 本地 zip：`/home/zaijia001/.cache/huggingface/lerobot/local/robot_mode_n_n7_fpose_hrot_noaxes_piper0515_6task_25ep.zip`。
- rclone 目标：`gdrive:piper/multi/6task/robot_mode_n_n7_fpose_hrot_noaxes_piper0515`；当前由脚本打印手动上传命令，不在本环境自动执行。

## 近期未解决或需要注意

- `make_l16_repaint_montage.py` 仍可能默认指向旧 `L16_human_replay_clean`；如果要复查 rightcam m003 的 wrist 视角，需要显式支持或传入新的 L16 root。
- 工作区已有用户/历史修改，提交时应只 stage 当前任务相关文件，避免混入 `COMMAND_LIBRARY.zh.md` 和已有 command log 修改。

## SKEYP 关键帧消融

- `skeyp` 使用方案 B：复用 `ours` 的 planner-output 转换格式，不创建 reinit 风格的 `world_targets_and_status.npz`。
- 总控脚本：`/home/zaijia001/ssd/RoboTwin/code_painting/run_skeyp_selected25_pipeline.sh`。
- tmux session：`skeyp_selected25_pipeline`。
- 复用选集：`/home/zaijia001/ssd/RoboTwin/code_painting/l16_ours_review_first25/selections/<TASK>/ours_review_selection.json`。
- Stage-1 输出：`/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_skeyp/stage1`，只去人手/手腕/手表/手臂，保留操作物体。
- Planner 输出：`/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/skeyp_selected25_rightcam_m003_20260708`。
- Stage-2 输出：`/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_skeyp_visible_reinit/e0_robot`，只 repaint 机器人，不重新贴物体。
- 最终数据：`/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_skeyp_piper0515_25ep`。
- 本地 zip：`/home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_piper0515_6task_25ep.zip`。
- 远端上传需要手动运行脚本打印的 `rclone copy ... gdrive:piper/multi/6task/robot_skeyp_piper0515` 命令。
