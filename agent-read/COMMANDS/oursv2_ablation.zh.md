# OursV2 数据量与 GraspNet 消融

## R：1 Robot + 49 Ours

目的：只改变 oursv2 的数据配比，不改变视频、planner、state/action 或 Piper0515 坐标转换逻辑。每个任务生成 49 条 ours，供训练集再加入 1 条真机 robot，形成每任务 50 条。

选择优先级固定为：y -> m -> 未标注 -> n。只有可用 planner/Stage-2 源总数仍不足 49 时，才循环重复最前面的 y。完整顺序和每条来源记录在：

~~~text
/home/zaijia001/ssd/RoboTwin/code_painting/l16_oursv2_review_49ep/selections/<TASK>/oursv2_49ep_selection_manifest.json
~~~

组成：

| task | y | m | 未标注 | n | 重复 y | unique | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| pick_diverse_bottles | 29 | 1 | 19 | 0 | 0 | 49 | 49 |
| place_bread_basket | 31 | 0 | 0 | 12 | 6 | 43 | 49 |
| handover_bottle | 29 | 10 | 0 | 8 | 2 | 47 | 49 |
| pnp_bread | 27 | 6 | 16 | 0 | 0 | 49 | 49 |
| pnp_tray | 26 | 4 | 19 | 0 | 0 | 49 | 49 |
| stack_cups | 7 | 30 | 0 | 4 | 8 | 41 | 49 |

运行入口：

~~~bash
tmux new-session -d -s oursv2_49ep_pipeline \
  'bash /home/zaijia001/ssd/RoboTwin/code_painting/run_oursv2_49ep_pipeline.sh'
~~~

输出：

~~~text
/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_oursv2_piper0515_49ep
/home/zaijia001/.cache/huggingface/lerobot/local/robot_oursv2_piper0515_6task_49ep.zip
gdrive:piper/multi/6task/robot_oursv2_piper0515_49ep
~~~

坐标复核：最终 repo 会同时转换 observation.state 与 action。Piper0515 base frame 中左右臂都是 +x=机械臂正前方、+y=机械臂左侧、+z=向上。当前 oursv2_piper0515_25ep 的 xyz 与真机同量级；旧 multi_piper_cartin_robot25_ours_human25_6tasks_300_repo 后 25 条仍是 world frame，不能用它判断当前修正版。

## S：GraspNet top-score 消融

目的：排查 ours 效果是否来自 AnyGrasp。复用 ours 的同一批 25 ID、关键帧、右腕相机、Stage-1/Stage-2、训练数据转换和 Piper0515 对齐；唯一主要变量是关键帧目标改为对应目标物体内 AnyGrasp 分数最高候选。

候选设置：

~~~text
candidate_selection_mode=top_score_auto
candidate_max_rotation_distance_deg=-1
candidate_keep_camera_up=0
~~~

top-score 不使用人手朝向阈值或人手旋转 tie-break；仍保留目标物体匹配和候选到物体的有效距离过滤。

运行入口：

~~~bash
tmux new-session -d -s graspnet_selected25_pipeline \
  'bash /home/zaijia001/ssd/RoboTwin/code_painting/run_graspnet_selected25_pipeline.sh'
~~~

中间与最终输出：

~~~text
planner:
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/S_graspnet_topscore_rightcam_m003_selected25

Stage-2:
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/stage2_color_graspnet_selected25/e0_robot_object

Piper0515:
/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_graspnet_piper0515_25ep

zip:
/home/zaijia001/.cache/huggingface/lerobot/local/robot_graspnet_piper0515_6task_25ep.zip

rclone:
gdrive:piper/multi/6task/robot_graspnet_piper0515
~~~
