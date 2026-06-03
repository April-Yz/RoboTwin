## 2026-06-02 00:00:00 +08

- 修复 Mode O viewer 环境传递：
  - `run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh` 在 `--viewer` 时传 `--gpu -1`。
  - `plan_first_frame_foundation_pick_diverse_bottles.py` 在 `--enable_viewer 1` 时移除 `CUDA_VISIBLE_DEVICES`，避免覆盖 wrapper 的 unset。
  - 新增/更新 O 节 viewer 探针说明。
- 验证：
  - `DISPLAY=:1.0 ... --viewer --viewer_wait_at_end 0 ...` 成功创建 SAPIEN viewer，日志显示 `CUDA_VISIBLE_DEVICES=None`。
- 新增 `COMMAND_LIBRARY.zh.md` O 节：
  - 第一帧 FoundationPose 直接策略抓取 `pick_diverse_bottles` 对照实验。
  - 推荐命令：`bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 0 --continue_on_error --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/first_frame_foundation_smoke`
- 新增命令入口：
  - `code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh`
  - 重要参数：`--foundation_frame`、`--grasp_surface_retreat_m`、`--approach_offset_m`、`--place_z_mode`、`--lift_m`、`--viewer`。
- 新增核心脚本：
  - `code_painting/plan_first_frame_foundation_pick_diverse_bottles.py`
  - 生成 `plan_summary_first_frame_foundation.json` 后通过 `--reuse_plan_summary_json` 调用现有 Piper planner。
- 验证：
  - Python `py_compile` 通过。
  - wrapper `bash -n` 通过。
  - `--ids 0 --dry_run` 路径解析通过。
  - `pick_diverse_bottles id0` 无 viewer smoke 完成；pregrasp reached，grasp 未双臂同时 reached，默认跳过 close/action。

## 2026-05-28 19:20:00 +08

- 新增 `COMMAND_LIBRARY.zh.md` L15.6：
  - 六任务各前 5 个 D435 summary 的 no-viewer 指令。
  - 六任务各前 5 个 D435 summary 的 viewer 指令，使用脚本 `--viewer` 和独立输出根目录。
  - 第一关键帧 debug 指令：`--debug_stop_after_keyframe1` 只执行 init -> pregrasp -> grasp，不关爪、不进入第二关键帧。
- 更新 `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`：
  - 新增 `--viewer`、`--output_root`、`--debug_stop_after_keyframe1`、`--trajectory_mode`、`--cartesian_auto_step_m`、`--joint_interp_waypoints`、`--replan_attempts`、`--allow_partial_dual_stage`、`--print_pose_every`。
  - 默认传 `--require_keyframe1_reached_before_close 1`，第一关键帧未 reached 时不关夹爪。
  - 恢复默认执行节奏为 `execute_interp_steps=24`、`joint_command_scene_steps=10`、`settle_steps=30`、`joint_target_wait_steps=25`。
- 新增 planner 参数：
  - `--debug_stop_after_keyframe1`
  - 用途：隔离第一关键帧可达性，判断失败是否发生在 keyframe-1 的 cartesian waypoint IK 阶段。
  - `--require_keyframe1_reached_before_close`
  - 用途：对齐旧 R1/V7 的行为意图，第一关键帧未到位时不进入抓取闭合。
  - `--print_execution_pose_every`
  - 用途：执行期间按步输出 TCP/EE world position，用于确认 waypoint 执行是否真的改变 pose。

## 2026-05-28 18:30:00 +08

- 新增 `COMMAND_LIBRARY.zh.md` L15.5：
  - `stack_cups id0` 的 D435 viewer 单条调试命令。
  - 先运行 `probe_sapien_viewer.py`，再用 `unset CUDA_VISIBLE_DEVICES` + `--enable_viewer 1` 打开 SAPIEN viewer。
  - 记录该 id 的 per-arm 关键帧：right 51/106，left 139/195。
  - 记录 planner `rank_previews` 与 J1.1 D435 preview 的区别：前者是 SAPIEN 3D 渲染并包含 `candidate_target_local_x_offset_m=-0.05` 的 5 cm TCP 补偿；`approach_offset_m=0.12` 只影响 pregrasp。
  - J1.1 源 preview 已复制到 `anygrasp_plan_keyframes_piper_d435_v1/stack_cups/foundation_input_0/preview_compare_d435/`。
  - 补充 viewer 探针失败条件：如果 `DISPLAY=` 为空并报 `Renderer does not support display`，需要从图形终端或正确 X11/Wayland forwarding 环境运行。
- 更新 `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md`：
  - 增加 L15.5 viewer 入口说明。

## 2026-05-28 09:30:00 +08

- 更新 `COMMAND_LIBRARY.zh.md` L15.1 说明：
  - 将 `dual_stage_require_all_plans` 与 `require_keyframe1_reached_before_action` 表述为对齐旧 V7 行为意图的显式约束，而不是旧 V7 原有参数。
- 新增 `COMMAND_LIBRARY.zh.md` L15.2：
  - 无 viewer 批跑版：保留 `CUDA_VISIBLE_DEVICES=${GPU}`，用于 id0-id10 稳定批处理。
  - viewer 单条调试版：先 `unset CUDA_VISIBLE_DEVICES`，避免 SAPIEN viewer 因 display GPU 被 mask 而无法弹窗。
  - 增加 `probe_sapien_viewer.py` 最小 viewer 探针命令。

## 2026-05-27 22:10:00 +08

- 新增 `COMMAND_LIBRARY.zh.md` L5.1：
  - 6 task 原始人手 head + pure replay action/wrist 的 processed HDF5 命令。
  - prompt 使用用户给定的 6 task 完整英文描述。
- 新增 `COMMAND_LIBRARY.zh.md` L5.2：
  - 新三任务当前可用的原始人手 head + D435 pure replay action/wrist 命令。
  - 记录 L5.1 skip 的原因：新三任务缺少 `h2_pure/<TASK>/id<ID>_z005`，但已有 `h2_pure_d435/<TASK>/id<ID>_d435_z005`。
- 新增 `COMMAND_LIBRARY.zh.md` I1.1/I3.5/L8.1：
  - I1.1：新三任务 Stage-1 人手抠除背景。
  - I3.5：新三任务 D435 visible-reinit robot repaint。
  - L8.1：把 I3.5 输出转成 processed HDF5。
- 新增 `COMMAND_LIBRARY.zh.md` L0/L10.5：
  - L0：按 human、robot replay、AnyGrasp replay 三条数据线说明执行顺序。
  - L10.5：把 L5.2 的新三任务 `human_head_pure_d435_action` 转成 LeRobot cache。
- 新增 `COMMAND_LIBRARY.zh.md` L11.1：
  - 为 3 个默认广角 robot replay 数据集和 3 个 AnyGrasp robot 数据集增加 25 episode 子集抽取命令。
  - 新增显式 zip 命令：`robot_replay_3task_25ep.zip` 和 `robot_anygrasp_3task_25ep.zip`，避免 `*_25ep` 把无关历史子集一起打包。
  - 新增 6 个 `_25ep` 输出的 `meta/info.json` 检查命令。
- 新增 `COMMAND_LIBRARY.zh.md` L11.2：
  - 把 25 episode 抽取扩展到 6 个 task：`pick_diverse_bottles`、`place_bread_basket`、`stack_cups`、`handover_bottle`、`pnp_bread`、`pnp_tray`。
  - 分别生成 `robot_replay_6task_25ep.zip` 和 `robot_anygrasp_6task_25ep.zip`。
- 新增 `COMMAND_LIBRARY.zh.md` L11.3：
  - 记录 task prompt 应在 processed data 的 `episode_*/instructions.json` 设置；当前 `convert_aloha_data_to_lerobot_R1.py --task` 不覆盖该文本。
- 补齐 `COMMAND_LIBRARY.zh.md` L6.1/L9.1/L10.4：
  - L6.1：6 task `pure_repaint` processed HDF5。
  - L9.1：6 task `anygrasp_repaint` processed HDF5。
  - L10.4：把 human_head_pure_action、pure_repaint、anygrasp_repaint 三类 processed HDF5 转成 LeRobot cache，作为 L11/L11.2 的源数据。
- 验证：
  - L5.1/L5.2/L6.1/L9.1/L10.4/L10.5/L11.1/L11.2/L11.3/I1.1/I3.5/L8.1 新增 bash 代码块抽取后通过 `bash -n`。

## 2026-05-27 21:45:00 +08

- 新增 `COMMAND_LIBRARY.zh.md` L15.1：
  - Piper AnyGrasp id0-id10 viewer 可视化版。
  - 执行节奏对齐旧 V7：`--execute_interp_steps 24`、`--joint_command_scene_steps 10`、`--settle_steps 30`、`--joint_target_wait_steps 25`。
  - 新增 `--require_keyframe1_reached_before_action 1`，第一关键帧 grasp 未 reached 时跳过第二关键帧 action。
- 新增 CLI：
  - `--require_keyframe1_reached_before_action`

## 2026-05-27 21:10:00 +08

- 更新 Piper AnyGrasp 批量运行命令：
  - `COMMAND_LIBRARY.zh.md` 新增 L15，给出 `pick_diverse_bottles` id0-id10 命令。
  - L15 不再传 `--keyframes 38 78`，改为依赖 `--reuse_preview_frame_mode annotated_json_keyframes` 读取每个 id 的手动标注关键帧。
  - 命令新增 `--dual_stage_require_all_plans 1`，确保双臂 stage 只有在左右臂都规划成功时才一起执行。
  - 补充说明 `settle_steps/joint_target_wait_steps` 与 IK 失败、视频停留帧之间的区别。

## 2026-05-27 14:00:00 +08

- 新增命令库小节：`COMMAND_LIBRARY.zh.md` C1.2。
- 新增内容：六个 H2O 任务的 FoundationPose replay D435 内参版命令。
- 关键参数：
  - `--image_width 640`
  - `--image_height 480`
  - `--fovy_deg 42.499880046655484`
  - `--camera_cv_axis_mode legacy_r1`
  - `--robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json`
- 输出约定：`/home/zaijia001/ssd/data/piper/hand/<TASK>/foundation_replay_d435`。
- 使用备注：该组命令用于让 FoundationPose 物体 replay 的 head 画面与后续 D435 robot pure replay 保持相同内参。

## 2026-05-26 21:30:00 +08

- 更新 Piper AnyGrasp 两关键帧规划命令：
  - `COMMAND_LIBRARY.zh.md` L14 增加 id0-id10 批量命令。
  - 推荐调试参数改为 `--urdfik_cartesian_interp_auto_step_m 0.02`、`--urdfik_max_position_threshold_m 0.02`、`--urdfik_max_rotation_threshold_rad 0.12`。
  - 位置优先调试阶段建议 `--reach_rot_tol_deg 180`；确认位置可达后再收紧旋转容差。
  - `--debug_visualize_ik_waypoints` 明确为 0/1 开关，使用 `1`。
  - 新增 `--head_only 0 --third_person_view 1 --vscode_compatible_video 1`，让 head/third 输出可直接在 VS Code 预览。
- 相关代码：
  - `code_painting/plan_anygrasp_keyframes_r1.py`
  - `code_painting/render_hand_retarget_piper_dual_npz_urdfik.py`
  - `code_painting/urdfik.py`

## 2026-05-25 16:10:00 +08

- 后续修复：
  - L1 明确为“原始人手 head + pure replay action/wrist”，并新增 `--head-dir-template '.' --head-video-name 'rgb_{id}.mp4'` 命令。
  - L1/L2/L3 都新增 `--review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json`，用于排除 `reject/discard/bad`。
  - 相关转换脚本已支持 `hand_keyframes_all.json` 的 `status` 字段过滤和 `{id}` 视频文件名模板。
  - 验证：L1/L2/L3 新命令通过 `bash -n`；L1 与 L2 单 id 临时转换通过。
- 后续补充：
  - 新增 `COMMAND_LIBRARY.zh.md` L5/L6/L7。
  - L5/L6 分别列出 `pick_diverse_bottles`、`place_bread_basket`、`stack_cups` 三个任务的可复制转换命令。
  - L7 新增 episode 数量统计、HDF5 结构检查和 `visualize_processed_hdf5_episode.py` review mp4 可视化命令。
  - 新增可视化入口：`policy/pi0/scripts/visualize_processed_hdf5_episode.py`。
- 继续补充：
  - 新增 `COMMAND_LIBRARY.zh.md` L8：D435 visible-reinit head + D435 pure replay action/wrist，分别列出三个任务命令。
  - 新增 `COMMAND_LIBRARY.zh.md` L9：AnyGrasp repaint head + planner action/wrist，分别列出三个任务命令，并注明需要 planner wrist 视频已补齐。
  - 验证：L8/L9 代表命令通过 `bash -n`。
- 继续补充：
  - 新增 `COMMAND_LIBRARY.zh.md` L10，记录 3 种已可用模式 x 3 个任务的 HDF5 -> LeRobot 转换命令。
  - L10 使用 `examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --use-wrist --mode video`，不使用 `convert_aloha_data_to_lerobot_robotwin.py`，因为当前 H2O processed HDF5 使用 `/observations/state`，而后者普通分支读取 `/observations/qpos`。
  - L10 明确 L6/L8 差异：L6 是默认广角 replay/repaint，L8 是 D435 visible-reinit replay/repaint。
- 更新 `COMMAND_LIBRARY.zh.md`，新增 `L. pi0 训练数据整理：原始人手、pure replay、AnyGrasp replay`。
- 新增命令说明：
  - 原始人手数据路径与既有 `policy/pi0/processed_data` 检查命令。
  - 未使用 AnyGrasp 的 pure replay：`process_repainted_headcam_with_wrist.py` 读取 repaint head、`world_targets_and_status.npz`、左右 wrist replay，并输出 pi0 HDF5。
  - AnyGrasp planner replay：`process_repainted_planner_outputs.py` 读取 repaint head、`pose_debug.jsonl`、左右 planner wrist 视频。
- 明确当前 AnyGrasp H2O planner 目录缺少 `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4`，L3 命令需要先补齐这两个输入才能稳定产出训练数据。
- 新增配套命令文档：
  - `agent-read/COMMANDS/pi0_h2o_training_data.zh.md`
  - `agent-read/COMMANDS/pi0_h2o_training_data.en.md`

## 2026-05-22 16:20:00 +08

- 修复 K1 Piper AnyGrasp planner 参数兼容性：
  - `plan_anygrasp_keyframes_r1.py` 新增并透传 `--debug_visualize_cameras`、`--debug_camera_axis_length`、`--debug_camera_axis_thickness`、`--target_local_forward_retreat_m`。
  - `plan_anygrasp_keyframes_r1_batch.py` 同步新增参数并传给单条 planner。
  - `COMMAND_LIBRARY.zh.md` K1 heredoc 命令显式加入默认值：
    - `--debug_visualize_cameras 0`
    - `--debug_camera_axis_length 0.16`
    - `--debug_camera_axis_thickness 0.006`
    - `--target_local_forward_retreat_m 0.0`
- 原因：Piper planner 最终继承 `HandRetargetR1Renderer`，该 renderer 已新增上述构造参数；AnyGrasp planner 路径未同步导致初始化 TypeError。

## 2026-05-22 15:55:00 +08

- 更新 `COMMAND_LIBRARY.zh.md` K1：
  - 将 `bash -lc '...'` 超长单行命令改为 heredoc 生成 `/tmp/run_h2o_k1_preview_resume.sh` 后执行。
  - 避免 zsh 中误用中文弯引号 `‘ ’` 导致进入续行状态，也避免直接依赖 zsh 支持 bash 内置 `mapfile`。

## 2026-05-22 15:45:00 +08

- 更新 `COMMAND_LIBRARY.zh.md` K1：
  - 规划命令不再直接扫描任务下所有 AnyGrasp 目录。
  - 改为从 `anygrasp_h2o_preview/<TASK>/foundation_input_<ID>/summary.json` 反推可用 id，再传入 `--ids`。
  - 增加 `--skip_existing 1 --continue_on_error 1`，支持部分任务安全续跑，避免未生成 K0.2 preview 的 id 反复报 `missing_preview_summary`。

## 2026-05-22 15:25:00 +08

- 更新 H2O 人工关键帧标注工具：
  - `Space` 继续标注整体关键帧，写入 `keyframes`
  - `l`/`L` 标注左手关键帧，写入 `left_keyframes`
  - `r` 标注右手关键帧，写入 `right_keyframes`
  - 原 `r` replay 功能改为 `R`，避免和右手标注冲突
- 更新命令库：
  - `COMMAND_LIBRARY.zh.md` K0 同步说明左右手分开标注字段。

## 2026-05-22 15:05:00 +08

- 新增 H2O AnyGrasp 人工关键帧标注入口：
  - `code_painting/annotate_hand_keyframes.py`
  - 支持逐视频标注 `hand_vis_gripper_*.mp4` 的关键帧，并把 JSON key 归一化为 `hand_vis_<id>.mp4`
  - 支持按 `d` 标记 `status=reject`，用于废弃坏视频/坏检测
- 更新 preview batch：
  - `code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh`
  - 自动跳过 `status=reject/discard/bad` 或少于两个关键帧的 id
- 更新命令库：
  - `COMMAND_LIBRARY.zh.md` K0 改为交互式标注脚本流程，保留可选物理移动废弃数据命令。

## 2026-05-06 11:40:00 +08

- 为 URDFIK replay 新增显式执行步数参数（可直接从命令行调）：
  - `--execute_waypoint_scene_steps`（每个关节轨迹点后仿真步数，默认 1）
  - `--execute_settle_scene_steps`（每帧结束附加仿真步数，默认 4）
  - `--urdfik_joint_interp_waypoints`（joint_interp 轨迹插值点数，默认 2）
- 适用入口：
  - `render_hand_retarget_piper_dual_npz_urdfik_main.py`
  - `render_hand_retarget_r1_npz_urdfik.py`
- 日志新增打印：`[execute-steps] ...`，便于核对实际执行配置。
- 同步更新：`/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`（D6 增加执行步数验证命令）。

## 2026-04-29 17:10:00 +08

- 更新命令工具：`run_piper_gripper_standard_pose_guess.sh`
  - 默认加入 `TARGET_DY=0.1 TARGET_DZ=0.1`（前上偏移）
  - 产物改为仅图片（zed/third PNG）+ `index.csv` + `world_targets_and_status.npz`
  - 增加 `left_status/right_status` 输出，方便定位是否 IK 未执行到位
- 命令库同步：`/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` D6
  - 明确“只保留 zed+third 图片”
  - 明确“Fail 常见由 IK 可达性引起，不等于朝向定义错误”

## 2026-04-29 16:30:00 +08

- 新增命令入口：
  - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_gripper_standard_pose_guess.sh ...`
- 用途：
  - 生成 Piper 夹爪“标准朝向猜测板”图片到单目录 `.../board/`
  - 同时输出 `index.csv`，便于逐图人工反馈与语义校准
- 相关代码：
  - `code_painting/run_piper_gripper_standard_pose_guess.sh`
  - `code_painting/run_piper_gripper_orientation_guess_board.sh`
- 同步更新命令库：`/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`（新增 D6）

## 2026-04-29 16:00:00 +08

- 更新 `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 的 HaMeR 指令：
  - GPU 命令从 `hamer-r1` 切换为 `hamer-r1-gpu`
  - GPU 命令增加 `unset LD_LIBRARY_PATH`（与原版 README 约定保持一致）
- 新增快速检测统计命令：
  - 直接读取 `hand_detections_*.npz` 的 `left_hand_detected/right_hand_detected` 计数
- 新增 debug 基准命令：
  - `CUDA_VISIBLE_DEVICES=2 + hamer-r1-gpu + video_id=0 + --no_visualize`

## 2026-04-27 17:35:00 +08

- `run_multi_object_pose_r1_npz_batch.sh` 所调用的 batch 入口现已支持：
  - `--save_pose_debug 1`
- Piper 标定 head cam 回放命令建议使用：
  - `--object star_fruit=/.../star.obj`（不要写成 `star fruit=`，避免对象名不匹配）

## 2026-04-27 14:55:00 +08

- 在 `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 增加了分对象重演命令：
  - `render_multi_object_pose_r1_npz_batch.py --objects pear ...`
  - `render_multi_object_pose_r1_npz_batch.py --objects star_fruit ...`
- 在 FoundationPose -> RoboTwin 批量回放命令中补充 `--save_pose_debug 1`。

## 2026-04-27 14:20:00 +08

- 将 FoundationPose 双物体命令中的星形物体提示词从 `star` 改为 `star fruit`（杨桃），减少 DINO 漏检。
- 同步更新：
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md`
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline.md`
  - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`

## 2026-04-27 13:55:00 +08

- 新增 FoundationPose 的 piper 准备与执行命令：
  - `conda run -n hamer-r1 python /home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py ...`
  - `bash /home/zaijia001/FoundationPose/run_piper_star_pear_foundation.sh`
- 新增 pear+star 双物体检测命令（run_realr1_dino_sam_batch.py + --object pear=... --object star=...）。
- 更新了 `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`，加入 FoundationPose 阶段与 RoboTwin 物体回放阶段。

## 2026-04-24 15:20:00 +08

- 新增跨项目手部处理指令文档：
  - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
- 新增内容采用“一行注释 + 一行命令”格式，包含：
  - Piper -> HaMeR 数据转换
  - HaMeR 单条/批量检测
  - `ffplay` 可视化检查
  - RoboTwin 下游 replay 命令
- 相关入口脚本：
  - `/home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py`
  - `/home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py`

## 2026-04-16 04:10:00 +08

- 更新 `code_painting/pika/visualize_calibrated_piper_pika_scene.py`
- 更新 `code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`
  - 新增参数：
    - `--viewer 1`
    - `--viewer-camera {overview,head}`
  - 这两个脚本现在可以打开交互 viewer。
- 更新命令文档：
  - `agent-read/COMMANDS/pika_scene_commands.en.md`
  - `agent-read/COMMANDS/pika_scene_commands.zh.md`

## 2026-04-16 03:55:00 +08

- 更新 `agent-read/COMMANDS/pika_scene_commands.en.md`
- 更新 `agent-read/COMMANDS/pika_scene_commands.zh.md`
  - 新增了“只导出 head cam 图”的明确命令
  - 新增了“当前标定脚本还不导出 wrist 视角”的明确说明

## 2026-04-16 03:40:00 +08

- 新增 `agent-read/COMMANDS/pika_scene_commands.en.md`
- 新增 `agent-read/COMMANDS/pika_scene_commands.zh.md`
  - 采用“一行注释 + 指令”格式
  - 包含手动桌面场景的 viewer 命令
  - 标明了当前标定场景脚本仍然只支持离屏渲染

## 2026-04-16 03:25:00 +08

- 新增 `code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`
  - 用途：
    - 用 version B 对齐方式重建标定场景
  - 与前一个标定场景脚本的关键区别：
    - 去掉 +90° 锚定旋转
    - 将左右分离尽量保持为 world y
    - 将桌子的约定旋转 90°
  - 示例：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`

## 2026-04-16 03:10:00 +08

- 新增 `code_painting/pika/visualize_calibrated_piper_pika_scene.py`
  - 用途：
    - 根据真实标定 bundle 重建模拟场景
  - 输入：
    - `robot_config_PiperPika_agx_dual_table.json`
    - `calibration_bundle_try2.json`
  - 输出：
    - `code_painting/pika/output_calibrated_scene/`
  - 示例：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/pika/visualize_calibrated_piper_pika_scene.py`

## 2026-04-16 02:50:00 +08

- 更新 `robot_config_PiperPika_agx_dual_table.json`
  - 将桌边这一侧的 base 位置从 `y = -0.60`（桌外）改为 `y = -0.30`（直接固定在桌边）
- 复用 `code_painting/visualize_piper_pika_agx_dual_table.py`
  - 导出桌边安装预览：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode oblique --output-dir code_painting/output_piper_pika_agx_dual_table_edge_mount --image-name piper_pika_agx_dual_table_edge_mount.png --video-name piper_pika_agx_dual_table_edge_mount.mp4`

## 2026-04-16 02:35:00 +08

- 更新 `robot_config_PiperPika_agx_dual_table.json`
  - 修正 base 朝向，使机械臂从桌子长边外侧朝桌面内部
  - 四元数：`[0.70710678, 0.0, 0.0, 0.70710678]`
- 更新 `code_painting/visualize_piper_pika_agx_dual_table.py`
  - 修正 `--camera-mode oblique` 为位于机械臂后方的正常斜视角
  - 在四元数修正后重新导出俯视结果
  - 示例：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode oblique --output-dir code_painting/output_piper_pika_agx_dual_table_oblique_fixed --image-name piper_pika_agx_dual_table_oblique_fixed.png --video-name piper_pika_agx_dual_table_oblique_fixed.mp4`
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode top_down --output-dir code_painting/output_piper_pika_agx_dual_table_topdown_fixed --image-name piper_pika_agx_dual_table_topdown_fixed.png --video-name piper_pika_agx_dual_table_topdown_fixed.mp4`

## 2026-04-16 02:20:00 +08

- 更新 `code_painting/visualize_piper_pika_agx_dual_table.py`
  - 进一步明确单侧长边双臂布局：
    - base 位于 x = ±0.30 m
    - 两 base 间距 = 0.60 m
  - 新增相机选择：
    - `--camera-mode top_down`
    - `--camera-mode oblique`
  - 俯视示例：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode top_down --output-dir code_painting/output_piper_pika_agx_dual_table_topdown --image-name piper_pika_agx_dual_table_topdown.png --video-name piper_pika_agx_dual_table_topdown.mp4`

## 2026-04-16 02:05:00 +08

- 新增 `code_painting/visualize_piper_pika_agx_dual_table.py`
  - 用途：
    - 预览新版彩色 Piper+Pika 在 120x60x75 cm 桌子上的双臂布局
  - 示例：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1`
- 新增 `robot_config_PiperPika_agx_dual_table.json`
  - 使用 UR 风格对称拆分，`embodiment_dis = 0.60`
  - 实际 base pose：
    - 左 `[-0.30, -0.60, 0.75]`
    - 右 `[0.30, -0.60, 0.75]`

## 2026-04-16 01:45:00 +08

- 新增一轮颜色统计分析（仅诊断，不改源文件）
  - 将导出的预览 PNG 与 Piper DAE 的 diffuse 参考色做近似比较
  - 使用前景颜色统计与 ΔE76 距离记录结果

## 2026-04-16 01:45:00 +08

- 新增 `code_painting/visualize_agx_arm_sim_source.py`
  - 目的：
    - 预览新的 `agx_arm_sim` 中 Piper/Pika 路由对应的模型
  - 目标选项：
    - `--target piper_only`
    - `--target pika_only`
    - `--target piper_pika_combo`
    - `--target all`
  - 示例：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_agx_arm_sim_source.py --target all --output-root code_painting/output_agx_arm_sim_preview --video-frames 36 --fps 12`

## 2026-04-16 01:30:00 +08

- 更新 `code_painting/visualize_original_source_urdfs.py`
  - 新增灯光预设参数：
    - `--lighting {bright,dark}`
  - 暗灯光分步验证示例命令：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_original_source_urdfs.py --target both --lighting dark --output-root code_painting/output_original_source_urdf_preview_dark --video-frames 36 --fps 12`

## 2026-04-16 01:15:00 +08

- 更新 `code_painting/visualize_piper_pika_single.py`
  - 新增灯光预设参数：
    - `--lighting {bright,dark}`
  - 暗灯光预览示例命令：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --lighting dark --output-dir code_painting/output_piper_pika_preview_dark --image-name piper_pika_dark.png --video-name piper_pika_dark.mp4 --video-frames 36 --fps 12`

## 2026-04-16 01:00:00 +08

- 新增原始源 URDF 预览命令
  - 入口：
    - `code_painting/visualize_original_source_urdfs.py`
  - 用途：
    - 直接预览下载目录中的原始 Piper 手臂 URDF 和原始 Pika gripper URDF
  - 主要参数：
    - `--target {piper_arm,pika_gripper,both}`
    - `--output-root`
    - `--video-frames`
    - `--fps`
  - 典型用法：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_original_source_urdfs.py --target both --output-root code_painting/output_original_source_urdf_preview --video-frames 30 --fps 12`

## 2026-04-16 00:45:00 +08

- 复用 `code_painting/visualize_piper_pika_single.py` 导出一版基于 DAE 的预览结果
  - 示例命令：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --output-dir code_painting/output_piper_pika_preview_dae --video-frames 24 --fps 12`
  - 用途：
    - 在将手臂 visual mesh 从 STL 切到 DAE 后，验证外观是否恢复

## 2026-04-16 00:30:00 +08

- 更新 `code_painting/visualize_piper_pika_single.py`
  - 新增预览导出能力：
    - 静态图片导出
    - 短 mp4 视频导出
  - 常用参数：
    - `--offscreen-only`
    - `--save-image`
    - `--save-video`
    - `--video-frames`
    - `--fps`
    - `--output-dir`
  - 默认输出文件：
    - `code_painting/output_piper_pika_preview/piper_pika_preview.png`
    - `code_painting/output_piper_pika_preview/piper_pika_preview.mp4`

## 2026-04-16 00:00:00 +08

- 新增独立的 piper_pika 可视化命令
  - 入口：
    - `code_painting/visualize_piper_pika_single.py`
  - 用途：
    - 加载并可视化新组装的 `assets/embodiments/piper_pika/piper_pika.urdf`
- 2026-05-26
  - 新增命令组：Piper hand `origin/` 原始数据可视化审核。
  - 相关命令：
    - `python code_painting/review_piper_hand_origin.py --task pnp_tray --delay-ms 45`
    - `python code_painting/review_piper_hand_origin.py --task handover_bottle --delay-ms 45`
    - `python code_painting/review_piper_hand_origin.py --task pnp_bread --delay-ms 45`
  - 重要参数：
    - `--task` 指定任务目录。
    - `--start-id` 从指定 episode id 开始。
    - `--dry-run` 只写审核日志不移动目录。
  - 代码位置：
    - `code_painting/review_piper_hand_origin.py`
  - 使用说明：
    - 在 `RoboTwin_openvla` 环境运行。
    - 按 `b` / `d` 会把当前 episode 从 `origin/` 移动到 `bad/`，后续预处理会自然跳过。

  - 关键参数：
    - `--urdf`
    - `--offscreen-only`
  - 典型用法：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py`
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1`

## 2026-04-14 12:30:00 +08

- 更新 `code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
  - 用途：
    - 防止复用的 R1 batch parser 在 Piper V2 运行中继续注入 `robot_config_R1.json`
  - 当用户未显式指定时，新增强制默认参数：
    - `--robot_config /home/zaijia001/ssd/RoboTwin/robot_config_Piper_dual_v2.json`
    - `--head_camera_local_quat_wxyz 1.0 0.0 0.0 0.0`
    - `--head_camera_local_pos 0.0 0.0 0.0`
  - 效果：
    - batch 打印出的命令现在会指向 Piper V2 robot config，而不是 R1 config

## 2026-04-14 12:00:00 +08

- 新增真正的 Piper V2 批量规划命令：`bash code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh ...`
  - 入口：
    - `code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh`
    - `code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
    - `code_painting/plan_anygrasp_keyframes_piper_v2.py`
  - 配套文件：
    - `code_painting/replay_piper_dual_h5.py`
    - `code_painting/render_hand_retarget_piper_dual_npz_urdfik.py`
    - `robot_config_Piper_dual_v2.json`
  - 用途：
    - 仿照现有 UR 配置风格，提供一个真正面向 Piper 的双单臂布局
    - 在 viewer/replay/URDFIK 执行阶段保留左右 Piper 独立 base pose
  - 关键行为：
    - 使用 `dual_arm_embodied=false`
    - 从 `assets/embodiments/piper/piper.urdf` 加载两个 Piper URDF 实例
    - 用 `embodiment_dis=0.80` 将其分开
    - 实际 base pose 为 `[-0.4, -0.65, 0.72]` 与 `[0.4, -0.65, 0.72]`
  - 典型用法：
    - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh <anygrasp_root> <replay_root> <hand_dir> <output_root> --planner_backend urdfik ...`

## 2026-04-14 00:00:00 +08

- 新增 Piper 批量规划命令：`bash code_painting/run_plan_anygrasp_keyframes_piper_batch.sh ...`
  - 入口：
    - `code_painting/run_plan_anygrasp_keyframes_piper_batch.sh`
    - `code_painting/plan_anygrasp_keyframes_piper_batch.py`
    - `code_painting/plan_anygrasp_keyframes_piper.py`
  - 用途：
    - 在不改原 R1 planner 的前提下提供一个面向 Piper 的入口
    - 复用现有 AnyGrasp batch launcher / 单视频 planner 流程
  - 关键行为：
    - 默认使用 `robot_config_Piper_dual.json`
    - 用户未显式指定时默认注入 `--head_camera_local_quat_wxyz 1 0 0 0`
    - `--planner_backend urdfik` 时使用 `assets/embodiments/piper/piper.urdf`
  - 相关配置：
    - `robot_config_Piper_dual.json`
  - 典型用法：
    - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_batch.sh <anygrasp_root> <replay_root> <hand_dir> <output_root> --planner_backend urdfik ...`

## 2026-04-03 00:00:00 +08

- 新增 smooth 处理专题说明：`agent-read/smooth/README.zh.md`
  - 相关命令：
    - `bash code_painting/run_plan_anygrasp_keyframes_r1_batch.sh ...`
    - `bash code_painting/run_replay_pose_debug_smooth.sh ...`
    - `bash code_painting/batch_smooth_planner_outputs.sh`
  - 重点参数：
    - `--urdfik_trajectory_mode cartesian_interp_ik`
    - `--urdfik_cartesian_interp_steps`
    - `--urdfik_cartesian_interp_auto_step_m`
    - `--settle_steps`
    - `--joint_target_wait_steps`
    - `--replan_until_reached`
    - `--replan_until_reached_max_attempts`
  - 用途：
    - 解释 planner smooth、settle/wait、post-hoc replay smooth 的区别
    - 记录如何在不改代码时平衡导出视频观感与末端执行精度
  - 本轮继续补充：
    - 评估“固定 1cm EE 采样 + 前一解 seed + joint jump threshold 拒绝”的增强型 waypoint IK 方案
    - 对比其与当前 `cartesian_interp_ik` 的重合与差异
    - 补充多种连续性/平滑改进方案的实现难度排序
    - 结合 V7 debug 命令分析 try/replan 与 `joint_target_wait_steps` 的精度/观感权衡
    - 记录并实现一个新增误差指标：点到目标前进轴的横向距离（axis lateral distance）
    - 新增输出字段：`lat_cm`
    - 补充后续建议实现方向：对象相对 target adapter（`T_obj_hand_demo + Δ_robot`）

## 2026-03-27 00:20:00 +08

- 新增 raw planner v7 串联命令：`bash run_planner_v7_repaint_review_pi0.sh`
  - 入口：
    - `run_planner_v7_repaint_review_pi0.sh`
  - 用途：
    - 对 `anygrasp_plan_keyframes_realoffset_batch_pure-v7` 直接执行 repaint
    - 手动 review repaint 视频
    - 再转成 pi0 / robotwin processed_data
  - 关键路径：
    - planner root: `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v7`
    - repaint root: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_v7`
    - review json: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_v7/video_review.json`
    - processed_data: `/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-27-planner-v7`

- 新增 review 驱动的 Step2+Step3+Step4 总控命令：`bash run_reviewed_smooth_repaint_pi0_pipeline.sh`
  - 入口：
    - `run_reviewed_smooth_repaint_pi0_pipeline.sh`
  - 用途：
    - 从 `video_review.json` 提取 `y`（或按模式包含 `m`）对应的 id
    - 批量执行 Step2 smooth、Step3 smooth repaint、Step4 pi0 处理
  - 关键参数：
    - `TASK_NAME`
    - `REVIEW_JSON`
    - `REVIEW_MODE`
    - `RUN_SMOOTH`
    - `RUN_REPAINT`
    - `RUN_PI0`
    - `DRY_RUN`
    - `EXPERT_DATA_NUM`
    - `PI0_OUTPUT_DIR`
  - 关键输出：
    - `${SMOOTH_OUTPUT_ROOT}/${TASK_NAME}_${idx}/...`
    - `${REPAINT_OUTPUT_ROOT}/id_${idx}_head_cam_arm_gripper_cup_bottle_pad_target/target_with_original_head_cam_plan.mp4`
    - `${PI0_OUTPUT_DIR}/episode_*/episode_*.hdf5`

- 新增 smooth bundle 命令：`bash code_painting/batch_smooth_planner_outputs.sh`
  - 入口：
    - `code_painting/batch_smooth_planner_outputs.sh`
    - `code_painting/smooth_planner_outputs_from_pose_debug.py`
  - 用途：
    - 对 Step1 planner 输出去徘徊帧并插值平滑
  - 关键参数：
    - `INPUT_ROOT`
    - `OUTPUT_ROOT`
    - `INTERP_FACTOR`
    - `FPS`
    - `KEEP_HOVER_FRAMES_EVERY`
    - `DEDUP_POS_THRESH_M`
    - `DEDUP_ROT_THRESH_DEG`
    - `DEDUP_JOINT_THRESH_RAD`
    - `DEDUP_GRIPPER_THRESH`
  - 关键输出：
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/head_cam_plan.mp4`
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/left_wrist_cam_plan.mp4`
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/right_wrist_cam_plan.mp4`
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/pose_debug.jsonl`
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/smooth_summary.json`

## 2026-03-27 00:30:00 +08

- 新增 planner 同源 pi0 数据转换命令：`process_repainted_planner_outputs.py`
  - 入口：
    - `policy/pi0/scripts/process_repainted_planner_outputs.py`
  - 用途：
    - 使用 repaint 后的 planner head + planner wrist + planner pose_debug 生成 `processed_data` HDF5
  - 关键参数：
    - `--head-root`
    - `--head-dir-template`
    - `--head-video-name`
    - `--planner-root`
    - `--planner-dir-template`
    - `--left-wrist-video-name`
    - `--right-wrist-video-name`
    - `--pose-debug-name`
    - `--review-json`
    - `--review-mode`
    - `--ids`
    - `--ignore-ids`
  - 典型命令：
    - `python scripts/process_repainted_planner_outputs.py d_pour_blue "pour water" 27 --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' --head-video-name target_with_original_head_cam_plan.mp4 --planner-root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3 --planner-dir-template 'd_pour_blue_{id}' --left-wrist-video-name left_wrist_cam_plan.mp4 --right-wrist-video-name right_wrist_cam_plan.mp4 --pose-debug-name pose_debug.jsonl --review-json /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json --review-mode strict --ignore-ids --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-27-planner`

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
    - `--review-mode`
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
- 2026-05-07
  - 新增 retarget 后旋转 board 视频命令：
    - `code_painting/run_piper_retarget_postrot_board_video.sh`
  - 关键环境变量：
    - `CASE_MODE=standard|axis90|grid`
    - `ORIENTATION_REMAP_LABEL`
    - `TARGET_DX/TARGET_DY/TARGET_DZ`
    - `ARMS`
    - `MAX_FRAMES`
  - 输出：
    - `board/board_zed.mp4`
    - `board/board_third.mp4`
    - `board/board_zed_frame0000.png`
    - `board/board_third_frame0000.png`
    - `index.csv`

- 2026-05-07
  - 扩展局部轴扫图命令：
    - `VIDEO_MODE=1`：生成 `board_all_zed.mp4` 与 `board_success_zed.mp4`
    - `CANDIDATE_MODE=semantic`：生成 `forward_from_*` / `open_from_*` 语义候选
    - `FRAME_END` / `MAX_FRAMES` / `FRAME_STRIDE`：控制多帧范围
  - 代表命令：
    - `GPU=3 FRAME_IDX=0 FRAME_END=-1 MAX_FRAMES=32 ARM=left EXECUTE=1 CANDIDATE_MODE=remap VIDEO_MODE=1 FPS=5 SAVE_WRIST_VIEWS=0 bash code_painting/run_piper_local_axis_sweep_board.sh <input_npz> <output_dir>`

- 2026-05-07
  - 新增局部轴扫图命令：
    - `bash code_painting/run_piper_local_axis_sweep_board.sh <input_npz> <output_dir>`
  - 相关代码：
    - `code_painting/build_piper_local_axis_sweep_board.py`
    - `code_painting/run_piper_local_axis_sweep_board.sh`
  - 重要参数：
    - `FRAME_IDX`
    - `ARM`
    - `EXECUTE`
    - `SAVE_WRIST_VIEWS`
  - 使用说明：
    - 默认固定当前 `PiperPika` 标定 head cam 位姿
    - 默认先做纯扫图，不执行 IK
    - 如果要检查“是不是执行不到位”，把 `EXECUTE=1` 打开

- 2026-05-11
  - 新增最终 HaMeR 轴 replay 批处理命令：
    - `code_painting/run_piper_hamer_axes_replay_batch.sh <input_npz_or_dir> <output_root>`
  - 关键环境变量：
    - `GPU`
    - `FPS`
    - `MAX_FRAMES`
    - `ARMS=both|left|right`
    - `ID_FILTER=0,2,5-8`
    - `KEEP_ONLY_ZED_THIRD=1`
  - 固定 replay 规则：
    - `--require_stored_gripper_pose 1`
    - `--pose_source gripper`
    - `--orientation_remap_label identity`
    - `--stored_orientation_post_rot_xyz_deg 0 0 0`
  - 输出：
    - 每个 ID 输出到 `<output_root>/id_<id>`
    - `frames/` 默认只保留 zed/third RGB PNG
  - 文档调整：
    - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` D 段只保留最终 replay 指令
    - 详细 debug 指令迁移到 `/home/zaijia001/ssd/PIPER_GRIPPER_ORIENTATION_DEBUG.zh.md`

- 2026-05-11
  - 新增“HaMeR 手 + FoundationPose 物体同场 replay”命令：
    - `code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh <hand_npz_or_dir> <foundation_obj_vis_root> <output_root>`
  - 关键环境变量：
    - `GPU`
    - `FPS`
    - `MAX_FRAMES`
    - `ARMS=both|left|right`
    - `ID_FILTER=0,2,5-8`
    - `KEEP_ONLY_ZED_THIRD=0|1`
    - `OBJECT_MISSING_FRAME_POLICY=hide|hold_last`
  - 新增底层 replay 参数：
    - `--object_replay_input_dir`
    - `--object_missing_frame_policy`
    - `--objects`
    - `--object NAME=/path/to/mesh.obj`
  - 用法位置：
    - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 的 E 段
  - 验证：
    - 1 帧 smoke test 通过，输出 `/tmp/piper_hamer_axes_with_objects_smoke/id_0/zed_replay.mp4` 与 `third_replay.mp4`

- 2026-05-11
  - 新增 gripper/wrist-retreat 到物体的世界轴向距离图命令：
    - `code_painting/plot_piper_gripper_wrist_object_axis_distances.py`
  - 关键参数：
    - `--hand_npz`
    - `--object_dir`
    - `--output_png`
    - `--target_world_offset_xyz`
    - `--left_object`
    - `--right_object`
  - 输出：
    - PNG 曲线图
    - 同名 CSV 数值表
  - 用法位置：
    - `/home/zaijia001/ssd/PIPER_GRIPPER_ORIENTATION_DEBUG.zh.md` 的 D-debug-9

- 2026-05-18
  - 更新 Piper replay 命令到 0515/new_table 标定：
    - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 的 C/D/E 段
  - 新增标定说明段：
    - `COMMAND_LIBRARY.zh.md` 的 C0
  - 当前默认配置：
    - `--robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json`
    - `--head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325`
    - `--head_camera_local_quat_wxyz -0.16972170030359832 0.6934883729683636 -0.6816465914025073 0.16008230830760367`
  - 同步更新 wrapper 默认值：
    - `code_painting/run_piper_hamer_axes_replay_batch.sh`
    - `code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`

- 2026-05-18
  - 修正 `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 的 Piper 标定命令说明：
    - C0 标定源从“三文件”补齐为“四文件”，新增 `left_wrist_new_table_eye_in_hand.json`
    - C0 记录左右 wrist `gripper_T_camera` 的 pos/quat，避免后续启用 wrist camera 时只接右腕
    - C0 增加当前 base/head camera 摆放估算
    - F 段距离曲线命令恢复为单行可复制命令
  - 关键路径：
    - `head_d435_new_table_0515_head_from_wrist.json`
    - `left_base_T_right_base_new_table.json`
    - `left_wrist_new_table_eye_in_hand.json`
    - `right_wrist_new_table_eye_in_hand.json`

- 2026-05-18
  - 新增 `place_bread_basket` 的 HaMeR/Piper 调试命令：
    - 位置：`/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 的 D0
  - 新增命令类型：
    - HaMeR GPU 检测：`detect_hands_realr1.py --data_dir .../place_bread_basket/harmer_input --output_dir .../place_bread_basket/harmer_output`
    - HaMeR 自带夹爪可视化查看：`hand_vis_gripper_*.mp4`
    - Piper replay 检查：`run_piper_hamer_axes_replay_batch.sh .../place_bread_basket/harmer_output .../output_place_bread_basket_piper_hamer_axes`
  - 重要参数：
    - `GPU_ID=2`
    - `GPU=2`
    - `ARMS=both`
    - `KEEP_ONLY_ZED_THIRD=1`
    - `ID_FILTER=0`

- 2026-05-18
  - 新增 Piper 标定 bundle 命令：
    - 位置：`/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 的 C0 和 D0
  - 新增工具：
    - `code_painting/build_piper_calibration_bundle.py`
    - `code_painting/visualize_piper_calibration_bundle.py`
  - 新增 replay 参数/环境变量：
    - `--piper_calibration_bundle`
    - `CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json`
  - 新增可视化命令：
    - 输出 `/home/zaijia001/ssd/RoboTwin/code_painting/output_piper_calibration_bundle_0515/axes_compare_old_head.png`
    - 使用 viewer 打开 `place_bread_basket` id0 的相机/场景检查

- 2026-05-18
  - 新增 third 视角内可见的 head camera marker 命令：
    - `DEBUG_VISUALIZE_CAMERAS=1`
    - `DEBUG_CAMERA_AXIS_LENGTH=0.22`
  - 新增三版本对比命令：
    - `output_place_bread_basket_camera_compare/old_manual`
    - `output_place_bread_basket_camera_compare/new_table_pre0515`
    - `output_place_bread_basket_camera_compare/new_table_0515`
  - 用法：
    - 查看各目录下 `id_0/frames/third_0000.png`
    - marker 颜色：白色机身，红/绿/蓝为局部 `+x/+y/+z`，黄色为相机 `-Z` 光轴

- 2026-05-18
  - 新增 viewer 排查命令：
    - `code_painting/probe_sapien_viewer.py`
  - 更新 viewer 命令：
    - 运行前打印 `DISPLAY/WAYLAND_DISPLAY/XDG_SESSION_TYPE`
    - hand replay 内部打印 `[viewer] ...`
    - 加入 `--debug_visualize_cameras 1`，确保 third 图里也能看到 head camera marker

- 2026-05-18
  - 修正 viewer 调试命令：
    - 移除 `CUDA_VISIBLE_DEVICES=2`
    - 日志增加 `CUDA_VISIBLE_DEVICES`
    - 增加 `CUDA_VISIBLE_DEVICES=2` 的最小 viewer probe 反例命令
  - 原因：
    - SAPIEN viewer 需要 display GPU 可见；设置 `CUDA_VISIBLE_DEVICES=2` 可能隐藏 VNC/X display 对应 GPU，导致 `Renderer does not support display`

- 2026-05-18
  - 修正 0515 Piper head camera 命令参数：
    - 旧散写 quaternion：raw/optical handeye quaternion
    - 新散写 quaternion：`0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399`
    - 新 quaternion 是 `raw_optical @ legacy_r1.T` 后的 render/SAPIEN 相机位姿，和 `--camera_cv_axis_mode legacy_r1` 配套使用
  - 更新位置：
    - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 的 C0/C1/C2/C3/D1/E2
    - `code_painting/run_piper_hamer_axes_replay_batch.sh`
    - `code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`
    - `code_painting/plot_piper_gripper_wrist_object_axis_distances.py`
  - bundle 命令保持不变，但 `build_piper_calibration_bundle.py` 现在会自动把 raw head 转成 replay 所需 render/SAPIEN head；已重新生成 0515 和 pre-0515 bundle。

- 2026-05-18
  - 将 `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 的 D 段 replay 命令改成 bundle 优先：
    - place_bread_basket 的 id0/id1/批处理命令显式加入 `CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json`
    - D1 单条底层 replay 命令改用 `--piper_calibration_bundle`
    - D2 最终批处理、id 过滤、右手检查命令显式加入 `CALIBRATION_BUNDLE`
  - 原因：
    - wrapper 默认值当前已是正确 0515 render/SAPIEN head 参数，但 bundle 形式更稳，后续换标定时优先重生成/替换 bundle，不需要人工同步散写 `robot_config/head_pos/head_quat`。

- 2026-05-19
  - 更新 `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 的 D 段 debug 命令：
    - 新增 HaMeR 只跑指定视频 ID 的命令：`--video_ids $(seq 0 10)`
    - 新增 Piper replay 只跑 id0-id10 的命令：`ID_FILTER=0-10`
    - 新增 `D4. HaMeR 可视化视频 + Piper retarget replay 横向拼接对比`
  - 新增 ffmpeg 拼接命令：
    - 单个 id：`hand_vis_gripper_<id>.mp4` + `zed_replay.mp4`
    - 单个 id：`hand_vis_gripper_<id>.mp4` + `zed_replay.mp4` + `third_replay.mp4`
    - 批量 id0-id10：逐个生成 `compare_hamer_gripper_piper_zed_third_<id>.mp4`
  - 验证：
    - 对 id0 运行三联拼接命令成功，生成 `output_place_bread_basket_piper_hamer_axes/id_0/compare_hamer_gripper_piper_zed_third_0.mp4`

- 2026-05-19
  - 修正 D4 视频拼接的时间对齐问题：
    - HaMeR `hand_vis_gripper` 通常是 30fps，Piper replay 是 5fps，直接 `hstack` 会导致左侧 HaMeR 播放过快
    - 新增对齐版命令：用 `ffprobe` 自动计算 `zed_replay_duration / hamer_duration`，再对 HaMeR 输入应用 `setpts=R*PTS,fps=5`
    - 新增单 id 对齐版输出：`compare_aligned_hamer_gripper_piper_zed_third_<id>.mp4`
    - 新增批量 id0-id10 对齐版拼接命令
  - 验证：
    - 对 id0 生成成功：`output_place_bread_basket_piper_hamer_axes/id_0/compare_aligned_hamer_gripper_piper_zed_third_0.mp4`
    - id0 自动计算的拉伸比例约 `R=6.0`，对应 HaMeR 30fps 到 Piper 5fps

- 2026-05-19
  - 新增 replay 局部蓝轴后退命令格式：
    - `/home/zaijia001/ssd/RoboTwin/COMMAND_LIBRARY.zh.md` 的 D2 增加 `TARGET_LOCAL_FORWARD_RETREAT_M=0.05`
    - 底层参数为 `--target_local_forward_retreat_m 0.05`
  - 参数含义：
    - 正数沿夹爪自身局部 `+Z` 蓝色前进/接近轴的反方向后退
    - 与 `--target_world_offset_xyz` 不同，它不是固定世界坐标偏移，而是跟随每帧 eepose 朝向
  - 相关入口：
    - `code_painting/render_hand_retarget_r1_npz.py`
    - `code_painting/run_piper_hamer_axes_replay_batch.sh`
    - `code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`

- 2026-05-20
  - 修正 C1 FoundationPose 双物体 replay 命令相关兼容问题：
    - `/home/zaijia001/ssd/RoboTwin/COMMAND_LIBRARY.zh.md` 第 182 行附近新增 pick_diverse_bottles 命令说明
    - 命令仍使用显式 `--robot_config`、`--head_camera_local_pos`、`--head_camera_local_quat_wxyz` 的 0515 标定参数
  - 相关入口：
    - `code_painting/run_multi_object_pose_r1_npz_batch.sh`
    - `code_painting/render_multi_object_pose_r1_npz.py`
    - `code_painting/render_object_pose_r1_npz.py`
  - 验证：
    - `CUDA_VISIBLE_DEVICES=2 ... --ids 0 --max_frames 1 --skip_existing 0 ...` smoke test 通过，不再出现 renderer 构造参数缺失错误

- 2026-05-20
  - 补充 E2 手 + 指定 FoundationPose video dir 的三任务单条命令：
    - pick_diverse_bottles：`hand_detections_0.npz` + `foundation_input_0`，对象 `right_bottle/left_bottle`
    - place_bread_basket：`hand_detections_0.npz` + `foundation_input_0`，对象 `basket/bread`
    - stack_cups：`hand_detections_0.npz` + `foundation_input_0`，对象 `right_dark_red_cup/left_light_pink_cup`
  - 关键参数：
    - `--target_local_forward_retreat_m 0.05`：沿夹爪局部蓝色 `+Z` 前进轴反方向后退 5cm
    - viewer 开启方式：追加 `--enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02`
  - 验证：
    - pick_diverse_bottles id0 的 `--max_frames 1` smoke test 通过

- 2026-05-20
  - 修改 E2.1/E2.2/E2.3 三个手 + 物体 replay 命令为 id0-id10 批处理格式：
    - `pick_diverse_bottles`
    - `place_bread_basket`
    - `stack_cups`
  - 命令格式：
    - `source ... && for ID in $(seq 0 10); do CUDA_VISIBLE_DEVICES=2 conda run ...; done`
    - `--input_npz`、`--output_dir`、`--object_replay_input_dir` 均使用 `${ID}` 展开
  - viewer 使用说明：
    - 批量命令默认不开 viewer；如果需要 viewer，先把循环改成单 ID，再追加 viewer 参数

- 2026-05-20
  - 新增 G 部分距离曲线命令：
    - `plot_piper_gripper_wrist_object_axis_distances.py`
    - 三任务 id0-id10 批量生成 gripper/wrist-retreat 到物体中心的世界轴向距离 PNG/CSV
  - 关键参数：
    - `--left_object` / `--right_object` 显式指定左右手对应物体目录
    - `--max_frames 300`
    - 输出目录跟随 H2O replay 的 `id${ID}_z005`
  - 目的：
    - 用 `dx/dy/dz` 曲线判断物体与人手 z 轴偏低更像检测问题还是 replay/标定链路问题

- 2026-05-26
  - 新增 `COMMAND_LIBRARY.zh.md` L12：直接修正已生成 LeRobot cache 的 task 文本。
  - 新增命令：
    - 备份 `meta/tasks.jsonl` 和 `meta/episodes.jsonl`，然后用 `perl -0pi` 将 `pick diverse bottles` 替换为完整英文指令。
  - 说明：
    - 当前 LeRobot parquet 只存 `task_index`，任务文本在 meta 文件中；只改已生成 cache 时无需改 parquet。

- 2026-05-26
  - 新增 `COMMAND_LIBRARY.zh.md` L11：从已生成 LeRobot cache 中抽取指定 episode 并重新编号。
  - 新增脚本命令：
    - `uv run python scripts/subset_lerobot_episodes.py --source <repo-or-path> --output-repo-id <repo> --episodes '0-24' --overwrite`
    - `--episodes` 支持 `0,1-5,7` 这种逗号列表和闭区间范围，脚本会去重、升序排序并输出连续 `0..N-1` episode。
  - 关键用途：
    - 从 `local/h2o_pick_diverse_bottles_human_head_pure_action` 等已转换数据中直接抽 25 个 episode，避免重新跑 HDF5 -> LeRobot 转换。
  - 相关代码：
    - `policy/pi0/scripts/subset_lerobot_episodes.py`

- 2026-05-21
  - 更新 G 部分距离曲线可视化规则：
    - `plot_piper_gripper_wrist_object_axis_distances.py` 默认 `--plot_clip_abs_m 0.5`
    - PNG 会压缩显示超过 `±0.5m` 的大异常值，CSV 保留原始值
    - 如果需要查看完整比例，命令追加 `--plot_clip_abs_m 0`
  - 兼容性：
    - FoundationPose 某个物体 track 缺失时输出 NaN 曲线而不中断批处理

- 2026-05-21
  - 新增 `COMMAND_LIBRARY.zh.md` H 部分：HaMeR 原始手点 + FoundationPose 原始物体点对比。
  - 新命令入口：`code_painting/make_hamer_foundation_point_compare_video.py`。
  - 关键参数：
    - `--hand_npz`：HaMeR `hand_detections_<id>.npz`
    - `--hand_video`：HaMeR `hand_vis_gripper_<id>.mp4`
    - `--object NAME=/path/to/foundation_input_<id>/<object>`：可重复传入多个物体目录
    - `--left_object` / `--right_object`：左右手 CSV 差值对应的物体名
    - `--output_video`：输出拼接视频，同名 `.csv` 自动生成
  - 已记录三任务 id0-id10 批处理命令：
    - pick_diverse_bottles：`left_bottle/right_bottle`
    - place_bread_basket：`basket/bread`
    - stack_cups：`left_light_pink_cup/right_dark_red_cup`
  - 使用说明：该对比不经过 Piper replay，只检查原始检测/物体 pose 层面的点位偏差。

- 2026-05-21
  - 修改 `COMMAND_LIBRARY.zh.md` H2 指令说明：
    - `make_hamer_foundation_point_compare_video.py` 现在默认同时输出 `*_distance.png`
    - 距离曲线与 G 部分一致，按帧画左右手对应物体的相机坐标系 `dx/dy/dz`
    - 新增参数说明：`--plot_clip_abs_m 0` 关闭曲线压缩显示；`--output_plot` 指定 PNG 路径
  - H6 增加查看 `*_hamer_foundation_points_distance.png` 的命令。

- 2026-05-21
  - 更新 `COMMAND_LIBRARY.zh.md` E/H 部分：
    - 新增 E2.0：三任务纯人手 replay，去掉 `--object_replay_input_dir` 和 `--object` 参数
    - E2.0 覆盖 pick_diverse_bottles、place_bread_basket、stack_cups 的 id0-id10 批处理
    - H1 后新增当前 H 原始 CSV 统计摘要，用于和 G/H1 world replay 统计对比
  - 验证：三条 E2.0 loop 命令 `bash -n` 通过。

- 2026-05-21
  - 更新 E2.0 纯人手 replay 命令：
    - `--save_png_frames 0` 用于避免生成 `frames/` 逐帧 PNG
    - 新增单视频转码命令，把 replay mp4 转成 VS Code 更兼容的 H.264/yuv420p
  - 验证：相关命令 `bash -n` 通过。

- 2026-05-21
  - 更新 `COMMAND_LIBRARY.zh.md`：
    - 新增 E0：三任务 H2O pure Piper replay，批量处理 `id0-id10`，只保留 `zed_replay.mp4` / `third_replay.mp4`，关闭 overlay text、target/camera axis 可视化，并转码为 VS Code 更兼容的 H.264/yuv420p。
    - 新增 I：按 `/home/zaijia001/usage.sh` 的两阶段 SAM repaint 流程，先生成人手抠除背景 `human_hand_bg.mp4`，再把 E0 pure robot replay 贴回背景。
    - 新增 J：对三个任务的 AnyGrasp 输出做 id0-id10 可用性筛选，并用 `render_anygrasp_ranked_preview.py` 生成和 HaMeR 人手朝向/目标物更接近的候选 preview/summary。
    - 新增 K：使用 J 的 `summary.json` 驱动 Piper AnyGrasp keyframe replay，并把 `head_cam_plan.mp4` 贴回 I1 背景。
  - 关键路径：
    - pure replay 输出：`code_painting/human_replay/h2_pure/<task>/id<ID>_z005`
    - repaint 输出：`/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2`
    - AnyGrasp preview：`code_painting/anygrasp_h2o_preview`
    - AnyGrasp plan：`code_painting/anygrasp_h2o_plan`
  - 验证：
    - 从 E0 起抽取 28 个 bash 代码块，`bash -n /tmp/command_library_new_blocks.sh` 通过。
    - 确认 `run_human_robot_inpaint_repaint.py`、`render_anygrasp_ranked_preview.py`、`run_plan_anygrasp_keyframes_piper_batch.sh` 存在。

- 2026-05-21
  - 修正 `COMMAND_LIBRARY.zh.md` I1/I2 SAM repaint 指令：
    - I1 现在只要求 `harmer_input/rgb_<id>.mp4` 存在，并使用一个已存在的 dummy robot video 满足 `run_human_robot_inpaint_repaint.py --robot_video` 必填参数。
    - 说明 I1 的 `human_hand_bg.mp4` 不依赖 dummy robot 是否对应同一个 task/id。
    - I2 仍严格要求每个 task/id 的 E0 pure `zed_replay.mp4` 存在，并分别打印缺失的 BG 或 pure robot 路径。
  - 原因：用户运行 I1 时大量 `[skip] missing HUMAN or ROBOT`，实际 `HUMAN` 存在，缺的是尚未生成完整的 `h2_pure/<task>/id<ID>_z005/zed_replay.mp4`。
  - 验证：抽取 I 段 4 个 bash 代码块，`bash -n /tmp/command_library_I_blocks.sh` 通过。

- 2026-05-22
  - 修正 `COMMAND_LIBRARY.zh.md` I2/K2 repaint 背景路径：
    - Stage-1 背景实际位于 `stage1_human_inpaint/removed_w_mask_*.mp4`，不是每次都会生成顶层 `human_hand_bg.mp4`。
    - I2/K2 现在先尝试顶层 `human_hand_bg.mp4`，不存在时自动 fallback 到 `stage1_human_inpaint/removed_w_mask_*.mp4`。
    - 缺失提示改为指向 `${BG_ROOT}/stage1_human_inpaint`，便于定位 I1 输出。
  - 同步修正 I1 输出检查命令，直接查找 `stage1_human_inpaint/removed_w_mask_*.mp4`。
  - 验证：I/K2 repaint 代码块 `bash -n` 通过；三任务抽样 id0/id1/id10 的 BG fallback 均解析到现有 `removed_w_mask_rgb_<ID>.mp4`。

- 2026-05-22
  - 在 `COMMAND_LIBRARY.zh.md` K1 前新增 K0 人工筛选流程：
    - K0.1：用 `mpv`/`ffplay` 人工查看 HaMeR gripper 视频并记录关键帧。
    - K0.2：把 `/tmp/h2o_manual_keyframes.tsv` 转成每个 task 的 `h2o_manual_review/<task>/hand_keyframes_all.json`。
    - K0.3：用 `--frame_selection_mode hand_keyframes_json` 按人工关键帧重跑 AnyGrasp preview summary，供 K1 的 `annotated_json_keyframes` 使用。
    - K0.4：新增 bad id 废弃命令，默认 `APPLY=0` dry-run，`APPLY=1` 时移动人手相关文件到 `_rejected_human_ids/` 并写入 `rejected_ids.json`。
  - 验证：抽取 K0 的 4 个 bash 代码块，`bash -n /tmp/command_library_K0_blocks.sh` 通过。

- 2026-05-22
  - 修正 H2O AnyGrasp 人工关键帧批处理命令：
    - `run_render_anygrasp_ranked_preview_keyframes_batch.sh` 新增 `VIDEO_PREFIX` 环境变量，默认仍为旧版 `d_pour_blue`，H2O 使用 `VIDEO_PREFIX=foundation_input`。
    - K0.3 改为调用 keyframe preview batch wrapper，对整个 task 批量生成 `summary.json`，不再手写单 id loop。
    - K1 去掉 `--ids $(seq 0 10)`，按 task 根目录批处理全部可用 `foundation_input_<id>`。
  - 旧版链路确认：先用人工关键帧 JSON 生成 preview summary，再由 planner 通过 `--reuse_preview_frame_mode annotated_json_keyframes` 消费 summary。
  - 验证：wrapper `bash -n` 通过；K0.3/K1 两个命令块 `bash -n` 通过。

- 2026-05-22
  - 新增命令组：`COMMAND_LIBRARY.zh.md` E2.4 / I3。
  - 新增内容：
    - E2.4：D435 正常 head 视角 pure Piper replay，使用 `--image_width 640 --image_height 480 --fovy_deg 42.499880046655484`。
    - I3：使用 E2.4 的 `zed_replay_d435.mp4` 贴回 I1 背景。
  - 重要路径：
    - D435 参数来源：`/home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_input/params_35.json`
    - replay 输出：`/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/<TASK>/id<ID>_d435_z005/`
    - repaint 输出：`/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_d435/e0_robot/<TASK>/id_<ID>_d435/`
  - 使用注意：默认旧 replay 的 `fovy_deg=90` 是广角虚拟相机；D435 对齐版本应使用 E2.4/I3，并保留 `d435` 文件名后缀。
  - 验证：新增命令块 `bash -n` 通过。
  - 后续澄清：E2.4 的 `fovy_deg=42.499880046655484` 是从实际 RGB camera_info 内参换算得到；官方 D435 depth FOV `85.2 x 58` 不适用于 `/camera/color/image_raw` RGB replay。
  - 后续诊断：新增 I3.1，记录 D435 Stage-2 mask 失效原因和复查命令；结论是旧版 Stage-2 参数不能直接复用于 D435 窄视角 robot replay。

- 2026-05-23
  - 更新命令组：`COMMAND_LIBRARY.zh.md` I3.2 / I3.3。
  - 新增内容：
    - I3.2：D435 首帧无机器人/误检背景的参数建议和可见起始帧 contact sheet 检查命令。
    - I3.3：SAM3 直接复用 I1 Stage-1 背景做 D435 robot repaint 的单条调试命令。
  - 关键参数：
    - 建议 D435 初始调参使用 `--robot_box_threshold 0.35 --robot_text_threshold 0.30 --max_mask_area_ratio 0.35`。
    - SAM3 命令使用 `remove_anything_video_sam3_robot.py --target_video "$BG"`，避免通过 SAM3 总入口重新跑 Stage-1 human inpainting。
  - 注意：
    - 当前 SAM2/SAM3 robot 脚本都固定从第 0 帧初始化；如果第 0 帧完全没有机器人，必须同步裁剪 robot/BG 或后续改代码支持非 0 初始化帧。
  - 验证：新增 bash 命令 `bash -n` 通过。

- 2026-05-26
  - 更新命令组：`COMMAND_LIBRARY.zh.md` L14。
  - 新增 Piper AnyGrasp 两关键帧规划调试命令：
    - 入口：`code_painting/plan_anygrasp_keyframes_piper.py`
    - 推荐参数：`--planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik --urdfik_cartesian_interp_steps -1 --urdfik_cartesian_interp_auto_step_m 0.01`
    - 调试参数：`--debug_visualize_targets 1 --debug_visualize_ik_waypoints 1 --save_pose_debug 1 --save_debug_execution_preview 1`
  - 相关代码：
    - `plan_anygrasp_keyframes_piper.py` 现在使用 Piper dual renderer，按 arm 保留左右 base。
  - 验证：
    - `plan_anygrasp_keyframes_piper.py` 语法编译通过。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` J0.1 / J1.1。
  - 新增内容：
    - J0.1：6 task D435 AnyGrasp / `foundation_replay_d435` / HaMeR 输入可用性检查。
    - J1.1：6 task D435 关键帧候选 preview/summary 生成命令，输出到 `anygrasp_h2o_preview_d435`。
    - 路径与默认广角区分：D435 使用 `foundation_replay_d435` 和独立 preview root。
  - 验证：
    - 新命令块通过 `bash -n`。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` L11.2.4。
  - 新增内容：
    - D435 robot replay `_25ep` 抽取改为按 processed `source_episode_id` 过滤 bad 原始 id。
    - 排除规则沿用 human replay：`handover_bottle=0,7,12,29`，`pnp_bread=0,1,2,3,4,5,6,22,70`。
    - 命令会自动补足前 25 个可用 LeRobot episode index，并由 `subset_lerobot_episodes.py` 重新编号为 `0..24`。
  - 验证：
    - 新命令块通过 `bash -n`。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` I3.5。
  - 新增内容：
    - 说明 `d435_final` 的输出路径和生成链路：`batch_visible_reinit_d435_repaint.py` -> `remove_anything_video_sam3_robot_visible_reinit.py` -> `final_repainted.mp4`。
    - 明确当前本机 I3.5 批处理实际为 SAM2/DINO2 fallback，虽然入口位于 `inpainting_sam3_robot`。
    - 新增“先补到至少 25 个 final”的批处理命令，使用 `--overwrite 0` 保留已有输出，并复用一次加载的 DINO/SAM checkpoint。
  - 验证：
    - 新命令块通过 `bash -n`。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` I1.1.1 / I3.5。
  - 新增内容：
    - I1.1.1：新三任务只补缺失 Stage-1 BG 的 resume 命令，已有 `removed_w_mask_*.mp4` 的 id 会跳过。
    - I3.5：新增 `--id_start 0 --id_end 80 --overwrite 0` 的 D435 visible-reinit repaint resume 命令，已有 `final_repainted.mp4` 的 id 会跳过。
  - 使用场景：
    - 当 L8.2 输出 episode 很少，且日志显示缺 `/results_repaint_piper_h2_d435_sam3_visible_reinit/.../final_repainted.mp4` 时，先跑 I1.1.1，再跑 I3.5 resume，最后回到 L8.2。
  - 验证：
    - 新命令块通过 `bash -n`。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` L6.1 / L8.2 / L10.6 / L11.2.4。
  - 新增内容：
    - L6.1 增加默认广角 `h2_pure` 与 D435 `h2_pure_d435` 的边界说明，解释新三任务为何会 `No usable episodes were processed`。
    - L8.2 提供六任务 D435 visible-reinit robot replay 转 processed HDF5 命令。
    - L10.6 提供六任务 D435 robot replay processed HDF5 转 LeRobot cache 命令。
    - L11.2.4 提供六任务 D435 robot replay `_25ep` 抽取、zip 和 `rclone --dry-run` 命令。
  - 运行顺序：
    - 旧三任务：I1 -> I3.4 -> L8/L8.2 -> L10.3/L10.6 -> L11.2.4。
    - 新三任务：I1.1 -> I3.5 -> L8.1/L8.2 -> L10.6 -> L11.2.4。
  - 验证：
    - 新命令块通过 `bash -n`。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` J0.1 / J1.1。
  - 新增说明：
    - J0.1 的 `MISS` 要按 `anygrasp/replay/hand` 三列判断；`seq 0 120` 超过真实 episode 数时出现 MISS 是正常的。
    - J1.1 现在支持 `left_keyframes/right_keyframes`，并会把它们写入 `effective_keyframes_by_arm`。
    - L15.4 改为调用 `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`，避免 zsh 不支持 `mapfile`。
    - L15.4 增加每任务前 5 个 summary 的测试命令，并建议使用 `--continue_on_error`。
  - 相关代码：
    - `code_painting/render_anygrasp_ranked_preview.py`
    - `code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh`
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`
  - 验证：
    - `place_bread_basket id0` D435 preview 单条生成成功，summary 中记录 per-arm effective keyframes。
    - 六任务脚本 dry-run 可正确解析每个任务第一个 summary。
    - `pick_diverse_bottles id0` D435 planner 单条执行通过脚本层面验证并生成视频/summary。
    - L15.4 六任务 D435 planner 命令块通过 `bash -n`。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` J1.2 / L15.3。
  - 新增：
    - J1.2：D435 候选 preview/summary 的指定 id 重跑命令，示例使用 `pick_diverse_bottles` 的 `2 17 18 19 20 21`。
    - L15.3：D435 AnyGrasp planner 命令，强制使用 `foundation_replay_d435` 和 `anygrasp_h2o_preview_d435` 的 `summary.json`。
  - 关键参数：
    - `--image_width 640 --image_height 480 --fovy_deg 42.499880046655484`
    - `--reuse_preview_summary_json .../anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json`
    - `--replay_dir .../foundation_replay_d435/foundation_input_<ID>`
  - 用法说明：
    - D435 summary 缺失时直接 skip，应先重跑 J1.1/J1.2，不要 fallback 到默认广角 `anygrasp_h2o_preview`。
  - 验证：
    - 新增 bash 命令块通过 `bash -n`。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` L15.11。
  - 新增内容：
    - `--execute_partial_cartesian_plan` viewer/no-viewer 调试命令。
    - 说明该开关只对 `cartesian_interp_ik` 生效，执行成功 waypoint 前缀但不算 reached。
    - 记录位置优先/朝向优先 IK 的当前状态和后续可改方向。
  - 验证：
    - L15.11 bash block 通过 `bash -n`。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` L15.9/L15.10。
  - 调整内容：
    - L15.9 主目录 preview 命令改为 `--candidate_target_local_x_offset_m 0.0`。
    - L15.10 改为 offset -5cm 对比命令，输出到 `anygrasp_h2o_preview_d435_offset_minus_5cm_compare`。
    - 补充 `planner_selected_orientation_rank1.png` 输出说明。
  - 验证：
    - L15.9/L15.10 bash block 通过 `bash -n`。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` L15.10。
  - 新增内容：
    - raw/no-offset 对比版 D435 preview 生成命令，设置 `--candidate_target_local_x_offset_m 0.0`，输出到 `anygrasp_h2o_preview_d435_no_offset_compare`。
    - 增加 summary 对比脚本，用于比较 `translation_cam`、`visual_translation_cam` 和 `translation_world`。
  - 验证：
    - L15.10 bash block 通过 `bash -n`。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` L15.9。
  - 新增内容：
    - 在文件末尾追加复制安全版三步运行指令：D435 preview/summary 重生成、无 viewer replay、viewer target 可视化 replay。
    - 第一步改用 `bash <<'BASH'`，避免 zsh 断行导致 `OUT_ROOT=/ home/...`、参数 fallback 到默认 `replay_m_obj_pose_d_pour_blue_norobot` 或 `--candidate_target_local_x_offset_m: command not found`。
  - 验证：
    - L15.9 bash block 通过 `bash -n`。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` L15.8。
  - 新增内容：
    - 记录 D435 preview 与 planner target 的真实映射关系：`translation_cam` 为原始 AnyGrasp，`visual_translation_cam/translation_world` 为 offset 后的实际绘制/规划目标。
    - 补充六任务 D435 preview 重新生成命令，确保后续 `anygrasp_h2o_preview_d435` 图片和 summary/planner 目标一致。
    - 补充 offset 修正后的无 viewer replay 与 viewer target 可视化命令。
  - 验证：
    - L15.8 命令块通过 `bash -n`。
    - `pick_diverse_bottles id0` offsetfix debug preview 生成成功。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` L15.7。
  - 新增内容：
    - D435/Piper 六任务 wrapper 默认 `--reach_error_pose_source ee`，并支持显式覆盖。
    - 记录当前关键帧执行逻辑：D435 summary per-arm keyframes -> 第一关键帧 `pregrasp/grasp` -> reached 后 close/action。
    - 补齐六任务分别前 5 个 episode 的严格同步命令。
    - 补齐六任务分别前 5 个 episode 的 partial + `joint_interp` + `--print_pose_every 5` 诊断命令。
  - 关键说明：
    - `tcp` reached 检查会留下约 12 cm TCP/EE 偏移；当前 Piper D435 planner 默认按 `ee` 判定到位。
    - `--allow_partial_dual_stage` 只用于诊断，不建议作为最终数据设置。
  - 验证：
    - wrapper `bash -n` 通过。
    - 六任务 dry-run 通过。
    - `pick_diverse_bottles id0` partial 复跑确认右臂 EE reached。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` L15.7。
  - 新增内容：
    - viewer 命令统一增加 `--visualize_targets` 示例，用于显示目标 axis 和 active candidate gripper。
    - 记录 `<OUT>/source_preview_compare/` 输出，包括 D435/legacy preview 原图和 `selected_candidate_mapping.json`。
  - 验证：
    - L15.7 新增 bash block 通过 `bash -n`。
    - `pick_diverse_bottles id0` 生成 source preview compare 输出。

- 2026-05-28
  - 更新命令组：`COMMAND_LIBRARY.zh.md` L11.1.3。
  - 新增内容：
    - L10.5 后续专用 `_25ep` 抽取命令，源 repo 为 `local/h2o_<TASK>_human_head_pure_d435_action`。
    - 通过 processed `instructions.json/source_episode_id` 排除新三任务中的已知坏原始 id，并自动补足 25 个 episode。
    - 新增 `human_d435_action_3task_25ep.zip` 打包和 `rclone --dry-run` 上传检查命令。
  - 关键区别：
    - L11.2.1 的 `local/h2o_<TASK>_pure_repaint` 只用于 L6/L6.1 robot replay，不能作为 L10.5 的后续。
  - 验证：
    - 新增命令块通过 `bash -n`。

- 2026-05-25
  - 新增命令/模式设计：`COMMAND_LIBRARY.zh.md` I3.4。
  - 新增内容：
    - 记录“可见帧重初始化 SAM 模式”的设计，用于 D435 窄视角 robot replay。
    - 新模式建议使用 `--init_policy first_visible --reinit_policy on_lost --empty_mask_when_lost 1` 等待实现接口。
    - 新模式输出根目录建议为 `results_repaint_piper_h2_d435_sam3_visible_reinit`，和当前 SAM2/SAM3 输出并列比较。
  - 注意：
    - 这是待实现接口和状态机设计，不是当前可运行脚本；本轮没有修改任何运行代码。
  - 验证：I3.4 待实现命令 `bash -n` 通过。

- 2026-05-25
  - 更新命令组：`COMMAND_LIBRARY.zh.md` I3.4。
  - 新增可运行入口：
    - 单视频：`/home/zaijia001/ssd/inpainting_sam3_robot/remove_anything_video_sam3_robot_visible_reinit.py`
    - 批处理：`/home/zaijia001/ssd/inpainting_sam3_robot/batch_visible_reinit_d435_repaint.py`
  - 新增命令：
    - 单 id 调试命令，输出到 `results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/<TASK>/id_<ID>_d435`。
    - 三 task 批处理命令，使用 `--id_start/--id_end` 控制范围，并通过单进程复用 DINO/SAM checkpoint。
    - dry-run 命令，用于检查 BG/ROBOT 输入是否存在。
  - 关键参数：
    - `--init_policy first_visible`
    - `--reinit_policy on_lost`
    - `--empty_mask_when_lost 1`
    - `--detector_stride 1`
    - `--lost_patience 2`
    - `--max_mask_area_ratio 0.35`
    - `--max_white_pixel_ratio_in_mask 0.60`
  - 验证：
    - 两个新脚本 `py_compile` 通过。
    - 两个新脚本 `--help` 通过。
    - 批处理 dry-run 能解析 `pick_diverse_bottles id0` 的输入路径。
    - I3.4 命令 `bash -n` 通过。

- 2026-05-25
  - 更新命令组说明：`COMMAND_LIBRARY.zh.md` I3.3 / I3.4。
  - 标题调整：
    - I3.3 改为“当前 SAM3 项目首帧初始化：直接复用 I1 背景做 D435 robot repaint”。
    - I3.4 改为“新逻辑：可见帧重初始化 SAM2/SAM3 模式”。
  - 新增说明：
    - 原 SAM2 指令、当前 SAM3 项目指令、新可见帧重初始化指令的区别。
    - 当前环境日志 `[backend] SAM=sam2, DINO=dino2` 表示实际 fallback 到 SAM2/GroundingDINO2；如果看到 `[backend] SAM=sam3, DINO=dino3` 才是真正 SAM3/DINO3。
    - 新脚本已局部修复 transformers/GroundingDINO 的 `BertModel.get_head_mask` 兼容问题。
  - 验证：
    - 新脚本可完成模型加载。
    - 两个新脚本 `py_compile` 通过。

- 2026-05-25
  - 更新命令组：`COMMAND_LIBRARY.zh.md` I3.0 / I3.3 / I3.4。
  - 新增 I3.0 对照表和独立命令：
    - I3.0.1：原 SAM2 固定第 0 帧初始化单 id 调试命令。
    - I3.0.2：SAM3 项目固定第 0 帧初始化单 id 调试命令。
    - 真正 SAM3 backend 模板：设置 `GROUNDED_SAM3_DIR=/path/to/Grounded_SAM_3` 后运行 I3.0.2 或 I3.4，并检查日志 `[backend] SAM=sam3, DINO=dino3`。
    - I3.0.3：新逻辑可见帧重初始化单 id 调试命令。
    - I3.0.4：新逻辑可见帧重初始化批处理复用 checkpoint 命令。
  - 兼容修复记录：
    - `remove_anything_video_sam3_robot.py` 和 `remove_anything_video_sam3_robot_visible_reinit.py` 现在都会 patch 旧 GroundingDINO 需要的 BERT helper，兼容 transformers 5.3.0。
  - 验证：
    - I3.0 命令块 `bash -n` 通过。
    - DINO forward smoke test 通过。
## 2026-05-28（L15.12 Piper AnyGrasp 轴修正与 IK 阈值 wrapper）

- 新增/更新：
  - `COMMAND_LIBRARY.zh.md` 新增 L15.12，记录 Piper preview gripper 与 URDF link6 的轴转换关系。
  - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` 新增 `--ik_max_position_threshold_m`、`--ik_max_rotation_threshold_rad`。
  - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --viewer` 默认不再停在每个 id 末尾；新增 `--viewer_wait_at_end 1` 用于需要手动停留检查的情况。
  - `COMMAND_LIBRARY.zh.md` 新增 L15.13，补充六任务分别跑 id0-10 的 viewer 命令和轴颜色说明。
  - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` 新增 `--id_start`、`--id_end`、`--ids`、`--piper_apply_global_trans_to_ik`。
- 关键参数：
  - 默认仍为 `--ik_max_rotation_threshold_rad 0.12`，保持严格完整姿态 IK。
  - 诊断位置优先可临时使用 `--ik_max_rotation_threshold_rad 3.14`，确认静止是否来自朝向约束。
- 相关代码：
  - `code_painting/render_hand_retarget_piper_dual_npz_urdfik.py`
  - `code_painting/plan_anygrasp_keyframes_r1.py`
  - `code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`
## 2026-05-29（L15.14 Viewer per-arm active frame 检查）

- 更新命令文档：
  - `COMMAND_LIBRARY.zh.md` 新增 L15.14。
  - `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` 增加 viewer 目标夹爪按左右手关键帧显示的说明。
- 相关命令：
  - dry-run 检查 `pick_diverse_bottles id0-10` 的 viewer 轴检查命令是否能解析 D435 summary。
  - `jq -c '{stage,active_frame,active_frame_by_arm}' <OUT>/pose_debug.jsonl | head -n 20` 用于确认 pregrasp/grasp 显示第一关键帧、action 显示第二关键帧。
  - 若当前 shell 没有 `jq`，文档同步提供 `head ... | sed -E ...` 的检查版本。
- 重要参数：
  - `--visualize_targets`
  - `--id_start 0 --id_end 10`
  - `--piper_apply_global_trans_to_ik 0`
  - `--ik_max_rotation_threshold_rad 3.14`
## 2026-05-29（L15.15 Stack Cups id0 无碰撞 target-only 命令）

- 新增命令文档：
  - `COMMAND_LIBRARY.zh.md` 增加 L15.15。
  - `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` 同步 stack_cups id0 target-only 调试命令。
- 新增 wrapper 参数：
  - `--disable_execution_collisions`
  - `--target_axes_only`
  - `--debug_candidate_top_k`
  - `--debug_common_candidate_top_k`
  - `--debug_visualize_selected_keyframe_axes`
  - `--debug_visualize_ik_waypoints`
- 用途：
  - 排除执行物体碰撞。
  - 只保留当前执行 target 轴，避免 viewer 中 target/candidate/selected/waypoint 多套坐标系混淆。
## 2026-05-29（L15.16 Direct Piper hand replay viewer 对照命令）

- `COMMAND_LIBRARY.zh.md` 新增 L15.16。
- 新增 direct replay 对照命令：
  - 入口：`code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py`
  - 示例：`stack_cups id0`
  - 用途：不走 AnyGrasp，直接 replay HaMeR NPZ 中存好的 gripper pose。
- 关键参数：
  - `--debug_visualize_targets 1`
  - `--debug_mode 1 --debug_post_execute 1`
  - `--save_world_targets 1`
  - `--enable_viewer 1 --viewer_wait_at_end 1`
## 2026-05-29（L15.17 Direct replay / AnyGrasp 轴约定对照）

- `COMMAND_LIBRARY.zh.md` 新增 L15.17。
- 新增说明：
  - direct Piper hand replay：local `+Z` 蓝轴是 approach/forward。
  - AnyGrasp preview/planner：local `+X` 红轴是 wireframe finger-depth。
- 新增对照参数：
  - `--candidate_orientation_remap_label swap_red_blue`
- 用途：
  - 测试是否需要把 AnyGrasp local `+X` 映射到 direct replay local `+Z`，以解释/修正 viewer 中 gripper 轴与真实机器人朝向不一致的问题。

## 2026-05-29（L15.18 Replay-axis AnyGrasp 六任务命令）

- `COMMAND_LIBRARY.zh.md` 新增 L15.18。
- 新增入口：
  - `code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh`
- 新增/透传参数：
  - `--candidate_orientation_remap_label`
  - `--candidate_target_local_x_offset_m`
  - `--candidate_target_local_z_offset_m`
  - `--approach_axis`
  - `--approach_offset_m`
- replay-axis wrapper 固定：
  - `--candidate_orientation_remap_label swap_red_blue`
  - `--candidate_target_local_x_offset_m 0.0`
  - `--candidate_target_local_z_offset_m -0.05`
  - `--approach_axis local_z`
  - `--approach_offset_m 0.12`
- L15.18 已写入六任务前 5 个 no-viewer 命令、viewer 命令，以及 `stack_cups id0-10` 小范围 viewer 调试命令。

## 2026-05-29（L15.19 筛选阶段 frame 统一设计命令）

- `COMMAND_LIBRARY.zh.md` 新增 L15.19。
- 记录长期设计：
  - 在 `render_anygrasp_ranked_preview.py` 候选筛选阶段统一 AnyGrasp raw frame 与 robot/replay frame。
  - 目标 frame 为 `robot local +Z = AnyGrasp raw local +X`、`robot local +Y = AnyGrasp raw local +Y`、`robot local +X = -AnyGrasp raw local +Z`。
- 新增当前可运行对照命令：
  - 入口仍为 `run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh`。
  - 输出根目录改为 `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper`。
- 说明：该命令目前仍使用 L15.18 wrapper，不代表筛选阶段统一 frame 已实现。

## 2026-05-29（L15.19.1 robot-frame preview/planner 命令）

- 新增 robot-frame preview 生成命令入口：
  - `run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh`
- 新增 robot-frame planner 命令入口：
  - `run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh`
- 新增关键参数：
  - `--candidate_frame_mode robot_replay`
  - `--candidate_target_local_z_offset_m`
  - `--preview_root`
  - `--debug_gripper_actor_forward_axis local_z`
- 输出：planner viewer 结果写入 `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper`。

## 2026-05-29（robot-frame planner 自动补 preview）

- 修改入口：
  - `code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh`
  - `code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh`
- 新行为：
  - planner wrapper 会先自动生成缺失的 robot-frame preview summary，再运行 planner。
  - 自动生成范围跟随 `--tasks`、`--ids`、`--id_start/--id_end`、`--max_per_task`。
- 新参数：
  - `--skip_preview_generation`：planner wrapper 不自动生成 summary。
  - `--skip_existing`：preview wrapper 是否跳过已有 summary，默认 `1`。
  - `--source_preview_root`：preview wrapper 用于确定可用 id 顺序的源 D435 preview root。

## 2026-05-29（L15.19.2 robot-frame 指定 id viewer 命令）

- `COMMAND_LIBRARY.zh.md` 新增：
  - `stack_cups id4` robot-frame viewer 命令。
  - 六任务分别指定 id 的 viewer 模板。
  - 六任务同时指定 `--ids 0 1 2 3 4` 的 viewer 命令。
- 关键参数：
  - `--ids <ID>` 精确指定 episode。
  - robot-frame wrapper 会自动补缺失 summary；如需禁用可加 `--skip_preview_generation`。

## 2026-06-02（Mode O gripper 轴约定说明）

- 命令格式未变，仍使用：
  - `code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh`
  - `code_painting/plan_first_frame_foundation_pick_diverse_bottles.py`
- 文档补充：
  - Mode O 当前保存给 planner 的 target frame 使用 local `+Z` 作为接近/前进轴。
  - 该约定与 Piper direct replay / robot-frame AnyGrasp 一致，但不同于原始 ALOHA-AgileX local `+X` 指尖深度约定。
  - 如需严格 ALOHA-style 对比，后续应新增 local-X 分支并配合 planner `--approach_axis local_x`。

## 2026-06-02（Mode O gripper frame 验证命令）

- 新增可视化入口：
  - `code_painting/visualize_mode_o_gripper_frame_conventions.py`
- 新增 wrapper 参数：
  - `--target_frame_convention piper_local_z|aloha_local_x_y_up|aloha_local_x_z_up`
  - `--plan_only`
- 静态可视化命令：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python /home/zaijia001/ssd/RoboTwin/code_painting/visualize_mode_o_gripper_frame_conventions.py --video_id 0 --foundation_frame 0 --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/mode_o_frame_convention_debug`
- ALOHA-style local-X plan-only 对照：
  - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 0 --plan_only --target_frame_convention aloha_local_x_z_up --output_root /tmp/mode_o_aloha_local_x_plan_only`

## 2026-06-02（O.0 原始 demo_clean Piper 数据生成命令）

- 新增任务名：
  - `pick_diverse_bottles_piper`
- 新增配置：
  - `demo_clean_piper`
- 新增命令：
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0`
- 对照命令：
  - `bash collect_data.sh pick_diverse_bottles demo_clean 0`
- 代码位置：
  - `envs/pick_diverse_bottles_piper.py`
  - `task_config/demo_clean_piper.yml`
  - `description/task_instruction/pick_diverse_bottles_piper.json`
- 说明：O.0 走原始 RoboTwin demo 数据生成链路，不走 FoundationPose / AnyGrasp / replay target frame。

## 2026-06-02（修正 O.0 collect_data 完整命令）

- 修正配置：
  - `task_config/demo_clean_piper.yml` 从 `embodiment: [piper]` 改为 `embodiment: [piper, piper, 0.60]`。
  - 原因：`[piper]` 会让 RoboTwin 查找不存在的 `assets/embodiments/piper/curobo_left.yml`；三元配置会加载两只单臂 Piper 并使用 `curobo.yml`。
- 推荐完整命令：
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0`
- 旧错误：
  - 在 `~` 下直接执行会找不到 `collect_data.sh`。
  - 不激活 `RoboTwin_bw` 时，`collect_data.sh` 内部的 `python` 可能找不到。

## 2026-06-03（O.0 标定 Piper/Pika 数据生成命令）

- 新增 embodiment：
  - `piper_pika_agx_calibrated`
- 新增/修改配置：
  - `assets/embodiments/piper_pika_agx/config.yml`
  - `task_config/demo_clean_piper.yml`
  - `task_config/demo_clean_piper_calibrated.yml`
  - `task_config/_embodiment_config.yml`
- 推荐命令：
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0`
- 说明：
  - 旧 `demo_clean_piper` 输出目录已经包含使用 RoboTwin 自带 `assets/embodiments/piper/piper.urdf` 生成的数据。
  - `demo_clean_piper_calibrated` 输出到新目录，使用标定 `piper_pika_agx.urdf` 和左右 base pose。

## 2026-06-03（O.0 head-only 与 viewer 调试配置）

- 修改配置：
  - `task_config/demo_clean_piper.yml`
  - `task_config/demo_clean_piper_calibrated.yml`
  - `task_config/demo_clean_piper_calibrated_viewer.yml`
- 推荐采集命令：
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0`
- 单 episode viewer/head-only 调试命令：
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50`
- 参数变化：
  - 正式采集与 viewer 配置均为 `collect_head_camera: true`、`collect_wrist_camera: false`。
  - viewer 配置额外设置 `render_freq: 1`、`episode_num: 1`、`collect_data: false`。
  - 该 viewer 配置只用于观察 seed/premotion，不保存 hdf5。
  - `run_view_pick_diverse_bottles_piper_scene.sh` 不进入 `play_once` 规划，会自动跳过不稳定 seed，用于纯场景 viewer 检查。
  - `run_collect_piper_calibrated_viewer.sh` 不再调用不存在的 `script/.update_path.sh`，但它仍会进入原始 demo 规划，不作为首选 viewer 命令。

## 2026-06-03（O.0 viewer 完成语义与 no-viewer 生成命令补充）

- viewer 命令说明更新：
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50`
  - 该命令只查看场景，会停在窗口循环，关闭窗口或 `Ctrl-C` 才退出。
- no-viewer 生成命令说明更新：
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0`
  - 该命令当前会进入原始 `play_once/grasp_actor`，但 `tmux gen1` 显示仍因瓶子不稳定和 `target_pose cannot be None for move action` 无法完成 episode。

## 2026-06-03（Mode M/N viewer 命令 CUDA mask 语义）

- 相关命令：
  - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh --gpu 2 --ids <ID> --viewer --tasks <TASK> ...`
  - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh --gpu 2 --ids <ID> --viewer --tasks <TASK> ...`
- 变更：
  - viewer 模式下，bash wrapper 和 Python 中间层都会移除传给 planner 的 `CUDA_VISIBLE_DEVICES`。
  - 非 viewer 模式不变，仍按 `--gpu` 设置计算 GPU。
- 使用说明：
  - 如果最小 `probe_sapien_viewer.py` 在 `unset CUDA_VISIBLE_DEVICES` 后能打开 viewer，Mode M/N viewer 命令也应在同一图形终端中打开 viewer。
  - 若仍失败，先检查日志 `[viewer] creating interactive viewer ...` 中的 `DISPLAY` 与 `CUDA_VISIBLE_DEVICES`。
- 相关代码：
  - `code_painting/plan_keyframes_human_replay.py`
  - `code_painting/plan_keyframes_foundation_pose.py`
  - `code_painting/run_plan_keyframes_human_replay_piper_d435.sh`
  - `code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh`

## 2026-06-03（O.0 motion baseline 数据生成与带运动 viewer 命令）

- 新增任务/配置：
  - `envs/pick_diverse_bottles_piper_motion.py`
  - `task_config/demo_clean_piper_motion.yml`
  - `task_config/demo_clean_piper_motion_viewer.yml`
  - `description/task_instruction/pick_diverse_bottles_piper_motion.json`
  - `run_pick_diverse_bottles_piper_motion_viewer.sh`
- 已跑通的无 viewer/head-only 数据生成命令：
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_motion demo_clean_piper_motion 0`
- 已跑通结果：
  - seed 0/1 因 `Objects is unstable` 跳过，seed 2 成功。
  - 输出 `data/pick_diverse_bottles_piper_motion/demo_clean_piper_motion/data/episode0.hdf5`、`video/episode0.mp4`、`_traj_data/episode0.pkl` 和 `instructions/episode0.json`。
- 带运动 viewer 命令：
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_pick_diverse_bottles_piper_motion_viewer.sh`
- 使用说明：
  - 该命令进入 `pick_diverse_bottles_piper_motion`，不是纯 scene viewer；在 `tmux gen1-1` 已跑到 seed 2 premotion。
  - 原 `run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50` 仍保留为只看稳定场景的 viewer，不会执行动作。
  - 原 `collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0` 仍会失败于原始 `grasp_actor`，不是本次推荐的可跑通 O.0 运动数据命令。
