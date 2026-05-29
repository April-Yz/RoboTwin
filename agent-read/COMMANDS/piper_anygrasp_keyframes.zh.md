# Piper AnyGrasp 两关键帧规划

## 用途

对 Piper H2O 数据使用 AnyGrasp rank preview 中的两个关键帧候选做双臂执行。当前推荐入口是：

```text
code_painting/plan_anygrasp_keyframes_piper.py
```

## 适用版本

PiperPika AGX 双单臂配置，推荐使用：

```text
robot_config_PiperPika_agx_dual_table_0515.json
```

## 关键实现

- 入口复用 `PiperDualReplayRenderer`，保留左右臂各自 base pose。
- URDFIK 使用 `HandRetargetPiperDualURDFIKRenderer`。
- IK 前按 arm 调用 `world_pose_to_base_pose_for_arm(...)`，避免沿用 R1 单 base 坐标转换。
- 推荐 `cartesian_interp_ik`，先生成 TCP 中间 waypoint，再逐点求 IK。
- 当前 Piper AnyGrasp 执行会在每个 stage 尝试规划到 pregrasp/grasp/action。若第 38 帧 grasp 未到位，脚本仍会继续 close/action，因此第 78 帧通常会更不到位；先用较宽旋转容差确认位置可达，再调 gripper 朝向。
- `--dual_stage_require_all_plans 1` 会让双臂 stage 必须左右臂都规划成功才执行，避免一只手 IK 失败时另一只手先单独运动。
- `--require_keyframe1_reached_before_action 1` 会在第一关键帧 grasp 未 reached 时跳过第二关键帧 action，避免“第一关键帧没到位就开始下一关键帧”。
- 使用 `--reuse_preview_frame_mode annotated_json_keyframes` 时不要再写 `--keyframes 38 78`；脚本会读取当前 id 的 `frame_selection.annotated_keyframes[:2]`。

## 命令格式

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && CUDA_VISIBLE_DEVICES=<GPU> conda run -n RoboTwin_bw python code_painting/plan_anygrasp_keyframes_piper.py \
  --anygrasp_dir <ANYGRASP_ID_DIR> \
  --replay_dir <FOUNDATION_REPLAY_ID_DIR> \
  --hand_npz <HAND_DETECTIONS_ID_NPZ> \
  --output_dir <OUTPUT_DIR> \
  --reuse_preview_summary_json <PREVIEW_SUMMARY_JSON> \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --arm auto \
  --execute_both_arms 1 \
  --dual_stage_require_all_plans 1 \
  --require_keyframe1_reached_before_action 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps -1 \
  --urdfik_cartesian_interp_auto_step_m 0.02 \
  --urdfik_max_position_threshold_m 0.02 \
  --urdfik_max_rotation_threshold_rad 0.12 \
  --reach_rot_tol_deg 180 \
  --debug_visualize_ik_waypoints 1 \
  --vscode_compatible_video 1
```

## 重要参数

- `--reuse_preview_summary_json`: 必须和当前 id 对应，不能用 id2 的 preview 判断 id0 的执行。
- D435 链路中 `--reuse_preview_summary_json` 必须指向 `code_painting/anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json`，并且 `--replay_dir` 必须指向 `foundation_replay_d435/foundation_input_<ID>`；不要 fallback 到默认广角 `anygrasp_h2o_preview`。
- `--reuse_preview_frame_mode annotated_json_keyframes`: 使用人工标注 keyframes。
- `--keyframes`: 在 annotated preview 复用模式下不要传；标注关键帧来自 preview summary。
- `frame_selection.effective_keyframes_by_arm`: 当标注 JSON 的全局 `keyframes` 不足两帧时，preview 会用 `left_keyframes/right_keyframes` 和全局 `keyframes` 为每只手补足两帧；planner 复用 summary 时会按 arm 使用这里的两帧。
- `--dual_stage_require_all_plans 1`: 双臂同步 stage 中任一 arm 规划失败时，两个 arm 都不执行该 stage。
- `--require_keyframe1_reached_before_action 1`: 第一关键帧 grasp 未到位时不进入第二关键帧 action。
- `--urdfik_trajectory_mode cartesian_interp_ik`: 使用 TCP waypoint 逐点 IK。
- `--urdfik_cartesian_interp_auto_step_m 0.02`: 自动 waypoint 间距。`0.01` 更密但更容易因为某个中间点 IK 阈值失败。
- `--urdfik_max_position_threshold_m 0.02`: Piper 调试时允许 URDFIK 最多放宽到 2 cm，避免 5 mm-2 cm 的可接受中间点被硬判失败。
- `--urdfik_max_rotation_threshold_rad 0.12`: 允许 IK 内部旋转阈值放宽到约 6.9 度；执行是否算 reached 仍由 `--reach_rot_tol_deg` 判断。
- `--reach_rot_tol_deg 180`: 位置优先调试用。确认位置能到后，再逐步收紧到 60/40/20 并调整候选朝向。
- `--debug_visualize_ik_waypoints 1`: 0/1 开关，输出中间 waypoint 可视化，适合调试目标是否可达。
- `--head_only 0 --third_person_view 1 --vscode_compatible_video 1`: 同时输出 head/third，并自动转码成 VS Code 可直接预览的 H.264/yuv420p/faststart。
- `--candidate_target_local_x_offset_m`: AnyGrasp 目标沿 gripper local X 的补偿。
- `--approach_offset_m`: pregrasp 相对 grasp 的 local X 后退距离。

## D435 版本

D435 候选选择先运行 `COMMAND_LIBRARY.zh.md` 的 `J1.1` 或 `J1.2`，生成：

```text
code_painting/anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json
```

下游 planner 使用 `L15.3`。该命令和默认广角 L15/L15.2 的核心区别是：

- `--replay_dir` 使用 `/home/zaijia001/ssd/data/piper/hand/<TASK>/foundation_replay_d435/foundation_input_<ID>`。
- `--reuse_preview_summary_json` 使用 `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json`。
- 相机参数使用 `--image_width 640 --image_height 480 --fovy_deg 42.499880046655484`，匹配 D435 color replay。
- 如果 D435 summary 缺失，应先重跑 J1.1/J1.2，而不是复用默认广角 summary。

当前六任务的 D435 基础输入链路是：

```text
AnyGrasp grasps + foundation_replay_d435/head_anygrasp_frames + HaMeR hand_detections
  -> J1.1/J1.2 生成 anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json
  -> L15.3 复用 D435 summary 和 D435 replay 执行 planner
```

如果 J0.1 粗扫看到很多 `MISS`，先确认是否只是 `seq 0 120` 超过了任务真实 episode 数。当前本机基础输入交集约为：`pick_diverse_bottles=102`、`place_bread_basket=92`、`stack_cups=47`、`handover_bottle=47`、`pnp_bread=72`、`pnp_tray=51`。其中 `place_bread_basket/stack_cups/handover_bottle/pnp_bread` 之前没有生成 D435 summary 的主要原因是旧 wrapper 只看全局 `keyframes`，没有使用 `left_keyframes/right_keyframes`；现在已经改为支持 per-arm effective keyframes。

六任务 D435 planner 推荐入口是：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 1
```

不要在 zsh 中直接粘贴旧版含 `mapfile` 的长命令；`mapfile` 是 bash 内建。需要全量运行时去掉 `--max_per_task 1`。如果是六任务批量摸底，建议加 `--continue_on_error`，避免单个 id 崩溃后中断后续任务。

每个任务前 5 个 id 的测试入口：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error
```

`stack_cups id0` 的 viewer 单条调试命令见 `COMMAND_LIBRARY.zh.md` L15.5。该命令使用 `unset CUDA_VISIBLE_DEVICES` 打开 SAPIEN viewer，并保留 D435 summary/replay 路径。如果 viewer 探针打印 `DISPLAY=` 为空并报 `Renderer does not support display`，需要从图形终端或正确 X11/Wayland forwarding 环境运行。

L15.6 增加了统一的 viewer/no-viewer 与第一关键帧 debug 入口：

```bash
# 无 viewer：六任务各跑前 5 个 summary
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v1

# viewer：必须在有 DISPLAY 的图形环境中运行
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_viewer

# viewer 运动确认：更接近旧 R1/V7 观感的 joint_interp
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --trajectory_mode joint_interp --joint_interp_waypoints 40 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_viewer_joint_interp

# 第一关键帧 debug：只执行 init -> pregrasp -> grasp，不关爪、不进入第二关键帧
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --debug_stop_after_keyframe1 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_keyframe1_debug
```

脚本默认传 `--require_keyframe1_reached_before_close 1` 与 `--require_keyframe1_reached_before_action 1`。第一关键帧 grasp 未 reached 时不会关夹爪，也不会进入第二关键帧。

`--debug_stop_after_keyframe1` 用于隔离第一关键帧可达性。`stack_cups id0` 的实测失败发生在第一关键帧 cartesian waypoint IK：pregrasp 左臂 waypoint 13/23、右臂 28/48 失败；grasp 左臂 16/28、右臂 25/45 失败。由于 `--dual_stage_require_all_plans 1`，任意 arm plan 失败都会跳过双臂 stage，所以视频中会表现为没有执行 waypoint。`execute_interp_steps/joint_command_scene_steps/settle_steps/joint_target_wait_steps` 只在 plan 已经 `Success` 后才生效；如果已经 `[plan-fail]`，要改的是轨迹模式、候选目标或 IK 参数。

脚本当前默认执行节奏为旧 R1/V7 风格：`execute_interp_steps=24`、`joint_command_scene_steps=10`、`settle_steps=30`、`joint_target_wait_steps=25`。不要同时把 `execute_interp_steps` 和 `joint_command_scene_steps` 设到几千；这会让 viewer 看起来像卡在 waypoint。

终端 TCP/EE 位置变化检查：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 1 --continue_on_error --tasks pick_diverse_bottles --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_posemove_debug
```

`--print_pose_every 5` 会输出 `[exec-pose]`，包含 left/right 的 TCP 和 EE world position。`--allow_partial_dual_stage` 只用于诊断：它允许右臂 plan 成功时先运动，从而确认执行链会改变 pose。最终数据仍建议保持严格双臂同步。

L15.7 记录了当前 D435/Piper 关键修正：

- 六任务 wrapper 默认 `--reach_error_pose_source ee`。Piper AnyGrasp 目标送入 IK 时使用 wrist/endlink 约定；如果用 `tcp` 判定 reached，会固定留下约 12 cm TCP/EE 偏移。
- `target_pose_for_error(..., ee)` 对双 Piper 按 arm 使用独立 base，避免右臂目标被转换到左臂 base。
- partial 诊断模式会 hold 住 plan 失败的 arm，避免另一只 arm 执行时失败臂漂移。
- 当前关键帧流程是：复用 D435 summary 的 per-arm effective keyframes -> 第一关键帧 `pregrasp -> grasp` -> grasp reached 后才 close gripper 和执行第二关键帧 action。
- `pick_diverse_bottles id0` 复查：右臂在 `ee` 判定下 grasp 位置误差约 `0.0057m`，说明执行链和 waypoint 是能动的；整体失败来自左臂第一关键帧 IK/目标失败，严格双臂同步因此阻断 stage。
- wrapper 支持 `--visualize_targets`，用于 viewer 调试时显示目标 axis 和 active frame 候选 gripper；该开关会自动关闭 `pure_scene_output`。
- planner 会把复用 summary 的原始最佳候选图复制到 `<OUT>/source_preview_compare/`，并写 `selected_candidate_mapping.json` 记录 `candidate_idx/rank/raw_pose/target_pose/local_x_offset`。

六任务分别跑前 5 个的严格同步入口见 `COMMAND_LIBRARY.zh.md` L15.7，典型形式如下：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pick_diverse_bottles --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/strict-ee
```

六任务 partial 诊断入口同样见 L15.7，典型形式如下：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pick_diverse_bottles --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/partial-ee
```

viewer 目标可视化示例：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --visualize_targets --tasks pick_diverse_bottles --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-partial-ee
```

## 输出

```text
<OUTPUT_DIR>/plan_summary.json
<OUTPUT_DIR>/rank_previews/
<OUTPUT_DIR>/debug_execution_metrics.jsonl
<OUTPUT_DIR>/pose_debug.jsonl
<OUTPUT_DIR>/head_cam_plan.mp4
<OUTPUT_DIR>/third_cam_plan.mp4
<OUTPUT_DIR>/debug_execution_preview.mp4
```

## 检查命令

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('<OUTPUT_DIR>/plan_summary.json')
d = json.load(open(p))
print('execution_success:', d.get('execution_success'))
print('execution_failed:', d.get('execution_failed'))
print('failed_stage_records:', d.get('failed_stage_records'))
print('rank_preview_images:', d.get('rank_preview_images'))
PY
```

## 常见问题

- 如果 `plan_solution_by_arm` 为空，通常表示 IK 没有解，不是已经规划到了错误位置。
- `--settle_steps` 和 `--joint_target_wait_steps` 只等待已有关节目标收敛，不能修复 IK 无解或目标姿态不可达。
- 如果右臂目标明显偏到左侧，优先检查是否走了 Piper dual renderer，而不是旧 R1 单 base wrapper。
- 如果 rank preview 看起来正确但执行失败，先确认 preview summary、anygrasp dir、replay dir、hand npz 都是同一个 id。
- 如果第 38 帧只执行一部分，第 78 帧更不到位，优先看 `failed_stage_records`：第 38 帧 grasp 没到位后仍会继续进入 close/action，第 78 帧是在错误机器人状态上重新规划。

## L15.8 Preview 与 Planner 映射一致性

本轮确认了一个真实的显示/执行不一致：`render_anygrasp_ranked_preview.py` 之前在 summary 的 `translation_world` 中应用了 `--candidate_target_local_x_offset_m -0.05`，但预览图上的 grasp wireframe 仍然画原始 AnyGrasp `translation_cam/rotation_matrix`。planner 复用 summary 时使用的是 offset 后的 world target，所以会看到 planner/rank preview 的夹爪比源 preview 靠后。

现在 preview 绘制已改为使用同一套 remap/post-rot/local-X offset 后的 camera-frame target pose。summary 额外记录 `translation_cam`、`visual_translation_cam`、`translation_world`、`rotation_matrix`、`visual_rotation_matrix`。`translation_cam` 是原始 AnyGrasp 候选；`visual_translation_cam` 是实际绘制和 planner target 对应的相机坐标。

D435 planner 必须复用 `code_painting/anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json`，不要用默认广角 `code_painting/anygrasp_h2o_preview/<TASK>/foundation_input_<ID>/summary.json`。`pick_diverse_bottles id0 frame 38` 的 D435 rank1 是 left candidate `16`、right candidate `11`；默认广角 preview 的 rank1 candidate id 不同，因此不能跨 preview root 对图。

重新生成六任务 D435 preview：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do case "$TASK" in pick_diverse_bottles) LEFT_OBJ=left_bottle; RIGHT_OBJ=right_bottle ;; place_bread_basket) LEFT_OBJ=basket; RIGHT_OBJ=bread ;; stack_cups) LEFT_OBJ=left_light_pink_cup; RIGHT_OBJ=right_dark_red_cup ;; handover_bottle) LEFT_OBJ=right_bottle; RIGHT_OBJ=right_bottle ;; pnp_bread) LEFT_OBJ=left_bread; RIGHT_OBJ=right_bread ;; pnp_tray) LEFT_OBJ=left_dark_red_cup; RIGHT_OBJ=right_bottle ;; esac; ANN=/home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json; ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output; [[ -d "$ANY_ROOT" ]] || ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output_old_cam; REPLAY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435; HAND_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output; OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/${TASK}; [[ -f "$ANN" ]] || { echo "[skip] task=${TASK} missing annotation $ANN"; continue; }; [[ -d "$ANY_ROOT" ]] || { echo "[skip] task=${TASK} missing ANY_ROOT=$ANY_ROOT"; continue; }; [[ -d "$REPLAY_ROOT" ]] || { echo "[skip] task=${TASK} missing REPLAY_ROOT=$REPLAY_ROOT"; continue; }; VIDEO_PREFIX=foundation_input CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh "$ANY_ROOT" "$REPLAY_ROOT" "$HAND_ROOT" "$OUT_ROOT" --hand_keyframes_json "$ANN" --left_target_object "$LEFT_OBJ" --right_target_object "$RIGHT_OBJ" --anygrasp_score_weight 0.25 --orientation_score_weight 0.75 --max_rotation_distance_deg 90 --candidate_target_local_x_offset_m -0.05 --draw_object_overlay 1 --draw_hand_reference 1 --debug_dump_object_distances 1 --top_k 20 --camera_cv_axis_mode legacy_r1; done
```

planner/replay 无 viewer：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/strict-ee-offsetfix
```

viewer 目标可视化：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --visualize_targets --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-partial-ee-offsetfix
```

planner 会在 `<OUT>/source_preview_compare/selected_candidate_mapping.json` 中记录 source summary、raw pose、offset 后 target pose。`planner_raw_pose_world_wxyz` 与 `planner_target_pose_world_wxyz` 相差约 5cm 时，是 `--candidate_target_local_x_offset_m -0.05` 的预期效果。

## L15.9 复制安全版三步运行

`COMMAND_LIBRARY.zh.md` 末尾新增 L15.9，提供按顺序运行的 3 段复制安全命令：

1. 用 `bash <<'BASH'` 重新生成六任务 D435 AnyGrasp no-offset preview/summary 到主目录 `anygrasp_h2o_preview_d435`，避免 zsh 断行导致参数丢失。
2. 无 viewer 跑六任务 D435 planner/replay，输出到 `anygrasp_plan_keyframes_piper_d435_v2/strict-ee-offsetfix`。
3. viewer 跑六任务 D435 planner/replay，并打开 `--visualize_targets` 显示 gripper 目标。

运行第 1 步时必须确认日志中的 replay root 是：

```text
/home/zaijia001/ssd/data/piper/hand/<TASK>/foundation_replay_d435
```

如果出现 `replay_m_obj_pose_d_pour_blue_norobot`，说明 shell 复制/换行仍然导致脚本没有收到 D435 replay 参数。

L15.9 的 preview 说明：

- `orientation_rank.png` 只按手部朝向相似度排序；当前 downstream planner 默认使用 `orientation rank1`。
- `fused_rank.png` 按 `anygrasp_score * 0.25 + orientation_score * 0.75` 排序，仅作为参考。
- `planner_selected_orientation_rank1.png` 只画当前 planner 默认会选择的 `orientation rank1`。
- 候选 AnyGrasp gripper：左手蓝色系，右手橙色系。
- 人手参考 gripper：左手绿色，右手紫色。

## L15.10 Offset -5cm 对比命令

`COMMAND_LIBRARY.zh.md` 末尾新增 L15.10，提供 offset -5cm 对比版命令。它把 `--candidate_target_local_x_offset_m` 改为 `-0.05`，输出到：

```text
code_painting/anygrasp_h2o_preview_d435_offset_minus_5cm_compare/<TASK>/foundation_input_<ID>
```

用途是和 L15.9 的主目录 no-offset preview 做视觉和 summary 对比，确认 5cm local-X 补偿是否就是“夹爪靠后”的来源。该对比目录不覆盖主 `anygrasp_h2o_preview_d435`。

注意：如果要让 preview 和 planner 的最终执行 target 完全一致，preview 和 planner 的 `--candidate_target_local_x_offset_m` 必须相同。当前主目录按用户要求保存 no-offset 原始候选。

## L15.11 Cartesian Partial Prefix 执行

`COMMAND_LIBRARY.zh.md` 末尾新增 L15.11，提供 `--execute_partial_cartesian_plan` 调试入口。该开关只对 `--trajectory_mode cartesian_interp_ik` 生效：如果中间 Cartesian waypoint IK 失败，但前面的 waypoint 已经求解成功，则返回 `status=Partial` 并执行成功前缀到最后一个可达 waypoint。

用途：

- 避免 viewer 中因为整段 plan 判 `Fail` 而完全不运动。
- 观察 arm 实际可以沿目标方向走到哪里。
- 配合 `[plan-partial] failed_waypoint=N/M solved_prefix=K` 和 `[exec-pose]` 定位不可达点。

限制：

- `Partial` 会执行，但不会算 reached。
- 第一关键帧没有 reached 时，close/action guard 仍会阻止关爪和进入第二关键帧。
- `joint_interp` 没有 Cartesian waypoint 前缀，因此不触发该逻辑。

位置优先/朝向优先说明也记录在 L15.11：当前 reached 判定可以用 `--reach_rot_tol_deg 180` 做位置优先调试，但 IK 求解本身仍同时约束位置和朝向。真正的 IK 位置优先需要后续增加分级 IK 策略，例如先完整姿态，失败后放宽 rotation threshold，再失败后只保留位置或采样朝向。

## L15.12 Piper 夹爪轴修正与位置优先诊断

`COMMAND_LIBRARY.zh.md` 末尾新增 L15.12，记录 Piper D435 AnyGrasp planner 当前使用的局部轴约定：

```text
Preview gripper frame:
  local +X = approach / forward axis = rotation_matrix[:, 0]

Piper reported visible gripper frame:
  R_report = R_link6 @ global_trans_matrix @ delta_matrix

Current Piper config:
  global_trans_matrix = diag(1, -1, -1)
  delta_matrix = I
```

送入 URDFIK 的 link6 目标必须使用：

```text
R_link6_target = R_preview_gripper @ inv(global_trans_matrix @ delta_matrix)
```

旧 Piper URDFIK 路径只反掉了 `delta_matrix`，没有反掉 `global_trans_matrix`，所以 viewer 中实际渲染的 gripper 可能相对 preview 绕 local +X 翻转。现在 `render_hand_retarget_piper_dual_npz_urdfik.py` 的 `_target_tcp_world_to_ee_base()` 已补上该逆变换。Piper dual 的 `reach_error_pose_source=ee` 也保持可见 gripper 坐标系，不再把目标转成 raw link6 坐标系。

六任务 wrapper 新增：

```text
--ik_max_position_threshold_m <meters>
--ik_max_rotation_threshold_rad <radians>
```

这用于位置优先诊断。默认 `--ik_max_rotation_threshold_rad 0.12` 下，`pick_diverse_bottles id0` 仍在第一个 Cartesian waypoint IK 被完整姿态约束拒绝；改成 `--ik_max_rotation_threshold_rad 3.14` 后 smoke test 出现连续 `[exec-pose]`，说明 waypoint 执行链是通的，静止来自 IK 拒绝，不是 `settle_steps` 或 `joint_target_wait_steps` 太小。

partial Cartesian 模式还增加了首个 waypoint 失败时的小步 fallback：在当前 pose 到第一个失败 waypoint 之间采样更短的小步，执行最远的可解小步。若仍然不动，说明当前完整姿态约束下连第一小步也不可解。

viewer 批跑行为也已调整：`--viewer` 默认使用 `--viewer_wait_at_end 0`，一个 id 结束后自动进入下一个 id；如果需要单 id 结束后停住检查，再显式加 `--viewer_wait_at_end 1`。L15.12 的 viewer 轴检查命令现在显式加入 `--ik_max_rotation_threshold_rad 3.14`，避免用户误用默认 `0.12rad` 时看到完全不运动。

## L15.13 Viewer 轴检查与 id 范围命令

`COMMAND_LIBRARY.zh.md` 末尾新增 L15.13，补充六任务分别跑 id0-10 的 viewer 命令。wrapper 新增：

```text
--id_start <ID>
--id_end <ID>
--ids <ID...>
--piper_apply_global_trans_to_ik <0|1>
```

默认 `--piper_apply_global_trans_to_ik 0`，保持和 direct Piper hand replay 相同的 URDFIK 约定；`1` 只用于对照诊断。

viewer 轴判断规则：

- preview gripper wireframe 的 local +X 是 `rotation_matrix[:, 0]`，视觉上是从掌根横杆指向两根手指指尖/缺口方向。
- viewer target 坐标轴 actor：红色 = local +X，绿色 = local +Y，蓝色 = local +Z。
- 判断 preview target 和 viewer target 是否一致时，优先比较 preview 手指方向与 viewer 红轴。
- Piper mesh/link 的 Y/Z 可能因 `global_trans_matrix=diag(1,-1,-1)` 和 target 坐标轴看起来相反；不要用绿/蓝轴单独判断 target 错误。

## L15.14 Viewer 目标夹爪按左右手关键帧显示

用户观察到 L15.13 的 viewer 中 gripper target 像是后一帧，第一关键帧没有成功显示。根因不是 D435 preview candidate 本身，而是执行预览状态之前只有一个全局 `active_frame`：双臂模式下它取左手帧，`record_frame()` 和 candidate debug actor 也只按这个单帧刷新。左右手 effective keyframes 不同或阶段切换后，就会出现 viewer 目标显示成错误帧的现象。

修正后：

- `DebugExecutionState` 增加 `active_frame_by_arm`。
- `pregrasp/grasp` 使用每只手的第一关键帧，`action` 使用每只手的第二关键帧。
- `record_frame()`、`update_candidate_debug_visuals()`、debug execution preview 都按 `active_frame_by_arm` 显示候选和 selected gripper。
- `pose_debug.jsonl` 与 `execution_metrics.jsonl` 记录 `active_frame_by_arm`，可用 `jq` 检查当前 viewer 显示帧。

复查命令：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --tasks pick_diverse_bottles --id_start 0 --id_end 10 --continue_on_error --viewer --visualize_targets --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10
```

实际运行后检查：

```bash
jq -c '{stage,active_frame,active_frame_by_arm}' /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10/pick_diverse_bottles/foundation_input_0/pose_debug.jsonl | head -n 20
```

如果当前 shell 没有 `jq`，使用：

```bash
head -n 20 /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10/pick_diverse_bottles/foundation_input_0/pose_debug.jsonl | sed -E 's/.*"active_frame": ([^,]+), "active_frame_by_arm": \{([^}]*)\}, "stage": "([^"]+)".*/stage=\3 active_frame=\1 active_frame_by_arm={\2}/'
```

## L15.15 Stack Cups id0 无碰撞 target-only 调试

新增 wrapper 参数：

- `--disable_execution_collisions`：传入 planner 的 `--enable_grasp_action_object_collision 0`，关闭 grasp/action 物体碰撞和 contact-stop close 逻辑。
- `--target_axes_only`：自动打开 `--visualize_targets`，同时隐藏候选 gripper 轴、selected-keyframe 轴和 IK waypoint marker。
- `--debug_candidate_top_k`、`--debug_common_candidate_top_k`、`--debug_visualize_selected_keyframe_axes`、`--debug_visualize_ik_waypoints`：细粒度控制调试 actor。

`stack_cups id0` 会看到多套坐标系的原因：

- D435 summary 使用 per-arm keyframes：左手第一关键帧为 `139`，右手第一关键帧为 `51`。
- viewer 可能同时画当前执行 target 轴、selected-keyframe 轴、candidate gripper 轴和 IK waypoint marker。
- 如果只想判断实际执行 target 轴，使用 `--target_axes_only`；此时红轴仍是 target local +X。

viewer 调试命令：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --tasks stack_cups --ids 0 --continue_on_error --viewer --target_axes_only --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/stackcups-id0-nocollision-targetaxes
```

无 viewer、只调第一关键帧：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --tasks stack_cups --ids 0 --continue_on_error --target_axes_only --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 20 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --debug_stop_after_keyframe1 --output_root /tmp/stack_cups_id0_no_collision_target_axes_only
```

本机无 viewer 实测确认 `enable_grasp_action_object_collision=0`，但 `stack_cups id0` 仍未到位。pregrasp 末端误差约 left `0.386m/147deg`、right `0.337m/82deg`；grasp 仍失败或 partial。这个结果说明当前主要问题不是物体碰撞，而是 IK 解和实际关节跟踪/执行后偏差较大。

## L15.16 Direct Piper Hand Replay Viewer 对照

`COMMAND_LIBRARY.zh.md` 末尾新增 L15.16，用于回看直接 replay 人手 gripper pose 的 Piper URDFIK 执行效果。该命令不走 AnyGrasp candidate 选择，只读取 `hand_detections_0.npz` 中存好的 gripper pose。

关键参数：

- `--debug_visualize_targets 1`：viewer 中显示每一帧目标 gripper 坐标轴。
- `--debug_mode 1 --debug_post_execute 1`：打印每帧执行后的 target 与实际 TCP/EE 误差。
- `--save_world_targets 1`：保存 `world_targets_and_status.npz`，用于检查 target pose 和状态字段。
- `--enable_viewer 1 --viewer_wait_at_end 1`：打开 viewer 并在结束后停住。

推荐先用 `stack_cups id0` 对照 AnyGrasp planner：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output/hand_detections_0.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/direct_replay_debug_piper_d435/stack_cups/id0_viewer_axes --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --fps 5 --frame_start 0 --frame_end 220 --max_frames 221 --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m 0.05 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 1 --debug_post_execute 1 --debug_frame_limit -1 --debug_visualize_targets 1 --debug_target_axis_length 0.10 --debug_visualize_cameras 0 --save_world_targets 1 --clean_output 0 --overlay_text_enable 1 --save_png_frames 0 --lighting_mode front_no_shadow --enable_viewer 1 --viewer_frame_delay 0.02 --viewer_wait_at_end 1
```

## L15.17 Direct Replay 与 AnyGrasp 轴约定差异

direct Piper hand replay 和 AnyGrasp planner 当前不是同一个 gripper local frame：

- direct replay 的 stored gripper frame 使用 `local +Z` 作为 approach/forward 轴。`--target_local_forward_retreat_m` 也是沿蓝轴后退，运行时会打印 `along_local_plus_z_blue_m`。
- AnyGrasp preview 的 wireframe 使用 `rotation_matrix[:, 0]` 作为两根手指从掌根到指尖的 finger-depth 方向，即 local +X/红轴。
- 所以 direct replay 中“蓝轴是前进轴”与 AnyGrasp 中“红轴像是 gripper wireframe 指尖方向”并不矛盾；这是两套局部坐标系约定不同。

如果要测试把 AnyGrasp local +X 映射到 direct replay local +Z，可以在 planner 单条命令中加入：

```bash
--candidate_orientation_remap_label swap_red_blue
```

`COMMAND_LIBRARY.zh.md` L15.17 已补充 `stack_cups id0` 的完整无 viewer 对照命令。先无 viewer 生成 `pose_debug.jsonl` 和视频，确认蓝轴/执行误差是否更接近 direct replay；若方向相反，再测试 `swap_red_blue_keep_green` 或具体枚举标签。

## L15.18 Replay-Axis AnyGrasp Keyframe Planner

本节新增独立入口，不替换旧 AnyGrasp 命令：

```text
code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh
```

新增原因：direct Piper hand replay 的 stored gripper frame 使用 `local +Z` 蓝轴作为 approach/forward；AnyGrasp raw candidate 的 wireframe 使用 `local +X` 红轴作为掌根到指尖的 finger-depth。因此旧 planner 的 `identity + local-X offset/pregrasp` 不等价于 direct replay 的蓝轴逻辑。

新 wrapper 固定启用：

```text
--candidate_orientation_remap_label swap_red_blue
--candidate_target_local_x_offset_m 0.0
--candidate_target_local_z_offset_m -0.05
--approach_axis local_z
--approach_offset_m 0.12
```

含义：先把 AnyGrasp 原始 local +X 映射成执行 target 的 local +Z，再把 5cm target compensation 和 pregrasp retreat 都沿 local +Z 执行。旧六任务 wrapper 默认仍保留 local-X 逻辑。

六任务前 5 个 no-viewer 命令与 viewer 命令已写入 `COMMAND_LIBRARY.zh.md` L15.18。常用单任务 id0-10 viewer 调试：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --tasks stack_cups --id_start 0 --id_end 10 --continue_on_error --viewer --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/stack_cups_id0_10_viewer
```

验证记录：

- `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py` 通过。
- `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` 通过。
- `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh` 通过。
- 无 viewer 实测 `stack_cups id0 --debug_stop_after_keyframe1` 可运行并生成 `head_cam_plan.mp4` / `third_cam_plan.mp4`；`plan_summary.json` 中记录 `candidate_target_local_z_offset_m=-0.05`、`approach_axis=local_z`，keyframe1 grasp 左右手均 reached。
