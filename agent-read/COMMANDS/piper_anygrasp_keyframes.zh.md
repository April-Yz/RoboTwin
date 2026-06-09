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

## Mode M/N Viewer CUDA Mask 修复

Mode M `run_plan_keyframes_human_replay_piper_d435.sh` 和 Mode N `run_plan_keyframes_foundation_pose_piper_d435.sh` 的 viewer 命令需要同时满足两点：

- 当前终端有可用图形显示，例如 `DISPLAY=:1.0`。
- SAPIEN viewer 创建时不能被 `CUDA_VISIBLE_DEVICES=<compute_gpu>` 限制到非显示 GPU。

本次问题出现在第二点：bash wrapper 已经在 `--viewer` 时 `env -u CUDA_VISIBLE_DEVICES`，但 Python 中间层 `plan_keyframes_human_replay.py` / `plan_keyframes_foundation_pose.py` 在调用 planner 前又按 `--gpu` 写回了 `CUDA_VISIBLE_DEVICES=2`，导致日志中出现：

```text
[viewer] creating interactive viewer ... CUDA_VISIBLE_DEVICES=2
[viewer-warning] failed to create interactive viewer ... Renderer does not support display.
```

当前修复为：`--enable_viewer 1` 时 Python 中间层在传给 planner 的 env 中移除 `CUDA_VISIBLE_DEVICES`；非 viewer 模式仍按 `--gpu` 设置计算 GPU。

最小 viewer 探针仍是：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES; [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; echo "DISPLAY=$DISPLAY CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}" && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/probe_sapien_viewer.py
```

验证命令：

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_keyframes_foundation_pose.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_keyframes_human_replay.py

timeout 60s bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh --gpu 2 --ids 0 --viewer --viewer_wait_at_end 0 --tasks pnp_tray --foundation_pose_retreat_m 0.03 --output_root /tmp/robotwin_viewer_env_probe
```

第二条在非图形 shell 中预期仍会因为 `DISPLAY=None` fallback 到 offscreen，但关键日志应显示 `CUDA_VISIBLE_DEVICES=None`，证明 Python 中间层不再重新设置 CUDA mask。

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

## L15.19 筛选阶段统一 gripper/robot frame 的设计记录

`COMMAND_LIBRARY.zh.md` 末尾新增 L15.19，只记录分析和当前对照命令，不修改代码。L15.18 的 `swap_red_blue + local_z` 旧指令保留。

核心分析：

- viewer 坐标轴颜色仍是红色 local +X、绿色 local +Y、蓝色 local +Z。
- rank preview 中蓝色/橙色 gripper 是左右手/候选 actor 颜色，不是坐标轴颜色；橙色通常表示右手 gripper/candidate。
- AnyGrasp C 形 gripper 的 raw local +X 是 C 平面内掌根到指尖方向，raw local +Y 是开合宽度，raw local +Z 是 C 平面侧向法线。
- 如果要长期解决 gripper 可视化方向和 robot target 方向不一致，应在候选筛选阶段就把 AnyGrasp raw frame 转成 canonical robot/replay frame，而不是只在 planner 执行阶段 remap。

建议的长期 frame：

```text
robot/replay target local +Z = AnyGrasp raw local +X
robot/replay target local +Y = AnyGrasp raw local +Y
robot/replay target local +X = -AnyGrasp raw local +Z
```

按该逻辑实现后，蓝色轴仍表示 local +Z；橙色仍只是右手 C gripper 颜色；右手橙色 C gripper 的指尖方向应与 robot target 蓝色 local +Z 对齐。

L15.19 当前对照命令仍使用 L15.18 wrapper，只把结果保存到：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
```

### L15.19.1 已实现 Robot-Frame Preview + Planner

新增代码入口：

```text
code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh
code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh
```

实现点：

- `render_anygrasp_ranked_preview.py` 新增 `--candidate_frame_mode robot_replay`。
- robot-frame summary 中 candidate rotation 满足 `target local +Z = AnyGrasp raw local +X`。
- preview 支持 `--candidate_target_local_z_offset_m`。
- planner C gripper actor 支持 `--debug_gripper_actor_forward_axis local_z`，让 C gripper 指尖方向沿蓝色 local +Z 显示。
- planner robot-frame wrapper 使用 `identity + local_z`，不再依赖执行阶段 `swap_red_blue`。
- `run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh` 会自动按当前 `--tasks`、`--ids`、`--id_start/--id_end`、`--max_per_task` 范围补齐缺失的 robot-frame preview summary，然后再运行 planner。
- 若只想使用已有 summary，不自动生成，可加 `--skip_preview_generation`。

可单独生成 robot-frame preview：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh --gpu 2 --tasks pick_diverse_bottles --ids 0
```

也可以直接运行 viewer_gripper planner；wrapper 会先补缺失 preview：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --viewer --tasks pick_diverse_bottles --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
```

### L15.19.2 Robot-Frame Planner 指定 id Viewer 命令

指定 episode 用 `--ids <ID>`；多个 episode 可写成 `--ids 0 4 8`。wrapper 会先补缺失的 robot-frame preview summary，再运行 planner。

`stack_cups id4`：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --ids 4 --continue_on_error --viewer --tasks stack_cups --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
```

六任务同一组 id：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --ids 0 1 2 3 4 --continue_on_error --viewer --tasks pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
```

## O. 第一帧 FoundationPose 直接策略抓取对照

### O.0 Piper/Pika 数据生成与 IK 诊断命令

O.0 现在只保留可复用的 4 条命令。不要再把 `pick_diverse_bottles_piper demo_clean_piper_calibrated` 当成推荐采集命令；它会进入原始 `grasp_actor`，目前在 `tmux gen1-1/gen1-2` 中持续失败。

最新 `tmux gen1-1/gen1-2` 结果：

```text
seed 72-115: Objects is unstable / target_pose cannot be None for move action
final: user interrupted with Ctrl-C
```

含义：原始 `pick_diverse_bottles.py` 的 `choose_grasp_pose/grasp_actor` 仍不能为标定 Piper/Pika 稳定生成可执行抓取 target。这不是 viewer 错误；`No left camera link` / `No right camera link` 只是当前 URDF 没有 wrist camera link 的 fallback 警告。当前 `piper_pika_agx.urdf` 没有 `left_camera`、`right_camera` 或 `camera` link，所以 O.0 配置保持 `collect_wrist_camera: false`，只保存 head 视角。

保留入口：

```text
envs/pick_diverse_bottles_piper.py
envs/pick_diverse_bottles_piper_motion.py
task_config/demo_clean_piper_motion.yml
task_config/demo_clean_piper_motion_viewer.yml
task_config/demo_clean_piper_ik_orig_tcp.yml
assets/embodiments/piper_pika_agx_ik_orig_tcp/config.yml
assets/embodiments/piper_pika_agx_ik_orig_tcp/curobo.yml
assets/embodiments/piper_pika_agx_ik_orig_tcp/collision_piper_pika.yml
description/task_instruction/pick_diverse_bottles_piper_motion.json
run_pick_diverse_bottles_piper_motion_viewer.sh
run_view_pick_diverse_bottles_piper_scene.sh
```

#### O.0-1 已跑通：无 viewer 生成 head-only 数据

用途：生成可保存的 O.0 对照数据。该命令走 `pick_diverse_bottles_piper_motion`，复用原始瓶子随机采样和稳定性检查，但不调用原始 IK；动作为标定 Piper/Pika 的关节空间 motion baseline。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_motion demo_clean_piper_motion 0
```

状态：已测试成功。输出示例：

```text
data/pick_diverse_bottles_piper_motion/demo_clean_piper_motion/data/episode0.hdf5
data/pick_diverse_bottles_piper_motion/demo_clean_piper_motion/video/episode0.mp4
data/pick_diverse_bottles_piper_motion/demo_clean_piper_motion/_traj_data/episode0.pkl
data/pick_diverse_bottles_piper_motion/demo_clean_piper_motion/instructions/episode0.json
```

#### O.0-2 已跑通：带运动 viewer 检查 motion baseline

用途：在 SAPIEN viewer 中看 O.0 motion baseline 的后续运动。该命令调用 `view_pick_diverse_bottles_piper_motion.py`，每次都会重新找稳定 seed 并执行一次 `play_once()`，不会被 `collect_data.py` 的旧 `seed.txt` 进度短路；`collect_data: false`，不保存 hdf5。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_pick_diverse_bottles_piper_motion_viewer.sh
```

无窗口 smoke 验证：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && DISPLAY=:1.0 timeout 120s bash run_pick_diverse_bottles_piper_motion_viewer.sh --seed 0 --max_seed_tries 3 --hold 0
```

状态：2026-06-04 已验证通过；seed 0/1 因瓶子不稳定被跳过，seed 2 加载后完成 `play_once()`。默认会显示 debug 坐标轴：红/绿/蓝分别是局部 +X/+Y/+Z，白色小方块只是每个坐标轴 marker 的原点，不代表 Piper base。当前会显示两个瓶子中心、左右放置目标、当前左右 `link6` EE，以及 `pregrasp/grasp_lower/lift/move_out` 的左右 EE 目标轴。

终端阶段日志：

```text
[piper-motion][setup] bottle ranges left=x[-0.30,-0.18],y[-0.20,-0.10] right=x[0.30,0.46],y[-0.20,-0.10]
[piper-motion][target-axis] pregrasp left_pos=... right_pos=...
[piper-motion][stage] pregrasp: planning joint interpolation
[piper-motion][stage] grasp_lower: planning joint interpolation
[piper-motion][stage] close_gripper: start/finished
[piper-motion][stage] lift: planning joint interpolation
[piper-motion][stage] move_out: planning joint interpolation
[piper-motion][stage] open_gripper: start/finished
```

说明：O.0 motion baseline 不再使用原始 ALOHA/AgileX 的瓶子范围 `y=[0.03,0.23]`，而是在 `envs/pick_diverse_bottles_piper_motion.py` 中覆盖为更靠近当前标定 Piper/Pika FK 工作区的范围。当前 FK 打印显示阶段 EE 目标仍在 `y≈-0.40~-0.47`，所以它仍是关节空间可视化 baseline，不是最终贴瓶抓取策略。

#### O.0-3 只看场景：纯 scene viewer，不执行动作

用途：只检查标定 Piper/Pika、桌面、瓶子随机摆放、目标坐标轴和 viewer 是否能打开。该命令不进入 `play_once`，不会执行运动，也不会保存数据；现在会通过 `skip_planner=True` 跳过 Curobo planner 初始化，避免 scene-only viewer 卡在 Curobo warmup。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50
```

无窗口 smoke 验证：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && DISPLAY=:1.0 timeout 90s python view_pick_diverse_bottles_piper_scene.py --seed 0 --max_seed_tries 3 --hold 0
```

状态：2026-06-04 已验证通过；seed 0/1 因瓶子不稳定被跳过，seed 2 加载稳定场景，添加坐标轴后渲染一帧退出。窗口打开后会停在渲染循环，关闭 SAPIEN 窗口或 `Ctrl-C` 才退出。

如果要只看 O.0 motion baseline 的场景和所有阶段目标轴，但不执行动作，用这个命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_motion --task_config demo_clean_piper_motion_viewer --seed 0 --max_seed_tries 50
```

#### O.0-4 诊断用：原始 IK/规划链路，不作为当前采集命令

用途：验证“标定 Piper/Pika URDF + 原始 `pick_diverse_bottles.py` 的 IK/规划逻辑”。该命令使用 `pick_diverse_bottles_piper`，会进入原始 `grasp_actor/place_actor -> robot.left/right_plan_path -> CuroboPlanner.plan_path` 链路；embodiment 使用 `piper_pika_agx_ik_orig_tcp`，即标定 `piper_pika_agx.urdf` 加 RoboTwin 自带 Piper TCP 转换。

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && timeout 120s bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_ik_orig_tcp 0
```

状态：已确认能进入 `Embodiment Config: piper_pika_agx_ik_orig_tcp+piper_pika_agx_ik_orig_tcp` 和原始 task/IK 链路，但没有完成 episode。最新 `tmux gen1-1/gen1-2` 仍显示大量 `Objects is unstable` 和 `target_pose cannot be None for move action`。因此它只用于定位 IK/抓取候选问题，不是当前推荐的数据生成命令。

原始 IK 链路说明：

```text
pick_diverse_bottles.py
-> grasp_actor/place_actor
-> Action(move target_pose)
-> Base_Task.move
-> robot.left_plan_path / robot.right_plan_path
-> robot._trans_from_gripper_to_endlink
-> CuroboPlanner.plan_path
```

`gen-23` 错误原因：

- 在 `~` 下直接执行 `bash collect_data.sh ...` 会找不到脚本；需要先进入 `/home/zaijia001/ssd/RoboTwin`。
- 旧 `demo_clean_piper.yml` 的 `embodiment: [piper]` 会触发双臂 embodiment 路径，RoboTwin 试图加载 `assets/embodiments/piper/curobo_left.yml`，但 Piper 目录只有 `curobo.yml`。
- 后续 `'Robot' object has no attribute 'left_planner'` 是第一次 planner 初始化失败后的残留状态次生错误。
- 第一版修正 `embodiment: [piper, piper, 0.60]` 能避免 `curobo_left.yml`，但仍加载 RoboTwin 自带 `assets/embodiments/piper/piper.urdf`。
- 当前标定版使用 `piper_pika_agx_calibrated`，加载 `assets/embodiments/piper_pika_agx/piper_pika_agx.urdf` 和标定 base pose。

`gen1-2` 错误原因：

- `No left camera link` / `No right camera link` 来自 `envs/robot/robot.py` 的 wrist camera link 查找；当前 `piper_pika_agx.urdf` 没有 `left_camera`、`right_camera` 或通用 `camera` link，所以代码打印警告并退回到第一个 link。这不是本次 seed 失败的直接异常。
- 实际失败发生在 `[Start Seed and Pre Motion Data Collection]`，episode 0 从 seed 421 到 730 反复失败，没有找到可用 premotion seed。
- 失败主要分两类：`Objects is unstable ... 001_bottle` 表示瓶子初始物理 settle 后不稳定；`target_pose cannot be None for move action` 表示原始 `pick_diverse_bottles.py` 的 scripted `grasp_actor`/move 逻辑没有为当前标定 Piper/Pika 生成可执行目标。
- 因此，只关 wrist 保存可以避免 wrist 数据依赖和日志干扰，但如果仍然连续出现 `target_pose cannot be None`，下一步应新增 Piper/Pika 专用任务逻辑，重写抓取候选/物体采样区域或直接用固定 side-grasp target，而不是继续完全复用 ALOHA-AgileX 的原始 demo 规划。

`COMMAND_LIBRARY.zh.md` 末尾新增 Mode O，用于 `pick_diverse_bottles` 的更简单对照实验：不使用手工关键帧、不使用人手朝向、不使用 AnyGrasp candidate，只用第 0 帧 FoundationPose 的两个 bottle 世界位置，按 env 任务逻辑生成抓取和放置目标。

新增入口：

```text
code_painting/plan_first_frame_foundation_pick_diverse_bottles.py
code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh
```

核心逻辑：

- 读取 `foundation_replay_d435/foundation_input_<ID>/multi_object_world_poses.npz`。
- `left_bottle` 固定左臂，`right_bottle` 固定右臂。
- 左臂 local `+Z` 设为世界 `[+1,0,0]`，右臂 local `+Z` 设为世界 `[-1,0,0]`，表示两臂从外侧水平夹取。
- grasp target = bottle center 沿接近轴反方向退 `--grasp_surface_retreat_m`，默认 `0.03m`。
- pregrasp 由 planner 沿 local `+Z` 后退 `--approach_offset_m`，默认 `0.08m`，对齐 `envs/pick_diverse_bottles.py` 的 `pre_grasp_dis=0.08`。
- action target 默认是 env 的左右目标：left `[-0.06,-0.105,1.0]`，right `[0.06,-0.105,1.0]`。
- 通过 `--reuse_plan_summary_json` 复用原有 Piper planner 的 pregrasp/grasp/close/action、IK、视频和 debug 逻辑。

夹爪朝向检查：

- Piper/Pika 配置与原始 ALOHA-AgileX 在 `global_trans_matrix=diag(1,-1,-1)`、`delta_matrix=I`、`grasp_perfect_direction=["front_right","front_left"]` 上一致。
- 差异在 URDF 夹爪结构轴和 target frame 约定：ALOHA-AgileX 的夹爪深度/指尖方向更自然对应 link6 local `+X`，开合沿 local `+/-Y`；Piper/Pika 夹爪开合轴在 gripper base local `Z/-Z`，结构轴不完全同构。
- 当前 Mode O 沿用已标定 Piper/replay 管线，保存给 planner 的 target frame 使用 local `+Z`（蓝轴）作为接近/前进轴；这和前面的 direct replay / robot-frame AnyGrasp 逻辑一致，但不是原始 ALOHA-style local `+X` 指尖深度约定。
- 如果要做严格 ALOHA-style local-X 对比，需要新增 Mode O 分支，把接近轴写入 target local `+X`，并让 planner 使用 `--approach_axis local_x`。

新增验证入口：

```text
code_painting/visualize_mode_o_gripper_frame_conventions.py
```

该脚本只读取 FoundationPose 物体位置，不跑 IK。它会在同一抓取点画出 `piper_local_z`、`aloha_local_x_y_up`、`aloha_local_x_z_up` 三套 frame，并输出每个局部轴与物理接近方向的夹角。

静态可视化命令：

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python /home/zaijia001/ssd/RoboTwin/code_painting/visualize_mode_o_gripper_frame_conventions.py --video_id 0 --foundation_frame 0 --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/mode_o_frame_convention_debug
```

Mode O wrapper 新增：

- `--target_frame_convention piper_local_z|aloha_local_x_y_up|aloha_local_x_z_up`
- `--plan_only`：只写 `plan_summary_first_frame_foundation.json`，不调用 planner。

ALOHA-style local-X plan-only 对照：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 0 --plan_only --target_frame_convention aloha_local_x_z_up --output_root /tmp/mode_o_aloha_local_x_plan_only
```

注意：`multi_object_world_poses.npz` 的键名包含 `pose_world_wxyz`，但实际数组顺序与 planner 一致，是 `[x, y, z, qw, qx, qy, qz]`。

推荐 smoke：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 0 --continue_on_error --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/first_frame_foundation_smoke
```

Viewer 注意事项：

- 你遇到的 `Renderer does not support display` 不是目标生成失败，而是 SAPIEN interactive viewer 创建失败。
- 这次日志中 viewer 创建时仍显示 `CUDA_VISIBLE_DEVICES=2`，原因是 Mode O Python 入口在调用 planner 前又把 wrapper 已经 unset 的环境变量写回去了。
- 已修复：viewer 模式下 wrapper 传 `--gpu -1` 给 Python，Python 再调用 planner 时也会移除 `CUDA_VISIBLE_DEVICES`。
- 修复后如果最小 viewer 探针仍失败，问题就是当前 `DISPLAY`/Vulkan 图形会话不可用，需要换到能打开 SAPIEN 的 VNC/图形终端。

最小 viewer 探针：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES; [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; echo "DISPLAY=$DISPLAY CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}" && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/probe_sapien_viewer.py
```

已验证：

- `python -m py_compile code_painting/plan_first_frame_foundation_pick_diverse_bottles.py` 通过。
- `bash -n code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh` 通过。
- `--ids 0 --dry_run` 路径解析通过。
- `pick_diverse_bottles id0` 无 viewer smoke 可生成 `plan_summary_first_frame_foundation.json`、`pose_debug.jsonl`、`head_cam_plan.mp4`、`third_cam_plan.mp4`。本次 pregrasp 左右臂均 reached，但 grasp 未双臂同时 reached，因此默认安全约束阻止 close/action。

## Mode N-4：Foundation 位置 + 人手朝向的 C 型夹爪预览

用途：调试 Mode N 合成目标是否合理。Mode N 使用 FoundationPose 物体世界位置作为目标位置，并复用人手 gripper 旋转矩阵作为目标朝向；`rank_previews/*.png` 会叠加 2D C 型夹爪和 RGB 局部轴，方便检查“Foundation 位置 + 人手朝向”的组合是否跑偏。

关键修复：

- `plan_keyframes_foundation_pose.py` 和 `plan_keyframes_human_replay.py` 写入 planner summary 的 pose 数组顺序已修正为 `[x, y, z, qw, qx, qy, qz]`。
- 字段名仍沿用历史名字 `pose_world_wxyz` / `raw_pose_world_wxyz`，但 planner 中 `pose_wxyz_to_matrix()` 实际按前三位位置、后四位 quaternion 解析。
- 修复前 Mode N/M wrapper 写成 `[qw, qx, qy, qz, x, y, z]`，会让 planner 把 quaternion 当作世界位置，导致目标位置明显异常。
- N-4 继续修正 Foundation replay 读取侧：`multi_object_world_poses.npz` 中 `object__pose_world_wxyz` 和 `head_camera_pose_world_wxyz` 的实际顺序也是 `[x, y, z, qw, qx, qy, qz]`。旧 N-2/N-3 在 Mode N 里取 `object_pose[4:7]`，实际取到 quaternion，因此 C 型夹爪会飞到 head camera 视野外；这不是 Foundation 物体本身出画，也不是 head 视角坏了。
- `rank_previews/*.png` 现在额外画 2D C 型夹爪：左臂蓝色、右臂橙色；局部轴为 X=红、Y=绿、Z=蓝。Mode N 默认以 local `+Z` 蓝轴作为 `foundation_pose_retreat_m` 的前进/后退轴。
- N-4/N-5 预览图额外打印 target xyz、object->target 偏移和 `proj=inside/offscreen/behind_camera`。如果 C 型夹爪不在 head camera 视野内，图片会明确标记，而不是看起来像没有生成。
- `--debug_viewer_overlay` 会关闭 clean video 模式，在 viewer/视频里显示每次规划目标的 target axis、top-1 C 型夹爪 actor、camera axes 和 SAPIEN camera frustum。
- rank preview 会按 `(frame, arm)` 标记 selected candidate，并会为左右臂各自的 debug frame 补齐预览帧。
- 默认规划模式是 `cartesian_interp_ik`：每个 stage 从当前 TCP 到目标 TCP 做位置线性插值和四元数 Slerp，然后逐 waypoint 求 IK。若 IK 中途换到另一组腕/肘解，画面上可能出现“先朝下再扭回目标”的末端朝向变化；这通常是 IK 解分支跳变，不代表关键帧 1 的目标朝向本身反向。
- N-5 使用 `--foundation_pose_retreat_m 0.08 --approach_offset_m 0.07`：grasp target 离 object center 8cm，pregrasp 总 retreat 15cm，pregrasp 到 grasp 前进 7cm。

推荐命令：

```bash
# 0609 Mode N-5：Foundation 物体位置 + 人手 gripper 朝向，输出带 2D C 型夹爪/局部轴/投影状态的 rank_previews
for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do
  bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh \
    --gpu 1 --ids 0 1 2 3 4 --continue_on_error --tasks $TASK \
    --foundation_pose_retreat_m 0.08 \
    --approach_offset_m 0.07 \
    --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-5_pregrasp15_grasp8
done
```

viewer 演示：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh \
  --gpu 2 --ids 1 --viewer --viewer_wait_at_end 1 --tasks pick_diverse_bottles \
  --debug_viewer_overlay \
  --foundation_pose_retreat_m 0.08 \
  --approach_offset_m 0.07 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-5_pregrasp15_grasp8_debug_viewer
```

单条 smoke/debug：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh \
  --gpu 1 --ids 0 --continue_on_error --tasks pick_diverse_bottles \
  --foundation_pose_retreat_m 0.08 \
  --approach_offset_m 0.07 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-5_pregrasp15_grasp8_smoke
```

检查输出：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-5_pregrasp15_grasp8/<TASK>/foundation_input_<ID>/plan_summary_foundation_pose.json
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-5_pregrasp15_grasp8/<TASK>/foundation_input_<ID>/rank_previews/keyframe_<FRAME>_rank_1.png
```
