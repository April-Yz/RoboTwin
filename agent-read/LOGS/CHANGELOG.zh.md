# CHANGELOG.zh

## 2026-06-26（stack_cups redprotect 后处理 debug）

- 新增 `code_painting/recompose_l16_stack_redprotect.py`：读取 I3.6.2 B 方案 Stage-2 foreground alpha，并把 L16 源视频中的红/粉杯颜色区域 OR 回 alpha，输出到独立 `e0_robot_object_b_points_negative_redprotect` 目录。
- 记录原因：前五个任务和 stack 的白背景阈值相同；stack 的异常更像白背景框/反选把红杯归入背景区域，而不是阈值不一致。当前 SAM3 robot 脚本没有 negative text prompt 参数，因此采用后处理保护。
- 已生成 `stack_cups id_0..4` redprotect 结果和 P4 风格 montage。
- 文档同步更新 `COMMAND_LIBRARY.zh.md` 的 I3.6.2.4/P6，以及 `agent-read/COMMANDS/pi0_h2o_training_data.*.md`。



## 2026-06-25（L16 whitebg mask 反选 debug）

- 新增 `code_painting/make_l16_whitebg_mask_debug.py`：
  - 纯后处理读取已有 Stage-2 输出，不重新跑 SAM/GPU。
  - 并排显示 L16 source、Stage1 BG、保存出的 alpha mask、alpha binary、inverse background check 和 final repaint。
  - 明确记录当前 `run_l16_whitebg_repaint_task.sh --invert_mask` 下 `mask_head_cam_plan/*.jpg` 已经是 foreground alpha。
- 生成 stack_cups id0 前 120 帧 debug：`code_painting/l16_whitebg_mask_debug/stack_cups/id_0/whitebg_invert_debug_stack_cups_id0.mp4`。
- 统计确认非 stack 五个任务均已达到 25 条以上 `Y` 标注；stack 当前还未达到。


## 2026-06-25（L16 ours 单任务标注脚本与调速键修正）

- 新增六个任务级标注脚本：`code_painting/annotate_l16_ours_<TASK>.sh`。
  - 默认 `TARGET_COUNT=25`、`OVERWRITE_MONTAGE=1`、`INITIAL_SPEED=1.0`。
  - 支持在脚本后追加 `--ids` 等参数，便于局部 review。
- 更新 `review_l16_ours_montages.py` 调速键：
  - 推荐 `+`/`-` 或 `=`/`_` 调速，并在终端打印当前倍率。
  - `[`/`]` 保留兼容，但文档提示若被窗口/播放器解释为逐帧，就改用 `+/-`。
- 文档更新：
  - `COMMAND_LIBRARY.zh.md` P4 改为任务级标注脚本入口。
  - `agent-read/COMMANDS/pi0_h2o_training_data.zh.md` / `.en.md` 同步。


## 2026-06-25（L16 ours 七面板 review 与单任务 25 条统计）

- 更新 L16 ours 可视化筛选：
  - `make_l16_repaint_montage.py` 默认加入 `left_wrist_cam_plan.mp4` 和 `right_wrist_cam_plan.mp4`。
  - 超过 4 个面板时自动用两行 `xstack` 拼接，避免七面板横向视频过宽。
  - `review_l16_ours_montages.py` 新增 `--target_count`，窗口和启动终端显示当前任务 accepted/remaining/maybe/reject/unreviewed/total。
- 文档更新：
  - `COMMAND_LIBRARY.zh.md` P1/P4 更新为七面板和按任务单独筛选命令。
  - `agent-read/COMMANDS/pi0_h2o_training_data.zh.md` / `.en.md` 同步 ours review 流程。
- 验证：`py_compile` 通过；`make_l16_repaint_montage.py --help` 和 `review_l16_ours_montages.py --help` 通过；`pick_diverse_bottles id0` 1 秒七面板 smoke 输出 `1704x640`；`git diff --check` 通过。

## 2026-06-02（Mode O 第一帧 FoundationPose 直接策略抓取）

- 后续 viewer 修复：
  - 用户运行 Mode O viewer 时日志显示 `CUDA_VISIBLE_DEVICES=2`，随后 SAPIEN 报 `Renderer does not support display`。
  - 原因：wrapper 在 viewer 模式下已用 `env -u CUDA_VISIBLE_DEVICES`，但 `plan_first_frame_foundation_pick_diverse_bottles.py` 调 planner 前又根据 `--gpu 2` 把 `CUDA_VISIBLE_DEVICES` 写回子进程环境。
  - 修复：viewer 模式下 wrapper 传 `--gpu -1`，Python 入口在 `enable_viewer=1` 时显式移除 `CUDA_VISIBLE_DEVICES`。
  - 验证：`DISPLAY=:1.0 bash code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 0 --viewer --viewer_wait_at_end 0 --continue_on_error --output_root /tmp/mode_o_viewer_env_check` 中 viewer 创建日志变为 `CUDA_VISIBLE_DEVICES=None`，并成功打印 `[viewer] interactive viewer created`。
- 新增 `pick_diverse_bottles` 对比实验 Mode O：
  - `code_painting/plan_first_frame_foundation_pick_diverse_bottles.py`
  - `code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh`
- 设计逻辑：
  - 不使用手工关键帧、人手朝向或 AnyGrasp 候选。
  - 从 `foundation_replay_d435/foundation_input_<ID>/multi_object_world_poses.npz` 读取第 0 帧两个 bottle 世界位置。
  - 固定左瓶左臂、右瓶右臂，按外侧水平夹取生成 grasp target。
  - 复用现有 Piper planner 的 `--reuse_plan_summary_json` 执行 pregrasp/grasp/close/action。
  - 默认 pregrasp 距离 `0.08m`，对齐 `envs/pick_diverse_bottles.py` 的 `pre_grasp_dis=0.08`。
  - 默认 action 目标沿用 env：left `[-0.06,-0.105,1.0]`，right `[0.06,-0.105,1.0]`。
- 代码注意：
  - `multi_object_world_poses.npz` 中 `pose_world_wxyz` 键名易误导；实际数组顺序为 `[x, y, z, qw, qx, qy, qz]`，新脚本按 planner 实际约定处理。
- 文档更新：
  - `COMMAND_LIBRARY.zh.md` 新增 O 节。
  - `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` 同步 Mode O 说明。
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_first_frame_foundation_pick_diverse_bottles.py` 通过。
  - `bash -n /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh` 通过。
  - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --ids 0 --dry_run` 通过。
  - `pick_diverse_bottles id0` 无 viewer smoke 完成，输出在 `code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/first_frame_foundation_smoke/pick_diverse_bottles/foundation_input_0`；pregrasp 左右臂均 reached，但 grasp 未双臂同时 reached，默认安全约束跳过 close/action。

## 2026-05-28（Piper AnyGrasp gripper 坐标轴修正与 IK 位置优先诊断）

- 问题定位：
  - D435 preview 中的 gripper wireframe 使用 AnyGrasp 可见夹爪坐标系，`local +X = rotation_matrix[:, 0]`。
  - Piper robot 的可见 gripper 姿态满足 `R_report = R_link6 @ global_trans_matrix @ delta_matrix`。
  - 当前 Piper 配置为 `global_trans_matrix=diag(1,-1,-1)`、`delta_matrix=I`。
  - 曾怀疑旧 Piper URDFIK 路径少反 `global_trans_matrix`，但和 direct Piper hand replay 的正确表现对照后，不能把该变换硬编码为默认行为。
- 代码更新：
  - `render_hand_retarget_piper_dual_npz_urdfik.py` 的 `_target_tcp_world_to_ee_base()` 默认保持 direct Piper hand replay 的旧 URDFIK 约定；新增 `urdfik_apply_global_trans_to_ik` 诊断开关用于对照 `R_link6_target = R_preview @ inv(global_trans_matrix @ delta_matrix)`。
  - `plan_anygrasp_keyframes_r1.py` 的 Piper dual `reach_error_pose_source=ee` 保持可见 gripper 坐标系，不再把目标转成 raw link6 坐标系做误差判定。
  - `cartesian_interp_ik` 的 partial 模式在第一个 waypoint 失败时会额外尝试当前 pose 到失败 waypoint 之间的更短小步。
  - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` 新增 `--ik_max_position_threshold_m` 和 `--ik_max_rotation_threshold_rad`，便于直接做位置优先诊断。
  - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` 的 `--viewer` 默认改为 `--viewer_wait_at_end 0`，批跑时一个 id 结束后自动进入下一个 id；需要暂停时显式传 `--viewer_wait_at_end 1`。
  - planner/wrapper 对应新增 `--piper_urdfik_apply_global_trans_to_ik` / `--piper_apply_global_trans_to_ik`。
  - wrapper 新增 `--id_start`、`--id_end`、`--ids`，用于明确跑指定 id 范围。
- 验证：
  - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/render_hand_retarget_piper_dual_npz_urdfik.py code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --max_per_task 1 --tasks pick_diverse_bottles --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --visualize_targets --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --output_root /tmp/axis_fix_pos_first_dryrun`
  - 默认 `--ik_max_rotation_threshold_rad 0.12` 的 smoke 仍在第一个 Cartesian waypoint IK 失败，说明静止不是 step/settle 不够。
  - 放宽到 `--ik_max_rotation_threshold_rad 3.14` 后，`pick_diverse_bottles id0` 输出连续 `[exec-pose]`，说明 waypoint 执行链是通的，主要阻断来自完整姿态 IK 的朝向约束。
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --max_per_task 1 --tasks pick_diverse_bottles --viewer --viewer_wait_at_end 1 --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --visualize_targets --ik_max_rotation_threshold_rad 3.14 --output_root /tmp/viewer_wait_dryrun`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --tasks pick_diverse_bottles --id_start 0 --id_end 10 --viewer --visualize_targets --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /tmp/id_filter_dryrun`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --tasks pick_diverse_bottles --ids 0 --continue_on_error --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --visualize_targets --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/debug-replay-convention-rot314`

## 2026-05-28（D435 AnyGrasp 支持左右手独立关键帧）

- 追加定位：
  - `tmux anygrasp-view-18` 中的 viewer 批跑多次被 `Ctrl-C` 中断；已有输出显示严格同步时 `pick_diverse_bottles id0` 左臂 plan `Fail`、右臂可能 `Success`，但 `dual_require_all=1` 会整体跳过 stage，因此 debug metrics 中 TCP 位移为 0。
  - 发现脚本曾被改成 `--execute_interp_steps 2400 --joint_command_scene_steps 1000 --settle_steps 300 --joint_target_wait_steps 250`，这会导致 viewer 看起来卡在 waypoint。已恢复默认 R1/V7 风格节奏：24/10/30/25。
  - 使用 `--allow_partial_dual_stage --print_pose_every 5` 验证执行链会改变 pose：`pick_diverse_bottles id0` 右臂 TCP 从约 `(0.561,-0.044,0.931)` 移动到 `(0.187,0.224,1.018)`，stdout 输出 `[exec-pose]`。
  - viewer 中“waypoint 出现后没有执行，直接闭合夹爪”的原因是旧逻辑只用 `--require_keyframe1_reached_before_action` 阻断第二关键帧，没有阻断 `close_gripper`；当 pregrasp/grasp plan 失败时，stage 被跳过但仍进入 close。
  - `stack_cups id0` 用 `--debug_stop_after_keyframe1` 复现到第一关键帧即失败，排除 close/action/第二关键帧影响。
  - 失败原因是 `cartesian_interp_ik` 的中间 waypoint IK 失败，而不是 `settle_steps` 或 `joint_target_wait_steps` 太小：pregrasp 左臂 waypoint 13/23、右臂 28/48 失败；grasp 左臂 16/28、右臂 25/45 失败。
  - 因为 `--dual_stage_require_all_plans 1`，任一 arm 的 plan 为 `Fail` 时双臂 stage 被整体跳过，所以视频看起来几乎没有执行 waypoint。
- 代码更新：
  - `plan_anygrasp_keyframes_r1.py` 新增 `--debug_stop_after_keyframe1`，只执行 init -> pregrasp -> grasp，不关爪、不进入第二关键帧。
  - `plan_anygrasp_keyframes_r1.py` 新增 `--require_keyframe1_reached_before_close`，第一关键帧未 reached 时不关夹爪。
  - `plan_anygrasp_keyframes_r1.py` 新增 `--print_execution_pose_every`，执行期间按步输出 TCP/EE world position。
  - planner 在 dual stage 中新增 `[plan-fail]` 日志，打印失败原因和失败 waypoint。
  - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` 默认传 `--require_keyframe1_reached_before_close 1`，并新增 `--viewer`、`--output_root`、`--debug_stop_after_keyframe1`、`--trajectory_mode`、`--cartesian_auto_step_m`、`--joint_interp_waypoints`、`--replan_attempts`、`--allow_partial_dual_stage`、`--print_pose_every`。
- 文档更新：
  - `COMMAND_LIBRARY.zh.md` 新增 L15.6，补充六任务 5 episode 的 no-viewer、viewer 和第一关键帧 debug 指令。
  - `agent-read/COMMANDS/piper_anygrasp_keyframes.*.md` 同步 L15.6。
- 验证：
  - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_piper.py`
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --max_per_task 1 --tasks stack_cups`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --max_per_task 1 --tasks stack_cups --viewer --output_root /tmp/viewer_out`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --max_per_task 1 --tasks stack_cups --debug_stop_after_keyframe1`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 1 --continue_on_error --tasks stack_cups --debug_stop_after_keyframe1`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 1 --continue_on_error --tasks stack_cups --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_close_guard_check` 验证第一关键帧失败后输出 `stopping before close_gripper`，不再关夹爪。
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 1 --continue_on_error --tasks pick_diverse_bottles --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_posemove_debug` 验证 stdout pose 随 waypoint 变化。

- 问题定位：
  - 用户在 zsh 中直接粘贴 L15.4 旧版长命令，触发 `zsh: command not found: mapfile`；`mapfile` 是 bash 内建，导致 `IDS` 为空并错误访问 `foundation_input_`。
  - J0.1 的 `seq 0 120` 会扫描超过任务真实 episode 数的 id，因此会出现很多正常的 `MISS`。
  - `place_bread_basket/stack_cups/handover_bottle/pnp_bread` 基础输入存在，但旧 `run_render_anygrasp_ranked_preview_keyframes_batch.sh` 只检查全局 `keyframes`，没有使用人工标注里的 `left_keyframes/right_keyframes`，导致 D435 preview summary 被跳过。
  - `stack_cups id0` 的 D435 summary 是 per-arm 关键帧：right `[51, 106]`，left `[139, 195]`。旧 planner rank preview 只按全局前两帧 `[51, 106]` 输出，因此只看到右手候选，容易误判为候选偏移。
  - planner `rank_previews` 是 SAPIEN 3D 夹爪渲染，J1.1 preview 是 D435 raw 图上的投影；`approach_offset_m=0.12` 不影响 rank preview，rank preview 只包含 `candidate_target_local_x_offset_m=-0.05` 的 5 cm TCP 补偿。
- 代码更新：
  - `render_anygrasp_ranked_preview.py` 新增 effective keyframes 逻辑：全局 `keyframes` 不足两帧时，用每个 arm 的 `left_keyframes/right_keyframes` 加全局 `keyframes` 补足。
  - preview summary 新增 `frame_selection.annotated_keyframes_by_arm`、`effective_keyframes`、`effective_keyframes_by_arm`。
  - `run_render_anygrasp_ranked_preview_keyframes_batch.sh` 的 annotation 检查改为检查左右手 effective keyframes。
  - `plan_anygrasp_keyframes_r1.py` 复用 preview summary 时，优先按 arm 使用 `effective_keyframes_by_arm`。
  - 修复 `load_reused_preview_summary()` 中 `resolved_map` 作用域错误，避免 D435 planner 复用 summary 时 `NameError`。
  - dual mode 下 `selected_keyframes` 改为左右臂 execution sequences 的并集，rank preview 也输出所有实际执行帧。
  - 新增 `code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`，作为 zsh 中可安全调用的六任务 D435 planner 入口。
  - 六任务脚本新增 `--continue_on_error`。
  - `COMMAND_LIBRARY.zh.md` 新增 L15.5：`stack_cups id0` viewer 单条调试命令。
  - 将 J1.1 的 D435 原始 preview 图复制到 `anygrasp_plan_keyframes_piper_d435_v1/stack_cups/foundation_input_0/preview_compare_d435/` 便于对比。
- 验证：
  - `python3 -m py_compile code_painting/render_anygrasp_ranked_preview.py code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_piper.py`
  - `bash -n code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh`
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --max_per_task 1`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 1 --tasks pick_diverse_bottles` 成功生成 id0 的 `plan_summary.json` 和 head/third 视频；执行层面左臂 IK 未成功，双臂同步逻辑按预期跳过 stage。
  - 重跑 `stack_cups id0` 后，planner `rank_previews` 输出 4 张图：right 51/106、left 139/195。
  - L15.5 viewer 命令块通过 `bash -n`。
  - 当前 shell 的 viewer 探针失败：`DISPLAY=` 为空，SAPIEN 报 `Renderer does not support display`；已在 L15.5 记录该情况，需从图形终端或正确 X11/Wayland forwarding 环境运行 viewer。
  - `COMMAND_LIBRARY.zh.md` L15.4 六任务 D435 planner 命令块通过 `bash -n`。
  - 单条 `place_bread_basket id0` D435 preview 成功生成到 `/tmp/d435_preview_place_id0_check/foundation_input_0`，summary 中 left 为 `[34, 64]`、right 为 `[103, 119]`。

## 2026-05-28（D435 AnyGrasp 候选与 planner 路径绑定）

- 文档更新：
  - `COMMAND_LIBRARY.zh.md` 新增 J1.2 和 L15.3。
  - `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` 同步 D435 planner 路径约束。
- 目的：
  - 明确 D435 AnyGrasp 候选选择和下游 planner 必须成对使用 `foundation_replay_d435` 与 `anygrasp_h2o_preview_d435`。
  - 避免 D435 replay 下游误复用默认广角 `anygrasp_h2o_preview` 的 `summary.json`。
- 验证：
  - 新增命令块通过 `bash -n`。

## 2026-05-28（Piper AnyGrasp 最终 viewer/no-viewer 命令拆分）

- 更新 `COMMAND_LIBRARY.zh.md`：
  - L15.1 说明改为：`dual_stage_require_all_plans` 与 `require_keyframe1_reached_before_action` 是为了显式对齐旧 V7 的行为意图，即“双臂 stage 不单臂偷跑，第一关键帧 reached 后再进入第二关键帧”。
  - 新增 L15.2，提供无 viewer 批跑版和 viewer 单条调试版。
  - viewer 版不设置 `CUDA_VISIBLE_DEVICES=2`，而是 `unset CUDA_VISIBLE_DEVICES`，避免 SAPIEN viewer 因 display GPU 被 CUDA mask 隐藏而无法打开。
  - L15.2 增加 `probe_sapien_viewer.py` 最小探针命令。
- 验证：
  - 抽取 L15.2 的 bash 代码块并通过 `bash -n /tmp/l15_2_blocks.sh`。

## 2026-05-27（LeRobot robot/AnyGrasp 25 episode 子集命令）

- 更新命令文档：
  - 新增 L5.1，补齐 6 task 的“原始人手 head + pure replay action/wrist” processed HDF5 命令。
  - 新增 L5.2，记录新三任务当前只有 `h2_pure_d435` 可用时的 human-head + D435 action/wrist 处理命令。
  - 新增 I1.1/I3.5/L8.1，记录新三任务从 Stage-1 背景、D435 visible-reinit robot repaint 到 processed HDF5 的对比链路。
  - 新增 L0 总览，按 human / robot replay / AnyGrasp replay 三类数据梳理运行顺序。
  - 新增 L10.5，补齐 L5.2 输出 `human_head_pure_d435_action` 转 LeRobot 的命令。
  - `COMMAND_LIBRARY.zh.md` 新增 L11.1，只新增不覆盖已有 L11。
  - L11.1 覆盖 6 个 robot 数据集：3 个默认广角 `pure_repaint` 和 3 个 `anygrasp_repaint`。
  - 新增显式 zip/rclone dry-run 命令，分别生成 `robot_replay_3task_25ep.zip` 和 `robot_anygrasp_3task_25ep.zip`。
  - 继续新增 L11.2，覆盖 FoundationPose 章节里的 6 个 H2O task：`pick_diverse_bottles`、`place_bread_basket`、`stack_cups`、`handover_bottle`、`pnp_bread`、`pnp_tray`。
  - 新增 L11.3，说明当前 LeRobot 转换的 prompt 来源是 processed episode 的 `instructions.json`，不是 `convert_aloha_data_to_lerobot_R1.py --task`。
  - 补齐 L6.1/L9.1/L10.4，明确 6 task 数据应先生成 processed HDF5，再转 LeRobot cache，最后才能用 L11/L11.2 抽 `_25ep`。
  - L10.4 新增 human_head_pure_action 六任务 LeRobot 转换，并统一使用用户提供的六任务 prompt。
- 同步更新：
  - `agent-read/COMMANDS/pi0_h2o_training_data.zh.md`
  - `agent-read/COMMANDS/pi0_h2o_training_data.en.md`
- 验证：
  - 抽取 L11.1 新增 bash 代码块并通过 `bash -n /tmp/l11_1_blocks.sh`。
  - 抽取 L11.2/L11.3 新增 bash 代码块并通过 `bash -n /tmp/l11_2_l11_3_blocks.sh`。
  - 抽取 L6.1/L9.1/L10.4 新增 bash 代码块并通过 `bash -n /tmp/l6_1_l9_1_l10_4_blocks.sh`。
  - 抽取 L5.1/L10.4 更新后的 bash 代码块并通过 `bash -n /tmp/l5_1_l10_4_blocks.sh`。
  - 抽取 L5.2 新增 bash 代码块并通过 `bash -n /tmp/l5_2_blocks.sh`。
  - 抽取 I1.1/I3.5/L8.1 新增 bash 代码块并通过 `bash -n /tmp/i1_1_i3_5_l8_1_blocks.sh`。
  - 抽取 L10.5 新增 bash 代码块并通过 `bash -n /tmp/l10_5_blocks.sh`。

## 2026-05-27（Piper AnyGrasp 增加第一关键帧到位后才执行 action）

- 检查 `tmux anygrasp` 当前输出：
  - 多个 id 的 pregrasp/grasp 阶段仍为 `reached=0`，随后进入 action，导致观察上像“第一关键帧没执行到位就开始第二关键帧”。
  - 新增的 `dual_stage_require_all_plans=1` 已生效：双臂阶段任一 arm plan 失败时会打印 `[dual-plan] skip stage execution...`。
- 更新代码：
  - `plan_anygrasp_keyframes_r1.py` 新增 `--require_keyframe1_reached_before_action`。
  - 该参数为 1 时，如果第一关键帧 grasp 未 reached，第二关键帧 action 会记录为 `Skipped`，不再执行 action trajectory。
  - `plan_summary.json` 新增 `require_keyframe1_reached_before_action`、`execute_interp_steps`、`settle_steps`、`reach_pos_tol_m`、`reach_rot_tol_deg`、collision/table 相关关键参数，便于后续复盘命令是否和 V7 对齐。
- 更新命令文档：
  - `COMMAND_LIBRARY.zh.md` 新增 L15.1 viewer 可视化版，执行节奏对齐旧 V7：`execute_interp_steps=24`、`joint_command_scene_steps=10`、`settle_steps=30`、`joint_target_wait_steps=25`。
  - L15.1 默认开启 `--require_keyframe1_reached_before_action 1`。
- 验证：
  - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_piper.py`
  - `conda run -n RoboTwin_bw python code_painting/plan_anygrasp_keyframes_piper.py --help | rg "require_keyframe1_reached_before_action|dual_stage_require_all_plans"`

## 2026-05-27（Piper AnyGrasp 批量命令改为标注关键帧与双臂同阶段计划约束）

- 修正 Piper AnyGrasp 双臂同步执行逻辑：
  - `plan_anygrasp_keyframes_r1.py` 新增 `--dual_stage_require_all_plans`，默认 `1`。
  - 双臂 stage 中若左/右任一 arm 规划失败，两个 arm 都不会执行该 stage，避免一只手先单独运动。
  - `plan_summary.json` 记录 `dual_stage_require_all_plans`。
- 更新命令文档：
  - `COMMAND_LIBRARY.zh.md` 将旧 L13 标记为历史命令，说明其中 `--keyframes 38 78` 不适合批量。
  - 新增 L15：id0-id10 批量命令不再传 `--keyframes`，由 `--reuse_preview_frame_mode annotated_json_keyframes` 读取每个 id 的手动标注关键帧。
  - 记录 `settle_steps/joint_target_wait_steps` 只等待已有 joint target 收敛，不能修复 IK 无解或目标不可达。
- 验证：
  - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_piper.py`
  - `conda run -n RoboTwin_bw python code_painting/plan_anygrasp_keyframes_piper.py --help | rg "dual_stage_require_all_plans|reuse_preview_frame_mode"`

## 2026-05-27（FoundationPose replay 增加 D435 内参版命令）

- 更新 `COMMAND_LIBRARY.zh.md`：
  - 在 C1 后新增 C1.2，覆盖 `pick_diverse_bottles`、`place_bread_basket`、`stack_cups`、`handover_bottle`、`pnp_bread`、`pnp_tray` 六个 H2O 任务。
  - C1.2 复用 0515 Piper head D435 外参和 `legacy_r1`，但显式加入 `--image_width 640 --image_height 480 --fovy_deg 42.499880046655484`，与 E2.4 机器人 pure replay 的 D435 RGB 内参一致。
  - 输出目录改为 `foundation_replay_d435`，避免覆盖 C1 的默认 replay。
- 新增命令文档：`agent-read/COMMANDS/piper_foundation_replay.{zh,en}.md`。
- 验证：确认 `run_multi_object_pose_r1_npz_batch.sh` 下游 `render_multi_object_pose_r1_npz_batch.py` 支持 `--image_width`、`--image_height`、`--fovy_deg` 透传。

## 2026-05-26（Piper AnyGrasp IK 阈值与 VS Code 视频兼容）

- 更新 AnyGrasp Piper 规划路径：
  - `urdfik.py` 支持从调用方设置 IK 初始阈值、最大放宽阈值和 seed 数。
  - `render_hand_retarget_piper_dual_npz_urdfik.py` 将上述阈值透传到左右 Piper URDFIK solver。
  - `plan_anygrasp_keyframes_r1.py` 新增 `--urdfik_max_position_threshold_m`、`--urdfik_max_rotation_threshold_rad`、`--urdfik_num_seeds` 等参数，并写入 `plan_summary.json`。
  - `plan_anygrasp_keyframes_r1.py` 新增 `--vscode_compatible_video 1`，在 `ffmpeg` 可用时把 head/third 输出转成 H.264 yuv420p faststart，方便 VS Code 直接预览。
  - 修正 Piper dual renderer 下 `plan-solution` 诊断对左右臂独立 base 的处理，避免沿用 R1 单 base FK 评估。
- 更新命令文档：
  - `COMMAND_LIBRARY.zh.md` L14 增加 id0-id10 AnyGrasp 批量命令。
  - `agent-read/COMMANDS/piper_anygrasp_keyframes.{zh,en}.md` 记录当前 Piper AnyGrasp 使用逻辑、推荐参数和第 38/78 帧失败分析。
- 验证：
  - `python3 -m py_compile code_painting/urdfik.py code_painting/render_hand_retarget_piper_dual_npz_urdfik.py code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_piper.py`
  - `conda run -n RoboTwin_bw python code_painting/plan_anygrasp_keyframes_piper.py --help | rg "urdfik_max_position|urdfik_max_rotation|urdfik_num_seeds|vscode_compatible|urdfik_cartesian_interp_auto_step"`

## 2026-05-25（H2O pi0 训练数据转换命令整理）

- 后续修复：
  - `process_repainted_headcam_with_wrist.py` 现在只对 `Success` 且四元数非零的帧做四元数转欧拉角，避免失败帧中的零四元数触发 `ValueError: Found zero norm quaternions in quat`。
  - `process_repainted_headcam_with_wrist.py` 与 `process_repainted_planner_outputs.py` 的 `--review-json` 现在支持读取 `hand_keyframes_all.json`，并跳过 `status=reject/discard/bad` 的 episode。
  - `--head-video-name` 支持 `{id}` 模板，可用 `--head-dir-template '.' --head-video-name 'rgb_{id}.mp4'` 表达“原始人手 head + pure replay action/wrist”。
  - 更新 `COMMAND_LIBRARY.zh.md` L1/L2/L3，三类转换都加入 manual review 过滤。
  - 验证：两个转换脚本 `py_compile` 通过；`hand_keyframes_all.json` 可正确解析 `pick_diverse_bottles` 的 102 个 id；pure repaint id0 和原始人手 head id0 均成功转换到 `/tmp` HDF5。
- 后续补充：
  - 新增 `policy/pi0/scripts/visualize_processed_hdf5_episode.py`，用于把一个 pi0 `processed_data` episode 的 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 拼成 review mp4。
  - `COMMAND_LIBRARY.zh.md` 新增 L5/L6/L7：三任务分别转换命令、生成数量检查、HDF5 结构检查、review mp4 可视化命令。
  - 验证：新可视化脚本 `py_compile` 通过；用 `processed_data/d_pour_blue-48/episode_0` 成功生成 `/tmp/pi0_review_probe.mp4`。
- 继续补充：
  - `COMMAND_LIBRARY.zh.md` 新增 L8/L9，分别覆盖 D435 visible-reinit 模式和 AnyGrasp planner 模式的三任务独立转换命令。
  - 验证：L8/L9 代表命令通过 `bash -n`。
- 继续补充：
  - `convert_aloha_data_to_lerobot_R1.py` 对齐 H2O processed HDF5 后续转换：压缩图像解码后转 RGB、缺相机时跳过 episode、HDF5 文件排序并在空目录时报错。
  - `COMMAND_LIBRARY.zh.md` 新增 L10，记录 3 种已可用模式 x 3 个任务的 LeRobot 转换命令。
  - 验证：`convert_aloha_data_to_lerobot_R1.py` `py_compile` 通过；L10 代表命令通过 `bash -n`。
- 在 `COMMAND_LIBRARY.zh.md` 末尾新增 L 节，区分三类输入：
  - 原始人手数据与既有 `policy/pi0/processed_data`
  - 未使用 AnyGrasp 的 pure replay 数据
  - AnyGrasp planner replay 数据
- 明确 pure replay 转换依赖 `world_targets_and_status.npz`、`left_wrist_replay.mp4`、`right_wrist_replay.mp4`。
- 明确 AnyGrasp planner 转换依赖 `pose_debug.jsonl`、`left_wrist_cam_plan.mp4`、`right_wrist_cam_plan.mp4`。
- 记录当前检查结果：`anygrasp_h2o_plan` 下暂未发现 planner wrist 视频，因此 AnyGrasp 的 L3 训练转换还需要先补齐 wrist 输入。
- 新增双语命令文档：`agent-read/COMMANDS/pi0_h2o_training_data.{zh,en}.md`。

## 2026-05-22（修复 K1 Piper AnyGrasp planner renderer 缺参）

- 修复 `plan_anygrasp_keyframes_r1.py` / batch wrapper：
  - 为 planner CLI 补齐 `HandRetargetR1Renderer` 新增的 `debug_visualize_cameras`、`debug_camera_axis_length`、`debug_camera_axis_thickness`、`target_local_forward_retreat_m` 默认参数。
  - 在 batch 命令生成时同步透传这 4 个参数，避免 Piper K1 规划进入 renderer 初始化时报 `missing required positional arguments`。
- 更新 `COMMAND_LIBRARY.zh.md` K1：
  - 在 heredoc 批处理命令中显式写出上述 4 个默认参数，便于后续按需打开相机可视化或设置局部前进轴 retreat。
- 验证：
  - `conda run -n RoboTwin_bw python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py code_painting/plan_anygrasp_keyframes_piper.py code_painting/plan_anygrasp_keyframes_piper_batch.py`
  - `conda run -n RoboTwin_bw python code_painting/plan_anygrasp_keyframes_piper.py --help`
  - `conda run -n RoboTwin_bw python code_painting/plan_anygrasp_keyframes_piper_batch.py --help`

## 2026-05-22（K1 续跑命令改为 heredoc 脚本）

- 更新 `COMMAND_LIBRARY.zh.md` K1：
  - 使用 `cat > /tmp/run_h2o_k1_preview_resume.sh <<'BASH' ... BASH` 生成 bash 脚本再执行。
  - 避免用户在 zsh 里粘贴 `bash -lc ‘...’` 时因中文弯引号进入 `cmdand for>` 续行状态。
- 验证：
  - 检查 K1 heredoc 片段结构。

## 2026-05-22（K1 规划命令改为只跑已有 preview 的 id）

- 更新 `COMMAND_LIBRARY.zh.md` K1：
  - 从 K0.2 preview summary 目录自动收集可运行 id。
  - 给 planner batch 显式传 `--ids`，避免未标注/未生成 preview 的 id 被扫描后报错。
  - 保留 `--skip_existing 1` 用于续跑，已有 `plan_summary.json` 的 id 会跳过。
- 验证：
  - 通过 shell 片段检查 `pick_diverse_bottles` preview summary 能正确反推出 id 列表。

## 2026-05-22（H2O 标注器支持左右手分开关键帧）

- 更新 `code_painting/annotate_hand_keyframes.py`：
  - 保留旧 `keyframes` 字段用于现有 AnyGrasp preview/planner。
  - 新增 `left_keyframes` 与 `right_keyframes`，分别由 `l`/`L` 和 `r` 标注。
  - 原 `r` 重新播放功能改为大写 `R`。
- 更新 `COMMAND_LIBRARY.zh.md` K0 说明。
- 验证：
  - `conda run -n RoboTwin_bw python code_painting/annotate_hand_keyframes.py --help`

## 2026-05-22（H2O 人工关键帧标注与废弃视频标记）

- 新增 `code_painting/annotate_hand_keyframes.py`：
  - 从旧 `d_pour_blue` 交互标注逻辑迁移为 repo 内通用工具。
  - 支持 `Space` 标注关键帧、`d` 标记/恢复废弃视频、`n/p/q` 保存导航。
  - 输出 `hand_keyframes_all.json`，并可把 `hand_vis_gripper_<id>.mp4` 归一化为下游读取的 `hand_vis_<id>.mp4`。
- 更新 `code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh`：
  - 读取标注 JSON 后跳过 `reject/discard/bad` 或关键帧少于 2 个的 id。
- 更新 `COMMAND_LIBRARY.zh.md` K0，明确 `ffplay/mpv` 只是查看器，正式标注依赖交互脚本。
- 验证：
  - `python code_painting/annotate_hand_keyframes.py --help`
  - `bash -n code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh`

## 2026-04-29（Piper 朝向猜测：仅图片 + 前上偏移修复）

- 调整 `code_painting/run_piper_gripper_standard_pose_guess.sh`：
  - 输出改为“仅保留 zed/third 图片 + index.csv + world_targets_and_status.npz”，自动删除各 case 的 replay mp4。
  - 新增默认目标偏移：`target_world_offset_xyz=(0.0, +0.1, +0.1)`，用于提升 IK 可达率。
  - `index.csv` 新增 `left_status/right_status` 字段，便于区分“姿态定义问题”与“IK不可达问题”。
- 核验结果（frame0）：
  - 偏移后标准 8 case 中出现可达项（如 `backward_guess/open_left_right_guess`），不再全 Fail。
  - 图片输出目录可直接人工标注语义：`.../output_piper_gripper_standard_pose_guess_check2/board/`。

## 2026-04-29（Piper 夹爪标准朝向猜测板工具）

- 新增脚本：`code_painting/run_piper_gripper_standard_pose_guess.sh`
  - 作用：固定 `video_id/frame`，批量生成 8 组标准朝向猜测（前/后/左/右 + 开合轴上下/左右）
  - 每个 case 跑 1 帧 replay，并把 `zed_replay.mp4` 首帧抽取到统一 `board/` 目录
  - 自动生成 `board/index.csv`，方便人工逐图标注“真实语义朝向”。
- 新增脚本：`code_painting/run_piper_gripper_orientation_guess_board.sh`
  - 作用：基于 orientation sweep 生成候选朝向调试目录（保留原始 sweep 结果供深入分析）。
- 更新命令库：`/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
  - 新增 D6：一键生成 Piper 夹爪标准朝向猜测板。

## 2026-04-29（HaMeR 全0检测问题排查）

- 现象：`detect_hands_realr1.py` 在 `hamer-r1` 环境下 GPU 跑完但统计全为 `0/0`，且覆盖了原 `hand_detections_0.npz`。
- 输入链路检查通过：
  - `pnp_star_pear_hamer_input` 下 `rgb_0..15.mp4 / params_0..15.json` 齐全。
  - 抽帧检查 `rgb_0` 可见双手，不是空画面/黑图问题。
  - `params_0.json` 内参字段正常（`fx/fy/cx/cy/width/height`）。
- 根因定位：命令文档里 GPU 指令使用了 `hamer-r1`（CPU-safe 环境），该环境在 Blackwell 卡上存在 CUDA 架构不匹配风险，推理路径异常导致逐帧无有效手结果。
- 验证修复：改用 `hamer-r1-gpu`（并 `unset LD_LIBRARY_PATH`）后，`video_id=0` 复测结果恢复为：
  - Left: `128/128`
  - Right: `128/128`
  - Both: `128/128`
  - 输出目录：`pnp_star_pear_hamer_output_dbg_gpuenv`
- 文档更新：`/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
  - A2 的 GPU 命令统一改为 `conda run -n hamer-r1-gpu ... --device cuda`
  - 新增“检测帧数快速统计”命令与 debug 基准命令。

## 2026-04-27（Piper 物体 replay：head cam link 缺失兼容修复）

- 修复 `code_painting/replay_r1_h5.py`：
  - 当机器人配置中不存在 R1 的 `zed_link/head_camera` 时，不再直接报错退出。
  - 新增 fallback：使用 `robot_base_pose + head_camera_local_*` 计算 head cam 姿态。
- 修复 `code_painting/render_multi_object_pose_r1_npz_batch.py`：
  - 增加并转发 `--save_pose_debug` 参数，避免 batch 模式参数不识别。
- 更新文档命令：
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md`
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline.md`
  - 回放阶段 mesh override 名称统一为 `star_fruit=...`（与对象目录名一致）。

## 2026-04-27（补充两阶段 I/O 格式 + 分对象 replay）

- 更新了 `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md` 与英文对应文档，补充：
  - HaMeR 阶段输入/输出根路径与关键格式字段
  - FoundationPose 阶段输入/输出根路径与关键格式字段
  - FoundationPose 输出对象目录命名（`pear`、`star_fruit`）
- 新增“分别重演轨迹/pos”命令：
  - 仅重演 `pear`
  - 仅重演 `star_fruit`
- 补充了 replay 输出关键文件说明：
  - `head_cam_replay.mp4`
  - `multi_object_world_poses.npz`
  - `pose_debug.jsonl`

## 2026-04-27（FoundationPose 提示词修正 + tmux 退出问题定位）

- 结论：`star` 提示词在当前数据上会导致 Grounding DINO 初始化失败；改为 `star fruit` 可正常进入跟踪。
- 更新文档命令：
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md`
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline.md`
  - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
- 记录了 tmux 面板“像被 kill” 的原因：`source_foundationpose_env.sh` 中 `set -e` 会传播到当前 shell，失败时直接退出当前 pane。

## 2026-04-27（pnp_star_pear：FoundationPose pear+star 阶段补齐）

- 新增 FoundationPose 的 Piper 专用准备脚本：
  - `/home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py`
- 新增 FoundationPose 的 pear+star 专用运行脚本：
  - `/home/zaijia001/FoundationPose/run_piper_star_pear_foundation.sh`
- 更新流程文档：
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md`
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline.md`
- 新增并补全了“HaMeR -> FoundationPose(pear+star) -> RoboTwin 回放”的完整命令链。
- 验证：
  - 已将 `pnp_star_pear` 的 16 个 episode 转换到 `pnp_star_pear_foundation_input`（含 `depth_<id>/*.npy` metric depth）。

## 2026-04-16（标定场景脚本新增 viewer 支持）

- 更新 `code_painting/pika/visualize_calibrated_piper_pika_scene.py`
- 更新 `code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`
- 新增交互 viewer 参数：
  - `--viewer 1`
  - `--viewer-camera overview|head`
- 修复了之前 `unrecognized arguments: --viewer 1` 的问题。
- 同步更新了命令库文档，补充标定场景的 viewer 命令。


## 2026-04-24（Piper 新手部数据 -> HaMeR -> RoboTwin 文档补全）

- 新增文档：
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md`
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline.md`
- 更新索引：
  - `agent-read/README.md`（加入新文档链接）
- 记录了针对新数据结构（`episode*/camera/color|depth/headD435`）的完整操作链路：
  - 数据转换为 `rgb_<id>.mp4/depth_<id>.mp4/params_<id>.json`
  - HaMeR 检测输出 `hand_detections_<id>.npz`
  - 可视化检查 `hand_vis_<id>.mp4`
  - RoboTwin 下游回放入口命令
- 验证：
  - `video_id=0` 已生成 `hand_vis_0.mp4` 与 `hand_vis_gripper_0.mp4`

## 2026-04-16（head cam / wrist 指令说明补充）

- 更新 `agent-read/COMMANDS/pika_scene_commands.en.md`
- 更新 `agent-read/COMMANDS/pika_scene_commands.zh.md`
- 明确了哪些命令会导出 head cam 视角图。
- 明确了当前标定场景脚本还没有 wrist 视角导出能力。


## 2026-04-16（viewer/head cam 说明补充）

- 更新 `agent-read/COMMANDS/pika_scene_commands.en.md`
- 更新 `agent-read/COMMANDS/pika_scene_commands.zh.md`
- 补充说明：
  - 标定场景脚本确实使用了 calibration bundle 来放置第二台机械臂和 head cam
  - head cam 在总览图里目前只是一个较小的 marker 加 RGB 坐标轴，所以不一定一眼能看出来
  - 目前只有手动桌面场景脚本支持交互 viewer 模式


## 2026-04-16（pika 指令库 + base 长在桌侧原因分析）

- 在 `agent-read/COMMANDS/` 下新增指令库文档：
  - `README.en.md`
  - `README.zh.md`
  - `pika_scene_commands.en.md`
  - `pika_scene_commands.zh.md`
- 新增了可直接复制的命令，覆盖：
  - 手动桌面场景的交互 viewer 运行
  - 手动桌面场景的离屏重导出
  - 标定场景导出
  - 标定 version-B 场景导出
- 分析了为什么 version-B 里 base 看起来长在桌子侧面：
  - version B 把桌子的约定旋转了 90 度
  - 然后又把之前手动内缩量复用成了 world-x 内缩
  - 所以 base 会有意靠近旋转后桌子的侧面，而不是原先理解中的前边缘


## 2026-04-16（标定场景 Version B）

- 在 `code_painting/pika/` 下实现了 version B 对齐方式：
  - 去掉第一台机械臂原先手动加的 +90° 锚定旋转
  - 尽量保持标定里的左右分离继续对齐到 world y
  - 通过旋转桌子约定，而不是旋转锚定机械臂，来适配场景
- 新增脚本：
  - `code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`
- 生成输出：
  - `code_painting/pika/output_calibrated_scene_vb/calibrated_scene_vb_overview.png`
  - `code_painting/pika/output_calibrated_scene_vb/calibrated_scene_vb_overview.mp4`
  - `code_painting/pika/output_calibrated_scene_vb/calibrated_scene_vb_headcam.png`
- 明确记录了 `robot_config_PiperPika_agx_dual_table.json` 中哪些字段在 version B 中被复用，哪些被有意忽略。


## 2026-04-16（在 code_painting/pika 下重建真实标定场景）

- 读取了真实场景标定输入：
  - `CALIBRATION_TRANSFORMS_README.md`
  - `calibration_bundle_try2.json`
- 在 `code_painting/pika/` 下重建了一个模拟场景，使用：
  - 第一台机械臂 = `robot_config_PiperPika_agx_dual_table.json` 中当前手动调好的桌面摆位
  - 第二台机械臂 = `left_base_T_right_base`
  - head camera = `left_base_T_head_camera`
- 新增脚本：
  - `code_painting/pika/visualize_calibrated_piper_pika_scene.py`
- 生成输出：
  - `code_painting/pika/output_calibrated_scene/calibrated_scene_overview.png`
  - `code_painting/pika/output_calibrated_scene/calibrated_scene_overview.mp4`
  - `code_painting/pika/output_calibrated_scene/calibrated_scene_headcam.png`
- 验证时打印了 left base、right base、head camera 的世界位姿。


## 2026-04-16（桌边安装修正）

- 确认之前的双臂桌边配置其实并没有固定在桌面上：
  - 桌子长边边线在 `y = -0.30`
  - 机器人 base 在 `y = -0.60`
  - 所以虽然 `z = 0.75` 和桌面高度一致，但平面位置仍在桌外
- 更新 `robot_config_PiperPika_agx_dual_table.json`
  - 将共享 base pose 的 `y` 从 `-0.60` 改成 `-0.30`
  - 保留已经修正好的朝向四元数
- 新的验证 base pose：
  - 左 `[-0.30, -0.30, 0.75]`
  - 右 `[0.30, -0.30, 0.75]`
- 生成桌边安装预览：
  - `code_painting/output_piper_pika_agx_dual_table_edge_mount/piper_pika_agx_dual_table_edge_mount.png`
  - `code_painting/output_piper_pika_agx_dual_table_edge_mount/piper_pika_agx_dual_table_edge_mount.mp4`


## 2026-04-16（双臂桌边朝向与斜视相机修正）

- 检查了 RoboTwin 中现有的 UR 风格相机参考：
  - `code_painting/replay_piper_dual_h5.py` 使用固定的 overview/head camera fallback
  - `code_painting/render_hand_retarget_r1_npz.py` 的第三视角是根据机器人前向和世界上方向构造的
- 定位了桌边布局的朝向问题：
  - 之前配置使用单位四元数 `[1, 0, 0, 0]`
  - 这会让机械臂正前方沿 `+x`，也就是平行于桌子长边
  - 这与“从桌子长边一侧伸向桌面操作物体”的目标不一致
- 修正 `robot_config_PiperPika_agx_dual_table.json`
  - 将 base 四元数改为 `[0.70710678, 0.0, 0.0, 0.70710678]`
  - 两个机械臂现在都朝向 `+y`
- 更新 `code_painting/visualize_piper_pika_agx_dual_table.py`
  - 把 oblique 视角修正为位于机械臂后方、朝桌面看的正常斜视角
- 生成修正后的输出：
  - `code_painting/output_piper_pika_agx_dual_table_oblique_fixed/piper_pika_agx_dual_table_oblique_fixed.png`
  - `code_painting/output_piper_pika_agx_dual_table_oblique_fixed/piper_pika_agx_dual_table_oblique_fixed.mp4`
  - `code_painting/output_piper_pika_agx_dual_table_topdown_fixed/piper_pika_agx_dual_table_topdown_fixed.png`
  - `code_painting/output_piper_pika_agx_dual_table_topdown_fixed/piper_pika_agx_dual_table_topdown_fixed.mp4`


## 2026-04-16（双臂桌边俯视相机调整）

- 进一步明确了双臂 Piper+Pika 桌边布局解释：
  - 两个机械臂位于桌子同一条长边外侧
  - 两个 base 分别距离左右短边 0.30 m
  - 两个 base 之间间距为 0.60 m
- 更新 `code_painting/visualize_piper_pika_agx_dual_table.py`
  - 新增 `--camera-mode {top_down,oblique}`
  - 临时俯视相机放在两个 base 中点上方，直接向下看桌面
- 生成临时俯视输出：
  - `code_painting/output_piper_pika_agx_dual_table_topdown/piper_pika_agx_dual_table_topdown.png`
  - `code_painting/output_piper_pika_agx_dual_table_topdown/piper_pika_agx_dual_table_topdown.mp4`
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode top_down --output-dir code_painting/output_piper_pika_agx_dual_table_topdown --image-name piper_pika_agx_dual_table_topdown.png --video-name piper_pika_agx_dual_table_topdown.mp4`


## 2026-04-16（piper_pika_agx 双臂桌边布局）

- 新增一个彩色版 Piper+Pika 组合 embodiment：
  - `assets/embodiments/piper_pika_agx/piper_pika_agx.urdf`
- 这个新组合使用：
  - Piper 手臂：原始 DAE 版本
  - Pika 夹爪：来自 `agx_arm_sim` 的 `pika2_gripper.urdf` + DAE mesh
- 新增适用于 120x60x75 cm 桌子的双臂布局配置：
  - `robot_config_PiperPika_agx_dual_table.json`
- 本轮采用的布局假设：
  - 对称的 UR 风格左右拆分
  - 分裂前共享 base pose：`[0.0, -0.60, 0.75]`
  - `embodiment_dis = 0.60`
  - 实际左右 base：
    - 左 `[-0.30, -0.60, 0.75]`
    - 右 `[0.30, -0.60, 0.75]`
- 新增预览脚本：
  - `code_painting/visualize_piper_pika_agx_dual_table.py`
- 生成输出：
  - `code_painting/output_piper_pika_agx_dual_table/piper_pika_agx_dual_table.png`
  - `code_painting/output_piper_pika_agx_dual_table/piper_pika_agx_dual_table.mp4`
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1`


## 2026-04-16（agx_arm_sim 新来源检查）

- 检查了新引入的仓库：
  - `/home/zaijia001/Downloads/agx_arm_sim`
- 确认：
  - `ros2 launch agx_arm_description display.launch.py arm_type:=piper end_effector:=pika`
  - 实际路由到的是 **Piper + Pika** 组合模型
- 发现一个关键仓库状态问题：
  - 当前 checkout 里的 `agx_arm_description/agx_arm_urdf/` 是空目录
  - 因而 xacro 引用的 Piper 手臂资源并不完整地包含在这份仓库快照里
- 确认新的 `pika2_gripper.urdf` 使用的是带内嵌颜色/材质信息的 DAE mesh
- 新增预览脚本：
  - `code_painting/visualize_agx_arm_sim_source.py`
- 生成预览输出：
  - `code_painting/output_agx_arm_sim_preview/piper_only.png`
  - `code_painting/output_agx_arm_sim_preview/piper_only.mp4`
  - `code_painting/output_agx_arm_sim_preview/pika_only.png`
  - `code_painting/output_agx_arm_sim_preview/pika_only.mp4`
  - `code_painting/output_agx_arm_sim_preview/piper_pika_combo.png`
  - `code_painting/output_agx_arm_sim_preview/piper_pika_combo.mp4`
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_agx_arm_sim_source.py --target all --output-root code_painting/output_agx_arm_sim_preview --video-frames 36 --fps 12`


## 2026-04-16（Piper 颜色诊断与量化）

- 新增一轮只做诊断、不改代码的整理，针对当前观察：
  - 原始 Piper 单独显示为灰色
  - 原始 Pika 单独显示为白色
  - 组合 `piper_pika` 后整体发白
- 确认了源证据：
  - Piper 的 DAE 内含深灰 diffuse（`0.113725 0.113725 0.113725`）
  - 组合版 `piper_pika.urdf` 中 Piper 手臂 link 仍保留显式浅色 URDF material（`rgba="0.792156862745098 0.819607843137255 0.933333333333333 1"`）
- 增加了渲染图像颜色的近似统计，对比：
  - 原始 Piper 亮灯光
  - 原始 Piper 暗灯光
  - 组合版亮灯光
  - 组合版暗灯光
- 主要结论：
  - 用户的总结是对的：Piper 原始灰色来自 DAE，而组合后发白极可能是被 URDF material 覆盖


## 2026-04-16（暗灯光分步验证）

- 新增了一轮暗灯光分步验证，分别检查：
  - 原始 Piper 手臂
  - 原始 Pika gripper
  - 组合后的 `piper_pika`
- 新增/更新脚本：
  - `code_painting/visualize_original_source_urdfs.py` 现在支持 `--lighting {bright,dark}`
  - `code_painting/visualize_piper_pika_single.py` 已支持 `--lighting {bright,dark}`
- 生成输出：
  - `code_painting/output_original_source_urdf_preview_dark/piper_arm.png`
  - `code_painting/output_original_source_urdf_preview_dark/piper_arm.mp4`
  - `code_painting/output_original_source_urdf_preview_dark/pika_gripper.png`
  - `code_painting/output_original_source_urdf_preview_dark/pika_gripper.mp4`
  - `code_painting/output_piper_pika_preview_dark/piper_pika_dark.png`
  - `code_painting/output_piper_pika_preview_dark/piper_pika_dark.mp4`
- 主要结论：
  - 组合模型发白的主要原因，很可能不是单纯灯光，而是 `assets/embodiments/piper_pika/piper_pika.urdf` 中 Piper 手臂 link 仍保留了显式的浅色 URDF material 覆盖


## 2026-04-16（piper_pika 暗灯光预览）

- 更新 `code_painting/visualize_piper_pika_single.py`，支持两种灯光预设：
  - `bright`
  - `dark`
- 为当前组合模型（有色 Piper 手臂 + 白色 Pika 夹爪）生成了一套暗灯光预览：
  - `code_painting/output_piper_pika_preview_dark/piper_pika_dark.png`
  - `code_painting/output_piper_pika_preview_dark/piper_pika_dark.mp4`
- 目的：
  - 降低过亮环境，便于观察 Piper 手臂更接近原始的深灰外观
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --lighting dark --output-dir code_painting/output_piper_pika_preview_dark --image-name piper_pika_dark.png --video-name piper_pika_dark.mp4 --video-frames 36 --fps 12`


## 2026-04-16（原始源 URDF 预览对比）

- 新增 `code_painting/visualize_original_source_urdfs.py`，用于直接预览下载目录中的原始源 URDF。
- 测试对象：
  - `/home/zaijia001/Downloads/agx_arm_urdf/piper/urdf/piper_description.urdf`
  - `/home/zaijia001/Downloads/pika_ros/src/pika_gripper_description/urdf/pika_gripper_description.urdf`
- 生成输出：
  - `code_painting/output_original_source_urdf_preview/piper_arm.png`
  - `code_painting/output_original_source_urdf_preview/piper_arm.mp4`
  - `code_painting/output_original_source_urdf_preview/pika_gripper.png`
  - `code_painting/output_original_source_urdf_preview/pika_gripper.mp4`
- 目的：
  - 在继续修改组装版本前，先比较原始 Piper 手臂和原始 Pika gripper 的外观
  - 验证发白现象是否来自原始源资产本身
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_original_source_urdfs.py --target both --output-root code_painting/output_original_source_urdf_preview --video-frames 30 --fps 12`


## 2026-04-16（piper_pika DAE 颜色恢复）

- 将组装后的 `assets/embodiments/piper_pika/piper_pika.urdf` 中手臂 visual mesh 切回 DAE，以尽量恢复原始 Piper 手臂外观。
- 新增复制的 DAE 资源目录：
  - `assets/embodiments/piper_pika/meshes/dae/*`
- 颜色来源结论：
  - 原始 Piper 手臂颜色看起来是嵌在 `/home/zaijia001/Downloads/agx_arm_urdf/piper/meshes/dae/` 下的 Collada DAE 文件里
  - 当前检查到的 Pika gripper 源码树中，没有发现 DAE/OBJ/MTL/纹理资源，只发现 STL mesh 和 URDF 里写死的白色 material
- 生成了新的基于 DAE 的预览输出：
  - `code_painting/output_piper_pika_preview_dae/piper_pika_preview.png`
  - `code_painting/output_piper_pika_preview_dae/piper_pika_preview.mp4`
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --output-dir code_painting/output_piper_pika_preview_dae --video-frames 24 --fps 12`


## 2026-04-16（piper_pika 预览导出）

- 增强了 `code_painting/visualize_piper_pika_single.py`。
- 新增功能：
  - 一个能完整拍到单臂的固定预览相机位姿
  - 静态图片导出
  - 带小幅关节动作的短视频导出
- 默认输出：
  - `code_painting/output_piper_pika_preview/piper_pika_preview.png`
  - `code_painting/output_piper_pika_preview/piper_pika_preview.mp4`
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/visualize_piper_pika_single.py`
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --video-frames 24 --fps 12`
  - 已确认两份预览文件成功写出


## 2026-04-16

- 在原有 assets 目录下新增了一个独立组装的 URDF：
  - `assets/embodiments/piper_pika/piper_pika.urdf`
  - `assets/embodiments/piper_pika/meshes/*`
- 输入来源：
  - 手臂 URDF/mesh 来自 `/home/zaijia001/Downloads/agx_arm_urdf/piper/`
  - 夹爪 URDF/mesh 来自 `/home/zaijia001/Downloads/pika_ros/src/pika_gripper_description/`
- 组装说明：
  - 以现成的组合参考 `piper_pika_gripper_description.urdf` 为起点
  - 将 package mesh URI 转换成本地相对 `meshes/...` 路径
  - 将 robot 名称改为 `piper_pika`
  - 将 `dummy_link` 重命名为 `piper_pika_dummy_link`
  - 将 `gripper_base` 重命名为 `pika_gripper_base`
  - 将所需手臂与夹爪 mesh 复制到了新 embodiment 文件夹中
- 新增了一个最小单臂可视化脚本：
  - `code_painting/visualize_piper_pika_single.py`
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/visualize_piper_pika_single.py`
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1`
  - 离屏加载确认 active joints：
    - `joint1 joint2 joint3 joint4 joint5 joint6 joint7 joint8`
- 仓库说明：
  - 当前 `.gitignore` 忽略了 `assets/*`，所以新的 embodiment 文件已在本地创建，但不会出现在 `git status` 中


## 2026-04-14（Piper V2 batch 修复）

- 修复了 `code_painting/plan_anygrasp_keyframes_piper_v2_batch.py` 的 Piper V2 batch 参数污染问题。
- 根因：
  - 复用的 `plan_anygrasp_keyframes_r1_batch.py` parser 仍然会给出默认 `--robot_config robot_config_R1.json`
  - 导致 batch 模式启动 Piper V2 单视频脚本时，显式传入了 R1 config，因此 viewer/rendering 仍然表现成 R1 风格
- 修复方式：
  - 在调用复用的 batch launcher 前，先注入 Piper 默认参数：
    - `--robot_config /home/zaijia001/ssd/RoboTwin/robot_config_Piper_dual_v2.json`
    - `--head_camera_local_quat_wxyz 1.0 0.0 0.0 0.0`
    - `--head_camera_local_pos 0.0 0.0 0.0`
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
  - 命令探针确认 batch 打印出的命令现在包含：
    - `--robot_config /home/zaijia001/ssd/RoboTwin/robot_config_Piper_dual_v2.json`
    - `--head_camera_local_quat_wxyz 1.0 0.0 0.0 0.0`


## 2026-04-14

- 新增一个真正的 Piper V2 双臂风格实现，仿照现有 UR 的双单臂拼接方式，且不修改任何 R1 / R1 Pro 文件。
- 新增文件：
  - `robot_config_Piper_dual_v2.json`
  - `code_painting/replay_piper_dual_h5.py`
  - `code_painting/render_hand_retarget_piper_dual_npz_urdfik.py`
  - `code_painting/plan_anygrasp_keyframes_piper_v2.py`
  - `code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
  - `code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh`
  - `agent-read/V2.0_piper_dual_ur_style.md`
  - `agent-read/V2.0_piper_dual_ur_style_ZH.md`
- V2 实现说明：
  - 使用 `dual_arm_embodied=false`，加载两个独立的 Piper URDF 实例
  - 保留 left/right 各自独立的 base pose，不再把两臂压回同一个 root pose
  - 新增独立的 Piper replay renderer 与 Piper URDFIK renderer
  - 左右 URDF 加载与 IK 都使用 `assets/embodiments/piper/piper.urdf`
  - 已验证实际 base pose：
    - left = `[-0.4, -0.65, 0.72]`
    - right = `[0.4, -0.65, 0.72]`
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/replay_piper_dual_h5.py code_painting/render_hand_retarget_piper_dual_npz_urdfik.py code_painting/plan_anygrasp_keyframes_piper_v2.py code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh`
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/plan_anygrasp_keyframes_piper_v2.py --help`
  - 直接配置探针确认：
    - `left_urdf_path=./assets/embodiments/piper/piper.urdf`
    - `right_urdf_path=./assets/embodiments/piper/piper.urdf`
    - `left_origin=[-0.4, -0.65, 0.72]`
    - `right_origin=[0.4, -0.65, 0.72]`
  - renderer 探针确认：
    - `[piper-v2-bases] left=[-0.4, -0.65, 0.72] right=[0.4, -0.65, 0.72]`


## 2026-04-14

- 新增一个兼容 Piper 的 AnyGrasp planner 包装层，不修改原有 R1 / R1 Pro planner 文件。
- 新增文件：
  - `robot_config_Piper_dual.json`
  - `code_painting/plan_anygrasp_keyframes_piper.py`
  - `code_painting/plan_anygrasp_keyframes_piper_batch.py`
  - `code_painting/run_plan_anygrasp_keyframes_piper_batch.sh`
  - `agent-read/2026-04-14_piper_anygrasp_wrapper.md`
  - `agent-read/2026-04-14_piper_anygrasp_wrapper_ZH.md`
- 实现方式：
  - 保持 `plan_anygrasp_keyframes_r1.py` 原始执行链不变
  - 在运行时注入 Piper robot config
  - 将 replay / URDFIK renderer 切换为 Piper 专用适配类
  - URDF IK 使用 `assets/embodiments/piper/piper.urdf`
  - 为了兼容现有 left/right 双执行结构，当前将其映射为两个 Piper 实例
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_piper.py code_painting/plan_anygrasp_keyframes_piper_batch.py`
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_batch.sh`
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/plan_anygrasp_keyframes_piper.py --help`


## 2026-04-03

- 新增 smooth 专题文档：
  - `agent-read/smooth/README.zh.md`
  - `agent-read/smooth/README.en.md`
- 目的：
  - 记录 AnyGrasp keyframe planner 当前 smooth 相关处理方式
  - 梳理 `joint_target_wait_steps` 过大时，为什么导出视频会出现跳变/瞬移感
  - 在“不改代码”的前提下，总结降低跳变并兼顾精度的可行办法，以及各自优缺点
- 文档覆盖内容：
  - 当前路径结构：`init -> pregrasp -> grasp -> action`
  - 当前常用 offset 语义：`candidate_target_local_x_offset_m=-0.03` 与 `approach_offset_m=0.08`
  - 当前平滑方式：EE/TCP pose 插值后逐 waypoint 求 IK
  - 现有 post-hoc smooth 工具：
    - `code_painting/replay_pose_debug_smooth.py`
    - `code_painting/smooth_planner_outputs_from_pose_debug.py`
    - `code_painting/batch_smooth_planner_outputs.sh`
- 继续补充新方案分析：
  - 对“每 1cm 采样一个 EE 点 + 前一解作 seed + 相邻 joint 跳变超阈值则整段拒绝”的方案做了专门评估
  - 明确指出它与当前 `cartesian_interp_ik` 的关系：当前已具备 waypoint IK + previous-seed，缺的是更密采样与显式 jump-threshold 拒绝
  - 补充了多种备选方案的优缺点与实现难度：
    - 固定步长密采样
    - 位置+旋转双阈值采样
    - joint jump threshold 过滤
    - IK 软连续性偏好
    - waypoint IK 后 joint smoothing
    - 增加语义中间 pose
    - 切到全局轨迹优化
- 继续补充 V7 debug 分析：
  - 说明为什么“不使用 try / replan”时更难到精确位置：当前系统更像 `plan-execute-correct` 闭环，而不是一次 open-loop 就完全到位
  - 说明为什么 try 能提精度但会让视频更 segmented：阶段内多次短修正 + settle 尾段未逐帧录制
  - 新增一个建议诊断量：
    - 点到目标前进轴的横向距离（axis lateral distance）
    - 用于区分“前后没到位”和“横向偏离目标前进轴”
- 本轮落实代码改动：
  - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 新增误差分解字段：
    - `lateral_to_forward_axis_m`
    - `lateral_to_forward_axis_cm`
  - 新增终端输出字段：
    - `lat_cm`
  - 接入范围：
    - 单臂 / 双臂 `plan-request`
    - 单臂 / 双臂 `plan-solution`
    - 单臂 / 双臂 `attempt`
    - 单臂 `attempt-supervision`
    - `attempt_history` / supervision error 结构
- 继续补充一种不改现有主链的设计方向：
  - 在物体坐标系里表达人手相对位姿 `T_obj_hand_demo`
  - 再通过机器人专用修正 `Δ_robot` 生成更可执行的机器人 target
  - 推荐把这层做成独立 target adapter，而不是直接改 IK solver 核心
  - 建议优先尝试：
    - 常量刚体修正
    - 分阶段修正
    - 物体类别/尺寸自适应修正
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py`
- 验证：
  - 文档整理任务，无代码改动，未运行额外脚本验证

## 2026-03-27

- 新增 raw planner v7 → repaint → review → pi0 串联脚本：
  - `run_planner_v7_repaint_review_pi0.sh`
  - 用途：
    - 不经过 smooth，直接消费 `anygrasp_plan_keyframes_realoffset_batch_pure-v7`
    - 调用原始 `batch_head_cam_repaint_with_auto_pad.sh`
    - 调用 `review_repaint_videos.py` 做人工筛选
    - 调用 `process_repainted_planner_outputs.py` 生成 pi0 / robotwin processed_data
  - 验证：
    - `bash -n run_planner_v7_repaint_review_pi0.sh`

- 新增 smooth bundle 脚本：
  - `code_painting/smooth_planner_outputs_from_pose_debug.py`
  - `code_painting/batch_smooth_planner_outputs.sh`
  - `run_reviewed_smooth_repaint_pi0_pipeline.sh`
  - 目的：
    - 针对 Step1 planner 输出去掉徘徊/近重复帧
    - 对关键状态做插值平滑
    - 重新导出同源的：head / left wrist / right wrist / pose_debug
  - 关键输出目录：
    - `code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth`
  - 验证：
    - `python -m py_compile code_painting/smooth_planner_outputs_from_pose_debug.py`
    - `bash -n code_painting/batch_smooth_planner_outputs.sh`
    - `bash -n run_reviewed_smooth_repaint_pi0_pipeline.sh`
    - `DRY_RUN=1 bash run_reviewed_smooth_repaint_pi0_pipeline.sh`
    - 单样本输出：`/tmp/d_pour_blue_0_smooth_bundle`

- 新增脚本：
  - `policy/pi0/scripts/process_repainted_planner_outputs.py`
  - 目的：使用同源 planner 数据做 pi0 处理：
    - repaint 后的 planner head
    - planner 的 `left_wrist_cam_plan.mp4`
    - planner 的 `right_wrist_cam_plan.mp4`
    - planner 的 `pose_debug.jsonl`
  - 不再混用 hand-retarget 的 wrist / `world_targets_and_status.npz`
  - 最小验证：
    - `python -m py_compile policy/pi0/scripts/process_repainted_planner_outputs.py`
    - 单样本测试输出：`/tmp/pi0_planner_repaint_test`

- 新增分析文档：
  - `agent-read/2026-03-27_repaint_duration_mismatch_analysis_ZH.md`
  - `agent-read/2026-03-27_repaint_duration_mismatch_analysis.md`
  - 目的：记录为什么 `process_repainted_headcam_with_wrist.py` 生成的序列明显短于 `head_cam_plan.mp4`
  - 结论：
    - 当前脚本按最短帧数裁切，而不是按真实秒数对齐
    - 真正限制长度的是 `world_targets_and_status.npz` 与左右 wrist replay 的帧数
    - 如果后续按更高 fps 查看，会进一步主观感觉“只有约 1 秒”
  - 本轮不改代码，只记录检查结果与原因

- 新增来源一致性分析文档：
  - `agent-read/2026-03-27_head_source_vs_wrist_source_analysis_ZH.md`
  - `agent-read/2026-03-27_head_source_vs_wrist_source_analysis.md`
  - 结论：
    - `batch_head_cam_repaint_with_auto_pad.sh` 使用的 head 来自 planner 目录下的 `head_cam_plan.mp4`
    - 当前 pi0 处理用的 wrist 来自 hand-retarget 目录下的 `left/right_wrist_replay.mp4`
    - hand-retarget 目录内部的 `zed_replay / wrist / world_targets` 长度基本一致
    - 但 planner head 与 hand-retarget wrist 不是同源流，因此不应期待帧数一致

- 新增 `policy/pi0/scripts/process_repainted_headcam_with_wrist.py`
  - 目的：
    - 把“新的 SAM/repaint 后 head cam 视频 + 左右 wrist replay + world_targets_and_status.npz”统一转成 pi0 的 `processed_data` HDF5 中间格式
  - 主要能力：
    - 支持独立指定 `--head-root` 和 `--retarget-root`
    - 支持模板化目录名：
      - `--head-dir-template`
      - `--retarget-dir-template`
    - 支持新的 head 视频文件名：
      - `target_with_original_head_cam_plan.mp4`
    - 支持 `--review-json`，默认只处理人工筛选为 `y` / `usable=true` 的视频
    - 支持 `--review-mode include_ambiguous`，把 `m` / `ambiguous` 一起纳入处理
    - 输出仍与现有 pi0 `processed_data/<task>-<num>/episode_x/*.hdf5` 兼容
  - 相关文档：
    - `agent-read/2026-03-27_pi0_repaint_wrist_to_hdf5_ZH.md`
    - `agent-read/2026-03-27_pi0_repaint_wrist_to_hdf5.md`
  - 验证：
    - `cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && python -m py_compile scripts/process_repainted_headcam_with_wrist.py scripts/process_data_retageted_human.py scripts/process_data_R1.py`

## 2026-03-25

- 新增 base 遮挡板（visual-only）以挡住底盘：
  - 文件：
    - `code_painting/render_hand_retarget_r1_npz.py`
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/render_object_pose_r1_npz.py`
  - 动机：
    - head / wrist 视频里常能看到机器人底盘，影响画面
    - 用户希望有一个可自定义高度和大小、且不参与碰撞的挡板
  - 实现：
    - 新增 visual-only `base_occluder` actor，不创建 collision
    - 挡板跟随 robot base pose 更新
    - 支持 CLI 参数：
      - `--base_occluder_enable`
      - `--base_occluder_local_pos X Y Z`
      - `--base_occluder_half_size HX HY HZ`
      - `--base_occluder_color R G B`
  - 语义：
    - `local_pos` 在机器人 base 坐标系下定义，因此随机器人朝向一起转
    - `half_size` 为 SAPIEN box half-size
    - 当前为 visual-only，不参与碰撞、IK 障碍或抓取碰撞
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_r1_npz.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py /home/zaijia001/ssd/RoboTwin/code_painting/render_object_pose_r1_npz.py`

- 为 base 遮挡板补充首次位姿日志：
  - 文件：
    - `code_painting/render_hand_retarget_r1_npz.py`
  - 变更：
    - 遮挡板首次更新位姿时打印
      - `world_p`
      - `half_size`
      - `color`
  - 目的：
    - 便于确认挡板是否创建成功，以及它是否落在预期位置/尺寸

- 修正 R1 planner 的 wrist 相机挂载定义，取消导出后图片旋转：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/CAMERA_DEBUG_NOTES_R1.md`
  - 根因：
    - 当前 R1 planner 之前沿用了更接近 `galaxea_sim/robots/r1_pro.py` 的 wrist 本地姿态
    - 但 `galaxea_sim/robots/r1.py` 的 wrist 相机只包含 `rx=-10°`，不包含额外 `z=-90°`
    - 这就是为什么必须不断对导出图片做 `90°/180°` 补丁，且横宽比例总是不自然
  - 修复：
    - 在 `plan_anygrasp_keyframes_r1.py` 内为 R1 planner 单独覆写 wrist 本地四元数，使其与 `galaxea_sim/robots/r1.py` 一致
    - `rotate_wrist_rgb_for_export(...)` 改为直通，不再对 wrist 图片做导出后旋转
    - wrist writer 尺寸恢复为原始横版 `(image_width, image_height)`
  - 影响：
    - wrist 视角由相机真实挂载姿态决定，而不是导出阶段的图像平面旋转
    - 不修改 `render_hand_retarget_r1_npz.py` 的全局默认值，避免影响 R1 Pro 相关链路
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- 再次修正 planner wrist 视频导出方向：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 用户反馈：
    - 上一轮 `180°` 校正后，导出结果仍相当于“正确视角逆时针转了 90 度”
  - 修复：
    - `rotate_wrist_rgb_for_export(...)` 由 `cv2.ROTATE_180` 改为 `cv2.ROTATE_90_COUNTERCLOCKWISE`
    - planner wrist writer 尺寸同步改回旋转后对应的 `(image_height, image_width)`
  - 当前行为：
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` 导出前做 `90°` 逆时针图像平面旋转
    - 输出尺寸与旋转后的帧一致
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- 修正 planner wrist 视频导出方向与尺寸：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 问题：
    - 上一轮将 wrist 视频误判为需要 `90°` 旋转
    - 导出结果变成竖屏，并且用户确认画面仍然是上下颠倒
  - 修复：
    - `rotate_wrist_rgb_for_export(...)` 由 `cv2.ROTATE_90_CLOCKWISE` 改为 `cv2.ROTATE_180`
    - planner wrist writer 尺寸由竖屏 `(image_height, image_width)` 改回横屏 `(image_width, image_height)`
  - 当前行为：
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` 保持横版尺寸
    - 导出前仅做 `180°` 图像平面旋转校正
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- 修复 planner 导出的 wrist 视频朝向：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 问题：
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` 视觉上相当于期望视角逆时针转了 90 度
  - 排查结论：
    - 对比 `render_hand_retarget_r1_npz.py` 与 `galaxea_sim/robots/r1_pro.py` 后，R1 与 R1 Pro 的 wrist 相机挂载本地姿态本质一致：
      - 基础四元数 `[0.5, 0.5, -0.5, 0.5]`
      - 额外 RPY 偏移 `[-10deg, 0, -90deg]`
    - 因此这次不改相机挂载定义，而是在 planner wrist 视频写出前做图像平面纠正
  - 修复：
    - 新增 `rotate_wrist_rgb_for_export(...)`
    - 在 `record_frame(...)` 写出 left/right wrist 视频前统一做 `cv2.ROTATE_90_CLOCKWISE`
    - 先旋转，再叠字/转 BGR，保证 debug 文本方向也正常
  - 影响：
    - 不改变相机世界位姿、planner target、候选坐标系转换或 head 视频
    - 只修正 planner wrist 视频导出方向
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- 修复 wrist 视频旋转后的 writer 尺寸不匹配问题：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 问题：
    - 上一轮对 planner wrist 视频做了 `90° 顺时针旋转`
    - 但 `cv2.VideoWriter` 仍按原始 `(image_width, image_height)` = `640x360` 打开
    - 旋转后的帧实际变成 `360x640`
    - 结果是 writer 成功创建文件，但无法写入有效视频流，只留下约 `258B` 的空壳 mp4
  - 修复：
    - planner wrist writer 现在改为按旋转后尺寸 `(image_height, image_width)` 打开
  - 影响：
    - `head_cam_plan.mp4` 保持原尺寸不变
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` 现在会输出为竖屏尺寸，与旋转后的图像方向一致
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- 调整 URDF IK waypoint 可视化，并补充对 stage 收敛参数的分析说明：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 可视化改动：
    - `--debug_visualize_ik_waypoints 1` 现在除中间 waypoint 外，也会显示起点和终点 marker
    - 起点与终点统一使用红色 point+forward-axis marker
    - 中间 waypoint marker 尺寸缩小，避免遮挡 viewer 中的手和目标
  - 分析说明：
    - 当前 `init` 仍只执行 `apply_robot_init_pose(...)` 的一次性 joint 命令后短暂 `step_scene(settle_steps)`，不会像 stage 末尾那样再调用 `settle_arms_to_targets(...)` 等待完全收敛
    - `--settle_steps` 默认值为 `4`，其作用是：
      - init 后推进少量物理步
      - 每段 stage 主轨迹发完后，再额外推进若干 physics scene step
    - `--joint_target_wait_steps` 默认值为 `60`，其作用是：
      - 在 stage 轨迹结束后，继续逐步等待实际关节逼近最终 joint target
  - 影响：
    - 本轮不改变规划或执行逻辑，只改 viewer/debug 呈现
  - 相关代码位置：
    - waypoint marker 更新：
      - `update_ik_waypoint_visuals(...)`
      - `_ensure_ik_waypoint_marker_actors(...)`
      - `_ensure_ik_waypoint_endpoint_actors(...)`
    - init 与 stage 收敛：
      - `apply_robot_init_pose(...)`
      - `execute_single_arm_plan(...)`
      - `execute_dual_arm_plan(...)`
      - `settle_arms_to_targets(...)`
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- 修复 pure 模式 `pose_debug.jsonl` 导出时的 EE pose 类型兼容问题：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 问题：
    - `record_frame(...)` 新增的 `current_left_ee_pose_world_wxyz` / `current_right_ee_pose_world_wxyz` 导出逻辑，误把 `robot.get_*_ee_pose()` 当成 `sapien.Pose`
    - 但该机器人接口实际返回的是 7 维列表 `[x, y, z, qw, qx, qy, qz]`
    - 导致 pure 模式批处理在首帧落盘时触发：
      - `AttributeError: 'list' object has no attribute 'p'`
  - 修复：
    - 新增 `pose_like_to_world_wxyz(...)`
    - 统一兼容 `sapien.Pose` 与 7 维 pose 列表两种输入
    - `record_frame(...)` 中的 head/ee pose 导出改为统一走该 helper
  - 影响：
    - 不改变主规划/执行逻辑
    - 只修复 pure 模式下 `pose_debug.jsonl` 的数据序列化
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- 新增 CLI 参数 `--urdfik_cartesian_interp_auto_step_m`，用于控制 `--urdfik_cartesian_interp_steps=-1` 时自动 waypoint 模式的平移密度阈值。
- 旧逻辑中 `0.05m` 为硬编码；现在变为参数，默认仍为 `0.05`，固定步数模式不受影响。
- `render_hand_retarget_r1_npz_urdfik.py` 现在会在 `[ik-trajectory]` 与 `[ik-waypoints]` 日志中打印当前 `auto_step_m`。
- 新增 AnyGrasp 执行层关节收敛调试与补偿参数：
  - `--joint_command_scene_steps`
  - `--joint_target_wait_steps`
  - `--joint_target_wait_tol_rad`
- 修改位置：
  - `code_painting/plan_anygrasp_keyframes_r1.py`
  - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
- 改动目的：
  - 解决 `plan-request` / `plan-solution` 已经显示规划终点开始回退，但 `attempt` 仍然长期停留在前方偏差的问题。
  - 让执行阶段在每个 joint waypoint 后推进更多 physics scene step，并在整段轨迹结束后继续等待，直到实际关节更接近最终命令目标再测 reach error。
- 当前分析结论：
  - 先前对 `plan_vs_current_fwd_cm` 的语义有过误读。
  - 正确解释是：在 `plan-solution` 中，`plan_vs_current_fwd_cm > 0` 表示“当前姿态在规划终点前方”，也就是规划终点其实在当前姿态后方。
  - 因此在较后期的长 try 段里，IK 终点已经开始回退，但执行层没有充分收敛到该终点。
- 额外文档：
  - `agent-read/2026-03-25_ik_execution_regression/README.zh.md`
  - `agent-read/2026-03-25_ik_execution_regression/README.en.md`

- 增加“同目标时跳过 grasp 重执行”的逻辑：
  - 当 `pregrasp` 和 `grasp` 目标 pose 实际相同（典型情形是 `--approach_offset_m 0.0`）
  - 不再在到达 `pregrasp` 后，对同一个目标再做一次 `grasp` 重规划和执行
  - 直接把 `grasp` 视为复用 `pregrasp` 结果，并写入 `grasp_skipped_same_target`
- 这样做的原因：
  - 在 `approach_offset_m=0.0` 时，日志显示 `pregrasp` 已可到位
  - 但随后对同一目标再次进入 `grasp` 会把末端重新拉离正确位置
- 另外记录：
  - 本轮曾尝试在 `urdfik.py` 中加入 seeded/unseeded FK 后验评分
  - 该尝试导致 `pregrasp` 第一轮就出现大幅错误姿态
  - 已回退，不保留该修改

- 新增对 `cartesian_interp_ik` 的第三轮修正：
  - 把 `urdfik.py` 默认位置阈值从 `0.005m` 收紧到 `0.001m`
  - 把默认旋转阈值从 `0.05rad` 收紧到 `0.02rad`
  - 阈值放宽不再无上限扩大到 `0.1m`，而是限制在小范围内
  - `render_hand_retarget_r1_npz_urdfik.py` 现在会根据 IK 阈值自动缩减过细的 cartesian waypoint 数
- 直接原因：
  - `action` 阶段日志显示目标总位移大约只有几厘米到 9 厘米，但 `--urdfik_cartesian_interp_steps 30` 会把单步平移切到约 `3mm`
  - 旧的 IK 成功阈值是 `5mm`
  - 这会让求解器在很多 waypoint 上“几乎不动也算成功”，最终整条路径理论终点仍然离目标很远
- 结论更新：
  - 对这个问题，“更多 try”不是主修复手段
  - “继续加大插值步数”反而可能更差
  - 需要先保证单个 waypoint 的目标分辨率高于 IK 成功阈值

- 在此基础上又增加了 waypoint 级别的 `seeded/unseeded` 候选比较：
  - 位置：`code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - 行为：对同一个 waypoint，同时尝试“使用当前 seed”与“无 seed”两种 IK 解
  - 再用 FK 后验比较两者对该 waypoint 的 `ee` 目标误差，保留更接近的一支
- 目的：
  - 解决在 `action` 阶段中，插值分辨率修正后仍存在的“左手被当前 seed 锁在局部解里”的问题

- 调整终端调试输出格式：
  - `plan-request`
  - `plan-solution`
  - `attempt`
- 改动内容：
  - 双臂日志改为左右手分行打印
  - `theory` 从长字符串改为：
    - `forward`
    - `backward`
    - `aligned`
  - `fwd_cm` 增加 ANSI 颜色高亮，正负更容易区分

- 进一步压缩单样本结束时的终端总结：
  - 不再打印完整的 `statuses_by_arm={...}` 大字典
  - 改为短格式：
    - `arms`
    - `arm`
    - `obj`
    - `fXX=cYY`
    - `pre/gr/act`
    - `video`

## 2026-03-25 11:51:21 +08

- 修复日志格式化回归：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 问题：
    - `colorize_forward_cm()` 在前一轮日志重构后被错误缩进到 `short_direction_label()` 作用域内
    - 运行双臂 `plan-request` 时触发 `NameError: name 'colorize_forward_cm' is not defined`
  - 修复：
    - 将 `colorize_forward_cm()` 恢复为模块级函数
    - 同时修正其内部 `if/elif` 缩进，保证正值/负值/近零颜色分支正常工作
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

## 2026-03-25 12:08:00 +08

- 新增 grasp/action 期物体碰撞开关：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - 新参数：
    - `--enable_grasp_action_object_collision 0|1`
  - 行为：
    - 默认 `0`，保持原来的无碰撞模式不变
    - 设为 `1` 时，为被执行臂选中的执行物体保留 collision geometry
    - 在 `pregrasp` 阶段仍关闭这些物体的碰撞
    - 在 `close_gripper` 前开启所选物体碰撞，并保持到 `action` 阶段结束
    - 不修改对象附着逻辑、TCP 相对位姿、目标位姿生成或其它相对变换
  - 实现说明：
    - 通过缓存并恢复 SAPIEN collision groups 做阶段性启停
    - 未被选中的其它物体仍保持原有行为
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-25 12:22:00 +08

- 调整 `plan_anygrasp_keyframes_r1.py` 的默认可视化行为：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 改动：
    - 将目标位姿坐标轴可视化恢复为默认开启
    - 在这条规划脚本里默认隐藏 left/right wrist camera，使保存视频和 viewer 中不再出现 wrist 相机视野框
  - 说明：
    - 这只影响 `plan_anygrasp_keyframes_r1.py` 这条脚本的默认行为
    - 不修改底层通用 renderer 的其它脚本用途
    - 不影响 head camera / third-person 输出本身
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

## 2026-03-25 13:05:00 +08

- 为 `plan_anygrasp_keyframes_r1.py` 增加纯净/调试可视化控制：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 新参数：
    - `--debug_visualize_targets 0|1`
    - `--viewer_show_camera_frustums 0|1`
  - 改动：
    - 将之前硬编码常开的目标坐标轴恢复为显式参数，默认仍为开启
    - 在 viewer 路径里默认关闭 SAPIEN `ControlWindow.show_camera_linesets`
    - 这样 wrist frustum 隐藏后，剩余的 zed/third 相机视野线框也会一起默认关闭
  - 说明：
    - `pure_scene_output` 继续负责主视频去掉文字、候选 gripper、target axis
    - `viewer_show_camera_frustums=0` 负责去掉 viewer 里的相机线框
    - `debug_visualize_targets=1` 可保留 target axis 作为 debug 模式
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- 修复 batch 包装层未透传纯净/调试可视化参数的问题：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - 改动：
    - 为 batch 脚本补充 `--debug_visualize_targets`
    - 为 batch 脚本补充 `--viewer_show_camera_frustums`
    - 在 `build_single_command()` 中将这两个参数继续透传给 `plan_anygrasp_keyframes_r1.py`
  - 原因：
    - 之前纯净模式命令在 batch 层被 argparse 拒绝，主脚本参数无法到达
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-25 13:35:00 +08

- 为 `--enable_grasp_action_object_collision=1` 增加最小版“渐进闭合直到接触/停滞”逻辑：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 问题分析：
    - 之前该开关只是在 `grasp/action` 阶段重新启用所选物体碰撞
    - 夹爪仍然使用一次性 `set_grippers(close)`，且仅推进极少量物理步
    - 因此即使物体存在 collision shape，手指仍可能视觉上完全闭合并穿过物体
  - 改动：
    - 新增 `close_grippers_progressively_with_collision_stop()`
    - 在 `close_gripper` 阶段按小步命令闭合夹爪
    - 每步推进物理，并读取夹爪关节 `qpos`
    - 同时检查所选物体与当前执行臂夹爪 links 的接触
    - 当“已接触且夹爪关节位移停滞”时提前停止闭合
    - 默认无碰撞模式保持原来的 `renderer.set_grippers(...)` 行为不变
  - 备注：
    - 这是最小修复，不改变 `pregrasp/grasp/action` 目标构造，也不改变物体附着到 TCP 的相对位姿
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- 修复渐进闭合逻辑中的 SAPIEN API 兼容问题：
  - 问题：
    - `PhysxArticulationJoint` 在当前环境没有 `get_qpos()`，导致批处理在进入 `close_gripper` 时崩溃
  - 修复：
    - 改为从 articulation 的 `entity.get_qpos()` 读取整条 `qpos`
    - 再按 `active_joints` 和 `joint.get_dof()` 累积偏移，提取夹爪 joint 对应的实际关节值
  - 代码位置：
    - `code_painting/plan_anygrasp_keyframes_r1.py:_get_gripper_joint_positions`

## 2026-03-25 14:05:00 +08

- 为 `grasp` 未到位但仍继续 `close_gripper` 的场景增加显式警告：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 行为：
    - 在进入 `close_gripper` 前，如果 `grasp` 阶段 `reached=False`，打印 `[warn] grasp_not_reached_before_close ...`
    - 仅增加日志，不改变原有执行逻辑

- 为 URDF IK 的 Cartesian waypoint 模式增加自动步数模式：
  - 文件：
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - 新行为：
    - `--urdfik_cartesian_interp_steps > 1` 时，继续保留原有固定步数模式
    - `--urdfik_cartesian_interp_steps -1` 时启用自动模式
  - 自动模式规则：
    - 绝对平移距离 `<= 5cm` 时，不增加中间 TCP waypoint（等价于只保留起点和终点）
    - 平移距离 `> 5cm` 后，每增加一个 `5cm` 档位增加一个中间 waypoint
    - 示例：
      - `10cm` 平移 -> `1` 个中间 waypoint
      - `15cm` 平移 -> `2` 个中间 waypoint
  - 说明：
    - 该自动模式只根据 TCP 平移距离调节 waypoint 数，不改变现有 IK 目标或执行逻辑

- 记录 `cartesian_interp_ik` 的实际执行语义说明：
  - 新文档：
    - `agent-read/V1.15_urdfik_cartesian_interp_execution_semantics_ZH.md`
    - `agent-read/V1.15_urdfik_cartesian_interp_execution_semantics.md`
  - 结论：
    - 当前中间 `ee/tcp` waypoint 确实用于逐点 IK 求解
    - 最新修复后，执行阶段也会逐段消费这些 waypoint 对应的 `joint_waypoints`
    - `cartesian_interp_ik` 现在既影响 IK 解，也更直接影响最终执行轨迹

## 2026-03-25 14:25:00 +08

- 修复 `urdfik` 执行层未真正消费 `joint_waypoints` 的问题：
  - 文件：
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - 改动：
    - `execute_plans(...)` 不再只从 `current_joints` 到 `target_joints` 做一次端点插值
    - 现在直接消费 `plan["position"]` / `plan["velocity"]`
    - 双臂执行时按左右臂相对轨迹进度交错推进，和基础 renderer 的轨迹执行语义保持一致
    - `_execute_single_ik_plan(...)` 也优先执行完整 `plan["position"]` 轨迹
  - 影响：
    - `cartesian_interp_ik` 生成的 `joint_waypoints` 现在会成为真实执行轨迹的一部分
# 2026-03-25

- pure 模式输出增强：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 改动：
    - `pure_scene_output=1` 时不再生成 `debug_selection_preview.mp4`
    - 规划主流程现在会同时写出：
      - `head_cam_plan.mp4`
      - `left_wrist_cam_plan.mp4`
      - `right_wrist_cam_plan.mp4`
    - pure 模式会自动启用 `pose_debug.jsonl`，即使未显式传 `--save_pose_debug 1`
    - `pose_debug.jsonl` 现在额外记录：
      - 左右腕部相机 pose
      - 左右 TCP / EE pose
      - 左右臂 6 维 qpos
      - 左右夹爪 finger-joint qpos
      - 物体 actor pose / replay pose
    - `plan_summary.json` 新增对应的视频和数据路径字段
  - 文档：
    - `agent-read/2026-03-25_pure_mode_outputs_ZH.md`
    - `agent-read/2026-03-25_pure_mode_outputs.md`
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py code_painting/render_hand_retarget_r1_npz_urdfik.py`
    - `git diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- 新增 `--debug_visualize_ik_waypoints` 调试参数，用于在 `--planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik` 下，把中间 `tcp_waypoints_world` 可视化到 viewer/debug 输出中。
- 可视化形式为“小球 + 局部前进轴”，仅显示中间 waypoint，不显示起点和终点；终点继续使用原有 target axis。
- 该改动只影响调试显示，不改变 waypoint 生成、IK 求解、轨迹执行或碰撞逻辑。
- `plan_anygrasp_keyframes_r1_batch.py` 已同步透传该参数。
- 验证：本轮改动后运行 `python -m py_compile` 与 `git diff --check`。
# 2026-03-25

- 修复 `urdfik` / `ReplayRenderer` 路径下 `base_occluder` 未初始化更新的问题：
  - 文件：
    - `code_painting/replay_r1_h5.py`
  - 原因：
    - `ReplayRenderer._load_robot()` 重写了基类机器人加载流程，但没有同步接入后续新增的 `base_occluder` 逻辑
    - 结果是在 `plan_anygrasp_keyframes_r1.py --planner_backend urdfik` 路径下：
      - 不会打印 `[base-occluder] ...` 调试日志
      - 挡板不会按修正后的锚点逻辑更新到目标位置
  - 改动：
    - 在 `ReplayRenderer._load_robot()` 中补上：
      - `self._base_occluder_link = self._find_robot_link(["base_link"])`
      - `self._update_base_occluder_pose()`
    - 让 replay / urdfik 执行链与基类 renderer 的挡板行为保持一致
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/replay_r1_h5.py code_painting/render_hand_retarget_r1_npz.py`

- 进一步修正 `base_occluder` 挡板的高度语义：
  - 文件：
    - `code_painting/render_hand_retarget_r1_npz.py`
  - 原因：
    - `base_link` 的平面位置更接近可见底盘，但其 `z` 原点并不等于用户直觉中的“离地高度参考”
    - 直接使用 `base_link` 的完整 3D pose 会导致挡板落到地面以下或高度明显异常
  - 改动：
    - 挡板现在使用混合锚定：
      - `x/y` 仍然跟随 `base_link` 的平面位置
      - `z` 改为相对于 renderer root/base pose 的世界高度解释
      - 朝向仅保留底座 yaw，不再继承可能干扰高度直觉的完整 3D link 姿态
    - 调试日志新增：
      - `anchor_p=...`
      - `root_z=...`
    - 这样 `--base_occluder_local_pos X Y Z` 中的 `Z` 可以按“离地高度”理解
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/render_hand_retarget_r1_npz.py`

- 修复 `base_occluder` 挡板位置偏离机器人底座的问题：
  - 文件：
    - `code_painting/render_hand_retarget_r1_npz.py`
  - 原因：
    - 挡板此前跟随 renderer 内部的 root/base pose，而不是机器人可见底盘对应的 `base_link`
    - 在当前 R1 配置下，这两者存在偏移，导致 viewer 中挡板看起来离机器人很远
  - 改动：
    - `base_occluder` 现在优先锚定到 `base_link`
    - 若未找到 `base_link`，再回退到原来的 root/base pose
    - 调试日志新增 `anchor_link=...`，便于确认当前实际使用的锚点
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/render_hand_retarget_r1_npz.py`

- 新增阶段性运行分析文档：
  - `agent-read/2026-03-25_overall_run_analysis_ZH.md`
  - `agent-read/2026-03-25_overall_run_analysis.md`
- 记录了当前版本的主要结论：
  - 轨迹形状已有改善，但 planner/IK 终点仍经常错误
  - `grasp` 未到位是 `close_gripper` 无接触的重要原因
  - 当前夹爪接触检测监控的是 finger joints 的 `child_link`，而不是 `left/right_gripper_link` 本体
- 本轮未修改运行逻辑，仅补充分析记录。

- 增强夹爪闭合阶段的碰撞调试输出，并新增“实心物体”测试模式：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - 新增参数：
    - `--debug_collision_report 1`
    - `--execution_object_collision_mode solid_bbox`
  - 改动：
    - `close_grippers_progressively_with_collision_stop(...)` 现在可在 debug 模式下打印：
      - 目标物体 collision shape 摘要
      - `left/right_gripper_link` collision shape 摘要
      - finger link collision shape 摘要
      - 每次渐进闭合迭代中的：
        - `finger_contact`
        - `base_contact`
        - `finger_pairs`
        - `base_pairs`
    - 常规 `[gripper-close]` 输出新增：
      - `base_contact=...`
    - 执行物体 collision 新增两种模式：
      - `convex`：保持原来的 `add_convex_collision_from_file`
      - `solid_bbox`：读取 mesh bounds，并用单个 axis-aligned box 作为实心碰撞体
    - `solid_bbox` 只影响 execution object collision，不改视觉 mesh
    - batch 脚本已支持转发这两个新参数
  - 说明：
    - 当前停机判据仍然只用 finger link 的接触，不改变原有行为
    - `base_contact` 目前只作为 debug 额外输出，帮助判断是否是 gripper base 先碰到物体
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `git diff --check -- code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`

- 记录一次 `d_pour_blue_0` 的碰撞调试结论：
  - 命令：
    - 使用 `--execution_object_collision_mode solid_bbox --debug_collision_report 1`
  - 关键输出：
    - `planned_object_cup(shapes=1,types=PhysxCollisionShapeBox)`
    - `planned_object_bottle(shapes=1,types=PhysxCollisionShapeBox)`
    - `left_gripper_link(shapes=0)`
    - `right_gripper_link(shapes=0)`
    - `left_gripper_finger_link1/2(shapes=0)`
    - `right_gripper_finger_link1/2(shapes=0)`
    - 闭合全过程 `contact=0` 且 `base_contact=0`
  - 结论：
    - 物体侧 collision 已经成功生效
    - 当前运行实例里，夹爪 base 和 finger links 在现有取 shape 路径下都显示 `shapes=0`
    - 因此闭合阶段完全闭合到底，与“物体无碰撞”不同，更像是“夹爪侧当前没有可被检测到的 collision shape”

- 新增夹爪接触监控范围参数：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - 新增参数：
    - `--gripper_contact_monitor_mode {fingers,fingers_and_base,all_robot_links}`
  - 说明：
    - `fingers`
      - 保持原来的 finger-only 停机监控
    - `fingers_and_base`
      - 在 finger 基础上也监控 `left/right_gripper_link`
    - `all_robot_links`
      - 将整套 robot articulation links 都纳入闭合接触监控，主要用于 debug 当前夹爪 link collision 是否缺失

## 2026-03-25 23:10:00 +08

- 新增最小 gripper/object 碰撞探针脚本并完成首轮验证：
  - 文件：
    - `code_painting/minimal_gripper_collision_probe.py`
    - `agent-read/2026-03-25_minimal_gripper_collision_probe_ZH.md`
    - `agent-read/2026-03-25_minimal_gripper_collision_probe.md`
  - 目的：
    - 在不经过 AnyGrasp/IK/stage 流程的前提下，单独验证 R1 gripper 与测试物体在 `close_gripper` 时是否真的产生 raw physics contact
    - 区分“主流程 contact debug helper 失明”和“物理引擎里根本没有 robot-object 碰撞”
  - 最小实验：
    - box probe：
      - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/minimal_gripper_collision_probe.py --arm left --object_kind box --probe_local_offset 0.04 0.0 0.0 --max_iters 20 --settle_steps_per_iter 8`
    - mesh + solid_bbox probe：
      - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/minimal_gripper_collision_probe.py --arm left --object_kind mesh --mesh_path /home/zaijia001/ssd/data/R1/hand/obj_mesh/blue_cup/blue_cup.obj --mesh_collision_mode solid_bbox --probe_local_offset 0.04 0.0 0.0 --max_iters 20 --settle_steps_per_iter 8`
  - 新结论：
    - 当前 helper/component 摘要仍会打印 robot links `shapes=0`
    - 但最小隔离实验里，`scene.get_contacts()` 已经能稳定观测到：
      - `left_gripper_finger_link1<->probe_box`
      - `left_gripper_finger_link2<->probe_box`
      - `left_gripper_finger_link1<->probe_mesh`
      - `left_gripper_finger_link2<->probe_mesh`
      - 以及更大 mesh 情况下的 `left_gripper_link<->probe_mesh`
    - 因此 `shapes=0` 不能再直接解释为“机器人没有碰撞体”或“物理中没有接触”
    - 当前主流程里 `contact=0` 更像是监控/匹配/时序问题，或 close 阶段物体真实位姿与视频观感不一致
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/minimal_gripper_collision_probe.py`
    - `git diff --check -- code_painting/minimal_gripper_collision_probe.py`

- 为主流程 `close_gripper` 调试增加 raw target contact 输出并完成回灌验证：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `agent-read/2026-03-25_minimal_gripper_collision_probe_ZH.md`
    - `agent-read/2026-03-25_minimal_gripper_collision_probe.md`
  - 改动：
    - `debug_collision_report=1` 时，`close_grippers_progressively_with_collision_stop(...)` 现在额外打印：
      - `raw_target_contacts`
      - `raw_target_contact_total`
    - `[gripper-close]` 汇总新增：
      - `raw_target_contact=0|1`
  - 回灌验证命令：
    - `bash code_painting/run_plan_anygrasp_keyframes_r1_batch.sh ... --debug_collision_report 1 --gripper_contact_monitor_mode all_robot_links --enable_viewer 0`
  - 新结论：
    - 在当前 `d_pour_blue_0` case 里，主流程 close 阶段不仅 monitor/helper 为 0，连 raw target contacts 也持续为 0
    - 因此当前问题不能只解释为“monitor 漏报”，而更像是主流程 close 阶段根本没有与目标物体发生 raw physics contact

- 为主流程 close 阶段增加目标物体 pose / collision debug 输出：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 改动：
    - `debug_collision_report=1` 时，`[collision-debug-init]` 现在额外打印：
      - `target_pose=...`
      - `target_collision_debug=...`
    - `[collision-debug-step]` 现在额外打印每次 close 迭代中的：
      - `target_pose=...`
  - 本轮回灌结论：
    - 当前 `d_pour_blue_0` close 阶段里，target pose 基本稳定不变，且 raw target contact 始终为 0
    - 因此问题更像是 visual mesh 与 `solid_bbox` collision primitive 存在显著几何偏差，而不是 close 阶段 object pose 被不断重设

- 新增 execution object collision bbox 可视化：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - 新参数：
    - `--debug_visualize_object_collision_bbox 0|1`
  - 行为：
    - 当 execution object 使用 `solid_bbox` collision 时，为其额外创建一个 visual-only bbox actor
    - bbox actor 采用与 collision 相同的局部 `center/half_size`
    - 并跟随 object actor 的 pose 更新
  - 目的：
    - 在 viewer/视频里直接比较 visual mesh 与 collision bbox
    - 判断当前“穿模”是否只是 visual mesh 现象
  - 验证：
    - `python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - 主流程回灌：`... --debug_visualize_object_collision_bbox 1 ...`

- 增加 `convex` 对照实验并更新判断：
  - 验证命令：
    - `... --execution_object_collision_mode convex --debug_collision_report 1 --gripper_contact_monitor_mode all_robot_links --enable_viewer 0`
  - 结果：
    - 当前 `d_pour_blue_0` case 在 `convex` 模式下，close 阶段 raw target contacts 仍持续为 0
  - 结论：
    - 问题不只与 `solid_bbox` 盒子近似有关
    - 即便改用 `convex` mesh collision，主流程 close 阶段仍没有 raw physics contact

- 新增 close 起始姿态导出与“从 pregrasp 就开启物体碰撞”实验：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - 新参数：
    - `--grasp_action_object_collision_start_stage {close_gripper,grasp,pregrasp}`
  - 新行为：
    - close 前写出 `close_stage_snapshot_*.json`
    - 可将 selected execution objects 的碰撞启用时机前移到 `grasp` 或 `pregrasp`
  - 关键实验结果：
    - 在 `pregrasp + convex` 实验中，主流程首次稳定出现 raw target contacts
    - 但 monitor/helper 仍然输出 `monitor_contact=0`
  - 结论更新：
    - 旧默认 `close_gripper` 才启用物体碰撞，确实太晚
    - 同时当前 monitor/contact 匹配逻辑仍存在漏报问题

- 新增 execution object 缩放覆盖参数，用于单独缩小 `cup` / `bottle` 的执行视觉与碰撞模型：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/README_anygrasp_keyframe_planner.md`
  - 新参数：
    - `--execution_object_scale_override NAME=S`
    - `--execution_object_scale_override NAME=SX,SY,SZ`
  - 行为：
    - 同时缩放 execution object 的 visual mesh 与 collision shape
    - `solid_bbox` 模式下同步缩放 bbox center / half_size
  - 典型用法：
    - `--execution_object_scale_override cup=0.9 --execution_object_scale_override bottle=0.9`
  - 备注：
    - 若要保留原来的“仅在 close_gripper 前启用物体碰撞”逻辑，继续使用：
      - `--grasp_action_object_collision_start_stage close_gripper`

- 运行“缩小物体 + 保留 close_gripper 才启用碰撞”的对照实验：
  - 输出目录：
    - `code_painting/anygrasp_single_scaled_close_only_probe/d_pour_blue_0`
  - 参数：
    - `--grasp_action_object_collision_start_stage close_gripper`
    - `--execution_object_scale_override cup=0.9`
    - `--execution_object_scale_override bottle=0.9`
  - 结果：
    - close init 到 iter 20 始终 `raw_target_contact_total=0`
    - 最终 `raw_target_contact=0`
  - 结论：
    - 仅把执行物体缩小到 0.9，不足以让旧的 `close_gripper`-only 碰撞启用逻辑检测到接触

- 运行“全程碰撞 + 缩放 0.8 / 0.5”实验：
  - 输出目录：
    - `code_painting/anygrasp_single_all_collision_scale08/d_pour_blue_0`
    - `code_painting/anygrasp_single_all_collision_scale05/d_pour_blue_0`
  - 公共参数：
    - `--grasp_action_object_collision_start_stage pregrasp`
    - `--execution_object_collision_mode convex`
  - 结果：
    - `0.8`：left/right 在 close 阶段都能稳定检测到 raw target contact
    - `0.5`：right 从 close init 就有 raw contact，left 在 close 中后段开始出现，最终 left/right 都是 `raw_target_contact=1`
  - 结论：
    - 全程碰撞输出本身是正常且可信的；真正异常的是 monitor/contact 统计链仍漏报
    - 在启碰撞时机足够早的前提下，缩小到 `0.8` / `0.5` 仍不会让 raw contact 消失

- 分析 `0.5 + pregrasp` 的全程碰撞结果，并新增 `0.5 + close_gripper + fingers` 对照实验：
  - 新输出目录：
    - `code_painting/anygrasp_single_scale05_close_only_fingers/d_pour_blue_0`
  - 结果分析：
    - `0.5 + pregrasp` 更接近“夹爪闭合到物体后被挡住、视觉上不完全闭合”的效果
    - 但当前 close 停止逻辑仍依赖 monitor_contact，因此 reason 仍可能显示 `target_reached`
  - 新对照实验结果：
    - `close_gripper + fingers + scale=0.5` 下，left 仍 `raw_target_contact=0`，right 为 `raw_target_contact=1`
  - 结论：
    - 缩小到 `0.5` 后，若仍只在 close 才启用碰撞，则问题没有完全消失；更可靠的方案仍是从 `pregrasp` 开始启用碰撞

- 新增 execution object visual/collision 分离缩放参数：
  - 新参数：
    - `--execution_object_visual_scale_override`
    - `--execution_object_collision_scale_override`
  - 兼容保留：
    - `--execution_object_scale_override`
  - 语义：
    - 可分别控制 visual mesh 与 collision shape 的缩放比例
    - 若同一物体同时给了统一缩放和专用缩放，则专用缩放优先

- 新增对“已经进入碰撞体内部后才开启碰撞，为何不会立刻卡住”的机制分析：
  - 结论要点：
    - 晚启碰撞不会回退到首次接触边界
    - close 停止逻辑依赖 monitor_contact + stall，而不是 raw contact 直接判停
    - gripper 控制是 arm 级耦合，不是单指独立控制
    - 执行对象是 kinematic actor，不会像动态物体那样被自然挤开形成干净接触边界
- 2026-05-07
  - 新增 Piper 夹爪朝向规则文档：
    - `agent-read/PIPER_GRIPPER_ORIENTATION_RULES.zh.md`
    - `agent-read/PIPER_GRIPPER_ORIENTATION_RULES.en.md`
  - 内容：
    - 解释为什么当前 debug 默认先跑左手
    - 总结 HaMeR/NPZ gripper 局部轴定义
    - 总结 `stored_orientation_post_rot_xyz_deg` / `orientation_remap_label` 的作用顺序
    - 总结 head camera 到 world、world 到 Piper base、gripper target 到 `link6` / URDFIK 的转换
    - 记录当前观察结论：蓝色 `+Z` 更像前进轴，绿色 `+Y` 更像开合轴，红色 `+X` 更像侧向/法向轴

- 2026-05-07
  - 新增 `code_painting/run_piper_retarget_postrot_board_video.sh`
  - 用途：
    - 复用完整 Piper retarget 回放链路
    - 只对 `stored_orientation_post_rot_xyz_deg` 做候选旋转扫描
    - 将每个候选的 `zed_replay.mp4` 与 `third_replay.mp4` 拼成 `board_zed.mp4` / `board_third.mp4`
  - 适用场景：
    - 当局部轴直接扫图 fail 太多时，用更接近原始 retarget 回放的方式观察朝向候选
  - 验证：
    - `bash -n code_painting/run_piper_retarget_postrot_board_video.sh`
    - 1 帧 `standard` smoke test 成功生成 `/tmp/piper_retarget_postrot_board_smoke/board/board_zed.mp4` 与 `board_third.mp4`

- 2026-05-07
  - 扩展 Piper 局部轴扫图工具：
    - 新增多帧视频模式 `--video_mode 1`
    - 新增 `board_all_zed.mp4` 和 `board_success_zed.mp4`
    - 新增 `--candidate_mode semantic`，输出 `forward_from_xp/xm/yp/ym/zp/zm` 与 `open_from_xp/xm/yp/ym/zp/zm` 语义候选
  - 目标：
    - 让 id0 的夹爪位置随帧移动，同时用固定候选朝向做大拼图视频
    - 降低 24 个 remap 中大量 fail 对人工观察的干扰
  - 验证：
    - `python3 -m py_compile code_painting/build_piper_local_axis_sweep_board.py`
    - `bash -n code_painting/run_piper_local_axis_sweep_board.sh`
    - 2 帧 smoke test 成功生成 `/tmp/piper_axis_video_smoke/board_all_zed.mp4` 与 `/tmp/piper_axis_video_smoke/board_success_zed.mp4`

- 2026-05-07
  - 新增 `code_painting/build_piper_local_axis_sweep_board.py` 与 `code_painting/run_piper_local_axis_sweep_board.sh`
  - 用途：
    - 固定当前 `PiperPika` 场景和 head cam 标定值
    - 对单帧单臂枚举所有合法右手系局部轴 remap
    - 导出 `board_zed.png` / `board_third.png` / `summary.json` / `summary.csv`
  - 解决的问题：
    - 现有 `orientation_sweep` 更偏向世界系目标姿态扫描，不够直接回答“HaMeR/重算夹爪局部 `x/y/z` 分别代表什么”
    - 新脚本直接在图上标出每个候选的红绿蓝轴相对机器人 `forward/left/up` 的语义，便于先确定局部轴定义，再进入执行误差 debug
  - 验证：
    - `python3 -m py_compile code_painting/build_piper_local_axis_sweep_board.py`
    - `bash -n code_painting/run_piper_local_axis_sweep_board.sh`

- 2026-05-11
  - 新增 `code_painting/run_piper_hamer_axes_replay_batch.sh`
  - 用途：
    - 批量回放 `hand_detections_*.npz`
    - 使用最终确认的 HaMeR/NPZ 夹爪轴规则：`orientation_remap_label=identity` 且 `stored_orientation_post_rot_xyz_deg=0 0 0`
    - 默认按 `ARMS=both` 同时回放左右手，支持 `ID_FILTER` 选择单个、多个或范围 ID
    - 默认 `KEEP_ONLY_ZED_THIRD=1`，清理 `frames/` 下 depth/wrist PNG，只保留 zed/third RGB 帧
  - 文档：
    - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 的 D 段已压缩为最终 replay 指令
    - 夹爪朝向 debug/扫图/历史 remap 指令迁移到 `/home/zaijia001/ssd/PIPER_GRIPPER_ORIENTATION_DEBUG.zh.md`
  - 验证：
    - `bash -n code_painting/run_piper_hamer_axes_replay_batch.sh`

- 2026-05-11
  - 扩展 `code_painting/render_hand_retarget_r1_npz.py`
  - 新增功能：
    - 在 Piper/HaMeR 手 replay 场景中叠加 FoundationPose 物体轨迹
    - 新参数 `--object_replay_input_dir` 指向 video-level FoundationPose 输出目录
    - 新参数 `--object_missing_frame_policy hide|hold_last`
    - 新参数 `--objects` 和 `--object NAME=/path/to/mesh.obj` 用于选择和覆盖物体 mesh
  - 新增批处理脚本：
    - `code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`
    - 自动按 `hand_detections_<id>.npz` 匹配 FoundationPose 目录中同 ID 的物体轨迹
  - 文档：
    - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 新增 E 段“HaMeR 手 + FoundationPose 物体同场 replay”
  - 验证：
    - `python3 -m py_compile code_painting/render_hand_retarget_r1_npz.py`
    - `bash -n code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`
    - 1 帧 smoke test 成功生成 `/tmp/piper_hamer_axes_with_objects_smoke/id_0/zed_replay.mp4` 与 `third_replay.mp4`

- 2026-05-11
  - 新增 `code_painting/plot_piper_gripper_wrist_object_axis_distances.py`
  - 用途：
    - 用同一套 Piper head camera 标定把 HaMeR gripper/wrist-retreat 点与 FoundationPose 物体 pose 转到世界系
    - 输出左手对 `pear`、右手对 `star_fruit` 的世界轴向距离曲线
    - 曲线包含 gripper `dx/dy/dz` 与 wrist-retreat `dx/dy/dz`
  - 文档：
    - `/home/zaijia001/ssd/PIPER_GRIPPER_ORIENTATION_DEBUG.zh.md` 新增 D-debug-9 指令
  - 验证：
    - `python3 -m py_compile code_painting/plot_piper_gripper_wrist_object_axis_distances.py`
    - id0 成功生成 `output_piper_replay_hamer_axes_with_objects_all/id_0/gripper_wrist_object_axis_distance_id0.png`

- 2026-05-18
  - 新增 0515/new_table Piper 标定配置：
    - `robot_config_PiperPika_agx_dual_table_0515.json`
  - 标定源：
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/head_d435_new_table_0515_head_from_wrist.json`
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/left_base_T_right_base_new_table.json`
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/right_wrist_new_table_eye_in_hand.json`
  - 更新默认 head/base 参数：
    - `code_painting/run_piper_hamer_axes_replay_batch.sh`
    - `code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`
    - `code_painting/plot_piper_gripper_wrist_object_axis_distances.py`
  - 更新用户指令：
    - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 的 C/D/E Piper replay 命令
    - 新增 C0 记录下次重标定时需要同步修改的文件和值
  - 验证：
    - `bash -n` 两个 Piper replay wrapper 通过
    - `python3 -m json.tool robot_config_PiperPika_agx_dual_table_0515.json` 通过
    - `python -m py_compile code_painting/plot_piper_gripper_wrist_object_axis_distances.py` 通过
    - 1 帧 id0 smoke test 可启动并输出视频；日志确认 right base 为 `[0.5562, -0.2718, 0.7698]`，但该旧 id0 目标在新标定下左右 IK 均为 Fail，后续需要基于新标定检查 target offset/可达性

- 2026-05-18
  - 修正 0515/new_table Piper 标定说明，补齐此前遗漏的左腕相机外参：
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/left_wrist_new_table_eye_in_hand.json`
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/right_wrist_new_table_eye_in_hand.json`
  - 更新 `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`：
    - C0 现在明确列出 head 0515、left/right wrist、left_base_T_right_base 四个当前应使用的 new_table JSON
    - 明确同目录旧 `head_d435_new_table_head_from_wrist.json` 不用于当前 replay
    - 增加当前 base/head camera 摆放估算，便于复查现实桌面布置是否一致
    - 修复 F 段距离曲线命令被断行导致不能直接复制执行的问题
  - 说明：
    - 当前 D/E 主 replay 仍只消费 `robot_config + head_camera`；左右 wrist 外参记录在命令库中，后续启用真实 wrist camera 渲染时再接入左右分别的 local pose 参数
  - 验证：
    - 读取四个 new_table JSON 并计算摆放关系通过

- 2026-05-18
  - 更新 `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 的 D 段：
    - 新增 `D0. place_bread_basket：HaMeR 检测结果与人手夹爪可视化`
    - 记录 `place_bread_basket/harmer_input -> harmer_output` 的 HaMeR GPU 检测命令
    - 增加 `hand_vis_gripper_*.mp4` 查看命令，便于直接检查 HaMeR 输出的人手夹爪点/轴可视化
    - 增加把 `place_bread_basket/harmer_output/hand_detections_*.npz` 接入 Piper HaMeR axes replay 的单 ID 和批处理命令
  - 验证：
    - 确认 `harmer_output` 下已有 `hand_detections_0..10.npz` 和 `hand_vis_gripper_0..10.mp4`

- 2026-05-18
  - 新增 Piper 标定 bundle 工作流：
    - `code_painting/build_piper_calibration_bundle.py`
    - `code_painting/visualize_piper_calibration_bundle.py`
    - `/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json`
  - 更新 replay 入口：
    - `render_hand_retarget_r1_npz.py` 新增 `--piper_calibration_bundle`
    - `run_piper_hamer_axes_replay_batch.sh` 新增 `CALIBRATION_BUNDLE`
    - `run_piper_hamer_axes_with_objects_replay_batch.sh` 新增 `CALIBRATION_BUNDLE`
  - 用途：
    - 从 head/base/left_wrist/right_wrist 四个 handeye JSON 生成单个自包含标定 JSON
    - replay 时由 bundle 自动生成本次输出目录下的 `calibration_bundle_robot_config.json`，并覆盖 head camera local pos/quat
    - 提供 `axes_compare_old_head.png` 可视化 base/head camera 坐标轴，对比旧 head 外参
  - 观察：
    - 旧 head 参数到 0515 head 参数的 local 平移差约 `0.123 m`，旋转差约 `120.57 deg`
    - 同目录旧 `head_d435_new_table_head_from_wrist.json` 到 0515 head 的差异较小：平移约 `0.039 m`，旋转约 `4.00 deg`
  - 验证：
    - bundle 生成通过
    - 坐标轴 PNG 生成通过
    - `bash -n` 两个 batch wrapper 通过
    - `py_compile` 三个 Python 文件通过
    - `CALIBRATION_BUNDLE=...` 对 `place_bread_basket` id0 运行 1 帧 smoke test 通过，程序正确加载 bundle 并输出 replay 视频；该帧 IK 仍 fail，属于目标可达性问题，不影响 bundle 读取

- 2026-05-18
  - 新增 head camera 场景内可视化：
    - `render_hand_retarget_r1_npz.py` 新增 `--debug_visualize_cameras`
    - 可在 third-person 渲染里画出 head camera 的白色机身、红绿蓝局部 xyz 轴、黄色 `-Z` 光轴
    - `run_piper_hamer_axes_replay_batch.sh` 和 `run_piper_hamer_axes_with_objects_replay_batch.sh` 新增 `DEBUG_VISUALIZE_CAMERAS/DEBUG_CAMERA_AXIS_LENGTH/DEBUG_CAMERA_AXIS_THICKNESS`
  - 更新 `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`：
    - 说明 SAPIEN viewer 需要 `DISPLAY/WAYLAND_DISPLAY`，当前纯 SSH 且 `DISPLAY=None` 时不会弹窗
    - 补充 `place_bread_basket` 的三版本 head camera marker 对比命令：
      - 最早手写 head 参数 + 旧 robot config
      - pre-0515 new_table head bundle
      - 0515 new_table head bundle
  - 验证：
    - `DEBUG_VISUALIZE_CAMERAS=1` 对 `place_bread_basket` id0 运行 1 帧 smoke test 通过
    - 成功生成 `/tmp/place_bread_basket_camera_marker_smoke/id_0/frames/third_0000.png`
    - `bash -n` 两个 batch wrapper 通过
    - `py_compile` replay 与 bundle 可视化脚本通过

- 2026-05-18
  - 增强 SAPIEN viewer 排查：
    - `render_hand_retarget_r1_npz.py` 在创建 viewer 前打印 `DISPLAY/WAYLAND_DISPLAY/XDG_SESSION_TYPE`
    - viewer 创建成功时打印 `[viewer] interactive viewer created`
    - viewer 创建失败时捕获更宽泛异常并打印异常类型
    - 新增 `code_painting/probe_sapien_viewer.py`，用于在 VNC 终端中独立验证最小 SAPIEN viewer 是否可显示
  - 更新 `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`：
    - viewer 命令前打印 display 环境
    - viewer 命令加入 `--debug_visualize_cameras 1 --debug_camera_axis_length 0.22`
    - 增加最小 viewer probe 命令
  - 说明：
    - 之前的 `output_place_bread_basket_piper_viewer_probe` 命令没有开启 `debug_visualize_cameras`，因此 third 图不会显示 head camera marker
  - 验证：
    - `DEBUG_VISUALIZE_CAMERAS=1 DEBUG_CAMERA_AXIS_LENGTH=0.5` 生成 `/tmp/place_bread_basket_camera_marker_big_smoke/id_0/frames/third_0000.png`，可见 head camera 坐标轴 marker

- 2026-05-18
  - 定位 viewer 不弹窗原因：
    - VNC 终端中 `probe_sapien_viewer.py` 不设置 `CUDA_VISIBLE_DEVICES` 可以创建 viewer
    - hand replay 设置 `CUDA_VISIBLE_DEVICES=2` 时 SAPIEN 报 `Renderer does not support display`
    - 结论：viewer 需要看到驱动 VNC/X display 的 GPU，`CUDA_VISIBLE_DEVICES=2` 会把该 display GPU 从进程可见设备中隐藏
  - 更新：
    - `render_hand_retarget_r1_npz.py` 的 viewer 日志增加 `CUDA_VISIBLE_DEVICES`
    - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 的 viewer 命令移除 `CUDA_VISIBLE_DEVICES=2`
    - 增加带 `CUDA_VISIBLE_DEVICES=2` 的最小 probe 反例命令
  - 验证：
    - `bash -n code_painting/run_piper_hamer_axes_replay_batch.sh`
    - `python -m py_compile code_painting/render_hand_retarget_r1_npz.py code_painting/probe_sapien_viewer.py`

- 2026-05-18
  - 定位并修正 0515 head camera 朝向偏差的根因：
    - handeye JSON 中的 `left_base_T_head_camera` 是 raw/optical 相机坐标
    - replay 命令在 `--camera_cv_axis_mode legacy_r1` 下需要 render/SAPIEN 相机位姿
    - 正确关系为 `T_render = T_raw_optical @ legacy_r1.T`
  - 关键验证：
    - 最早手写 head 位置与 `head_d435_try2_head_from_wrist.json` raw translation 仅差 `3.3e-7 m`
    - 最早手写 head quaternion 与 try2 raw rotation 相差 `120.0 deg`
    - 最早手写 head quaternion 与 `try2_raw @ legacy_r1.T` 相差 `0.0 deg`
    - 因此原先看到的 `120 deg` 主要是相机轴约定差异，不是物理标定漂移
  - 更新：
    - `build_piper_calibration_bundle.py` 生成 bundle 时保存 raw optical head，同时把 replay 使用的 `head_camera.left_base_T_head_camera` 转为 render/SAPIEN 位姿
    - 重新生成 `calibration_bundle_piper_new_table_0515.json` 与 `calibration_bundle_piper_new_table_pre0515.json`
    - 更新 D/E wrapper 和 `plot_piper_gripper_wrist_object_axis_distances.py` 默认 head quaternion 为 0515 render/SAPIEN quaternion
    - 更新 `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` 中散写 head quaternion 的 C/D/E 命令
  - 标定质量观察：
    - 0515 head residual mean/max：`0.607/2.193 deg`，`0.0048/0.0167 m`
    - pre-0515 head residual mean/max：`0.574/1.298 deg`，`0.0059/0.0151 m`
    - 左 wrist residual mean/max：`0.563/0.985 deg`，`0.0129/0.0193 m`
    - 右 wrist residual mean/max：`1.795/3.567 deg`，`0.0103/0.0180 m`
  - 验证：
    - `py_compile` 通过：`build_piper_calibration_bundle.py`、`render_hand_retarget_r1_npz.py`、`plot_piper_gripper_wrist_object_axis_distances.py`
    - `bash -n` 通过：两个 Piper HaMeR batch wrapper
    - `json.tool` 通过：两个 new_table calibration bundle
    - `CALIBRATION_BUNDLE=...new_table_0515.json DEBUG_VISUALIZE_CAMERAS=1 MAX_FRAMES=1 ID_FILTER=0` smoke test 通过，bundle 正确加载并生成 `/tmp/place_bread_basket_camera_axis_fixed_smoke/id_0/third_replay.mp4`

- 2026-05-19
  - 新增 replay 目标沿夹爪局部蓝色 `+Z` 前进轴后退的参数：
    - `render_hand_retarget_r1_npz.py` 新增 `--target_local_forward_retreat_m`
    - 正数含义：`target_position -= distance * local(+Z)`，即沿可视化蓝色前进轴反方向后退
    - 该局部后退在 camera-to-world 之后、普通 `target_world_offset_xyz` 之前应用，因此跟随每帧夹爪朝向，而不是固定世界 XYZ
  - 更新 wrapper：
    - `run_piper_hamer_axes_replay_batch.sh` 新增 `TARGET_LOCAL_FORWARD_RETREAT_M`
    - `run_piper_hamer_axes_with_objects_replay_batch.sh` 新增 `TARGET_LOCAL_FORWARD_RETREAT_M`
  - 兼容性修复：
    - `build_piper_local_axis_sweep_board.py` 和 `plot_piper_gripper_wrist_object_axis_distances.py` 补齐 renderer 构造参数
  - 验证：
    - `py_compile` 通过：`render_hand_retarget_r1_npz.py`、`build_piper_local_axis_sweep_board.py`、`plot_piper_gripper_wrist_object_axis_distances.py`
    - `bash -n` 通过：两个 Piper HaMeR batch wrapper
    - `TARGET_LOCAL_FORWARD_RETREAT_M=0.05 MAX_FRAMES=1 ID_FILTER=0` smoke test 通过，日志打印 `[target-local-retreat] along_local_plus_z_blue_m=0.0500`

- 2026-05-20
  - 修复 FoundationPose 多物体 replay 的 renderer 构造兼容问题：
    - `HandRetargetR1Renderer.__init__` 新增 camera debug 与局部蓝轴后退参数后，`render_object_pose_r1_npz.py` 仍使用旧构造参数，导致 `run_multi_object_pose_r1_npz_batch.sh` 报 `missing 4 required positional arguments`
    - 在 `render_object_pose_r1_npz.py`、`replay_r1_h5.py`、`minimal_gripper_collision_probe.py` 的 `ReplayRenderer(...)` 构造调用中补齐默认参数
  - 更新命令库：
    - `/home/zaijia001/ssd/RoboTwin/COMMAND_LIBRARY.zh.md` C1 中 pick_diverse_bottles 第 182 行附近增加说明，明确这是 Piper 0515 head/base 标定的 FoundationPose 双物体 replay 命令
  - 验证：
    - `py_compile` 通过：`render_object_pose_r1_npz.py`、`replay_r1_h5.py`、`minimal_gripper_collision_probe.py`、`render_multi_object_pose_r1_npz.py`
    - `bash -n` 通过：`run_multi_object_pose_r1_npz_batch.sh`
    - 对 pick_diverse_bottles id0 跑 `--max_frames 1 --skip_existing 0` smoke test 成功，生成 `/tmp/pick_diverse_bottles_foundation_replay_smoke/foundation_input_0/head_cam_replay.mp4` 和 `multi_object_world_poses.npz`

- 2026-05-20
  - 更新 `COMMAND_LIBRARY.zh.md` 的 E2 单条 replay 命令：
    - 新增 pick_diverse_bottles、place_bread_basket、stack_cups 三个任务的人手 + FoundationPose 物体同场 replay 指令
    - 三个命令都使用 `--piper_calibration_bundle calibration_bundle_piper_new_table_0515.json`
    - 三个命令都加入 `--target_local_forward_retreat_m 0.05`，用于沿夹爪局部蓝色 `+Z` 前进轴反方向后退 5cm
    - 在 E2 中注明 viewer 开启方式：追加 `--enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02`
  - 验证：
    - 对 pick_diverse_bottles id0 运行 `--max_frames 1` smoke test 成功
    - 日志确认 `[target-local-retreat] along_local_plus_z_blue_m=0.0500`
    - 日志确认加载 FoundationPose 物体 `['left_bottle', 'right_bottle']`

- 2026-05-20
  - 将 `COMMAND_LIBRARY.zh.md` 的 E2.1/E2.2/E2.3 三个 H2O replay 命令从单个 id0 改为批量 id0-id10：
    - 使用 `for ID in $(seq 0 10)` 循环
    - 输入改为 `hand_detections_${ID}.npz`
    - FoundationPose 物体目录改为 `foundation_input_${ID}`
    - 输出目录改为 `id${ID}_z005`
  - viewer 说明同步更新：
    - 若要开 viewer，建议先把 `seq 0 10` 改为单个 ID，例如 `seq 0 0`
    - 再追加 `--enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02`
  - 验证：
    - 对三个批量 loop 命令做 `bash -n` 语法检查，通过；未实际运行 33 个 replay

- 2026-05-20
  - 在 `COMMAND_LIBRARY.zh.md` 末尾新增 G 部分：
    - 添加 H2O 三任务 id0-id10 的 gripper/wrist-retreat 到 FoundationPose 物体中心的世界轴向距离曲线命令
    - 三个任务分别为 pick_diverse_bottles、place_bread_basket、stack_cups
    - 输出每个 id 的 PNG 与同名 CSV 到 `code_painting/human_object_replay/h2o/.../id${ID}_z005/`
  - 文档补充读图规则：
    - 若 `dz` 多任务/多 id 稳定同向偏移，优先怀疑 head/depth/camera-to-world 或 replay 标定链路
    - 若只在某个物体或某些帧跳变，优先怀疑 FoundationPose pose/depth/mesh 估计
  - 验证：
    - 三条 id0-id10 loop 命令 `bash -n` 通过
    - pick_diverse_bottles id0 运行 `--max_frames 2` smoke test 成功，生成 `/tmp/pick_diverse_bottles_axis_distance_id0_smoke.png` 和 `.csv`

- 2026-05-26
  - 新增 LeRobot cache episode 子集抽取脚本：
    - 新增 `policy/pi0/scripts/subset_lerobot_episodes.py`，用于从已生成的 `/home/zaijia001/.cache/huggingface/lerobot/local/<dataset>` 中抽取指定 episode。
    - 支持 `--episodes '0-24'`、`--episodes '0,1-5,7'`，解析后去重并按旧 episode id 升序排序。
    - 输出到新的 `--output-repo-id`，并重写 `episode_index`、`frame_index`、全局 `index`、`meta/info.json`、`meta/episodes.jsonl`、`meta/episodes_stats.jsonl`，使新数据连续编号为 `0..N-1`。
    - 更新 `COMMAND_LIBRARY.zh.md` L11 和 `agent-read/COMMANDS/pi0_h2o_training_data.*.md`，记录 25 episode 抽取命令和检查命令。
  - 验证：
    - `uv run python -m py_compile scripts/subset_lerobot_episodes.py` 通过。
    - 用 `local/h2o_pick_diverse_bottles_human_head_pure_action` 抽取 `0,1-2,7` 到临时 repo，检查得到 `total_episodes=4`、episode 重新编号 `0..3`、parquet 和三路视频均存在；验证后删除临时 repo。

- 2026-05-21
  - 更新距离曲线脚本 `plot_piper_gripper_wrist_object_axis_distances.py`：
    - 新增 `--plot_clip_abs_m`，默认 `0.5`
    - PNG 绘图时将超过 `±plot_clip_abs_m` 的值压到边界显示，便于观察 0.5m 以内趋势
    - CSV 仍保留未裁剪原始值，图标题会标注 clipping 与被裁剪数量
    - 某个 FoundationPose 物体目录缺失 `poses.npz` 时不再中断，改为打印 warning 并将该侧曲线写为 NaN
  - 数据观察：
    - H2O 三任务 id0-id10 已有 33 个 CSV；place_bread_basket id5/id6 缺 `bread/poses.npz`，脚本已允许缺失并生成左侧 basket 曲线
    - 统计时将 `|value|>0.5m` 作为大异常；正常帧整体 dz 中位数约 gripper `+0.150m`、wrist `+0.169m`
  - 验证：
    - `py_compile` 通过
    - pick_diverse_bottles id0 `--max_frames 2` clipped smoke test 通过
    - place_bread_basket id5/id6 在缺 bread track 的情况下生成 PNG/CSV

- 2026-05-21
  - 新增 HaMeR 原始手点与 FoundationPose 原始物体点的对比工具：
    - `code_painting/make_hamer_foundation_point_compare_video.py`
    - 输入 HaMeR `hand_detections_<id>.npz`、`hand_vis_gripper_<id>.mp4` 和 FoundationPose object dir
    - 输出横向拼接视频：HaMeR 手点面板 + 每个物体 `mesh_overlay.mp4` 面板
    - 视频叠加 thumb tip、index tip、thumb/index midpoint 与物体中心投影
    - 同名 CSV 记录相机坐标系下 `hand_midpoint - object_center` 的 `dx/dy/dz`
  - 更新 `COMMAND_LIBRARY.zh.md` H 部分：
    - 记录 H2O 三任务 id0-id10 的 G 部分 replay CSV 统计摘要
    - 新增 pick_diverse_bottles、place_bread_basket、stack_cups 三任务 id0-id10 原始点位对比命令
  - 数据观察：
    - 正常帧整体 `abs dz median` 约 gripper `15.1cm`、wrist-retreat `17.0cm`
    - signed dz median 约 gripper `+15.0cm`、wrist-retreat `+16.9cm`
    - pick/place 存在少量米级 z outlier，stack_cups 无 `>0.5m` outlier
  - 验证：
    - `py_compile` 通过
    - place_bread_basket id0 `--max_frames 5` smoke test 成功，生成 `/tmp/hamer_foundation_point_compare_place_bread_basket_id0.mp4` 和 `.csv`

- 2026-05-21
  - 扩展 `make_hamer_foundation_point_compare_video.py`：
    - 新增默认距离曲线 PNG 输出，文件名为 `*_distance.png`
    - 曲线显示 HaMeR thumb/index midpoint 到 FoundationPose object center 的相机坐标轴向 `dx/dy/dz`
    - 新增 `--output_plot` 可指定曲线输出路径
    - 新增 `--plot_clip_abs_m`，默认 `0.5`，与 G 部分一致只压缩 PNG 显示，不改 CSV 原始值
  - 更新 `COMMAND_LIBRARY.zh.md` H2/H6：
    - H2 说明视频、CSV、距离曲线 PNG 三种输出
    - H6 增加查找 `*_hamer_foundation_points_distance.png` 的命令
  - 验证：
    - `py_compile` 通过
    - place_bread_basket id0 `--max_frames 5` smoke test 成功，生成视频、CSV 和距离曲线 PNG

- 2026-05-21
  - 统计 H 部分新生成的原始 HaMeR/FoundationPose CSV：
    - 当前检测到 `pick_diverse_bottles` id0-id10，共 11 个 CSV
    - 相机坐标系 `hand_midpoint - object_center` 的正常帧整体 `abs dz median` 约 `5.1cm`
    - 对比 G/H1 的 world replay 统计，pick 的 gripper/wrist `abs dz median` 约 `14.6cm/16.5cm`
    - 结论：15cm 级 z 偏差不主要来自原始检测点位，而更可能来自 camera-to-world、`target_world_offset_xyz`、retreat 点定义和 replay 坐标链路叠加
  - 更新 `COMMAND_LIBRARY.zh.md`：
    - 在 E2.1/E2.2/E2.3 前新增 E2.0，只 replay 三个 H2O 任务的人手，不加载 FoundationPose 物体
    - 在 H1 后补充当前 H 原始 CSV 统计和与 G/H1 world replay 统计的差异说明
  - 验证：
    - E2.0 三条 id0-id10 pure hand replay loop 命令 `bash -n` 通过

- 2026-05-21
  - 更新 `COMMAND_LIBRARY.zh.md` E2.0：
    - 三个纯人手 replay 命令从 `--save_png_frames 1` 改为 `--save_png_frames 0`
    - 说明 `--save_png_frames 0` 不再保存 `frames/` 下逐帧 PNG，只保存 replay mp4/npz 等主输出
    - 增加 VS Code 预览兼容转码命令：`ffmpeg -c:v libx264 -pix_fmt yuv420p -movflags +faststart`
  - 说明：原始 replay mp4 可能使用 VS Code/Chromium 不支持的 codec 或 pixel format；对比视频能预览是因为 ffmpeg hstack 命令重编码成了 H.264/yuv420p
  - 验证：
    - E2.0 三条 loop 命令和单条 ffmpeg 转码命令 `bash -n` 通过

- 2026-05-21
  - 扩展 `COMMAND_LIBRARY.zh.md` 的 Piper H2O 调试/生成流程：
    - 增加 E0 pure replay，用干净 zed/third RGB 视频作为后续 repaint 的机器人源。
    - 增加 I/J/K，串起 SAM 人手抠除、pure replay 贴回、AnyGrasp 候选筛选、AnyGrasp keyframe replay 和 repaint。
  - 该轮只修改文档命令和 agent-read 日志，没有运行长耗时渲染/重绘任务。
  - 验证：
    - 新增命令区块抽取后 `bash -n` 通过。
    - 检查到 repaint/AnyGrasp 入口脚本存在。

- 2026-05-21
  - 修正 I 段 SAM repaint 文档命令，避免 Stage-1 人手抠除因为 E0 pure robot 视频未全部生成而全量跳过。
  - I1 改为使用 dummy robot video；I2 保持使用 E0 pure robot，并输出更具体的缺失路径。
  - 验证：I 段 bash 代码块 `bash -n` 通过。

- 2026-05-22
  - 修正 `COMMAND_LIBRARY.zh.md` I2/K2 的 Stage-1 背景输入路径。
  - 背景文件实际在 `stage1_human_inpaint/removed_w_mask_*.mp4`，因此命令加入 fallback，兼容不存在顶层 `human_hand_bg.mp4` 的情况。
  - 同步更新 I1 输出检查命令。
  - 验证：I/K2 repaint 命令块 `bash -n` 通过；抽样三任务 id0/id1/id10 均能找到现有 Stage-1 背景文件。

- 2026-05-22
  - 更新 `COMMAND_LIBRARY.zh.md` K 部分：
    - 在 K1 前补充 K0 人工关键帧筛选流程。
    - 增加基于 TSV 生成 `hand_keyframes_all.json` 的命令。
    - 增加按人工关键帧重跑 AnyGrasp preview summary 的命令。
    - 增加 bad id dry-run/移动命令，并同步记录 `_rejected_human_ids/rejected_ids.json`。
  - 验证：K0 bash 代码块 `bash -n` 通过。

- 2026-05-22
  - 参数化 `run_render_anygrasp_ranked_preview_keyframes_batch.sh` 的视频目录前缀：
    - 默认保持 `d_pour_blue`，兼容旧 AnyGrasp 流程。
    - 新增 `VIDEO_PREFIX=foundation_input` 支持 H2O task 的 `foundation_input_<id>` 目录。
  - 更新 `COMMAND_LIBRARY.zh.md` K0.3/K1：
    - K0.3 使用 batch wrapper 按整 task 生成人工关键帧 preview summary。
    - K1 改为处理整 task，不再限制 `id0-id10`。
  - 验证：wrapper 和 K0.3/K1 文档命令均 `bash -n` 通过。

- 2026-05-22
  - 在 `COMMAND_LIBRARY.zh.md` 中补充 D435 正常 head 视角记录：
    - 记录 `pick_diverse_bottles/origin/episode35/head_d435_rgbd_meta.json` 与 `harmer_input/params_35.json` 中的 headD435 来源、RGB/depth topic、`fx/fy/cx/cy`、`640x480` 分辨率。
    - 记录由真实内参换算的 `fovy_deg=42.499880046655484`，并说明默认 replay 使用 `640x360 + fovy_deg=90` 导致画面广角、Piper 投影偏小。
    - 新增 E2.4 D435 pure Piper replay 命令和 I3 D435 inpainting/repainting 命令，输出路径使用 `h2_pure_d435` 与 `results_repaint_piper_h2_d435`，文件名保留 `d435`。
  - 验证：D435 replay/repaint 文档命令已抽取做 `bash -n` 语法检查。
  - 继续补充 D435 FOV 来源说明：
    - 明确 `42.499880046655484°` 来自录制保存的 RGB camera_info 内参 `fy=617.160888671875, height=480`，不是根据图片尺寸猜测；仅凭 `640x480` 无法推断 FOV。
    - 补充官方 D435 depth FOV `85.2 x 58` 与 color camera FOV 标称 `H:69 / V:42 / D:77` 的区别；当前 replay 对齐 `/camera/color/image_raw`，因此保留 `fovy_deg=42.499880046655484`。
  - 继续补充 I3 D435 repaint 异常诊断：
    - 确认 I3 的 `BG` 路径仍指向 I1 的 `removed_w_mask_rgb_<id>.mp4`，不是直接选错背景。
    - 记录 Stage-2 合成语义：只在 robot mask 区域从 `robot_video` 拷贝原像素到 target BG。
    - 对比 id0 的旧版/D435 `w_mask/w_box/final`，确认 D435 的 GroundingDINO/SAM2 mask 覆盖了大片仿真背景，导致最终把 robot replay 背景一起贴回。
    - 记录原始人手视频与 D435 replay 均为 `640x480`；旧版 replay 为 `640x360`，但 Stage-2 会 resize 到 target BG 尺寸。

- 2026-05-23
  - 继续更新 `COMMAND_LIBRARY.zh.md` I3 D435 repaint 诊断：
    - 补充用户确认的根因：D435 窄 FOV 下部分视频第 0 帧/前几帧没有拍到机器人，Stage-2 首帧检测会误把白色仿真背景或桌面当作 robot mask，后续传播沿着错误目标走。
    - 记录减少该问题的设置建议：优先让 Stage-2 初始化帧包含清楚可见的机器人；仅靠阈值不能修复首帧完全无机器人的情况。
    - 建议 D435 从更保守的 `--robot_box_threshold 0.30~0.40`、`--robot_text_threshold 0.25~0.35`、`--robot_max_mask_area_ratio 0.20~0.35` 和更明确的 robot prompt 开始调参。
    - 新增 SAM3 直接复用 I1 背景的 Stage-2 robot repaint 指令，说明 SAM3 当前也固定首帧初始化，应先确认第 0 帧能看到机器人。
  - 验证：
    - 新增 I3.2/I3.3 bash 命令通过 `bash -n` 语法检查。

- 2026-05-25
  - 继续更新 `COMMAND_LIBRARY.zh.md` I3：
    - 新增 I3.4“可见帧重初始化 SAM 模式”设计记录，只写文档，不修改现有代码。
    - 对比当前固定第 0 帧初始化的 SAM2/SAM3 Stage-2 与新模式：新模式会先逐帧检测 robot，只有检测到有效 robot 后才初始化 SAM；中间 robot 消失时输出空 mask，重新出现后再次初始化。
    - 记录建议状态机、有效 mask 判断条件、待实现接口参数和独立输出目录 `results_repaint_piper_h2_d435_sam3_visible_reinit`，用于后续和当前 SAM 输出做 A/B 对比。
  - 验证：
    - I3.4 中的待实现命令通过 `bash -n` 语法检查。

- 2026-05-26
  - 修复 Piper AnyGrasp 两关键帧规划入口：
    - `code_painting/plan_anygrasp_keyframes_piper.py` 不再使用旧的 R1 单 base Piper wrapper。
    - 入口现在复用 `PiperDualReplayRenderer` 和 `HandRetargetPiperDualURDFIKRenderer`，与直接 replay 人手坐标的 Piper dual 路径一致。
    - 左右臂保留 `robot_config_PiperPika_agx_dual_table_0515.json` 中的独立 base pose，IK 前按 arm 做 world-to-base 转换。
  - 更新文档：
    - `COMMAND_LIBRARY.zh.md` 新增 L14，给出 Piper AnyGrasp dual-base Cartesian waypoint 调试命令。
    - 新增 `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md`。
  - 验证：
    - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_piper.py` 通过。

- 2026-05-28
  - 补充 D435 版本 AnyGrasp 候选筛选命令：
    - `COMMAND_LIBRARY.zh.md` 新增 J0.1，检查 6 task 的 AnyGrasp、`foundation_replay_d435`、HaMeR NPZ 是否齐全。
    - 新增 J1.1，使用人工关键帧 `hand_keyframes_all.json` 为 6 task 生成 D435 候选 preview/summary。
    - D435 summary 输出到 `code_painting/anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json`，避免和默认广角 `anygrasp_h2o_preview` 混用。
    - `place_bread_basket` 当前命令支持 fallback 到 `place_bread_basket_output_old_cam`。
    - 同步更新 `agent-read/COMMANDS/pi0_h2o_training_data.zh.md` / `.en.md`。
  - 验证：
    - 检查了 6 task 的 `foundation_replay_d435` 输入和若干 AnyGrasp 根目录。
    - 新增 J0.1/J1.1 命令块通过 `bash -n`。

- 2026-05-28
  - 调整 L11.2.4 D435 robot replay `_25ep` 抽取逻辑：
    - 不再盲抽 LeRobot episode `0-24`。
    - 改为读取 `processed_data/h2o_<TASK>_pure_d435_visible_reinit-120/episode_*/instructions.json/source_episode_id`，按原始 id 对齐。
    - 排除 `handover_bottle` 原始 bad id `0,7,12,29` 和 `pnp_bread` 原始 bad id `0,1,2,3,4,5,6,22,70`，再补足 25 个 episode。
    - 同步更新 `agent-read/COMMANDS/pi0_h2o_training_data.zh.md` / `.en.md`。
  - 验证：
    - 检查了 D435 processed data 中的 `source_episode_id` 字段。
    - 新增 L11.2.4 命令块通过 `bash -n`。

- 2026-05-28
  - 补充解释 D435 final 的生成来源和 SAM2 fallback 批处理入口：
    - `d435_final` 是 `results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/<TASK>/id_<ID>_d435/final_repainted.mp4`。
    - 该文件由 I3.5 的 `batch_visible_reinit_d435_repaint.py` 生成，不是 L8.2 生成。
    - 当前机器没有 `Grounded_SAM_3`，因此 I3.5 批处理实际日志为 `[backend] SAM=sam2, DINO=dino2`。
    - 该批处理会打印 `loading DINO once` 和 `loading SAM image predictor once`，即只加载一次 checkpoint 后循环处理 task/id。
    - `COMMAND_LIBRARY.zh.md` I3.5 新增“先补到至少 25 个 final”的 SAM2/DINO2 fallback 批处理命令。
  - 验证：
    - 检查了 `batch_visible_reinit_d435_repaint.py` 和 `remove_anything_video_sam3_robot_visible_reinit.py` 中的 checkpoint 加载和 `final_repainted.mp4` 复制逻辑。
    - 新增命令块通过 `bash -n`。

- 2026-05-28
  - 检查 `pro5-17` 中新三任务 D435 robot replay processed HDF5 数量偏少的问题：
    - `pro5-17` 正在/刚刚运行的是 L8.2 转换阶段，它只读取已有 `final_repainted.mp4`，不会生成缺失的 D435 repaint。
    - 当前新三任务 `h2_pure_d435` retarget 数量充足：`handover_bottle=51`、`pnp_bread=81`、`pnp_tray=51`。
    - 少的是 Stage-1 BG 和 I3.5 D435 repaint final；例如当前 final 数量约为 `handover_bottle=12`、`pnp_bread=2`、`pnp_tray=14`。
    - `COMMAND_LIBRARY.zh.md` 新增 I1.1.1，提供只补缺失 Stage-1 BG 的命令。
    - I3.5 新增 `--overwrite 0` 的 0..80 resume 命令，用于补齐缺失 D435 repaint final 后再运行 L8.2。
    - 同步更新 `agent-read/COMMANDS/pi0_h2o_training_data.zh.md` / `.en.md`。
  - 验证：
    - 检查了 `tmux capture-pane -pt pro5-17`，确认日志是 L8.2 缺 `final_repainted.mp4` 的 skip。
    - 新增命令块做了 `bash -n` 语法检查。

- 2026-05-28
  - 修复并复查 Piper D435 AnyGrasp 关键帧执行不到位问题：
    - `plan_anygrasp_keyframes_r1.py` 的 `target_pose_for_error(..., ee)` 现在对 Piper dual 按 arm 使用 `world_pose_to_base_pose_for_arm/base_pose_to_world_pose_for_arm`，避免右臂目标误用左臂 base。
    - `planned_eval_pose_from_plan()` 不再直接把 `target_pose_world` 当作 planned pose，而是通过 target joints 的 FK 和 robot gripper/endlink 静态变换做评估。
    - partial 诊断执行中，plan 失败的 arm 会持续 hold 当前关节，避免另一只 arm 执行时失败臂在物理仿真里漂移。
    - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` 新增 `--reach_error_pose_source` 参数，默认改为 `ee`。旧 `tcp` 检查会固定留下约 12 cm TCP/EE 偏移。
  - `pick_diverse_bottles id0` 复查结论：
    - 旧 `tcp` 检查下右臂 grasp 显示约 `0.125m` 误差；改用 `ee` 并修复 per-arm base 后，右臂 grasp 位置误差约 `0.0057m`。
    - id0 仍整体失败的直接原因是左臂第一关键帧 IK/目标失败；严格双臂同步会阻断整个 stage。
  - 文档：
    - `COMMAND_LIBRARY.zh.md` 新增 L15.7，记录当前关键帧执行逻辑、EE 到位判定原因、六任务分开运行命令。
    - 同步更新 `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md`。
  - 验证：
    - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_piper.py` 通过。
    - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` 通过。
    - six-task wrapper dry-run 覆盖六个任务各 1 个 summary 通过。
    - 复跑 `pick_diverse_bottles id0` partial/joint_interp/pose 打印，确认右臂 EE reached，左臂失败被 hold。

- 2026-05-28
  - 补充 Piper D435 AnyGrasp preview 对照和 viewer 目标可视化：
    - planner 现在会把复用 summary 对应的原始 D435 preview 图和同路径 legacy preview 图复制到 `<OUT>/source_preview_compare/`。
    - 新增 `selected_candidate_mapping.json`，记录每个 frame/arm 的 selected `candidate_idx`、rank、source translation、planner raw pose、planner target pose 和 `candidate_target_local_x_offset_m`。
    - six-task wrapper 新增 `--visualize_targets`，viewer 调试时自动关闭 `pure_scene_output` 并显示目标 axis/active candidate gripper。
  - 验证：
    - `pick_diverse_bottles id0` 输出 `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/2-pdb/pick_diverse_bottles/foundation_input_0/source_preview_compare/`，已包含 frame 38/78 的 D435 与 legacy `orientation/fused` 图和 mapping JSON。
    - `plan_summary.json` 已记录 `source_preview_compare` 13 个条目。
    - `py_compile` 和 wrapper `bash -n` 通过。

- 2026-05-28
  - 新增 Cartesian waypoint partial-prefix 执行诊断：
    - `plan_anygrasp_keyframes_r1.py` 新增 `--execute_partial_cartesian_plan`。
    - Piper/R1 URDFIK renderer 在 `cartesian_interp_ik` 中间 waypoint 失败时，可返回 `status=Partial` 和已成功求解的 waypoint 前缀。
    - 执行逻辑现在会执行 `Success` 或 `Partial` 且带有 trajectory 的 plan；`Partial` 不算 reached。
    - 六任务 wrapper 新增 `--execute_partial_cartesian_plan` 转发。
    - `COMMAND_LIBRARY.zh.md` 新增 L15.11 viewer/no-viewer 调试命令，并记录位置优先/朝向优先说明。
  - 验证：
    - 相关 Python 文件 `py_compile` 通过。
    - wrapper `bash -n` 通过。
    - L15.11 bash block 通过 `bash -n`。

- 2026-05-28
  - 根据用户要求调整 D435 AnyGrasp preview 约定：
    - 主目录 `anygrasp_h2o_preview_d435` 的 L15.9 命令改为 no-offset：`--candidate_target_local_x_offset_m 0.0`。
    - L15.10 改为 offset -5cm 对比目录：`anygrasp_h2o_preview_d435_offset_minus_5cm_compare`。
    - `render_anygrasp_ranked_preview.py` 新增 `planner_selected_image`，输出 `frame_XXXXXX_left_right_planner_selected_orientation_rank1.png`，只画 downstream planner 默认选择的 orientation rank1。
    - 文档补充 orientation/fused 区别与颜色含义：候选左手蓝色系、候选右手橙色系，人手参考左手绿色、右手紫色。
  - 验证：
    - `python3 -m py_compile code_painting/render_anygrasp_ranked_preview.py` 通过。
    - L15.9/L15.10 bash block 通过 `bash -n`。

- 2026-05-28
  - 在 `COMMAND_LIBRARY.zh.md` 末尾追加 L15.10 raw/no-offset 对比命令：
    - 使用 `--candidate_target_local_x_offset_m 0.0` 生成独立对比目录 `anygrasp_h2o_preview_d435_no_offset_compare`。
    - 提供 summary 对比脚本，用于检查 raw/original candidate 和 offset 后 visual/planner target 的差别。
  - 验证：
    - L15.10 bash block 通过 `bash -n`。

- 2026-05-28
  - 在 `COMMAND_LIBRARY.zh.md` 末尾追加 L15.9 复制安全版三步运行指令：
    - 第一步使用 `bash <<'BASH'` 重新生成六任务 D435 preview/summary，避免 zsh 断行丢参。
    - 第二步提供无 viewer D435 planner/replay。
    - 第三步提供 viewer + `--visualize_targets` gripper 目标可视化 replay。
  - 验证：
    - L15.9 bash block 通过 `bash -n`。

- 2026-05-28
  - 修正 AnyGrasp D435 preview 与 planner target 的可视化映射不一致：
    - `render_anygrasp_ranked_preview.py` 现在绘制 grasp wireframe 时使用 remap/post-rot/local-X offset 后的 `visual_translation_cam/visual_rotation_matrix`。
    - summary 继续保留原始 `translation_cam/rotation_matrix`，并新增 `visual_translation_cam/visual_rotation_matrix`，用于区分原始 AnyGrasp candidate 与实际 planner target。
    - 确认 `pick_diverse_bottles id0 frame 38` 的 D435 rank1 为 left candidate `16`、right candidate `11`；默认广角 preview 的候选 id 不同，不能混用。
  - 验证：
    - `python3 -m py_compile code_painting/render_anygrasp_ranked_preview.py code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_piper.py` 通过。
    - 单 id D435 preview 重生成到 `code_painting/anygrasp_h2o_preview_d435_offsetfix_debug/pick_diverse_bottles/foundation_input_0`，summary 可读出 raw/visual/world 三组坐标。

- 2026-05-28
  - 梳理并补齐 6 任务 D435 robot replay 数据处理链路：
    - `COMMAND_LIBRARY.zh.md` 在 L6.1 明确说明该命令只适用于默认广角 `h2_pure`，新三任务缺少 `h2_pure` 和默认广角 repaint head，所以运行 L6.1 会全部 skip。
    - 新增 L8.2：六任务 D435 visible-reinit robot repaint head + `h2_pure_d435` action/wrist 转 processed HDF5。
    - 新增 L10.6：六任务 `h2o_<TASK>_pure_d435_visible_reinit-120` 转 LeRobot cache。
    - 新增 L11.2.4：六任务 D435 robot replay `_25ep` 抽取、打包和上传检查。
    - 同步更新 `agent-read/COMMANDS/pi0_h2o_training_data.zh.md` / `.en.md` 的 D435 流程说明。
  - 当前路径检查：
    - 旧三任务有默认广角 `h2_pure` 和 D435 `h2_pure_d435`。
    - 新三任务只有 D435 `h2_pure_d435`，没有默认广角 `h2_pure`。
    - 新三任务 D435 visible-reinit repaint 当前只有部分输出，后续需要补跑 I1.1/I3.5 后再跑 L8.2/L10.6/L11.2.4。
  - 验证：
    - 新增 L8.2/L10.6/L11.2.4 命令块做了 `bash -n` 语法检查。

- 2026-05-28
  - 更新 H2O pi0 数据处理命令文档：
    - `COMMAND_LIBRARY.zh.md` 新增 L11.1.3，作为 L5.2 -> L10.5 后的新三任务 human head + D435 action/wrist `_25ep` 抽取步骤。
    - 明确 L10.5 后续不能使用 `local/h2o_<TASK>_pure_repaint`；该 repo 属于 L6/L6.1 robot replay 数据。
    - L11.1.3 会读取 processed data 的 `instructions.json/source_episode_id`，排除 `handover_bottle` 原始 id `0,7,12,29` 和 `pnp_bread` 原始 id `0,1,2,3,4,5,6,22,70`，再补足 25 个可用 episode。
    - 同步更新 `agent-read/COMMANDS/pi0_h2o_training_data.zh.md` / `.en.md`。
  - 验证：
    - 本轮检查到直接抽 `0-24` 会包含 `handover_bottle` 原始 id `7,12` 和 `pnp_bread` 原始 id `22`。
    - 文档命令块已做 `bash -n` 语法检查。

- 2026-05-26
  - 新增 Piper hand 原始 `origin/` 数据审核脚本：
    - `code_painting/review_piper_hand_origin.py`
    - 直接读取 `/home/zaijia001/ssd/data/piper/hand/<TASK>/origin/episode*/camera/color/headD435/*.png`，避免等到 AnyGrasp / foundation 阶段才筛坏数据。
    - 交互按 `b` / `d` 会把当前 episode 目录移动到同级 `bad/`，并写入 `<TASK>/origin_bad_review.json`。
  - 已按要求将 `/home/zaijia001/ssd/data/piper/hand/handover_bottle/origin/episode*` 全部移动到 `/home/zaijia001/ssd/data/piper/hand/handover_bottle/bad/`，移动后 `origin/` 中 episode 数量为 0，`bad/` 中 episode 数量为 51。
  - 更新文档：
    - `COMMAND_LIBRARY.zh.md` 新增 L13。
    - 新增 `agent-read/COMMANDS/piper_hand_origin_review.zh.md` / `.en.md`。
  - 验证：
    - `RoboTwin_openvla` 环境中 `python code_painting/review_piper_hand_origin.py --help` 通过。
    - 默认系统 `python3` 缺少 `cv2`，因此命令文档显式激活 `RoboTwin_openvla`。

- 2026-05-25
  - 实现 I3.4 可见帧重初始化 SAM 模式：
    - 新增 `/home/zaijia001/ssd/inpainting_sam3_robot/remove_anything_video_sam3_robot_visible_reinit.py`，不修改原有 SAM2/SAM3 脚本接口。
    - 新增 `/home/zaijia001/ssd/inpainting_sam3_robot/batch_visible_reinit_d435_repaint.py`，批处理时只加载一次 DINO/SAM checkpoint，然后循环处理 task/id。
    - 单视频脚本逻辑：`inactive` 状态逐帧检测 robot；检测到有效 candidate 后初始化 SAM；`tracking` 状态用上一帧 bbox 扩张后作为当前帧 SAM box prompt；mask 失效或 robot 消失时输出空 mask，并等待后续帧重新检测/初始化。
    - 更新 `COMMAND_LIBRARY.zh.md` I3.4，把原“设计记录”改为已实现模式说明，并补充单视频、批处理和 dry-run 命令。
  - 验证：
    - `python3 -m py_compile` 通过两个新脚本。
    - 在 `inpainting-sam3-dino3` 环境中两个新脚本 `--help` 均可运行。
    - 批处理 `pick_diverse_bottles id0` dry-run 可正确解析 I1 BG 与 D435 robot replay 输入。
    - I3.4 文档中的单视频/批处理/dry-run 命令通过 `bash -n`。

- 2026-05-25
  - 修复 I3.4 新脚本在 `inpainting-sam3-dino3` 环境中的 GroundingDINO 加载兼容问题：
    - 用户运行时报 `AttributeError: 'BertModel' object has no attribute 'get_head_mask'`。
    - 原因是 `Grounded_SAM_2` 的旧 GroundingDINO `BertModelWarper` 依赖 transformers 旧版 `BertModel.get_head_mask`，当前环境的 transformers 已不再暴露该方法。
    - 在 `remove_anything_video_sam3_robot_visible_reinit.py` 中新增局部兼容 patch：运行新脚本时动态给 `transformers.BertModel` 补 `get_head_mask`，不修改第三方源码，不影响原始 SAM2/SAM3 脚本接口。
  - 更新 `COMMAND_LIBRARY.zh.md`：
    - 把 I3.3 标题改清楚为“当前 SAM3 项目首帧初始化”，说明实际后端以 `[backend] SAM=...` 日志为准。
    - 把 I3.4 标题改清楚为“新逻辑：可见帧重初始化 SAM2/SAM3 模式”。
    - 明确列出原来 SAM2 指令、当前 SAM3 项目指令、新逻辑指令三者的区别。
  - 验证：
    - patch 前当前环境 `BertModel.get_head_mask=False`，patch 后为 `True`。
    - `VisibleReinitRobotSegmenter` 在 `inpainting-sam3-dino3` 环境中可完成 DINO/SAM 模型加载，输出 `[ok] model loaded`。
    - 两个新脚本 `py_compile` 通过。

- 2026-05-25
  - 继续修复 `inpainting-sam3-dino3` 环境中的旧 GroundingDINO / 新 transformers 兼容问题：
    - 用户继续遇到 `TypeError: to() received an invalid combination of arguments - got (dtype=torch.device, )`。
    - 原因是旧 `BertModelWarper` 按旧 transformers API 调用 `get_extended_attention_mask(attention_mask, input_shape, device)`，但当前 transformers 5.3.0 的第三个参数是 `dtype`。
    - 在 `/home/zaijia001/ssd/inpainting_sam3_robot/remove_anything_video_sam3_robot.py` 中新增统一兼容函数 `patch_transformers_bert_for_groundingdino()`，同时 patch `get_head_mask` 和 `get_extended_attention_mask`。
    - `/home/zaijia001/ssd/inpainting_sam3_robot/remove_anything_video_sam3_robot_visible_reinit.py` 改为复用该统一兼容函数。
  - 更新 `COMMAND_LIBRARY.zh.md`：
    - 新增 I3.0 对照小节，明确列出原 SAM2 固定首帧、SAM3 项目固定首帧、新逻辑可见帧重初始化三套指令的入口、后端、初始化逻辑和输出目录。
    - 补充 I3.0.1 原 SAM2 单 id 指令、I3.0.2 SAM3 项目单 id 指令、真正 SAM3 backend 模板、I3.0.3 新逻辑单 id 指令、I3.0.4 新逻辑批处理指令。
    - 明确当前机器只检测到 `Grounded_SAM_2`，没有 `Grounded_SAM_3`，因此 SAM3 项目命令当前会 fallback 到 SAM2/DINO2。
  - 验证：
    - 三个相关脚本 `py_compile` 通过。
    - 在 `inpainting-sam3-dino3` 环境中完成一次 DINO forward smoke test，输出 `[ok] detect call completed 0`，确认 `get_head_mask` 和 `get_extended_attention_mask` 两个兼容错误都已绕过。
    - I3.0 新增命令块 `bash -n` 通过。
## 2026-05-29（修正 D435 AnyGrasp viewer 第一关键帧目标显示）

- 修正 `plan_anygrasp_keyframes_r1.py` 的执行预览帧选择：
  - `DebugExecutionState` 增加 `active_frame_by_arm`。
  - 双臂 pregrasp/grasp 阶段按左右手各自第一关键帧显示 target/candidate gripper；action 阶段才切到各自第二关键帧。
  - `record_frame()`、`update_candidate_debug_visuals()` 和 debug execution preview 现在使用 per-arm active frame，避免只用左手 `active_frame` 导致 viewer 中显示后一帧或错误帧。
  - `pose_debug.jsonl` 和 `execution_metrics.jsonl` 记录 `active_frame_by_arm`，用于复查 viewer 当前显示帧。
- 文档同步：
  - `COMMAND_LIBRARY.zh.md` 新增 L15.14，说明该问题的原因、修正和 `jq` 检查方式。
  - `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` 同步 L15.14。
- 验证：
  - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py` 通过。
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` 通过。
  - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --tasks pick_diverse_bottles --id_start 0 --id_end 10 ...` 可解析出 id0-10 共 11 个 D435 summary。
  - 非 viewer 实测 `pick_diverse_bottles id0 --debug_stop_after_keyframe1` 输出到 `/tmp/anygrasp_active_frame_by_arm_check`，`pose_debug.jsonl` 显示 `stage=pregrasp active_frame_by_arm={"left": 38, "right": 38}`，第一关键帧显示状态正确。
## 2026-05-29（Stack Cups id0 无碰撞 target-only 调试）

- 增加 D435/Piper 六任务 wrapper 调试开关：
  - `--disable_execution_collisions`：将 planner 的 `--enable_grasp_action_object_collision` 设为 `0`，用于排除 grasp/action 物体碰撞和 contact-stop close 逻辑。
  - `--target_axes_only`：只保留当前执行 target 轴，隐藏候选 gripper 轴、selected-keyframe 轴和 IK waypoint marker，避免 viewer 中多套坐标系混淆。
  - 增加 `--debug_candidate_top_k`、`--debug_common_candidate_top_k`、`--debug_visualize_selected_keyframe_axes`、`--debug_visualize_ik_waypoints` 透传参数。
- `plan_anygrasp_keyframes_r1.py` 增加 `--debug_visualize_selected_keyframe_axes`，执行预览可隐藏 selected-keyframe axis actors。
- 复查 `stack_cups/foundation_input_0`：
  - D435 summary 是 per-arm keyframes：left `[139, 195]`，right `[51, 106]`。
  - 因此 pregrasp 阶段 `active_frame_by_arm={"left": 139, "right": 51}` 是预期行为。
  - 之前 viewer 中“四个坐标系”来自当前 target 轴、selected-keyframe 轴、candidate gripper 轴和 IK waypoint marker 同时显示。
- 验证：
  - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py` 通过。
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` 通过。
  - `--dry_run --tasks stack_cups --ids 0 --target_axes_only --disable_execution_collisions ...` 可正确解析。
  - 无 viewer 实测 `/tmp/stack_cups_id0_no_collision_target_axes_only` 确认 `enable_grasp_action_object_collision=0`，但 stack_cups id0 仍未到位；说明当前主要问题不是物体碰撞，而是 IK/轨迹执行后的关节跟踪偏差和候选姿态可达性。
## 2026-05-29（补充 direct Piper hand replay viewer 对照命令）

- `COMMAND_LIBRARY.zh.md` 新增 L15.16：
  - 提供 `stack_cups id0` 直接 replay HaMeR NPZ gripper pose 的 viewer 命令。
  - 同时显示目标 gripper 坐标轴和机器人执行。
  - 打开 `--debug_mode 1 --debug_post_execute 1`，用于查看目标 EE/TCP pose 与实际执行之间的误差。
  - 保存 `world_targets_and_status.npz`，便于后续读取 target pose/status。
- 同步更新 `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` 和 command changelog。
## 2026-05-29（补充 direct replay 与 AnyGrasp 轴约定差异）

- 记录 direct replay 与 AnyGrasp planner 的 gripper frame 差异：
  - direct Piper hand replay 的 stored gripper frame 使用 local `+Z` 蓝轴作为 approach/forward 轴。
  - AnyGrasp preview wireframe 使用 `rotation_matrix[:, 0]`，即 local `+X` 红轴，作为掌根到指尖的 finger-depth 方向。
- `plan_anygrasp_keyframes_r1.py` 的参数说明和诊断注释已改为明确区分 AnyGrasp local `+X` 与 direct replay local `+Z`。
- `COMMAND_LIBRARY.zh.md` 新增 L15.17，提供 `stack_cups id0` 的 `--candidate_orientation_remap_label swap_red_blue` 单条对照命令，用于测试是否应把 AnyGrasp local `+X` 映射到 direct replay local `+Z`。
- `.gitignore` 增加 `policy/openvla-oft/runs/`，避免 96G 运行输出进入 git；同时忽略 `*.bak` / `*.bak2` 备份文件。

## 2026-05-29（新增 replay-axis AnyGrasp 关键帧执行入口）

- 新增 planner 参数：
  - `--candidate_target_local_z_offset_m`：沿 target local `+Z` 对 AnyGrasp target 做补偿。
  - `--approach_axis local_x|local_z`：控制 pregrasp retreat 使用 local `+X` 还是 local `+Z`。
- 旧默认不变：
  - `--candidate_target_local_x_offset_m` 默认语义仍保留。
  - `--approach_axis` 默认 `local_x`，保持旧 AnyGrasp planner 行为。
- 六任务 D435 wrapper 新增透传参数：
  - `--candidate_orientation_remap_label`
  - `--candidate_target_local_x_offset_m`
  - `--candidate_target_local_z_offset_m`
  - `--approach_axis`
  - `--approach_offset_m`
- 新增独立入口 `code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh`：
  - 固定 `swap_red_blue`。
  - 固定关闭 local-X compensation，启用 `--candidate_target_local_z_offset_m -0.05`。
  - 固定 `--approach_axis local_z`。
- 文档同步：
  - `COMMAND_LIBRARY.zh.md` 新增 L15.18，说明 direct replay local +Z 蓝轴与 AnyGrasp local +X 红轴的差异，以及六任务 viewer/no-viewer 命令。
  - `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` 同步 L15.18。
- 验证：
  - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py` 通过。
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` 通过。
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh` 通过。
  - `run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --tasks stack_cups --ids 0 --debug_stop_after_keyframe1 ...` 无 viewer 实测通过，输出到 `/tmp/stack_cups_id0_replay_axes_check/stack_cups/foundation_input_0`；`plan_summary.json` 记录 `candidate_target_local_z_offset_m=-0.05`、`approach_axis=local_z`，并生成 VSCode 兼容 `head_cam_plan.mp4` / `third_cam_plan.mp4`。该调试 run 中 keyframe1 grasp 左右手均 reached。

## 2026-05-29（L15.19 筛选阶段统一 gripper/robot frame 设计记录）

- 按用户要求仅更新文档，不修改代码。
- `COMMAND_LIBRARY.zh.md` 末尾新增 L15.19：
  - 记录为什么应在候选筛选阶段统一 AnyGrasp gripper frame 与 robot/replay frame。
  - 明确 viewer 坐标轴颜色仍是红/绿/蓝对应 local +X/+Y/+Z。
  - 明确 rank preview 中橙色是右手/candidate gripper 颜色，不是坐标轴颜色。
  - 记录长期推荐 frame：`robot local +Z = AnyGrasp raw local +X`，`robot local +Y = AnyGrasp raw local +Y`，`robot local +X = -AnyGrasp raw local +Z`。
  - 增加当前可运行对照命令，输出到 `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper`。
- 同步更新 `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md`。

## 2026-05-29（实现 L15.19 robot-frame preview 与 planner 路径）

- 代码实现：
  - `render_anygrasp_ranked_preview.py` 新增 `--candidate_frame_mode robot_replay` 和 `--candidate_target_local_z_offset_m`。
  - robot-frame candidate rotation 保存为 `target local +Z = AnyGrasp raw local +X`、`target local +Y = AnyGrasp raw local +Y`、`target local +X = -AnyGrasp raw local +Z`。
  - 2D preview 的 C 形 gripper wireframe 支持 local-Z 作为指尖方向。
  - `plan_anygrasp_keyframes_r1.py` 新增 `--debug_gripper_actor_forward_axis local_z`，用于 viewer/rank preview 中把 C gripper actor 沿蓝色 local +Z 画出指尖方向。
  - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` 新增 `--preview_root` 和 `--debug_gripper_actor_forward_axis` 透传。
- 新增脚本：
  - `code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh`
  - `code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh`
- 文档：
  - `COMMAND_LIBRARY.zh.md` 新增 L15.19.1，记录生成 robot-frame preview 与 viewer_gripper planner 的运行命令。
  - `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` 同步命令。
- 验证：
  - `python3 -m py_compile code_painting/render_anygrasp_ranked_preview.py code_painting/plan_anygrasp_keyframes_r1.py` 通过。
  - `bash -n code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh`、`bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh`、`bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` 均通过。
  - `run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh --tasks pick_diverse_bottles --ids 0` 实测生成 robot-frame summary，`summary.json` 记录 `candidate_frame_mode=robot_replay`、`candidate_target_local_z_offset_m=-0.05`。
  - `run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --tasks pick_diverse_bottles --ids 0 --debug_stop_after_keyframe1 ...` 无 viewer smoke run 成功生成 `head_cam_plan.mp4` / `third_cam_plan.mp4`。该调试 run 验证新路径可执行；右手仍未完全 reach，说明后续仍需继续调 IK/候选可达性，而不是入口或 frame 写入失败。

## 2026-05-29（修复 robot-frame planner 只跑 id0 的问题）

- 问题原因：
  - `run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh` 按 robot-frame preview summary 列表运行。
  - 当时 `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435_robot_frame` 里只有 `pick_diverse_bottles/foundation_input_0/summary.json`，所以 planner 只能跑 id0。
- 修复：
  - robot-frame preview wrapper 新增 `--max_per_task`、`--id_start`、`--id_end`、`--skip_existing` 和 `--source_preview_root`。
  - robot-frame planner wrapper 现在会在 planner 前按同样的 task/id/max 范围自动补齐缺失的 robot-frame summary。
  - 新增 `--skip_preview_generation`，用于只使用已有 summary，不自动生成。
- 验证：
  - `bash -n code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh` 通过。
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh` 通过。
  - `run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --max_per_task 2 --dry_run --tasks pick_diverse_bottles ...` 自动补齐 id1 summary，并在 planner dry-run 中列出 id0/id1。
  - `run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --max_per_task 1 --dry_run ...` 对六个任务均能进入 planner dry-run；部分任务首个可用 id 不是 0，这是由已有 D435 preview summary 可用 id 决定。

## 2026-05-29（补充 robot-frame 指定 id viewer 命令）

- `COMMAND_LIBRARY.zh.md` 新增 L15.19.2：
  - `stack_cups id4` viewer 调试命令。
  - 六任务分别指定 id 的 viewer 模板。
  - 六任务同时指定同一组 id 的 viewer 命令。
- `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` 同步指定 id 命令。
- 验证：
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh` 通过。
  - `bash -n code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh` 通过。
  - `run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --ids 4 --dry_run --tasks stack_cups ...` 正确列出 `stack_cups/foundation_input_4/summary.json`。

## 2026-06-02（Mode O Piper gripper 轴约定检查与中文注释）

- 检查 Piper/Pika 与 ALOHA-AgileX 的夹爪朝向定义：
  - 高层配置 `global_trans_matrix=diag(1,-1,-1)`、`delta_matrix=I`、`grasp_perfect_direction=["front_right","front_left"]` 一致。
  - URDF 夹爪结构轴不同：ALOHA-AgileX 的 finger depth/指尖方向更自然对应 link6 local `+X`，Piper/Pika 的夹爪开合轴在 gripper base local `Z/-Z`。
  - 当前 Mode O 沿用 Piper/replay target frame，使用 local `+Z` 作为接近/前进轴；这和 direct replay / robot-frame AnyGrasp 一致，但不是原始 ALOHA-style local `+X` 约定。
- `code_painting/plan_first_frame_foundation_pick_diverse_bottles.py` 增加中文注释，说明文件位置、目标生成逻辑、pose 存储顺序、local `+Z` 接近轴约定，以及和 ALOHA-style local `+X` 的差异。
- `COMMAND_LIBRARY.zh.md` 与 `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` 同步新增 Mode O 夹爪朝向检查说明。
- 验证：
  - `python3 -m py_compile code_painting/plan_first_frame_foundation_pick_diverse_bottles.py` 通过。

## 2026-06-02（Mode O gripper frame 可视化与 ALOHA-style local-X 对照）

- 新增 `code_painting/visualize_mode_o_gripper_frame_conventions.py`：
  - 不跑 IK，不启动 SAPIEN viewer。
  - 读取 `pick_diverse_bottles` FoundationPose 物体位置。
  - 在同一 grasp target 上画出 `piper_local_z`、`aloha_local_x_y_up`、`aloha_local_x_z_up` 三套 frame。
  - 输出 PNG 与 JSON，JSON 记录 local X/Y/Z 与物理接近方向的夹角。
- `code_painting/plan_first_frame_foundation_pick_diverse_bottles.py` 新增：
  - `--target_frame_convention piper_local_z|aloha_local_x_y_up|aloha_local_x_z_up`
  - `--plan_only`
  - 当使用 `aloha_local_x_*` 时，planner 自动使用 `--approach_axis local_x` 和 `--debug_gripper_actor_forward_axis local_x`。
- `code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh` 同步透传 `--target_frame_convention` 与 `--plan_only`。
- `.gitignore` 放行新增可视化脚本，继续忽略生成的 PNG/JSON 调试输出。
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_first_frame_foundation_pick_diverse_bottles.py code_painting/visualize_mode_o_gripper_frame_conventions.py` 通过。
  - `bash -n code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh` 通过。
  - 可视化脚本对 `pick_diverse_bottles id0 frame0` 成功输出 PNG/JSON；结果显示 `piper_local_z` 的 local Z 与物理接近方向夹角为 `0deg`，两个 ALOHA-style local-X 变体的 local X 与物理接近方向夹角为 `0deg`。
  - `--plan_only --target_frame_convention aloha_local_x_z_up` 成功写出 summary，记录 `target_frame_convention=aloha_local_x_z_up`、`planner_approach_axis=local_x`。

## 2026-06-02（O.0 原始 demo_clean 逻辑 Piper 数据生成入口）

- 新增 `envs/pick_diverse_bottles_piper.py`：
  - 继承原始 `pick_diverse_bottles`。
  - 不修改 `envs/pick_diverse_bottles.py`。
  - 保留原始瓶子随机采样、随机旋转、左右区域、`grasp_actor`、lift 和 place 逻辑。
- 新增 `task_config/demo_clean_piper.yml`：
  - 基于 `demo_clean.yml`。
  - `embodiment` 改为 `[piper, piper, 0.60]`。
- 新增 `description/task_instruction/pick_diverse_bottles_piper.json`：
  - 复用原始 `pick_diverse_bottles` 指令模板，保证 `collect_data.py` 结束后能生成 episode instructions。
- `.gitignore` 放行 `task_config/demo_clean_piper.yml`。
- 推荐命令：
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper 0`
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile envs/pick_diverse_bottles_piper.py script/collect_data.py` 通过。
  - conda 环境中动态 import `envs.pick_diverse_bottles_piper` 成功，class MRO 显示继承 `pick_diverse_bottles`。
  - `task_config/demo_clean_piper.yml` 解析得到 `embodiment=['piper','piper',0.60]`、`episode_num=50`。
  - `description/task_instruction/pick_diverse_bottles_piper.json` 解析得到 `seen=50`、`unseen=10`。

## 2026-06-02（修复 gen-23 O.0 collect_data 指令）

- `gen-23` 原始错误：
  - 在 `~` 目录执行 `bash collect_data.sh ...`，找不到脚本。
  - 进入仓库后，旧 `demo_clean_piper.yml` 的 `embodiment: [piper]` 触发双臂 embodiment 路径，RoboTwin 试图加载不存在的 `assets/embodiments/piper/curobo_left.yml`。
  - 后续 `'Robot' object has no attribute 'left_planner'` 是 planner 初始化失败后 task 复用残留 robot 的次生错误。
- 修复：
  - `task_config/demo_clean_piper.yml` 改为 `embodiment: [piper, piper, 0.60]`，使用两只单臂 Piper 与 `curobo.yml`。
  - O.0 文档命令改为带 conda 激活和仓库路径的完整命令。
- 验证：
  - `task_config/demo_clean_piper.yml` 解析得到 `['piper', 'piper', 0.6]`。
  - 带 conda 激活的 `timeout 35s bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper 0` 未再出现 `curobo_left.yml` 或 `left_planner` 错误。
  - 清理了该 smoke test 产生的 `data/pick_diverse_bottles_piper/demo_clean_piper/` 临时轨迹输出，避免影响用户后续从 episode 0 开始采集。

## 2026-06-03（O.0 切换到标定 Piper/Pika embodiment）

- 问题：
  - 用户成功生成了 `data/pick_diverse_bottles_piper/demo_clean_piper/`，但画面不像标定 Piper。
  - 检查发现旧 `demo_clean_piper.yml` 虽然不再使用 ALOHA-AgileX，但加载的是 RoboTwin 自带 `assets/embodiments/piper/config.yml` 和 `piper.urdf`，不是标定 `robot_config_PiperPika_agx_dual_table_0515.json` 对应的 `piper_pika_agx`。
- 修复：
  - 新增 `assets/embodiments/piper_pika_agx/config.yml`，使用标定 `piper_pika_agx.urdf`、Piper/Pika 夹爪 joint、左右 base pose、`delta_matrix=I`、`global_trans_matrix=diag(1,-1,-1)`。
  - `task_config/_embodiment_config.yml` 新增 `piper_pika_agx_calibrated`。
  - `task_config/demo_clean_piper.yml` 改为 `embodiment: [piper_pika_agx_calibrated, piper_pika_agx_calibrated, 0.0]`。
  - 新增 `task_config/demo_clean_piper_calibrated.yml`，建议用它生成新的独立输出目录，避免和旧 `demo_clean_piper` 数据混淆。
  - `.gitignore` 放行 `assets/embodiments/piper_pika_agx/config.yml` 和 `task_config/demo_clean_piper_calibrated.yml`，但继续忽略生成数据。
- 新推荐命令：
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0`
- 验证：
  - Python/YAML 检查确认 `demo_clean_piper` 和 `demo_clean_piper_calibrated` 都解析到 `assets/embodiments/piper_pika_agx/piper_pika_agx.urdf`，gripper base 为 `left_joint`，robot pose 为标定左右 base pose。

## 2026-06-03（gen1-2 O.0 head-only 配置与错误定位）

- 检查 `tmux gen1-2`：
  - `No left camera link` / `No right camera link` 是 `envs/robot/robot.py` 找不到 wrist camera link 后的 fallback 警告，不是本次 seed 失败的直接异常。
  - 真正失败发生在 seed/premotion 阶段；episode 0 从 seed 421 到 730 反复失败。
  - 失败类型主要是瓶子物理不稳定 `Objects is unstable ... 001_bottle`，以及原始 scripted demo 逻辑生成不了可执行 move target 的 `target_pose cannot be None for move action`。
- 修改：
  - `task_config/demo_clean_piper.yml` 和 `task_config/demo_clean_piper_calibrated.yml` 改为 `collect_wrist_camera: false`，只保存 head 视角。
  - 新增 `task_config/demo_clean_piper_calibrated_viewer.yml`：`render_freq: 1`、`episode_num: 1`、`collect_data: false`、`collect_wrist_camera: false`，用于单 episode viewer/head-only 调试。
  - `.gitignore` 放行新增 viewer 配置。
- 结论：
  - 关闭 wrist 保存可以减少 wrist link 依赖和日志干扰。
  - 如果仍连续出现 `target_pose cannot be None`，问题在原始 ALOHA-style `pick_diverse_bottles.py` demo 规划与标定 Piper/Pika 的几何/可达性不匹配，下一步应写 Piper/Pika 专用任务逻辑。

## 2026-06-03（修复 O.0 viewer 命令无输出与纯场景查看）

- 重新检查 `tmux gen1-2`：
  - 最小 `probe_sapien_viewer.py` 已能创建 viewer，说明当前 VNC/显示环境本身可用。
  - 之前文档中的 viewer 命令包含 `bash ./script/.update_path.sh ... && python ...`，但仓库没有 `script/.update_path.sh`；该命令被重定向吞掉错误后直接返回，所以看起来“无反应”。
  - 直接用 `script/collect_data.py` 作为 viewer 入口也不合适，因为它会继续进入原始 `play_once` / `grasp_actor` 规划，当前标定 Piper/Pika 下仍会出现 `target_pose cannot be None for move action`。
- 修改：
  - 新增 `run_collect_piper_calibrated_viewer.sh`，去掉不存在的 `.update_path.sh`，并在 viewer 模式下 `unset CUDA_VISIBLE_DEVICES`、自动设置 NVIDIA Vulkan ICD。
  - 新增 `view_pick_diverse_bottles_piper_scene.py` 和 `run_view_pick_diverse_bottles_piper_scene.sh`，只加载 `pick_diverse_bottles_piper` 场景，不进入 `play_once` 规划，自动跳过不稳定 seed 后停在 SAPIEN viewer。
  - 文档中的首选 viewer 命令改为 `bash run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50`。

## 2026-06-03（gen1 viewer 完成语义与 no-viewer 生成说明）

- 重新检查 `tmux gen1`：
  - 纯场景 viewer 已成功加载标定 Piper/Pika；seed 0/1 因瓶子不稳定被跳过，seed 2 加载成功。
  - 该 viewer 入口用于交互检查，会停在渲染循环等待用户关闭窗口或 `Ctrl-C`，不会自动执行完整 demo 或生成数据。
  - 关闭 SAPIEN 窗口时曾出现 `AttributeError: 'NoneType' object has no attribute 'should_close'`。
  - 无 viewer 生成命令 `bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0` 能启动 head-only 标定配置，但仍在 seed 搜索阶段失败：`Objects is unstable` 和 `target_pose cannot be None for move action`。
- 修改：
  - `view_pick_diverse_bottles_piper_scene.py` 在 viewer window 关闭或 `window=None` 时优雅退出，避免关闭窗口后的 traceback。
  - `COMMAND_LIBRARY.zh.md` 与 O.0 命令文档补充 viewer 不自动结束的语义，以及 no-viewer 生成命令和当前失败原因。

## 2026-06-03（修复 Mode M/N viewer CUDA mask 回写）

- 检查 `tmux modeln-4` 后确认：
  - Mode N `pnp_tray` 命令确实传入了 `--viewer`，planner 也尝试创建 interactive viewer。
  - 失败日志为 `[viewer] creating interactive viewer ... CUDA_VISIBLE_DEVICES=2` 后接 `Renderer does not support display`。
  - 用户的最小探针在同一类图形环境中 `unset CUDA_VISIBLE_DEVICES` 后可以打开 SAPIEN viewer。
- 根因：
  - bash wrapper 在 viewer 模式已经 `env -u CUDA_VISIBLE_DEVICES`。
  - 但 `plan_keyframes_human_replay.py` 和 `plan_keyframes_foundation_pose.py` 调 planner 前又根据 `--gpu` 写回 `CUDA_VISIBLE_DEVICES=2`。
- 修复：
  - 两个 Python 中间层在 `--enable_viewer 1` 时从传给 planner 的环境中移除 `CUDA_VISIBLE_DEVICES`。
  - 非 viewer 模式仍按 `--gpu` 设置计算 GPU。
- 文档：
  - `COMMAND_LIBRARY.zh.md` 的 Mode M/N viewer 段落补充 viewer 需要 `DISPLAY` 和 unset CUDA mask。
  - `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` 补充排查依据、最小探针和验证命令。
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_keyframes_foundation_pose.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_keyframes_human_replay.py` 通过。
  - `timeout 60s bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh --gpu 2 --ids 0 --viewer --viewer_wait_at_end 0 --tasks pnp_tray --foundation_pose_retreat_m 0.03 --output_root /tmp/robotwin_viewer_env_probe` 通过；当前非图形 shell 中 `DISPLAY=None`，因此 viewer fallback 到 offscreen，但关键日志已变为 `CUDA_VISIBLE_DEVICES=None`。

## 2026-06-03（O.0 motion baseline 跑通标定 Piper/Pika 后续运动）

- 问题复查：
  - `run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50` 是纯场景 viewer；它只加载稳定 seed 并停在 SAPIEN viewer，不会执行 `play_once` 或后续运动。
  - `collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0` 仍会进入原始 `pick_diverse_bottles.py` 的 `grasp_actor`；`tmux gen1-1/gen1-2` 中反复出现 `Objects is unstable` 和 `target_pose cannot be None for move action`，根因是原始 ALOHA-style EE 抓取目标生成不适配标定 Piper/Pika。
- 修改：
  - 新增 `envs/pick_diverse_bottles_piper_motion.py`，继承原始 `pick_diverse_bottles` 的瓶子随机采样、旋转、左右摆放区域和稳定性检查，但不调用原始 `choose_grasp_pose/grasp_actor`。
  - 该任务使用标定 Piper/Pika home pose 附近的保守关节空间阶段：approach、lower、close gripper、lift/retract、move outward、open gripper。
  - 新增 `task_config/demo_clean_piper_motion.yml` 用于无 viewer/head-only 数据保存，新增 `task_config/demo_clean_piper_motion_viewer.yml` 和 `run_pick_diverse_bottles_piper_motion_viewer.sh` 用于带运动 viewer 检查。
  - 新增 `description/task_instruction/pick_diverse_bottles_piper_motion.json`，让 `collect_data.py` 能保存 instruction 文件。
  - 在 `envs/pick_diverse_bottles_piper_motion.py` 中加入中文注释，明确它是 joint-space motion baseline，不代表真实瓶子抓取已经解决。
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile envs/pick_diverse_bottles_piper_motion.py view_pick_diverse_bottles_piper_scene.py` 通过。
  - `bash -n run_pick_diverse_bottles_piper_motion_viewer.sh run_view_pick_diverse_bottles_piper_scene.sh run_collect_piper_calibrated_viewer.sh` 通过。
  - YAML/JSON 解析检查通过，`demo_clean_piper_motion` 与 viewer 配置均使用 `piper_pika_agx_calibrated+piper_pika_agx_calibrated`，且 `collect_wrist_camera: false`。
  - 无 viewer 命令 `timeout 180s bash collect_data.sh pick_diverse_bottles_piper_motion demo_clean_piper_motion 0` 已成功：seed 0/1 因瓶子不稳定跳过，seed 2 `simulate data episode 0 success`，保存 64 帧 head-camera 数据、`episode0.hdf5`、`episode0.mp4`、`episode0.pkl` 和 instruction json。
  - `tmux gen1-1` 中执行 `bash run_pick_diverse_bottles_piper_motion_viewer.sh` 已成功跑到 seed 2 premotion 并返回 shell。

## 2026-06-03（O.0 原始 IK/规划链路与 Piper/Pika TCP 实验）

- 问题复查：
  - 当前能生成数据的 `pick_diverse_bottles_piper_motion` 没有调用原始 `pick_diverse_bottles.py` 的 IK/规划链路，而是直接生成关节空间插值。
  - 原始任务链路为 `grasp_actor/place_actor -> Action(move target_pose) -> Base_Task.move -> robot.left/right_plan_path -> _trans_from_gripper_to_endlink -> CuroboPlanner.plan_path`。
  - 检查发现现有 `assets/embodiments/piper_pika_agx/curobo.yml` 仍指向旧 `/assets/embodiments/piper/piper.urdf`，并使用旧 Piper 的 `link7/link8/joint7/joint8`，与 `piper_pika_agx.urdf` 的 Pika gripper link/joint 不一致。
- 修改：
  - 新增 `piper_pika_agx_ik_orig_tcp` embodiment，保持标定 `piper_pika_agx.urdf` 和左右 base pose，但把 `delta_matrix/global_trans_matrix` 改为 RoboTwin 自带 Piper TCP 转换。
  - 新增配套 `curobo.yml` 和 `collision_piper_pika.yml`，让 Curobo planner 也使用 `piper_pika_agx.urdf`、`gripper_base_link/gripper_left_link/gripper_right_link` 和 `left_joint/right_joint`。
  - 新增 `task_config/demo_clean_piper_ik_orig_tcp.yml`，命令仍使用 `pick_diverse_bottles_piper`，从而真正进入原始 `pick_diverse_bottles.py` 的 `grasp_actor/place_actor` IK/规划链路。
- 验证：
  - YAML 解析检查通过：新 task config、embodiment config、Curobo config、collision config 均可解析，`_embodiment_config.yml` 能找到 `piper_pika_agx_ik_orig_tcp`。
  - `py_compile envs/pick_diverse_bottles_piper.py envs/pick_diverse_bottles_piper_motion.py` 通过。
  - `git diff --check` 通过。
  - `timeout 120s bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_ik_orig_tcp 0` 已确认进入 `Embodiment Config: piper_pika_agx_ik_orig_tcp+piper_pika_agx_ik_orig_tcp` 和原始 `pick_diverse_bottles_piper` task，但 120 秒内没有完成 episode；失败仍主要是 `Objects is unstable` 与 `target_pose cannot be None for move action`。

## 2026-06-04（O.0 命令整理与 gen1 错误复查）

- 复查 tmux：
  - 当前没有名为 `gen1` 的单独 pane；实际 session 为 `gen1-1` 与 `gen1-2`。
  - 两个 pane 的最新输出一致：seed 72 到 115 之间持续出现 `Objects is unstable` 和 `target_pose cannot be None for move action`，最后用户 `Ctrl-C` 中断。
  - 结论：运行的是原始 task/IK 链路，不是已跑通的 `pick_diverse_bottles_piper_motion`；失败点仍在原始 `choose_grasp_pose/grasp_actor` 不能为标定 Piper/Pika 稳定生成可执行 target。
- 文档整理：
  - 重写 `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` 的 O.0 小节，只保留 4 条带用途标题的命令：无 viewer 数据生成、带运动 viewer、纯场景 viewer、原始 IK 诊断。
  - 重写 `COMMAND_LIBRARY.zh.md` 的 O.0 小节，并删除文件末尾重复的旧 O.0 head-only/motion 段落。
  - `pick_diverse_bottles_piper demo_clean_piper_calibrated` 不再作为推荐命令保留，只说明其会进入原始 `grasp_actor` 并持续失败。

## 2026-06-04（O.0 viewer-only 跳过 planner 与 motion viewer 修复）

- tmux 复查：
  - `gen1-1` / `gen1-2` 的原始 IK 诊断命令确实进入 `envs/pick_diverse_bottles.py -> grasp_actor -> choose_grasp_pose -> CuroboPlanner.plan_batch`。
  - 失败原因分两类：`Objects is unstable` 是瓶子初始随机摆放后的稳定性检查失败；`target_pose cannot be None for move action` 是原始抓取候选没有为标定 Piper/Pika 生成可执行抓取 target。
  - `No left camera link` / `No right camera link` 只是当前 `piper_pika_agx.urdf` 没有 wrist camera link 的 fallback 警告，不是 IK 失败根因；O.0 继续使用 head-only 配置。
- 修改：
  - `Base_Task.load_robot()` 新增 `skip_planner=True` 支持，viewer-only 场景可以不初始化 Curobo planner。
  - `Robot` 初始化默认提供 gripper-only planner 占位，只支持初始开合夹爪插值，正常 IK/采集链路仍会被 `set_planner()` 覆盖为 Curobo planner。
  - `view_pick_diverse_bottles_piper_scene.py` 新增 `--show_axes` 和 `--hold`；默认显示两个瓶子中心与左右放置目标的 RGB 坐标轴，并且 scene-only 跳过 planner/play_once。
  - 新增 `view_pick_diverse_bottles_piper_motion.py`，绕开 `collect_data.py` 的 `seed.txt` 续跑逻辑，每次 viewer 调试都会重新找稳定 seed 并执行一次 `play_once()`。
  - `run_pick_diverse_bottles_piper_motion_viewer.sh` 改为调用新的 motion viewer 入口。
- 验证：
  - `python -m py_compile envs/_base_task.py envs/robot/robot.py view_pick_diverse_bottles_piper_scene.py view_pick_diverse_bottles_piper_motion.py` 通过。
  - `bash -n run_pick_diverse_bottles_piper_motion_viewer.sh run_view_pick_diverse_bottles_piper_scene.sh` 通过。
  - `DISPLAY=:1.0 timeout 90s python view_pick_diverse_bottles_piper_scene.py --seed 0 --max_seed_tries 3 --hold 0` 通过：seed 0/1 不稳定跳过，seed 2 加载稳定场景，添加坐标轴并渲染一帧退出。
  - `DISPLAY=:1.0 timeout 120s bash run_pick_diverse_bottles_piper_motion_viewer.sh --seed 0 --max_seed_tries 3 --hold 0` 通过：seed 0/1 不稳定跳过，seed 2 加载后完成一次 `play_once()`。

## 2026-06-04（O.0 Piper motion 阶段日志与 EE 目标轴）

- 问题复查：
  - viewer 中白色小方块只是每个坐标轴 marker 的原点，不是 Piper base 或初始位姿。
  - 之前只显示瓶子中心和左右放置目标，没有显示夹爪运动阶段目标。
  - 当前标定 Piper/Pika 的 home FK 约为左 `(-0.30,-0.48,0.77)`、右 `(0.56,-0.50,0.80)`；这和原始 ALOHA/AgileX 的瓶子范围 `y=[0.03,0.23]` 不一致。
- 修改：
  - `envs/pick_diverse_bottles_piper_motion.py` 覆盖 `load_actors()`，把 O.0 motion baseline 的瓶子范围改为更靠近当前 Piper/Pika FK 工作区的 `left=x[-0.30,-0.18],y[-0.20,-0.10]` 与 `right=x[0.30,0.46],y[-0.20,-0.10]`。
  - 新增 `[piper-motion][stage]` 阶段日志，覆盖 `play_once/pregrasp/grasp_lower/close_gripper/lift/move_out/open_gripper`。
  - 新增 `get_debug_axis_poses()`，用当前 URDF/SAPIEN FK 计算当前左右 `link6` EE，以及 `pregrasp/grasp_lower/lift/move_out` 的左右 EE 目标轴。
  - `view_pick_diverse_bottles_piper_scene.py` 会调用任务的 `get_debug_axis_poses()` 并把这些 EE 目标轴加入 viewer。
- 验证：
  - `python -m py_compile envs/pick_diverse_bottles_piper_motion.py view_pick_diverse_bottles_piper_scene.py view_pick_diverse_bottles_piper_motion.py` 通过。
  - `DISPLAY=:1.0 timeout 120s bash run_pick_diverse_bottles_piper_motion_viewer.sh --seed 0 --max_seed_tries 10 --hold 0` 通过：seed 0/1 不稳定跳过，seed 2 稳定加载，输出所有 `[target-axis]` 与 `[stage]` 日志，并完成 `play_once()`。
- 剩余事实：
  - 当前阶段 EE 目标 FK 仍约在 `y=-0.40~-0.47`，所以 O.0 motion 仍是关节空间可视化 baseline；若要真正贴瓶抓取，需要后续重新设计 Piper EE 目标或使用适配 Piper/Pika 的 IK/抓取策略。

## 2026-06-08（Mode N-1 Foundation 目标 pose 顺序与 C 型夹爪预览）

- 问题复查：
  - Mode N 的目标定义应为 Foundation 物体世界位置 + 人手 gripper 旋转矩阵，但 wrapper 写入的 `pose_world_wxyz` 顺序与 planner 实际解析顺序不一致。
  - planner 内部实际按 `[x, y, z, qw, qx, qy, qz]` 解析 pose；历史字段名仍叫 `pose_world_wxyz`。
  - 修复前 `plan_keyframes_foundation_pose.py` / `plan_keyframes_human_replay.py` 写成 `[qw, qx, qy, qz, x, y, z]`，会让 planner 把 quaternion 当作位置。
- 修改：
  - 修正 Mode N 和 Mode M wrapper 输出的 `raw_pose_world_wxyz` / `pose_world_wxyz` 为 `[x, y, z, qw, qx, qy, qz]`。
  - `rank_previews/*.png` 增加 2D C 型夹爪投影和 RGB 局部轴；左臂蓝色、右臂橙色，X=红、Y=绿、Z=蓝，蓝色 local `+Z` 是 Mode N 的前进/后退轴。
  - rank preview 的 selected 标记改为按 `(frame, arm)` 判断，并补齐各 arm 的 debug frames。
  - `COMMAND_LIBRARY.zh.md` 的 `# 0608` Mode N 命令输出目录改为 `N-1_foundation_pose_viewer`。
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_keyframes_foundation_pose.py code_painting/plan_keyframes_human_replay.py` 通过。
  - `bash -n code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh` 通过。
  - `pick_diverse_bottles id2` smoke 已生成 `N-1_foundation_pose_viewer/pick_diverse_bottles/foundation_input_2/plan_summary_foundation_pose.json` 和 `rank_previews/keyframe_000036_rank_1.png`、`rank_previews/keyframe_000053_rank_1.png`。

## 2026-06-09（Mode N-3 rank preview 投影状态与 viewer 调试叠加）

- 问题复查：
  - 用户生成的 `N-2_foundation_pose_viewer/pick_diverse_bottles/foundation_input_0/rank_previews` 中没有肉眼可见的 C 型夹爪。
  - 复查 `plan_summary_foundation_pose.json` 和 smoke 日志后确认，新代码已经运行；该样例的合成目标投到 head camera 后位于相机后方，尤其右臂 keyframe 38 目标约为 `(0.644,-0.343,-0.297)`，相对右瓶低约 1.04m。
- 修改：
  - `rank_previews/*.png` 现在额外打印 target xyz、object->target 偏移和 `proj=inside/offscreen/behind_camera`。
  - 2D C 型夹爪投影支持越界裁剪；目标在视野外时画边缘标记，目标在相机后方时在图底部标注 `behind camera`。
  - `run_plan_keyframes_foundation_pose_piper_d435.sh` 新增 `--debug_viewer_overlay`，会把 `pure_scene_output` 置为 0，并在 viewer/视频里显示 target axis 与 top-1 C 型夹爪 actor。
  - Mode N 默认传 `--debug_candidate_top_k 1` 给 planner，便于 viewer 调试规划目标。
  - `COMMAND_LIBRARY.zh.md` 的 N 模块更新为 `N-3_foundation_pose_viewer`，并新增单条 `pick_diverse_bottles id1 --viewer --debug_viewer_overlay` 演示命令。
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_keyframes_foundation_pose.py` 通过。
  - `bash -n code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh` 通过。
  - `pick_diverse_bottles id0` smoke 已生成 `N-3_foundation_pose_viewer_smoke/.../rank_previews/keyframe_000038_rank_1.png`，图中明确显示 `L: proj=behind_camera` 与 `R: proj=behind_camera`。

## 2026-06-09（Mode N-4 Foundation replay pose 顺序修复与 camera 可视化）

- 问题复查：
  - 进一步验证 `foundation_replay_d435/foundation_input_0/multi_object_world_poses.npz` 后确认，`left_bottle__pose_world_wxyz[:3]` 与 `left_bottle__pose_world_matrix[:3,3]` 一致，都是真实物体位置。
  - 旧 Mode N 读取 `pose_world_wxyz[4:7]` 作为物体位置，实际读到 quaternion 的后三位。例如 frame 38 左瓶真实位置约 `(-0.0395, 0.0943, 0.7323)`，旧读取为 `(0.3273, 0.6319, 0.6086)`。
  - 因此 N-2/N-3 的 C 型夹爪飞出不是 Foundation 物体不在视野内，也不是 head camera 图像本身错误，而是 Mode N 读取 Foundation pose 顺序错误。
- 修改：
  - `plan_keyframes_foundation_pose.py` 中 object position 改为读取 `pose_world_wxyz[:3]`。
  - `plan_keyframes_foundation_pose.py` 与 `plan_keyframes_human_replay.py` 的 head camera pose 解析改为 `pos=[:3]`、`quat=[3:7]`。
  - 2D rank-preview 投影函数补上 SAPIEN camera frame 到 OpenCV frame 的转换：`[x, y, z]_cv = [x, -y, -z]_sapien_camera`。
  - `--debug_viewer_overlay` 现在同时打开 `--debug_visualize_cameras 1` 和 `--viewer_show_camera_frustums 1`，viewer 中可看到 head/third camera 轴与 frustum。
  - `COMMAND_LIBRARY.zh.md` Mode N 命令更新到 `N-4_foundation_pose_order_fix` 和 `N-4_foundation_pose_debug_viewer`。
- 验证：
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_keyframes_foundation_pose.py code_painting/plan_keyframes_human_replay.py` 通过。
  - `bash -n code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh` 通过。
  - `pick_diverse_bottles id0` smoke 输出中 frame 38 目标变为左 `(-0.058, 0.071, 0.735)`、右 `(0.253, 0.095, 0.747)`，与瓶子真实位置只差约 3cm retreat。
  - 新 `rank_previews/keyframe_000038_rank_1.png` 显示 `L: proj=inside(...)` 与 `R: proj=inside(...)`，2D C 型线框与 3D C 型 actor 均在瓶子附近。

## 2026-06-09（Mode N-5 retreat 参数与插值说明）

- 问题复查：
  - 当前 Mode N 不是直接在关键帧 0 和关键帧 1 之间做一次目标姿态插值，而是分 stage 规划：pregrasp、grasp、action。每个 stage 调用 URDF IK planner。
  - 默认 `cartesian_interp_ik` 会在当前 TCP 和 stage 目标 TCP 之间做位置线性插值、四元数 Slerp，然后逐 waypoint 求 IK。
  - 如果某个 waypoint 的 IK 从当前腕/肘分支切到另一组可行解，执行视频中会看到末端先朝下或扭转，之后再回到目标；这更像 IK 解分支跳变，而不是 keyframe1 的目标朝向本身错误。
- 修改：
  - `COMMAND_LIBRARY.zh.md` 的 Mode N 模块更新为 N-5。
  - N-5 命令使用 `--foundation_pose_retreat_m 0.08 --approach_offset_m 0.07`，对应 grasp retreat 8cm、pregrasp 总 retreat 15cm、pregrasp 到 grasp 前进 7cm。
  - 同步更新 `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` 和命令日志。
- 验证：
  - 本次只改文档和命令参数说明，未改 planner 代码。

## 2026-06-10（Mode N-6 id1 viewer 复查与问题归因）

- 问题复查：
  - 用户在 `modeln-4` 中运行 `pick_diverse_bottles id=1 --viewer --debug_viewer_overlay --foundation_pose_retreat_m 0.10 --approach_offset_m 0.07`。
  - 输出目录为 `N-5_pregrasp15_grasp8_debug_viewer/.../foundation_input_1`，但实际参数是 grasp retreat 10cm、pregrasp 总 retreat 17cm；目录名已在命令文档中改为 N-6，避免继续误导。
- 观察结果：
  - `plan_summary.json` 显示 pregrasp/grasp 阶段位置误差约 1-2.4cm，基本可用。
  - action 阶段左臂第 3 次 replan 后 reached，位置误差约 2.9cm；右臂第 3 次 replan 后未 reached，位置误差约 38.9cm。
  - `pose_debug.jsonl` 显示 init->pregrasp 左右腕部末端关节累计转动约 4.25/4.23rad，但净变化很小；action 右臂也出现较大的关节累计转动和分支跳变。
- 归因：
  - Foundation 物体位置与投影不再是主要问题。
  - 当前 Mode N 允许 `reach_rot_tol_deg=180` 和 `urdfik_max_rotation_threshold_rad=3.14`，且 `candidate_keep_camera_up=0`；绕前进轴 180 度等价的姿态会被接受，腕部相机朝上与 roll 连续性没有被强制。
  - 当前 `cartesian_interp_ik` 逐 waypoint 求 IK，IK 可在中途选择 unseeded/另一组腕肘解，因此会出现视觉上先朝下、再扭回目标或绕前进轴大转的路径。
- 修改：
  - 仅更新 `COMMAND_LIBRARY.zh.md` 与 agent-read 命令文档，把推荐命令改为 N-6：`--foundation_pose_retreat_m 0.10 --approach_offset_m 0.07`，输出目录 `N-6_pregrasp17_grasp10`。
  - 本次没有修改 planner 代码；roll/up 约束、>180 度错误旋转限制、IK 分支连续性仍待后续实现。

## 2026-06-10（Piper IK 连续轨迹 v2 与 V1-V4 采集修复）

- 根因：旧实现把四段 move 都从 home 规划；lift 的 x/y 参考错误；运动后未等待 PD 收敛；place 把瓶子功能点目标当成夹爪目标；Phase 1/2 还能混用无版本旧 pickle。
- 修改：四段 move 改为逐段规划执行，下一段使用上一段 IK 末端关节状态；lift 保留 grasp x/y 和姿态，仅增加 z；close 后测量瓶子到末端偏移并修正 place；每段增加 endpoint settling。
- 轨迹：新增 `piper_ik_cartesian` schema v2、IK 版本、动作名、目标、shape/finite/nonempty 校验，并拒绝旧格式。
- V3：MotionGen 不可用、异常或规划失败时，回退到同一有效 IK 终点的三次插值。
- Viewer/采集：viewer 默认要求真实成功；采集断点续跑识别 `_succ/_fail.hdf5`；新增 `demo_piper_ik_seq_v1..v4` 隔离旧数据。
- 相机：`third_camera` 改为右侧视角，新增对向俯视 `opposite_top_camera`；所有 RGB camera 自动输出 MP4。
- 验证：V1-V4 viewer 均在 seed 3 真实成功；V1 完整采集 5 集，成功 seed 为 3/6/10/14/18；V2/V3/V4 各完成 1 集 smoke，均生成 `_succ.hdf5`、v2 pickle、instruction 和六路视频。V1 重跑确认跳过现有 `_succ.hdf5`。
- 最终检查：`py_compile`、四份 YAML、prompt JSON、`bash -n collect_data.sh`、`git diff --check` 和旧 pickle 拒绝测试均通过。
- 清理：验证后删除可重复生成的 V2/V3/V4 smoke 配置和临时输出；保留忽略目录下的 V1 完整验证数据。

## 2026-06-11（O.1 Foundation OBJ 抓取修复）

- O.1 现在实际加载 NPZ 指定的 cola/bottle OBJ，不再用 `001_bottle` 替代。
- 修正 actor 原点与 OBJ 几何中心混淆；按旋转后最低网格点计算桌面 clearance。
- 默认使用 bounds 圆柱代理碰撞；精确凸碰撞保留为可选项。
- 为当前窄 OBJ 增加距离门控 grasp assist，并在 open/关闭场景时释放 drive。
- 轨迹绑定 Foundation 输入、frame、碰撞模式和 mesh 几何；语言描述改为直接语义文本并补齐 task prompt。
- replay 模式跳过 planner 初始化，解决 V3 Phase 2 多相机渲染卡顿。
- 验证：V1-V4 viewer 均 `physical_success=True`；V1-V4 完整采集均生成 validated replay、`episode0_succ.hdf5`、六路视频和 instructions。V3 MotionGen 失败后正常回退。
- 清理：结构检查后删除本轮可重复生成的 O.1 HDF5、视频、缓存、临时 config 和日志。

## 2026-06-11（O.1 无瞬移抓取门控与关键帧模式）

- 根因：旧 O.1 在 close 前执行 `actor.set_pose(settled_pose)`，因此 pregrasp/grasp 撞倒的瓶子会在 close 时瞬移回夹爪。完整瓶身圆柱碰撞也会与张开夹爪的接近路径发生提前碰撞。
- 修改：删除物体 pose reset；默认改为底部 `support_proxy`；close 后检查位移、旋转、link6 距离、双指投影和径向距离，通过后才在物体当前 pose 建立 drive。
- O.1.1：读取 `hand_keyframes_all.json` 中 episode 的前两个关键帧，用第一帧设置 Foundation OBJ pose。
- O.1.2：第一帧完成 pregrasp/grasp/close，第二帧从 `world_targets_and_status.npz` 取左右 EE xyz，以保留 grasp 朝向的单一 action 代替 lift/place。
- 轨迹上下文新增 mode、episode ID、关键帧、annotation/action 来源、pregrasp 距离和抓取门控参数，防止跨模式回放。
- 验证：`py_compile` 和 `bash -n` 通过；V1 的 O.1/O.1.1/O.1.2 viewer 与完整两阶段采集通过；O.1.2 的 V2/V3/V4 完整采集通过。每次采集均生成 v2 pickle、validated replay、`episode0_succ.hdf5`、instructions 和六路 MP4。
- 环境边界：当前非交互 shell 的 X11 socket 对 SAPIEN 报 `Renderer does not support display`，因此 V2-V4 未在该 shell 重复 GUI 建窗；离屏完整采集已覆盖同一规划和回放逻辑。
- 清理：检查产物结构后删除本轮六个可重复生成的 validation 输出目录、临时 YAML 和 `/tmp` 日志；未修改已有采集数据。

## 2026-06-11（Mode N-7 action 朝向与 dual replan 冻结）

- 复查 `modeln-4` 的 N-6/N-7 输出后确认，Foundation 位置和投影不是当前主要问题；`pick_diverse_bottles id=1` 的 N-6 在 pregrasp/grasp 约 1-2.4cm，但 action 右臂第 3 次 replan 后约 38.9cm miss。
- 新增 `--foundation_pose_action_orientation_source grasp`：第二关键帧 action 继续使用第二帧 Foundation 物体 xyz，但朝向和 retreat 方向沿用第一关键帧 grasp 朝向，借鉴 O.1.2 的“action 保持 grasp quaternion”逻辑。
- 新增 `--dual_stage_freeze_reached_arms_on_replan`：dual-stage replan 中某只手已达标后，后续 attempt 保持该手关节不动，只补偿未达标手。
- 校对 R1 的 `candidate_keep_camera_up` 后确认它按 local X 作为 forward；Mode N 当前用 local +Z 作为前进/retreat 轴，因此不能直接照搬。新增的 N 专用 `--foundation_pose_keep_top_axis_up` 绕 local +Z 做 180 度二选一，但 `top_axis=y` 在 id=1 变差，推荐暂不启用。
- 验证：`py_compile` 和 `bash -n` 通过；`N-7_action_grasp_rot_freeze_smoke` 在 `pick_diverse_bottles id=1` 上 action 左/右误差约 2.78cm/2.07cm，双臂达标。未解决的问题是 IK 仍接受约 170 度 roll 等价姿态，朝向误差不能作为严格成功条件。

## 2026-06-11（Mode M Human Replay IK 连续性修复）

- 根因：Mode M 未向底层转发 seed 参数；旧 IK 倾向按 pose error 选 unseeded 分支；失败的 Cartesian prefix 仍可执行；第一帧 grasp 失败后第二帧被安全门控跳过。
- 修改：修正 CuRobo seed tensor 为 `[batch, num_seeds, dof]`；增加显式扰动 seed 与关节连续性选解；增加 cubic joint smoothstep；action 默认保留 grasp quaternion；冻结已 reached 手；执行失败返回非零退出码。
- 姿态结论：当前没有 local +Z 前进轴的严格 roll 限制。Piper 固定 `global_trans_matrix` 与 target/report frame 约定不一致，约 178-180 度旋转误差暂不能作为严格成功条件；`apply_global_trans_to_ik=1` 实测更差。
- 验证：`pick_diverse_bottles` ID 1、2 完整成功；ID 0 pregrasp/grasp 成功，但 action 左右误差约 4.39cm/6.41cm，超过 4cm 阈值。旧问题是共享 IK/执行问题，不是 ID 1 标注独有。

## 2026-06-11（O.1.2 有限重试与标定腕相机）

- tmux 复查确认 `gen2-10`、`genikv2-11`、`genikv3-12`、`genikv4-13` 已回到 shell；历史任务由 `Killed`/Ctrl-C 结束，并非仍在采集。
- 修正批处理放大因素：V1 从每 ID 10 episodes 改为 1；四配置增加 `max_seed_tries: 3`；通用采集器达到上限后返回非零，避免确定性失败无限循环。
- `collect_foundation_piper_ik.sh` 新增可选 `run_tag`，生成隔离 config/output，并强制 `episode_num: 1`。
- 四份 Foundation 配置启用左右 wrist；从 `calibration_bundle_piper_new_table_0515.json` 读取独立外参，以 planner gripper pose 为父坐标并转换 optical/render 轴。
- 验证：V1 O.1.2 两阶段采集成功，生成 HDF5、instruction 和八路 MP4；左右腕各 38 帧、320x240，移动约 0.37m/0.46m。V4 ID 9 在三个 seed 上均因右 grasp 旋转约 25.6 度超过 15 度门限失败，并按上限退出。

## 2026-06-11（Wrist 坐标适配与 Foundation 视频索引）

- 复核 0515/new-table 左右 wrist 手眼文件及历史版本：平移和姿态趋势稳定；右腕约 45 度 roll 持续存在，属于实机安装差异。
- 修复实机 TCP 外参与 RoboTwin Pika CAD 父坐标混用问题。Foundation 配置改为 `urdf_end_link`，bundle 提供 `piper_pika_agx` 平移净空 adapter；保留标定光轴、左右符号和 roll。
- Viewer 新增 `--wrist_preview 1`，实时显示左右 wrist RGB 拼接窗口。
- 新增 `script/index_foundation_piper_ik_videos.py`，把每个独立目录的 `episode0_*` 按 Foundation ID 映射为聚合目录 `episode<ID>_*`；默认软链接并拒绝覆盖，显式 `--replace-episode` 才替换目标 MP4。
- 验证：V1/O.1.2 `foundation_input_0` 完整两阶段采集成功，生成 38 帧左右腕视频、HDF5、instruction 和 validated trajectory；抽帧确认两路均看到对应瓶子和夹爪。`DISPLAY=:1.0` viewer 连同 `--wrist_preview 1` 完整运行并通过 `physical_success=True`。

## 2026-06-15（O.1.2.1 wrist 外壳与 roll 校正）

- tmux `gen1/gen2/genikv2/genikv3/genikv4` 均已回到 shell。历史批处理漏设 `RUN_TAG`，产生 `o12_v4__failures.log` 并把多套相机配置混入无 tag 目录。
- 官方 Pika gripper URDF 不含相机；官方 Piper+Pika 的 `joint6_to_gripper_base` 带 `rpy="0 -1.57 0"`，当前 AGX 合并 URDF 使用 identity 连接，说明外观对齐不能替代父帧对齐。
- `envs/camera/camera.py` 新增逐侧 `forward_offset_m` 与 `image_roll_deg`。基础 YAML 采用左 `0.125 m/-15 deg`、右 `0.11 m/-60 deg`，保留 0515 原始 JSON 和 IK 不变。
- Viewer 新增四个临时覆盖参数，可在实时双腕窗口中微调而不修改 YAML。
- 验证：`py_compile`、四份 YAML 解析通过；V4 ID 0 O.1.2 tag `wrist_o121_verified_0615_smoke` 完成两阶段采集，生成 38 帧左右 wrist MP4、HDF5、instructions 和其余六路视频。抽帧确认外壳退出画面且右侧 roll 扶正；最终 viewer 含 `--wrist_preview 1` 完整运行并报告 `physical_success=True`。
- 清理：删除三组可重复生成的中间 tuning smoke，只保留最终 `wrist_o121_verified_0615_smoke` 供结果检查；所有数据目录均由 `data/*` 忽略。

## 2026-06-15（O.1.2.1 viewer wrist debug 录制）

- Viewer 新增独立 wrist debug recorder，同一渲染帧同时写左、右和带标签拼接 MP4，并保存 camera/tuning/task/Foundation 上下文 JSON。
- `--wrist_debug_tag` 拒绝覆盖已有非空目录；当前限制单次一个 episode，防止多 episode 共用 tag。
- 明确坐标结论：矩阵可拼接，错误来自缺失的 `link6_T_real_tcp`，不是 0515 `real_tcp_T_camera` 本身；tuning 是缺失机械外参的经验补偿。
- 验证：V1/O.1.2 ID 0 viewer `physical_success=True`；最终保留的无窗口三路 MP4 各 511 帧，左/右 320x240、拼接 640x240，30 FPS，JSON 含 task/config/ID/mode/seed 且参数一致；已清理较早的不含 context 录制。

## 2026-06-15（Wrist Debug H.264 与无 viewer 正式参数覆盖）

- 根因：旧 recorder 使用 `mp4v`，文件完整但 VS Code/Chromium codec 支持不稳定。改为 FFmpeg `libx264`、`yuv420p`、`faststart`，与正式采集视频的 H.264 逻辑一致。
- 已将 `data/wrist_camera_debug` 现有 6 个旧 `mp4v` 文件原地转为 `h264/avc1/yuv420p`，文件名不变。
- `collect_foundation_piper_ik.sh` 新增四个全量环境变量覆盖，写入 generated YAML；缺任一参数或数值非法时立即失败。
- 验证：新 debug 三路各 511 帧 H.264，`moov` 位于文件前部；V1 ID 0 O.1.2 无 viewer 正式采集完整成功，左右 wrist 各 38 帧 H.264，generated YAML 为左 `0.125/-15`、右 `0.11/-60`。
- 清理：删除重复的 `o121_h264_smoke_0615`，保留已转码的原 debug 目录和正式采集验证结果。

## 2026-06-15（Viewer wrist/head 相机框线）

- `view_pick_diverse_bottles_piper_ik_motion.py` 新增 `--show_camera_frustums`，使用 SAPIEN `show_camera_linesets` 显示相机视锥，并在启用时校验左右 wrist 与 head camera 均已加载。
- 修复 `--hold 1` 未被使用的问题；单 episode 现在保持最终窗口直到关窗或 `Ctrl-C`。
- `gen1` 失败原因：两次复用了非空 debug tag；一次换新 tag 后 viewer 实际完整成功；随后 `unset DISPLAY` 且误用 `set DISPLAY`，导致 SAPIEN 无显示。正确恢复为 `export DISPLAY=:1.0`。
- 验证：`:1.0` 通过 `xdpyinfo`；V1/O.1.2 ID 0 viewer 日志列出 `left_camera/right_camera/head_camera`，最终 `physical_success=True`；`hold=1` 进入持续保持并可由 `Ctrl-C` 无 traceback 退出。

## 2026-06-16（Piper IK 实时 SAPIEN 运动）

- 根因：`envs/pick_diverse_bottles_piper_ik.py` 的自定义 move、末端 settle 和 gripper settle 循环只调用 `_update_render()`，实时更新了 wrist 图像，却遗漏 `viewer.render()`；因此 SAPIEN 只在末尾 hold 时显示最终状态。
- 新增统一 `_render_execution_step()`：始终刷新 observation cameras，并按 `render_freq` 实时绘制 SAPIEN；viewer 模式结束后校验并打印实时运动帧数。
- 模式 1 验证：运动期间检测到 1920x1080 `SAPIEN` 窗口，实时绘制 510 帧，`physical_success=True`。
- 模式 2 验证：运动期间同时检测到 `SAPIEN` 和 640x299 `RoboTwin wrist cameras`，实时绘制 510 帧，`physical_success=True`。

## 2026-06-16（Wrist 相机前向轴诊断）

- 新增 `script/diagnose_piper_wrist_camera_axes.py`，复用 `envs/camera/camera.py` 的 `legacy_r1` axis conversion 和 `piper_pika_agx` adapter，离线计算左右 wrist camera forward。
- 当前结果：左 forward `[0.999974, -0.003184, 0.006511]`，右 forward `[0.999622, -0.014664, 0.023248]`，均在 gripper frame 中接近 Pika 物理 `+X`。
- 开合轴 `Y` 平面误差：左 `-0.182 deg`、右 `-0.840 deg`；若只消除 `Y` 分量，需要绕 gripper `+Z` 的微小 yaw 左 `+0.182 deg`、右 `+0.840 deg`。
- 记录结论：旧 debug 蓝色 `+Z` 是 IK/debug 目标姿态约定，不应直接作为 Pika 物理前向；否则会误算出约 `-89 deg` 的大旋转。

## 2026-06-16（Wrist parent yaw 参数）

- `envs/camera/camera.py` 新增 `parent_yaw_deg` tuning，在 gripper/link6 父坐标系绕 `+Z` 微调 wrist camera 朝向；它不同于绕光轴的 `image_roll_deg`。
- `view_pick_diverse_bottles_piper_ik_motion.py` 新增 `--wrist_left_yaw_deg` 与 `--wrist_right_yaw_deg`，用于 viewer 临时覆盖。
- `script/diagnose_piper_wrist_camera_axes.py` 现在在末尾输出可直接复制的 yaw 参数；当前为左 `0.182 deg`、右 `0.840 deg`。
- 验证：带 yaw 的 V1/O.1.2 实时 SAPIEN + wrist viewer 完成 510 帧实时绘制，`physical_success=True`，日志确认 tuning 含 `parent_yaw_deg`。
## 2026-06-16（Wrist 距离与俯视角复查）

- 当前 wrist forward offset：左 `0.125m`、右 `0.11m`；若沿光轴前移 2cm，viewer 参数为左 `0.145m`、右 `0.13m`。
- 以 nominal tip `[0.12,0,0]` 估算，当前相机到 tip 欧氏距离约 `11.9cm`，沿相机 forward 到 tip 约 `6.8cm`；加 2cm 后沿 forward 到 tip 约 `4.8cm`。
- 原始 0515 和 yaw 后 forward 都接近 Pika 物理 `+X`，不是明显俯视夹爪；若精确看向 nominal tip，需要约 `54deg` gripper-`Y` pitch，实际应小步试俯视 pitch。
- 右相机中心 `Y=-2.74cm`，左相机 `Y=+2.07cm`，右侧偏心略大，yaw/roll 不能完全替代 lateral offset。



## 2026-06-16（Foundation gripper 抓取距离 +2cm）

- 将四份 `demo_piper_ik_foundation_v1-v4.yml` 的 `foundation_grasp_standoff` 默认值从 `0.085m` 改为 `0.105m`。
- `view_pick_diverse_bottles_piper_ik_motion.py` 新增 `--foundation_grasp_standoff_m`，用于实时 viewer 临时覆盖抓取规划距离。
- `collect_foundation_piper_ik.sh` 新增 `FOUNDATION_GRASP_STANDOFF_M` 环境变量覆盖，并在生成 config 时写入该值。
- 语义更正：这是 gripper base/EE grasp 目标相对物体中心的 standoff；wrist camera `forward_offset_m` 只调相机外参，不改变抓取深度。

验证：`py_compile` 通过；`bash -n collect_foundation_piper_ik.sh` 通过；`view_pick_diverse_bottles_piper_ik_motion.py --help` 显示 `--foundation_grasp_standoff_m`；V1/O.1.2 headless 最小运行使用 `--foundation_grasp_standoff_m 0.105` 完成，日志确认 `grasp_standoff=0.105m` 且 `physical_success=True`。


## 2026-06-16（Wrist 原始标定角度表与 pitch/lateral 调参）

- 复查 0515 wrist 标定：原始/adapter 后 forward 均基本与 gripper `+X` 共面，left `plane_err_y=-0.182deg`，right `-0.840deg`。
- 原始标定没有明显俯视夹爪：forward 到 gripper `+X` 仅 left `0.415deg`、right `1.575deg`；当前 viewer yaw 后为 left `0.373deg`、right `1.332deg`。
- 新增 viewer 调参：`--wrist_left_pitch_deg`、`--wrist_right_pitch_deg`、`--wrist_left_lateral_offset_m`、`--wrist_right_lateral_offset_m`。
- 建议第一轮试调：左右 pitch `15deg`，右手 lateral `+0.0067m`。

验证：`py_compile envs/camera/camera.py view_pick_diverse_bottles_piper_ik_motion.py` 通过；viewer `--help` 显示 pitch/lateral 新参数；带 `--wrist_left_pitch_deg 15 --wrist_right_pitch_deg 15 --wrist_right_lateral_offset_m 0.0067` 的 V1/O.1.2 headless 最小运行完成，日志确认 camera tuning 含 `parent_pitch_deg` 与 `parent_lateral_offset_m`，并报告 `physical_success=True`。


## 2026-06-16（Wrist gripper 中线修正说明）

- 明确区分 mirror correction 与 centerline correction：右手 `+0.0067m` 只让 right `Y=-2.74cm` 接近 left 的镜像 `-2.07cm`；不是把相机放到 `Y=0`。
- 若目标是 gripper 中线，相机 lateral 应设为 left `-0.0207m`、right `+0.0274m`。
- 坐标约定：gripper `+X` 是 wrist 到 tip 的前进轴，`+Y` 是夹爪开合/左右偏心方向。
- 验证：带中线修正和左右 pitch `15deg` 的 V1/O.1.2 headless 最小运行成功，日志确认 `parent_lateral_offset_m` 进入 camera tuning，`physical_success=True`。


## 2026-06-16（O.1.2 verified grasp/wrist v2 与真实抓取 debug）

- 记录用户实测效果好的 viewer 参数：`foundation_grasp_standoff_m=0.14`、wrist forward left/right `0.145/0.13`、pitch `15deg`、lateral left/right `-0.0207/0.0274`。
- `view_pick_diverse_bottles_piper_ik_motion.py` 新增 Foundation debug 覆盖参数：collision mode、collision padding、grasp assist、require contact、radial tolerance、assist max distance。
- O.1.2 grasp-assist 关闭时现在仍会执行 grasp-state validation 并打印 `contacts/projection/radial`，但不会创建 object-gripper drive。

验证：`py_compile` 通过；viewer `--help` 显示新增 Foundation debug 参数；verified v2 headless 在 `radial_tolerance=0.08`、`assist_max_distance=0.16` 下完成并 `physical_success=True`。默认门控 `0.065/0.14` 会因 left `radial=0.071m` / `ee_distance=0.143m` 失败，因此文档命令已显式加入门控阈值。纯物理档 `--foundation_grasp_assist 0 --foundation_collision_mode cylinder_proxy` 能运行到 validation 并打印 contacts/projection/radial；当前 seed 0 会失败，说明纯接触还没真实夹住。


## 2026-06-16（Verified 采集 wrapper 与 O.2 pnp_tray）

- 新增 `collect_foundation_piper_ik_verified.sh`，统一生成 verified v2 task config 并调用 `collect_data.sh`；支持 `pick_diverse_bottles` 和 `pnp_tray`，支持 `DRY_RUN=1`。
- pick_diverse_bottles 采集固定使用稳定 A 档：`support_proxy + grasp_assist=true + require_contact=false + standoff=0.14 + radial=0.08 + assist_max_distance=0.16`，并写入当前确认的 head/wrist 相机参数。
- 新增 `envs/pnp_tray_piper_ik_foundation.py`，复用 Foundation IK 基类，将对象映射为左 `left_dark_red_cup`、右 `right_bottle`，动作末尾打开夹爪。
- Foundation 基类抽出对象 key、actor id、annotation path、hand target pattern 和 `foundation_open_after_action`，O.1 默认行为保持关闭 open-after-action。
- 新增 `task_config/demo_pnp_tray_piper_ik_foundation_v1-v4.yml` 和 `description/task_instruction/pnp_tray_piper_ik_foundation.json`。
- AB/C 结论：A 档是当前可稳定采集模式；B 档需要完整侧面 collision 才有接触意义；C 档关闭 assist 后会暴露当前 pregrasp/grasp 与物体碰撞的问题。

验证：`py_compile` 通过；`DRY_RUN=1` 对 pick_diverse_bottles 和 pnp_tray 均能生成 config；pick_diverse V1/O.1.2 headless 使用 `standoff=0.14` 和放宽门控成功；pnp_tray V1/ID0/O.2 headless 使用 `standoff=0.105` 成功，日志包含 `open_after_action=True` 和 `open_gripper`。


## 2026-06-17（O.2 pnp_tray action 目标修正与预抓取避障试验）

- 根因：旧 O.2 沿用 O.1.2 的第二关键帧 EE target，ID0 的 EE target `Y≈0.266`，比 Foundation 第二关键帧物体中心 `Y≈0.18` 更靠前，导致 close 后 gripper 看起来向前移动过远。
- 修正：`pnp_tray_piper_ik_foundation` 默认 `foundation_action_target_source=object_keyframe`，action gripper target 由第二关键帧 OBJ center 加当前 grasp 相对偏移得到。
- 新增 viewer 参数 `--foundation_action_target_source` 与 `--foundation_pregrasp_clearance_m`；wrapper 支持 `FOUNDATION_PREGRASP_CLEARANCE_M` 生成独立避障试验 config。
- 可选避障：`foundation_pregrasp_clearance=0.06m` 在 pregrasp 前插入抬高 waypoint；默认仍为 `0`，不影响无避障命令。

验证：O.2 V1/ID0 headless 默认 object-keyframe 成功，action target 从旧 EE `Y≈0.266` 改为 gripper `Y≈0.075/0.069`，object error 约 `4.2cm/3.3cm`；`foundation_pregrasp_clearance=0.06` 成功；`0.10` 失败，左杯旋转约 `16.3deg` 超过门限。

## 2026-06-18（Piper wrist camera 状态备份与恢复）

- 先按用户要求提交当前疑似损坏状态：`2feaa0b Backup current Piper wrist camera state`，改动涉及 `COMMAND_LIBRARY.zh.md` 和 `code_painting/plan_anygrasp_keyframes_r1.py`。
- 随后用非破坏性 `git revert` 恢复到备份提交之前的 planner/命令文档状态：`916d9f6 Revert "Backup current Piper wrist camera state"`。
- 恢复后确认 `plan_anygrasp_keyframes_r1.py` 和 `run_plan_keyframes_human_replay_piper_d435.sh` 仍支持 `--wrist_*`、`--joint_trajectory_interpolation`、`--dual_stage_freeze_reached_arms_on_replan`、`--fail_on_execution_failure` 和 `--piper_calibration_bundle` 参数。

验证：`git status --short` 干净；`/home/zaijia001/ssd/miniconda3/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py` 通过。


## 2026-06-18（Mode M no-viewer 相机轴可视化隔离）

- `run_plan_keyframes_human_replay_piper_d435.sh` 现在只在 `--viewer` 模式转发 `--wrist_preview 1`、`--viewer_show_camera_frustums 1` 和 `--debug_visualize_cameras 1`。
- no-viewer 批量采集不再绘制 wrist/head 相机框线或相机 RGB 轴；wrist 外参参数仍按命令原样转发，因此批量 wrist 视角与 viewer debug 的相机外参一致。
- 记录 L16 `pick_diverse_bottles` 实测 wrist 参数：left/right forward `-0.04/-0.01`，roll `14.635/-44.649`，yaw `0.182/0.840`，pitch `-90/-90`，lateral `-0.0207/0.0274`。

验证：`bash -n code_painting/run_plan_keyframes_human_replay_piper_d435.sh` 通过；L16 viewer/no-viewer 命令 `--dry_run` 均通过。


## 2026-06-24（L16 六任务人手+物体 inpaint/repaint 指令）

- 在 `COMMAND_LIBRARY.zh.md` 新增 I3.6/I3.7：L16 六任务的人手+物体 Stage-1 inpaint debug 指令和全量批处理指令。
- Stage-1 改为直接调用 `remove_anything_video_sam2.py`，避免 `run_human_robot_inpaint_repaint.py` 在只想生成背景时仍要求 Stage-2 composite。
- 新输出目录为 `results_repaint_piper_h2_l16/stage1_human_object` 与 `results_repaint_piper_h2_l16_visible_reinit/e0_robot_object`，不覆盖旧 I1/I3.5。

验证：对新增 bash 代码块做静态语法检查，结果记录在本轮命令输出中。


## 2026-06-24（修正 L16 repaint 指令可运行性）

- 修正 I3.6/I3.7 命令：将 `set -euo pipefail` 改为 `set -eo pipefail`，避免 conda activate 脚本因 `ADDR2LINE` 未绑定变量失败。
- 在 Stage-1 调用 `remove_anything_video_sam2.py` 前增加 `mkdir -p "$S1OUT"`，避免批处理 `save_mask_frames=0` 时输出目录不存在导致 `FileNotFoundError`。

验证：重新提取 I3.6/I3.7 bash 块并通过 `bash -n`。

## 2026-06-24 13:20 +08 - L16 可视化拼接脚本

- 新增 `code_painting/make_l16_repaint_montage.py`，用于把 HaMeR gripper、Foundation object replay、L16 robot plan 横向拼接；如果 Stage1 inpaint 和 final repaint 已存在，会自动加入面板。
- 在新 tmux `l16_vis_id0` 中测试 `pick_diverse_bottles id0` 成功，输出 `code_painting/l16_repaint_montage/pick_diverse_bottles/id_0/compare_hamer_foundation_l16_repaint_pick_diverse_bottles_id0.mp4`。
- 验证：`python3 -m py_compile code_painting/make_l16_repaint_montage.py` 通过；`ffprobe` 显示输出视频为 `2130x320`、`5 fps`、`21.4s`、`107` 帧。

## 2026-06-24（L16 白背景反选 repaint 指令）

- 将 `COMMAND_LIBRARY.zh.md` 的 I3.5/I3.6/I3.7 repaint 说明整理为 I3.5.1/I3.5.2/I3.5.3，保留原 D435 visible-reinit 和 L16 robot/object prompt 路线。
- 新增 I3.6：L16 白色背景 SAM + `--invert_mask` 反选路线。该路线不修改现有 Python 代码，使用 `remove_anything_video_sam3_robot.py` 保存反选 mask，再在命令内用 inline 合成步骤生成 `target_with_original_head_cam_plan.mp4` 和 `final_repainted.mp4`。
- 新输出根目录：`/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object`。

Validation: 已抽取 I3.6 debug bash block 并通过 `bash -n /tmp/i36_whitebg_debug_block.sh`；已抽取 inline compose Python 并通过 `python3 -m py_compile /tmp/i36_whitebg_inline_compose.py`。本轮未运行实际 SAM 推理。

## 2026-06-24（修正 I3.6 白背景反选 mask 帧目录）

- 修正 `COMMAND_LIBRARY.zh.md` I3.6 inline compose 片段：`remove_anything_video_sam3_robot.py` 实际把逐帧 mask 保存到 `mask_head_cam_plan/`，不是 `mask/`。旧命令会在第一条 `pick_diverse_bottles id0` 合成时报 `no inverted mask frames under .../mask`，因此循环提前退出。
- 同步更新 I3.6 的输出检查说明和 `agent-read/COMMANDS/pi0_h2o_training_data.*.md`。

Validation: 已重新抽取 I3.6 debug/batch bash block 并做 `bash -n`；已抽取 inline compose Python 并做 `python3 -m py_compile`。

## 2026-06-24（I3.6 默认改用 human-object 背景）

- 将 `COMMAND_LIBRARY.zh.md` I3.6 默认运行方式从 `BG_MODE=hand_only` 改为 `BG_MODE=human_object`，避免白背景反选 repaint 时使用只抠除人手的背景导致真实物体残留。
- 修正 `stack_cups` 的 Stage-1/robot-object prompt：去掉泛化 `cups`，只保留 `left red cup, right red cup`，减少绿色杯子被误 inpaint 的风险。
- 同步更新 `agent-read/COMMANDS/pi0_h2o_training_data.*.md`。

Validation: 已重新抽取 I3.6 debug/batch bash block 并做 `bash -n`；已抽取 inline compose Python 并做 `python3 -m py_compile`。

## 2026-06-24（stack_cups prompt 细化为粉杯/深红杯）

- 将 `stack_cups` 的 Stage-1/robot-object prompt 从 `left red cup, right red cup` 进一步细化为 `left light pink cup, right dark red cup`，与 AnyGrasp/Foundation 对象命名一致，减少 `red cup` 泛化导致绿色杯子也被选中的风险。

Validation: 已重新抽取 I3.6 debug/batch bash block 并做 `bash -n`；已抽取 inline compose Python 并做 `python3 -m py_compile`。

## 2026-06-24（I3.6 背景按比例拉伸合成）

- 修改 `COMMAND_LIBRARY.zh.md` I3.6 inline compose：最终输出帧数跟随 `mask_head_cam_plan/*.jpg`/robot replay，Stage-1 BG 帧数较短时按 `round(i * (bg_len - 1) / (out_len - 1))` 比例采样 BG 帧。
- 修改 `/home/zaijia001/ssd/inpainting_sam3_robot/remove_anything_video_sam3_robot.py`：当 `--save_removed_video 0` 时不再构建/执行 STTN inpainter，只输出 mask/box，避免白背景反选 Stage-2 因无用 inpainting OOM。

Validation: `python3 -m py_compile /home/zaijia001/ssd/inpainting_sam3_robot/remove_anything_video_sam3_robot.py`；抽取 I3.6 debug/batch bash block 做 `bash -n`；抽取 inline compose Python 做 `python3 -m py_compile`；用 `pick_diverse_bottles id0` 做纯合成验证，robot=107 帧、BG=106 帧、final=107 帧。

## 2026-06-24（L16 白背景反选任务级脚本）

- 新增 `code_painting/run_l16_stage1_human_object_task.sh`：按单任务补/重跑 Stage-1 人手+物体 inpaint。
- 新增 `code_painting/run_l16_whitebg_repaint_task.sh`：按单任务执行白背景 SAM + 反选 repaint，并在 compose 中按比例拉伸短 BG。
- `COMMAND_LIBRARY.zh.md` 新增 I3.6.1，记录五个非 `stack_cups` 任务的并行 tmux 指令。

Validation: `bash -n code_painting/run_l16_stage1_human_object_task.sh code_painting/run_l16_whitebg_repaint_task.sh`。

## 2026-06-24（I3.6.1 GPU 编号修正）

- 将 I3.6.1 中 `pnp_tray` 的示例 GPU 从不存在的 GPU4 改成第二波复用 GPU0；当前机器可用 GPU 编号为 0-3，五个任务不能全部独占不同 GPU 同时跑。

## 2026-06-25（stack_cups 绿色杯保护 debug runner）

- 新增 `code_painting/l16_stack_cups_debug_variants.py` 和 `code_painting/run_l16_stack_cups_debug_variants.sh`，用于对 `stack_cups id_0..4` 跑四种 Stage-1 debug 方案。
- 四种方案分别是 DINO green-cup protect mask 扣除、SAM2 正/负点、HSV 绿色保护扣除、严格 DINO prompt/threshold 基线。
- 输出目录：`/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/<VARIANT>/stack_cups/id_<ID>/stage1_human_inpaint/`。

Validation: `python -m py_compile code_painting/l16_stack_cups_debug_variants.py` 通过；已启动 tmux `l16_stack_debug_variants_gpu1` 跑 `id_0..4`。

## 2026-06-25（stack_cups B 方案全量入口）

- 用户检查四方案 debug 后确认 B `B_points_negative` 和 C `C_hsv_green_protect` 可用；A `A_protect_dino` 与 D `D_tight_dino` 记录为错误路线。
- `l16_stack_cups_debug_variants.py` 新增 `--variants` 参数；`run_l16_stack_cups_debug_variants.sh` 新增 `VARIANTS` 环境变量，因此可只跑 B 方案。
- `COMMAND_LIBRARY.zh.md` 新增 Q 节，记录 B/C 结论、A/D 错误原因，以及 B 全量 Stage-1 和独立 Stage-2 输出路径。

Validation: `python -m py_compile code_painting/l16_stack_cups_debug_variants.py` 与 `bash -n code_painting/run_l16_stack_cups_debug_variants.sh` 通过。

## 2026-06-25（L16 指令归回 I 段并补训练格式转换链路）

- 将 `COMMAND_LIBRARY.zh.md` 末尾的 `Q. L16 stack_cups 绿色杯保护...` 移回 `I3.6.2`，不再保留 Q 段。
- 新增 L9.2：L16 whitebg repaint head + `L16_human_replay_clean` planner state/wrist 转 processed HDF5。
- 新增 L10.7：L16 processed HDF5 转 LeRobot cache。
- 新增 L11.2.5：L16 LeRobot cache 抽取 `_25ep`、zip、rclone dry-run 上传检查。
- 同步更新 `agent-read/COMMANDS/pi0_h2o_training_data.zh.md` 和英文版本，明确 L16 没有 `world_targets_and_status.npz`，应使用 `process_repainted_planner_outputs.py` 而不是 D435 pure replay 的 `process_repainted_headcam_with_wrist.py`。

Validation: `git diff --check`。


## 2026-06-25（L16 ours review 与 selected pipeline）

- `make_l16_repaint_montage.py` 新增可配置 final 路径参数，可复用 P 五联可视化逻辑查看 whitebg/B 方案 final。
- 新增 `code_painting/review_l16_ours_montages.py`：自动发现 final、生成 P montage、交互式按 `y/n/m` 选择 episode，并输出兼容 `--review-json` 的 per-task JSON。
- 新增 `code_painting/run_l16_ours_selected_pipeline.sh`：按 review JSON 生成 `h2o_<TASK>_ours-120`、`local/h2o_<TASK>_ours`、`local/h2o_<TASK>_ours_25ep`，并打包 `robot_ours_<TASK_GROUP>_25ep.zip` 后执行 rclone dry-run/上传。
- 同步更新 `COMMAND_LIBRARY.zh.md` P4 和 L11.2.6，以及 `agent-read/COMMANDS/pi0_h2o_training_data.*.md`。

Validation: `python3 -m py_compile code_painting/make_l16_repaint_montage.py code_painting/review_l16_ours_montages.py`；`bash -n code_painting/run_l16_ours_selected_pipeline.sh`。
