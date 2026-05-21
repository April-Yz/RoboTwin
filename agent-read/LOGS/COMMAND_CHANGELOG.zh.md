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
