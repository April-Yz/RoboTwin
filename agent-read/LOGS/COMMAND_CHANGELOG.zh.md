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
