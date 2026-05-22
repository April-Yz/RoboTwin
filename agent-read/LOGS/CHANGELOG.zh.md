# CHANGELOG.zh

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
