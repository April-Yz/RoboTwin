# 项目理解

## 主要结构

- `envs/`：任务、机器人控制、观测与 HDF5 转换。
- `assets/embodiments/`：机器人 URDF、关节和静态相机配置。
- `task_config/`：任务采集配置。
- `description/`：任务 prompt 与 episode instruction 生成。
- `code_painting/`：手部/物体 replay、FoundationPose 和 AnyGrasp 相关工作流。

## Piper IK 数据流

输入是带 seed 的双瓶场景、左右瓶功能点、放置目标和 IK 配置。任务生成 pregrasp、grasp、lift、place 四个 Cartesian move 目标。每段 IK 输出 `position[N,6]` 和 `velocity[N,6]`，随后按上一段末端关节状态继续规划。

轨迹 pickle 输出 schema、版本、IK 版本、动作名、Cartesian 目标和左右关节轨迹。Phase 2 只接受当前 v2 schema，并在相同 seed 下回放。观测最终写入 HDF5，各 RGB camera 写入独立 MP4。

## 模型/后端职责

- V1/V2/V4：基于 Piper URDF 的 IK 求解，分别使用线性、三次和多种子策略。
- V3：先求有效 IK 终点，再尝试 MotionGen；MotionGen 不可用或失败时回退到该终点的三次轨迹。
- SAPIEN：执行关节命令、接触仿真、相机渲染和成功判定。

## 已知边界

- V3 的 MotionGen 优化在当前场景可能失败，但回退路径已实测成功。
- viewer 与采集轨迹逻辑一致；采集多一层 pickle schema 校验和观测保存。Piper IK 自定义执行循环必须在 `_update_render()` 后调用 `viewer.render()` 才能实时刷新主窗口；当前 move、settle、gripper 均遵循该顺序。Motion viewer 可同时显示动态 wrist/静态 head 相机视锥和双腕 RGB。Wrist 前向诊断区分 Pika 物理/CAD `+X` 与旧 debug `+Z`，避免把坐标约定差异误判为 90 度外参错误；viewer yaw 参数用于父坐标系 `+Z` 朝向微调，roll 参数只用于绕光轴旋转画面。
- `save_all_episodes` 仅用于调试，不应用于正式成功数据筛选。

## O.1 Foundation OBJ 数据流

O.1 从 `multi_object_world_poses.npz` 读取位置、可选朝向和原始 OBJ 路径。trimesh bounds 用于几何中心和桌面 clearance。input 0 的 OBJ 直径约 6.6cm，完整瓶身碰撞会在 pregrasp/grasp 接近时被当前 Pika 夹爪推倒，因此默认用底部 `support_proxy` 保留桌面支撑。close 后不重置物体 pose，而是校验物体位移/旋转、link6 距离以及双指几何夹持状态，通过后才在当前 pose 建立 drive。

O.1.1 从 `hand_keyframes_all.json` 取第一关键帧的 Foundation OBJ pose。O.1.2 进一步从 `world_targets_and_status.npz` 取第二关键帧的左右 EE xyz，以保留 grasp 朝向的单个 action 替代 lift/place。轨迹绑定 mode、episode ID、关键帧、action 来源、抓取门控参数和 mesh 几何，防止跨设定回放。

Phase 2 只消费已验证关节路径，不创建 IK 或 MotionGen planner，避免 V3 回放占用 GPU 并拖慢多相机渲染。

Foundation 的左右腕相机从 `calibration_bundle_piper_new_table_0515.json` 分别加载 gripper-to-camera 外参。官方 Pika gripper URDF 没有相机 link，官方 Piper+Pika 与当前 AGX 合并模型的 `link6 -> gripper` 轴表达也不同。相机每帧跟随 raw `link6`，应用基础 `piper_pika_agx` adapter、optical-to-render 转换，再应用逐侧 `forward_offset_m/image_roll_deg`；当前左为 `0.125/-15`，右为 `0.11/-60`。该最后一层只统一仿真训练视角，不改标定文件或 IK。严格坐标链还缺 `link6_T_real_tcp`；0515 仅提供 `real_tcp_T_camera`，所以当前 tuning 属于缺失机械外参的经验估计。Debug recorder 从同一渲染帧保存 H.264/faststart 左/右/拼接 MP4 和上下文 JSON；正式 wrapper 可把四个环境变量写入 generated YAML。所有 observation RGB camera 由现有 HDF5/video 合并流程自动导出。

Foundation 数据按 ID 写入独立目录，每个目录内部从 `episode0` 开始。`script/index_foundation_piper_ik_videos.py` 负责在不改源数据的前提下把 ID N 映射为聚合目录中的 `episodeN`，并用 manifest 记录来源；已有目标 episode 默认视为冲突。

批采集 wrapper 强制每 ID 一个 episode并支持 run tag。`script/collect_data.py` 的 `max_seed_tries` 给 seed 搜索设置硬上限；这是针对几何确定性失败的终止条件，不把失败 episode 伪装成成功数据。

## O.2 pnp_tray Foundation OBJ 数据流

O.2 复用 O.1.2 Foundation IK 基类，但通过 `pnp_tray_piper_ik_foundation` 覆盖对象和关键帧路径：左手对象是 `left_dark_red_cup`，右手对象是 `right_bottle`；Foundation 输入来自 `data/piper/hand/pnp_tray/foundation_replay_d435/foundation_input_<ID>`；第二关键帧 EE target 来自 `code_painting/human_replay/h2_pure_d435/pnp_tray/id<ID>_d435_z005/world_targets_and_status.npz`。

O.2 的动作顺序为 `pregrasp -> grasp -> close -> action -> open_gripper`，其中 `action` 是第二关键帧 EE xyz，朝向沿用 grasp 朝向。pnp_tray 的左杯比瓶子更矮更小，实测 `foundation_grasp_standoff=0.14` 会在 ID0 把杯子推偏；当前 O.2 默认使用 `0.105`。正式采集使用 `collect_foundation_piper_ik_verified.sh pnp_tray ...`，仍保存 head 和左右 wrist 视频。
