# RoboTwin 项目概览

本仓库在 RoboTwin 仿真、数据采集和策略工作流基础上，维护 Piper/Pika 双臂场景、Cartesian IK 抓放、手部/物体 replay 与 AnyGrasp 规划工具。

当前会话上下文和近期数据状态优先见 `agent-read/ACTIVE_MEMORY.zh.md`。

## 当前推荐工作流

- Piper IK 双瓶抓放：使用 `pick_diverse_bottles_piper_ik` 与 `demo_piper_ik_seq_v1..v4`。
- Foundation OBJ 对比：使用 `pick_diverse_bottles_piper_ik_foundation` 与 `demo_piper_ik_foundation_v1..v4`。O.1 使用显式 frame，O.1.1 用第一标注关键帧建场，O.1.2 再用第二关键帧 EE 位置替代 lift/place。O.2 新增 `pnp_tray_piper_ik_foundation`，左手抓 `left_dark_red_cup`、右手抓 `right_bottle`，默认按第二关键帧 OBJ center 放置并打开夹爪。
- 默认 IK：V1；V2 使用三次插值，V3 使用 MotionGen 并带 IK 插值回退，V4 使用多种子 IK。
- 数据流程：Phase 1 找稳定且真实成功的 seed 并保存版本化轨迹；Phase 2 在相同 seed 场景中校验、回放并保存 HDF5/视频/instruction。
- Dense Replay：旧结果保留不动；需要修复 Piper 规划/执行固定偏差时使用隔离的 `run_dense_replay_urdfmatch_v2.sh`。六任务批量入口为 `run_dense_replay_urdfmatch_v2_batch.sh`，独立写入 `h2_pure_d435_urdfmatch_v2` 并支持完整性检查续跑。该版本修复 Curobo/SAPIEN link6 固定轴差和 TCP/执行收敛，不消除人手姿态对机器人的不可达性，也不把 V1 repaint 自动升级为 V2。
- Selection Strategy Audit V4：由脚本只读比较 OursV2、Orientation、Fused 与真实 Top-score 候选；扁平文件名、抗重叠颜色/线型和 agreement/xyz-distance 统计已纳入。当前 Fused 与 Orientation 同候选 93.75%，与 Top canonical 同候选 8.87%，详见 `SELECTION_STRATEGY_AUDIT_V4.zh.md`。
- 相机：head、front、side、右侧 `third_camera`、对向俯视 `opposite_top_camera`、top-level `third_view`；Foundation 配置使用 0515 左右手眼标定、Pika 基础 adapter 和逐侧 wrist 前移/roll tuning 输出独立腕视频，并支持 viewer 实时双腕拼接。
- Foundation 抓取默认使用底部 `support_proxy` 和 close 后几何状态门控；不会通过 `set_pose` 把倾倒物体瞬移回夹爪。
- Foundation 批采集推荐使用独立 run tag、每 ID 一个 episode、最多三次 seed 和外层 timeout，避免复用旧 head-only HDF5 或无限重试。当前 verified wrapper 是 `collect_foundation_piper_ik_verified.sh`，按任务写入稳定 grasp/wrist 参数并保存 head 与左右 wrist 视频。Wrist debug recorder 保存 VS Code 兼容的 H.264 左右/拼接 MP4 和参数 JSON；Motion viewer 支持 `--show_camera_frustums 1` 显示并校验左右 wrist/head 相机框线，Piper IK 执行期间逐步刷新 SAPIEN 主窗口，并可与实时双腕 RGB 窗口同时运行。`script/diagnose_piper_wrist_camera_axes.py` 可复算 wrist camera forward 与 Pika 物理 `+X`、旧 debug `+Z` 和开合轴 `Y` 的关系，并输出可用于 viewer 的 per-side parent yaw 参数。
- `script/index_foundation_piper_ik_videos.py` 可把独立输出中的 Foundation ID N 安全映射为聚合目录的 `episodeN_*`；默认软链接并拒绝覆盖已有 episode。

## 环境与入口

- Conda：`RoboTwin_bw`
- 采集：`collect_data.sh`、`script/collect_data.py`
- Piper IK viewer：`view_pick_diverse_bottles_piper_ik_motion.py`
- 任务：`envs/pick_diverse_bottles_piper_ik.py`
- IK：`envs/robot/piper_ik.py`

命令详见 `agent-read/COMMANDS/piper_ik_cartesian.zh.md`、`piper_ik_foundation.zh.md` 和 `selection_strategy_audit_v4.zh.md`，版本关系详见 `agent-read/VERSION_SUMMARY.zh.md`。
