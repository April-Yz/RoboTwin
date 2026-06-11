# Piper IK Foundation OBJ 抓放

## 用途

使用 Foundation NPZ 中的位置和原始 OBJ，在 SAPIEN 中运行 Piper V1-V4 Cartesian 抓放、viewer 和两阶段数据采集。

## 前置条件

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw
cd /home/zaijia001/ssd/RoboTwin
```

## 模式

- `o1`：使用显式 `foundation_frame`完成 `pregrasp -> grasp -> close -> lift -> place -> open`。
- `o1.1`：从 `hand_keyframes_all.json` 取 episode 的第一标注关键帧作为物体初始 pose，动作序列与 O.1 相同。
- `o1.2`：第一关键帧用于建场和抓取；第二关键帧从 `world_targets_and_status.npz` 取左右 EE xyz，以单个 `action` 取代 lift/place，朝向保留 grasp 姿态。

## Viewer

```bash
python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --task_config demo_piper_ik_foundation_v1 \
  --ik_version v1 --foundation_id 0 --foundation_frame 0 --foundation_mode o1 \
  --seed 0 --max_seed_tries 1 --require_success 1
```

切换 O.1.1/O.1.2 时改为 `--foundation_mode o1.1` 或 `--foundation_mode o1.2`。Viewer 会创建 SAPIEN 窗口，需要可用的 X11/Vulkan display；不要对 viewer 使用 `unset DISPLAY`。

## 数据采集

```bash
# 固定 input 0
bash collect_data.sh pick_diverse_bottles_piper_ik_foundation demo_piper_ik_foundation_v1 0

# 参数：IK 版本、foundation_input ID、frame、GPU ID、mode、可选 run_tag
bash collect_foundation_piper_ik.sh v1 0 0 0 o1
bash collect_foundation_piper_ik.sh v1 0 0 0 o1.1
bash collect_foundation_piper_ik.sh v1 0 0 0 o1.2 wrist0515
```

独立采集脚本按版本、ID、frame/mode/run tag 生成独立 config 与输出目录，并强制每个 ID 只采一个 episode。O.1.1/O.1.2 的 frame 由标注第一关键帧覆盖。启用腕相机后必须使用新 run tag；已有 HDF5 会被跳过，旧 head-only 目录不会自动补出腕部视频。不能用 `sed -i` 修改基础 YAML 后复用同一个轨迹目录。

推荐批量形式：

```bash
RUN_TAG=wrist0515
FAIL_LOG=/tmp/o12_v1_${RUN_TAG}_failures.log
: > "$FAIL_LOG"
for id in $(seq 0 120); do
  timeout 600s bash collect_foundation_piper_ik.sh v1 "$id" 0 0 o1.2 "$RUN_TAG" \
    || echo "FAIL v1 id=$id status=$?" | tee -a "$FAIL_LOG"
done
```

V2/V3/V4 分别把版本和 GPU 改为 `v2/1`、`v3/2`、`v4/3`。实际输入当前为 ID 0-101；扫描到 102-120 时 wrapper 会因缺少 NPZ 快速失败并记入日志。

## 执行逻辑

- NPZ 提供左右位置、四元数和 OBJ 路径。
- OBJ bounds 的几何中心是抓取基准；actor 原点通常接近瓶底。
- visual 使用原始 OBJ；collision 默认只使用 OBJ 底部的薄圆柱 `support_proxy`，避免完整瓶身在接近时被夹爪推倒。
- 默认直立朝向。FoundationPose 朝向可选，但可能物理不稳定。
- close 前后不再执行 `set_pose`，不会把被撞倒的瓶子瞬移回夹爪。
- close 后检查物体位移、旋转、link6 距离、双指区间投影和径向距离；只有通过才在物体当前 pose 建立 grasp drive。
- 默认 `support_proxy` 没有完整瓶身接触，因此 `foundation_grasp_require_contact=false` 使用几何夹持门控。这是无瞬移的受控 grasp assist，不是纯接触物理。
- Phase 1 顺序规划并保存；Phase 2 校验 Foundation source 和轨迹 schema 后回放，不初始化 planner。
- 每个版本最多尝试 `max_seed_tries: 3` 个 seed；确定性失败会返回非零状态，不再无限循环伪装成卡住。

## 腕部相机

- 四份 Foundation 配置均启用 `collect_wrist_camera: true`。
- 左右相机分别读取 `calibration_bundle_piper_new_table_0515.json` 中的 `left_gripper_T_camera` / `right_gripper_T_camera`，不能共用一份外参。
- 标定的 gripper frame 对应 planner EE frame，因此使用 `wrist_camera_pose_reference: planner_gripper`；直接挂 raw `link6` 会让画面落在夹爪底座内。
- `wrist_camera_axis_mode: legacy_r1` 把标定的 OpenCV optical frame 转为 SAPIEN render frame。
- Phase 2 会自动把所有 observation RGB camera 写入 MP4，无需额外视频导出器。新增文件是 `episode0_succ_left_camera.mp4` 和 `episode0_succ_right_camera.mp4`，HDF5 中也包含同名 observation camera。

## 抓取调试

默认关键值：

- `foundation_pregrasp_distance: 0.12`
- `foundation_grasp_max_displacement: 0.025`
- `foundation_grasp_max_rotation_deg: 15.0`
- `foundation_capture_radial_tolerance: 0.065`
- `foundation_capture_segment_margin: 0.15`

若仍在 pregrasp 撞到物体，先增大 `foundation_pregrasp_distance`并在 viewer 检查轴线；若在 grasp 时撞到，调整 `foundation_grasp_standoff` 或夹爪朝向。严格接触实验可使用 `cylinder_proxy` / `exact_convex` 加 `foundation_grasp_require_contact: true`，但 input 0 当前会直接暴露碰撞失败，不会再被 pose reset 掩盖。

## 已验证状态

2026-06-11 使用 `foundation_input_0`：O.1/O.1.1/O.1.2 的 V1 viewer 与完整两阶段采集均成功；O.1.2 的 V2/V3/V4 完整采集也成功。input 0 的两个关键帧是 38/78。修正腕相机 reference 后，V1 O.1.2 完整生成 validated replay、`episode0_succ.hdf5`、instructions 和八路视频；左右腕视频各 38 帧、320x240，且随末端分别移动约 0.37m/0.46m。

tmux 批处理复查显示会话已经回到 shell，并非仍在运行。旧任务出现 `Killed`/Ctrl-C；V1 基础配置曾把每个 ID 设为 10 episodes，而失败搜索又没有上限。V4 ID 9 在多个 seed 上稳定触发右侧 grasp 旋转约 25.6 度、超过 15 度门限，继续换 seed 不会解决同一关键帧几何问题。当前配置把每 ID 改为 1 episode、最多 3 次 seed，并建议外层再加 `timeout`。

## 相关代码

- `envs/pick_diverse_bottles_piper_ik_foundation.py`
- `envs/pick_diverse_bottles_piper_ik.py`
- `view_pick_diverse_bottles_piper_ik_motion.py`
- `collect_foundation_piper_ik.sh`
- `task_config/demo_piper_ik_foundation_v1.yml` 至 `v4.yml`
