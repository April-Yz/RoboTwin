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

# 参数：IK 版本、foundation_input ID、frame、GPU ID、mode
bash collect_foundation_piper_ik.sh v1 0 0 0 o1
bash collect_foundation_piper_ik.sh v1 0 0 0 o1.1
bash collect_foundation_piper_ik.sh v1 0 0 0 o1.2
```

独立采集脚本按版本、ID、frame/mode 生成独立 config 与输出目录。O.1.1/O.1.2 的 frame 由标注第一关键帧覆盖。不能用 `sed -i` 修改基础 YAML 后复用同一个轨迹目录。

## 执行逻辑

- NPZ 提供左右位置、四元数和 OBJ 路径。
- OBJ bounds 的几何中心是抓取基准；actor 原点通常接近瓶底。
- visual 使用原始 OBJ；collision 默认只使用 OBJ 底部的薄圆柱 `support_proxy`，避免完整瓶身在接近时被夹爪推倒。
- 默认直立朝向。FoundationPose 朝向可选，但可能物理不稳定。
- close 前后不再执行 `set_pose`，不会把被撞倒的瓶子瞬移回夹爪。
- close 后检查物体位移、旋转、link6 距离、双指区间投影和径向距离；只有通过才在物体当前 pose 建立 grasp drive。
- 默认 `support_proxy` 没有完整瓶身接触，因此 `foundation_grasp_require_contact=false` 使用几何夹持门控。这是无瞬移的受控 grasp assist，不是纯接触物理。
- Phase 1 顺序规划并保存；Phase 2 校验 Foundation source 和轨迹 schema 后回放，不初始化 planner。

## 抓取调试

默认关键值：

- `foundation_pregrasp_distance: 0.12`
- `foundation_grasp_max_displacement: 0.025`
- `foundation_grasp_max_rotation_deg: 15.0`
- `foundation_capture_radial_tolerance: 0.065`
- `foundation_capture_segment_margin: 0.15`

若仍在 pregrasp 撞到物体，先增大 `foundation_pregrasp_distance`并在 viewer 检查轴线；若在 grasp 时撞到，调整 `foundation_grasp_standoff` 或夹爪朝向。严格接触实验可使用 `cylinder_proxy` / `exact_convex` 加 `foundation_grasp_require_contact: true`，但 input 0 当前会直接暴露碰撞失败，不会再被 pose reset 掩盖。

## 已验证状态

2026-06-11 使用 `foundation_input_0`：O.1/O.1.1/O.1.2 的 V1 viewer 与完整两阶段采集均成功；O.1.2 的 V2/V3/V4 完整采集也成功。input 0 的两个关键帧是 38/78。四个 O.1.2 后端均生成 validated replay、`episode0_succ.hdf5`、instructions 和六路视频。

## 相关代码

- `envs/pick_diverse_bottles_piper_ik_foundation.py`
- `envs/pick_diverse_bottles_piper_ik.py`
- `view_pick_diverse_bottles_piper_ik_motion.py`
- `collect_foundation_piper_ik.sh`
- `task_config/demo_piper_ik_foundation_v1.yml` 至 `v4.yml`
