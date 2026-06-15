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
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --seed 0 --max_seed_tries 1 --require_success 1 --wrist_preview 1
```

`--wrist_preview 1` 会额外显示左右 wrist RGB 拼接窗口。Viewer 需要可用的 X11/Vulkan display；远程桌面可显式使用 `DISPLAY=:1.0`，不要使用 `unset DISPLAY`。

## 数据采集

```bash
# 固定 input 0
bash collect_data.sh pick_diverse_bottles_piper_ik_foundation demo_piper_ik_foundation_v1 0

# 参数：IK 版本、foundation_input ID、frame、GPU ID、mode、可选 run_tag
bash collect_foundation_piper_ik.sh v1 0 0 0 o1
bash collect_foundation_piper_ik.sh v1 0 0 0 o1.1
bash collect_foundation_piper_ik.sh v1 0 0 0 o1.2 wrist_o121_verified_0615
```

独立采集脚本按版本、ID、frame/mode/run tag 生成独立 config 与输出目录，并强制每个 ID 只采一个 episode。O.1.1/O.1.2 的 frame 由标注第一关键帧覆盖。启用腕相机后必须使用新 run tag；已有 HDF5 会被跳过，旧 head-only 目录不会自动补出腕部视频。不能用 `sed -i` 修改基础 YAML 后复用同一个轨迹目录。

推荐批量形式：

```bash
RUN_TAG=wrist_o121_verified_0615
mkdir -p data/tmp
FAIL_LOG="$PWD/data/tmp/o12_v1_${RUN_TAG}_failures.log"
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
- 相机以 `wrist_camera_pose_reference: urdf_end_link` 跟随仿真 `link6`，使用 `wrist_camera_simulation_adapter: piper_pika_agx` 处理基础 TCP/CAD 平移，再用逐侧 `wrist_camera_tuning` 处理镜头前移和训练画面 roll。
- 当前 tuning：左 `0.125 m/-15 deg`，右 `0.11 m/-60 deg`。它不改写 0515 JSON，也不改变 IK；左右前移差与两份外参约 1.43 cm 的 X 平移差一致。
- `wrist_camera_axis_mode: legacy_r1` 把标定的 OpenCV optical frame 转为 SAPIEN render frame。
- Phase 2 会自动把所有 observation RGB camera 写入 MP4，无需额外视频导出器。新增文件是 `episode0_succ_left_camera.mp4` 和 `episode0_succ_right_camera.mp4`，HDF5 中也包含同名 observation camera。

## ID 到 Episode 视频索引

Foundation 每个 ID 的独立目录内部使用 `episode0_*`。使用以下命令将源 ID N 映射为聚合目录中的 `episodeN_*`：

```bash
python script/index_foundation_piper_ik_videos.py \
  --version v4 --mode o1.2 --run-tag wrist_o121_verified_0615 \
  --output-video-dir data/pick_diverse_bottles_piper_ik/demo_piper_ik_foundation_v4_o1_2_wrist_o121_verified_0615/video \
  --method symlink --dry-run
```

确认后去掉 `--dry-run`。脚本写入 `foundation_episode_index.json`。已有 episode 默认报冲突；`--replace-episode` 会删除目标目录中该 ID 的旧 MP4，因此仅在明确替换时使用。

当前已有的 V4/O.1.2 目录没有 run tag，因此索引旧数据时省略 `--run-tag`。对 `demo_piper_ik_v4_3/video` 的 dry-run 实测可索引 ID 0-8，并在已有 ID 0-4 上报告冲突；ID 9 没有成功视频。

## 抓取调试

默认关键值：

- `foundation_pregrasp_distance: 0.12`
- `foundation_grasp_max_displacement: 0.025`
- `foundation_grasp_max_rotation_deg: 15.0`
- `foundation_capture_radial_tolerance: 0.065`
- `foundation_capture_segment_margin: 0.15`

若仍在 pregrasp 撞到物体，先增大 `foundation_pregrasp_distance`并在 viewer 检查轴线；若在 grasp 时撞到，调整 `foundation_grasp_standoff` 或夹爪朝向。严格接触实验可使用 `cylinder_proxy` / `exact_convex` 加 `foundation_grasp_require_contact: true`，但 input 0 当前会直接暴露碰撞失败，不会再被 pose reset 掩盖。

## 已验证状态

2026-06-15 使用 `foundation_input_0` 验证 O.1.2 V4：最终 tuning 完整生成 validated replay、`episode0_succ.hdf5`、instructions 和八路视频；左右腕视频各 38 帧、320x240，抽帧确认外壳退出画面且右侧 roll 扶正。tmux 中旧无 tag 输出由未初始化 `RUN_TAG` 导致，不能与新结果混用。

Viewer 可临时覆盖而不改 YAML：`--wrist_left_forward_offset_m`、`--wrist_right_forward_offset_m`、`--wrist_left_roll_deg`、`--wrist_right_roll_deg`。官方 Pika gripper URDF 不包含 camera link；当前 wrist 内参仍使用 D435 预设，严格 D405 对齐还需要实测内参和镜头 optical-center CAD 变换。

tmux 批处理复查显示会话已经回到 shell，并非仍在运行。旧任务出现 `Killed`/Ctrl-C；V1 基础配置曾把每个 ID 设为 10 episodes，而失败搜索又没有上限。V4 ID 9 在多个 seed 上稳定触发右侧 grasp 旋转约 25.6 度、超过 15 度门限，继续换 seed 不会解决同一关键帧几何问题。当前配置把每 ID 改为 1 episode、最多 3 次 seed，并建议外层再加 `timeout`。

## 相关代码

- `envs/pick_diverse_bottles_piper_ik_foundation.py`
- `envs/pick_diverse_bottles_piper_ik.py`
- `view_pick_diverse_bottles_piper_ik_motion.py`
- `collect_foundation_piper_ik.sh`
- `script/index_foundation_piper_ik_videos.py`
- `code_painting/build_piper_calibration_bundle.py`
- `task_config/demo_piper_ik_foundation_v1.yml` 至 `v4.yml`

## O.1.2.1 坐标结论与 Viewer Debug 录制

正确链条是 `world_T_link6 @ link6_T_real_tcp @ real_tcp_T_camera @ optical_T_render`。0515 只提供 `real_tcp_T_camera`，当前缺少实机 TCP/支架/镜头中心对应的 `link6_T_real_tcp`。因此矩阵可以拼接，问题是省略了未知机械外参；当前 tuning 是经验补偿，不是新的物理标定。

forward offset 正值沿每台相机自身视线前移。当前 render frame 中 roll 正值顺时针、负值逆时针。加 `--wrist_debug_record 1 --wrist_debug_tag <TAG>` 会保存左右原始 MP4、带标签拼接 MP4 和参数 JSON 到 `data/wrist_camera_debug/<TAG>/`。

Debug MP4 使用 `H.264/avc1 + yuv420p + faststart`，可直接在 VS Code 预览。无窗口快速录制加 `--render_freq 0 --show_axes 0 --wrist_preview 0`。若要走正式采集链路，在 `collect_foundation_piper_ik.sh` 前同时设置 `WRIST_LEFT_FORWARD_OFFSET_M`、`WRIST_RIGHT_FORWARD_OFFSET_M`、`WRIST_LEFT_ROLL_DEG`、`WRIST_RIGHT_ROLL_DEG`；wrapper 会把参数写入隔离的 generated YAML，Phase 2 继续生成正式 HDF5 和 8 路 H.264 视频。

```bash
unset DISPLAY
WRIST_LEFT_FORWARD_OFFSET_M=0.125 \
WRIST_RIGHT_FORWARD_OFFSET_M=0.11 \
WRIST_LEFT_ROLL_DEG=-15 \
WRIST_RIGHT_ROLL_DEG=-60 \
timeout 600s bash collect_foundation_piper_ik.sh \
  v1 0 0 0 o1.2 wrist_left125_right110_roll_m15_m60
```

## 带相机框线的 Viewer

```bash
export DISPLAY=:1.0
TAG="o121_v1_viewer_$(date +%Y%m%d_%H%M%S)"
python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --render_freq 1 --show_axes 1 --show_camera_frustums 1 \
  --wrist_preview 1 --hold 1 \
  --wrist_left_forward_offset_m 0.125 --wrist_right_forward_offset_m 0.11 \
  --wrist_left_roll_deg -15 --wrist_right_roll_deg -60 \
  --wrist_debug_record 1 --wrist_debug_tag "$TAG" \
  --max_seed_tries 1 --require_success 1
```

`--show_camera_frustums 1` 启用 SAPIEN 相机视锥，并验证 `left_camera`、`right_camera`、`head_camera` 都存在。`--wrist_preview 1` 是独立的双腕 RGB 窗口；`--hold 1` 会保持最终 viewer。执行过 `unset DISPLAY` 后必须用 `export DISPLAY=:1.0` 恢复，`set DISPLAY` 无效。时间戳 tag 避免已有非空输出目录导致 `FileExistsError`。
