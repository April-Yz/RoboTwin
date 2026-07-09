# VR 人手数据处理命令

## 目的

记录 `/home/zaijia001/ssd/data/piper/vr` 下 Meta Quest VR hand-tracking 数据（本地 metadata 写 Quest 3；用户口头说明为 Quest 2，需以后在采集端确认）的结构、统计和诊断可视化命令。

## Q.1 JPG + 手部关节诊断可视化

入口脚本：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/visualize_vr_hand_data.py
```

全量运行：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/visualize_vr_hand_data.py \
  --overwrite \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/vr_hand_visualization
```

快速调试：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/visualize_vr_hand_data.py \
  --episodes NTU-PINE_20260708_145454 \
  --max-frames 30 \
  --overwrite \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/vr_hand_visualization_smoke
```

输出：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/vr_hand_visualization/<EPISODE>/<EPISODE>_hand_overlay_vscode.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/vr_hand_visualization/vr_data_stats.md
/home/zaijia001/ssd/RoboTwin/code_painting/vr_hand_visualization/vr_data_stats.json
```

## 数据字段

- 主 JSON：`format_version=2.0`、`format_type=real_world`、`metadata`、`frames`。
- 每帧包含 `timestamp_seconds`、左右手 `is_tracked/joint_names/poses`、`center/left/right_eye_pose`、左右 `validation`。
- 每只手 26 个 joint；每个 joint pose 为 `[x, y, z, qx, qy, qz, qw]`，四元数约定是 `xyzw_scalar_last`。
- metadata JSON 记录 `env_id=NTU-PINE-v1`、Quest 3 device、`coordinate_frame=RUF`、`joint_capture_mode=FullJointPoses`、JPEG 相机采集配置、episode 成功/时长/reward/tracking_method。
- `camera_real/*_video.json` 记录图片帧数、FPS、分辨率、部分 episode 的 camera intrinsics，以及 `frame_integrity`。

## 相机参数结论

- 13/16 个 episode 有内参：`focal_length=[640,640]`、`principal_point=[640,640]`、`sensor_resolution=[1280,1280]`。
- 0/16 个 episode 有 `cameras` 外参/相机位姿记录。
- 因缺外参，当前叠加视频不是标定投影；脚本按 episode 内 3D joint `x/z` 范围归一化到图像上，只用于诊断。

## 本次统计

- episode: 16
- 可视化视频: 15；`NTU-PINE_20260703_211357` 没有 JPG，所以无视频。
- JSON frames: 3772
- JPG frames: 3231
- left/right tracked 总帧数：1998 / 1905
- left/right validation valid 总帧数：995 / 1160


## Q.2 坐标轴投影 debug 与结果目录

旧版可视化复制结果：

```text
/home/zaijia001/ssd/data/piper/vr/0vis/datav1/<EPISODE>/<EPISODE>_hand_overlay_vscode.mp4
```

`visualize_vr_hand_data.py` 支持 `--projection-mode norm_xz|norm_xy|norm_yz|norm_zx|eye_center`。`norm_*` 是 episode 内坐标轴归一化诊断，`eye_center` 使用 `center_eye_pose + intrinsics` 做近似投影；二者都不是外部相机标定投影。

生成四种 debug 投影：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
for MODE in norm_xy norm_yz norm_zx eye_center; do
  python code_painting/visualize_vr_hand_data.py \
    --overwrite \
    --projection-mode "$MODE" \
    --output-suffix "_${MODE}" \
    --output-root "/home/zaijia001/ssd/data/piper/vr/0vis/datav1_axis_debug/${MODE}"
done
```

输出目录：

```text
/home/zaijia001/ssd/data/piper/vr/0vis/datav1_axis_debug/<MODE>/<EPISODE>/
```

## Q.3 VR JPG 复用 HaMeR 检测

入口脚本：

```text
code_painting/prepare_vr_hamer_input.py
code_painting/compare_vr_hamer_results.py
```

VR episode 转 HaMeR flat input：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/prepare_vr_hamer_input.py \
  --overwrite \
  --output-root /home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1
```

HaMeR 检测：

```bash
unset LD_LIBRARY_PATH && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && \
cd /home/zaijia001/ssd/RoboTwin && \
CUDA_VISIBLE_DEVICES=2 conda run -n hamer-r1-gpu python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py \
  --data_dir /home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/input \
  --output_dir /home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/output \
  --video_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 \
  --no_depth \
  --device cuda
```

对比统计与横向视频：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/compare_vr_hamer_results.py \
  --overwrite \
  --hamer-root /home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1 \
  --vr-vis-root /home/zaijia001/ssd/data/piper/vr/0vis/datav1
```

输出：

```text
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/input/rgb_<ID>.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/input/params_<ID>.json
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/output/hand_vis_gripper_<ID>.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/output/hand_detections_<ID>.npz
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare/id_<ID>_<EPISODE>_vr_vs_hamer_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare/compare_vr_hamer_stats.md
```

## Q.4 坐标系和相机参数结论

- VR JSON 里的 26 个手部 joint 是 VR tracking/world 坐标，metadata 写 `coordinate_frame=RUF`。
- HaMeR 是从 RGB 像素直接检测手，输出的 `hand_vis_gripper_*.mp4` 是图像空间叠加，更适合检查人手位置是否贴合图片。
- 当前 `camera_real/*_video.json` 没有 `cameras` 外参/相机 pose；只有部分 episode 有 `camera_intrinsics`。
- `center_eye_pose/left_eye_pose/right_eye_pose` 是眼/头相关 pose，不等同于 `camera_real` JPG 的 RGB 相机外参。
- 因此 VR joint overlay 偏移不是简单交换轴能修好的问题；缺少外部相机到 VR world/headset 的外参。
- 官方 Meta PCA 文档显示，正确采集方式应保存 PassthroughCameraAccess 返回的 intrinsics、extrinsics、timestamp 或 camera pose。官方 PCA 当前面向 Quest 3/Quest 3S；如果实际设备是 Quest 2，不能假设存在能匹配这些 JPG 的固定官方内参。


## Q.5 20260708 VR-HaMeR eye/lag 对齐 sweep

入口脚本：

```text
code_painting/analyze_vr_hamer_alignment.py
```

用途：只使用 `NTU-PINE_20260708_*` episode，比较 `world_xyz`、`center_eye_xyz`、`left_eye_xyz`、`right_eye_xyz` 四组候选坐标，并对每组分别拟合：

```text
linear_xyz: u_norm,v_norm = linear(x,y,z)
perspective_xy_over_z: u_norm,v_norm = linear(x/z,y/z)
lag: -10..+10，约定 hamer_frame = vr_frame + lag
```

运行命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/analyze_vr_hamer_alignment.py \
  --episode-substr 20260708 \
  --lag-min -10 \
  --lag-max 10 \
  --min-samples 20 \
  --render-videos 1 \
  --overwrite \
  --out-dir /home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_bestfit_20260708
```

输出：

```text
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_bestfit_20260708/alignment_sweep_20260708.md
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_bestfit_20260708/alignment_sweep_20260708.json
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_bestfit_20260708/videos/id_<ID>_<EPISODE>_bestfit_vr_vs_hamer_vscode.mp4
```

本轮结果：11 个 20260708 episode 中，10 个生成 best-fit compare 视频；`NTU-PINE_20260708_143622` 因为 VR tracked + HaMeR detected 的匹配样本少于 20，未拟合。

关键结论：

```text
best pose counts: right_eye_xyz=6, left_eye_xyz=4, center_eye_xyz=0, world_xyz=0
best lag counts: -6=3, -3=3, -5=2, -1=1, +10=1；+10 来自低样本 id5，不建议重视
best model: 9/10 是 linear_xyz，1/10 是 perspective_xy_over_z
```

解释：`left_eye_pose/right_eye_pose` 有帮助，但没有单一 eye pose 稳定胜出；center/left/right 的全局加权指标非常接近，说明 eye 选择不是主因。多数 episode 的最佳 lag 是负数，按 `hamer_frame = vr_frame + lag` 约定，意味着同一事件更像是 `VR hand frame` 需要取比 RGB/HaMeR 更晚的帧，存在约 3-6 帧（100-200ms @30fps）的时间偏移。`linear_xyz` 明显优于 raw world 和大多 perspective，说明当前图像更像 Quest 用户视角/屏幕合成空间，而不是未裁剪的 raw pinhole passthrough camera。

## Q.6 20260708 VR-HaMeR 3D diagnostic visualization

入口脚本：

```text
code_painting/visualize_vr_hamer_3d_diagnostics.py
```

用途：只使用 `NTU-PINE_20260708_*` episode，在 VR/world RUF 坐标中解释“HaMeR 2D overlay 贴图像手，但 VR hand joint overlay 偏移”的原因。脚本读取 VR hand joints、`is_tracked`、`center_eye_pose/left_eye_pose/right_eye_pose`、HaMeR 2D detections，以及 Q.5 生成的 best pose/model/lag。

运行命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/visualize_vr_hamer_3d_diagnostics.py \
  --episode-substr 20260708 \
  --overwrite \
  --out-dir /home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708
```

输出：

```text
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708/summary_3d_20260708.md
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708/summary_3d_20260708.json
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708/id_<ID>_<EPISODE>/high_back_3d_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708/id_<ID>_<EPISODE>/front_3d_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708/id_<ID>_<EPISODE>/top_3d_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708/id_<ID>_<EPISODE>/quadview_3d_vscode.mp4
```

可视化规则：

- 3D 场景统一使用 VR/world RUF：`X right`、`Y up`、`Z forward`。
- 画左右 VR hand skeleton、最近轨迹、`center/left/right_eye_pose` 坐标轴和 frustum。
- HaMeR 不被当成真实 world 3D；只画图像检测点和从 best eye pose 发出的 2D back-projection ray。
- quadview 第一格是原图 + HaMeR 2D + bestfit pseudo VR overlay；其余三格是 high_back、front、top 3D 视角。
- 每帧叠加 episode id、frame、best pose/model/lag、R2/RMSE、valid samples。

本次运行结果：

- 输入：11 个 `NTU-PINE_20260708_*` episode。
- 成功：10 个 episode，每个输出 4 个 VSCode 可直接查看的 mp4，共 40 个视频。
- 跳过：`NTU-PINE_20260708_143622`，原因是所有 pose/model/lag 组合都少于 20 个 matched tracked+detected 样本，Q.5 没有可靠 bestfit。

优先查看：

```text
id_11_NTU-PINE_20260708_145659/quadview_3d_vscode.mp4
id_9_NTU-PINE_20260708_145601/quadview_3d_vscode.mp4
id_8_NTU-PINE_20260708_145546/quadview_3d_vscode.mp4
id_6_NTU-PINE_20260708_145454/quadview_3d_vscode.mp4
id_14_NTU-PINE_20260708_145845/quadview_3d_vscode.mp4
```

这些 episode 的 Q.5 bestfit 指标相对可靠，可用于判断 VR world hand trajectory 是否平滑、HaMeR back-projection ray 是否大致穿过 VR 手部体积。反例建议看 `id_12_NTU-PINE_20260708_145729`、`id_7_NTU-PINE_20260708_145505`、`id_10_NTU-PINE_20260708_145638`，它们更适合观察纵向失配、低 R2 或疑似时间/投影问题。

当前结论：

- 好的 episode 中 VR hand world 轨迹本身是连续的，不像简单轴交换错误。
- best pose 在 left/right eye 之间切换，center/world 不胜出，说明 eye pose 只是近似视角，不是 `camera_real` JPG 的真实外参。
- 多数 best lag 为 -3 到 -6 帧，说明存在约 100-200ms 级别的同步偏移。
- 大多数 best model 是 `linear_xyz` 而不是 pinhole-style perspective，支持“录屏/用户视角经过合成、裁剪、缩放或 warp”的判断。
- 可做 episode-local 粗校正：id6、id8、id9、id11、id14；id13 中等；id7/id10 只能局部参考；id12 不可信；id5 样本少且 +10 lag，需要谨慎；id4 不可用。

## Q.7 20260708 hand-local alignment validation

入口脚本：

```text
code_painting/validate_vr_hamer_local_hand_alignment.py
```

用途：只使用 `NTU-PINE_20260708_*` episode，验证 VR hand joints 和 HaMeR 2D keypoints 的手部局部骨架关系与运动趋势是否一致。该流程不要求 `camera_real` JPG 的真实 raw camera 外参，也不要求 world/camera 全局投影对齐。

运行命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/validate_vr_hamer_local_hand_alignment.py \
  --episode-substr 20260708 \
  --overwrite \
  --out-dir /home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708
```

不重转视频、只复算 summary 时可以去掉 `--overwrite`。

输出：

```text
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708/summary_local_hand_alignment_20260708.md
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708/summary_local_hand_alignment_20260708.json
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708/summary_local_hand_alignment_20260708.csv
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708/id_<ID>_<EPISODE>/image_overlay_local_alignment_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708/id_<ID>_<EPISODE>/local_skeleton_comparison_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708/id_<ID>_<EPISODE>/error_heatmap_timeplot_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708/id_<ID>_<EPISODE>/motion_trend_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708/id_<ID>_<EPISODE>/quadview_local_hand_alignment_vscode.mp4
```

VR 26 joint 到 HaMeR 21 keypoint 的映射：

```text
wrist -> wrist
thumb_cmc/mcp/ip/tip -> thumb_metacarpal/proximal/distal/tip
index_mcp/pip/dip/tip -> index_proximal/intermediate/distal/tip
middle_mcp/pip/dip/tip -> middle_proximal/intermediate/distal/tip
ring_mcp/pip/dip/tip -> ring_proximal/intermediate/distal/tip
pinky_mcp/pip/dip/tip -> little_proximal/intermediate/distal/tip
```

忽略的 VR joint：`palm`、`index_metacarpal`、`middle_metacarpal`、`ring_metacarpal`、`little_metacarpal`。原因是 HaMeR 21 点没有这些额外 palm/metacarpal 点；这里优先比较可见手指关节。

局部坐标定义：

- HaMeR 侧：以 wrist 为原点，用 `index_mcp - pinky_mcp` 定义局部 x 轴，用 `middle_mcp - wrist` 的正交分量定义局部 y 轴，并用 palm scale 归一化。
- VR 侧：以 wrist 为原点，用对应的 VR index/pinky/middle 点构造 3D palm basis，再投到局部 2D palm plane，并用同类 palm scale 归一化。
- 每个 episode 独立 sweep `lag=-10..10`，每只手分别评估 similarity/Procrustes 与 2D affine，对每个 episode 选择最佳 side/lag/alignment。

指标：

- `rmse_mean`：best alignment 后的 21 点局部 RMSE。
- `per_keypoint_rmse` / `per_finger_rmse`：逐点和逐手指误差，保存在 JSON。
- `pairwise_distance_rmse`：局部归一化点集的两两距离矩阵误差。
- `bone_length_log_error`：骨长比例误差。
- `joint_angle_deg_error`：手指关节角误差。
- `velocity_corr` / `delta_direction_agreement`：hand-local 形状变化趋势的一致性。
- `local_shape_score`：综合局部形状和运动趋势的 0-100 分。

本次运行结果：

- 输入：11 个 `NTU-PINE_20260708_*` episode。
- 成功：9 个 episode，每个输出 5 个 VSCode 可读 mp4，共 45 个视频。
- 跳过：`id4 NTU-PINE_20260708_143622` 和 `id5 NTU-PINE_20260708_143721`，原因是所有 side/lag 的 VR tracked + HaMeR detected 匹配样本都少于 20。
- 分类：9 个成功 episode 全部为 `medium`，没有 `good` / `bad`。

解释：

- `id6/id8/id9/id11/id14`：hand-local medium，并且上一轮 global bestfit 也相对合理；这些 episode 更适合继续做 episode-local 粗校正。
- `id7/id10/id12/id13`：hand-local medium，但上一轮 global/world projection 指标弱；更像录屏/用户视角合成、裁剪、同步或外参缺失问题，而不是 VR hand local tracking 完全失败。
- `id4/id5`：样本不足，不建议用于 hand-local 或 world trajectory 判断。
- 所有 rendered episode 的 best alignment 都是 affine；similarity 结果保存在 JSON 的 `best_similarity`。这说明局部运动趋势很强，但严格刚性/相似变换下的骨架形状只能算中等，不应把这些数据当作精确 3D 手骨架标定结果。

## Q.8 20260708 both-hands v2 local alignment validation

入口脚本：

```text
code_painting/validate_vr_hamer_local_hand_alignment_bothhands.py
```

用途：在 Q.7 的 hand-local 验证基础上，不再只选单个 best side，而是分别输出 `left_only`、`right_only`、`both_hands`。该流程继续只使用 `NTU-PINE_20260708_*`，不要求真实 raw camera 外参或 world/camera 全局投影对齐。

运行命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/validate_vr_hamer_local_hand_alignment_bothhands.py \
  --episode-substr 20260708 \
  --overwrite \
  --out-dir /home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands
```

输出：

```text
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands/summary_bothhands_local_alignment_20260708.md
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands/summary_bothhands_local_alignment_20260708.json
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands/summary_bothhands_local_alignment_20260708.csv
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands/id_<ID>_<EPISODE>/left_quadview_local_alignment_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands/id_<ID>_<EPISODE>/right_quadview_local_alignment_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands/id_<ID>_<EPISODE>/bothhands_quadview_local_alignment_vscode.mp4
```

每个成功 episode 实际还会为 `left/right/bothhands` 各自输出四个组件视频：`image_overlay`、`local_skeleton_comparison`、`error_heatmap_timeplot`、`motion_trend`，以及一个 `quadview`。

mapping hypotheses：

```text
identity: VR_left -> HaMeR_left, VR_right -> HaMeR_right
swapped:  VR_left -> HaMeR_right, VR_right -> HaMeR_left
```

mirror/canonical variants：

```text
none, mirror_x, mirror_y, mirror_xy
```

这些 mirror 变体是在 VR hand-local 坐标进入 similarity/affine alignment 前应用的，用于显式测试 HaMeR 可能使用 right-hand MANO 或 canonicalized hand 约定导致的左右/镜像问题。注意：affine alignment 本身也可以吸收 reflection，因此严格判断局部骨架形状时要看 JSON 中的 `best_similarity`，不要只看 best affine。

本次运行结果：

- 输入：11 个 `NTU-PINE_20260708_*` episode。
- 成功：9 个 episode，每个输出 15 个 VSCode 可读 mp4，共 135 个视频。
- 跳过：`id4 NTU-PINE_20260708_143622` 和 `id5 NTU-PINE_20260708_143721`，原因是 `left_only/right_only/both_hands` 的匹配样本都不足。
- `left_only`：9 个成功 episode 均为 `medium`。
- `right_only`：9 个成功 episode 均为 `medium`。
- `both_hands`：9 个成功 episode 均为 `medium`。
- both-hands 最佳 mapping：`identity=8`、`swapped=1`。`swapped` 只出现在 `id13 NTU-PINE_20260708_145832`。
- both-hands 最佳 mirror：全部是 `none`，没有 episode 需要 mirror_x/mirror_y/mirror_xy 才达到最佳。

解释：

- 双手 v2 进一步支持：大多数 20260708 episode 的左右手 side label 在双手关系上更接近 identity，不需要整体左右交换。
- 单手 `left_only` 经常会独立选择 `VR_left -> HaMeR_right`，但 both-hands 联合约束下大多数 episode 仍选择 identity；说明单手局部形状 alone 不足以稳定判断 side label，双手相对关系更可靠。
- `id13` 的 both-hands 选择 swapped，且 score 仅中等偏低，建议作为 side-label/同步异常样例检查，不要直接用于世界坐标双手轨迹。
- 所有成功 episode 仍是 `medium` 而非 `good`，说明局部运动趋势可用，但严格骨架形状与双手关系不能当成精确标定结果。
