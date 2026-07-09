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
