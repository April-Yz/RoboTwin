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
