# VR 人手数据处理命令

## 目的

记录 `/home/zaijia001/ssd/data/piper/vr` 下 Meta Quest 3 VR hand-tracking 数据的结构、统计和诊断可视化命令。

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
