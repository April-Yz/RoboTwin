# Piper / Pika 场景指令库

## 用途
整理手动桌面摆位、标定场景、version-B 标定场景的常用命令。

## 指令

- 用斜视相机在交互 viewer 中打开手动桌面场景。
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/visualize_piper_pika_agx_dual_table.py \
  --offscreen-only 0 \
  --camera-mode oblique \
  --output-dir /home/zaijia001/ssd/RoboTwin/code_painting/pika/output_piper_pika_agx_dual_table_manual_viewer
```

- 用俯视相机在交互 viewer 中打开手动桌面场景。
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/visualize_piper_pika_agx_dual_table.py \
  --offscreen-only 0 \
  --camera-mode top_down \
  --output-dir /home/zaijia001/ssd/RoboTwin/code_painting/pika/output_piper_pika_agx_dual_table_manual_topdown_viewer
```

- 修改 `robot_config_PiperPika_agx_dual_table.json` 后，用斜视相机重新导出手动桌面场景。
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/visualize_piper_pika_agx_dual_table.py \
  --offscreen-only 1 \
  --camera-mode oblique \
  --output-dir /home/zaijia001/ssd/RoboTwin/code_painting/pika/output_piper_pika_agx_dual_table_manual \
  --image-name piper_pika_agx_dual_table_manual.png \
  --video-name piper_pika_agx_dual_table_manual.mp4
```

- 修改 `robot_config_PiperPika_agx_dual_table.json` 后，用俯视相机重新导出手动桌面场景。
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/visualize_piper_pika_agx_dual_table.py \
  --offscreen-only 1 \
  --camera-mode top_down \
  --output-dir /home/zaijia001/ssd/RoboTwin/code_painting/pika/output_piper_pika_agx_dual_table_manual_topdown \
  --image-name piper_pika_agx_dual_table_manual_topdown.png \
  --video-name piper_pika_agx_dual_table_manual_topdown.mp4
```

- 导出基于当前手动桌面摆位的真实标定场景重建结果。
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene.py
```

- 导出去掉 +90° 锚定旋转后的 version-B 标定场景。
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py
```

- 在交互 viewer 中打开标定场景 A（默认 overview 视角）。
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene.py \
  --viewer 1 \
  --viewer-camera overview \
  --save-image 0 \
  --save-video 0 \
  --save-headcam-image 0
```

- 在交互 viewer 中打开标定场景 A（head cam 视角）。
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene.py \
  --viewer 1 \
  --viewer-camera head \
  --save-image 0 \
  --save-video 0 \
  --save-headcam-image 0
```

- 只导出标定场景 A 的 head cam 视角图（不导总览图、不导视频）。
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene.py \
  --save-image 0 \
  --save-video 0 \
  --save-headcam-image 1 \
  --headcam-image-name calibrated_scene_headcam.png
```

- 在交互 viewer 中打开标定场景 version-B（默认 overview 视角）。
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py \
  --viewer 1 \
  --viewer-camera overview \
  --save-image 0 \
  --save-video 0 \
  --save-headcam-image 0
```

- 在交互 viewer 中打开标定场景 version-B（head cam 视角）。
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py \
  --viewer 1 \
  --viewer-camera head \
  --save-image 0 \
  --save-video 0 \
  --save-headcam-image 0
```

- 只导出 version-B 标定场景的 head cam 视角图（不导总览图、不导视频）。
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py \
  --save-image 0 \
  --save-video 0 \
  --save-headcam-image 1 \
  --headcam-image-name calibrated_scene_vb_headcam.png
```

## 标定场景说明
- 这两个标定场景脚本确实使用了你的 calibration bundle 来放置第二台机械臂和 head cam。
- 但当前总览图里，head cam 只被画成一个比较小的灰色 marker 加 RGB 坐标轴，所以不一定一眼能看出来。

## Viewer 说明
- 当前可用的交互 viewer 命令：
  - `visualize_piper_pika_agx_dual_table.py --offscreen-only 0 --camera-mode oblique`
  - `visualize_piper_pika_agx_dual_table.py --offscreen-only 0 --camera-mode top_down`
  - `visualize_calibrated_piper_pika_scene.py --viewer 1 --viewer-camera overview|head`
  - `visualize_calibrated_piper_pika_scene_vb.py --viewer 1 --viewer-camera overview|head`
- 注意：viewer 命令主要用于交互查看；wrist 视角图片导出目前还没实现。

## Wrist 视角现状
- 当前这两版标定场景脚本还没有实现 left/right wrist 相机图像导出。
- 目前可导出的是 overview 和 head cam 视角；wrist 视角需要后续补功能。

## Piper 夹爪局部轴扫图

- 生成 Piper 夹爪局部 `x/y/z` 轴扫图板。
```bash
GPU=1 FRAME_IDX=0 ARM=left EXECUTE=0 SAVE_WRIST_VIEWS=0 \
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_local_axis_sweep_board.sh \
  /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_local_axis_sweep_board
```

- 用途：
  - 固定 `PiperPika` 双臂场景与当前 head cam 标定值
  - 对单帧单臂枚举全部合法右手系局部轴 remap
  - 每个候选都会输出带红绿蓝轴的 `zed/third` 图，并在图上直接标注 `x/y/z` 相对机器人 `forward/left/up` 的语义

- 关键输出：
  - `per_case/*.png`
  - `board_zed.png`
  - `board_third.png`
  - `summary.json`
  - `summary.csv`

- 常用环境变量：
  - `FRAME_IDX`：选哪一帧
  - `FRAME_END` / `MAX_FRAMES` / `FRAME_STRIDE`：多帧视频范围控制；`MAX_FRAMES=-1` 表示不截断
  - `ARM`：`left` 或 `right`
  - `EXECUTE`：`0` 只看局部轴，不执行；`1` 会额外走一遍 IK 执行并记录执行误差
  - `CANDIDATE_MODE`：`remap` 扫 24 个右手系 remap；`semantic` 扫 `forward_from_*` 和 `open_from_*` 语义候选
  - `VIDEO_MODE`：`1` 时写出 `board_all_zed.mp4` 和 `board_success_zed.mp4`
  - `SAVE_WRIST_VIEWS`：`1` 时额外导出 wrist 视角扫图板

- 生成 id0 多帧 remap 大拼图视频。
```bash
GPU=3 FRAME_IDX=0 FRAME_END=-1 MAX_FRAMES=32 ARM=left EXECUTE=1 \
CANDIDATE_MODE=remap VIDEO_MODE=1 FPS=5 SAVE_WRIST_VIEWS=0 \
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_local_axis_sweep_board.sh \
  /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_local_axis_sweep_video_remap
```

- 生成更少的语义候选视频。
```bash
GPU=3 FRAME_IDX=0 FRAME_END=-1 MAX_FRAMES=32 ARM=left EXECUTE=1 \
CANDIDATE_MODE=semantic VIDEO_MODE=1 FPS=5 SAVE_WRIST_VIEWS=0 \
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_local_axis_sweep_board.sh \
  /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_local_axis_sweep_video_semantic
```

- 多帧输出：
  - `board_all_zed.mp4`：全部候选大拼图视频
  - `board_success_zed.mp4`：只把 success 候选紧凑拼接的视频
  - `board_frames/frame_XXXX_all_zed.png`
  - `board_frames/frame_XXXX_success_zed.png`
