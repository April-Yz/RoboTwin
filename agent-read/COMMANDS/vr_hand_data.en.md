# VR Hand Data Commands

## Purpose

Document the structure, statistics, and diagnostic visualization commands for the Meta Quest 3 VR hand-tracking data under `/home/zaijia001/ssd/data/piper/vr`.

## Q.1 JPG + Hand-Joint Diagnostic Visualization

Entrypoint:

```text
/home/zaijia001/ssd/RoboTwin/code_painting/visualize_vr_hand_data.py
```

Full run:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/visualize_vr_hand_data.py \
  --overwrite \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/vr_hand_visualization
```

Quick debug:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/visualize_vr_hand_data.py \
  --episodes NTU-PINE_20260708_145454 \
  --max-frames 30 \
  --overwrite \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/vr_hand_visualization_smoke
```

Outputs:

```text
/home/zaijia001/ssd/RoboTwin/code_painting/vr_hand_visualization/<EPISODE>/<EPISODE>_hand_overlay_vscode.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/vr_hand_visualization/vr_data_stats.md
/home/zaijia001/ssd/RoboTwin/code_painting/vr_hand_visualization/vr_data_stats.json
```

## Data Fields

- Main JSON: `format_version=2.0`, `format_type=real_world`, `metadata`, and `frames`.
- Each frame contains `timestamp_seconds`, left/right hand `is_tracked/joint_names/poses`, `center/left/right_eye_pose`, and left/right `validation`.
- Each hand has 26 joints. Each joint pose is `[x, y, z, qx, qy, qz, qw]`; quaternion convention is `xyzw_scalar_last`.
- The metadata JSON records `env_id=NTU-PINE-v1`, Quest 3 device metadata, `coordinate_frame=RUF`, `joint_capture_mode=FullJointPoses`, JPEG camera capture config, and episode success/time/reward/tracking_method.
- `camera_real/*_video.json` records image frame count, FPS, resolution, camera intrinsics for some episodes, and `frame_integrity`.

## Camera Parameter Conclusion

- 13/16 episodes have intrinsics: `focal_length=[640,640]`, `principal_point=[640,640]`, `sensor_resolution=[1280,1280]`.
- 0/16 episodes have `cameras` extrinsics/camera pose records.
- Because extrinsics are missing, the overlay videos are not calibrated projections. The script normalizes episode-level 3D joint `x/z` into image space for diagnostic review only.

## Current Statistics

- Episodes: 16
- Visualization videos: 15; `NTU-PINE_20260703_211357` has no JPG frames, so no video was generated.
- JSON frames: 3772
- JPG frames: 3231
- left/right tracked frame totals: 1998 / 1905
- left/right validation-valid frame totals: 995 / 1160
