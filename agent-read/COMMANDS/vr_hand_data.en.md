# VR Hand Data Commands

## Purpose

Document the structure, statistics, and diagnostic visualization commands for the Meta Quest VR hand-tracking data (local metadata says Quest 3; the user currently describes the device as Quest 2, so the capture-side metadata should be checked later) under `/home/zaijia001/ssd/data/piper/vr`.

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


## Q.2 Axis-Projection Debug and Result Paths

Copied baseline visualization result:

```text
/home/zaijia001/ssd/data/piper/vr/0vis/datav1/<EPISODE>/<EPISODE>_hand_overlay_vscode.mp4
```

`visualize_vr_hand_data.py` supports `--projection-mode norm_xz|norm_xy|norm_yz|norm_zx|eye_center`. The `norm_*` modes are episode-normalized axis diagnostics. `eye_center` uses `center_eye_pose + intrinsics` as an approximation. Neither is a calibrated external-camera projection.

Generate four debug projections:

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

Output directory:

```text
/home/zaijia001/ssd/data/piper/vr/0vis/datav1_axis_debug/<MODE>/<EPISODE>/
```

## Q.3 Reusing HaMeR on VR JPG Frames

Entrypoints:

```text
code_painting/prepare_vr_hamer_input.py
code_painting/compare_vr_hamer_results.py
```

Convert VR episodes to HaMeR flat input:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/prepare_vr_hamer_input.py \
  --overwrite \
  --output-root /home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1
```

Run HaMeR detection:

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

Generate comparison statistics and side-by-side videos:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/compare_vr_hamer_results.py \
  --overwrite \
  --hamer-root /home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1 \
  --vr-vis-root /home/zaijia001/ssd/data/piper/vr/0vis/datav1
```

Outputs:

```text
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/input/rgb_<ID>.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/input/params_<ID>.json
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/output/hand_vis_gripper_<ID>.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/output/hand_detections_<ID>.npz
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare/id_<ID>_<EPISODE>_vr_vs_hamer_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare/compare_vr_hamer_stats.md
```

## Q.4 Coordinate-System and Camera-Parameter Conclusions

- The 26 hand joints in the VR JSON are in the VR tracking/world frame; metadata records `coordinate_frame=RUF`.
- HaMeR detects hands directly from RGB pixels; `hand_vis_gripper_*.mp4` is image-space overlay and is better for checking visual alignment.
- Current `camera_real/*_video.json` files have no `cameras` extrinsics/camera pose. Only some episodes have `camera_intrinsics`.
- `center_eye_pose/left_eye_pose/right_eye_pose` are eye/head-related poses and are not the external RGB camera pose for the `camera_real` JPG frames.
- Therefore the offset in the VR joint overlay is not something a simple axis swap can fully fix; the external camera-to-VR-world/headset extrinsic is missing.
- Meta's official PCA docs indicate the proper capture path is to save the intrinsics, extrinsics, timestamp, or camera pose returned by `PassthroughCameraAccess`. Official PCA currently targets Quest 3/Quest 3S; if the real device is Quest 2, do not assume there is a fixed official intrinsic table matching these JPG frames.
