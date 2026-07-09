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


## Q.5 20260708 VR-HaMeR Eye/Lag Alignment Sweep

Entrypoint:

```text
code_painting/analyze_vr_hamer_alignment.py
```

Purpose: use only `NTU-PINE_20260708_*` episodes, compare four coordinate candidates (`world_xyz`, `center_eye_xyz`, `left_eye_xyz`, `right_eye_xyz`), and fit each candidate with:

```text
linear_xyz: u_norm,v_norm = linear(x,y,z)
perspective_xy_over_z: u_norm,v_norm = linear(x/z,y/z)
lag: -10..+10, with hamer_frame = vr_frame + lag
```

Command:

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

Outputs:

```text
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_bestfit_20260708/alignment_sweep_20260708.md
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_bestfit_20260708/alignment_sweep_20260708.json
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_bestfit_20260708/videos/id_<ID>_<EPISODE>_bestfit_vr_vs_hamer_vscode.mp4
```

Current result: out of 11 20260708 episodes, 10 produced best-fit comparison videos. `NTU-PINE_20260708_143622` was not fit because it had fewer than 20 matched VR-tracked + HaMeR-detected samples.

Key conclusion:

```text
best pose counts: right_eye_xyz=6, left_eye_xyz=4, center_eye_xyz=0, world_xyz=0
best lag counts: -6=3, -3=3, -5=2, -1=1, +10=1; +10 is from low-sample id5 and should be treated cautiously
best model: 9/10 are linear_xyz, 1/10 is perspective_xy_over_z
```

Interpretation: `left_eye_pose/right_eye_pose` help, but no single eye pose consistently wins. The global weighted metrics for center/left/right are very close, so eye choice is not the main issue. Most best lags are negative; with `hamer_frame = vr_frame + lag`, this means the matching VR hand frame is usually later than the RGB/HaMeR frame by about 3-6 frames, or 100-200 ms at 30 fps. `linear_xyz` beating most perspective fits suggests the images are closer to a Quest user-view/screen-composited space than to an uncropped raw pinhole passthrough camera.

## Q.6 20260708 VR-HaMeR 3D Diagnostic Visualization

Entrypoint:

```text
code_painting/visualize_vr_hamer_3d_diagnostics.py
```

Purpose: use only `NTU-PINE_20260708_*` episodes to explain why the HaMeR 2D overlay follows the hands in the RGB image while the VR hand-joint overlay is offset. The scene is rendered in VR/world RUF coordinates. The script reads VR hand joints, `is_tracked`, `center_eye_pose/left_eye_pose/right_eye_pose`, HaMeR 2D detections, and the best pose/model/lag from Q.5.

Command:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/visualize_vr_hamer_3d_diagnostics.py \
  --episode-substr 20260708 \
  --overwrite \
  --out-dir /home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708
```

Outputs:

```text
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708/summary_3d_20260708.md
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708/summary_3d_20260708.json
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708/id_<ID>_<EPISODE>/high_back_3d_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708/id_<ID>_<EPISODE>/front_3d_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708/id_<ID>_<EPISODE>/top_3d_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708/id_<ID>_<EPISODE>/quadview_3d_vscode.mp4
```

Visualization rules:

- The 3D scene uses VR/world RUF: `X right`, `Y up`, `Z forward`.
- Draw left/right VR hand skeletons, recent trajectories, and `center/left/right_eye_pose` axes plus frustums.
- HaMeR is not treated as true world 3D. It is rendered as image detections and a 2D back-projection ray from the best eye pose.
- The quadview first panel is RGB + HaMeR 2D + bestfit pseudo VR overlay; the other panels are high_back, front, and top 3D views.
- Each frame overlays the episode id, frame, best pose/model/lag, R2/RMSE, and valid sample count.

Current run result:

- Input: 11 `NTU-PINE_20260708_*` episodes.
- Rendered: 10 episodes, each with four VSCode-viewable mp4 files, 40 videos total.
- Skipped: `NTU-PINE_20260708_143622`, because every pose/model/lag configuration had fewer than 20 matched tracked+detected samples, so Q.5 had no reliable bestfit.

Recommended first videos:

```text
id_11_NTU-PINE_20260708_145659/quadview_3d_vscode.mp4
id_9_NTU-PINE_20260708_145601/quadview_3d_vscode.mp4
id_8_NTU-PINE_20260708_145546/quadview_3d_vscode.mp4
id_6_NTU-PINE_20260708_145454/quadview_3d_vscode.mp4
id_14_NTU-PINE_20260708_145845/quadview_3d_vscode.mp4
```

These episodes have relatively reliable Q.5 bestfit metrics and are useful for checking whether the VR world hand trajectory is smooth and whether the HaMeR back-projection ray roughly passes through the VR hand volume. For failure modes, inspect `id_12_NTU-PINE_20260708_145729`, `id_7_NTU-PINE_20260708_145505`, and `id_10_NTU-PINE_20260708_145638`; they are better examples of vertical mismatch, low R2, or likely time/projection issues.

Current conclusion:

- In good episodes, the VR hand trajectory is continuous in world space, so this does not look like a simple axis-swap bug.
- The best pose alternates between left and right eye; center/world do not win. This supports the interpretation that eye poses are only approximate user/render views, not the true `camera_real` JPG extrinsic.
- Most best lags are -3 to -6 frames, indicating an approximately 100-200 ms synchronization offset.
- Most best models are `linear_xyz` rather than pinhole-style perspective, supporting the hypothesis that the RGB is a composited/cropped/scaled/warped Quest user view rather than an uncropped raw camera image.
- Episode-local coarse correction is plausible for id6, id8, id9, id11, and id14; id13 is moderate; id7/id10 are only partially useful; id12 is not trusted; id5 has low samples and +10 lag, so treat it cautiously; id4 is unusable.

## Q.7 20260708 Hand-Local Alignment Validation

Entrypoint:

```text
code_painting/validate_vr_hamer_local_hand_alignment.py
```

Purpose: use only `NTU-PINE_20260708_*` episodes to validate whether VR hand joints and HaMeR 2D keypoints agree in hand-local bone structure and motion trend. This workflow does not require the true raw-camera extrinsic for `camera_real` JPG frames and does not require global world/camera projection alignment.

Command:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/validate_vr_hamer_local_hand_alignment.py \
  --episode-substr 20260708 \
  --overwrite \
  --out-dir /home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708
```

Remove `--overwrite` when only recomputing summaries and reusing existing videos.

Outputs:

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

VR-26 to HaMeR-21 mapping:

```text
wrist -> wrist
thumb_cmc/mcp/ip/tip -> thumb_metacarpal/proximal/distal/tip
index_mcp/pip/dip/tip -> index_proximal/intermediate/distal/tip
middle_mcp/pip/dip/tip -> middle_proximal/intermediate/distal/tip
ring_mcp/pip/dip/tip -> ring_proximal/intermediate/distal/tip
pinky_mcp/pip/dip/tip -> little_proximal/intermediate/distal/tip
```

Ignored VR joints: `palm`, `index_metacarpal`, `middle_metacarpal`, `ring_metacarpal`, and `little_metacarpal`. HaMeR-21 has no matching extra palm/metacarpal points, so this diagnostic compares the visible finger joints.

Local coordinate definition:

- HaMeR: wrist origin; `index_mcp - pinky_mcp` defines local x; the orthogonal component of `middle_mcp - wrist` defines local y; coordinates are palm-scale normalized.
- VR: wrist origin; corresponding VR index/pinky/middle points build a 3D palm basis; points are projected into the local 2D palm plane and palm-scale normalized.
- Each episode independently sweeps `lag=-10..10`; each hand is evaluated separately with similarity/Procrustes and 2D affine alignment; the best side/lag/alignment is selected per episode.

Metrics:

- `rmse_mean`: local 21-keypoint RMSE after the best alignment.
- `per_keypoint_rmse` / `per_finger_rmse`: per-keypoint and per-finger errors in JSON.
- `pairwise_distance_rmse`: pairwise distance-matrix error for normalized local keypoints.
- `bone_length_log_error`: bone-length ratio error.
- `joint_angle_deg_error`: finger joint-angle error.
- `velocity_corr` / `delta_direction_agreement`: consistency of hand-local shape-motion trends.
- `local_shape_score`: combined local shape and motion score from 0 to 100.

Current run result:

- Input: 11 `NTU-PINE_20260708_*` episodes.
- Rendered: 9 episodes, each with five VSCode-readable mp4 files, 45 videos total.
- Skipped: `id4 NTU-PINE_20260708_143622` and `id5 NTU-PINE_20260708_143721`, because every side/lag had fewer than 20 matched VR-tracked + HaMeR-detected samples.
- Classification: all 9 rendered episodes are `medium`; none are `good` or `bad`.

Interpretation:

- `id6/id8/id9/id11/id14`: hand-local medium and previous global bestfit is also relatively reasonable; these are better candidates for episode-local coarse correction.
- `id7/id10/id12/id13`: hand-local medium but previous global/world projection metrics are weak; this points more to recording/user-view composition, cropping, synchronization, or missing extrinsics than to total VR local hand-tracking failure.
- `id4/id5`: insufficient samples, not recommended for hand-local or world-trajectory conclusions.
- Every rendered episode selected affine as the best alignment; similarity summaries are stored under `best_similarity` in JSON. This means local motion trend is strong, but strict rigid/similarity hand shape is only medium quality, so these data should not be treated as precise calibrated 3D hand skeletons.

## Q.8 20260708 Both-Hands v2 Local Alignment Validation

Entrypoint:

```text
code_painting/validate_vr_hamer_local_hand_alignment_bothhands.py
```

Purpose: build on Q.7 but stop selecting only one best side. This v2 workflow separately reports `left_only`, `right_only`, and `both_hands`. It still uses only `NTU-PINE_20260708_*` and does not require a true raw-camera extrinsic or global world/camera projection alignment.

Command:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/validate_vr_hamer_local_hand_alignment_bothhands.py \
  --episode-substr 20260708 \
  --overwrite \
  --out-dir /home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands
```

Outputs:

```text
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands/summary_bothhands_local_alignment_20260708.md
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands/summary_bothhands_local_alignment_20260708.json
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands/summary_bothhands_local_alignment_20260708.csv
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands/id_<ID>_<EPISODE>/left_quadview_local_alignment_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands/id_<ID>_<EPISODE>/right_quadview_local_alignment_vscode.mp4
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands/id_<ID>_<EPISODE>/bothhands_quadview_local_alignment_vscode.mp4
```

Each rendered episode also writes four component videos for each of `left/right/bothhands`: `image_overlay`, `local_skeleton_comparison`, `error_heatmap_timeplot`, and `motion_trend`, plus the quadview.

Mapping hypotheses:

```text
identity: VR_left -> HaMeR_left, VR_right -> HaMeR_right
swapped:  VR_left -> HaMeR_right, VR_right -> HaMeR_left
```

Mirror/canonical variants:

```text
none, mirror_x, mirror_y, mirror_xy
```

The mirror variants are applied to VR hand-local coordinates before similarity/affine alignment to explicitly test possible right-hand MANO or canonicalized-hand conventions in HaMeR. Note that affine alignment can also absorb reflection, so use the JSON `best_similarity` fields when strict local bone shape matters.

Current run result:

- Input: 11 `NTU-PINE_20260708_*` episodes.
- Rendered: 9 episodes, each with 15 VSCode-readable mp4 files, 135 videos total.
- Skipped: `id4 NTU-PINE_20260708_143622` and `id5 NTU-PINE_20260708_143721`, because `left_only/right_only/both_hands` all had insufficient matched samples.
- `left_only`: all 9 rendered episodes are `medium`.
- `right_only`: all 9 rendered episodes are `medium`.
- `both_hands`: all 9 rendered episodes are `medium`.
- Best both-hands mapping: `identity=8`, `swapped=1`. The only swapped case is `id13 NTU-PINE_20260708_145832`.
- Best both-hands mirror: all rendered episodes use `none`; no episode required mirror_x/mirror_y/mirror_xy to win.

Interpretation:

- Both-hands v2 further supports that most 20260708 episodes have side labels closer to identity under a joint two-hand constraint.
- Single-hand `left_only` often independently selects `VR_left -> HaMeR_right`, but most episodes return to identity under the both-hands constraint. This means single-hand local shape alone is not stable enough to infer side labels; the two-hand relation is more reliable.
- `id13` selects swapped under both-hands and has only a medium-low score, so treat it as a side-label/sync anomaly example and do not use it directly for world-coordinate two-hand trajectories.
- All rendered episodes remain `medium`, not `good`; local motion trends are usable, but strict local shape and both-hand relation should not be treated as precisely calibrated 3D hand data.

## Q.9 20260708 Cross-Episode Transform Pattern Aggregation

Entrypoint:

```text
code_painting/analyze_vr_hamer_cross_episode_transform_patterns.py
```

Purpose: use only `NTU-PINE_20260708_*` episodes and aggregate Q.5 bestfit, Q.6 3D diagnostic, Q.7 hand-local, and Q.8 both-hands results to decide whether the VR-hand to HaMeR-image relation is better explained by one global transform, two standing-position cluster transforms, or episode-local transforms. The script does not modify raw data.

Command:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && \
cd /home/zaijia001/ssd/RoboTwin && \
python code_painting/analyze_vr_hamer_cross_episode_transform_patterns.py \
  --episode-substr 20260708 \
  --overwrite \
  --out-dir /home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/cross_episode_transform_patterns_20260708
```

Inputs:

```text
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_bestfit_20260708/alignment_sweep_20260708.json
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/compare_3d_20260708/summary_3d_20260708.json
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708/summary_local_hand_alignment_20260708.json
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/local_hand_alignment_20260708_bothhands/summary_bothhands_local_alignment_20260708.json
/home/zaijia001/ssd/data/piper/vr/data/NTU-PINE_20260708_*/NTU-PINE_20260708_*.json
```

Outputs:

```text
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/cross_episode_transform_patterns_20260708/cross_episode_transform_patterns_20260708.json
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/cross_episode_transform_patterns_20260708/cross_episode_transform_patterns_20260708.md
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/cross_episode_transform_patterns_20260708/episode_transform_table_20260708.csv
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/cross_episode_transform_patterns_20260708/cluster_transform_table_20260708.csv
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/cross_episode_transform_patterns_20260708/transform_parameter_scatter_20260708.png
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/cross_episode_transform_patterns_20260708/cluster_eye_pose_20260708.png
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/cross_episode_transform_patterns_20260708/lag_distribution_20260708.png
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/cross_episode_transform_patterns_20260708/axis_contribution_heatmap_20260708.png
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/cross_episode_transform_patterns_20260708/global_cluster_self_r2_rmse_20260708.png
/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1/cross_episode_transform_patterns_20260708/transform_similarity_distance_heatmap_20260708.png
```

Current result:

- Input: 11 `20260708` episodes. 9 episodes entered cross-episode fitting: `6,7,8,9,10,11,12,13,14`. `id4` has no bestfit; `id5` has only 20 bestfit samples and skipped local/bothhands validation, so it is excluded from global/cluster fitting.
- Best common global config: `right_eye_xyz + linear_xyz`, 2873 samples across 9 episodes. The shared global transform has weighted mean R2 `0.492` and weighted mean RMSE `122.6 px`, so a single global transform is not supported.
- k=2 separates standing/eye-pose clusters, but cluster transforms are only partially useful. Cluster 1 (`6,13,14`) reaches about `67.1 px / 0.832 R2` and can be used as a group-transform reference; cluster 0 (`7,8,9,10,11,12`) is about `136.8 px / 0.417 R2` and remains diagnostic-only.
- Axis pattern: `no_axis_swap_xy=7`, `perspective_xy_over_z=1`, `yz_axis_candidate=1`, `zx_axis_candidate=1`. The data do not support “axis swap + small-angle tuning”; they better support eye-frame `u≈+x`, `v≈-y` plus lag, z/crop/warp terms, and episode-local affine correction.
- Lag: 9 of the 10 bestfit episodes have negative lag, median lag `-4`. With `hamer_frame = vr_frame + lag`, this means the matching VR hand frame is usually later than the RGB/HaMeR frame.
- Both-hands mapping: `identity=8`, `swapped=1`; `id13` is explicitly marked as side-label swapped.
- Recommendation: `id6,id8,id9,id11,id14` are usable for world-coordinate coarse correction; `id7,id10,id13` are reference-only; `id4,id5,id12` should be dropped or kept only as failure examples.
