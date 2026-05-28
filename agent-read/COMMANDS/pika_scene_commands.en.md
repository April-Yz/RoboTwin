# Piper / Pika Scene Commands

## Purpose
Reusable commands for manual tabletop previews, calibrated scenes, and version-B calibrated scenes.

## Commands

- Open the manual tabletop scene in interactive viewer with the oblique camera.
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/visualize_piper_pika_agx_dual_table.py \
  --offscreen-only 0 \
  --camera-mode oblique \
  --output-dir /home/zaijia001/ssd/RoboTwin/code_painting/pika/output_piper_pika_agx_dual_table_manual_viewer
```

- Open the manual tabletop scene in interactive viewer with the top-down camera.
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/visualize_piper_pika_agx_dual_table.py \
  --offscreen-only 0 \
  --camera-mode top_down \
  --output-dir /home/zaijia001/ssd/RoboTwin/code_painting/pika/output_piper_pika_agx_dual_table_manual_topdown_viewer
```

- Re-export the manual tabletop scene after editing `robot_config_PiperPika_agx_dual_table.json` using the oblique camera.
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/visualize_piper_pika_agx_dual_table.py \
  --offscreen-only 1 \
  --camera-mode oblique \
  --output-dir /home/zaijia001/ssd/RoboTwin/code_painting/pika/output_piper_pika_agx_dual_table_manual \
  --image-name piper_pika_agx_dual_table_manual.png \
  --video-name piper_pika_agx_dual_table_manual.mp4
```

- Re-export the manual tabletop scene after editing `robot_config_PiperPika_agx_dual_table.json` using the top-down camera.
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/visualize_piper_pika_agx_dual_table.py \
  --offscreen-only 1 \
  --camera-mode top_down \
  --output-dir /home/zaijia001/ssd/RoboTwin/code_painting/pika/output_piper_pika_agx_dual_table_manual_topdown \
  --image-name piper_pika_agx_dual_table_manual_topdown.png \
  --video-name piper_pika_agx_dual_table_manual_topdown.mp4
```

- Export the calibrated real-scene reconstruction anchored to the current manual tabletop placement.
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene.py
```

- Export the calibrated version-B scene that removes the previous +90 degree anchor rotation.
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py
```

- Open calibrated scene A in interactive viewer (overview view).
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene.py \
  --viewer 1 \
  --viewer-camera overview \
  --save-image 0 \
  --save-video 0 \
  --save-headcam-image 0
```

- Open calibrated scene A in interactive viewer (head-camera view).
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene.py \
  --viewer 1 \
  --viewer-camera head \
  --save-image 0 \
  --save-video 0 \
  --save-headcam-image 0
```

- Export only the calibrated head-cam image (no overview image, no video).
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene.py \
  --save-image 0 \
  --save-video 0 \
  --save-headcam-image 1 \
  --headcam-image-name calibrated_scene_headcam.png
```

- Open calibrated scene version-B in interactive viewer (overview view).
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py \
  --viewer 1 \
  --viewer-camera overview \
  --save-image 0 \
  --save-video 0 \
  --save-headcam-image 0
```

- Open calibrated scene version-B in interactive viewer (head-camera view).
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py \
  --viewer 1 \
  --viewer-camera head \
  --save-image 0 \
  --save-video 0 \
  --save-headcam-image 0
```

- Export only the calibrated version-B head-cam image (no overview image, no video).
```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py \
  --save-image 0 \
  --save-video 0 \
  --save-headcam-image 1 \
  --headcam-image-name calibrated_scene_vb_headcam.png
```

## Calibrated-scene note
- The calibrated scene scripts do use the calibration bundle for the second robot and head camera.
- In the current overview render, the calibrated head camera is shown only as a small gray marker with RGB axes, so it may be hard to notice.
- The calibrated scene scripts are offscreen renderers only right now; they do not open an interactive viewer window yet.

## Viewer-related note
- Currently available interactive viewer commands:
  - `visualize_piper_pika_agx_dual_table.py --offscreen-only 0 --camera-mode oblique`
  - `visualize_piper_pika_agx_dual_table.py --offscreen-only 0 --camera-mode top_down`
  - `visualize_calibrated_piper_pika_scene.py --viewer 1 --viewer-camera overview|head`
  - `visualize_calibrated_piper_pika_scene_vb.py --viewer 1 --viewer-camera overview|head`
- Note: viewer commands are for interactive inspection; wrist-view image export is still not implemented.

## Wrist-view status
- Left/right wrist image export is not implemented yet in the two calibrated-scene scripts.
- Currently exported views are overview + head-cam.

## Piper Local-Axis Sweep

- Generate a local `x/y/z` axis sweep board for the Piper gripper.
```bash
GPU=1 FRAME_IDX=0 ARM=left EXECUTE=0 SAVE_WRIST_VIEWS=0 \
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_local_axis_sweep_board.sh \
  /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_local_axis_sweep_board
```

- Purpose:
  - Keep the calibrated `PiperPika` dual-arm scene and current head-camera calibration fixed
  - Enumerate every valid right-handed local-axis remap for one arm on one frame
  - Save `zed/third` images with RGB axes and annotate each candidate with how local `x/y/z` align with robot `forward/left/up`

- Key outputs:
  - `per_case/*.png`
  - `board_zed.png`
  - `board_third.png`
  - `summary.json`
  - `summary.csv`

- Useful environment variables:
  - `FRAME_IDX`: frame to inspect
  - `FRAME_END` / `MAX_FRAMES` / `FRAME_STRIDE`: multi-frame video range controls; `MAX_FRAMES=-1` means no truncation
  - `ARM`: `left` or `right`
  - `EXECUTE`: `0` for pure axis semantics, `1` to also run IK execution and record execution error
  - `CANDIDATE_MODE`: `remap` scans 24 right-handed remaps; `semantic` scans `forward_from_*` and `open_from_*` semantic candidates
  - `VIDEO_MODE`: set to `1` to write `board_all_zed.mp4` and `board_success_zed.mp4`
  - `SAVE_WRIST_VIEWS`: set to `1` to export extra wrist-view sweep boards

- Generate a multi-frame remap board video for id0.
```bash
GPU=3 FRAME_IDX=0 FRAME_END=-1 MAX_FRAMES=32 ARM=left EXECUTE=1 \
CANDIDATE_MODE=remap VIDEO_MODE=1 FPS=5 SAVE_WRIST_VIEWS=0 \
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_local_axis_sweep_board.sh \
  /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_local_axis_sweep_video_remap
```

- Generate a smaller semantic-candidate video.
```bash
GPU=3 FRAME_IDX=0 FRAME_END=-1 MAX_FRAMES=32 ARM=left EXECUTE=1 \
CANDIDATE_MODE=semantic VIDEO_MODE=1 FPS=5 SAVE_WRIST_VIEWS=0 \
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_local_axis_sweep_board.sh \
  /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_local_axis_sweep_video_semantic
```

- Multi-frame outputs:
  - `board_all_zed.mp4`: board video containing all candidates
  - `board_success_zed.mp4`: compact board video containing only success candidates
  - `board_frames/frame_XXXX_all_zed.png`
  - `board_frames/frame_XXXX_success_zed.png`
