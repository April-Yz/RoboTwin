# R1 Camera Debug Notes

## Summary

This note records the camera-pose bug found while replaying hand retargeting in `render_hand_retarget_r1_npz.py`.

The main conclusion is:

- `rx=0, ry=0, rz=0` is now the correct extra debug rotation for the head camera.
- This does **not** mean "no camera conversion is needed".
- It means the script now includes the fixed mounted-camera local rotation that was previously missing.

## Root Cause

The original replay script treated `zed_link` as if it were already the actual head camera pose.

That assumption was wrong.

In `RoboTwin2_repainting`, the real head camera is mounted on `zed_link` with an additional local pose:

- Source: [RoboTwin2_repainting/galaxea_sim/robots/r1_pro.py](/home/zaijia001/RoboTwin2_repainting/galaxea_sim/robots/r1_pro.py)
- Definition:
  - `mount = zed_link`
  - `local_pose = sapien.Pose([0, 0, 0], [1, 1, -1, 1] / 2)`

That quaternion is in `wxyz` order.

So the correct head camera world pose is:

`world_T_head_camera = world_T_zed_link * local_T_head_camera`

Previously the script used only:

`world_T_head_camera = world_T_zed_link`

That missing fixed rotation was the main reason the head camera kept looking in the wrong direction and required fake `yaw_p90 / yaw_n90` compensation.

## Why `rx=ry=rz=0` Works Now

After the mounted-camera local quaternion was added back, the default camera pose became consistent with the simulator definition.

So:

- old behavior:
  - missing fixed local rotation
  - needed manual debug rotation to "patch" the view
- current behavior:
  - fixed local rotation is already included
  - the extra debug rotation should usually stay at `0, 0, 0`

So `000` is not "the raw zed_link happened to be correct".
It is "the script now matches the real mounted camera definition, so no extra patch rotation is needed".

## Camera Conventions Used

There are two independent transforms to keep separate:

1. Mounted camera local pose
   - Example for head camera: `zed_link -> head_camera`
   - Example for wrist cameras: `left/right_realsense_link -> wrist_camera`

2. Camera coordinate convention conversion
   - Used when converting hand detections from CV camera coordinates into world coordinates
   - Current replay default uses `camera_cv_axis_mode=legacy_r1`

These two transforms solve different problems and must not be mixed together.

## Wrist Camera Definition

In `RoboTwin2_repainting`, the wrist cameras are also mounted with a local rotation:

- Source: [RoboTwin2_repainting/galaxea_sim/robots/r1_pro.py](/home/zaijia001/RoboTwin2_repainting/galaxea_sim/robots/r1_pro.py)
- Base local quaternion: `[0.5, 0.5, -0.5, 0.5]` in `wxyz`
- Extra RPY offset: `[-10 deg, 0, -90 deg]`

So wrist camera debugging should start from this mounted pose, not from the raw `left_realsense_link` / `right_realsense_link` pose.

## Third View Note

An earlier version of `third_view` was placed using robot-base forward plus a left offset, which made it look more like a side camera near the robot's left side.

It has been changed to be roughly opposite to the current head camera forward direction instead.

## Useful Commands

Head camera single-frame check:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_camera_single_zero \
  5 \
  --frame_start 0 \
  --frame_end 0 \
  --max_frames 1 \
  --save_png_frames 1 \
  --debug_mode 1 \
  --debug_frame_limit 1 \
  --camera_cv_axis_mode legacy_r1 \
  --head_camera_local_quat_wxyz 1 1 -1 1
```

Wrist camera sweep:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_left_wrist_sweep \
  5 \
  --camera_sweep_enable 1 \
  --camera_debug_target left_wrist \
  --camera_sweep_steps_deg -90 0 90 \
  --debug_mode 1 \
  --debug_frame_limit 1
```

## Current Status

- Head camera:
  - mounted local pose bug identified
  - default `000` extra debug rotation is now plausible
- Third view:
  - previously too lateral
  - now adjusted to be opposite to head forward
- Wrist camera:
  - mounted local pose has been added
  - orientation tuning/debug is the next target
