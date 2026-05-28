# Piper FoundationPose Replay Commands

## Purpose

Record reusable RoboTwin replay commands for Piper H2O FoundationPose outputs.

## Applicable Version

- Current recommended calibration: the 0515 head D435 / base calibration represented by `calibration_bundle_piper_new_table_0515.json`.
- Current command library location: C1 / C1.2 in `/home/zaijia001/ssd/RoboTwin/COMMAND_LIBRARY.zh.md`.

## D435 Intrinsics Replay

C1.2 adds D435-intrinsics FoundationPose replay commands for six H2O tasks. It reuses the C1 FoundationPose inputs, object meshes, Piper head extrinsics, and `--camera_cv_axis_mode legacy_r1`, but changes the render camera to match the later robot pure replay commands in E2.4:

```text
--image_width 640 --image_height 480 --fovy_deg 42.499880046655484
```

The output directory convention is:

```text
/home/zaijia001/ssd/data/piper/hand/<TASK>/foundation_replay_d435
```

## Covered Tasks

- `pick_diverse_bottles`
- `place_bread_basket`
- `stack_cups`
- `handover_bottle`
- `pnp_bread`
- `pnp_tray`

## Related Code

- Batch wrapper: `/home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh`
- Batch Python entry: `/home/zaijia001/ssd/RoboTwin/code_painting/render_multi_object_pose_r1_npz_batch.py`
- Single replay entry: `/home/zaijia001/ssd/RoboTwin/code_painting/render_multi_object_pose_r1_npz.py`

## Notes

- C1.2 only changes render intrinsics and output directories. It does not change FoundationPose inputs, meshes, or head extrinsics.
- The D435 FOV is inherited from the E2.4 human RGB camera_info note: `fovy = 2 * atan(height / (2 * fy))`.
