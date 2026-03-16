# AnyGrasp Orientation Check

## Goal

Check whether the AnyGrasp grasp orientation path inside RoboTwin is missing a fixed orientation conversion compared with the earlier hand-to-gripper replay pipeline.

## Short conclusion

A missing fixed orientation conversion is very likely.

The hand replay path and the AnyGrasp path were not using the same orientation-processing chain:

### Hand replay path
`rotation_cam -> remap_target_rotation() -> camera_to_world_pose() -> apply_target_world_offset() -> align_target_orientation()`

The key point is `remap_target_rotation()`, which applies:
- `stored_orientation_post_rot_matrix`
- `orientation_remap_matrix`

### AnyGrasp path before this update
`rotation_matrix -> camera_to_world_pose()`

That means the AnyGrasp path previously skipped the same kind of fixed local-axis conversion used by hand replay.

## Why this matters

AnyGrasp grasp frames are defined in the grasp detector's own gripper convention. RoboTwin planning, IK, and TCP execution are using the robot's end-effector convention. If those two conventions differ by a fixed local rotation, the grasp position can still look reasonable while the grasp orientation looks consistently wrong.

This matches the failure pattern you described.

## What was added

The planner now exposes a fixed orientation-conversion layer for AnyGrasp candidates:
- `--candidate_orientation_remap_label`
- `--candidate_post_rot_xyz_deg`

These are applied before converting the candidate from camera frame to world frame.

Current candidate conversion path is now:
`rotation_matrix -> candidate_post_rot_xyz_deg -> candidate_orientation_remap_label -> camera_to_world_pose()`

## Practical debug recommendation

### Step 1: Reduce visualization to top1
To look only at the nearest candidate for each arm:
```bash
--debug_candidate_top_k 1 --debug_common_candidate_top_k 0
```

### Step 2: Compare fixed orientation variants
Try a few fixed post-rotations, for example:
```bash
--candidate_post_rot_xyz_deg 0 0 0
--candidate_post_rot_xyz_deg 90 0 0
--candidate_post_rot_xyz_deg -90 0 0
--candidate_post_rot_xyz_deg 0 90 0
--candidate_post_rot_xyz_deg 0 0 90
```

### Step 3: Try remap labels copied from the hand replay utilities
Useful starting points:
```bash
--candidate_orientation_remap_label identity
--candidate_orientation_remap_label swap_red_blue
--candidate_orientation_remap_label swap_red_blue_keep_green
```

## Suggested debug command

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes \
  --ids 1 \
  --keyframes 1 22 \
  --lighting_mode front_no_shadow \
  --planner_backend urdfik \
  --enforce_target_object_constraint 0 \
  --enforce_candidate_distance_constraint 0 \
  --debug_show_all_candidates 1 \
  --debug_common_candidate_top_k 0 \
  --debug_candidate_top_k 1 \
  --candidate_orientation_remap_label identity \
  --candidate_post_rot_xyz_deg 0 0 0 \
  --save_debug_preview 1 \
  --save_debug_execution_preview 1 \
  --enable_viewer 1 \
  --viewer_frame_delay 0.02 \
  --viewer_wait_at_end 1
```

## Additional viewer change

The debug gripper actor now uses the AnyGrasp `width` value to visualize the opening. So the viewer is no longer showing a fixed-width debug gripper; the finger spacing follows the AnyGrasp candidate.

## Current confidence level

High confidence that a fixed orientation conversion layer was missing.

Not yet proven which exact remap/post-rotation is the correct one for your AnyGrasp export. That still needs one round of visual verification in the viewer or videos.
