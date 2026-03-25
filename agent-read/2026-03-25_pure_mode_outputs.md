# 2026-03-25 Pure-Mode Output Notes

## Background

This round updates the `pure_scene_output=1` workflow of `plan_anygrasp_keyframes_r1.py` / `run_plan_anygrasp_keyframes_r1_batch.sh` so that pure mode keeps the main video clean while still saving the synchronized videos and state logs needed for downstream processing.

## Current pure-mode behavior

When the command includes:

```bash
--pure_scene_output 1
```

the current behavior is:

- Main output video stays clean:
  - `head_cam_plan.mp4` has no text overlay
  - candidate grippers are hidden
  - target axes are hidden
- `debug_selection_preview.mp4` is no longer generated
- `debug_execution_preview.mp4` is still controlled by its own flag
- Wrist-camera plan videos are also saved:
  - `left_wrist_cam_plan.mp4`
  - `right_wrist_cam_plan.mp4`
- `pose_debug.jsonl` is automatically saved even if the command does not explicitly pass `--save_pose_debug 1`

## Main output files

In a typical pure-mode output directory, the relevant files are:

- `head_cam_plan.mp4`
  - head-camera planning video
- `left_wrist_cam_plan.mp4`
  - left wrist-camera planning video
  - planner export now applies a 90-degree CCW in-plane correction so the saved wrist videos match the expected viewing orientation
- `right_wrist_cam_plan.mp4`
  - right wrist-camera planning video
  - planner export now applies a 90-degree CCW in-plane correction so the saved wrist videos match the expected viewing orientation
  - wrist-video dimensions now follow the rotated frame size
- `pose_debug.jsonl`
  - one JSON record per saved planning frame, meant for post-processing and video alignment
- `plan_summary.json`
  - summary of arguments, selected candidates, stage results, and output paths

If the command still keeps:

```bash
--save_debug_execution_preview 1
```

then the following are also produced:

- `debug_execution_preview.mp4`
- `debug_execution_metrics.jsonl`

## `pose_debug.jsonl` schema

`pose_debug.jsonl` is a JSON Lines file. Each line corresponds to one `record_frame(...)` call that writes a planning frame. Current fields include:

- `record_index`
  - monotonically increasing planner-record index starting from 0
- `active_frame`
  - current replay/object frame index; may be `null` when no keyframe is active
- `stage`
  - current stage such as `init`, `pregrasp`, `grasp`, `close_gripper`, or `action`
- `overlay_lines`
  - the text lines that would normally be used for overlay; in pure mode the video is clean, but the logical frame annotation is still preserved here

### Camera poses

- `current_head_camera_pose_world_wxyz`
- `current_left_wrist_camera_pose_world_wxyz`
- `current_right_wrist_camera_pose_world_wxyz`
- `replay_head_camera_pose_world_wxyz`

All camera poses use the world-frame 7D format:

```text
[x, y, z, qw, qx, qy, qz]
```

### Robot end-effector poses

- `current_left_tcp_pose_world_wxyz`
- `current_right_tcp_pose_world_wxyz`
- `current_left_ee_pose_world_wxyz`
- `current_right_ee_pose_world_wxyz`

Notes:

- `tcp` follows the planner / gripper-reference convention
- `ee` follows the wrist/endlink convention
- In the current implementation:
  - `current_*_tcp_pose_world_wxyz` already comes from planner-side 7D pose arrays
  - `current_*_ee_pose_world_wxyz` comes from `robot.get_*_ee_pose()`, which returns a 7D list rather than a `sapien.Pose`
  - the export layer now accepts both representations and writes them uniformly as `[x, y, z, qw, qx, qy, qz]`

### Robot joint states

- `current_left_arm_qpos_rad`
- `current_right_arm_qpos_rad`
  - 6D arm joint positions in radians
- `current_left_gripper_joint_qpos_rad`
- `current_right_gripper_joint_qpos_rad`
  - actual finger-joint qpos values used by the gripper contact logic
- `current_left_gripper_command`
- `current_right_gripper_command`
  - current scalar gripper command values from the robot state

### Object states

- `object_actor_poses`
  - dictionary keyed by object name, e.g. `cup` / `bottle`
  - each object always contains:
    - `actor_pose_world_wxyz`
  - if the current `active_frame` aligns with replay object tracks, it also contains:
    - `replay_pose_world_wxyz`

### Extra debug data

- `frame_metrics`
  - a per-frame snapshot of target/current/object relative metrics for the active stage

## Code locations

- Main logic:
  - `code_painting/plan_anygrasp_keyframes_r1.py`
- Pure-mode skip for selection preview:
  - `generate_debug_preview(...)`
- Head / wrist video writing:
  - `record_frame(...)`
- `pose_debug.jsonl` writing:
  - `record_frame(...)`
- Output-path summary:
  - `plan_summary.json` construction block

## Recommended usage

If you want “clean main video + wrist videos + synchronized state logs”, the recommended option set is:

```bash
--pure_scene_output 1 \
--overlay_text 0 \
--debug_visualize_targets 0 \
--save_debug_execution_preview 0
```

Notes:

- `pure_scene_output=1` now auto-enables `pose_debug.jsonl`
- if you still want the execution debug video, switch `--save_debug_execution_preview` back to `1`

## Suggested downstream alignment

For later processing, the most useful alignment keys are:

- `record_index`
- `stage`
- `active_frame`

Use them jointly with:

- `head_cam_plan.mp4`
- `left_wrist_cam_plan.mp4`
- `right_wrist_cam_plan.mp4`
- `pose_debug.jsonl`

This gives a synchronized view of:

- video frames
- TCP/EE trajectories
- arm joint positions
- gripper motion
- object poses
