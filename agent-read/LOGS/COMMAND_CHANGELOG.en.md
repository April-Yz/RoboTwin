## 2026-03-25 12:08:00 +08

- Added flag: `--enable_grasp_action_object_collision 0|1`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/run_plan_anygrasp_keyframes_r1_batch.sh`
  - Purpose:
    - enable collision blocking for the selected execution object during `close_gripper` and `action`
    - default `0` preserves the previous no-collision execution mode
  - Usage:
    - single-run command: append `--enable_grasp_action_object_collision 1`
    - batch: append `--enable_grasp_action_object_collision 1` to the batch command as well
  - Notes:
    - this flag does not change target-pose construction for `pregrasp/grasp/action`
    - it also does not change the relative transform used when attaching the object to TCP
## 2026-03-25 13:05:00 +08

- Added visualization-mode flags to `plan_anygrasp_keyframes_r1.py`:
  - New flags:
    - `--debug_visualize_targets 0|1`
    - `--viewer_show_camera_frustums 0|1`
  - Usage:
    - `debug_visualize_targets=0` disables target-axis actors globally
    - `viewer_show_camera_frustums=0` disables SAPIEN camera frustum lines in the interactive viewer
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`

- Synced the same visualization-mode flags through `plan_anygrasp_keyframes_r1_batch.py`:
  - Forwarded flags:
    - `--debug_visualize_targets`
    - `--viewer_show_camera_frustums`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-25 13:35:00 +08

- `--enable_grasp_action_object_collision 1`
  - Behavior enhancement:
    - The `close_gripper` stage no longer always closes fully in one shot
    - It now closes progressively and stops early once contact with the selected object is present and gripper joint motion has stalled
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`

## 2026-03-25 14:10:00 +08

- Added parameter: `--urdfik_cartesian_interp_auto_step_m`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Only active when `--urdfik_cartesian_interp_steps=-1`, controlling the translation threshold of automatic waypoint mode.
  - Default:
    - `0.05`
  - Example:
    - `--urdfik_cartesian_interp_steps -1 --urdfik_cartesian_interp_auto_step_m 0.03`
  - Note:
    - Smaller values create denser intermediate waypoints.
  - Usage:
    - Command syntax is unchanged; keep using `--enable_grasp_action_object_collision 1`
    - With the default `0`, the original fast no-collision closing behavior is preserved

## 2026-03-25 14:05:00 +08

- `--urdfik_cartesian_interp_steps`
  - New convention:
    - `-1` enables automatic waypoint mode
  - Automatic rule:
    - no intermediate waypoint when translation is `<= 0.05m`
    - add one intermediate waypoint for each additional `0.05m` bucket beyond that threshold
  - Example:
    - `--urdfik_cartesian_interp_steps -1`
  - Related code:
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`

## 2026-03-25 14:25:00 +08

- `planner_backend=urdfik` + `urdfik_trajectory_mode=cartesian_interp_ik`
  - Behavior fix:
    - the execution layer now truly consumes the full `joint_waypoints` stored in `plan["position"]`
    - it no longer executes only a straight endpoint interpolation from `current_joints` to `target_joints`
  - Related code:
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`
# 2026-03-25

- `--pure_scene_output 1`
  - Behavior update:
    - no longer generates `debug_selection_preview.mp4`
    - now keeps and writes:
      - `head_cam_plan.mp4`
      - `left_wrist_cam_plan.mp4`
      - `right_wrist_cam_plan.mp4`
    - auto-enables `pose_debug.jsonl`
  - Main `pose_debug.jsonl` fields:
    - `record_index`
    - `stage`
    - `active_frame`
    - `current_*_camera_pose_world_wxyz`
    - `current_*_tcp_pose_world_wxyz`
    - `current_*_ee_pose_world_wxyz`
    - `current_*_arm_qpos_rad`
    - `current_*_gripper_joint_qpos_rad`
    - `object_actor_poses`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Reference docs:
    - `agent-read/2026-03-25_pure_mode_outputs_ZH.md`
    - `agent-read/2026-03-25_pure_mode_outputs.md`

- Added command parameter: `--debug_visualize_ik_waypoints`
  - Entry points:
    - `/home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `/home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Show the intermediate TCP/EE smoothing waypoints of `cartesian_interp_ik` in debug/viewer output, helping distinguish “bad waypoint generation” from “bad IK/execution follow-through”.
  - Display:
    - Position point plus local forward-axis marker for each intermediate waypoint.
  - Default:
    - `0`
