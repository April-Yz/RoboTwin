## 2026-03-27 00:00:00 +08

- Added a new pi0 data-conversion command: `process_repainted_headcam_with_wrist.py`
  - Entry point:
    - `policy/pi0/scripts/process_repainted_headcam_with_wrist.py`
  - Purpose:
    - convert the newer head-cam repaint outputs plus retarget wrist replays into `processed_data` HDF5 episodes
  - Key arguments:
    - `--head-root`
    - `--head-dir-template`
    - `--head-video-name`
    - `--retarget-root`
    - `--retarget-dir-template`
    - `--ids`
    - `--ignore-ids`
  - Typical command:
    - `python scripts/process_repainted_headcam_with_wrist.py d_pour_blue "pour water" 48 --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' --head-video-name target_with_original_head_cam_plan.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue --retarget-dir-template 'hand_detections_{id}' --ignore-ids`

## 2026-03-25 19:15:00 +08

- Added visual-only base-occluder parameters
  - Flags:
    - `--base_occluder_enable 0|1`
    - `--base_occluder_local_pos X Y Z`
    - `--base_occluder_half_size HX HY HZ`
    - `--base_occluder_color R G B`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/render_hand_retarget_r1_npz.py`
  - Purpose:
    - add a white occluder that follows the robot base and hides the chassis/base in videos
    - the current implementation is visual-only and creates no collision
  - Usage notes:
    - useful for cleaning up pure/debug videos
    - `local_pos` controls height and local offset
    - `half_size` controls the occluder box dimensions

## 2026-03-25 18:55:00 +08

- R1 planner wrist-camera export semantics updated
  - Behavior:
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - no longer depend on post-export image rotation
    - instead use an R1-specific wrist local pose inside `plan_anygrasp_keyframes_r1.py` that matches `galaxea_sim/robots/r1.py`
    - output size returns to the original landscape `image_width x image_height`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Notes:
    - this override is specific to the R1 planner path and does not change the global default in `render_hand_retarget_r1_npz.py`
    - the goal is to obtain the correct wrist view from the mounted camera pose itself rather than from an export-time image rotation patch

## 2026-03-25 18:35:00 +08

- Planner wrist-video export fine-tuned again
  - Behavior:
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - now use a uniform `90°` CCW rotation before export
    - writer dimensions match the rotated frames
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Notes:
    - this change is based on the user's actual result: the previous `180°` fix still looked like the correct view rotated `90°` CCW
    - no new CLI parameter was added

## 2026-03-25 18:20:00 +08

- Planner wrist-video export corrected again
  - Behavior:
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - no longer use a `90°` rotation; they now use a uniform `180°` in-plane rotation before export
    - output size returns to landscape `image_width x image_height`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Notes:
    - no new CLI parameter was added
    - this correction is based on the user's actual exported result: the previous version produced portrait videos that were still upside down
    - the new fix only changes image-plane export orientation, not the camera mount or planner coordinate conventions

## 2026-03-25 16:45:00 +08

- `--debug_visualize_ik_waypoints 1`
  - Visualization enhancement:
    - now shows start and goal markers in addition to intermediate TCP waypoints
    - both start and goal use red point+forward-axis markers
    - intermediate waypoint markers are smaller so the hands, target axes, and path are easier to inspect together
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Usage:
    - command syntax is unchanged; keep using `--debug_visualize_ik_waypoints 1`
    - this flag only affects viewer/debug visualization and does not change planning or execution logic

## 2026-03-25 17:10:00 +08

- Planner wrist-video orientation correction
  - Behavior:
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - are now rotated 90 degrees clockwise before being written
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Notes:
    - no new CLI parameter was added
    - this only affects planner wrist-video orientation
    - it does not change world camera pose or planner coordinate definitions

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

- Added command parameter: `--debug_collision_report`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Print stronger collision debug information during the progressive `close_gripper` stage.
  - Main output:
    - `[collision-debug-init]`
    - `[collision-debug-step]`
    - regular `[gripper-close]` now also includes `base_contact=...`
  - Debug focus:
    - distinguish `finger_contact` from `base_contact`
    - inspect `finger_pairs` / `base_pairs`
    - inspect collision-shape summaries of the target object, `left/right_gripper_link`, and finger links
  - Default:
    - `0`

- Added command parameter: `--execution_object_collision_mode {convex,solid_bbox}`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Control which collision geometry execution objects use at runtime.
  - Modes:
    - `convex`
      - keeps the current `add_convex_collision_from_file`
    - `solid_bbox`
      - reads mesh bounds
      - creates one axis-aligned solid box collision
  - Notes:
    - affects execution collision only, not the visual mesh
    - if `--replay_objects_ignore_collision 1` and the object is not enabled for grasp/action collision, collision is still omitted
  - Default:
    - `convex`

- Added command parameter: `--gripper_contact_monitor_mode {fingers,fingers_and_base,all_robot_links}`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Control which robot links are allowed to trigger contact monitoring during `close_gripper`.
  - Modes:
    - `fingers`
      - finger links only
    - `fingers_and_base`
      - finger links plus `left/right_gripper_link`
    - `all_robot_links`
      - all links of the articulation corresponding to the current arm
  - Notes:
    - this changes the monitoring set used by the early-stop logic, not just the printed debug output
    - it is especially useful for debugging cases where finger/base links report `shapes=0` but other links may still carry collision

## 2026-03-25 23:10:00 +08

- Added a minimal collision-probe command:
  - Entry script:
    - `code_painting/minimal_gripper_collision_probe.py`
  - Purpose:
    - verify raw scene contacts between the R1 gripper and either a simple box or a mesh (`solid_bbox` / `convex`) object without going through the AnyGrasp/IK/stage main pipeline
  - Representative commands:
    - box:
      - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/minimal_gripper_collision_probe.py --arm left --object_kind box --probe_local_offset 0.04 0.0 0.0 --max_iters 20 --settle_steps_per_iter 8`
    - mesh:
      - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/minimal_gripper_collision_probe.py --arm left --object_kind mesh --mesh_path /home/zaijia001/ssd/data/R1/hand/obj_mesh/blue_cup/blue_cup.obj --mesh_collision_mode solid_bbox --probe_local_offset 0.04 0.0 0.0 --max_iters 20 --settle_steps_per_iter 8`
  - Key outputs:
    - step-by-step terminal logs of:
      - `qpos`
      - `raw_contact_total`
      - `target_contact_total`
      - `target_contacts`
    - JSON outputs:
      - `code_painting/minimal_gripper_collision_probe/probe_left_box.json`
      - `code_painting/minimal_gripper_collision_probe/probe_left_mesh.json`

- Enhanced `--debug_collision_report 1` output:
  - added raw target-contact reporting during the close stage:
    - `raw_target_contacts`
    - `raw_target_contact_total`
    - `[gripper-close] ... raw_target_contact=0|1`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Purpose:
    - distinguish “the monitor/helper failed to match a contact” from “raw physics contact does not exist at all”

- Further enhanced `--debug_collision_report 1` output:
  - Added:
    - `target_pose=...`
    - `target_collision_debug=...`
  - Purpose:
    - directly inspect whether the target-object actor pose stays stable during close
    - directly inspect the `solid_bbox` `center/half_size`

- Added flag: `--debug_visualize_object_collision_bbox 0|1`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - when `execution_object_collision_mode=solid_bbox`, additionally display the object's collision bbox
  - Typical usage:
    - append to the existing command:
      - `--debug_visualize_object_collision_bbox 1`

- Added flag: `--grasp_action_object_collision_start_stage {close_gripper,grasp,pregrasp}`
  - Purpose:
    - control from which stage selected execution objects start participating in collision
  - Typical experiment:
    - `--enable_grasp_action_object_collision 1 --grasp_action_object_collision_start_stage pregrasp --execution_object_collision_mode convex`

- Added pre-close pose export:
  - Output files:
    - `close_stage_snapshot_dual_before_close.json`
    - `close_stage_snapshot_<arm>_before_close.json`

- Added flag: `--execution_object_scale_override NAME=S|SX,SY,SZ`
  - Purpose:
    - independently scale an execution object's visual mesh and collision geometry together
  - Typical examples:
    - `--execution_object_scale_override cup=0.9`
    - `--execution_object_scale_override bottle=0.9`

- Added flags:
  - `--execution_object_visual_scale_override NAME=S|SX,SY,SZ`
  - `--execution_object_collision_scale_override NAME=S|SX,SY,SZ`
  - Purpose: independently control execution-object visual-mesh scale and collision-shape scale
