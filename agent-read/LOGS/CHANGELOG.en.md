# CHANGELOG.en

## 2026-03-25

- Added a visual-only base occluder to hide the chassis in rendered videos:
  - Files:
    - `code_painting/render_hand_retarget_r1_npz.py`
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/render_object_pose_r1_npz.py`
  - Motivation:
    - the robot chassis/base is often visible in head and wrist videos and hurts the composition
    - the user wanted a configurable occluder with custom height/size that does not participate in collision
  - Implementation:
    - added a visual-only `base_occluder` actor with no collision shapes
    - the occluder follows the robot base pose
    - exposed CLI flags:
      - `--base_occluder_enable`
      - `--base_occluder_local_pos X Y Z`
      - `--base_occluder_half_size HX HY HZ`
      - `--base_occluder_color R G B`
  - Semantics:
    - `local_pos` is defined in the robot-base frame, so it rotates with the robot heading
    - `half_size` uses the SAPIEN box half-size convention
    - the occluder is visual-only and does not affect collision, IK obstacles, or grasp contact
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_r1_npz.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py /home/zaijia001/ssd/RoboTwin/code_painting/render_object_pose_r1_npz.py`

- Corrected the R1 planner wrist-camera mount definition and removed post-export image rotation:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/CAMERA_DEBUG_NOTES_R1.md`
  - Root cause:
    - the R1 planner path had been using a wrist local pose closer to `galaxea_sim/robots/r1_pro.py`
    - but `galaxea_sim/robots/r1.py` only includes `rx=-10°` and does not include the extra local `z=-90°`
    - that is why wrist exports kept needing `90°/180°` image-plane patches and still looked geometrically unnatural
  - Fix:
    - override the wrist local quaternion specifically inside `plan_anygrasp_keyframes_r1.py` so the R1 planner matches `galaxea_sim/robots/r1.py`
    - `rotate_wrist_rgb_for_export(...)` now becomes a pass-through; planner wrist exports no longer rotate the saved image afterward
    - wrist-writer size returns to the original landscape `(image_width, image_height)`
  - Impact:
    - the wrist view now comes from the actual mounted camera pose rather than an export-time image-plane rotation
    - no change to the global default in `render_hand_retarget_r1_npz.py`, so R1 Pro-related paths are not affected
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Corrected planner wrist-video export orientation again:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - User feedback:
    - after the previous `180°` correction, the exported result still looked like the correct view rotated 90 degrees CCW
  - Fix:
    - changed `rotate_wrist_rgb_for_export(...)` from `cv2.ROTATE_180` to `cv2.ROTATE_90_COUNTERCLOCKWISE`
    - changed planner wrist-writer size back to the rotated-frame size `(image_height, image_width)`
  - Current behavior:
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` are exported with a `90°` CCW image-plane rotation
    - output dimensions match the rotated frames
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Corrected planner wrist-video export orientation and frame size:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - the previous round incorrectly treated the wrist videos as needing a `90°` rotation
    - the exported files became portrait, and the user confirmed the image was still upside down
  - Fix:
    - changed `rotate_wrist_rgb_for_export(...)` from `cv2.ROTATE_90_CLOCKWISE` to `cv2.ROTATE_180`
    - changed planner wrist-writer size back from portrait `(image_height, image_width)` to landscape `(image_width, image_height)`
  - Current behavior:
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` stay in landscape format
    - export now applies only an in-plane `180°` correction
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed the orientation of planner-exported wrist videos:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` appeared 90 degrees CCW relative to the expected viewing orientation
  - Investigation result:
    - after comparing `render_hand_retarget_r1_npz.py` and `galaxea_sim/robots/r1_pro.py`, the R1 and R1 Pro wrist-camera mount poses are effectively the same:
      - base quaternion `[0.5, 0.5, -0.5, 0.5]`
      - extra RPY offset `[-10deg, 0, -90deg]`
    - so this round does not change the mounted camera definition; it corrects the exported image plane instead
  - Fix:
    - added `rotate_wrist_rgb_for_export(...)`
    - apply `cv2.ROTATE_90_CLOCKWISE` before writing left/right wrist videos in `record_frame(...)`
    - rotation now happens before overlay/BGR conversion so debug text stays upright as well
  - Impact:
    - no change to world camera pose, planner targets, candidate coordinate conversion, or head videos
    - this only corrects planner wrist-video export orientation
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed the post-rotation writer-size mismatch for wrist videos:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - the previous round rotated planner wrist videos by `90° clockwise`
    - but the `cv2.VideoWriter` instances were still opened with the original `(image_width, image_height)` = `640x360`
    - the rotated frames are actually `360x640`
    - as a result the writer created tiny placeholder mp4 files (around `258B`) without valid video frames
  - Fix:
    - planner wrist writers now use the rotated frame size `(image_height, image_width)`
  - Impact:
    - `head_cam_plan.mp4` keeps its original landscape size
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` now use portrait dimensions matching the rotated wrist frames
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Adjusted URDF-IK waypoint visualization and documented the stage-settling parameters:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Visualization changes:
    - with `--debug_visualize_ik_waypoints 1`, the viewer now shows start and goal markers in addition to intermediate waypoints
    - both start and goal use red point+forward-axis markers
    - intermediate waypoint markers are now smaller to reduce clutter around the hands and targets
  - Analysis notes:
    - `init` still only applies `apply_robot_init_pose(...)` once and advances a short `step_scene(settle_steps)` window; unlike later stages, it does not call `settle_arms_to_targets(...)` to wait for full convergence
    - `--settle_steps` defaults to `4` and is used to:
      - advance a few physics steps after init
      - add a short settle window after each stage trajectory is sent
    - `--joint_target_wait_steps` defaults to `60` and is used to:
      - keep stepping physics after the trajectory ends until the measured joints get closer to the final target
  - Impact:
    - no planning or execution logic changed in this round; this is a visualization-only update
  - Relevant code:
    - waypoint markers:
      - `update_ik_waypoint_visuals(...)`
      - `_ensure_ik_waypoint_marker_actors(...)`
      - `_ensure_ik_waypoint_endpoint_actors(...)`
    - init / stage settling:
      - `apply_robot_init_pose(...)`
      - `execute_single_arm_plan(...)`
      - `execute_dual_arm_plan(...)`
      - `settle_arms_to_targets(...)`
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed pure-mode EE-pose serialization compatibility in `pose_debug.jsonl`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - the newly added `current_left_ee_pose_world_wxyz` / `current_right_ee_pose_world_wxyz` export path in `record_frame(...)` incorrectly assumed that `robot.get_*_ee_pose()` returns a `sapien.Pose`
    - in this robot implementation the API actually returns a 7D list `[x, y, z, qw, qx, qy, qz]`
    - pure-mode batch runs therefore crashed on the first frame write with:
      - `AttributeError: 'list' object has no attribute 'p'`
  - Fix:
    - added `pose_like_to_world_wxyz(...)`
    - made the serializer accept both `sapien.Pose` objects and 7D pose lists
    - switched the relevant head/ee pose exports in `record_frame(...)` to use the shared helper
  - Impact:
    - no change to planning or execution behavior
    - this only fixes pure-mode `pose_debug.jsonl` serialization
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Added CLI parameter `--urdfik_cartesian_interp_auto_step_m` to control the translation-density threshold used by automatic waypoint mode when `--urdfik_cartesian_interp_steps=-1`.
- The old `0.05m` threshold was hardcoded; it is now configurable with the same default `0.05`. Fixed waypoint mode is unchanged.
- `render_hand_retarget_r1_npz_urdfik.py` now prints the active `auto_step_m` in `[ik-trajectory]` and `[ik-waypoints]` logs.
- Added execution-side joint-convergence tuning parameters for the AnyGrasp planner:
  - `--joint_command_scene_steps`
  - `--joint_target_wait_steps`
  - `--joint_target_wait_tol_rad`
- Modified files:
  - `code_painting/plan_anygrasp_keyframes_r1.py`
  - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
- Purpose:
  - address the case where `plan-request` / `plan-solution` already indicate that the planned endpoint is moving backward, but `attempt` still remains stuck at a forward-biased pose for many retries.
  - make the execution path advance more physics scene steps per joint waypoint and wait for final joint convergence before measuring reach error.
- Current diagnosis:
  - the earlier interpretation of `plan_vs_current_fwd_cm` was partially wrong.
  - the correct meaning is: in `plan-solution`, `plan_vs_current_fwd_cm > 0` means the current pose is ahead of the planned endpoint, so the planned endpoint is actually behind the current pose.
  - therefore, in the long-tail retry regime, IK is already commanding backward motion, but the execution layer is not converging far enough to that command.
- Extra docs:
  - `agent-read/2026-03-25_ik_execution_regression/README.zh.md`
  - `agent-read/2026-03-25_ik_execution_regression/README.en.md`

- Added "skip grasp re-execution when the target is the same" logic:
  - when `pregrasp` and `grasp` resolve to the same pose (the typical case is `--approach_offset_m 0.0`)
  - the system no longer replans and re-executes `grasp` after already reaching `pregrasp`
  - instead it reuses the `pregrasp` result and records `grasp_skipped_same_target`
- Why:
  - under `approach_offset_m=0.0`, logs showed that `pregrasp` could already converge
  - but the subsequent redundant `grasp` replanning to the exact same target pulled the end effector away again
- Additional note:
  - this round briefly tried a seeded/unseeded FK post-check scorer inside `urdfik.py`
  - that attempt produced clearly wrong pregrasp poses on the first retry
  - it has been reverted and is not kept

- Added a third correction for `cartesian_interp_ik`:
  - tightened the default position threshold in `urdfik.py` from `0.005m` to `0.001m`
  - tightened the default rotation threshold from `0.05rad` to `0.02rad`
  - stopped threshold relaxation from expanding toward `0.1m`
  - `render_hand_retarget_r1_npz_urdfik.py` now automatically reduces overly dense cartesian waypoint counts based on IK thresholds
- Immediate reason:
  - the `action` logs showed total target motion on the order of a few centimeters to about 9 cm, while `--urdfik_cartesian_interp_steps 30` reduced each translation step to roughly `3mm`
  - the old IK success threshold was `5mm`
  - that allowed the solver to accept many waypoint solves with almost no motion, so the final theoretical endpoint stayed far from the target
- Updated conclusion:
  - for this bug, “more retries” is not the main fix
  - increasing interpolation density can actually make it worse
  - the waypoint resolution must first remain larger than the IK success threshold

- Added a waypoint-level `seeded/unseeded` candidate comparison on top of that:
  - location: `code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - behavior: for each waypoint, try both “use current seed” and “no seed”
  - then use FK-based post-checking against that waypoint's `ee` target and keep the closer candidate
- Purpose:
  - address the remaining `action` failure mode where the interpolation-density fix helped, but the left arm was still being trapped in a local solution basin by the current seed

- Adjusted terminal debug output formatting:
  - `plan-request`
  - `plan-solution`
  - `attempt`
- Changes:
  - dual-arm logs are now printed as separate left/right lines
  - `theory` is shortened to:
    - `forward`
    - `backward`
    - `aligned`
  - `fwd_cm` now uses ANSI color highlighting so sign changes are easier to spot

- Further compressed the end-of-sample terminal summary:
  - it no longer prints the full `statuses_by_arm={...}` dictionary
  - it now uses a short format with:
    - `arms`
    - `arm`
    - `obj`
    - `fXX=cYY`
    - `pre/gr/act`
    - `video`

## 2026-03-25 11:51:21 +08

- Fixed a logging-format regression:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - `colorize_forward_cm()` was accidentally indented inside `short_direction_label()` during the previous log refactor
    - dual-arm `plan-request` logging then crashed with `NameError: name 'colorize_forward_cm' is not defined`
  - Fix:
    - restored `colorize_forward_cm()` to module scope
    - corrected its internal branch indentation so positive/negative/near-zero color cases work again
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

## 2026-03-25 12:08:00 +08

- Added a grasp/action object-collision toggle:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - New flag:
    - `--enable_grasp_action_object_collision 0|1`
  - Behavior:
    - default `0` preserves the original no-collision execution mode
    - when set to `1`, execution objects selected by the executed arm keep collision geometry
    - collision stays disabled during `pregrasp`
    - collision is enabled right before `close_gripper` and remains enabled through `action`
    - object attachment logic, TCP-relative transforms, target generation, and other pose transforms are unchanged
  - Implementation note:
    - stage-local collision enable/disable is implemented by caching and restoring SAPIEN collision groups
    - non-selected objects keep their original behavior
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-25 12:22:00 +08

- Adjusted the default visualization behavior of `plan_anygrasp_keyframes_r1.py`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Changes:
    - restored target-pose axes to be visible by default
    - hid the left/right wrist cameras by default in this planner script so wrist-camera frustums no longer appear in saved videos / viewer output
  - Notes:
    - this affects only the default behavior of `plan_anygrasp_keyframes_r1.py`
    - it does not change the shared base renderer behavior for other scripts
    - it does not affect head-camera or third-person capture itself
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

## 2026-03-25 13:05:00 +08

- Added pure/debug visualization controls to `plan_anygrasp_keyframes_r1.py`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - New flags:
    - `--debug_visualize_targets 0|1`
    - `--viewer_show_camera_frustums 0|1`
  - Changes:
    - restored target-axis rendering to an explicit flag instead of a hardcoded always-on path while keeping the default enabled
    - disabled SAPIEN `ControlWindow.show_camera_linesets` by default in the viewer path
    - after the wrist frustums are hidden, this also removes the remaining zed/third camera frustum lines by default
  - Notes:
    - `pure_scene_output` still controls clean main videos: no overlay text, no candidate grippers, no target axes
    - `viewer_show_camera_frustums=0` now controls viewer camera-line visibility
    - `debug_visualize_targets=1` keeps target axes available for debug runs
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed the batch wrapper so the new pure/debug visualization flags are forwarded:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Changes:
    - added `--debug_visualize_targets` to the batch parser
    - added `--viewer_show_camera_frustums` to the batch parser
    - forwarded both flags in `build_single_command()`
  - Reason:
    - the pure-mode batch command was being rejected by batch-layer argparse before the single-run script could receive the flags
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-25 13:35:00 +08

- Added a minimal "progressive close until contact/stall" path for `--enable_grasp_action_object_collision=1`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Analysis:
    - Previously the flag only re-enabled collision for the selected object during `grasp/action`
    - The gripper still used a one-shot `set_grippers(close)` update and only advanced a very small number of physics steps
    - As a result, the fingers could still visually close through the object even when collision shapes were present
  - Change:
    - Added `close_grippers_progressively_with_collision_stop()`
    - The `close_gripper` stage now closes in small command increments
    - Each increment advances physics, reads gripper joint `qpos`, and checks contacts between the selected object and the executing gripper links
    - Closure stops early once contact is present and gripper joint motion has stalled
    - The original one-shot `renderer.set_grippers(...)` path remains unchanged when the collision flag is disabled
  - Note:
    - This is a minimal fix; it does not change `pregrasp/grasp/action` target construction or the TCP-to-object attachment transform
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed a SAPIEN API compatibility issue in the progressive-close path:
  - Problem:
    - `PhysxArticulationJoint` in the current environment does not provide `get_qpos()`, which caused a crash when entering `close_gripper`
  - Fix:
    - read the full articulation `qpos` from `entity.get_qpos()`
    - then recover each gripper joint's actual position by accumulating offsets over `active_joints` with `joint.get_dof()`
  - Code location:
    - `code_painting/plan_anygrasp_keyframes_r1.py:_get_gripper_joint_positions`

## 2026-03-25 14:05:00 +08

- Added an explicit warning when `grasp` has not reached the target but execution still proceeds to `close_gripper`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Behavior:
    - before `close_gripper`, if the `grasp` stage reports `reached=False`, the script now prints `[warn] grasp_not_reached_before_close ...`
    - this is logging only and does not change the execution logic

- Added an automatic waypoint mode for URDF IK Cartesian interpolation:
  - File:
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - New behavior:
    - when `--urdfik_cartesian_interp_steps > 1`, the original fixed-step mode is preserved
    - when `--urdfik_cartesian_interp_steps -1`, an automatic mode is enabled
  - Automatic mode rule:
    - if absolute translation distance is `<= 5cm`, no intermediate TCP waypoint is added (start and goal only)
    - once translation exceeds `5cm`, one additional intermediate waypoint is added per extra `5cm` bucket
    - examples:
      - `10cm` translation -> `1` intermediate waypoint
      - `15cm` translation -> `2` intermediate waypoints
  - Note:
    - this automatic mode only adjusts waypoint count from TCP translation distance and does not change the IK target or execution semantics

- Recorded the actual execution semantics of `cartesian_interp_ik`:
  - New documents:
    - `agent-read/V1.15_urdfik_cartesian_interp_execution_semantics_ZH.md`
    - `agent-read/V1.15_urdfik_cartesian_interp_execution_semantics.md`
  - Conclusion:
    - intermediate `ee/tcp` waypoints are indeed used for per-waypoint IK solving
    - after the latest fix, the execution stage now also replays the corresponding `joint_waypoints`
    - `cartesian_interp_ik` now affects both IK solving and the final executed path more directly

## 2026-03-25 14:25:00 +08

- Fixed the `urdfik` execution layer so it actually consumes `joint_waypoints`:
  - File:
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - Changes:
    - `execute_plans(...)` no longer performs only an endpoint interpolation from `current_joints` to `target_joints`
    - it now directly consumes `plan["position"]` / `plan["velocity"]`
    - in dual-arm execution, left and right trajectories are interleaved by relative trajectory progress, matching the base renderer semantics
    - `_execute_single_ik_plan(...)` also now prefers replaying the full `plan["position"]` trajectory
  - Effect:
    - the `joint_waypoints` generated by `cartesian_interp_ik` now become part of the real executed path
# 2026-03-25

- Enhanced pure-mode outputs:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Changes:
    - `pure_scene_output=1` no longer generates `debug_selection_preview.mp4`
    - the main planning run now also writes:
      - `head_cam_plan.mp4`
      - `left_wrist_cam_plan.mp4`
      - `right_wrist_cam_plan.mp4`
    - pure mode now auto-enables `pose_debug.jsonl` even without explicitly passing `--save_pose_debug 1`
    - `pose_debug.jsonl` now additionally records:
      - left/right wrist-camera poses
      - left/right TCP / EE poses
      - left/right 6D arm qpos
      - left/right gripper finger-joint qpos
      - object actor poses / replay poses
    - `plan_summary.json` now includes the corresponding video and data paths
  - Documentation:
    - `agent-read/2026-03-25_pure_mode_outputs_ZH.md`
    - `agent-read/2026-03-25_pure_mode_outputs.md`
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py code_painting/render_hand_retarget_r1_npz_urdfik.py`
    - `git diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- Added a `--debug_visualize_ik_waypoints` debug parameter for `--planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik`, rendering intermediate `tcp_waypoints_world` into viewer/debug output.
- The visualization is a “point + local forward-axis” marker and only shows intermediate waypoints; the start and final target remain excluded, with the final target still represented by the existing target axis.
- This change affects debug display only and does not change waypoint generation, IK solving, trajectory execution, or collision behavior.
- `plan_anygrasp_keyframes_r1_batch.py` now forwards the parameter as well.
- Validation: `python -m py_compile` and `git diff --check` are run after this round.
# 2026-03-25

- Added a stage-level run analysis document:
  - `agent-read/2026-03-25_overall_run_analysis_ZH.md`
  - `agent-read/2026-03-25_overall_run_analysis.md`
- Recorded the current conclusions:
  - trajectory shape has improved, but planner/IK endpoints are still frequently wrong
  - unreached `grasp` is a major reason why `close_gripper` sees no contact
  - current gripper contact detection monitors finger-joint `child_link`s rather than the `left/right_gripper_link` base links
- No runtime logic was changed in this round; this was documentation-only analysis.
