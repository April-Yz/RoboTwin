# CHANGELOG.en

## 2026-03-25

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
    - but the execution stage does not replay those waypoints segment by segment
    - the final motion is still mainly a joint-space linear interpolation from `current_joints` to the final `target_joints`
