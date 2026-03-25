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
