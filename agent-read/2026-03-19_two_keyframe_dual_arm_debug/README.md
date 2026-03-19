# Two-Keyframe Dual-Arm Debug Note (2026-03-19)

## Problem

The user observed that:

1. A first-keyframe-only command could bring both hands to the requested frame-1 targets reliably.
2. A later two-keyframe command did not reach frame 1 as well and also did not transition cleanly to the second target.
3. In the viewer, target grippers and axes were not clear enough to tell which hand and which keyframe target was active.

## Root Cause Breakdown

### 1. The two commands were not equivalent

The successful first command included the following important settings:

- `--arm auto`
- `--execute_both_arms 1`
- `--init_prefix_frames 15`
- `--keyframes 1 1`

The later command that underperformed changed behavior in three ways:

- It omitted `--execute_both_arms 1`, so execution could fall back to single-arm semantics.
- It omitted `--init_prefix_frames 15`, so the execution did not begin from the same stabilized visual prefix.
- It changed `--keyframes 1 1` to `--keyframes 1 21`, which means the second pose was no longer the same manually validated frame-1 pose. Frame 21 was still being selected automatically unless separately overridden.

Because of those differences, the second run was not a direct extension of the first successful setup.

### 2. Dual-arm execution existed, but dual-arm target visualization was incomplete

The dual-arm stage controller was already synchronized:

- in each stage, both arms were planned,
- both were executed in lockstep,
- and stage transition required both sides to satisfy reach checks or exhaust the retry budget.

However, the debug visualization still behaved like a primary-arm view in two places:

- keyframe axis actors were stored one-per-frame instead of one-per-frame-per-arm,
- dual-arm execution populated `debug_execution_state.selected_keyframes` from only one arm's sequence.

That meant:

- when left and right both had a selected candidate at the same keyframe,
- one arm's keyframe axis could overwrite the other,
- and the viewer could make the right-hand target look ambiguous even though the dual-arm stage logic was active.

### 3. Current IK backend

For the user's command, the backend is:

- `--planner_backend urdfik`

Code path:

- `plan_anygrasp_keyframes_r1.py` chooses `render_hand_retarget_r1_npz_urdfik.HandRetargetR1URDFIKRenderer`
- that renderer uses `URDFInverseKinematics`
- it solves against:
  - `left_gripper_link`
  - `right_gripper_link`

This is not the curobo path.

## Code Changes Made

Updated:
- `code_painting/plan_anygrasp_keyframes_r1.py`

Behavior changes:

1. Keyframe axis actors are now indexed by `(frame, arm)` instead of only `frame`.
2. Dual-arm execution now pushes both arms' selected keyframes into `debug_execution_state`.
3. Dual-arm stage transitions now explicitly update both left and right target visuals before:
   - `pregrasp`
   - `grasp`
   - `action`

Result:

- the viewer can show both hands' selected targets for the active keyframe,
- the right-hand target is no longer overwritten by the left-hand target when both belong to the same frame,
- dual-arm target axes now track the actual current stage target for both arms.

## Validation Guidance

To compare apples-to-apples with the earlier successful single-keyframe setup, the two-keyframe test should preserve:

- `--arm auto`
- `--execute_both_arms 1`
- `--init_prefix_frames 15`

and then add the second keyframe target.

If frame 21 is not manually pinned, the second keyframe still depends on automatic candidate selection.

## Recommended Test Command

```bash
CUDA_VISIBLE_DEVICES=3 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes \
  --ids 1 \
  --keyframes 1 21 \
  --arm auto \
  --execute_both_arms 1 \
  --lighting_mode front_no_shadow \
  --planner_backend urdfik \
  --enforce_target_object_constraint 0 \
  --enforce_candidate_distance_constraint 0 \
  --debug_show_all_candidates 1 \
  --debug_common_candidate_top_k 0 \
  --debug_candidate_top_k 1 \
  --candidate_orientation_remap_label identity \
  --candidate_post_rot_xyz_deg 0 0 0 \
  --manual_candidate 1 left 5 \
  --manual_candidate 1 right 11 \
  --replan_until_reached 1 \
  --replan_until_reached_max_attempts 20 \
  --reach_error_pose_source ee \
  --save_debug_preview 1 \
  --save_debug_execution_preview 1 \
  --enable_viewer 1 \
  --viewer_frame_delay 0.02 \
  --viewer_wait_at_end 1 \
  --init_prefix_frames 15
```

If frame 21 also needs to be frozen for both arms, add:

```bash
--manual_candidate 21 left <IDX> \
--manual_candidate 21 right <IDX>
```
