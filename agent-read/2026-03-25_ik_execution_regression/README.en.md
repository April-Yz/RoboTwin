# 2026-03-25 IK / Execution Backtracking Debug Note

## 1. Background

Under the following configuration, the `pregrasp` stage can keep retrying for a long time:

- `planner_backend=urdfik`
- `urdfik_trajectory_mode=cartesian_interp_ik`
- `reach_error_pose_source=ee`
- `approach_offset_m=0.0`
- `candidate_target_local_x_offset_m=0.0`

Observed pattern:

- `fwd_cm` starts negative and approaches zero
- then becomes positive
- then stabilizes around `+11cm ~ +13cm`

## 2. Meaning of the current logs

The current code already emits three log families:

1. `plan-request`
   - current measured pose vs target pose
   - `theory` says whether the motion should go forward or backward along the target forward axis

2. `plan-solution`
   - forward-kinematics result of the planned `target_joints`
   - compared both against:
     - planned endpoint vs target
     - planned endpoint vs current

3. `attempt`
   - the actual error measured after execution

## 3. Corrected interpretation

`plan_vs_current_fwd_cm` had been partially misread before.

The correct reading is:

- `plan_request_diagnostics(current, planned)` treats `planned` as the target and `current` as the current pose
- therefore `plan_vs_current_fwd_cm > 0` means:
  - the current pose is ahead of the planned endpoint
  - equivalently, the planned endpoint is behind the current pose

So in the long-tail retry regime:

- `plan-request` already says the arm should move backward
- `plan-solution` also says the planned endpoint is behind the current pose
- but `attempt` still does not move backward enough

This makes the late-stage issue look more like an execution-layer convergence problem than a purely wrong IK direction.

## 4. Changes in this round

### 4.1 More scene steps per commanded waypoint

Added:

- `--joint_command_scene_steps`

Effect:

- each commanded arm waypoint no longer uses only `step_scene(1)`
- more physics steps are given so the drive target can actually be followed

### 4.2 Final convergence wait after the trajectory

Added:

- `--joint_target_wait_steps`
- `--joint_target_wait_tol_rad`

Effect:

- after the whole trajectory has been sent, the code keeps checking the actual arm joints against the final commanded target
- if the joint error is still above tolerance, it keeps advancing the physics scene until convergence or timeout

### 4.3 Final joint-error overlay

The execution overlay now records:

- single-arm: `joint_max_err=...rad`
- dual-arm: `left_joint_max_err=...rad` / `right_joint_max_err=...rad`

This is meant to separate:

- “the plan was correct but execution did not reach it”
- from
- “the plan itself is wrong”

## 5. Current hypothesis

This round is testing the following idea first:

- if the main issue is that execution does not settle to the final joint target
- then more scene steps plus a final convergence wait should reduce the observed `fwd_cm` significantly

If the reduction is still weak, the next round should go back to:

- the single-seed plus threshold-relaxation policy in `urdfik.py`
- and whether waypoint IK enters a bad basin early

## 6. 2026-03-25 second correction

After rerunning `d_pour_blue_0`, a more direct issue appeared:

- under `approach_offset_m=0.0`
- the `pregrasp` target and the `grasp` target are actually identical

That means:

- once `pregrasp` is already correct
- replanning and re-executing `grasp` to the exact same pose is redundant
- and the logs showed that this redundant `grasp` stage could pull the end effector away again

So the code now adds:

- `grasp_skipped_same_target`

Behavior:

- if `pregrasp_pose` and `grasp_pose` are effectively the same in both position and rotation
- the planner skips the extra `grasp` planning/execution stage
- and directly reuses the `pregrasp` result

This rule is meant to match the actual semantics of your current command, instead of blindly relying on more retries.

## 7. 2026-03-25 third correction

Further analysis of the `action` logs exposed a lower-level issue:

- `plan-request` in `action` already clearly says the arm should move backward along the target forward axis
- but `plan-solution` shows that the planned endpoint is only a few millimeters behind the current pose
- while it is still 5 to 11 cm away from the actual final target

That means the problem is not simply “execution failed to follow the plan”. Instead:

- `cartesian_interp_ik` is splitting the motion too finely
- each waypoint translation becomes smaller than the IK success threshold
- the solver can therefore mark a waypoint as success while barely moving
- after many such small successes, the full trajectory still does not reach the target

In this command, the mismatch is direct:

- `--urdfik_cartesian_interp_steps 30`
- total `action` translation is only on the order of a few centimeters to about 9 cm
- per-waypoint translation becomes roughly `3mm`
- the old `urdfik.py` position success threshold was `5mm`

So the third correction is:

1. Tighten IK success thresholds
   - position threshold: `0.005m -> 0.001m`
   - rotation threshold: `0.05rad -> 0.02rad`

2. Tighten threshold relaxation
   - no longer allow it to expand toward `0.1m`
   - only relax within a small bounded range

3. Automatically compress overly dense cartesian waypoint counts
   - if the requested waypoint count makes each translation/rotation step too small relative to the IK thresholds
   - the code now reduces the requested count to a safer effective count

The key takeaway for this round is:

- for this bug, increasing interpolation density is not the fix
- it can actually make waypoint IK easier to satisfy without moving
- so the first correction is to restore a sensible scale relationship between waypoint resolution and IK success thresholds

## 8. 2026-03-25 fourth correction

After the third correction, `pregrasp` could reach the threshold again, but `action` still showed an asymmetric residual issue:

- the right arm improved clearly
- the left arm still stalled around `5cm`

At that point the logs already showed:

- waypoint counts had been compressed to small values
- so “waypoints are too fine” was no longer the main cause for left-arm `action`
- the remaining issue looked more like the current seed locking IK into a local basin

So the fourth correction does not introduce another global scorer. Instead it performs a narrow per-waypoint candidate comparison:

1. solve once with the current seed
2. solve once again without a seed
3. use FK post-checking against that waypoint's `ee` target
4. keep only the candidate that is actually closer to the waypoint

The intent is:

- keep the continuity advantage of seeded waypoint IK when it works
- but still give each waypoint one chance to escape a bad local basin when the seed is clearly trapping it
