# 2026-03-25 Overall Run Analysis

## Context

This note summarizes batch-run behavior around the following configuration:

- `--planner_backend urdfik`
- `--urdfik_trajectory_mode cartesian_interp_ik`
- `--urdfik_cartesian_interp_steps -1`
- `--reach_error_pose_source tcp`
- `--enable_grasp_action_object_collision 1`

The user observed:

- Trajectory shape looks better than before
- But there are still frequent failures in:
  - `pregrasp / grasp / action`
  - `close_gripper` often never contacts the object
  - suspicion that the gripper-contact link selection itself may be wrong

## Overall Assessment

At this point, the main issue is no longer primarily “bad execution trajectory shape”. It is more likely:

1. `URDF IK / planner` often produces a final solution that is still far from target
2. `grasp` often fails to bring the TCP close enough to the object
3. Because `grasp` is already off, `close_gripper` often never touches the object at all

In short:

- The problem of “how the arm moves” has improved
- The problem of “where the plan ends up” is still dominated by planner / IK solution quality

## What the Logs Suggest

### 1. `action` failure is mostly not an execution-following problem

Typical pattern:

- `plan-request` already says `theory=backward`
- but `plan-solution` still shows:
  - large positive `plan_vs_target_fwd_cm`
  - `plan_vs_current_fwd_cm` near zero or small positive

This means:

- from the current state, the correct motion should be backward
- but the planner / IK “theoretical endpoint” is still in front of the target
- execution is mostly just realizing that bad endpoint

So these failures should not mainly be blamed on execution failing to follow the requested path.

### 2. `close_gripper` logs currently indicate “did not reach object”, not “touched object and kept closing through it”

Typical log:

```text
[warn] grasp_not_reached_before_close ...
[gripper-close] left:reason=target_reached,cmd=0.000,contact=0 right:reason=target_reached,cmd=0.000,contact=0
```

Meaning:

- the system already knows `grasp` was not reached before closing starts
- during progressive closing, no object contact was detected
- the fingers simply reached command target `0.0` instead of stopping due to contact

So this is more consistent with:

- the gripper never touched the object
- rather than “the object was touched but contact detection failed and the fingers still closed through it”

### 3. The `execute_plans(...)` fix has already moved “trajectory shape” into a secondary role

Previously, `cartesian_interp_ik` used intermediate waypoints only during planning, while execution ignored the full `joint_waypoints`.

The current version already:

- builds `tcp_waypoints_world`
- solves IK per waypoint into `joint_waypoints`
- executes the full `plan["position"]` trajectory

Therefore:

- if the video trajectory looks more reasonable now, that is expected
- the remaining failures are more likely caused by waypoint quality or IK solution quality than by execution ignoring the path entirely

## Which Links Are Actually Used for Gripper Contact Detection

## Code Path

Contact detection is implemented in:

- `plan_anygrasp_keyframes_r1.py`
  - `_get_gripper_link_entities(...)`
  - `_contact_involves_entities(...)`
  - `close_grippers_progressively_with_collision_stop(...)`

Gripper joint definitions come from:

- `envs/robot/robot.py`
  - `init_joints()`
  - `get_gripper_joints(...)`

Configuration comes from:

- `robot_config_R1.json`

### What is actually being monitored

Current `_get_gripper_link_entities(...)` logic:

- reads `renderer.robot.left_gripper` / `renderer.robot.right_gripper`
- for each gripper joint, takes `joint_info[0].child_link`

So the contact detector is **not** monitoring:

- `left_gripper_link`
- `right_gripper_link`

Instead, it monitors the child links of the finger joints.

From `robot_config_R1.json`, the configured gripper joints are:

- left:
  - `left_gripper_finger_joint1`
  - `left_gripper_finger_joint2`
- right:
  - `right_gripper_finger_joint1`
  - `right_gripper_finger_joint2`

Therefore, the actual monitored links are:

- `left_gripper_finger_joint1.child_link`
- `left_gripper_finger_joint2.child_link`
- `right_gripper_finger_joint1.child_link`
- `right_gripper_finger_joint2.child_link`

That is, the two finger links on each hand, not the whole gripper base link.

### What this implies

This detection design has two important consequences:

1. If the object is really between the two fingers, this is a reasonable detection strategy
2. If the object mainly contacts:
   - the palm / gripper base
   - `left_gripper_link` / `right_gripper_link`
   - any non-finger child link
   then the current detector will not count that as “gripper-object contact”

So the user’s suspicion about contact-link selection is reasonable. However, from the current logs:

- the stronger evidence still points to `grasp` itself being off
- rather than the contact detector being definitively wrong

A conservative conclusion is:

- the detector currently covers finger child links only
- so it may miss palm/base contact
- but the current logs still more strongly suggest “the gripper never reached the object”

## Current Stage Conclusion

### What is already clear

- executing full `joint_waypoints` has improved trajectory shape
- the progressive `close_gripper` logic itself is not obviously contradictory
- many current failures are still best explained by:
  - bad planner / IK endpoints
  - `grasp` not reaching target
  - therefore `close_gripper` seeing no contact

### What still needs confirmation

- whether the intermediate `cartesian_interp_ik` waypoints themselves are reasonable
- whether IK falls into bad basins in some replans
- whether finger-only contact detection misses meaningful palm/base contact cases

## Suggested Next Debug Signals

Without changing code, prioritize three log groups:

1. `plan-request`
   - whether the current state should theoretically move forward or backward
2. `plan-solution`
   - whether the planner endpoint still sits clearly in front of the target
3. `gripper-close`
   - whether it stops by `contact_stall` or ends at `target_reached, contact=0`

If you see:

- `grasp_not_reached_before_close`
- `gripper-close ... contact=0`

then the first interpretation should be:

- the grasp pose never got close enough,
- not that close-phase contact detection necessarily failed.
