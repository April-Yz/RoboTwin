# Right-Hand Supervision Analysis (2026-03-17)

## Question
Why does the left arm appear to reach target in the video while the right arm does not?

## Key Findings
- In this run, the planner executes only one arm at a time (`selected_arm`).
- For the provided result (`d_pour_blue_1`), `selected_arm` is `left`.
- The right arm target from manual candidate selection is used as a supervision-only target (error accounting), not an execution target.
- Therefore, in the current workflow, right-arm non-arrival is expected behavior, not evidence of a short execution horizon.

## Evidence
- `plan_summary.json` contains `selected_arm: left`.
- Per-attempt records include `supervision_errors.right` while stage command is still `arm=left`.
- In executor code, stage rollout calls `renderer.robot.set_arm_joints(..., arm)` only for the selected arm.

## Root Cause Classification
- Not primarily "execution time too short".
- Not primarily "right-arm unreachable" within this specific run setup.
- Main reason: right arm is not actuated in this single-arm execution pipeline.

## This Update
- Added explicit execution-mode metadata in planner output:
  - `executed_arms`
  - `supervision_only_arms`
- Added runtime log:
  - `[exec-mode] selected_arm=... supervision_only_arms=...`

## Suggested Validation Runs
1. Keep current command, compare right supervision error trend to confirm it stays mostly flat when right arm is not actuated.
2. Run with `--arm right` and the same right manual candidate to test true right-arm reachability.
3. If dual-arm behavior is desired, add a dual-arm execution mode (new feature, not included in this patch).
