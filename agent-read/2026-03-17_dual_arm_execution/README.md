# Dual-Arm Synchronized Stage Execution Note (2026-03-17)

## Problem
In single-arm AnyGrasp execution, non-selected arm remained supervision-only and did not move to its target.

## Change
Added `--execute_both_arms` to AnyGrasp keyframe planning:
- Works with `--arm auto`.
- Executes both arms in synchronized stage control.
- In each stage (`pregrasp/grasp/action`), both arms are replanned and executed together.
- Stage ends only when both arms satisfy tolerance or attempt budget is exhausted.

## Output Semantics
`plan_summary.json` now includes:
- `execute_both_arms`
- `stages_by_executed_arm`
- `selected_objects_by_executed_arm`
- `selected_candidates_by_executed_arm`
- `supervision_targets_by_executed_arm`
- `supervision_only_arms_by_executed_arm`

Compatibility fields are preserved:
- `stages`, `selected_candidates`, `supervision_targets`, `supervision_only_arms`
  are still present for the primary executed arm.

## How to Validate
1. Run with `--arm auto --execute_both_arms 1` and manual candidates for both sides.
2. Confirm runtime logs include `selected_arm=both` and the summary `execution_mode=dual_sync`.
3. Check `stages_by_executed_arm.left.reached` and `stages_by_executed_arm.right.reached`.
