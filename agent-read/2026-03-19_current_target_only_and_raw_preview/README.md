# 2026-03-19 Current-Target-Only Viewer And Raw AnyGrasp Preview

## Problem

Two issues remained in the AnyGrasp debug path:

1. Execution viewer/debug videos could show selected axes from a later keyframe while the current stage was still moving to the earlier keyframe.
2. Manual candidate inspection still depended on RoboTwin rank previews, even when the user only wanted a direct view of AnyGrasp's own result image.

## Root Cause

### 1. Current vs future keyframe visualization

`record_frame()` restored every selected keyframe axis from `debug_execution_state.selected_keyframes`, regardless of `active_frame`.

That meant:
- during frame-1 execution, frame-21 axes could still be visible
- in dual-arm mode, the user could not isolate the current pair of targets cleanly

### 2. No direct AnyGrasp-only preview

The repo had:
- RoboTwin rank preview PNGs
- RoboTwin debug videos

It did not have a script that:
- skipped RoboTwin completely
- read `grasp_*.json` and `vis/grasp_result_*.png`
- overlaid `candidate_idx` labels directly on the AnyGrasp image

## Fix

### A. Current-target-only visualization

Updated `code_painting/plan_anygrasp_keyframes_r1.py`:

- Added `selected_keyframes_for_active_frame(...)`
- `record_frame()` now restores only the selected keyframes whose `source_frame == active_frame`
- debug label overlay now also uses only the currently active keyframe set
- arm-specific red-selection logic in `annotate_candidate_labels()` now keys on `(frame, arm)` instead of just `frame`

Result:
- during execution, only the current target pair is shown
- the next keyframe target is not shown early
- dual-arm same-frame display no longer loses one arm's selected label

### B. Raw AnyGrasp direct preview

Added:
- `code_painting/render_anygrasp_ranked_preview.py`

This script:
- reads `grasps/grasp_<frame>.json`
- reads `vis/grasp_result_<frame>.png`
- sorts by raw AnyGrasp score
- uses hand orientation only to distinguish left/right reference ranking context
- writes left, right, and combined annotated images

## Output

Direct AnyGrasp preview outputs:
- `frame_<frame>_left_score_rank.png`
- `frame_<frame>_right_score_rank.png`
- `frame_<frame>_left_right_score_rank.png`
- `summary.json`

## Command Reference

See:
- `agent-read/V1.8_command_log.md`
- `agent-read/V1.8_command_log_ZH.md`
