# CHANGELOG

## 2026-03-12

### AnyGrasp planner debug visualization
- Fixed the `plan_anygrasp_keyframes_r1.py` call chain after introducing separate common-candidate and arm-specific candidate overlays.
- Added explicit left/right candidate distinction in debug visualization:
  - left arm: white marker above the candidate gripper
  - right arm: white marker below the candidate gripper
- Kept color semantics consistent across debug videos:
  - green = common/all displayed candidates
  - blue = arm-specific top-ranked candidates
  - red = selected candidate
- Added `arm_debugs` to `plan_summary.json` so downstream inspection can see per-arm diagnostics and whether a hand was actually selected.
- Updated `code_painting/README_anygrasp_keyframe_planner.md` to document the left/right visualization markers.

### 2026-03-15
- Shrunk candidate IDs in AnyGrasp planner debug videos and removed background boxes.
- Switched from marker-based left/right distinction to color-based distinction: blue for left, orange for right, red for selected.
- Added a no-code reproduction guide at `code_painting/README_anygrasp_keyframe_planner_repro.md`.
- Fixed a regression where debug execution video recording passed one extra argument into `update_candidate_debug_visuals`, causing runtime failure before planning started.
- Fixed a second label-overlay regression where `record_frame()` called `annotate_candidate_labels()` without `selected_keyframes`, causing a runtime crash before debug execution video writing.
- Reduced candidate label size again, removed visual clutter from background boxes, and made selected candidates larger. Added `--debug_common_candidate_top_k` so raw green candidates can be hidden or capped independently from per-arm top-k display.
