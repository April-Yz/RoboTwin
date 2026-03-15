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
- Added projected candidate IDs to AnyGrasp planner debug videos.
- Switched left/right arm markers from white to black for better contrast on white backgrounds.
- Added a no-code reproduction guide at `code_painting/README_anygrasp_keyframe_planner_repro.md`.
