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
- Added AnyGrasp orientation-debug parameters (`--candidate_orientation_remap_label`, `--candidate_post_rot_xyz_deg`) and documented the likely missing fixed orientation conversion. Debug gripper actors now visualize the AnyGrasp opening width, and top1-only debugging is documented.
- Decoupled `urdfik` workflows from eager `curobo` imports by switching `replay_r1_h5.py` to import `Robot` from `envs.robot.robot` directly and by making `CuroboPlanner` a lazy import inside `Robot.set_planner()`. This prevents urdfik-only runs from failing on curobo CUDA OOM during module import.

- Added offscreen-safe rank preview PNG export for the AnyGrasp planner (`--save_rank_preview_images`, `--rank_preview_top_n`). Each keyframe can now produce rank-1..N still images that show the left candidate in blue and the right candidate in orange, making manual candidate selection possible even when the interactive viewer fails to open.

- Added manual AnyGrasp candidate overrides via repeated `--manual_candidate FRAME ARM CANDIDATE_IDX`. Partial overrides reorder and surface the chosen candidates for debug; complete two-keyframe overrides for one arm are used directly by the planner.

- Shrunk the green common-candidate labels again in AnyGrasp debug outputs so first-keyframe manual inspection is less cluttered.

- Changed the AnyGrasp planner viewer path so debug axes and candidate grippers remain visible persistently in a headed SAPIEN viewer instead of flashing for a single frame. Offscreen video output still hides them unless the debug video is being written.
