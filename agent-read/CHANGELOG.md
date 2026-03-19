# CHANGELOG

## 2026-03-19

- Added command-oriented documentation in:
  - `agent-read/V1.8_command_log.md`
  - `agent-read/V1.8_command_log_ZH.md`
- Added a dated debug note `agent-read/2026-03-19_current_target_only_and_raw_preview/README.md` covering:
  - current-target-only execution visualization
  - direct AnyGrasp raw preview generation without RoboTwin
- Added a consolidated workflow documentation set for `code_painting/`:
  - `agent-read/V1.7_pipeline_index.md`
  - `agent-read/V1.7_hand_pipeline.md`
  - `agent-read/V1.7_object_pipeline.md`
  - `agent-read/V1.7_anygrasp_pipeline.md`
- Updated `agent-read/README.md` so the repo overview now points to the end-to-end pipeline docs instead of only the AnyGrasp planner notes.
- Documented the current project boundary explicitly: hand extraction and FoundationPose extraction are upstream producers; RoboTwin starts from their outputs and handles replay, export, visualization, and execution.
- Added matching Chinese docs for the same workflow set:
  - `agent-read/V1.7_pipeline_index_ZH.md`
  - `agent-read/V1.7_hand_pipeline_ZH.md`
  - `agent-read/V1.7_object_pipeline_ZH.md`
  - `agent-read/V1.7_anygrasp_pipeline_ZH.md`
- Investigated the gap between a successful single-keyframe dual-arm AnyGrasp run and a weaker two-keyframe run. Recorded the findings in `agent-read/2026-03-19_two_keyframe_dual_arm_debug/README.md`.
- Fixed dual-arm AnyGrasp target visualization so selected keyframe axes are tracked per `(frame, arm)` instead of per `frame`, and dual-arm execution now keeps both arms' selected keyframes active in the debug viewer.
- Tightened execution debug rendering so only the currently active keyframe's selected axes are shown; later keyframe targets are hidden until their stage becomes active.
- Added `code_painting/render_anygrasp_ranked_preview.py`, a non-RoboTwin preview tool that annotates AnyGrasp's own `vis/grasp_result_*.png` with `candidate_idx` labels while separating left/right outputs and preserving raw-score rank order.
- Refined `render_anygrasp_ranked_preview.py` so it can use raw replay RGB frames as the base image (`--base_image_dir`, `--base_image_mode raw`) instead of the dense AnyGrasp `vis` image. Candidate labels are now plain black text without colored number styling.
- Added lightweight grasp wireframe rendering to `render_anygrasp_ranked_preview.py` via `--draw_grasp_boxes 1`, so raw-image previews can still show a simple grasp shape without inheriting the dense AnyGrasp visualization.
- Added `--pause_after_keyframe1_seconds` to the AnyGrasp planner and batch wrapper. After keyframe-1 is reached and the gripper is closed, the robot can now hold that pose for a fixed duration before starting keyframe-2. The planner also prints an explicit terminal log when keyframe-1 has been reached and when the pause begins.
- Added `agent-read/V1.9_anygrasp_ik_execution_logic_ZH.md`, documenting the current `urdfik` chain, the TCP->endlink conversion, why execution shows about 25 interpolated plan steps, and what would happen in the viewer if interpolation were removed.

## 2026-03-17

- Investigated why right hand appears not to reach target in AnyGrasp keyframe videos while left hand does.
- Confirmed from code path and summary semantics that current stage execution is single-arm; non-selected arm is supervision-only.
- Updated `plan_anygrasp_keyframes_r1.py` to export explicit execution-mode metadata in `plan_summary.json`:
  - `executed_arms`
  - `supervision_only_arms`
- Added runtime log line `[exec-mode] selected_arm=... supervision_only_arms=...` to reduce ambiguity during debugging.
- Added analysis note at `agent-read/2026-03-17_right_hand_supervision_analysis/README.md`.
- Added versioned documentation note `agent-read/V1.4_anygrasp_execution_mode_supervision.md`.
- Added `--execute_both_arms` to AnyGrasp planner and batch wrapper. With `--arm auto --execute_both_arms 1`, both arms are executed sequentially and each arm gets independent stage reach results in `stages_by_executed_arm`.
- Added dual-arm analysis note `agent-read/2026-03-17_dual_arm_execution/README.md` and versioned record `agent-read/V1.5_anygrasp_dual_arm_execution.md`.
- Updated the above dual-arm feature to synchronized stage execution: each stage now plans/executes both arms together and only advances when both arms reach tolerance (or retries are exhausted).
- Added init-pose start support in `plan_anygrasp_keyframes_r1.py`: execution now explicitly re-applies renderer init joints before stage planning, and `--init_prefix_frames N` can emit the first N frames as fixed init-pose frames for easier post-trim/deletion.

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

- Added per-candidate axis actors for the headed AnyGrasp viewer, so non-selected manual candidates now keep their own coordinate axes instead of showing only the gripper geometry.

- Added `--replan_until_reached` and `--replan_until_reached_max_attempts` to the AnyGrasp planner. The stage executor can now keep replanning from the current state until the reach tolerance is met or an extended attempt budget is exhausted, and each stage records `attempt_history` in the summary for accuracy debugging.

- Added `--reach_error_pose_source {tcp,ee}` to the AnyGrasp planner and stage summaries. Reach-error computation can now be switched between fingertip TCP and wrist/endlink space. The planner also records non-selected-arm supervision errors (for example the manually chosen right-hand candidate while executing the left hand) inside per-attempt histories.
- Fixed a regression in the AnyGrasp planner reach-debug path where `execute_stage_until_reached()` forwarded `supervision_targets` into `execute_single_arm_plan()`, causing `TypeError: unexpected keyword argument 'supervision_targets'` before any stage execution. The supervision targets are now only used for post-plan error accounting, as intended.
- Added `--replay_objects_during_action` and `--replay_objects_ignore_collision` to the AnyGrasp planner. The formal execution path can now replay `replay_dir/multi_object_world_poses.npz` during the second keyframe `action` stage so bottle/cup follow their recorded trajectories instead of being attached to the TCP. When collision-ignore mode is enabled, replayed objects are created as visual-only kinematic actors.
- Validation: `python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`
