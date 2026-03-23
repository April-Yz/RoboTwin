# CHANGELOG

## 2026-03-20

- Added explicit Backend-A trajectory submodes to the AnyGrasp planner:
  - `--urdfik_trajectory_mode joint_interp`
  - `--urdfik_trajectory_mode cartesian_interp_ik`
  - `--urdfik_cartesian_interp_steps`
- `joint_interp` preserves the original behavior: solve one endpoint IK, then interpolate in joint space.
- `cartesian_interp_ik` is the new behavior: interpolate TCP waypoints in Cartesian space, solve IK waypoint by waypoint with the previous solution as the seed, then execute the resulting multi-point joint trajectory.
- Exposed the new parameters in both:
  - `code_painting/plan_anygrasp_keyframes_r1.py`
  - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
- Extended `plan_summary.json` to record:
  - `planner_backend`
  - `urdfik_trajectory_mode`
  - `urdfik_cartesian_interp_steps`
- Updated `agent-read/ik_analyze/anygrasp_keyframe_ik_planning_analysis.md`
- Updated `agent-read/ik_analyze/anygrasp_keyframe_ik_planning_analysis.en.md`
- Updated command docs so there are now explicit runnable command examples for:
  - Backend A original mode
  - Backend A Cartesian-waypoint IK mode
  - Backend B cuRobo mode
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/render_hand_retarget_r1_npz_urdfik.py code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`

- Added per-frame execution diagnostics to the AnyGrasp planner debug path.
- `debug_execution_preview.mp4` now renders a colored metric panel for left/right arms, including:
  - target minus current end-effector pose
  - planned object pose minus actual object pose
  - target pose minus actual object pose
  - current end-effector pose minus actual object pose
- Added `debug_execution_metrics.jsonl` to the planner output directory. This file is written alongside execution preview frames and records, per frame:
  - current planner stage
  - active keyframe
  - current head-camera pose
  - replay-export head-camera pose for the same active frame
  - per-arm target object name
  - target pose, current evaluated pose, current TCP pose
  - target/current xyz deltas
  - planned-object/actual-object xyz deltas
  - target-object and current-object xyz deltas
- Added automatic PNG chart export under `output_dir/analysis_plots/`:
  - `{left,right}_target_current_error_vs_time.png`
  - `{left,right}_object_distance_vs_time.png`
- The new charts are generated from `debug_execution_metrics.jsonl` and focus on time-vs-error inspection, especially the relationship between target/current pose error and object-relative distances.
- This change is meant to make coordinate-frame and object-follow debugging easier than reading only stage-level `attempt_history` summaries.
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`

- Added `--candidate_target_local_x_offset_m` to the AnyGrasp planner and batch wrapper. This applies an explicit local-`+X` translation to each imported AnyGrasp target pose before visualization and planning, so the workflow can switch cleanly between "raw candidate behaves like wrist/endlink" and "planner input should represent fingertip TCP" without rewriting the TCP->endlink chain.
- Extended `plan_summary.json` so selected and ranked candidates now record both `raw_pose_world_wxyz` and the final planning/visualization `pose_world_wxyz`, plus the top-level `candidate_target_local_x_offset_m` used for the run.
- Added paired analysis docs:
  - `agent-read/V1.11_anygrasp_target_pose_offset.md`
  - `agent-read/V1.11_anygrasp_target_pose_offset_ZH.md`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`

- Added `--candidate_keep_camera_up` and `--candidate_camera_top_axis` to the AnyGrasp planner and batch wrapper.
- The initial camera-up rule over-constrained roll by trying to align the configured top axis too aggressively with world up. This could distort sideward grasps visually even though the forward axis stayed fixed.
- Refined the camera-up rule so it now keeps local gripper `+X` fixed and only chooses between the original orientation and a `180`-degree roll flip around local `+X`, whichever places the configured top axis more upward. This matches the intended semantics: keep the camera on the upper side overall, not pin the gripper to one exact roll.
- Extended `plan_summary.json` to record:
  - `candidate_keep_camera_up`
  - `candidate_camera_top_axis`
  - per-candidate `top_axis_up_dot`
  - `original_top_axis_up_dot`
  - `camera_up_flip_applied`
  - `forward_axis_change_deg`
- Added paired analysis docs:
  - `agent-read/V1.10_anygrasp_camera_up_rule.md`
  - `agent-read/V1.10_anygrasp_camera_up_rule_ZH.md`
- Documented the distinction between recent behavior-changing commits (`d877396`, `830f057`) and recent debug-only commits (`27a4bc9`, `6a0974f`) so future debugging can separate motion changes from pure instrumentation.
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python - <<'PY' ... constrain_roll_keep_top_axis_up(...) ... PY`

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
- Added `--object_mesh_override NAME=/abs/path/to/mesh.obj` to the AnyGrasp planner and batch wrapper so broken replay meshes can be swapped at runtime without regenerating `multi_object_world_poses.npz`. This is intended for cases like `cup.obj` whose `.mtl` references invalid texture paths.
- Validation: `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`
- Added `--save_pose_debug` and `--pose_debug_print_frames` to `render_multi_object_pose_r1_npz.py`. Multi-object replay can now dump per-frame `robot_base_pose_world_wxyz`, `head_camera_pose_world_wxyz`, left/right TCP poses, raw object `pose_cam_matrix`, and converted object `pose_world_wxyz` into `pose_debug.jsonl` for coordinate-frame debugging.
- Validation: `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/render_multi_object_pose_r1_npz.py`
- Extended multi-object replay export so `multi_object_world_poses.npz` now also stores `head_camera_pose_world_wxyz` for each exported frame. Added planner-side `--save_pose_debug` support so `plan_anygrasp_keyframes_r1.py` can dump current planner head-camera pose, left/right TCP poses, current object actor poses, and replay-export head-camera poses into `pose_debug.jsonl`.
- Validation: `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/render_multi_object_pose_r1_npz.py code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-23

- Updated `code_painting/render_anygrasp_ranked_preview.py` again so old replay exports without `head_camera_pose_world_wxyz` no longer hard-fail.
- The script now falls back to a fixed head-camera world pose computed from:
  - `robot_config`
  - fixed `torso_qpos`
  - `head_camera_local_pos`
  - `head_camera_local_quat_wxyz`
- Verified on a real replay directory:
  - the script computed a fallback head pose and completed frame rendering instead of raising the previous replay-export error
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python /home/zaijia001/ssd/RoboTwin/code_painting/render_anygrasp_ranked_preview.py --anygrasp_dir /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results/d_pour_blue_1 --replay_dir /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue/d_pour_blue_1 --hand_npz /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/hand_detections_1.npz --base_image_dir /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue/d_pour_blue_1/head_anygrasp_frames --base_image_mode raw --output_dir /tmp/anygrasp_ranked_preview_real_fallback --frames 1 --top_k 2 --left_target_object cup --right_target_object bottle --draw_grasp_boxes 0`

- Updated `code_painting/render_anygrasp_ranked_preview.py` from a raw-score-only preview tool into a staged candidate-inspection tool.
- The script now supports:
  - replay-backed left/right target-object filtering
  - per-arm orientation-only ranking after object filtering
  - fused ranking with `0.5 * anygrasp_score + 0.5 * orientation_score` by default
  - combined-only preview outputs instead of saving separate left/right images
  - terminal count logs before/after object filtering for each frame
- Added a replay-export prerequisite note: object filtering requires `multi_object_world_poses.npz` to contain `head_camera_pose_world_wxyz`.
- Updated:
  - `agent-read/V1.13_anygrasp_candidate_ranking_logic.md`
  - `agent-read/V1.13_anygrasp_candidate_ranking_logic_ZH.md`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/render_anygrasp_ranked_preview.py`

- Added paired ranking-logic notes for the AnyGrasp candidate pipeline:
  - `agent-read/V1.13_anygrasp_candidate_ranking_logic.md`
  - `agent-read/V1.13_anygrasp_candidate_ranking_logic_ZH.md`
- Documented the exact code path behind candidate ranking and preview generation, including:
  - the wrapper role of `code_painting/run_plan_anygrasp_keyframes_r1_batch.sh`
  - the raw-score-only visualization semantics of `code_painting/render_anygrasp_ranked_preview.py`
  - the actual planner-side ranking and cross-keyframe selection logic in `code_painting/plan_anygrasp_keyframes_r1.py`
  - the distinction between `ranked_candidates_per_frame`, `all_candidates_per_frame`, manual overrides, and final selected keyframes
- Validation:
  - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- agent-read/V1.13_anygrasp_candidate_ranking_logic.md agent-read/V1.13_anygrasp_candidate_ranking_logic_ZH.md agent-read/CHANGELOG.md`

- Added `agent-read/V1.12_object_and_gripper_replay_logic_ZH.md` and `agent-read/V1.12_object_and_gripper_replay_logic.md` to document the current combined AnyGrasp keyframe execution plus object-track replay logic.
- Updated selected-keyframe camera-up behavior in `code_painting/plan_anygrasp_keyframes_r1.py`: keyframe 1 still resolves the local-`+X` 180-degree roll ambiguity using the upward-facing rule, while later keyframes now choose the equivalent roll variant with the smaller rotation change relative to the previous selected keyframe. This prevents large extra roll spins between keyframe 1 and keyframe 22.
- Added `camera_up_selection_mode` to selected-candidate debug / summary outputs so it is explicit whether a keyframe used `keyframe1_keep_up`, `follow_previous_base`, or `follow_previous_flip180`.
- Updated `agent-read/ik_analyze/anygrasp_keyframe_ik_planning_analysis.md` and `.en.md` with the new sequential camera-up rule.
- Fixed a regression in the sequential keyframe camera-up rule: the new follow-previous branch referenced `rotation_distance_deg_matrix(...)`, but the file only defines `rotation_distance_deg(...)`. The roll-continuity path now uses the existing helper and no longer crashes during keyframe selection.
- Validation: `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py`
