## 2026-06-02 00:00:00 +08

- Fixed Mode O viewer environment forwarding:
  - `run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh` now passes `--gpu -1` when `--viewer` is enabled.
  - `plan_first_frame_foundation_pick_diverse_bottles.py` now removes `CUDA_VISIBLE_DEVICES` when `--enable_viewer 1`, so it does not undo the wrapper's unset.
  - Added/updated section O viewer probe notes.
- Validation:
  - `DISPLAY=:1.0 ... --viewer --viewer_wait_at_end 0 ...` successfully created the SAPIEN viewer and logged `CUDA_VISIBLE_DEVICES=None`.
- Added section O to `COMMAND_LIBRARY.zh.md`:
  - First-frame FoundationPose direct-strategy `pick_diverse_bottles` comparison experiment.
  - Recommended command: `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 0 --continue_on_error --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/first_frame_foundation_smoke`
- Added command entrypoint:
  - `code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh`
  - Important parameters: `--foundation_frame`, `--grasp_surface_retreat_m`, `--approach_offset_m`, `--place_z_mode`, `--lift_m`, and `--viewer`.
- Added core script:
  - `code_painting/plan_first_frame_foundation_pick_diverse_bottles.py`
  - It generates `plan_summary_first_frame_foundation.json` and invokes the existing Piper planner with `--reuse_plan_summary_json`.
- Validation:
  - Python `py_compile` passed.
  - wrapper `bash -n` passed.
  - `--ids 0 --dry_run` resolved paths.
  - The no-viewer `pick_diverse_bottles id0` smoke run completed; pregrasp reached, grasp did not reach for both arms, and the default safety gate skipped close/action.

## 2026-05-28 23:30:00 +08

- Updated `COMMAND_LIBRARY.zh.md` L15.11:
  - Added viewer/no-viewer debug commands for `--execute_partial_cartesian_plan`.
  - Documented that the flag only applies to `cartesian_interp_ik`, executes the solved waypoint prefix, and does not count as reached.
  - Recorded the current position-priority/orientation-priority IK behavior and possible future fallback strategy.
- Validation:
  - The L15.11 bash blocks passed `bash -n`.

## 2026-05-28 23:15:00 +08

- Updated `COMMAND_LIBRARY.zh.md` L15.9/L15.10:
  - L15.9 main preview command now uses `--candidate_target_local_x_offset_m 0.0`.
  - L15.10 is now the offset -5cm comparison command, writing to `anygrasp_h2o_preview_d435_offset_minus_5cm_compare`.
  - Added documentation for the new `planner_selected_orientation_rank1.png` output.
- Validation:
  - The L15.9/L15.10 bash blocks passed `bash -n`.

## 2026-05-28 23:05:00 +08

- Added `COMMAND_LIBRARY.zh.md` L15.10:
  - Raw/no-offset D435 preview comparison command with `--candidate_target_local_x_offset_m 0.0`, writing to `anygrasp_h2o_preview_d435_no_offset_compare`.
  - Added a summary comparison script for `translation_cam`, `visual_translation_cam`, and `translation_world`.
- Validation:
  - The L15.10 bash blocks passed `bash -n`.

## 2026-05-28 22:55:00 +08

- Added `COMMAND_LIBRARY.zh.md` L15.9:
  - Appended copy-safe three-step commands at the end of the file: regenerate D435 previews/summaries, run no-viewer replay, and run viewer target-visualization replay.
  - Step 1 uses `bash <<'BASH'` to avoid zsh line-wrap issues such as `OUT_ROOT=/ home/...`, fallback to the default `replay_m_obj_pose_d_pour_blue_norobot`, or `--candidate_target_local_x_offset_m: command not found`.
- Validation:
  - The L15.9 bash blocks passed `bash -n`.

## 2026-05-28 22:40:00 +08

- Added `COMMAND_LIBRARY.zh.md` L15.8:
  - Documents the D435 preview vs planner target mapping: `translation_cam` is the raw AnyGrasp candidate, while `visual_translation_cam/translation_world` are the offset visual/planner targets.
  - Adds a six-task D435 preview regeneration command so future `anygrasp_h2o_preview_d435` images match the summary/planner target convention.
  - Adds offset-fix no-viewer replay and viewer target-visualization commands.
- Validation:
  - The L15.8 command blocks passed `bash -n`.
  - The `pick_diverse_bottles id0` offset-fix debug preview generated successfully.

## 2026-05-28 19:20:00 +08

- Added `COMMAND_LIBRARY.zh.md` L15.6:
  - No-viewer command for the first five D435 summaries of all six tasks.
  - Viewer command for the first five D435 summaries of all six tasks, using script `--viewer` and a separate output root.
  - First-keyframe debug command: `--debug_stop_after_keyframe1` executes only init -> pregrasp -> grasp, without closing the gripper or entering keyframe 2.
- Updated `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`:
  - Added `--viewer`, `--output_root`, `--debug_stop_after_keyframe1`, `--trajectory_mode`, `--cartesian_auto_step_m`, `--joint_interp_waypoints`, `--replan_attempts`, `--allow_partial_dual_stage`, and `--print_pose_every`.
  - Now passes `--require_keyframe1_reached_before_close 1` by default so the gripper does not close when keyframe 1 is not reached.
  - Restored the default execution cadence to `execute_interp_steps=24`, `joint_command_scene_steps=10`, `settle_steps=30`, and `joint_target_wait_steps=25`.
- Added planner parameter:
  - `--debug_stop_after_keyframe1`
  - Purpose: isolate first-keyframe reachability and determine whether failure occurs during keyframe-1 Cartesian waypoint IK.
  - `--require_keyframe1_reached_before_close`
  - Purpose: match the old R1/V7 behavioral intent by skipping gripper close when keyframe 1 is not reached.
  - `--print_execution_pose_every`
  - Purpose: print TCP/EE world positions during execution to confirm whether waypoint execution is actually changing pose.

## 2026-05-28 18:30:00 +08

- Added `COMMAND_LIBRARY.zh.md` L15.5:
  - Single D435 viewer debug command for `stack_cups id0`.
  - Runs `probe_sapien_viewer.py` first, then uses `unset CUDA_VISIBLE_DEVICES` plus `--enable_viewer 1` for the SAPIEN viewer.
  - Records the per-arm keyframes for this id: right 51/106, left 139/195.
  - Documents the difference between planner `rank_previews` and the J1.1 D435 preview: the former is a SAPIEN 3D render and includes the `candidate_target_local_x_offset_m=-0.05` 5 cm TCP compensation; `approach_offset_m=0.12` only affects pregrasp.
  - The J1.1 source previews were copied to `anygrasp_plan_keyframes_piper_d435_v1/stack_cups/foundation_input_0/preview_compare_d435/`.
  - Added the viewer probe failure condition: if `DISPLAY=` is empty and SAPIEN reports `Renderer does not support display`, run from a graphical terminal or a correctly forwarded X11/Wayland session.
- Updated `agent-read/COMMANDS/piper_anygrasp_keyframes.en.md`:
  - Added the L15.5 viewer entrypoint note.

## 2026-05-28 09:30:00 +08

- Updated the L15.1 explanation in `COMMAND_LIBRARY.zh.md`:
  - Described `dual_stage_require_all_plans` and `require_keyframe1_reached_before_action` as explicit constraints that match the old V7 behavioral intent, not as old V7 parameters.
- Added `COMMAND_LIBRARY.zh.md` L15.2:
  - No-viewer batch command: keeps `CUDA_VISIBLE_DEVICES=${GPU}` for stable id0-id10 batch processing.
  - Viewer single-id debug command: runs `unset CUDA_VISIBLE_DEVICES` first so SAPIEN viewer can see the display GPU.
  - Added a minimal `probe_sapien_viewer.py` viewer probe command.

## 2026-05-27 22:10:00 +08

- Added `COMMAND_LIBRARY.zh.md` L5.1:
  - Six-task original human head + pure replay action/wrist processed HDF5 command.
  - Prompts use the user's full six-task English descriptions.
- Added `COMMAND_LIBRARY.zh.md` L5.2:
  - Original human head + D435 pure replay action/wrist command for the currently available new-three-task outputs.
  - Documents why L5.1 skips those tasks: `h2_pure/<TASK>/id<ID>_z005` is absent, while `h2_pure_d435/<TASK>/id<ID>_d435_z005` exists.
- Added `COMMAND_LIBRARY.zh.md` I1.1/I3.5/L8.1:
  - I1.1: Stage-1 human-hand removal backgrounds for the new three tasks.
  - I3.5: D435 visible-reinit robot repaint for the new three tasks.
  - L8.1: convert I3.5 outputs to processed HDF5.
- Added `COMMAND_LIBRARY.zh.md` L0/L10.5:
  - L0: describes run order for the human, robot replay, and AnyGrasp replay data pipelines.
  - L10.5: converts L5.2 new-three-task `human_head_pure_d435_action` outputs to LeRobot cache.
- Added `COMMAND_LIBRARY.zh.md` L11.1:
  - Added 25-episode subset commands for three default-wide robot replay datasets and three AnyGrasp robot datasets.
  - Added explicit zip commands: `robot_replay_3task_25ep.zip` and `robot_anygrasp_3task_25ep.zip`, avoiding accidental inclusion of unrelated historical `*_25ep` subsets.
  - Added an output check command for the six `_25ep` datasets via `meta/info.json`.
- Added `COMMAND_LIBRARY.zh.md` L11.2:
  - Extended 25-episode subset commands to six tasks: `pick_diverse_bottles`, `place_bread_basket`, `stack_cups`, `handover_bottle`, `pnp_bread`, and `pnp_tray`.
  - Added `robot_replay_6task_25ep.zip` and `robot_anygrasp_6task_25ep.zip`.
- Added `COMMAND_LIBRARY.zh.md` L11.3:
  - Documented that task prompts should be set in processed data `episode_*/instructions.json`; the current `convert_aloha_data_to_lerobot_R1.py --task` does not override that text.
- Added `COMMAND_LIBRARY.zh.md` L6.1/L9.1/L10.4:
  - L6.1: six-task `pure_repaint` processed HDF5.
  - L9.1: six-task `anygrasp_repaint` processed HDF5.
  - L10.4: convert human_head_pure_action, pure_repaint, and anygrasp_repaint processed HDF5 datasets to LeRobot caches, which are the source datasets consumed by L11/L11.2.
- Validation:
  - Extracted the new L5.1/L5.2/L6.1/L9.1/L10.4/L10.5/L11.1/L11.2/L11.3/I1.1/I3.5/L8.1 bash blocks and verified them with `bash -n`.

## 2026-05-27 21:45:00 +08

- Added `COMMAND_LIBRARY.zh.md` L15.1:
  - Piper AnyGrasp id0-id10 viewer visualization command.
  - Execution cadence matches old V7: `--execute_interp_steps 24`, `--joint_command_scene_steps 10`, `--settle_steps 30`, `--joint_target_wait_steps 25`.
  - Adds `--require_keyframe1_reached_before_action 1`, so action is skipped when first-keyframe grasp is not reached.
- Added CLI:
  - `--require_keyframe1_reached_before_action`

## 2026-05-27 21:10:00 +08

- Updated the Piper AnyGrasp batch run command:
  - Added L15 to `COMMAND_LIBRARY.zh.md` with a `pick_diverse_bottles` id0-id10 command.
  - L15 no longer passes `--keyframes 38 78`; it relies on `--reuse_preview_frame_mode annotated_json_keyframes` to read each id's manual keyframes.
  - Added `--dual_stage_require_all_plans 1` so a dual-arm stage executes only when both left and right plans succeed.
  - Documented the difference between `settle_steps/joint_target_wait_steps`, IK failure, and visible video hold frames.

## 2026-05-27 14:00:00 +08

- Added command library section: C1.2 in `COMMAND_LIBRARY.zh.md`.
- Added content: D435-intrinsics FoundationPose replay commands for six H2O tasks.
- Key parameters:
  - `--image_width 640`
  - `--image_height 480`
  - `--fovy_deg 42.499880046655484`
  - `--camera_cv_axis_mode legacy_r1`
  - `--robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json`
- Output convention: `/home/zaijia001/ssd/data/piper/hand/<TASK>/foundation_replay_d435`.
- Usage note: these commands align the FoundationPose object replay head view with the later D435 robot pure replay intrinsics.

## 2026-05-26 21:30:00 +08

- Updated the Piper AnyGrasp two-keyframe planning command:
  - Added an id0-id10 batch command to `COMMAND_LIBRARY.zh.md` L14.
  - Recommended debug parameters now use `--urdfik_cartesian_interp_auto_step_m 0.02`, `--urdfik_max_position_threshold_m 0.02`, and `--urdfik_max_rotation_threshold_rad 0.12`.
  - Position-first debugging uses `--reach_rot_tol_deg 180`; tighten the rotation tolerance only after position reachability is confirmed.
  - Documented that `--debug_visualize_ik_waypoints` is a 0/1 switch and should be set to `1`.
  - Added `--head_only 0 --third_person_view 1 --vscode_compatible_video 1` so head/third outputs can be previewed directly in VS Code.
- Related code:
  - `code_painting/plan_anygrasp_keyframes_r1.py`
  - `code_painting/render_hand_retarget_piper_dual_npz_urdfik.py`
  - `code_painting/urdfik.py`

## 2026-05-25 16:10:00 +08

- Follow-up fix:
  - L1 is now explicitly "original human head + pure replay action/wrist" and includes a command using `--head-dir-template '.' --head-video-name 'rgb_{id}.mp4'`.
  - L1/L2/L3 now all pass `--review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json` to exclude `reject/discard/bad`.
  - The related conversion scripts now support `hand_keyframes_all.json` status filtering and `{id}` video filename templates.
  - Validation: the updated L1/L2/L3 commands pass `bash -n`; single-id temporary conversions passed for L1 and L2.
- Follow-up addition:
  - Added `COMMAND_LIBRARY.zh.md` L5/L6/L7.
  - L5/L6 list copyable conversion commands for `pick_diverse_bottles`, `place_bread_basket`, and `stack_cups`.
  - L7 adds episode-count checks, HDF5 structure checks, and review mp4 visualization via `visualize_processed_hdf5_episode.py`.
  - Added visualization entrypoint: `policy/pi0/scripts/visualize_processed_hdf5_episode.py`.
- Continued additions:
  - Added `COMMAND_LIBRARY.zh.md` L8: D435 visible-reinit head + D435 pure replay action/wrist, with separate commands for the three tasks.
  - Added `COMMAND_LIBRARY.zh.md` L9: AnyGrasp repaint head + planner action/wrist, with separate commands for the three tasks and a note that planner wrist videos must exist first.
  - Validation: representative L8/L9 commands pass `bash -n`.
- Continued additions:
  - Added `COMMAND_LIBRARY.zh.md` L10 with HDF5 -> LeRobot conversion commands for 3 usable modes x 3 tasks.
  - L10 uses `examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --use-wrist --mode video`, not `convert_aloha_data_to_lerobot_robotwin.py`, because the current H2O processed HDF5 uses `/observations/state` while the latter's normal branch reads `/observations/qpos`.
  - L10 documents the L6/L8 distinction: L6 is the default wide replay/repaint path, while L8 is the D435 visible-reinit replay/repaint path.
- Updated `COMMAND_LIBRARY.zh.md` with `L. pi0 training data organization: original human, pure replay, AnyGrasp replay`.
- Added command notes for:
  - Original human-data paths and existing `policy/pi0/processed_data` inspection.
  - Non-AnyGrasp pure replay: `process_repainted_headcam_with_wrist.py` reads the repainted head video, `world_targets_and_status.npz`, and left/right wrist replay videos to emit pi0 HDF5.
  - AnyGrasp planner replay: `process_repainted_planner_outputs.py` reads the repainted head video, `pose_debug.jsonl`, and left/right planner wrist videos.
- Documented that the current AnyGrasp H2O planner outputs are missing `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4`; L3 requires those inputs before it can reliably produce training data.
- Added paired command documentation:
  - `agent-read/COMMANDS/pi0_h2o_training_data.zh.md`
  - `agent-read/COMMANDS/pi0_h2o_training_data.en.md`

## 2026-05-22 16:20:00 +08

- Fixed K1 Piper AnyGrasp planner argument compatibility:
  - `plan_anygrasp_keyframes_r1.py` now defines and forwards `--debug_visualize_cameras`, `--debug_camera_axis_length`, `--debug_camera_axis_thickness`, and `--target_local_forward_retreat_m`.
  - `plan_anygrasp_keyframes_r1_batch.py` now accepts the same arguments and forwards them to the single-video planner.
  - `COMMAND_LIBRARY.zh.md` K1 heredoc command explicitly includes the default values:
    - `--debug_visualize_cameras 0`
    - `--debug_camera_axis_length 0.16`
    - `--debug_camera_axis_thickness 0.006`
    - `--target_local_forward_retreat_m 0.0`
- Cause: the Piper planner ultimately inherits `HandRetargetR1Renderer`, whose constructor gained these arguments; the AnyGrasp planner path had not been synchronized and failed during initialization.

## 2026-05-22 15:55:00 +08

- Updated `COMMAND_LIBRARY.zh.md` K1:
  - replaced the long `bash -lc '...'` one-liner with a heredoc that writes `/tmp/run_h2o_k1_preview_resume.sh` and then runs it.
  - avoids zsh continuation prompts caused by curly quotes `‘ ’`, and avoids relying on zsh support for bash's `mapfile` builtin.

## 2026-05-22 15:45:00 +08

- Updated `COMMAND_LIBRARY.zh.md` K1:
  - the planning command no longer scans every AnyGrasp directory under the task root.
  - it now derives usable ids from `anygrasp_h2o_preview/<TASK>/foundation_input_<ID>/summary.json` and passes them through `--ids`.
  - added `--skip_existing 1 --continue_on_error 1` for safe partial-task resume and to avoid repeated `missing_preview_summary` reports for ids that have not gone through K0.2.

## 2026-05-22 15:25:00 +08

- Updated the H2O manual keyframe annotator:
  - `Space` still annotates global keyframes into `keyframes`
  - `l`/`L` annotates left-hand keyframes into `left_keyframes`
  - `r` annotates right-hand keyframes into `right_keyframes`
  - the old replay shortcut moved from `r` to `R` to avoid conflict
- Updated command library:
  - `COMMAND_LIBRARY.zh.md` K0 now documents separate left/right hand keyframe fields.

## 2026-05-22 15:05:00 +08

- Added an H2O AnyGrasp manual keyframe annotation entrypoint:
  - `code_painting/annotate_hand_keyframes.py`
  - annotates `hand_vis_gripper_*.mp4` videos interactively and normalizes JSON keys to `hand_vis_<id>.mp4`
  - supports pressing `d` to mark `status=reject` for discarded bad videos/detections
- Updated preview batch:
  - `code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh`
  - skips ids with `status=reject/discard/bad` or fewer than two annotated keyframes
- Updated command library:
  - `COMMAND_LIBRARY.zh.md` K0 now documents the interactive annotator workflow and keeps the optional physical move command for discarded data.

## 2026-05-06 11:40:00 +08

- Added explicit URDFIK execution-step CLI controls:
  - `--execute_waypoint_scene_steps` (scene steps per trajectory waypoint, default 1)
  - `--execute_settle_scene_steps` (extra settle steps per frame, default 4)
  - `--urdfik_joint_interp_waypoints` (joint_interp waypoint count, default 2)
- Applied to:
  - `render_hand_retarget_piper_dual_npz_urdfik_main.py`
  - `render_hand_retarget_r1_npz_urdfik.py`
- Added runtime print: `[execute-steps] ...` for easier verification.
- Synced docs: `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` (D6 execution-step probe command).

## 2026-04-29 17:10:00 +08

- Updated command tool: `run_piper_gripper_standard_pose_guess.sh`
  - added default forward/up offsets (`TARGET_DY=0.1 TARGET_DZ=0.1`)
  - outputs are now images-only (`zed/third` PNG) + `index.csv` + `world_targets_and_status.npz`
  - emits `left_status/right_status` to diagnose IK reachability vs orientation-definition issues
- Synced command docs: `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` D6
  - clarified images-only usage
  - clarified that frequent `Fail` is typically IK reachability, not necessarily wrong orientation semantics

## 2026-04-29 16:30:00 +08

- Added command entry:
  - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_gripper_standard_pose_guess.sh ...`
- Purpose:
  - generate a single-folder Piper gripper “standard orientation guess board” under `.../board/`
  - emit `index.csv` for manual per-image semantic feedback
- Related code:
  - `code_painting/run_piper_gripper_standard_pose_guess.sh`
  - `code_painting/run_piper_gripper_orientation_guess_board.sh`
- Synced command library update: `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` (new D6)

## 2026-04-29 16:00:00 +08

- Updated HaMeR commands in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`:
  - switched GPU commands from `hamer-r1` to `hamer-r1-gpu`
  - added `unset LD_LIBRARY_PATH` to match the original README convention
- Added a quick detection-count check command:
  - reads `left_hand_detected/right_hand_detected` directly from `hand_detections_*.npz`
- Added a debug baseline command:
  - `CUDA_VISIBLE_DEVICES=2 + hamer-r1-gpu + video_id=0 + --no_visualize`

## 2026-04-27 17:35:00 +08

- Batch entry behind `run_multi_object_pose_r1_npz_batch.sh` now accepts:
  - `--save_pose_debug 1`
- For Piper calibrated head-cam replay, prefer:
  - `--object star_fruit=/.../star.obj` (avoid `star fruit=`, which can mismatch object-folder names)

## 2026-04-27 14:55:00 +08

- Added per-object replay commands to `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`:
  - `render_multi_object_pose_r1_npz_batch.py --objects pear ...`
  - `render_multi_object_pose_r1_npz_batch.py --objects star_fruit ...`
- Added `--save_pose_debug 1` to the FoundationPose -> RoboTwin batch replay command.

## 2026-04-27 14:20:00 +08

- Updated the FoundationPose multi-object prompt for star-shaped fruit from `star` to `star fruit` to reduce DINO miss-detections.
- Synced in:
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md`
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline.md`
  - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`

## 2026-04-27 13:55:00 +08

- Added FoundationPose preparation and execution commands for Piper:
  - `conda run -n hamer-r1 python /home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py ...`
  - `bash /home/zaijia001/FoundationPose/run_piper_star_pear_foundation.sh`
- Added pear+star multi-object detection command (`run_realr1_dino_sam_batch.py` with `--object pear=... --object star=...`).
- Updated `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` with FoundationPose stage and RoboTwin object-replay stage.

## 2026-04-24 15:20:00 +08

- Added cross-project hand-processing command doc:
  - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
- The doc uses a one-line-comment + one-line-command format and includes:
  - Piper -> HaMeR conversion
  - HaMeR single/batch detection
  - `ffplay` visualization checks
  - RoboTwin downstream replay command
- Related entry scripts:
  - `/home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py`
  - `/home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py`

## 2026-04-16 04:10:00 +08

- Updated `code_painting/pika/visualize_calibrated_piper_pika_scene.py`
- Updated `code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`
  - Added arguments:
    - `--viewer 1`
    - `--viewer-camera {overview,head}`
  - These scripts can now open interactive viewer windows.
- Updated command docs:
  - `agent-read/COMMANDS/pika_scene_commands.en.md`
  - `agent-read/COMMANDS/pika_scene_commands.zh.md`

## 2026-04-16 03:55:00 +08

- Updated `agent-read/COMMANDS/pika_scene_commands.en.md`
- Updated `agent-read/COMMANDS/pika_scene_commands.zh.md`
  - added explicit commands for exporting head-cam-only images
  - added explicit note that wrist views are not exported yet by calibrated-scene scripts

## 2026-04-16 03:40:00 +08

- Added `agent-read/COMMANDS/pika_scene_commands.en.md`
- Added `agent-read/COMMANDS/pika_scene_commands.zh.md`
  - one-line-comment + command format
  - includes viewer commands for the manual tabletop scene
  - notes that calibrated scene scripts are currently offscreen-only

## 2026-04-16 03:25:00 +08

- Added `code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`
  - Purpose:
    - reconstruct the calibrated scene using version B alignment
  - Key differences from the previous calibrated scene script:
    - removes the +90deg anchor rotation
    - treats left/right spread as world y
    - rotates the table convention by 90deg
  - Example:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`

## 2026-04-16 03:10:00 +08

- Added `code_painting/pika/visualize_calibrated_piper_pika_scene.py`
  - Purpose:
    - reconstruct a simulated scene from the real calibration bundle
  - Inputs:
    - `robot_config_PiperPika_agx_dual_table.json`
    - `calibration_bundle_try2.json`
  - Outputs:
    - `code_painting/pika/output_calibrated_scene/`
  - Example:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/pika/visualize_calibrated_piper_pika_scene.py`

## 2026-04-16 02:50:00 +08

- Updated `robot_config_PiperPika_agx_dual_table.json`
  - moved the edge-side base position from `y = -0.60` (outside table) to `y = -0.30` (edge-mounted on tabletop)
- Reused `code_painting/visualize_piper_pika_agx_dual_table.py`
  - exported edge-mounted preview:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode oblique --output-dir code_painting/output_piper_pika_agx_dual_table_edge_mount --image-name piper_pika_agx_dual_table_edge_mount.png --video-name piper_pika_agx_dual_table_edge_mount.mp4`

## 2026-04-16 02:35:00 +08

- Updated `robot_config_PiperPika_agx_dual_table.json`
  - corrected base orientation to face the tabletop from the long-edge side
  - quaternion: `[0.70710678, 0.0, 0.0, 0.70710678]`
- Updated `code_painting/visualize_piper_pika_agx_dual_table.py`
  - corrected `--camera-mode oblique` to a proper behind-the-robots viewpoint
  - refreshed top-down output after quaternion fix
  - examples:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode oblique --output-dir code_painting/output_piper_pika_agx_dual_table_oblique_fixed --image-name piper_pika_agx_dual_table_oblique_fixed.png --video-name piper_pika_agx_dual_table_oblique_fixed.mp4`
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode top_down --output-dir code_painting/output_piper_pika_agx_dual_table_topdown_fixed --image-name piper_pika_agx_dual_table_topdown_fixed.png --video-name piper_pika_agx_dual_table_topdown_fixed.mp4`

## 2026-04-16 02:20:00 +08

- Updated `code_painting/visualize_piper_pika_agx_dual_table.py`
  - clarified the one-side long-edge dual-arm layout:
    - bases at x = ±0.30 m
    - base spacing = 0.60 m
  - added camera selection:
    - `--camera-mode top_down`
    - `--camera-mode oblique`
  - top-down example:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode top_down --output-dir code_painting/output_piper_pika_agx_dual_table_topdown --image-name piper_pika_agx_dual_table_topdown.png --video-name piper_pika_agx_dual_table_topdown.mp4`

## 2026-04-16 02:05:00 +08

- Added `code_painting/visualize_piper_pika_agx_dual_table.py`
  - Purpose:
    - preview the newer colored Piper+Pika dual-arm layout on a 120x60x75 cm table
  - Example:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1`
- Added `robot_config_PiperPika_agx_dual_table.json`
  - UR-style symmetric split using `embodiment_dis = 0.60`
  - effective base poses:
    - left `[-0.30, -0.60, 0.75]`
    - right `[0.30, -0.60, 0.75]`

## 2026-04-16 01:45:00 +08

- Added a color-statistics analysis step (diagnosis only, no source modification)
  - compared exported preview PNGs against the Piper DAE diffuse reference color
  - used approximate foreground color statistics and ΔE76 distances for documentation

## 2026-04-16 01:45:00 +08

- Added `code_painting/visualize_agx_arm_sim_source.py`
  - Purpose:
    - preview the new `agx_arm_sim` Piper/Pika source routing
  - Targets:
    - `--target piper_only`
    - `--target pika_only`
    - `--target piper_pika_combo`
    - `--target all`
  - Example:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_agx_arm_sim_source.py --target all --output-root code_painting/output_agx_arm_sim_preview --video-frames 36 --fps 12`

## 2026-04-16 01:30:00 +08

- Updated `code_painting/visualize_original_source_urdfs.py`
  - Added lighting preset argument:
    - `--lighting {bright,dark}`
  - Dark stepwise validation example:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_original_source_urdfs.py --target both --lighting dark --output-root code_painting/output_original_source_urdf_preview_dark --video-frames 36 --fps 12`

## 2026-04-16 01:15:00 +08

- Updated `code_painting/visualize_piper_pika_single.py`
  - Added lighting preset argument:
    - `--lighting {bright,dark}`
  - Example dark-preview command:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --lighting dark --output-dir code_painting/output_piper_pika_preview_dark --image-name piper_pika_dark.png --video-name piper_pika_dark.mp4 --video-frames 36 --fps 12`

## 2026-04-16 01:00:00 +08

- Added original-source URDF preview command
  - Entry point:
    - `code_painting/visualize_original_source_urdfs.py`
  - Purpose:
    - preview the original Piper arm URDF and original Pika gripper URDF directly from the download folders
  - Main arguments:
    - `--target {piper_arm,pika_gripper,both}`
    - `--output-root`
    - `--video-frames`
    - `--fps`
  - Typical usage:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_original_source_urdfs.py --target both --output-root code_painting/output_original_source_urdf_preview --video-frames 30 --fps 12`

## 2026-04-16 00:45:00 +08

- Reused `code_painting/visualize_piper_pika_single.py` to export a DAE-based preview variant
  - Example command:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --output-dir code_painting/output_piper_pika_preview_dae --video-frames 24 --fps 12`
  - Purpose:
    - verify the arm appearance after switching visual meshes from STL to DAE

## 2026-04-16 00:30:00 +08

- Updated `code_painting/visualize_piper_pika_single.py`
  - Added preview-export support:
    - still image export
    - short mp4 export
  - Useful arguments:
    - `--offscreen-only`
    - `--save-image`
    - `--save-video`
    - `--video-frames`
    - `--fps`
    - `--output-dir`
  - Default exported files:
    - `code_painting/output_piper_pika_preview/piper_pika_preview.png`
    - `code_painting/output_piper_pika_preview/piper_pika_preview.mp4`

## 2026-04-16 00:00:00 +08

- Added standalone piper_pika visualization command
  - Entry point:
    - `code_painting/visualize_piper_pika_single.py`
  - Purpose:
    - load and visualize the new assembled `assets/embodiments/piper_pika/piper_pika.urdf`
  - Key arguments:
    - `--urdf`
    - `--offscreen-only`
  - Typical usage:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py`
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1`

## 2026-04-14 12:30:00 +08

- Updated `code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
  - Purpose:
    - prevent the reused R1 batch parser from injecting `robot_config_R1.json` into Piper V2 runs
  - New enforced defaults when absent from user CLI:
    - `--robot_config /home/zaijia001/ssd/RoboTwin/robot_config_Piper_dual_v2.json`
    - `--head_camera_local_quat_wxyz 1.0 0.0 0.0 0.0`
    - `--head_camera_local_pos 0.0 0.0 0.0`
  - Result:
    - printed batch command now points to the Piper V2 robot config instead of the R1 config

## 2026-04-14 12:00:00 +08

- Added true Piper V2 batch planning command: `bash code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh ...`
  - Entry points:
    - `code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh`
    - `code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
    - `code_painting/plan_anygrasp_keyframes_piper_v2.py`
  - Supporting files:
    - `code_painting/replay_piper_dual_h5.py`
    - `code_painting/render_hand_retarget_piper_dual_npz_urdfik.py`
    - `robot_config_Piper_dual_v2.json`
  - Purpose:
    - provide a real Piper-oriented dual-single-arm setup modeled after the existing UR configuration style
    - keep left/right Piper bases separate in viewer/replay/URDFIK execution
  - Key behavior:
    - uses `dual_arm_embodied=false`
    - loads two Piper URDF instances from `assets/embodiments/piper/piper.urdf`
    - separates them with `embodiment_dis=0.80`
    - effective base poses are `[-0.4, -0.65, 0.72]` and `[0.4, -0.65, 0.72]`
  - Typical usage:
    - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh <anygrasp_root> <replay_root> <hand_dir> <output_root> --planner_backend urdfik ...`

## 2026-04-14 00:00:00 +08

- Added Piper batch planning command: `bash code_painting/run_plan_anygrasp_keyframes_piper_batch.sh ...`
  - Entry points:
    - `code_painting/run_plan_anygrasp_keyframes_piper_batch.sh`
    - `code_painting/plan_anygrasp_keyframes_piper_batch.py`
    - `code_painting/plan_anygrasp_keyframes_piper.py`
  - Purpose:
    - provide a Piper-oriented entry while keeping the original R1 planner code unchanged
    - reuse the existing AnyGrasp batch launcher / single-video planner flow
  - Important behavior:
    - defaults to `robot_config_Piper_dual.json`
    - defaults `--head_camera_local_quat_wxyz 1 0 0 0` unless user overrides it
    - when `--planner_backend urdfik`, uses `assets/embodiments/piper/piper.urdf`
  - Related config:
    - `robot_config_Piper_dual.json`
  - Typical usage:
    - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_batch.sh <anygrasp_root> <replay_root> <hand_dir> <output_root> --planner_backend urdfik ...`

## 2026-04-03 00:00:00 +08

- Added a smooth-handling note: `agent-read/smooth/README.en.md`
  - Related commands:
    - `bash code_painting/run_plan_anygrasp_keyframes_r1_batch.sh ...`
    - `bash code_painting/run_replay_pose_debug_smooth.sh ...`
    - `bash code_painting/batch_smooth_planner_outputs.sh`
  - Main arguments discussed:
    - `--urdfik_trajectory_mode cartesian_interp_ik`
    - `--urdfik_cartesian_interp_steps`
    - `--urdfik_cartesian_interp_auto_step_m`
    - `--settle_steps`
    - `--joint_target_wait_steps`
    - `--replan_until_reached`
    - `--replan_until_reached_max_attempts`
  - Purpose:
    - explain the difference between planner smoothing, settle/wait behavior, and post-hoc replay smoothing
    - record how to balance exported-video smoothness against final execution accuracy without changing code
  - Expanded this round with:
    - an evaluation of the strengthened waypoint-IK idea: fixed 1 cm EE sampling + previous-solution seed + joint jump-threshold rejection
    - a comparison between that idea and the current `cartesian_interp_ik` implementation
    - a ranked difficulty discussion for several continuity / smoothing improvement options
    - a V7-debug-specific analysis of the tradeoff among try/replan, `joint_target_wait_steps`, final accuracy, and exported-video smoothness
    - a new implemented diagnostic metric: lateral distance from the current point to the target forward axis
    - new output field: `lat_cm`
    - an additional future direction: an object-relative target adapter (`T_obj_hand_demo + Δ_robot`)

## 2026-03-27 00:20:00 +08

- Added raw-planner-v7 wrapper command: `bash run_planner_v7_repaint_review_pi0.sh`
  - Entry point:
    - `run_planner_v7_repaint_review_pi0.sh`
  - Purpose:
    - run repaint directly on `anygrasp_plan_keyframes_realoffset_batch_pure-v7`
    - manually review repaint videos
    - then convert them into pi0 / robotwin processed_data
  - Key paths:
    - planner root: `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v7`
    - repaint root: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_v7`
    - review json: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_v7/video_review.json`
    - processed_data: `/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-27-planner-v7`

- Added review-driven Step-2+Step-3+Step-4 controller command: `bash run_reviewed_smooth_repaint_pi0_pipeline.sh`
  - Entry point:
    - `run_reviewed_smooth_repaint_pi0_pipeline.sh`
  - Purpose:
    - extract ids labeled `y` from `video_review.json` (or include `m` by mode)
    - batch-run Step 2 smoothing, Step 3 smooth repaint, and Step 4 pi0 processing
  - Key arguments:
    - `TASK_NAME`
    - `REVIEW_JSON`
    - `REVIEW_MODE`
    - `RUN_SMOOTH`
    - `RUN_REPAINT`
    - `RUN_PI0`
    - `DRY_RUN`
    - `EXPERT_DATA_NUM`
    - `PI0_OUTPUT_DIR`
  - Key outputs:
    - `${SMOOTH_OUTPUT_ROOT}/${TASK_NAME}_${idx}/...`
    - `${REPAINT_OUTPUT_ROOT}/id_${idx}_head_cam_arm_gripper_cup_bottle_pad_target/target_with_original_head_cam_plan.mp4`
    - `${PI0_OUTPUT_DIR}/episode_*/episode_*.hdf5`

- Added smooth-bundle command: `bash code_painting/batch_smooth_planner_outputs.sh`
  - Entry point:
    - `code_painting/batch_smooth_planner_outputs.sh`
    - `code_painting/smooth_planner_outputs_from_pose_debug.py`
  - Purpose:
    - remove lingering frames and interpolate key states from the Step-1 planner outputs
  - Key arguments:
    - `INPUT_ROOT`
    - `OUTPUT_ROOT`
    - `INTERP_FACTOR`
    - `FPS`
    - `KEEP_HOVER_FRAMES_EVERY`
    - `DEDUP_POS_THRESH_M`
    - `DEDUP_ROT_THRESH_DEG`
    - `DEDUP_JOINT_THRESH_RAD`
    - `DEDUP_GRIPPER_THRESH`
  - Key outputs:
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/head_cam_plan.mp4`
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/left_wrist_cam_plan.mp4`
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/right_wrist_cam_plan.mp4`
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/pose_debug.jsonl`
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/smooth_summary.json`

## 2026-03-27 00:30:00 +08

- Added a planner-consistent pi0 data-conversion command: `process_repainted_planner_outputs.py`
  - Entry point:
    - `policy/pi0/scripts/process_repainted_planner_outputs.py`
  - Purpose:
    - generate `processed_data` HDF5 episodes from repaint planner head + planner wrist + planner pose_debug
  - Key arguments:
    - `--head-root`
    - `--head-dir-template`
    - `--head-video-name`
    - `--planner-root`
    - `--planner-dir-template`
    - `--left-wrist-video-name`
    - `--right-wrist-video-name`
    - `--pose-debug-name`
    - `--review-json`
    - `--review-mode`
    - `--ids`
    - `--ignore-ids`
  - Typical command:
    - `python scripts/process_repainted_planner_outputs.py d_pour_blue "pour water" 27 --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' --head-video-name target_with_original_head_cam_plan.mp4 --planner-root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3 --planner-dir-template 'd_pour_blue_{id}' --left-wrist-video-name left_wrist_cam_plan.mp4 --right-wrist-video-name right_wrist_cam_plan.mp4 --pose-debug-name pose_debug.jsonl --review-json /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json --review-mode strict --ignore-ids --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-27-planner`

## 2026-03-27 00:00:00 +08

- Added a new pi0 data-conversion command: `process_repainted_headcam_with_wrist.py`
  - Entry point:
    - `policy/pi0/scripts/process_repainted_headcam_with_wrist.py`
  - Purpose:
    - convert the newer head-cam repaint outputs plus retarget wrist replays into `processed_data` HDF5 episodes
  - Key arguments:
    - `--head-root`
    - `--head-dir-template`
    - `--head-video-name`
    - `--retarget-root`
    - `--retarget-dir-template`
    - `--review-json`
    - `--review-mode`
    - `--ids`
    - `--ignore-ids`
  - Typical command:
    - `python scripts/process_repainted_headcam_with_wrist.py d_pour_blue "pour water" 48 --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' --head-video-name target_with_original_head_cam_plan.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue --retarget-dir-template 'hand_detections_{id}' --ignore-ids`

## 2026-03-25 19:15:00 +08

- Added visual-only base-occluder parameters
  - Flags:
    - `--base_occluder_enable 0|1`
    - `--base_occluder_local_pos X Y Z`
    - `--base_occluder_half_size HX HY HZ`
    - `--base_occluder_color R G B`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/render_hand_retarget_r1_npz.py`
  - Purpose:
    - add a white occluder that follows the robot base and hides the chassis/base in videos
    - the current implementation is visual-only and creates no collision
  - Usage notes:
    - useful for cleaning up pure/debug videos
    - `local_pos` controls height and local offset
    - `half_size` controls the occluder box dimensions

## 2026-03-25 18:55:00 +08

- R1 planner wrist-camera export semantics updated
  - Behavior:
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - no longer depend on post-export image rotation
    - instead use an R1-specific wrist local pose inside `plan_anygrasp_keyframes_r1.py` that matches `galaxea_sim/robots/r1.py`
    - output size returns to the original landscape `image_width x image_height`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Notes:
    - this override is specific to the R1 planner path and does not change the global default in `render_hand_retarget_r1_npz.py`
    - the goal is to obtain the correct wrist view from the mounted camera pose itself rather than from an export-time image rotation patch

## 2026-03-25 18:35:00 +08

- Planner wrist-video export fine-tuned again
  - Behavior:
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - now use a uniform `90°` CCW rotation before export
    - writer dimensions match the rotated frames
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Notes:
    - this change is based on the user's actual result: the previous `180°` fix still looked like the correct view rotated `90°` CCW
    - no new CLI parameter was added

## 2026-03-25 18:20:00 +08

- Planner wrist-video export corrected again
  - Behavior:
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - no longer use a `90°` rotation; they now use a uniform `180°` in-plane rotation before export
    - output size returns to landscape `image_width x image_height`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Notes:
    - no new CLI parameter was added
    - this correction is based on the user's actual exported result: the previous version produced portrait videos that were still upside down
    - the new fix only changes image-plane export orientation, not the camera mount or planner coordinate conventions

## 2026-03-25 16:45:00 +08

- `--debug_visualize_ik_waypoints 1`
  - Visualization enhancement:
    - now shows start and goal markers in addition to intermediate TCP waypoints
    - both start and goal use red point+forward-axis markers
    - intermediate waypoint markers are smaller so the hands, target axes, and path are easier to inspect together
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Usage:
    - command syntax is unchanged; keep using `--debug_visualize_ik_waypoints 1`
    - this flag only affects viewer/debug visualization and does not change planning or execution logic

## 2026-03-25 17:10:00 +08

- Planner wrist-video orientation correction
  - Behavior:
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - are now rotated 90 degrees clockwise before being written
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Notes:
    - no new CLI parameter was added
    - this only affects planner wrist-video orientation
    - it does not change world camera pose or planner coordinate definitions

## 2026-03-25 12:08:00 +08

- Added flag: `--enable_grasp_action_object_collision 0|1`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/run_plan_anygrasp_keyframes_r1_batch.sh`
  - Purpose:
    - enable collision blocking for the selected execution object during `close_gripper` and `action`
    - default `0` preserves the previous no-collision execution mode
  - Usage:
    - single-run command: append `--enable_grasp_action_object_collision 1`
    - batch: append `--enable_grasp_action_object_collision 1` to the batch command as well
  - Notes:
    - this flag does not change target-pose construction for `pregrasp/grasp/action`
    - it also does not change the relative transform used when attaching the object to TCP
## 2026-03-25 13:05:00 +08

- Added visualization-mode flags to `plan_anygrasp_keyframes_r1.py`:
  - New flags:
    - `--debug_visualize_targets 0|1`
    - `--viewer_show_camera_frustums 0|1`
  - Usage:
    - `debug_visualize_targets=0` disables target-axis actors globally
    - `viewer_show_camera_frustums=0` disables SAPIEN camera frustum lines in the interactive viewer
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`

- Synced the same visualization-mode flags through `plan_anygrasp_keyframes_r1_batch.py`:
  - Forwarded flags:
    - `--debug_visualize_targets`
    - `--viewer_show_camera_frustums`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-25 13:35:00 +08

- `--enable_grasp_action_object_collision 1`
  - Behavior enhancement:
    - The `close_gripper` stage no longer always closes fully in one shot
    - It now closes progressively and stops early once contact with the selected object is present and gripper joint motion has stalled
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`

## 2026-03-25 14:10:00 +08

- Added parameter: `--urdfik_cartesian_interp_auto_step_m`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Only active when `--urdfik_cartesian_interp_steps=-1`, controlling the translation threshold of automatic waypoint mode.
  - Default:
    - `0.05`
  - Example:
    - `--urdfik_cartesian_interp_steps -1 --urdfik_cartesian_interp_auto_step_m 0.03`
  - Note:
    - Smaller values create denser intermediate waypoints.
  - Usage:
    - Command syntax is unchanged; keep using `--enable_grasp_action_object_collision 1`
    - With the default `0`, the original fast no-collision closing behavior is preserved

## 2026-03-25 14:05:00 +08

- `--urdfik_cartesian_interp_steps`
  - New convention:
    - `-1` enables automatic waypoint mode
  - Automatic rule:
    - no intermediate waypoint when translation is `<= 0.05m`
    - add one intermediate waypoint for each additional `0.05m` bucket beyond that threshold
  - Example:
    - `--urdfik_cartesian_interp_steps -1`
  - Related code:
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`

## 2026-03-25 14:25:00 +08

- `planner_backend=urdfik` + `urdfik_trajectory_mode=cartesian_interp_ik`
  - Behavior fix:
    - the execution layer now truly consumes the full `joint_waypoints` stored in `plan["position"]`
    - it no longer executes only a straight endpoint interpolation from `current_joints` to `target_joints`
  - Related code:
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`
# 2026-03-25

- `--pure_scene_output 1`
  - Behavior update:
    - no longer generates `debug_selection_preview.mp4`
    - now keeps and writes:
      - `head_cam_plan.mp4`
      - `left_wrist_cam_plan.mp4`
      - `right_wrist_cam_plan.mp4`
    - auto-enables `pose_debug.jsonl`
  - Main `pose_debug.jsonl` fields:
    - `record_index`
    - `stage`
    - `active_frame`
    - `current_*_camera_pose_world_wxyz`
    - `current_*_tcp_pose_world_wxyz`
    - `current_*_ee_pose_world_wxyz`
    - `current_*_arm_qpos_rad`
    - `current_*_gripper_joint_qpos_rad`
    - `object_actor_poses`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Reference docs:
    - `agent-read/2026-03-25_pure_mode_outputs_ZH.md`
    - `agent-read/2026-03-25_pure_mode_outputs.md`

- Added command parameter: `--debug_visualize_ik_waypoints`
  - Entry points:
    - `/home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `/home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Show the intermediate TCP/EE smoothing waypoints of `cartesian_interp_ik` in debug/viewer output, helping distinguish “bad waypoint generation” from “bad IK/execution follow-through”.
  - Display:
    - Position point plus local forward-axis marker for each intermediate waypoint.
  - Default:
    - `0`

- Added command parameter: `--debug_collision_report`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Print stronger collision debug information during the progressive `close_gripper` stage.
  - Main output:
    - `[collision-debug-init]`
    - `[collision-debug-step]`
    - regular `[gripper-close]` now also includes `base_contact=...`
  - Debug focus:
    - distinguish `finger_contact` from `base_contact`
    - inspect `finger_pairs` / `base_pairs`
    - inspect collision-shape summaries of the target object, `left/right_gripper_link`, and finger links
  - Default:
    - `0`

- Added command parameter: `--execution_object_collision_mode {convex,solid_bbox}`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Control which collision geometry execution objects use at runtime.
  - Modes:
    - `convex`
      - keeps the current `add_convex_collision_from_file`
    - `solid_bbox`
      - reads mesh bounds
      - creates one axis-aligned solid box collision
  - Notes:
    - affects execution collision only, not the visual mesh
    - if `--replay_objects_ignore_collision 1` and the object is not enabled for grasp/action collision, collision is still omitted
  - Default:
    - `convex`

- Added command parameter: `--gripper_contact_monitor_mode {fingers,fingers_and_base,all_robot_links}`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Control which robot links are allowed to trigger contact monitoring during `close_gripper`.
  - Modes:
    - `fingers`
      - finger links only
    - `fingers_and_base`
      - finger links plus `left/right_gripper_link`
    - `all_robot_links`
      - all links of the articulation corresponding to the current arm
  - Notes:
    - this changes the monitoring set used by the early-stop logic, not just the printed debug output
    - it is especially useful for debugging cases where finger/base links report `shapes=0` but other links may still carry collision

## 2026-03-25 23:10:00 +08

- Added a minimal collision-probe command:
  - Entry script:
    - `code_painting/minimal_gripper_collision_probe.py`
  - Purpose:
    - verify raw scene contacts between the R1 gripper and either a simple box or a mesh (`solid_bbox` / `convex`) object without going through the AnyGrasp/IK/stage main pipeline
  - Representative commands:
    - box:
      - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/minimal_gripper_collision_probe.py --arm left --object_kind box --probe_local_offset 0.04 0.0 0.0 --max_iters 20 --settle_steps_per_iter 8`
    - mesh:
      - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/minimal_gripper_collision_probe.py --arm left --object_kind mesh --mesh_path /home/zaijia001/ssd/data/R1/hand/obj_mesh/blue_cup/blue_cup.obj --mesh_collision_mode solid_bbox --probe_local_offset 0.04 0.0 0.0 --max_iters 20 --settle_steps_per_iter 8`
  - Key outputs:
    - step-by-step terminal logs of:
      - `qpos`
      - `raw_contact_total`
      - `target_contact_total`
      - `target_contacts`
    - JSON outputs:
      - `code_painting/minimal_gripper_collision_probe/probe_left_box.json`
      - `code_painting/minimal_gripper_collision_probe/probe_left_mesh.json`

- Enhanced `--debug_collision_report 1` output:
  - added raw target-contact reporting during the close stage:
    - `raw_target_contacts`
    - `raw_target_contact_total`
    - `[gripper-close] ... raw_target_contact=0|1`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Purpose:
    - distinguish “the monitor/helper failed to match a contact” from “raw physics contact does not exist at all”

- Further enhanced `--debug_collision_report 1` output:
  - Added:
    - `target_pose=...`
    - `target_collision_debug=...`
  - Purpose:
    - directly inspect whether the target-object actor pose stays stable during close
    - directly inspect the `solid_bbox` `center/half_size`

- Added flag: `--debug_visualize_object_collision_bbox 0|1`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - when `execution_object_collision_mode=solid_bbox`, additionally display the object's collision bbox
  - Typical usage:
    - append to the existing command:
      - `--debug_visualize_object_collision_bbox 1`

- Added flag: `--grasp_action_object_collision_start_stage {close_gripper,grasp,pregrasp}`
  - Purpose:
    - control from which stage selected execution objects start participating in collision
  - Typical experiment:
    - `--enable_grasp_action_object_collision 1 --grasp_action_object_collision_start_stage pregrasp --execution_object_collision_mode convex`

- Added pre-close pose export:
  - Output files:
    - `close_stage_snapshot_dual_before_close.json`
    - `close_stage_snapshot_<arm>_before_close.json`

- Added flag: `--execution_object_scale_override NAME=S|SX,SY,SZ`
  - Purpose:
    - independently scale an execution object's visual mesh and collision geometry together
  - Typical examples:
    - `--execution_object_scale_override cup=0.9`
    - `--execution_object_scale_override bottle=0.9`

- Added flags:
  - `--execution_object_visual_scale_override NAME=S|SX,SY,SZ`
  - `--execution_object_collision_scale_override NAME=S|SX,SY,SZ`
  - Purpose: independently control execution-object visual-mesh scale and collision-shape scale
- 2026-05-07
  - Added a retarget post-rotation board video command:
    - `code_painting/run_piper_retarget_postrot_board_video.sh`
  - Key environment variables:
    - `CASE_MODE=standard|axis90|grid`
    - `ORIENTATION_REMAP_LABEL`
    - `TARGET_DX/TARGET_DY/TARGET_DZ`
    - `ARMS`
    - `MAX_FRAMES`
  - Outputs:
    - `board/board_zed.mp4`
    - `board/board_third.mp4`
    - `board/board_zed_frame0000.png`
    - `board/board_third_frame0000.png`
    - `index.csv`

- 2026-05-07
  - Extended the local-axis sweep command:
    - `VIDEO_MODE=1`: generate `board_all_zed.mp4` and `board_success_zed.mp4`
    - `CANDIDATE_MODE=semantic`: generate `forward_from_*` / `open_from_*` semantic candidates
    - `FRAME_END` / `MAX_FRAMES` / `FRAME_STRIDE`: control the multi-frame range
  - Representative command:
    - `GPU=3 FRAME_IDX=0 FRAME_END=-1 MAX_FRAMES=32 ARM=left EXECUTE=1 CANDIDATE_MODE=remap VIDEO_MODE=1 FPS=5 SAVE_WRIST_VIEWS=0 bash code_painting/run_piper_local_axis_sweep_board.sh <input_npz> <output_dir>`

- 2026-05-07
  - Added a local-axis sweep command:
    - `bash code_painting/run_piper_local_axis_sweep_board.sh <input_npz> <output_dir>`
  - Related code:
    - `code_painting/build_piper_local_axis_sweep_board.py`
    - `code_painting/run_piper_local_axis_sweep_board.sh`
  - Important parameters:
    - `FRAME_IDX`
    - `ARM`
    - `EXECUTE`
    - `SAVE_WRIST_VIEWS`
  - Usage notes:
    - Keeps the current calibrated `PiperPika` head-camera pose by default
    - Defaults to a pure sweep without IK execution
    - Set `EXECUTE=1` when you specifically want to test whether execution lag/reaching error is the main issue

- 2026-05-11
  - Added a final HaMeR-axis replay batch command:
    - `code_painting/run_piper_hamer_axes_replay_batch.sh <input_npz_or_dir> <output_root>`
  - Key environment variables:
    - `GPU`
    - `FPS`
    - `MAX_FRAMES`
    - `ARMS=both|left|right`
    - `ID_FILTER=0,2,5-8`
    - `KEEP_ONLY_ZED_THIRD=1`
  - Fixed replay rule:
    - `--require_stored_gripper_pose 1`
    - `--pose_source gripper`
    - `--orientation_remap_label identity`
    - `--stored_orientation_post_rot_xyz_deg 0 0 0`
  - Outputs:
    - Each ID is written under `<output_root>/id_<id>`
    - `frames/` keeps only zed/third RGB PNGs by default
  - Documentation update:
    - The D section in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` now keeps only final replay commands
    - Detailed debug commands were moved to `/home/zaijia001/ssd/PIPER_GRIPPER_ORIENTATION_DEBUG.zh.md`

- 2026-05-11
  - Added a "HaMeR hands + FoundationPose objects in one replay" command:
    - `code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh <hand_npz_or_dir> <foundation_obj_vis_root> <output_root>`
  - Key environment variables:
    - `GPU`
    - `FPS`
    - `MAX_FRAMES`
    - `ARMS=both|left|right`
    - `ID_FILTER=0,2,5-8`
    - `KEEP_ONLY_ZED_THIRD=0|1`
    - `OBJECT_MISSING_FRAME_POLICY=hide|hold_last`
  - New low-level replay flags:
    - `--object_replay_input_dir`
    - `--object_missing_frame_policy`
    - `--objects`
    - `--object NAME=/path/to/mesh.obj`
  - Usage location:
    - Section E in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
  - Validation:
    - A 1-frame smoke test passed and generated `/tmp/piper_hamer_axes_with_objects_smoke/id_0/zed_replay.mp4` and `third_replay.mp4`

- 2026-05-11
  - Added a world-axis distance plot command for gripper/wrist-retreat points to objects:
    - `code_painting/plot_piper_gripper_wrist_object_axis_distances.py`
  - Key flags:
    - `--hand_npz`
    - `--object_dir`
    - `--output_png`
    - `--target_world_offset_xyz`
    - `--left_object`
    - `--right_object`
  - Outputs:
    - PNG plot
    - Matching CSV table
  - Usage location:
    - D-debug-9 in `/home/zaijia001/ssd/PIPER_GRIPPER_ORIENTATION_DEBUG.zh.md`

- 2026-05-18
  - Updated Piper replay commands to the 0515/new_table calibration:
    - Sections C/D/E in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
  - Added a calibration note section:
    - C0 in `COMMAND_LIBRARY.zh.md`
  - Current default configuration:
    - `--robot_config /home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json`
    - `--head_camera_local_pos 0.11210396690038413 -0.39189397826604927 0.4753892624100325`
    - `--head_camera_local_quat_wxyz -0.16972170030359832 0.6934883729683636 -0.6816465914025073 0.16008230830760367`
  - Synchronized wrapper defaults:
    - `code_painting/run_piper_hamer_axes_replay_batch.sh`
    - `code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`

- 2026-05-18
  - Corrected the Piper calibration command notes in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`:
    - C0 calibration sources now list four files instead of three, adding `left_wrist_new_table_eye_in_hand.json`
    - C0 records both left and right wrist `gripper_T_camera` pos/quat values so future wrist-camera rendering does not wire only the right wrist
    - C0 adds the current base/head-camera placement estimate
    - The section F distance-plot command was restored as a single copyable command
  - Key paths:
    - `head_d435_new_table_0515_head_from_wrist.json`
    - `left_base_T_right_base_new_table.json`
    - `left_wrist_new_table_eye_in_hand.json`
    - `right_wrist_new_table_eye_in_hand.json`

- 2026-05-18
  - Added `place_bread_basket` HaMeR/Piper debugging commands:
    - Location: D0 in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
  - New command types:
    - HaMeR GPU detection: `detect_hands_realr1.py --data_dir .../place_bread_basket/harmer_input --output_dir .../place_bread_basket/harmer_output`
    - HaMeR native gripper visualization inspection: `hand_vis_gripper_*.mp4`
    - Piper replay check: `run_piper_hamer_axes_replay_batch.sh .../place_bread_basket/harmer_output .../output_place_bread_basket_piper_hamer_axes`
  - Important parameters:
    - `GPU_ID=2`
    - `GPU=2`
    - `ARMS=both`
    - `KEEP_ONLY_ZED_THIRD=1`
    - `ID_FILTER=0`

- 2026-05-18
  - Added Piper calibration bundle commands:
    - Location: C0 and D0 in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
  - New tools:
    - `code_painting/build_piper_calibration_bundle.py`
    - `code_painting/visualize_piper_calibration_bundle.py`
  - New replay parameter/environment variable:
    - `--piper_calibration_bundle`
    - `CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json`
  - New visualization commands:
    - Outputs `/home/zaijia001/ssd/RoboTwin/code_painting/output_piper_calibration_bundle_0515/axes_compare_old_head.png`
    - Opens the viewer for `place_bread_basket` id0 camera/scene inspection

- 2026-05-18
  - Added commands for a head-camera marker visible in third-person renders:
    - `DEBUG_VISUALIZE_CAMERAS=1`
    - `DEBUG_CAMERA_AXIS_LENGTH=0.22`
  - Added three-version comparison commands:
    - `output_place_bread_basket_camera_compare/old_manual`
    - `output_place_bread_basket_camera_compare/new_table_pre0515`
    - `output_place_bread_basket_camera_compare/new_table_0515`
  - Usage:
    - Inspect `id_0/frames/third_0000.png` under each output directory
    - Marker colors: white body, red/green/blue for local `+x/+y/+z`, yellow for the camera `-Z` optical ray

- 2026-05-18
  - Added viewer diagnostic command:
    - `code_painting/probe_sapien_viewer.py`
  - Updated the viewer command:
    - Prints `DISPLAY/WAYLAND_DISPLAY/XDG_SESSION_TYPE` before running
    - The hand replay code prints `[viewer] ...` diagnostics internally
    - Adds `--debug_visualize_cameras 1` so the third-person output also shows the head-camera marker

- 2026-05-18
  - Corrected the viewer debugging command:
    - Removed `CUDA_VISIBLE_DEVICES=2`
    - Added `CUDA_VISIBLE_DEVICES` to viewer diagnostics
    - Added a negative-control minimal viewer probe with `CUDA_VISIBLE_DEVICES=2`
  - Reason:
    - SAPIEN viewer needs the display GPU to remain visible; setting `CUDA_VISIBLE_DEVICES=2` can hide the GPU backing the VNC/X display and produce `Renderer does not support display`

- 2026-05-18
  - Corrected the 0515 Piper head-camera command parameters:
    - Old direct quaternion: raw/optical hand-eye quaternion
    - New direct quaternion: `0.8524694864910365 -0.0011011947849308937 0.5226654778798345 0.010740586780925399`
    - The new quaternion is the render/SAPIEN camera pose after `raw_optical @ legacy_r1.T`, matching `--camera_cv_axis_mode legacy_r1`
  - Updated locations:
    - C0/C1/C2/C3/D1/E2 in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
    - `code_painting/run_piper_hamer_axes_replay_batch.sh`
    - `code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`
    - `code_painting/plot_piper_gripper_wrist_object_axis_distances.py`
  - The bundle-generation command format is unchanged, but `build_piper_calibration_bundle.py` now converts raw head transforms into the render/SAPIEN head pose required by replay; the 0515 and pre-0515 bundles were regenerated.

- 2026-05-18
  - Updated section D replay commands in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` to prefer the calibration bundle:
    - Explicitly added `CALIBRATION_BUNDLE=/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json` to the place_bread_basket id0/id1/batch commands
    - Changed the D1 low-level single replay command to use `--piper_calibration_bundle`
    - Explicitly added `CALIBRATION_BUNDLE` to the D2 final batch, ID-filtered, and right-hand-only commands
  - Reason:
    - The wrapper defaults already contain the correct 0515 render/SAPIEN head parameters, but the bundle form is safer. After the next calibration update, regenerate or replace the bundle first instead of manually syncing scattered `robot_config/head_pos/head_quat` values.

- 2026-05-19
  - Updated section D debug commands in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`:
    - Added a HaMeR command for selected video IDs only: `--video_ids $(seq 0 10)`
    - Added a Piper replay command for id0-id10 only: `ID_FILTER=0-10`
    - Added `D4. HaMeR visualization video + Piper retarget replay side-by-side comparison`
  - Added ffmpeg stacking commands:
    - Single ID: `hand_vis_gripper_<id>.mp4` + `zed_replay.mp4`
    - Single ID: `hand_vis_gripper_<id>.mp4` + `zed_replay.mp4` + `third_replay.mp4`
    - Batch id0-id10: generate `compare_hamer_gripper_piper_zed_third_<id>.mp4` for each ID
  - Validation:
    - Ran the three-panel command for id0 successfully and generated `output_place_bread_basket_piper_hamer_axes/id_0/compare_hamer_gripper_piper_zed_third_0.mp4`

- 2026-05-19
  - Fixed the time-alignment issue in the D4 video-stacking commands:
    - HaMeR `hand_vis_gripper` is usually 30fps while Piper replay is 5fps, so direct `hstack` makes the left HaMeR panel play too fast
    - Added aligned commands that use `ffprobe` to compute `zed_replay_duration / hamer_duration`, then apply `setpts=R*PTS,fps=5` to the HaMeR input
    - Added single-ID aligned output: `compare_aligned_hamer_gripper_piper_zed_third_<id>.mp4`
    - Added a batch aligned stacking command for id0-id10
  - Validation:
    - Successfully generated `output_place_bread_basket_piper_hamer_axes/id_0/compare_aligned_hamer_gripper_piper_zed_third_0.mp4`
    - The auto-computed id0 stretch ratio was about `R=6.0`, matching HaMeR 30fps to Piper 5fps

- 2026-05-19
  - Added a command format for retreating replay targets along the local blue axis:
    - Added `TARGET_LOCAL_FORWARD_RETREAT_M=0.05` to D2 in `/home/zaijia001/ssd/RoboTwin/COMMAND_LIBRARY.zh.md`
    - The underlying CLI flag is `--target_local_forward_retreat_m 0.05`
  - Parameter meaning:
    - Positive values retreat opposite the gripper-local `+Z` blue approach axis
    - Unlike `--target_world_offset_xyz`, this is not a fixed world-coordinate offset; it follows each frame's eepose orientation
  - Related entrypoints:
    - `code_painting/render_hand_retarget_r1_npz.py`
    - `code_painting/run_piper_hamer_axes_replay_batch.sh`
    - `code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`

- 2026-05-20
  - Fixed the compatibility issue behind the C1 FoundationPose two-object replay command:
    - Added a note near line 182 of `/home/zaijia001/ssd/RoboTwin/COMMAND_LIBRARY.zh.md` for the pick_diverse_bottles command
    - The command still uses explicit 0515 calibration arguments via `--robot_config`, `--head_camera_local_pos`, and `--head_camera_local_quat_wxyz`
  - Related entrypoints:
    - `code_painting/run_multi_object_pose_r1_npz_batch.sh`
    - `code_painting/render_multi_object_pose_r1_npz.py`
    - `code_painting/render_object_pose_r1_npz.py`
  - Validation:
    - `CUDA_VISIBLE_DEVICES=2 ... --ids 0 --max_frames 1 --skip_existing 0 ...` smoke test passed without the renderer-constructor missing-argument error

- 2026-05-20
  - Added three E2 single-command recipes for hand + specified FoundationPose video-dir replay:
    - pick_diverse_bottles: `hand_detections_0.npz` + `foundation_input_0`, objects `right_bottle/left_bottle`
    - place_bread_basket: `hand_detections_0.npz` + `foundation_input_0`, objects `basket/bread`
    - stack_cups: `hand_detections_0.npz` + `foundation_input_0`, objects `right_dark_red_cup/left_light_pink_cup`
- 2026-05-26
  - Added command group: Piper hand raw `origin/` visual review.
  - Related commands:
    - `python code_painting/review_piper_hand_origin.py --task pnp_tray --delay-ms 45`
    - `python code_painting/review_piper_hand_origin.py --task handover_bottle --delay-ms 45`
    - `python code_painting/review_piper_hand_origin.py --task pnp_bread --delay-ms 45`
  - Important parameters:
    - `--task` selects the task directory.
    - `--start-id` starts from a specific episode id.
    - `--dry-run` records decisions without moving directories.
  - Code location:
    - `code_painting/review_piper_hand_origin.py`
  - Usage notes:
    - Run in the `RoboTwin_openvla` environment.
    - Pressing `b` / `d` moves the current episode from `origin/` to `bad/`, so later preprocessing naturally skips it.

  - Key parameters:
    - `--target_local_forward_retreat_m 0.05`: retreat 5 cm opposite the gripper-local blue `+Z` approach axis
    - Viewer toggle: append `--enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02`
  - Validation:
    - The pick_diverse_bottles id0 `--max_frames 1` smoke test passed

- 2026-05-20
  - Changed the E2.1/E2.2/E2.3 hand + object replay commands to id0-id10 batch-loop format:
    - `pick_diverse_bottles`
    - `place_bread_basket`
    - `stack_cups`
  - Command format:
    - `source ... && for ID in $(seq 0 10); do CUDA_VISIBLE_DEVICES=2 conda run ...; done`
    - `--input_npz`, `--output_dir`, and `--object_replay_input_dir` all expand `${ID}`
  - Viewer usage note:
    - The batch commands keep the viewer off by default; to use the viewer, narrow the loop to a single ID and then append the viewer flags

- 2026-05-20
  - Added section G distance-curve commands:
    - `plot_piper_gripper_wrist_object_axis_distances.py`
    - Batch commands for all three tasks and id0-id10 to generate world-axis distance PNG/CSV files from gripper/wrist-retreat points to object centers
  - Key parameters:
    - `--left_object` / `--right_object` explicitly map hands to FoundationPose object folders
    - `--max_frames 300`
    - Outputs follow the H2O replay `id${ID}_z005` directories
  - Purpose:
    - Use `dx/dy/dz` curves to distinguish object/hand z-offset symptoms caused by detection from symptoms caused by the replay/calibration chain

- 2026-05-26
  - Added `COMMAND_LIBRARY.zh.md` L12: directly fix task text in an already generated LeRobot cache.
  - New command:
    - Back up `meta/tasks.jsonl` and `meta/episodes.jsonl`, then use `perl -0pi` to replace `pick diverse bottles` with the full English instruction.
  - Note:
    - The current LeRobot parquet files only store `task_index`; task text lives in meta files, so parquet files do not need to be edited for this cache-only fix.

- 2026-05-26
  - Added `COMMAND_LIBRARY.zh.md` L11: extract selected episodes from an existing LeRobot cache and reindex them.
  - New command:
    - `uv run python scripts/subset_lerobot_episodes.py --source <repo-or-path> --output-repo-id <repo> --episodes '0-24' --overwrite`
    - `--episodes` accepts comma lists and inclusive ranges such as `0,1-5,7`; the script deduplicates, sorts, and writes continuous `0..N-1` episodes.
  - Main use:
    - Directly create 25-episode subsets from already converted datasets such as `local/h2o_pick_diverse_bottles_human_head_pure_action`, without rerunning HDF5 -> LeRobot conversion.
  - Related code:
    - `policy/pi0/scripts/subset_lerobot_episodes.py`

- 2026-05-21
  - Updated the section G distance-plot visualization rule:
    - `plot_piper_gripper_wrist_object_axis_distances.py` now defaults to `--plot_clip_abs_m 0.5`
    - PNG plots compress large outliers beyond `±0.5m`, while CSV files keep raw values
    - Append `--plot_clip_abs_m 0` to see the full uncompressed plot scale
  - Compatibility:
    - Missing FoundationPose object tracks now produce NaN curves instead of aborting the batch

- 2026-05-21
  - Added section H to `COMMAND_LIBRARY.zh.md`: raw HaMeR hand points vs raw FoundationPose object points.
  - New command entrypoint: `code_painting/make_hamer_foundation_point_compare_video.py`.
  - Important parameters:
    - `--hand_npz`: HaMeR `hand_detections_<id>.npz`
    - `--hand_video`: HaMeR `hand_vis_gripper_<id>.mp4`
    - `--object NAME=/path/to/foundation_input_<id>/<object>`: repeatable object directory spec
    - `--left_object` / `--right_object`: object names used for left/right hand CSV deltas
    - `--output_video`: output comparison video; a paired `.csv` is generated by default
  - Added id0-id10 batch commands for three tasks:
    - pick_diverse_bottles: `left_bottle/right_bottle`
    - place_bread_basket: `basket/bread`
    - stack_cups: `left_light_pink_cup/right_dark_red_cup`
  - Usage note: this comparison bypasses Piper replay and checks raw detection/object-pose point offsets directly.

- 2026-05-21
  - Updated the H2 command notes in `COMMAND_LIBRARY.zh.md`:
    - `make_hamer_foundation_point_compare_video.py` now also emits `*_distance.png` by default
    - The distance plot mirrors section G and shows per-frame camera-frame `dx/dy/dz` for each hand/object pair
    - Documented `--plot_clip_abs_m 0` to disable plot clipping and `--output_plot` to override the PNG path
  - H6 now includes a finder command for `*_hamer_foundation_points_distance.png`.

- 2026-05-21
  - Updated the E/H sections in `COMMAND_LIBRARY.zh.md`:
    - Added E2.0 for pure hand replay without `--object_replay_input_dir` or `--object` arguments
    - E2.0 covers id0-id10 batches for pick_diverse_bottles, place_bread_basket, and stack_cups
    - Added a current H raw CSV statistics summary after H1 for comparison against the G/H1 world replay statistics
  - Validation: `bash -n` passed for all three E2.0 loop commands.

- 2026-05-21
  - Updated E2.0 pure hand replay commands:
    - `--save_png_frames 0` avoids generating per-frame PNG files under `frames/`
    - Added a single-video transcode command to make replay mp4 files more compatible with VS Code by using H.264/yuv420p
  - Validation: related commands passed `bash -n`.

- 2026-05-21
  - Updated `COMMAND_LIBRARY.zh.md`:
    - Added E0 for pure Piper replay on the three H2O tasks over `id0-id10`; it keeps only `zed_replay.mp4` / `third_replay.mp4`, disables overlay text and target/camera axis visualization, and transcodes to VS Code-friendly H.264/yuv420p.
    - Added I for the `/home/zaijia001/usage.sh`-style two-stage SAM repaint flow: first cache hand-removed backgrounds as `human_hand_bg.mp4`, then repaint E0 pure robot replay videos onto those backgrounds.
    - Added J to check id0-id10 AnyGrasp data availability for the three tasks and generate HaMeR-aligned ranked candidate previews/summaries with `render_anygrasp_ranked_preview.py`.
    - Added K to run Piper AnyGrasp keyframe replay from J summaries and repaint `head_cam_plan.mp4` onto the I1 backgrounds.
  - Key paths:
    - Pure replay output: `code_painting/human_replay/h2_pure/<task>/id<ID>_z005`
    - Repaint output: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2`
    - AnyGrasp preview: `code_painting/anygrasp_h2o_preview`
    - AnyGrasp plan: `code_painting/anygrasp_h2o_plan`
  - Validation:
    - Extracted 28 bash code blocks from E0 onward and `bash -n /tmp/command_library_new_blocks.sh` passed.
    - Confirmed `run_human_robot_inpaint_repaint.py`, `render_anygrasp_ranked_preview.py`, and `run_plan_anygrasp_keyframes_piper_batch.sh` exist.

- 2026-05-21
  - Fixed the `COMMAND_LIBRARY.zh.md` I1/I2 SAM repaint commands:
    - I1 now only requires `harmer_input/rgb_<id>.mp4` and uses an existing dummy robot video to satisfy the required `run_human_robot_inpaint_repaint.py --robot_video` argument.
    - Documented that I1 `human_hand_bg.mp4` does not depend on whether the dummy robot video matches the same task/id.
    - I2 still requires the per-task/id E0 pure `zed_replay.mp4` and now reports missing BG vs missing pure robot paths separately.
  - Reason: the user's I1 run printed many `[skip] missing HUMAN or ROBOT` lines; `HUMAN` existed, while most `h2_pure/<task>/id<ID>_z005/zed_replay.mp4` files had not been generated yet.
  - Validation: extracted 4 bash code blocks from section I and `bash -n /tmp/command_library_I_blocks.sh` passed.

- 2026-05-22
  - Fixed the I2/K2 repaint background paths in `COMMAND_LIBRARY.zh.md`:
    - The Stage-1 background actually lives under `stage1_human_inpaint/removed_w_mask_*.mp4`; a top-level `human_hand_bg.mp4` is not always produced.
    - I2/K2 now try the top-level `human_hand_bg.mp4` first, then fall back to `stage1_human_inpaint/removed_w_mask_*.mp4`.
    - Missing-file messages now point at `${BG_ROOT}/stage1_human_inpaint` to make I1 output checks easier.
  - Updated the I1 output check command to search for `stage1_human_inpaint/removed_w_mask_*.mp4` directly.
  - Validation: the I/K2 repaint command blocks passed `bash -n`; sampled id0/id1/id10 for all three tasks and the BG fallback resolved to existing `removed_w_mask_rgb_<ID>.mp4` files.

- 2026-05-22
  - Added a K0 manual review flow before K1 in `COMMAND_LIBRARY.zh.md`:
    - K0.1: inspect HaMeR gripper videos with `mpv`/`ffplay` and record keyframes manually.
    - K0.2: convert `/tmp/h2o_manual_keyframes.tsv` into per-task `h2o_manual_review/<task>/hand_keyframes_all.json` files.
    - K0.3: rerun AnyGrasp preview summaries with `--frame_selection_mode hand_keyframes_json`, producing metadata consumed by K1 `annotated_json_keyframes`.
    - K0.4: add a bad-id rejection command; it defaults to `APPLY=0` dry-run and moves human-side files into `_rejected_human_ids/` plus records `rejected_ids.json` when `APPLY=1`.
  - Validation: extracted the four K0 bash blocks and `bash -n /tmp/command_library_K0_blocks.sh` passed.

- 2026-05-22
  - Fixed the H2O AnyGrasp manual-keyframe batch commands:
    - `run_render_anygrasp_ranked_preview_keyframes_batch.sh` now supports a `VIDEO_PREFIX` environment variable; the default remains the legacy `d_pour_blue`, while H2O uses `VIDEO_PREFIX=foundation_input`.
    - K0.3 now calls the keyframe preview batch wrapper for whole-task summary generation instead of a handwritten per-id loop.
    - K1 no longer passes `--ids $(seq 0 10)`, so it processes all available `foundation_input_<id>` directories under each task root.
  - Confirmed the old flow: manual keyframe JSON first generates preview summaries, and the planner later consumes them with `--reuse_preview_frame_mode annotated_json_keyframes`.
  - Validation: the wrapper passed `bash -n`; the K0.3/K1 command blocks passed `bash -n`.

- 2026-05-22
  - Added command groups: `COMMAND_LIBRARY.zh.md` E2.4 / I3.
  - Added:
    - E2.4: normal D435 head-view pure Piper replay using `--image_width 640 --image_height 480 --fovy_deg 42.499880046655484`.
    - I3: repainting with E2.4 `zed_replay_d435.mp4` over the I1 background.
  - Important paths:
    - D435 parameters: `/home/zaijia001/ssd/data/piper/hand/pick_diverse_bottles/harmer_input/params_35.json`
    - replay output: `/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/<TASK>/id<ID>_d435_z005/`
    - repaint output: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_d435/e0_robot/<TASK>/id_<ID>_d435/`
  - Usage note: the old default replay uses a wide virtual camera with `fovy_deg=90`; use E2.4/I3 for the D435-aligned path and keep the `d435` filename suffix.
  - Validation: the new command blocks passed `bash -n`.
  - Follow-up clarification: E2.4 `fovy_deg=42.499880046655484` is computed from the actual RGB camera_info intrinsics; the official D435 depth FOV `85.2 x 58` is not the target for `/camera/color/image_raw` RGB replay.
  - Follow-up diagnosis: added I3.1 with the D435 Stage-2 mask failure analysis and reproduction commands; the conclusion is that the old Stage-2 parameters should not be reused directly for the narrow-FOV D435 robot replay.

- 2026-05-23
  - Updated command groups: `COMMAND_LIBRARY.zh.md` I3.2 / I3.3.
  - Added:
    - I3.2: mitigation guidance for D435 first-frame robot absence / background false positives, plus a contact-sheet command for checking the visible start frame.
    - I3.3: a single SAM3 debug command for D435 robot repaint that directly reuses the I1 Stage-1 background.
  - Key parameters:
    - Recommended D435 starting point: `--robot_box_threshold 0.35 --robot_text_threshold 0.30 --max_mask_area_ratio 0.35`.
    - The SAM3 command calls `remove_anything_video_sam3_robot.py --target_video "$BG"` directly, avoiding rerunning Stage-1 human inpainting through the SAM3 pipeline entrypoint.
  - Note:
    - The current SAM2/SAM3 robot scripts both initialize from frame 0. If frame 0 contains no robot, robot/BG must be trimmed in sync or the code must later support a nonzero initialization frame.
  - Validation: the new bash commands passed `bash -n`.

- 2026-05-26
  - Updated command group: `COMMAND_LIBRARY.zh.md` L14.
  - Added a Piper AnyGrasp two-keyframe planning debug command:
    - Entrypoint: `code_painting/plan_anygrasp_keyframes_piper.py`
    - Recommended parameters: `--planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik --urdfik_cartesian_interp_steps -1 --urdfik_cartesian_interp_auto_step_m 0.01`
    - Debug parameters: `--debug_visualize_targets 1 --debug_visualize_ik_waypoints 1 --save_pose_debug 1 --save_debug_execution_preview 1`
  - Related code:
    - `plan_anygrasp_keyframes_piper.py` now uses the Piper dual renderer and preserves separate arm bases.
  - Validation:
    - `plan_anygrasp_keyframes_piper.py` compiled successfully.

- 2026-05-28
  - Updated command groups: `COMMAND_LIBRARY.zh.md` J0.1 / J1.1.
  - Added:
    - J0.1: six-task D435 AnyGrasp / `foundation_replay_d435` / HaMeR input availability check.
    - J1.1: six-task D435 keyframe-based candidate preview/summary generation, writing to `anygrasp_h2o_preview_d435`.
    - Path separation from the default-wide path: D435 uses `foundation_replay_d435` and a separate preview root.
  - Validation:
    - The new command blocks passed `bash -n`.

- 2026-05-28
  - Updated command group: `COMMAND_LIBRARY.zh.md` L11.2.4.
  - Added:
    - D435 robot replay `_25ep` subsetting now filters original bad ids through processed `source_episode_id`.
    - The bad-id rules match the human replay path: `handover_bottle=0,7,12,29`, `pnp_bread=0,1,2,3,4,5,6,22,70`.
    - The command fills the first 25 usable LeRobot episode indices, and `subset_lerobot_episodes.py` reindexes the output to `0..24`.
  - Validation:
    - The new command block passed `bash -n`.

- 2026-05-28
  - Updated command group: `COMMAND_LIBRARY.zh.md` I3.5.
  - Added:
    - Explained the `d435_final` output path and generation chain: `batch_visible_reinit_d435_repaint.py` -> `remove_anything_video_sam3_robot_visible_reinit.py` -> `final_repainted.mp4`.
    - Clarified that the current I3.5 batch path is a SAM2/DINO2 fallback on this machine, despite living under `inpainting_sam3_robot`.
    - Added a batch command for first filling each task to at least 25 final videos, using `--overwrite 0` to keep existing outputs and reuse the one-time-loaded DINO/SAM checkpoints.
  - Validation:
    - The new command block passed `bash -n`.

- 2026-05-28
  - Updated command groups: `COMMAND_LIBRARY.zh.md` I1.1.1 / I3.5.
  - Added:
    - I1.1.1: a resume command for the new three tasks that only fills missing Stage-1 BGs and skips ids that already have `removed_w_mask_*.mp4`.
    - I3.5: a D435 visible-reinit repaint resume command with `--id_start 0 --id_end 80 --overwrite 0`, skipping ids that already have `final_repainted.mp4`.
  - Usage:
    - When L8.2 produces few episodes and the log shows missing `/results_repaint_piper_h2_d435_sam3_visible_reinit/.../final_repainted.mp4`, run I1.1.1, then the I3.5 resume command, then rerun L8.2.
  - Validation:
    - The new command blocks passed `bash -n`.

- 2026-05-28
  - Updated command groups: `COMMAND_LIBRARY.zh.md` L6.1 / L8.2 / L10.6 / L11.2.4.
  - Added:
    - L6.1 now explains the boundary between default-wide `h2_pure` and D435 `h2_pure_d435`, including why the new three tasks hit `No usable episodes were processed`.
    - L8.2 provides the six-task D435 visible-reinit robot replay to processed-HDF5 command.
    - L10.6 provides the six-task D435 robot replay processed-HDF5 to LeRobot-cache command.
    - L11.2.4 provides six-task D435 robot replay `_25ep` subset, zip, and `rclone --dry-run` commands.
  - Run order:
    - Old three tasks: I1 -> I3.4 -> L8/L8.2 -> L10.3/L10.6 -> L11.2.4.
    - New three tasks: I1.1 -> I3.5 -> L8.1/L8.2 -> L10.6 -> L11.2.4.
  - Validation:
    - The new command blocks passed `bash -n`.

- 2026-05-28
  - Updated command groups: `COMMAND_LIBRARY.zh.md` J0.1 / J1.1.
  - Added notes:
    - Interpret J0.1 `MISS` lines by the `anygrasp/replay/hand` columns; `MISS` is expected when `seq 0 120` exceeds the real episode count.
    - J1.1 now supports `left_keyframes/right_keyframes` and writes them as `effective_keyframes_by_arm`.
    - L15.4 now calls `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`, avoiding zsh's lack of the bash-only `mapfile` builtin.
    - L15.4 now includes first-5-summary test commands per task and recommends `--continue_on_error`.
  - Related code:
    - `code_painting/render_anygrasp_ranked_preview.py`
    - `code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh`
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`
  - Validation:
    - A single `place_bread_basket id0` D435 preview generated successfully and its summary recorded per-arm effective keyframes.
    - The six-task script dry-run resolved the first summary for every task.
    - A single `pick_diverse_bottles id0` D435 planner run passed the script-level check and generated video/summary outputs.
    - The L15.4 six-task D435 planner command block passed `bash -n`.

- 2026-05-28
  - Updated command groups: `COMMAND_LIBRARY.zh.md` J1.2 / L15.3.
  - Added:
    - J1.2: targeted-id rerun command for D435 candidate preview/summary, with `pick_diverse_bottles` ids `2 17 18 19 20 21` as the example.
    - L15.3: D435 AnyGrasp planner command that requires `foundation_replay_d435` and the `summary.json` under `anygrasp_h2o_preview_d435`.
  - Key parameters:
    - `--image_width 640 --image_height 480 --fovy_deg 42.499880046655484`
    - `--reuse_preview_summary_json .../anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json`
    - `--replay_dir .../foundation_replay_d435/foundation_input_<ID>`
  - Usage note:
    - If the D435 summary is missing, the command skips the id; rerun J1.1/J1.2 instead of falling back to the default wide-FOV `anygrasp_h2o_preview`.
  - Validation:
    - The new bash command blocks passed `bash -n`.

- 2026-05-28
  - Updated command group: `COMMAND_LIBRARY.zh.md` L15.7.
  - Added:
    - The D435/Piper six-task wrapper now defaults to `--reach_error_pose_source ee` and supports explicit override.
    - Documented the current keyframe execution logic: D435 summary per-arm keyframes -> first-keyframe `pregrasp/grasp` -> close/action only after reached.
    - Added separate strict-sync first-5 commands for all six tasks.
    - Added separate partial + `joint_interp` + `--print_pose_every 5` diagnostic commands for all six tasks.
  - Key note:
    - `tcp` reached checks leave an approximately 12 cm TCP/EE offset; the current Piper D435 planner defaults to `ee` reached checks.
    - `--allow_partial_dual_stage` is diagnostic only and should not be used for final data generation.
  - Validation:
    - The wrapper passed `bash -n`.
    - Six-task dry-run passed.
    - A partial rerun of `pick_diverse_bottles id0` confirmed right-arm EE reach.

- 2026-05-28
  - Updated command group: `COMMAND_LIBRARY.zh.md` L15.7.
  - Added:
    - Viewer command examples with `--visualize_targets` to display target axes and active candidate grippers.
    - Documentation for `<OUT>/source_preview_compare/`, including copied D435/legacy source preview images and `selected_candidate_mapping.json`.
  - Validation:
    - The new L15.7 bash block passed `bash -n`.
    - `pick_diverse_bottles id0` produced source preview compare outputs.

- 2026-05-28
  - Updated command group: `COMMAND_LIBRARY.zh.md` L11.1.3.
  - Added:
    - A dedicated post-L10.5 `_25ep` subset command using source repos `local/h2o_<TASK>_human_head_pure_d435_action`.
    - Filtering by processed `instructions.json/source_episode_id` to exclude known bad original ids for the new three tasks while still filling 25 episodes.
    - Packaging and `rclone --dry-run` upload-check commands for `human_d435_action_3task_25ep.zip`.
  - Key distinction:
    - L11.2.1's `local/h2o_<TASK>_pure_repaint` is only for L6/L6.1 robot replay and is not the post-L10.5 path.
  - Validation:
    - The new command blocks passed `bash -n`.

- 2026-05-25
  - Added command/mode design: `COMMAND_LIBRARY.zh.md` I3.4.
  - Added:
    - Documented a "visible-frame SAM reinitialization mode" design for narrow-FOV D435 robot replay.
    - Proposed future interface flags such as `--init_policy first_visible --reinit_policy on_lost --empty_mask_when_lost 1`.
    - Proposed output root `results_repaint_piper_h2_d435_sam3_visible_reinit` for side-by-side comparison with the current SAM2/SAM3 outputs.
  - Note:
    - This is a future interface and state-machine design, not a currently runnable script; no runtime code was changed in this round.
  - Validation: the proposed I3.4 command passed `bash -n`.

- 2026-05-25
  - Updated command group: `COMMAND_LIBRARY.zh.md` I3.4.
  - Added runnable entrypoints:
    - Single video: `/home/zaijia001/ssd/inpainting_sam3_robot/remove_anything_video_sam3_robot_visible_reinit.py`
    - Batch: `/home/zaijia001/ssd/inpainting_sam3_robot/batch_visible_reinit_d435_repaint.py`
  - Added commands:
    - A single-id debug command writing to `results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/<TASK>/id_<ID>_d435`.
    - A three-task batch command controlled by `--id_start/--id_end`, reusing DINO/SAM checkpoints in one process.
    - A dry-run command for checking BG/ROBOT input availability.
  - Key parameters:
    - `--init_policy first_visible`
    - `--reinit_policy on_lost`
    - `--empty_mask_when_lost 1`
    - `--detector_stride 1`
    - `--lost_patience 2`
    - `--max_mask_area_ratio 0.35`
    - `--max_white_pixel_ratio_in_mask 0.60`
  - Validation:
    - Both new scripts passed `py_compile`.
    - Both new scripts passed `--help`.
    - Batch dry-run resolved the `pick_diverse_bottles id0` input paths.
    - The I3.4 commands passed `bash -n`.

- 2026-05-25
  - Updated command group notes: `COMMAND_LIBRARY.zh.md` I3.3 / I3.4.
  - Title updates:
    - I3.3 is now "current SAM3-project first-frame initialization: directly reuse the I1 background for D435 robot repaint".
    - I3.4 is now "new logic: visible-frame reinitialization SAM2/SAM3 mode".
  - Added notes:
    - The difference among the original SAM2 command, the current SAM3-project command, and the new visible-frame reinitialization command.
    - A startup log of `[backend] SAM=sam2, DINO=dino2` means the current environment is falling back to SAM2/GroundingDINO2; `[backend] SAM=sam3, DINO=dino3` would indicate true SAM3/DINO3.
    - The new script locally fixes the transformers/GroundingDINO `BertModel.get_head_mask` compatibility issue.
  - Validation:
    - The new script can complete model loading.
    - Both new scripts pass `py_compile`.

- 2026-05-25
  - Updated command groups: `COMMAND_LIBRARY.zh.md` I3.0 / I3.3 / I3.4.
  - Added the I3.0 comparison table and standalone commands:
    - I3.0.1: original fixed-frame-0 SAM2 single-id debug command.
    - I3.0.2: SAM3-project fixed-frame-0 single-id debug command.
    - True-SAM3 backend template: set `GROUNDED_SAM3_DIR=/path/to/Grounded_SAM_3`, then run I3.0.2 or I3.4 and verify `[backend] SAM=sam3, DINO=dino3`.
    - I3.0.3: new visible-frame reinitialization single-id debug command.
    - I3.0.4: new visible-frame reinitialization batch command that reuses checkpoints.
  - Compatibility fix record:
    - `remove_anything_video_sam3_robot.py` and `remove_anything_video_sam3_robot_visible_reinit.py` now patch the BERT helpers required by old GroundingDINO, making them compatible with transformers 5.3.0.
  - Validation:
    - The I3.0 command blocks passed `bash -n`.
    - The DINO forward smoke test passed.
## 2026-05-28 (L15.12 Piper AnyGrasp Axis Fix And IK Threshold Wrapper)

- Added/updated:
  - Added L15.12 to `COMMAND_LIBRARY.zh.md`, documenting the axis conversion between the Piper preview gripper frame and URDF link6.
  - Added `--ik_max_position_threshold_m` and `--ik_max_rotation_threshold_rad` to `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`.
  - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --viewer` no longer pauses at the end of every id by default; use `--viewer_wait_at_end 1` when manual end-of-id inspection is needed.
  - Added L15.13 to `COMMAND_LIBRARY.zh.md`, with per-task id0-10 viewer commands and axis-color notes.
  - Added `--id_start`, `--id_end`, `--ids`, and `--piper_apply_global_trans_to_ik` to `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`.
- Important parameters:
  - The default remains `--ik_max_rotation_threshold_rad 0.12` for strict complete-pose IK.
  - For position-first diagnostics, temporarily use `--ik_max_rotation_threshold_rad 3.14` to check whether a static viewer is caused by the orientation constraint.
- Related code:
  - `code_painting/render_hand_retarget_piper_dual_npz_urdfik.py`
  - `code_painting/plan_anygrasp_keyframes_r1.py`
  - `code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`
## 2026-05-29 (L15.14 Viewer Per-Arm Active Frame Check)

- Command documentation updated:
  - Added L15.14 to `COMMAND_LIBRARY.zh.md`.
  - Updated `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` with the viewer target-gripper per-arm keyframe display note.
- Related commands:
  - Dry-run check for the `pick_diverse_bottles id0-10` viewer axis-check command to confirm D435 summary resolution.
  - `jq -c '{stage,active_frame,active_frame_by_arm}' <OUT>/pose_debug.jsonl | head -n 20` checks that pregrasp/grasp show keyframe 1 and action shows keyframe 2.
  - The docs also include a `head ... | sed -E ...` fallback for shells where `jq` is unavailable.
- Important parameters:
  - `--visualize_targets`
  - `--id_start 0 --id_end 10`
  - `--piper_apply_global_trans_to_ik 0`
  - `--ik_max_rotation_threshold_rad 3.14`
## 2026-05-29 (L15.15 Stack Cups id0 No-Collision Target-Only Commands)

- Command documentation updated:
  - Added L15.15 to `COMMAND_LIBRARY.zh.md`.
  - Updated `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` with the stack_cups id0 target-only debug commands.
- New wrapper parameters:
  - `--disable_execution_collisions`
  - `--target_axes_only`
  - `--debug_candidate_top_k`
  - `--debug_common_candidate_top_k`
  - `--debug_visualize_selected_keyframe_axes`
  - `--debug_visualize_ik_waypoints`
- Purpose:
  - Rule out execution-object collision.
  - Keep only active execution target axes, avoiding visual confusion from target/candidate/selected/waypoint coordinate frames.
## 2026-05-29 (L15.16 Direct Piper Hand Replay Viewer Comparison Command)

- Added L15.16 to `COMMAND_LIBRARY.zh.md`.
- Added a direct replay comparison command:
  - Entry point: `code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py`
  - Example: `stack_cups id0`
  - Purpose: replay the stored HaMeR NPZ gripper poses directly, without AnyGrasp candidate selection.
- Important parameters:
  - `--debug_visualize_targets 1`
  - `--debug_mode 1 --debug_post_execute 1`
  - `--save_world_targets 1`
  - `--enable_viewer 1 --viewer_wait_at_end 1`
## 2026-05-29 (L15.17 Direct Replay / AnyGrasp Axis Convention Comparison)

- Added L15.17 to `COMMAND_LIBRARY.zh.md`.
- Added the convention note:
  - Direct Piper hand replay: local `+Z` blue is approach/forward.
  - AnyGrasp preview/planner: local `+X` red is the wireframe finger-depth direction.
- Added comparison parameter:
  - `--candidate_orientation_remap_label swap_red_blue`
- Purpose:
  - Test whether AnyGrasp local `+X` should be mapped onto direct replay local `+Z`, explaining or correcting the mismatch between viewer target axes and the actual robot gripper orientation.

## 2026-05-29 (L15.18 Replay-Axis AnyGrasp Six-Task Commands)

- Added L15.18 to `COMMAND_LIBRARY.zh.md`.
- New entrypoint:
  - `code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh`
- Added/passthrough parameters:
  - `--candidate_orientation_remap_label`
  - `--candidate_target_local_x_offset_m`
  - `--candidate_target_local_z_offset_m`
  - `--approach_axis`
  - `--approach_offset_m`
- The replay-axis wrapper fixes:
  - `--candidate_orientation_remap_label swap_red_blue`
  - `--candidate_target_local_x_offset_m 0.0`
  - `--candidate_target_local_z_offset_m -0.05`
  - `--approach_axis local_z`
  - `--approach_offset_m 0.12`
- L15.18 now contains six-task first-five no-viewer commands, viewer commands, and a `stack_cups id0-10` small-range viewer debug command.

## 2026-05-29 (L15.19 Candidate-Stage Frame-Unification Design Command)

- Added L15.19 to `COMMAND_LIBRARY.zh.md`.
- Records the long-term design:
  - Unify the raw AnyGrasp frame with the robot/replay frame during candidate filtering in `render_anygrasp_ranked_preview.py`.
  - Target frame: `robot local +Z = AnyGrasp raw local +X`, `robot local +Y = AnyGrasp raw local +Y`, and `robot local +X = -AnyGrasp raw local +Z`.
- Added the current runnable comparison command:
  - Entry point remains `run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh`.
  - Output root is `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper`.
- Note: this command still uses the L15.18 wrapper; it does not mean candidate-stage frame unification has been implemented.

## 2026-05-29 (L15.19.1 Robot-Frame Preview/Planner Commands)

- Added robot-frame preview generation entrypoint:
  - `run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh`
- Added robot-frame planner entrypoint:
  - `run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh`
- New important parameters:
  - `--candidate_frame_mode robot_replay`
  - `--candidate_target_local_z_offset_m`
  - `--preview_root`
  - `--debug_gripper_actor_forward_axis local_z`
- Output: planner viewer results write to `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper`.

## 2026-05-29 (Robot-Frame Planner Auto-Fills Preview Summaries)

- Updated entrypoints:
  - `code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh`
  - `code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh`
- New behavior:
  - The planner wrapper auto-generates missing robot-frame preview summaries before running the planner.
  - The generation range follows `--tasks`, `--ids`, `--id_start/--id_end`, and `--max_per_task`.
- New parameters:
  - `--skip_preview_generation`: disables planner-wrapper summary generation.
  - `--skip_existing`: controls whether the preview wrapper skips existing summaries; default is `1`.
  - `--source_preview_root`: source D435 preview root used to determine available id ordering.

## 2026-05-29 (L15.19.2 Robot-Frame Viewer Commands With Explicit ids)

- Added to `COMMAND_LIBRARY.zh.md`:
  - `stack_cups id4` robot-frame viewer command.
  - Per-task viewer templates with explicit ids.
  - One viewer command for all six tasks with `--ids 0 1 2 3 4`.
- Important parameters:
  - `--ids <ID>` selects exact episodes.
  - The robot-frame wrapper auto-fills missing summaries; add `--skip_preview_generation` to disable that behavior.

## 2026-06-02 (Mode O Gripper Axis Convention Note)

- Command formats are unchanged and still use:
  - `code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh`
  - `code_painting/plan_first_frame_foundation_pick_diverse_bottles.py`
- Documentation added:
  - Mode O currently stores planner target frames with local `+Z` as the approach/forward axis.
  - This matches Piper direct replay and robot-frame AnyGrasp, but differs from the original ALOHA-AgileX local `+X` fingertip-depth convention.
  - A strict ALOHA-style comparison should add a local-X branch and invoke the planner with `--approach_axis local_x`.

## 2026-06-02 (Mode O Gripper Frame Validation Commands)

- Added visualization entrypoint:
  - `code_painting/visualize_mode_o_gripper_frame_conventions.py`
- Added wrapper parameters:
  - `--target_frame_convention piper_local_z|aloha_local_x_y_up|aloha_local_x_z_up`
  - `--plan_only`
- Static visualization command:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python /home/zaijia001/ssd/RoboTwin/code_painting/visualize_mode_o_gripper_frame_conventions.py --video_id 0 --foundation_frame 0 --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/mode_o_frame_convention_debug`
- ALOHA-style local-X plan-only comparison:
  - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 0 --plan_only --target_frame_convention aloha_local_x_z_up --output_root /tmp/mode_o_aloha_local_x_plan_only`

## 2026-06-02 (O.0 Original demo_clean Piper Data Generation Command)

- Added task name:
  - `pick_diverse_bottles_piper`
- Added config:
  - `demo_clean_piper`
- Added command:
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0`
- Comparison command:
  - `bash collect_data.sh pick_diverse_bottles demo_clean 0`
- Code locations:
  - `envs/pick_diverse_bottles_piper.py`
  - `task_config/demo_clean_piper.yml`
  - `description/task_instruction/pick_diverse_bottles_piper.json`
- Note: O.0 uses the original RoboTwin demo data-generation path and does not use FoundationPose, AnyGrasp, or replay target frames.

## 2026-06-02 (Corrected O.0 collect_data Full Command)

- Fixed config:
  - Changed `task_config/demo_clean_piper.yml` from `embodiment: [piper]` to `embodiment: [piper, piper, 0.60]`.
  - Reason: `[piper]` makes RoboTwin look for missing `assets/embodiments/piper/curobo_left.yml`; the three-item config loads two single-arm Piper instances and uses `curobo.yml`.
- Recommended full command:
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0`
- Old errors:
  - Running directly from `~` cannot find `collect_data.sh`.
  - Without activating `RoboTwin_bw`, the `python` used inside `collect_data.sh` may not exist.

## 2026-06-03 (O.0 Calibrated Piper/Pika Data Generation Command)

- Added embodiment:
  - `piper_pika_agx_calibrated`
- Added/updated configs:
  - `assets/embodiments/piper_pika_agx/config.yml`
  - `task_config/demo_clean_piper.yml`
  - `task_config/demo_clean_piper_calibrated.yml`
  - `task_config/_embodiment_config.yml`
- Recommended command:
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0`
- Notes:
  - The old `demo_clean_piper` output directory already contains data generated with RoboTwin's built-in `assets/embodiments/piper/piper.urdf`.
  - `demo_clean_piper_calibrated` writes to a new directory and uses the calibrated `piper_pika_agx.urdf` and left/right base poses.

## 2026-06-03 (O.0 Head-Only And Viewer Debug Config)

- Updated configs:
  - `task_config/demo_clean_piper.yml`
  - `task_config/demo_clean_piper_calibrated.yml`
  - `task_config/demo_clean_piper_calibrated_viewer.yml`
- Recommended collection command:
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0`
- One-episode viewer/head-only debug command:
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50`
- Parameter changes:
  - The collection and viewer configs both use `collect_head_camera: true` and `collect_wrist_camera: false`.
  - The viewer config additionally sets `render_freq: 1`, `episode_num: 1`, and `collect_data: false`.
  - The viewer config is for inspecting seed/premotion behavior and does not save hdf5 data.
  - `run_view_pick_diverse_bottles_piper_scene.sh` does not enter `play_once` planning and skips unstable seeds automatically, so it is the preferred pure scene viewer check.
  - `run_collect_piper_calibrated_viewer.sh` no longer calls the missing `script/.update_path.sh`, but it still enters the original demo planner and is not the preferred viewer command.

## 2026-06-03 (O.0 Viewer Completion Semantics And No-Viewer Generation Command)

- Updated viewer command notes:
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50`
  - This command only displays the scene and stays in the viewer loop until the window is closed or `Ctrl-C` is pressed.
- Updated no-viewer generation notes:
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0`
  - This command currently enters the original `play_once/grasp_actor` path, but `tmux gen1` shows it still cannot complete an episode because of bottle instability and `target_pose cannot be None for move action`.

## 2026-06-03 (Mode M/N Viewer Command CUDA Mask Semantics)

- Related commands:
  - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435.sh --gpu 2 --ids <ID> --viewer --tasks <TASK> ...`
  - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh --gpu 2 --ids <ID> --viewer --tasks <TASK> ...`
- Change:
  - In viewer mode, both the bash wrapper and the Python middle layer remove `CUDA_VISIBLE_DEVICES` from the planner environment.
  - Non-viewer mode is unchanged and still uses `--gpu` for compute GPU selection.
- Usage notes:
  - If the minimal `probe_sapien_viewer.py` opens a viewer after `unset CUDA_VISIBLE_DEVICES`, the Mode M/N viewer commands should also open the viewer from the same graphical terminal.
  - If they still fail, inspect the `[viewer] creating interactive viewer ...` log for `DISPLAY` and `CUDA_VISIBLE_DEVICES`.
- Related code:
  - `code_painting/plan_keyframes_human_replay.py`
  - `code_painting/plan_keyframes_foundation_pose.py`
  - `code_painting/run_plan_keyframes_human_replay_piper_d435.sh`
  - `code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh`

## 2026-06-03 (O.0 Motion Baseline Data Generation And Motion Viewer Commands)

- Added task/config:
  - `envs/pick_diverse_bottles_piper_motion.py`
  - `task_config/demo_clean_piper_motion.yml`
  - `task_config/demo_clean_piper_motion_viewer.yml`
  - `description/task_instruction/pick_diverse_bottles_piper_motion.json`
  - `run_pick_diverse_bottles_piper_motion_viewer.sh`
- Tested no-viewer/head-only data-generation command:
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_motion demo_clean_piper_motion 0`
- Tested result:
  - Seed 0/1 were skipped due to `Objects is unstable`; seed 2 succeeded.
  - Outputs include `data/pick_diverse_bottles_piper_motion/demo_clean_piper_motion/data/episode0.hdf5`, `video/episode0.mp4`, `_traj_data/episode0.pkl`, and `instructions/episode0.json`.
- Motion viewer command:
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_pick_diverse_bottles_piper_motion_viewer.sh`
- Usage notes:
  - This command enters `pick_diverse_bottles_piper_motion`; it is not the scene-only viewer. In `tmux gen1-1`, it successfully reached seed 2 premotion.
  - The original `run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50` remains a stable-scene viewer and does not execute motion.
  - The original `collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0` still fails in the original `grasp_actor` path and is not the currently recommended runnable O.0 motion-data command.

## 2026-06-03 (O.0 Original IK/Planning Path Experiment Command)

- Added embodiment/config:
  - `piper_pika_agx_ik_orig_tcp`
  - `assets/embodiments/piper_pika_agx_ik_orig_tcp/config.yml`
  - `assets/embodiments/piper_pika_agx_ik_orig_tcp/curobo.yml`
  - `assets/embodiments/piper_pika_agx_ik_orig_tcp/collision_piper_pika.yml`
  - `task_config/demo_clean_piper_ik_orig_tcp.yml`
- Experiment command:
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_ik_orig_tcp 0`
- Meaning:
  - The task still uses `pick_diverse_bottles_piper`, so it enters the original `pick_diverse_bottles.py` `grasp_actor/place_actor` IK/planning path.
  - The embodiment uses the calibrated `piper_pika_agx.urdf` and left/right base poses, but uses the built-in RoboTwin Piper TCP conversion matrices.
  - The Curobo config was adjusted to match the Pika URDF gripper link/joint names.
- Smoke result:
  - `timeout 120s ... demo_clean_piper_ik_orig_tcp 0` confirmed the new embodiment and original task path, but did not finish an episode.
  - Main failures remained `Objects is unstable` and `target_pose cannot be None for move action`.
  - This command is for validating the original IK path; it is not the currently recommended successful data-generation command. The successful data-generation path remains `pick_diverse_bottles_piper_motion demo_clean_piper_motion`.

## 2026-06-04 (O.0 Command List Consolidation)

- Kept commands:
  - `O.0-1 Tested: Generate head-only data without viewer`
    `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_motion demo_clean_piper_motion 0`
  - `O.0-2 Tested: Motion viewer for the motion baseline`
    `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_pick_diverse_bottles_piper_motion_viewer.sh`
  - `O.0-3 Scene only: viewer without motion`
    `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50`
  - `O.0-4 Diagnostic only: original IK/planning path`
    `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && timeout 120s bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_ik_orig_tcp 0`
- Removed/downgraded:
  - `pick_diverse_bottles_piper demo_clean_piper_calibrated` is no longer recommended as a collection command; `tmux gen1-1/gen1-2` shows it keeps failing with `Objects is unstable` and `target_pose cannot be None for move action`.

## 2026-06-04 (O.0 Viewer Command Fix And Axes Smoke Checks)

- Changed command:
  - `run_pick_diverse_bottles_piper_motion_viewer.sh` now calls `view_pick_diverse_bottles_piper_motion.py "$@"` instead of `script/collect_data.py`.
- Added/updated entrypoints:
  - `view_pick_diverse_bottles_piper_motion.py`
  - `view_pick_diverse_bottles_piper_scene.py --show_axes --hold`
- Recommended commands:
  - Motion viewer:
    `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_pick_diverse_bottles_piper_motion_viewer.sh`
  - Motion viewer smoke:
    `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && DISPLAY=:1.0 timeout 120s bash run_pick_diverse_bottles_piper_motion_viewer.sh --seed 0 --max_seed_tries 3 --hold 0`
  - Scene viewer:
    `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50`
  - Scene viewer smoke:
    `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && DISPLAY=:1.0 timeout 90s python view_pick_diverse_bottles_piper_scene.py --seed 0 --max_seed_tries 3 --hold 0`
- Notes:
  - The scene viewer uses `skip_planner=True`, so it does not enter Curobo warmup.
  - The motion viewer searches for a stable seed and executes `play_once()` once on every run, so it is not short-circuited by old `seed.txt` progress.
  - Debug axes are shown by default on the two bottle centers and the left/right place targets. Red/green/blue are local +X/+Y/+Z, and the small white cube is the origin.

## 2026-06-04 (O.0 Motion Stage Axes And Stage Log Commands)

- Updated command notes:
  - `bash run_pick_diverse_bottles_piper_motion_viewer.sh` now prints `[piper-motion][stage]` stage logs and `[piper-motion][target-axis]` EE target-axis positions.
  - The small white cube is only the axis-marker origin, not the Piper base.
- Added a scene-only command for inspecting the motion-baseline scene and staged target axes without executing motion:
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python view_pick_diverse_bottles_piper_scene.py --task_name pick_diverse_bottles_piper_motion --task_config demo_clean_piper_motion_viewer --seed 0 --max_seed_tries 50`
- Validation command:
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && DISPLAY=:1.0 timeout 120s bash run_pick_diverse_bottles_piper_motion_viewer.sh --seed 0 --max_seed_tries 10 --hold 0`
- Notes:
  - The O.0 motion-baseline bottle range is now overridden in `envs/pick_diverse_bottles_piper_motion.py` as `left=x[-0.30,-0.18],y[-0.20,-0.10]` and `right=x[0.30,0.46],y[-0.20,-0.10]`.
  - This range is closer to the current calibrated Piper/Pika FK workspace than the original ALOHA/AgileX `y=[0.03,0.23]`, but it is still not the final bottle-aligned grasping strategy.

## 2026-06-08 (Mode N-1 FoundationPose Target Preview Command)

- Changed command:
  - The `# 0608` Mode N batch command in `COMMAND_LIBRARY.zh.md` now writes `--output_root` to `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-1_foundation_pose_viewer`.
- Recommended command:
  - `for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh --gpu 1 --ids 0 1 2 3 4 --continue_on_error --tasks $TASK --foundation_pose_retreat_m 0.03 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-1_foundation_pose_viewer; done`
- Meaning:
  - Mode N uses the FoundationPose object world position plus the human gripper rotation matrix.
  - `rank_previews/*.png` shows the 2D C-shaped gripper and local axes: left arm blue, right arm orange, X=red, Y=green, Z=blue.
  - The `pose_world_wxyz` field name is historical; the wrapper and planner use `[x, y, z, qw, qx, qy, qz]`.
- Validation command:
  - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh --gpu 1 --ids 2 --continue_on_error --tasks pick_diverse_bottles --foundation_pose_retreat_m 0.03 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-1_foundation_pose_viewer`

## 2026-06-09 (Mode N-3 Projection Status And Viewer Overlay Command)

- Changed command:
  - The Mode N batch output root in `COMMAND_LIBRARY.zh.md` is now `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-3_foundation_pose_viewer`.
  - `run_plan_keyframes_foundation_pose_piper_d435.sh` now supports `--debug_viewer_overlay` for showing target axes and top-1 C-gripper actors.
- Batch command:
  - `for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh --gpu 1 --ids 0 1 2 3 4 --continue_on_error --tasks $TASK --foundation_pose_retreat_m 0.03 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-3_foundation_pose_viewer; done`
- Viewer demo command:
  - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh --gpu 2 --ids 1 --viewer --viewer_wait_at_end 1 --tasks pick_diverse_bottles --debug_viewer_overlay --foundation_pose_retreat_m 0.03 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-3_foundation_pose_debug_viewer`
- Notes:
  - `rank_previews/*.png` prints `proj=inside/offscreen/behind_camera`; if the C-gripper is not visible, check this field first to determine whether the target is behind the head camera or outside the image.

## 2026-06-09 (Mode N-4 Pose Order Fix Command)

- Changed command:
  - The Mode N batch output root is now `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-4_foundation_pose_order_fix`.
  - The Mode N viewer demo output root is now `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-4_foundation_pose_debug_viewer`.
  - `--debug_viewer_overlay` now shows target axes, top-1 C-gripper actors, camera axes, and the SAPIEN camera frustum.
- Batch command:
  - `for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh --gpu 1 --ids 0 1 2 3 4 --continue_on_error --tasks $TASK --foundation_pose_retreat_m 0.03 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-4_foundation_pose_order_fix; done`
- Viewer demo command:
  - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh --gpu 2 --ids 1 --viewer --viewer_wait_at_end 1 --tasks pick_diverse_bottles --debug_viewer_overlay --foundation_pose_retreat_m 0.03 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-4_foundation_pose_debug_viewer`

## 2026-06-09 (Mode N-5 Pregrasp/Grasp Retreat Parameters)

- Changed command:
  - Updated the Mode N block in `COMMAND_LIBRARY.zh.md` to `N-5_pregrasp15_grasp8`.
  - Batch and viewer commands now explicitly use `--foundation_pose_retreat_m 0.08 --approach_offset_m 0.07`.
- Parameter meaning:
  - `grasp target = object center - 0.08m * local +Z`.
  - `pregrasp = grasp - 0.07m * local +Z`, so the total pregrasp retreat is 0.15 m and pregrasp advances 0.07 m to grasp.
- Notes:
  - Documented the `cartesian_interp_ik` intermediate interpolation behavior: linear TCP position interpolation, TCP-orientation Slerp, and waypoint-by-waypoint IK. A mid-stage dip/twist usually comes from an IK wrist/elbow branch switch.

## 2026-06-10 (Mode N-6 Viewer Debug Command)

- Changed command:
  - Updated the recommended Mode N command to `N-6_pregrasp17_grasp10`.
  - Parameters are now `--foundation_pose_retreat_m 0.10 --approach_offset_m 0.07`.
- Parameter meaning:
  - The grasp target is 10 cm from the object center.
  - Pregrasp retreats another 7 cm behind grasp, for a total retreat of 17 cm.
- Usage notes:
  - The viewer command keeps `--debug_viewer_overlay` to show target axes, top-1 C-gripper actors, camera axes, and frustums.
  - The docs note that current Mode N does not enable the R1/AnyGrasp roll/up constraints. Enforcing wrist-camera-up or rejecting bad roll variants requires a later planner/wrapper change.

## 2026-06-10 (New Piper IK V1-V4 Commands)

- New configs: `demo_piper_ik_seq_v1`, `demo_piper_ik_seq_v2`, `demo_piper_ik_seq_v3`, and `demo_piper_ik_seq_v4`.
- Viewer: `python view_pick_diverse_bottles_piper_ik_motion.py --ik_version vN --seed 0 --max_seed_tries 50 --require_success 1`.
- Collection: `bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_vN 0`.
- Headless: use `unset DISPLAY` with `--render_freq 0 --show_axes 0 --hold 0`.
- Documentation: updated `COMMAND_LIBRARY.zh.md` and added `agent-read/COMMANDS/piper_ik_cartesian.en.md`.

## 2026-06-11 (Isolated O.1 Foundation Collection Command)

- Viewer adds `--task_config`, `--foundation_id`, and `--foundation_frame`.
- Added `collect_foundation_piper_ik.sh <v1-v4> <id> [frame] [gpu]`, generating isolated config and output names by version, ID, and frame.
- Updated O.1 in `COMMAND_LIBRARY.zh.md` and removed the unsafe base-config `sed -i` batch workflow.
- Added `agent-read/COMMANDS/piper_ik_foundation.en.md`.

## 2026-06-11 (O.1.1 / O.1.2 Commands)

- Viewer adds `--foundation_mode o1|o1.1|o1.2`.
- Collection wrapper format is now `bash collect_foundation_piper_ik.sh <v1-v4> <id> [frame] [gpu] [o1|o1.1|o1.2]`.
- O.1.1/O.1.2 output names include `o1_1` / `o1_2` and do not share O.1 `id<id>_frame<frame>` directories.
- Example: `bash collect_foundation_piper_ik.sh v1 0 0 0 o1.2`.

## 2026-06-11 (Mode N-7 Commands)

- `run_plan_keyframes_foundation_pose_piper_d435.sh` now supports:
  - `--foundation_pose_action_orientation_source keyframe|grasp`
  - `--foundation_pose_keep_top_axis_up 0|1`
  - `--foundation_pose_top_axis x|y`
  - `--dual_stage_freeze_reached_arms_on_replan 0|1`
- The recommended N-7 command uses `--foundation_pose_action_orientation_source grasp --dual_stage_freeze_reached_arms_on_replan 1` and writes to `N-7_action_grasp_rot_freeze`.
- Updated batch, viewer, and smoke commands in `COMMAND_LIBRARY.zh.md` and `agent-read/COMMANDS/piper_anygrasp_keyframes.en.md`.

## 2026-06-11 (Mode M-0611 Human Replay Command)

- `run_plan_keyframes_human_replay_piper_d435.sh` now exposes and forwards IK seed, joint-continuity, cubic interpolation, action-orientation, reached-arm freezing, and failure-exit settings.
- Recommended defaults are `joint_interp`, `joint_trajectory_interpolation=cubic`, `ik_solution_selection=joint_continuity`, six 0.05-rad perturbed seeds, `action_orientation_source=grasp`, and `reach_pos_tol_m=0.04`.
- Updated L15.20 and Mode M in `COMMAND_LIBRARY.zh.md`. Use `--ids 0 1 2 --continue_on_error` for cross-ID diagnosis.

## 2026-06-11 (O.1.2 Wrist Batch Collection Command)

- New Foundation wrapper format: `bash collect_foundation_piper_ik.sh <v1-v4> <id> [frame] [gpu] [mode] [run_tag]`.
- Use a new tag such as `wrist0515` so resume logic does not skip an existing head-only HDF5.
- Full O.1.2 commands now apply a 600-second timeout per ID and a version-specific failure log. V1-V4 remain assigned to GPUs 0/1/2/3.
- New outputs are `episode0_succ_left_camera.mp4` and `episode0_succ_right_camera.mp4`.

## 2026-06-11 (Wrist Viewer And ID-To-Episode Index Commands)

- Viewer: add `--wrist_preview 1` to the Foundation O.1.2 command for a live left/right wrist RGB mosaic.
- Batch collection: use a new `RUN_TAG=wrist0515_simfix` after the frame correction so old wrist outputs are not reused.
- Video indexing: added `python script/index_foundation_piper_ik_videos.py --version v4 --mode o1.2 --run-tag wrist0515_simfix --output-video-dir <DIR> --method symlink --dry-run`.
- Replacement rule: existing `episode<ID>_*.mp4` files are conflicts by default. Only explicit `--replace-episode` replaces that ID in the destination directory.
- Omit `--run-tag` for the current untagged V4/O.1.2 legacy outputs. The dry run found indexable IDs 0-8 and destination conflicts for IDs 0-4 in `demo_piper_ik_v4_3/video`.

## 2026-06-15 (O.1.2.1 Wrist Tuning And Self-Contained Batch Commands)

- New recommended tag: `wrist_o121_verified_0615`. Each tmux-pane command independently sets `RUN_TAG`, creates `data/tmp`, and initializes its failure log so an empty variable cannot write to untagged outputs.
- Viewer options added: `--wrist_left_forward_offset_m`, `--wrist_right_forward_offset_m`, `--wrist_left_roll_deg`, and `--wrist_right_roll_deg`.
- Defaults in all four Foundation YAML files are left `0.125/-15` and right `0.11/-60`.
- Full commands and the URDF/0515 analysis are in O.1.2.1 of `COMMAND_LIBRARY.zh.md` and `agent-read/COMMANDS/piper_ik_foundation.en.md`.

## 2026-06-15 (Viewer Wrist Debug Video Command)

- Added `--wrist_debug_record`, `--wrist_debug_tag`, `--wrist_debug_dir`, and `--wrist_debug_fps`.
- Append `--wrist_debug_record 1 --wrist_debug_tag <parameter-name>` to a tuning viewer command.
- Outputs are `wrist_debug_left.mp4`, `wrist_debug_right.mp4`, `wrist_debug_mosaic.mp4`, and `wrist_debug_config.json`.

## 2026-06-15 (Headless Formal Wrist Override Command)

- The formal wrapper accepts `WRIST_LEFT_FORWARD_OFFSET_M`, `WRIST_RIGHT_FORWARD_OFFSET_M`, `WRIST_LEFT_ROLL_DEG`, and `WRIST_RIGHT_ROLL_DEG`.
- All four variables are required together. Positional arguments remain `version id frame gpu mode run_tag`.
- Debug recorder output now uses H.264/yuv420p/faststart for direct VS Code playback.

## 2026-06-15 (Viewer Command With Wrist/Head Frustums)

- Added `--show_camera_frustums 1`, recommended with `--render_freq 1 --show_axes 1 --wrist_preview 1 --hold 1`.
- In tmux `gen1`, restore the GUI after `unset DISPLAY` with `export DISPLAY=:1.0`.
- Use a timestamped debug tag such as `TAG="o121_v1_viewer_$(date +%Y%m%d_%H%M%S)"` to avoid overwrite rejection.

## 2026-06-16 (Two Live Motion Commands)

- Mode 1: `--render_freq 1 --show_camera_frustums 1 --wrist_preview 0` for live SAPIEN plus camera frustums only.
- Mode 2: switch to `--wrist_preview 1` to show live SAPIEN and dual-wrist RGB concurrently.
- When not recording, both modes omit `TAG`, `--wrist_debug_record`, and `--wrist_debug_tag`.

## 2026-06-16 (Wrist Forward-Axis Diagnostic Command)

- Added `python script/diagnose_piper_wrist_camera_axes.py`.
- It reports per-side camera forward, opening-plane error, angles to Pika physical `+X` and legacy debug `+Z`, plus the tiny yaw needed to zero only the `Y` component.

## 2026-06-16 (Viewer Wrist Yaw Parameters)

- Added viewer parameters: `--wrist_left_yaw_deg` and `--wrist_right_yaw_deg`.
- Current values for zeroing the finger-opening `Y` component are `--wrist_left_yaw_deg 0.182 --wrist_right_yaw_deg 0.840`.
- `roll_deg` still rotates the image about the optical axis; `yaw_deg` adjusts extrinsic orientation about parent-frame `+Z`.


## 2026-06-16 (Foundation Gripper Standoff Commands)

- New viewer option: `--foundation_grasp_standoff_m <M>`, for example `--foundation_grasp_standoff_m 0.105`.
- New collection-wrapper environment variable: `FOUNDATION_GRASP_STANDOFF_M=0.105 bash collect_foundation_piper_ik.sh v1 0 0 0 o1.2 standoff105`.
- Default Foundation V1-V4 YAML files now use `foundation_grasp_standoff: 0.105`, so the new distance is active without explicit overrides.


## 2026-06-16 (Viewer Wrist Pitch/Lateral Command)

- Added viewer options: `--wrist_left_pitch_deg`, `--wrist_right_pitch_deg`, `--wrist_left_lateral_offset_m`, and `--wrist_right_lateral_offset_m`.
- Pitch rotates about gripper/link6 parent-frame `+Y`; positive values tilt the camera forward axis downward toward the nominal tip.
- Lateral offset translates along gripper/link6 parent-frame `+Y`; the right camera is currently negative in Y, so use positive values to move it toward center.
- Recommended first trial: `--wrist_left_pitch_deg 15 --wrist_right_pitch_deg 15 --wrist_right_lateral_offset_m 0.0067`.


## 2026-06-16 (Viewer Wrist Centerline Command)

- For a camera at gripper centerline `Y=0`, use `--wrist_left_lateral_offset_m -0.0207 --wrist_right_lateral_offset_m 0.0274`.
- To preserve raw/current calibrated positions, omit lateral parameters.
- `+X` is wrist-to-tip forward; `+Y` is finger-opening/lateral direction.


## 2026-06-16 (Foundation Real-Grasp Debug Commands)

- Added viewer options: `--foundation_collision_mode`, `--foundation_collision_radius_padding_m`, `--foundation_grasp_assist`, `--foundation_grasp_require_contact`, `--foundation_capture_radial_tolerance_m`, and `--foundation_grasp_assist_max_distance_m`.
- The daily success tier uses `foundation_grasp_standoff_m=0.14` with the current wrist v2 parameters.
- The contact-gated tier uses `--foundation_collision_mode cylinder_proxy --foundation_grasp_require_contact 1`.
- The pure-physics observation tier uses `--foundation_grasp_assist 0 --require_success 0`.

Update: the verified-v2 daily viewer command now includes `--foundation_capture_radial_tolerance_m 0.08 --foundation_grasp_assist_max_distance_m 0.16` to match the fingertip grasp geometry from `foundation_grasp_standoff_m=0.14`.


## 2026-06-16 (Verified Collection And O.2 pnp_tray Commands)

- Added the formal collection wrapper: `bash collect_foundation_piper_ik_verified.sh <pick_diverse_bottles|pnp_tray> <v1|v2|v3|v4> <foundation_id> [gpu_id] [run_tag]`.
- `DRY_RUN=1` writes the generated config without running collection.
- `pick_diverse_bottles` uses `foundation_grasp_standoff=0.14`; `pnp_tray` uses `foundation_grasp_standoff=0.105`.
- Added viewer/collection task name: `pnp_tray_piper_ik_foundation`.
- The minimal O.2 viewer command uses `--task_name pnp_tray_piper_ik_foundation --foundation_mode o1.2 --foundation_grasp_standoff_m 0.105`.


## 2026-06-17 (O.2 object-keyframe And Pregrasp Clearance Commands)

- Added viewer option: `--foundation_action_target_source hand_ee|object_keyframe`.
- Added viewer option: `--foundation_pregrasp_clearance_m <M>`; `0` is the default no-avoidance path, and a positive value inserts a lifted waypoint before pregrasp.
- `pnp_tray_piper_ik_foundation` defaults to `object_keyframe`; pick_diverse still defaults to `hand_ee`.
- Wrapper environment variable: `FOUNDATION_PREGRASP_CLEARANCE_M=0.06 bash collect_foundation_piper_ik_verified.sh pnp_tray v1 0 0 o2_pregrasp_clearance006`.


## 2026-06-18 (L16 Human Replay Wrist Batch Commands)

- Added `COMMAND_LIBRARY.zh.md` L16.1 with `pick_diverse_bottles` viewer-debug and no-viewer batch collection commands.
- The batch command uses `--ids 0-101 --continue_on_error` for 102 episodes and writes to `.../L16_human_replay_clean/pick_diverse_bottles/foundation_input_<ID>/`.
- Viewer and no-viewer commands use the same wrist extrinsics: left/right forward `-0.04/-0.01`, roll `14.635/-44.649`, yaw `0.182/0.840`, pitch `-90/-90`, and lateral `-0.0207/0.0274`.
- The wrapper now forwards wrist preview, camera frustum, and camera RGB-axis visualization arguments only in `--viewer` mode; no-viewer batch output keeps clean videos/obs.


## 2026-06-24 (Added L16 Six-Task Repaint Commands)

- New command locations: `COMMAND_LIBRARY.zh.md` I3.6/I3.7.
- I3.6: debug command for five available IDs per six-task run; it first creates human+object Stage-1 inpaint backgrounds, then visible-reinit repaints robot+object pixels from L16 `head_cam_plan.mp4`.
- I3.7: full batch command that enumerates `L16_human_replay_clean/<TASK>/foundation_input_<ID>/head_cam_plan.mp4` and skips existing Stage-1 BGs and `final_repainted.mp4` outputs.
- Key outputs: `results_repaint_piper_h2_l16/stage1_human_object/<TASK>/id_<ID>/...` and `results_repaint_piper_h2_l16_visible_reinit/e0_robot_object/<TASK>/id_<ID>_l16/final_repainted.mp4`.


## 2026-06-24 (Fix L16 Repaint Command Details)

- I3.6/I3.7 no longer use `set -u`, so they are compatible with the `inpainting-sam3-dino3` conda activation scripts.
- I3.6/I3.7 Stage-1 now explicitly creates the `stage1_human_inpaint` directory before writing `removed_w_mask_rgb_<ID>.mp4`.

## 2026-06-24 13:20 +08 - L16 Visualization Montage Commands

- Added `P. Visualization: L16 HaMeR / Foundation / repaint montage comparison` to the end of `COMMAND_LIBRARY.zh.md`.
- Added single-id tmux test command: `python3 code_painting/make_l16_repaint_montage.py --task pick_diverse_bottles --id 0 --overwrite`.
- Added batch guidance: use `--ids 0-4` for `pick_diverse_bottles/place_bread_basket/stack_cups/pnp_tray`, `--ids 1-5` for `handover_bottle`, and `--ids 7-11` for `pnp_bread`.

## 2026-06-24 (COMMAND_LIBRARY I3.6 White-Background Inverted-Mask Repaint)

- New command location: `COMMAND_LIBRARY.zh.md` I3.6.
- Purpose: for L16 six-task debug/batch runs, prompt the white background instead of robot+object, invert the mask, then compose the L16 source video's non-white-background pixels onto the Stage-1 background.
- Important parameters: `RUN_MODE=debug|batch`, `BG_MODE=hand_only|human_object`, `MASK_IDX`, `WHITE_PROMPT`, `COMPOSITE_ERODE`, `BLEND_ALPHA_SIGMA`.
- Output: `results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object/<TASK>/id_<ID>_l16_whitebg_<BG_MODE>/final_repainted.mp4`.

## 2026-06-24 (Fix COMMAND_LIBRARY I3.6 Mask Frame Directory)

- Fixed the compose input directory in the I3.6 white-background inverted-mask repaint command: `mask/` -> `mask_head_cam_plan/`.
- Impact: the previous debug command completed only the first SAM visualization output and then exited during compose; the fixed command can create `final_repainted.mp4` and continue to later task/id jobs.

## 2026-06-24 (COMMAND_LIBRARY I3.6 Defaults to Human-Object Background)

- Changed the I3.6 white-background inverted-mask repaint debug first line to `RUN_MODE=debug BG_MODE=human_object OVERWRITE=1 bash <<'BASH'`.
- Changed the I3.6 batch first line to `RUN_MODE=batch BG_MODE=human_object OVERWRITE=0 bash <<'BASH'`.
- Removed `cups` from the `stack_cups` object prompt to avoid selecting the green cup.

## 2026-06-24 (COMMAND_LIBRARY stack_cups Prompt Refinement)

- Changed the `stack_cups` inpaint/repaint prompt to `left light pink cup, right dark red cup`; it no longer uses the generic `red cup` / `cups` descriptions.

## 2026-06-24 (COMMAND_LIBRARY I3.6 Proportional BG Stretch)

- I3.6 repaint compose now outputs robot/mask frame count and proportionally samples shorter Stage-1 BG videos to match duration.
- Stage-2 with `--save_removed_video 0` now pairs with the script change that skips unnecessary STTN inpainting.

## 2026-06-24 (COMMAND_LIBRARY I3.6.1 Per-Task Parallel Commands)

- Added I3.6.1: use `run_l16_stage1_human_object_task.sh` and `run_l16_whitebg_repaint_task.sh` to run Stage-1 inpaint and Stage-2 repaint per task in parallel.

## 2026-06-24 (I3.6.1 GPU Index Fix)

- Changed the I3.6.1 `pnp_tray` example from non-existent GPU4 to a second-wave GPU0 command. This machine exposes GPUs 0-3, so five tasks cannot all run simultaneously on unique GPUs.
