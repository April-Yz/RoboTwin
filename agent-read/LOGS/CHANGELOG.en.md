# CHANGELOG.en

## 2026-06-18 (Restore Mode M/L Wrist Extrinsic Default Behavior)

- Reverted the previous incorrect wrist roll documentation:
  - `507782e` reverted `b297f32`, removing the `+14.635/-44.649` recommended-roll note.
- Updated `code_painting/run_plan_keyframes_human_replay_piper_d435.sh`:
  - `PIPER_CALIBRATION_BUNDLE` now defaults to empty instead of loading the 0515 bundle.
  - Without an explicit `--piper_calibration_bundle`, the wrapper no longer forwards any `--wrist_*` tuning to the Python planner.
  - If the outer command includes `--wrist_*` without a bundle, the wrapper prints a warning and ignores those parameters.
- Reason:
  - With the 0515 bundle and O.1 tuning as Mode M/L defaults, id1 extracted wrist frames looked at the gripper back shell or empty background, so the current L wrist parent-frame/axis convention cannot directly reuse the O.1 parameters.
- Validation:
  - `bash -n code_painting/run_plan_keyframes_human_replay_piper_d435.sh` passed.
  - A dry-run with `--wrist_*` and no bundle printed the warning.
  - The no-viewer id1 probe under `/tmp/robo_wrist_restored_baseline_probe/pick_diverse_bottles/foundation_input_1` executed successfully, and logs confirmed that no calibration bundle was loaded.

## 2026-06-02 (Mode O First-Frame FoundationPose Direct Strategy Grasp)

- Follow-up viewer fix:
  - The user's Mode O viewer run printed `CUDA_VISIBLE_DEVICES=2`, then SAPIEN reported `Renderer does not support display`.
  - Cause: the wrapper used `env -u CUDA_VISIBLE_DEVICES` in viewer mode, but `plan_first_frame_foundation_pick_diverse_bottles.py` restored `CUDA_VISIBLE_DEVICES` from `--gpu 2` before invoking the planner.
  - Fix: the wrapper now passes `--gpu -1` in viewer mode, and the Python entrypoint explicitly removes `CUDA_VISIBLE_DEVICES` when `enable_viewer=1`.
  - Validation: `DISPLAY=:1.0 bash code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 0 --viewer --viewer_wait_at_end 0 --continue_on_error --output_root /tmp/mode_o_viewer_env_check` now logs `CUDA_VISIBLE_DEVICES=None` and successfully prints `[viewer] interactive viewer created`.
- Added the Mode O comparison experiment for `pick_diverse_bottles`:
  - `code_painting/plan_first_frame_foundation_pick_diverse_bottles.py`
  - `code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh`
- Design logic:
  - Does not use manual keyframes, human hand orientation, or AnyGrasp candidates.
  - Reads both bottle world positions from frame 0 of `foundation_replay_d435/foundation_input_<ID>/multi_object_world_poses.npz`.
  - Assigns the left bottle to the left arm and the right bottle to the right arm, then builds fixed outside-in horizontal side-grasp targets.
  - Reuses the existing Piper planner through `--reuse_plan_summary_json` for pregrasp/grasp/close/action.
  - Defaults to a `0.08m` pregrasp distance, matching `envs/pick_diverse_bottles.py`'s `pre_grasp_dis=0.08`.
  - Defaults to the env placement targets: left `[-0.06,-0.105,1.0]`, right `[0.06,-0.105,1.0]`.
- Code note:
  - The `pose_world_wxyz` key name in `multi_object_world_poses.npz` is misleading. The actual array order is `[x, y, z, qw, qx, qy, qz]`, and the new script follows the planner convention.
- Documentation:
  - Added section O to `COMMAND_LIBRARY.zh.md`.
  - Synced Mode O notes into `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` and `.en.md`.
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_first_frame_foundation_pick_diverse_bottles.py` passed.
  - `bash -n /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh` passed.
  - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --ids 0 --dry_run` passed.
  - A no-viewer `pick_diverse_bottles id0` smoke run completed under `code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/first_frame_foundation_smoke/pick_diverse_bottles/foundation_input_0`; both arms reached pregrasp, but grasp did not reach for both arms, so the default safety gate skipped close/action.

## 2026-05-28 (Cartesian Partial Prefix Execution)

- Added a Cartesian waypoint partial-prefix execution diagnostic:
  - `plan_anygrasp_keyframes_r1.py` now supports `--execute_partial_cartesian_plan`.
  - The Piper/R1 URDFIK renderers can return `status=Partial` plus the solved waypoint prefix when `cartesian_interp_ik` fails at an intermediate waypoint.
  - Execution now runs plans that are `Success` or executable `Partial`; `Partial` is not counted as reached.
  - The six-task wrapper forwards `--execute_partial_cartesian_plan`.
  - `COMMAND_LIBRARY.zh.md` now includes L15.11 viewer/no-viewer debug commands and a note on position-priority vs orientation-priority IK.
- Validation:
  - Related Python files passed `py_compile`.
  - The wrapper passed `bash -n`.
  - L15.11 bash blocks passed `bash -n`.

## 2026-05-28 (D435 Preview No-Offset Main And Planner-Selected Image)

- Updated the D435 AnyGrasp preview convention per user request:
  - L15.9 now regenerates the main `anygrasp_h2o_preview_d435` previews with no offset: `--candidate_target_local_x_offset_m 0.0`.
  - L15.10 now writes the offset -5cm comparison output to `anygrasp_h2o_preview_d435_offset_minus_5cm_compare`.
  - `render_anygrasp_ranked_preview.py` now writes `frame_XXXXXX_left_right_planner_selected_orientation_rank1.png`, showing only the downstream planner's default `orientation rank1` selection.
  - Documentation now explains orientation vs fused ranking and the colors: candidate left is blue-ish, candidate right is orange-ish; human reference left is green and right is purple.
- Validation:
  - `python3 -m py_compile code_painting/render_anygrasp_ranked_preview.py` passed.
  - The L15.9/L15.10 bash blocks passed `bash -n`.

## 2026-05-28 (Raw/No-Offset D435 AnyGrasp Comparison)

- Appended L15.10 to the end of `COMMAND_LIBRARY.zh.md`:
  - Uses `--candidate_target_local_x_offset_m 0.0` to generate an independent comparison directory: `anygrasp_h2o_preview_d435_no_offset_compare`.
  - Adds a summary comparison script to inspect the difference between the raw/original candidate and the offset visual/planner target.
- Validation:
  - The L15.10 bash blocks passed `bash -n`.

## 2026-05-28 (Copy-Safe D435 AnyGrasp Commands)

- Appended L15.9 to the end of `COMMAND_LIBRARY.zh.md`:
  - Step 1 uses `bash <<'BASH'` to regenerate six-task D435 previews/summaries without zsh line-wrap argument loss.
  - Step 2 provides the no-viewer D435 planner/replay command.
  - Step 3 provides the viewer + `--visualize_targets` gripper-target visualization replay command.
- Validation:
  - The L15.9 bash blocks passed `bash -n`.

## 2026-05-28 (D435 AnyGrasp Preview/Planner Mapping Fix)

- Fixed an AnyGrasp D435 preview vs planner target visualization mismatch:
  - `render_anygrasp_ranked_preview.py` now draws grasp wireframes with the remapped/post-rotated/local-X-offset `visual_translation_cam/visual_rotation_matrix`.
  - The summary still keeps the raw `translation_cam/rotation_matrix` and now also writes `visual_translation_cam/visual_rotation_matrix`, separating the raw AnyGrasp candidate from the actual planner target.
  - Confirmed that `pick_diverse_bottles id0 frame 38` uses D435 rank1 left candidate `16` and right candidate `11`; the default wide-FOV preview uses different candidate ids and should not be mixed with D435 planner runs.
- Validation:
  - `python3 -m py_compile code_painting/render_anygrasp_ranked_preview.py code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_piper.py` passed.
  - A single-id D435 preview was regenerated under `code_painting/anygrasp_h2o_preview_d435_offsetfix_debug/pick_diverse_bottles/foundation_input_0`, and the summary exposes raw/visual/world coordinates.

## 2026-05-28 (D435 AnyGrasp Supports Per-Arm Keyframes)

- Additional diagnosis:
  - The `tmux anygrasp-view-18` viewer batch was repeatedly interrupted with `Ctrl-C`; existing outputs show that in strict sync mode, `pick_diverse_bottles id0` has a left-arm plan `Fail` while the right arm can have `Success`, but `dual_require_all=1` skips the whole stage, so debug metrics show zero TCP displacement.
  - Found that the script had been changed to `--execute_interp_steps 2400 --joint_command_scene_steps 1000 --settle_steps 300 --joint_target_wait_steps 250`, which makes the viewer appear stuck at a waypoint. Restored the default R1/V7-style cadence: 24/10/30/25.
  - Verified that execution does change pose with `--allow_partial_dual_stage --print_pose_every 5`: for `pick_diverse_bottles id0`, the right TCP moved from about `(0.561,-0.044,0.931)` to `(0.187,0.224,1.018)`, and stdout prints `[exec-pose]`.
  - The viewer behavior where waypoints appear but the arm does not move before the gripper closes was caused by the old guard only blocking keyframe-2 action via `--require_keyframe1_reached_before_action`; it did not block `close_gripper`. When pregrasp/grasp planning failed, the stage was skipped but close still ran.
  - Reproduced `stack_cups id0` with `--debug_stop_after_keyframe1`; it already fails at the first keyframe, ruling out close/action/keyframe-2 as the cause.
  - The failure is caused by intermediate `cartesian_interp_ik` waypoint IK failures, not by too-small `settle_steps` or `joint_target_wait_steps`: pregrasp fails at left waypoint 13/23 and right waypoint 28/48; grasp fails at left waypoint 16/28 and right waypoint 25/45.
  - Because `--dual_stage_require_all_plans 1` is enabled, any arm plan marked `Fail` skips the whole synchronized dual-arm stage, so the output video appears to execute almost no waypoints.
- Code updates:
  - Added `--debug_stop_after_keyframe1` to `plan_anygrasp_keyframes_r1.py`; it executes only init -> pregrasp -> grasp, without closing the gripper or entering keyframe 2.
  - Added `--require_keyframe1_reached_before_close` to `plan_anygrasp_keyframes_r1.py`; when enabled, the gripper does not close unless the first-keyframe grasp reached.
  - Added `--print_execution_pose_every` to `plan_anygrasp_keyframes_r1.py`; execution can now print TCP/EE world positions every N trajectory steps.
  - Added `[plan-fail]` logging in dual stages to print the failure reason and failed Cartesian waypoint.
  - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` now passes `--require_keyframe1_reached_before_close 1` by default and supports `--viewer`, `--output_root`, `--debug_stop_after_keyframe1`, `--trajectory_mode`, `--cartesian_auto_step_m`, `--joint_interp_waypoints`, `--replan_attempts`, `--allow_partial_dual_stage`, and `--print_pose_every`.
- Documentation updates:
  - Added L15.6 to `COMMAND_LIBRARY.zh.md` with six-task five-episode no-viewer, viewer, and first-keyframe debug commands.
  - Synchronized L15.6 in `agent-read/COMMANDS/piper_anygrasp_keyframes.*.md`.
- Validation:
  - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_piper.py`
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --max_per_task 1 --tasks stack_cups`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --max_per_task 1 --tasks stack_cups --viewer --output_root /tmp/viewer_out`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --max_per_task 1 --tasks stack_cups --debug_stop_after_keyframe1`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 1 --continue_on_error --tasks stack_cups --debug_stop_after_keyframe1`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 1 --continue_on_error --tasks stack_cups --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_close_guard_check` verified that first-keyframe failure now prints `stopping before close_gripper` and no longer closes the gripper.
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 1 --continue_on_error --tasks pick_diverse_bottles --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_posemove_debug` verified that stdout pose changes with waypoint execution.

- Root cause:
  - The user pasted the old L15.4 long command directly into zsh, triggering `zsh: command not found: mapfile`; `mapfile` is a bash builtin, leaving `IDS` empty and causing a bogus `foundation_input_` lookup.
  - J0.1 scans `seq 0 120`, which exceeds the real episode count for several tasks, so many `MISS` lines are expected.
  - `place_bread_basket/stack_cups/handover_bottle/pnp_bread` had the required base inputs, but the old `run_render_anygrasp_ranked_preview_keyframes_batch.sh` only checked global `keyframes` and ignored the manually annotated `left_keyframes/right_keyframes`, so D435 preview summaries were skipped.
  - `stack_cups id0` uses per-arm D435 summary keyframes: right `[51, 106]`, left `[139, 195]`. The old planner rank preview only rendered the global first two frames `[51, 106]`, so it only showed right-arm candidates and could be misread as a candidate offset issue.
  - Planner `rank_previews` are SAPIEN-rendered 3D grippers, while the J1.1 preview is a projection on the raw D435 image. `approach_offset_m=0.12` does not affect rank previews; the rank preview target only includes the `candidate_target_local_x_offset_m=-0.05` 5 cm TCP compensation.
- Code updates:
  - `render_anygrasp_ranked_preview.py` now computes effective keyframes: when global `keyframes` has fewer than two frames, each arm is filled from `left_keyframes/right_keyframes` plus global `keyframes`.
  - Preview summaries now include `frame_selection.annotated_keyframes_by_arm`, `effective_keyframes`, and `effective_keyframes_by_arm`.
  - `run_render_anygrasp_ranked_preview_keyframes_batch.sh` now checks left/right effective keyframes.
  - `plan_anygrasp_keyframes_r1.py` now prefers arm-specific `effective_keyframes_by_arm` when reusing a preview summary.
  - Fixed the `resolved_map` scope bug in `load_reused_preview_summary()`, avoiding a `NameError` when the D435 planner reuses a preview summary.
  - In dual mode, `selected_keyframes` is now the union of left/right execution sequences, and rank previews are exported for all frames that are actually used for execution.
  - Added `code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` as a zsh-safe six-task D435 planner entrypoint.
  - Added `--continue_on_error` to the six-task script.
  - Added L15.5 to `COMMAND_LIBRARY.zh.md`: a single `stack_cups id0` viewer debug command.
  - Copied the J1.1 D435 source preview images into `anygrasp_plan_keyframes_piper_d435_v1/stack_cups/foundation_input_0/preview_compare_d435/` for side-by-side inspection.
- Validation:
  - `python3 -m py_compile code_painting/render_anygrasp_ranked_preview.py code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_piper.py`
  - `bash -n code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh`
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --max_per_task 1`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 1 --tasks pick_diverse_bottles` generated id0 `plan_summary.json` and head/third videos successfully; execution-level left-arm IK still failed, and dual-sync gating skipped the stage as expected.
  - Rerunning `stack_cups id0` now generates four planner `rank_previews`: right 51/106 and left 139/195.
  - The L15.5 viewer command block passed `bash -n`.
  - The viewer probe failed in the current shell because `DISPLAY=` is empty and SAPIEN reports `Renderer does not support display`; L15.5 now documents that the viewer must be run from a graphical terminal or a correctly forwarded X11/Wayland session.
  - The L15.4 six-task D435 planner command block in `COMMAND_LIBRARY.zh.md` passed `bash -n`.
  - A single `place_bread_basket id0` D435 preview was generated successfully under `/tmp/d435_preview_place_id0_check/foundation_input_0`; its summary records left `[34, 64]` and right `[103, 119]`.

## 2026-05-28 (D435 AnyGrasp Candidate and Planner Path Binding)

- Documentation updates:
  - Added J1.2 and L15.3 to `COMMAND_LIBRARY.zh.md`.
  - Synchronized the D435 planner path requirements in `agent-read/COMMANDS/piper_anygrasp_keyframes.en.md`.
- Purpose:
  - Make the D435 AnyGrasp candidate-selection step and downstream planner use `foundation_replay_d435` together with `anygrasp_h2o_preview_d435`.
  - Prevent downstream D435 replay runs from accidentally reusing the default wide-FOV `anygrasp_h2o_preview` `summary.json`.
- Validation:
  - The new command blocks passed `bash -n`.

## 2026-05-28 (Piper AnyGrasp Final Viewer/No-Viewer Command Split)

- Updated `COMMAND_LIBRARY.zh.md`:
  - Reworded L15.1 to explain that `dual_stage_require_all_plans` and `require_keyframe1_reached_before_action` explicitly match the old V7 behavioral intent: no single-arm stage execution in dual mode, and only proceed to keyframe 2 after keyframe 1 reaches.
  - Added L15.2 with separate no-viewer batch and viewer single-id debug commands.
  - The viewer command does not set `CUDA_VISIBLE_DEVICES=2`; it runs `unset CUDA_VISIBLE_DEVICES` so SAPIEN viewer can see the display GPU.
  - Added a minimal `probe_sapien_viewer.py` probe command to L15.2.
- Validation:
  - Extracted the L15.2 bash blocks and passed `bash -n /tmp/l15_2_blocks.sh`.

## 2026-05-27 (LeRobot Robot/AnyGrasp 25-Episode Subset Commands)

- Updated command documentation:
  - Added L5.1 with six-task "original human head + pure replay action/wrist" processed HDF5 commands.
  - Added L5.2 documenting the human-head + D435 action/wrist command for the new three tasks when only `h2_pure_d435` outputs exist.
  - Added I1.1/I3.5/L8.1 documenting the new-three-task comparison path from Stage-1 background generation through D435 visible-reinit robot repaint to processed HDF5.
  - Added L0 as an overview of the human / robot replay / AnyGrasp replay data pipelines and their run order.
  - Added L10.5 to convert L5.2 `human_head_pure_d435_action` outputs to LeRobot.
  - Added L11.1 to `COMMAND_LIBRARY.zh.md` without changing the existing L11 commands.
  - L11.1 covers six robot datasets: three default-wide `pure_repaint` datasets and three `anygrasp_repaint` datasets.
  - Added explicit zip/rclone dry-run commands for `robot_replay_3task_25ep.zip` and `robot_anygrasp_3task_25ep.zip`.
  - Added L11.2 covering the six H2O tasks used by the FoundationPose sections: `pick_diverse_bottles`, `place_bread_basket`, `stack_cups`, `handover_bottle`, `pnp_bread`, and `pnp_tray`.
  - Added L11.3 explaining that the current LeRobot conversion reads prompts from processed episode `instructions.json` files, not from `convert_aloha_data_to_lerobot_R1.py --task`.
  - Added L6.1/L9.1/L10.4 to make the six-task pipeline explicit: create processed HDF5 first, convert to LeRobot cache second, then use L11/L11.2 to create `_25ep` subsets.
  - Extended L10.4 with six-task human_head_pure_action LeRobot conversion and aligned all six-task prompts to the user-provided descriptions.
- Synchronized:
  - `agent-read/COMMANDS/pi0_h2o_training_data.zh.md`
  - `agent-read/COMMANDS/pi0_h2o_training_data.en.md`
- Validation:
  - Extracted the new L11.1 bash blocks and verified them with `bash -n /tmp/l11_1_blocks.sh`.
  - Extracted the new L11.2/L11.3 bash blocks and verified them with `bash -n /tmp/l11_2_l11_3_blocks.sh`.
  - Extracted the new L6.1/L9.1/L10.4 bash blocks and verified them with `bash -n /tmp/l6_1_l9_1_l10_4_blocks.sh`.
  - Extracted the updated L5.1/L10.4 bash blocks and verified them with `bash -n /tmp/l5_1_l10_4_blocks.sh`.
  - Extracted the new L5.2 bash blocks and verified them with `bash -n /tmp/l5_2_blocks.sh`.
  - Extracted the new I1.1/I3.5/L8.1 bash blocks and verified them with `bash -n /tmp/i1_1_i3_5_l8_1_blocks.sh`.
  - Extracted the new L10.5 bash blocks and verified them with `bash -n /tmp/l10_5_blocks.sh`.

## 2026-05-27 (Piper AnyGrasp Requires Keyframe 1 Before Action)

- Inspected the current `tmux anygrasp` output:
  - Several ids still have `reached=0` in pregrasp/grasp and then proceed to action, which appears as "starting keyframe 2 before keyframe 1 is reached."
  - The new `dual_stage_require_all_plans=1` behavior is active: dual-arm stages print `[dual-plan] skip stage execution...` when either arm fails to plan.
- Updated code:
  - Added `--require_keyframe1_reached_before_action` to `plan_anygrasp_keyframes_r1.py`.
  - When this flag is 1 and first-keyframe grasp is not reached, the second-keyframe action is recorded as `Skipped` and its action trajectory is not executed.
  - `plan_summary.json` now records `require_keyframe1_reached_before_action`, `execute_interp_steps`, `settle_steps`, `reach_pos_tol_m`, `reach_rot_tol_deg`, and key collision/table parameters for easier comparison against V7 runs.
- Updated command documentation:
  - Added L15.1 viewer command to `COMMAND_LIBRARY.zh.md`, matching the old V7 execution cadence: `execute_interp_steps=24`, `joint_command_scene_steps=10`, `settle_steps=30`, `joint_target_wait_steps=25`.
  - L15.1 enables `--require_keyframe1_reached_before_action 1` by default.
- Validation:
  - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_piper.py`
  - `conda run -n RoboTwin_bw python code_painting/plan_anygrasp_keyframes_piper.py --help | rg "require_keyframe1_reached_before_action|dual_stage_require_all_plans"`

## 2026-05-27 (Piper AnyGrasp Batch Command Uses Annotated Keyframes and Dual-Stage Plan Gating)

- Fixed the Piper AnyGrasp dual-sync execution behavior:
  - `plan_anygrasp_keyframes_r1.py` now has `--dual_stage_require_all_plans`, defaulting to `1`.
  - In a dual-arm stage, if either left or right arm fails to plan, neither arm executes the stage, preventing one arm from moving alone.
  - `plan_summary.json` records `dual_stage_require_all_plans`.
- Updated command documentation:
  - Marked the old L13 command in `COMMAND_LIBRARY.zh.md` as historical and noted that `--keyframes 38 78` is not suitable for batch use.
  - Added L15: the id0-id10 batch command no longer passes `--keyframes`; `--reuse_preview_frame_mode annotated_json_keyframes` reads each id's manually annotated keyframes.
  - Documented that `settle_steps/joint_target_wait_steps` only wait for existing joint targets to settle and cannot fix IK failure or unreachable target poses.
- Validation:
  - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_piper.py`
  - `conda run -n RoboTwin_bw python code_painting/plan_anygrasp_keyframes_piper.py --help | rg "dual_stage_require_all_plans|reuse_preview_frame_mode"`

## 2026-05-27 (FoundationPose Replay D435 Intrinsics Commands)

- Updated `COMMAND_LIBRARY.zh.md`:
  - Added C1.2 after C1 for six H2O tasks: `pick_diverse_bottles`, `place_bread_basket`, `stack_cups`, `handover_bottle`, `pnp_bread`, and `pnp_tray`.
  - C1.2 reuses the 0515 Piper head D435 extrinsics and `legacy_r1`, but explicitly adds `--image_width 640 --image_height 480 --fovy_deg 42.499880046655484` to match the D435 RGB intrinsics used by the E2.4 robot pure replay commands.
  - Outputs are written to `foundation_replay_d435` to avoid overwriting the default C1 replay outputs.
- Added command documentation: `agent-read/COMMANDS/piper_foundation_replay.{zh,en}.md`.
- Validation: confirmed that `run_multi_object_pose_r1_npz_batch.sh` forwards `--image_width`, `--image_height`, and `--fovy_deg` through `render_multi_object_pose_r1_npz_batch.py`.

## 2026-05-26 (Piper AnyGrasp IK Thresholds and VS Code Video Compatibility)

- Updated the Piper AnyGrasp planning path:
  - `urdfik.py` now accepts caller-provided IK initial thresholds, maximum relaxed thresholds, and seed count.
  - `render_hand_retarget_piper_dual_npz_urdfik.py` forwards those threshold settings to both Piper URDFIK solvers.
  - `plan_anygrasp_keyframes_r1.py` adds `--urdfik_max_position_threshold_m`, `--urdfik_max_rotation_threshold_rad`, `--urdfik_num_seeds`, and related summary fields in `plan_summary.json`.
  - `plan_anygrasp_keyframes_r1.py` adds `--vscode_compatible_video 1`, which transcodes head/third outputs to H.264 yuv420p faststart when `ffmpeg` is available so VS Code can preview them directly.
  - Fixed `plan-solution` diagnostics under the Piper dual renderer so planned-pose evaluation respects per-arm bases instead of the old R1 single-base FK path.
- Updated command documentation:
  - Added an id0-id10 AnyGrasp batch command to `COMMAND_LIBRARY.zh.md` L14.
  - Updated `agent-read/COMMANDS/piper_anygrasp_keyframes.{zh,en}.md` with the current Piper AnyGrasp logic, recommended parameters, and the frame 38/78 failure interpretation.
- Validation:
  - `python3 -m py_compile code_painting/urdfik.py code_painting/render_hand_retarget_piper_dual_npz_urdfik.py code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_piper.py`
  - `conda run -n RoboTwin_bw python code_painting/plan_anygrasp_keyframes_piper.py --help | rg "urdfik_max_position|urdfik_max_rotation|urdfik_num_seeds|vscode_compatible|urdfik_cartesian_interp_auto_step"`

## 2026-05-25 (H2O pi0 Training Data Conversion Commands)

- Follow-up fix:
  - `process_repainted_headcam_with_wrist.py` now converts quaternions to Euler angles only for `Success` frames with nonzero quaternions, avoiding `ValueError: Found zero norm quaternions in quat` from failed frames.
  - `process_repainted_headcam_with_wrist.py` and `process_repainted_planner_outputs.py` now accept `hand_keyframes_all.json` through `--review-json` and skip episodes with `status=reject/discard/bad`.
  - `--head-video-name` supports `{id}`, so `--head-dir-template '.' --head-video-name 'rgb_{id}.mp4'` can represent "original human head + pure replay action/wrist".
  - Updated `COMMAND_LIBRARY.zh.md` L1/L2/L3 so all three conversions use the manual-review filter.
  - Validation: both conversion scripts pass `py_compile`; `hand_keyframes_all.json` correctly resolves 102 ids for `pick_diverse_bottles`; pure repaint id0 and original-human-head id0 both converted successfully to temporary HDF5 outputs under `/tmp`.
- Follow-up addition:
  - Added `policy/pi0/scripts/visualize_processed_hdf5_episode.py`, which combines `cam_high`, `cam_left_wrist`, and `cam_right_wrist` from one pi0 `processed_data` episode into a review mp4.
  - Added `COMMAND_LIBRARY.zh.md` L5/L6/L7 for three-task conversion commands, episode-count checks, HDF5 structure checks, and review mp4 visualization.
  - Validation: the new visualization script passes `py_compile`; `processed_data/d_pour_blue-48/episode_0` produced `/tmp/pi0_review_probe.mp4` successfully.
- Continued addition:
  - Added `COMMAND_LIBRARY.zh.md` L8/L9 for separate three-task conversion commands covering D435 visible-reinit mode and AnyGrasp planner mode.
  - Validation: representative L8/L9 commands pass `bash -n`.
- Continued addition:
  - Aligned `convert_aloha_data_to_lerobot_R1.py` for H2O processed HDF5 conversion: compressed images are converted to RGB after decoding, episodes with missing cameras are skipped, HDF5 files are sorted, and empty raw directories raise an error.
  - Added `COMMAND_LIBRARY.zh.md` L10 with LeRobot conversion commands for 3 usable modes x 3 tasks.
  - Validation: `convert_aloha_data_to_lerobot_R1.py` passes `py_compile`; representative L10 commands pass `bash -n`.
- Added section L to `COMMAND_LIBRARY.zh.md`, separating three input types:
  - Original human data and existing `policy/pi0/processed_data`
  - Non-AnyGrasp pure replay data
  - AnyGrasp planner replay data
- Documented that pure replay conversion depends on `world_targets_and_status.npz`, `left_wrist_replay.mp4`, and `right_wrist_replay.mp4`.
- Documented that AnyGrasp planner conversion depends on `pose_debug.jsonl`, `left_wrist_cam_plan.mp4`, and `right_wrist_cam_plan.mp4`.
- Recorded the current check result: planner wrist videos are not present under `anygrasp_h2o_plan`, so L3 training conversion needs those wrist inputs to be generated first.
- Added paired command docs: `agent-read/COMMANDS/pi0_h2o_training_data.{zh,en}.md`.

## 2026-05-22 (Fix missing renderer args in K1 Piper AnyGrasp planner)

- Fixed `plan_anygrasp_keyframes_r1.py` and the batch wrapper:
  - added CLI defaults for the `HandRetargetR1Renderer` arguments `debug_visualize_cameras`, `debug_camera_axis_length`, `debug_camera_axis_thickness`, and `target_local_forward_retreat_m`.
  - forwards the same four arguments through the batch command builder so Piper K1 no longer fails during renderer initialization with missing required arguments.
- Updated `COMMAND_LIBRARY.zh.md` K1:
  - the heredoc batch command now explicitly passes these four defaults, making camera visualization and local-forward retreat easy to adjust later.
- Validation:
  - `conda run -n RoboTwin_bw python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py code_painting/plan_anygrasp_keyframes_piper.py code_painting/plan_anygrasp_keyframes_piper_batch.py`
  - `conda run -n RoboTwin_bw python code_painting/plan_anygrasp_keyframes_piper.py --help`
  - `conda run -n RoboTwin_bw python code_painting/plan_anygrasp_keyframes_piper_batch.py --help`

## 2026-05-22 (K1 resume command rewritten as a heredoc script)

- Updated `COMMAND_LIBRARY.zh.md` K1:
  - writes a bash script via `cat > /tmp/run_h2o_k1_preview_resume.sh <<'BASH' ... BASH` and then executes it.
  - avoids zsh `cmdand for>` continuation prompts caused by pasting `bash -lc ‘...’` with curly quotes.
- Validation:
  - checked the K1 heredoc block structure.

## 2026-05-22 (K1 planning command now runs only ids with preview summaries)

- Updated `COMMAND_LIBRARY.zh.md` K1:
  - automatically collects runnable ids from the K0.2 preview summary directory.
  - passes explicit `--ids` into the planner batch to avoid scanning unannotated ids without preview summaries.
  - keeps `--skip_existing 1` for resume behavior, so ids with existing `plan_summary.json` are skipped.
- Validation:
  - checked that the shell snippet can derive ids from existing `pick_diverse_bottles` preview summaries.

## 2026-05-22 (H2O annotator supports separate left/right hand keyframes)

- Updated `code_painting/annotate_hand_keyframes.py`:
  - preserved the old `keyframes` field for the existing AnyGrasp preview/planner flow.
  - added `left_keyframes` and `right_keyframes`, annotated with `l`/`L` and `r` respectively.
  - moved the old replay shortcut from `r` to uppercase `R`.
- Updated `COMMAND_LIBRARY.zh.md` K0 instructions.
- Validation:
  - `conda run -n RoboTwin_bw python code_painting/annotate_hand_keyframes.py --help`

## 2026-05-22 (H2O manual keyframe annotation and discarded-video marking)

- Added `code_painting/annotate_hand_keyframes.py`:
  - migrated the old `d_pour_blue` interactive annotation workflow into a repo-local reusable tool.
  - supports `Space` for keyframes, `d` for discard/restore, and `n/p/q` for save/navigation.
  - writes `hand_keyframes_all.json` and can normalize `hand_vis_gripper_<id>.mp4` to downstream `hand_vis_<id>.mp4` keys.
- Updated `code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh`:
  - skips ids marked `reject/discard/bad` or ids with fewer than two annotated keyframes.
- Updated `COMMAND_LIBRARY.zh.md` K0 to clarify that `ffplay/mpv` is only a viewer and the formal annotation path uses the interactive script.
- Validation:
  - `python code_painting/annotate_hand_keyframes.py --help`
  - `bash -n code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh`

## 2026-04-29 (Piper orientation guess: images-only + forward/up offset fix)

- Updated `code_painting/run_piper_gripper_standard_pose_guess.sh`:
  - output is now images-only (`zed/third` PNGs + `index.csv` + `world_targets_and_status.npz`), replay mp4 files are removed automatically.
  - added default target offset `target_world_offset_xyz=(0.0, +0.1, +0.1)` to improve IK reachability.
  - `index.csv` now includes `left_status/right_status` to separate IK-failure from orientation-definition issues.
- Validation (frame0):
  - with offset, standard 8 cases are no longer all Fail (e.g. `backward_guess/open_left_right_guess` become reachable).
  - output board ready for manual semantic labeling at `.../output_piper_gripper_standard_pose_guess_check2/board/`.

## 2026-04-29 (Piper gripper standard-orientation guess board tool)

- Added script: `code_painting/run_piper_gripper_standard_pose_guess.sh`
  - Purpose: generate 8 canonical orientation guesses (front/back/left/right + opening-axis up/down/left/right) on a fixed frame.
  - Runs one-frame replay per case and extracts the first `zed_replay.mp4` frame into a single `board/` folder.
  - Produces `board/index.csv` for manual semantic labeling.
- Added script: `code_painting/run_piper_gripper_orientation_guess_board.sh`
  - Purpose: orientation-sweep-based candidate dump for deeper debugging.
- Updated command doc: `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
  - Added D6 with one-command generation of the Piper orientation guess board.

## 2026-04-29 (HaMeR all-zero detection debug)

- Symptom: `detect_hands_realr1.py` completed on GPU under `hamer-r1` but produced `0/0` detections for all frames, and overwrote previous `hand_detections_0.npz`.
- Input-chain validation passed:
  - `pnp_star_pear_hamer_input` contains complete `rgb_0..15.mp4 / params_0..15.json`.
  - Extracted RGB frames clearly show both hands (not blank/black input).
  - `params_0.json` intrinsics are valid (`fx/fy/cx/cy/width/height`).
- Root cause: GPU commands in docs were using `hamer-r1` (CPU-safe env). On Blackwell GPUs this can hit CUDA-arch mismatch behavior and silently degrade per-frame hand inference.
- Fix verified: switching to `hamer-r1-gpu` (with `unset LD_LIBRARY_PATH`) restored detection on `video_id=0` to:
  - Left: `128/128`
  - Right: `128/128`
  - Both: `128/128`
  - Output dir: `pnp_star_pear_hamer_output_dbg_gpuenv`
- Docs updated: `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
  - A2 GPU commands now use `conda run -n hamer-r1-gpu ... --device cuda`
  - Added quick detection-count check and debug baseline command.

## 2026-04-27 (Piper object replay: head-camera link fallback fix)

- Fixed `code_painting/replay_r1_h5.py`:
  - no longer exits when R1-style `zed_link/head_camera` is missing in non-R1 robot configs.
  - added fallback head-camera pose computed from `robot_base_pose + head_camera_local_*`.
- Fixed `code_painting/render_multi_object_pose_r1_npz_batch.py`:
  - added and forwarded `--save_pose_debug` so batch mode accepts the flag.
- Updated command docs:
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md`
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline.md`
  - replay-stage mesh override name normalized to `star_fruit=...` to match object folder naming.

## 2026-04-27 (added stage I/O formats + per-object replay)

- Updated `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md` and its English pair to include:
  - HaMeR stage input/output roots and key format fields
  - FoundationPose stage input/output roots and key format fields
  - FoundationPose object folder naming (`pear`, `star_fruit`)
- Added split replay commands for trajectory/pose inspection:
  - pear-only replay
  - star_fruit-only replay
- Added replay output notes:
  - `head_cam_replay.mp4`
  - `multi_object_world_poses.npz`
  - `pose_debug.jsonl`

## 2026-04-27 (FoundationPose prompt fix + tmux exit root-cause)

- Conclusion: prompt `star` fails Grounding DINO init on this dataset; `star fruit` enters tracking correctly.
- Updated command docs:
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md`
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline.md`
  - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
- Recorded tmux pane-exit root cause: `set -e` in `source_foundationpose_env.sh` propagates to caller shell and exits pane on non-zero commands.

## 2026-04-27 (pnp_star_pear: FoundationPose pear+star stage completed)

- Added a Piper-specific FoundationPose preparation script:
  - `/home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py`
- Added a dedicated pear+star FoundationPose runner:
  - `/home/zaijia001/FoundationPose/run_piper_star_pear_foundation.sh`
- Updated workflow docs:
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md`
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline.md`
- Added full command chain for "HaMeR -> FoundationPose(pear+star) -> RoboTwin replay".
- Validation:
  - converted all 16 episodes into `pnp_star_pear_foundation_input` (with metric `depth_<id>/*.npy`).

## 2026-04-16 (added viewer support for calibrated scene scripts)

- Updated `code_painting/pika/visualize_calibrated_piper_pika_scene.py`
- Updated `code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`
- Added interactive viewer support:
  - `--viewer 1`
  - `--viewer-camera overview|head`
- This fixes the previous error: `unrecognized arguments: --viewer 1`.
- Updated command-library docs to include calibrated-scene viewer commands.


## 2026-04-24 (Piper hand dataset -> HaMeR -> RoboTwin documentation refresh)

- Added docs:
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md`
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline.md`
- Updated index:
  - `agent-read/README.md` (new links added)
- Documented the full runnable path for the new dataset layout (`episode*/camera/color|depth/headD435`):
  - convert to `rgb_<id>.mp4/depth_<id>.mp4/params_<id>.json`
  - run HaMeR to produce `hand_detections_<id>.npz`
  - visualize with `hand_vis_<id>.mp4`
  - run RoboTwin downstream replay command
- Validation:
  - `video_id=0` produced `hand_vis_0.mp4` and `hand_vis_gripper_0.mp4`

## 2026-04-16 (command-library clarification for headcam/wrist)

- Updated `agent-read/COMMANDS/pika_scene_commands.en.md`
- Updated `agent-read/COMMANDS/pika_scene_commands.zh.md`
- Clarified exactly which commands export head-cam images.
- Clarified that wrist-view export is not implemented yet in the calibrated-scene scripts.


## 2026-04-16 (command-doc update for viewer/headcam clarification)

- Updated `agent-read/COMMANDS/pika_scene_commands.en.md`
- Updated `agent-read/COMMANDS/pika_scene_commands.zh.md`
- Clarified that:
  - calibrated scene scripts do use the calibration bundle for the second robot and head camera
  - the head camera may be hard to notice in the overview because it is currently rendered only as a small marker with RGB axes
  - only the manual tabletop script currently supports interactive viewer mode


## 2026-04-16 (pika command library + base-on-side analysis)

- Added command-library docs under `agent-read/COMMANDS/`:
  - `README.en.md`
  - `README.zh.md`
  - `pika_scene_commands.en.md`
  - `pika_scene_commands.zh.md`
- Added copyable commands for:
  - interactive viewer runs of the manual tabletop scene
  - offscreen re-export of manual tabletop previews
  - calibrated scene export
  - calibrated version-B export
- Analyzed why the version-B bases look attached to the table side:
  - version B rotates the table convention by 90 degrees
  - then reuses the previous manual inset magnitude as world-x inset
  - so the bases are intentionally placed close to the rotated table side face rather than the former front edge


## 2026-04-16 (calibrated scene version B)

- Implemented version B scene alignment under `code_painting/pika/`:
  - removed the earlier +90deg anchor rotation on the first robot
  - kept the calibration left/right spread aligned with world y as much as possible
  - rotated the table convention instead of rotating the anchor robot
- Added script:
  - `code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`
- Generated outputs:
  - `code_painting/pika/output_calibrated_scene_vb/calibrated_scene_vb_overview.png`
  - `code_painting/pika/output_calibrated_scene_vb/calibrated_scene_vb_overview.mp4`
  - `code_painting/pika/output_calibrated_scene_vb/calibrated_scene_vb_headcam.png`
- Explicitly documented which fields from `robot_config_PiperPika_agx_dual_table.json` are reused and which are intentionally ignored in version B.


## 2026-04-16 (calibrated real-scene reconstruction in code_painting/pika)

- Read real-scene calibration inputs:
  - `CALIBRATION_TRANSFORMS_README.md`
  - `calibration_bundle_try2.json`
- Reconstructed a simulated scene under `code_painting/pika/` using:
  - first robot = current manual tabletop placement from `robot_config_PiperPika_agx_dual_table.json`
  - second robot = `left_base_T_right_base`
  - head camera = `left_base_T_head_camera`
- Added script:
  - `code_painting/pika/visualize_calibrated_piper_pika_scene.py`
- Generated outputs:
  - `code_painting/pika/output_calibrated_scene/calibrated_scene_overview.png`
  - `code_painting/pika/output_calibrated_scene/calibrated_scene_overview.mp4`
  - `code_painting/pika/output_calibrated_scene/calibrated_scene_headcam.png`
- Validation printed derived world poses for left base, right base, and head camera.


## 2026-04-16 (edge-mount base correction)

- Confirmed that the previous dual-table configuration was not mounted on the tabletop:
  - table long-edge line was `y = -0.30`
  - robot bases were at `y = -0.60`
  - so the bases were outside the table even though `z = 0.75` matched tabletop height
- Updated `robot_config_PiperPika_agx_dual_table.json`
  - moved the shared base pose from `y = -0.60` to `y = -0.30`
  - kept the corrected tabletop-facing quaternion
- New validated base poses:
  - left `[-0.30, -0.30, 0.75]`
  - right `[0.30, -0.30, 0.75]`
- Generated edge-mounted preview:
  - `code_painting/output_piper_pika_agx_dual_table_edge_mount/piper_pika_agx_dual_table_edge_mount.png`
  - `code_painting/output_piper_pika_agx_dual_table_edge_mount/piper_pika_agx_dual_table_edge_mount.mp4`


## 2026-04-16 (dual table orientation + oblique camera fix)

- Checked the existing UR-style camera reference in RoboTwin:
  - `code_painting/replay_piper_dual_h5.py` uses a fixed overview/head camera fallback
  - `code_painting/render_hand_retarget_r1_npz.py` builds third-person views from robot forward + world up
- Diagnosed the table-layout orientation issue:
  - the previous config used identity quaternion `[1, 0, 0, 0]`
  - that made the robot front align with `+x`, i.e. parallel to the table long edge
  - this was inconsistent with the intended use of approaching tabletop objects from one long-edge side
- Fixed `robot_config_PiperPika_agx_dual_table.json`
  - updated base quaternion to `[0.70710678, 0.0, 0.0, 0.70710678]`
  - both robots now face `+y`
- Updated `code_painting/visualize_piper_pika_agx_dual_table.py`
  - corrected oblique camera to sit behind the robots and look toward the tabletop
- Generated corrected outputs:
  - `code_painting/output_piper_pika_agx_dual_table_oblique_fixed/piper_pika_agx_dual_table_oblique_fixed.png`
  - `code_painting/output_piper_pika_agx_dual_table_oblique_fixed/piper_pika_agx_dual_table_oblique_fixed.mp4`
  - `code_painting/output_piper_pika_agx_dual_table_topdown_fixed/piper_pika_agx_dual_table_topdown_fixed.png`
  - `code_painting/output_piper_pika_agx_dual_table_topdown_fixed/piper_pika_agx_dual_table_topdown_fixed.mp4`


## 2026-04-16 (dual table top-down camera adjustment)

- Refined the table-edge interpretation for the dual Piper+Pika layout:
  - both robots are on one long-edge side of the table
  - bases are 0.30 m from the two short-side ends
  - base-to-base spacing is 0.60 m
- Updated `code_painting/visualize_piper_pika_agx_dual_table.py`
  - added `--camera-mode {top_down,oblique}`
  - temporary top-down camera is placed above the midpoint between the two bases and looks downward to the tabletop
- Generated temporary top-down outputs:
  - `code_painting/output_piper_pika_agx_dual_table_topdown/piper_pika_agx_dual_table_topdown.png`
  - `code_painting/output_piper_pika_agx_dual_table_topdown/piper_pika_agx_dual_table_topdown.mp4`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode top_down --output-dir code_painting/output_piper_pika_agx_dual_table_topdown --image-name piper_pika_agx_dual_table_topdown.png --video-name piper_pika_agx_dual_table_topdown.mp4`


## 2026-04-16 (piper_pika_agx dual table layout)

- Added a new colored combo embodiment for the newer Piper+Pika appearance:
  - `assets/embodiments/piper_pika_agx/piper_pika_agx.urdf`
- The new combo uses:
  - Piper arm from the original DAE-based Piper source
  - Pika gripper from `agx_arm_sim` `pika2_gripper.urdf` + DAE meshes
- Added dual-arm layout config for a 120x60x75 cm table:
  - `robot_config_PiperPika_agx_dual_table.json`
- Layout assumption used in this round:
  - symmetric UR-style split
  - shared base pose before split: `[0.0, -0.60, 0.75]`
  - `embodiment_dis = 0.60`
  - effective bases:
    - left `[-0.30, -0.60, 0.75]`
    - right `[0.30, -0.60, 0.75]`
- Added preview script:
  - `code_painting/visualize_piper_pika_agx_dual_table.py`
- Generated outputs:
  - `code_painting/output_piper_pika_agx_dual_table/piper_pika_agx_dual_table.png`
  - `code_painting/output_piper_pika_agx_dual_table/piper_pika_agx_dual_table.mp4`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1`


## 2026-04-16 (agx_arm_sim source inspection)

- Inspected the newly introduced repository:
  - `/home/zaijia001/Downloads/agx_arm_sim`
- Confirmed that:
  - `ros2 launch agx_arm_description display.launch.py arm_type:=piper end_effector:=pika`
  - routes to a **Piper + Pika** combined model
- Found an important repository state issue:
  - `agx_arm_description/agx_arm_urdf/` is empty in the current checkout
  - so the xacro-referenced Piper arm assets are missing locally from this repo snapshot
- Confirmed that the new `pika2_gripper.urdf` uses DAE meshes with embedded color/material data
- Added preview script:
  - `code_painting/visualize_agx_arm_sim_source.py`
- Generated preview outputs:
  - `code_painting/output_agx_arm_sim_preview/piper_only.png`
  - `code_painting/output_agx_arm_sim_preview/piper_only.mp4`
  - `code_painting/output_agx_arm_sim_preview/pika_only.png`
  - `code_painting/output_agx_arm_sim_preview/pika_only.mp4`
  - `code_painting/output_agx_arm_sim_preview/piper_pika_combo.png`
  - `code_painting/output_agx_arm_sim_preview/piper_pika_combo.mp4`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_agx_arm_sim_source.py --target all --output-root code_painting/output_agx_arm_sim_preview --video-frames 36 --fps 12`


## 2026-04-16 (Piper color diagnosis and quantification)

- Added a diagnosis-only documentation pass for the current observation:
  - original Piper alone looks gray
  - original Pika alone looks white
  - combined `piper_pika` looks whitened
- Confirmed source evidence:
  - Piper DAE contains intrinsic dark-gray diffuse color (`0.113725 0.113725 0.113725`)
  - combined `piper_pika.urdf` still contains explicit light-colored URDF material blocks on Piper arm links (`rgba="0.792156862745098 0.819607843137255 0.933333333333333 1"`)
- Added approximate rendered-image color statistics comparing:
  - original Piper bright
  - original Piper dark
  - combined bright
  - combined dark
- Main conclusion:
  - the user's summary is correct: Piper's original gray comes from DAE, and the combined model is very likely whitened by the URDF material override


## 2026-04-16 (dark stepwise validation)

- Added a stepwise dark-lighting validation pass for:
  - original Piper arm
  - original Pika gripper
  - combined `piper_pika`
- Added/updated scripts:
  - `code_painting/visualize_original_source_urdfs.py` now supports `--lighting {bright,dark}`
  - `code_painting/visualize_piper_pika_single.py` already supports `--lighting {bright,dark}`
- Generated outputs:
  - `code_painting/output_original_source_urdf_preview_dark/piper_arm.png`
  - `code_painting/output_original_source_urdf_preview_dark/piper_arm.mp4`
  - `code_painting/output_original_source_urdf_preview_dark/pika_gripper.png`
  - `code_painting/output_original_source_urdf_preview_dark/pika_gripper.mp4`
  - `code_painting/output_piper_pika_preview_dark/piper_pika_dark.png`
  - `code_painting/output_piper_pika_preview_dark/piper_pika_dark.mp4`
- Main conclusion:
  - the combined model whitening is likely caused by explicit light-colored URDF material blocks still present on the Piper arm links inside `assets/embodiments/piper_pika/piper_pika.urdf`, not only by lighting


## 2026-04-16 (piper_pika dark preview)

- Updated `code_painting/visualize_piper_pika_single.py` to support two lighting presets:
  - `bright`
  - `dark`
- Generated a dark-lighting preview for the current combined model (colored Piper arm + white Pika gripper):
  - `code_painting/output_piper_pika_preview_dark/piper_pika_dark.png`
  - `code_painting/output_piper_pika_preview_dark/piper_pika_dark.mp4`
- Purpose:
  - reduce over-bright lighting so the original deeper Piper arm tone is easier to inspect
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --lighting dark --output-dir code_painting/output_piper_pika_preview_dark --image-name piper_pika_dark.png --video-name piper_pika_dark.mp4 --video-frames 36 --fps 12`


## 2026-04-16 (original source URDF preview comparison)

- Added `code_painting/visualize_original_source_urdfs.py` to preview the original source URDFs directly from the download folders.
- Tested sources:
  - `/home/zaijia001/Downloads/agx_arm_urdf/piper/urdf/piper_description.urdf`
  - `/home/zaijia001/Downloads/pika_ros/src/pika_gripper_description/urdf/pika_gripper_description.urdf`
- Generated outputs:
  - `code_painting/output_original_source_urdf_preview/piper_arm.png`
  - `code_painting/output_original_source_urdf_preview/piper_arm.mp4`
  - `code_painting/output_original_source_urdf_preview/pika_gripper.png`
  - `code_painting/output_original_source_urdf_preview/pika_gripper.mp4`
- Purpose:
  - compare the appearance of the original Piper arm and original Pika gripper before further assembly edits
  - verify whether the whiteness comes from the original source assets themselves
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_original_source_urdfs.py --target both --output-root code_painting/output_original_source_urdf_preview --video-frames 30 --fps 12`


## 2026-04-16 (piper_pika DAE color recovery)

- Switched the assembled `assets/embodiments/piper_pika/piper_pika.urdf` arm visual meshes back to DAE to better recover the original Piper arm appearance.
- Added copied DAE assets under:
  - `assets/embodiments/piper_pika/meshes/dae/*`
- Color-source findings:
  - the original Piper arm color appears to be embedded in the Collada DAE files under `/home/zaijia001/Downloads/agx_arm_urdf/piper/meshes/dae/`
  - the checked Pika gripper source tree did not contain DAE/OBJ/MTL/texture assets; only STL meshes plus white URDF material values were found
- Generated new DAE-based preview outputs:
  - `code_painting/output_piper_pika_preview_dae/piper_pika_preview.png`
  - `code_painting/output_piper_pika_preview_dae/piper_pika_preview.mp4`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --output-dir code_painting/output_piper_pika_preview_dae --video-frames 24 --fps 12`


## 2026-04-16 (piper_pika preview export)

- Enhanced `code_painting/visualize_piper_pika_single.py`.
- Added:
  - a fixed preview camera position that clearly sees the whole single arm
  - still image export
  - short preview video export with a small joint-space motion
- Default outputs:
  - `code_painting/output_piper_pika_preview/piper_pika_preview.png`
  - `code_painting/output_piper_pika_preview/piper_pika_preview.mp4`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/visualize_piper_pika_single.py`
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --video-frames 24 --fps 12`
  - confirmed both preview files were written successfully


## 2026-04-16

- Added a new standalone assembled URDF under the original assets tree:
  - `assets/embodiments/piper_pika/piper_pika.urdf`
  - `assets/embodiments/piper_pika/meshes/*`
- Source inputs:
  - arm URDF/meshes from `/home/zaijia001/Downloads/agx_arm_urdf/piper/`
  - gripper URDF/meshes from `/home/zaijia001/Downloads/pika_ros/src/pika_gripper_description/`
- Assembly notes:
  - used the existing combined reference `piper_pika_gripper_description.urdf` as the starting point
  - converted package mesh URIs to local relative `meshes/...` paths
  - renamed the robot to `piper_pika`
  - renamed `dummy_link` -> `piper_pika_dummy_link`
  - renamed `gripper_base` -> `pika_gripper_base`
  - copied required arm and gripper meshes into the new embodiment folder
- Added a minimal single-arm visualization script:
  - `code_painting/visualize_piper_pika_single.py`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/visualize_piper_pika_single.py`
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1`
  - offscreen load confirmed active joints:
    - `joint1 joint2 joint3 joint4 joint5 joint6 joint7 joint8`
- Repository note:
  - `assets/*` is currently ignored by `.gitignore`, so the new embodiment files exist locally but do not show up in `git status`


## 2026-04-14 (Piper V2 batch fix)

- Fixed Piper V2 batch argument pollution in `code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`.
- Root cause:
  - the reused `plan_anygrasp_keyframes_r1_batch.py` parser was still supplying its own default `--robot_config robot_config_R1.json`
  - as a result, batch mode launched the Piper V2 single-video script with an explicit R1 config, so viewer/rendering still showed R1-style behavior
- Fix:
  - inject Piper defaults before calling the reused batch launcher:
    - `--robot_config /home/zaijia001/ssd/RoboTwin/robot_config_Piper_dual_v2.json`
    - `--head_camera_local_quat_wxyz 1.0 0.0 0.0 0.0`
    - `--head_camera_local_pos 0.0 0.0 0.0`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
  - command probe confirmed printed batch command now contains:
    - `--robot_config /home/zaijia001/ssd/RoboTwin/robot_config_Piper_dual_v2.json`
    - `--head_camera_local_quat_wxyz 1.0 0.0 0.0 0.0`


## 2026-04-14

- Added a true Piper V2 dual-arm-style implementation following the existing UR dual-single-arm setup, without modifying any R1 / R1 Pro files.
- Added files:
  - `robot_config_Piper_dual_v2.json`
  - `code_painting/replay_piper_dual_h5.py`
  - `code_painting/render_hand_retarget_piper_dual_npz_urdfik.py`
  - `code_painting/plan_anygrasp_keyframes_piper_v2.py`
  - `code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
  - `code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh`
  - `agent-read/V2.0_piper_dual_ur_style.md`
  - `agent-read/V2.0_piper_dual_ur_style_ZH.md`
- V2 implementation notes:
  - uses `dual_arm_embodied=false` and two independent Piper URDF instances
  - keeps separate left/right base poses instead of collapsing both arms back to one shared root pose
  - uses a dedicated Piper replay renderer and a dedicated Piper URDFIK renderer
  - uses `assets/embodiments/piper/piper.urdf` for both left and right URDF loading and IK
  - validated effective base poses:
    - left = `[-0.4, -0.65, 0.72]`
    - right = `[0.4, -0.65, 0.72]`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/replay_piper_dual_h5.py code_painting/render_hand_retarget_piper_dual_npz_urdfik.py code_painting/plan_anygrasp_keyframes_piper_v2.py code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh`
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/plan_anygrasp_keyframes_piper_v2.py --help`
  - direct config probe confirmed:
    - `left_urdf_path=./assets/embodiments/piper/piper.urdf`
    - `right_urdf_path=./assets/embodiments/piper/piper.urdf`
    - `left_origin=[-0.4, -0.65, 0.72]`
    - `right_origin=[0.4, -0.65, 0.72]`
  - renderer probe confirmed:
    - `[piper-v2-bases] left=[-0.4, -0.65, 0.72] right=[0.4, -0.65, 0.72]`


## 2026-04-14

- Added a Piper-compatible AnyGrasp planner wrapper without modifying the original R1 / R1 Pro planner files.
- Added files:
  - `robot_config_Piper_dual.json`
  - `code_painting/plan_anygrasp_keyframes_piper.py`
  - `code_painting/plan_anygrasp_keyframes_piper_batch.py`
  - `code_painting/run_plan_anygrasp_keyframes_piper_batch.sh`
  - `agent-read/2026-04-14_piper_anygrasp_wrapper.md`
  - `agent-read/2026-04-14_piper_anygrasp_wrapper_ZH.md`
- Implementation notes:
  - keeps the original `plan_anygrasp_keyframes_r1.py` code path untouched
  - injects a Piper robot config at runtime
  - swaps the replay / URDFIK renderer classes to Piper-specific adapters
  - uses `assets/embodiments/piper/piper.urdf` for URDF IK
  - maps the existing left/right execution structure onto two Piper instances for compatibility
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_piper.py code_painting/plan_anygrasp_keyframes_piper_batch.py`
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_batch.sh`
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/plan_anygrasp_keyframes_piper.py --help`


## 2026-04-03

- Added smooth-focused documentation:
  - `agent-read/smooth/README.zh.md`
  - `agent-read/smooth/README.en.md`
- Purpose:
  - document the current smooth-related handling in the AnyGrasp keyframe planner
  - explain why large `joint_target_wait_steps` can create jump / teleport-like motion in exported videos
  - summarize practical no-code-change ways to reduce jumps while keeping accuracy, including pros and cons
- Covered topics:
  - current path structure: `init -> pregrasp -> grasp -> action`
  - current offset semantics: `candidate_target_local_x_offset_m=-0.03` and `approach_offset_m=0.08`
  - current smoothing logic: EE/TCP-pose interpolation followed by per-waypoint IK
  - existing post-hoc smooth tools:
    - `code_painting/replay_pose_debug_smooth.py`
    - `code_painting/smooth_planner_outputs_from_pose_debug.py`
    - `code_painting/batch_smooth_planner_outputs.sh`
- Expanded the analysis with new candidate methods:
  - added a dedicated evaluation of the “sample one EE point every 1 cm + use previous IK solution as the seed + reject the whole segment if adjacent joint deltas exceed a threshold” idea
  - clarified its relationship to the current `cartesian_interp_ik` mode: the current code already has waypoint IK + previous-seed chaining; what is missing is denser sampling and explicit jump-threshold rejection
  - added pros/cons and implementation difficulty for several alternatives:
    - fixed-step dense sampling
    - position+rotation dual-threshold sampling
    - joint jump-threshold filtering
    - soft continuity preferences inside IK
    - post-IK joint smoothing
    - extra semantic intermediate poses
    - switching to global trajectory optimization
- Expanded again with V7 debug analysis:
  - explained why accuracy drops without try / replanning: the current system behaves more like a `plan-execute-correct` closed loop than a single-shot open-loop path that fully lands inside tolerance
  - explained why try improves accuracy but makes the video look more segmented: multiple short corrections inside one stage plus settle tails that are not serialized frame by frame
  - added a proposed new diagnostic quantity:
    - lateral distance from the current point to the target forward axis
    - useful for separating front/back error from lateral miss relative to the intended approach line
- Implemented the code change in this round:
  - `code_painting/plan_anygrasp_keyframes_r1.py`
  - added new breakdown fields:
    - `lateral_to_forward_axis_m`
    - `lateral_to_forward_axis_cm`
  - added new terminal output field:
    - `lat_cm`
  - integrated into:
    - single-arm / dual-arm `plan-request`
    - single-arm / dual-arm `plan-solution`
    - single-arm / dual-arm `attempt`
    - single-arm `attempt-supervision`
    - `attempt_history` / supervision-error structures
- Added another analysis-only design direction that preserves the current main workflow:
  - express the human-hand target as an object-relative pose `T_obj_hand_demo`
  - then generate a more executable robot target through a robot-specific correction `Δ_robot`
  - recommend implementing this later as an independent target-adapter layer rather than modifying the IK solver core first
  - recommended progression:
    - constant rigid correction
    - stage-specific correction
    - object-type/size adaptive correction
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py`
- Validation:
  - documentation-only task, so no extra code validation command was run

## 2026-03-27

- Added a raw-planner-v7 -> repaint -> review -> pi0 wrapper script:
  - `run_planner_v7_repaint_review_pi0.sh`
  - Purpose:
    - skip smoothing and directly consume `anygrasp_plan_keyframes_realoffset_batch_pure-v7`
    - call the original `batch_head_cam_repaint_with_auto_pad.sh`
    - call `review_repaint_videos.py` for manual filtering
    - call `process_repainted_planner_outputs.py` to generate pi0 / robotwin processed_data
  - Validation:
    - `bash -n run_planner_v7_repaint_review_pi0.sh`

- Added smooth-bundle scripts:
  - `code_painting/smooth_planner_outputs_from_pose_debug.py`
  - `code_painting/batch_smooth_planner_outputs.sh`
  - `run_reviewed_smooth_repaint_pi0_pipeline.sh`
  - Purpose:
    - remove lingering / near-duplicate frames from the Step-1 planner outputs
    - interpolate key states to smooth jumps
    - re-export source-consistent head / left wrist / right wrist / pose_debug outputs
  - Key output root:
    - `code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth`
  - Validation:
    - `python -m py_compile code_painting/smooth_planner_outputs_from_pose_debug.py`
    - `bash -n code_painting/batch_smooth_planner_outputs.sh`
    - `bash -n run_reviewed_smooth_repaint_pi0_pipeline.sh`
    - `DRY_RUN=1 bash run_reviewed_smooth_repaint_pi0_pipeline.sh`
    - one-sample output: `/tmp/d_pour_blue_0_smooth_bundle`

- Added script:
  - `policy/pi0/scripts/process_repainted_planner_outputs.py`
  - Purpose: process pi0 data using source-consistent planner outputs:
    - repaint planner head
    - planner `left_wrist_cam_plan.mp4`
    - planner `right_wrist_cam_plan.mp4`
    - planner `pose_debug.jsonl`
  - This avoids mixing hand-retarget wrist / `world_targets_and_status.npz` with planner head videos
  - Minimal validation:
    - `python -m py_compile policy/pi0/scripts/process_repainted_planner_outputs.py`
    - one-sample output test: `/tmp/pi0_planner_repaint_test`

- Added analysis notes:
  - `agent-read/2026-03-27_repaint_duration_mismatch_analysis_ZH.md`
  - `agent-read/2026-03-27_repaint_duration_mismatch_analysis.md`
  - Purpose: record why `process_repainted_headcam_with_wrist.py` outputs are much shorter than `head_cam_plan.mp4`
  - Conclusion:
    - the current script truncates by minimum frame count rather than aligning by real duration in seconds
    - the effective length is actually limited by `world_targets_and_status.npz` and the left/right wrist replay streams
    - if a later viewer/export step plays the result at a higher fps, it will further feel like “only about 1 second”
  - No code changes in this round; only recorded findings and root cause

- Added source-consistency analysis notes:
  - `agent-read/2026-03-27_head_source_vs_wrist_source_analysis_ZH.md`
  - `agent-read/2026-03-27_head_source_vs_wrist_source_analysis.md`
  - Conclusion:
    - `batch_head_cam_repaint_with_auto_pad.sh` uses planner-side `head_cam_plan.mp4` as the head source
    - the current pi0 processing step uses hand-retarget `left/right_wrist_replay.mp4` as the wrist source
    - inside the hand-retarget directory, `zed_replay / wrist / world_targets` are basically aligned in length
    - but planner head and hand-retarget wrist are not the same source stream, so matching frame counts should not be expected

- Added `policy/pi0/scripts/process_repainted_headcam_with_wrist.py`
  - Purpose:
    - convert the newer “SAM/repainted head-cam video + left/right wrist replays + world_targets_and_status.npz” into the pi0 `processed_data` HDF5 intermediate format
  - Main capabilities:
    - separate `--head-root` and `--retarget-root`
    - templated episode directory names via:
      - `--head-dir-template`
      - `--retarget-dir-template`
    - support the new head-video filename:
      - `target_with_original_head_cam_plan.mp4`
    - support `--review-json`, which by default processes only videos manually marked `y` / `usable=true`
    - support `--review-mode include_ambiguous` to also include `m` / `ambiguous`
    - keep output compatible with the existing pi0 `processed_data/<task>-<num>/episode_x/*.hdf5` layout
  - Related docs:
    - `agent-read/2026-03-27_pi0_repaint_wrist_to_hdf5_ZH.md`
    - `agent-read/2026-03-27_pi0_repaint_wrist_to_hdf5.md`
  - Validation:
    - `cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && python -m py_compile scripts/process_repainted_headcam_with_wrist.py scripts/process_data_retageted_human.py scripts/process_data_R1.py`

## 2026-03-25

- Added a visual-only base occluder to hide the chassis in rendered videos:
  - Files:
    - `code_painting/render_hand_retarget_r1_npz.py`
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/render_object_pose_r1_npz.py`
  - Motivation:
    - the robot chassis/base is often visible in head and wrist videos and hurts the composition
    - the user wanted a configurable occluder with custom height/size that does not participate in collision
  - Implementation:
    - added a visual-only `base_occluder` actor with no collision shapes
    - the occluder follows the robot base pose
    - exposed CLI flags:
      - `--base_occluder_enable`
      - `--base_occluder_local_pos X Y Z`
      - `--base_occluder_half_size HX HY HZ`
      - `--base_occluder_color R G B`
  - Semantics:
    - `local_pos` is defined in the robot-base frame, so it rotates with the robot heading
    - `half_size` uses the SAPIEN box half-size convention
    - the occluder is visual-only and does not affect collision, IK obstacles, or grasp contact
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_r1_npz.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py /home/zaijia001/ssd/RoboTwin/code_painting/render_object_pose_r1_npz.py`

- Added a first-pose debug log for the base occluder:
  - File:
    - `code_painting/render_hand_retarget_r1_npz.py`
  - Change:
    - when the occluder pose is updated for the first time, print
      - `world_p`
      - `half_size`
      - `color`
  - Purpose:
    - helps confirm that the occluder is really created and whether it lands at the intended position and size

- Corrected the R1 planner wrist-camera mount definition and removed post-export image rotation:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/CAMERA_DEBUG_NOTES_R1.md`
  - Root cause:
    - the R1 planner path had been using a wrist local pose closer to `galaxea_sim/robots/r1_pro.py`
    - but `galaxea_sim/robots/r1.py` only includes `rx=-10°` and does not include the extra local `z=-90°`
    - that is why wrist exports kept needing `90°/180°` image-plane patches and still looked geometrically unnatural
  - Fix:
    - override the wrist local quaternion specifically inside `plan_anygrasp_keyframes_r1.py` so the R1 planner matches `galaxea_sim/robots/r1.py`
    - `rotate_wrist_rgb_for_export(...)` now becomes a pass-through; planner wrist exports no longer rotate the saved image afterward
    - wrist-writer size returns to the original landscape `(image_width, image_height)`
  - Impact:
    - the wrist view now comes from the actual mounted camera pose rather than an export-time image-plane rotation
    - no change to the global default in `render_hand_retarget_r1_npz.py`, so R1 Pro-related paths are not affected
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Corrected planner wrist-video export orientation again:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - User feedback:
    - after the previous `180°` correction, the exported result still looked like the correct view rotated 90 degrees CCW
  - Fix:
    - changed `rotate_wrist_rgb_for_export(...)` from `cv2.ROTATE_180` to `cv2.ROTATE_90_COUNTERCLOCKWISE`
    - changed planner wrist-writer size back to the rotated-frame size `(image_height, image_width)`
  - Current behavior:
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` are exported with a `90°` CCW image-plane rotation
    - output dimensions match the rotated frames
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Corrected planner wrist-video export orientation and frame size:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - the previous round incorrectly treated the wrist videos as needing a `90°` rotation
    - the exported files became portrait, and the user confirmed the image was still upside down
  - Fix:
    - changed `rotate_wrist_rgb_for_export(...)` from `cv2.ROTATE_90_CLOCKWISE` to `cv2.ROTATE_180`
    - changed planner wrist-writer size back from portrait `(image_height, image_width)` to landscape `(image_width, image_height)`
  - Current behavior:
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` stay in landscape format
    - export now applies only an in-plane `180°` correction
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed the orientation of planner-exported wrist videos:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` appeared 90 degrees CCW relative to the expected viewing orientation
  - Investigation result:
    - after comparing `render_hand_retarget_r1_npz.py` and `galaxea_sim/robots/r1_pro.py`, the R1 and R1 Pro wrist-camera mount poses are effectively the same:
      - base quaternion `[0.5, 0.5, -0.5, 0.5]`
      - extra RPY offset `[-10deg, 0, -90deg]`
    - so this round does not change the mounted camera definition; it corrects the exported image plane instead
  - Fix:
    - added `rotate_wrist_rgb_for_export(...)`
    - apply `cv2.ROTATE_90_CLOCKWISE` before writing left/right wrist videos in `record_frame(...)`
    - rotation now happens before overlay/BGR conversion so debug text stays upright as well
  - Impact:
    - no change to world camera pose, planner targets, candidate coordinate conversion, or head videos
    - this only corrects planner wrist-video export orientation
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed the post-rotation writer-size mismatch for wrist videos:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - the previous round rotated planner wrist videos by `90° clockwise`
    - but the `cv2.VideoWriter` instances were still opened with the original `(image_width, image_height)` = `640x360`
    - the rotated frames are actually `360x640`
    - as a result the writer created tiny placeholder mp4 files (around `258B`) without valid video frames
  - Fix:
    - planner wrist writers now use the rotated frame size `(image_height, image_width)`
  - Impact:
    - `head_cam_plan.mp4` keeps its original landscape size
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` now use portrait dimensions matching the rotated wrist frames
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Adjusted URDF-IK waypoint visualization and documented the stage-settling parameters:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Visualization changes:
    - with `--debug_visualize_ik_waypoints 1`, the viewer now shows start and goal markers in addition to intermediate waypoints
    - both start and goal use red point+forward-axis markers
    - intermediate waypoint markers are now smaller to reduce clutter around the hands and targets
  - Analysis notes:
    - `init` still only applies `apply_robot_init_pose(...)` once and advances a short `step_scene(settle_steps)` window; unlike later stages, it does not call `settle_arms_to_targets(...)` to wait for full convergence
    - `--settle_steps` defaults to `4` and is used to:
      - advance a few physics steps after init
      - add a short settle window after each stage trajectory is sent
    - `--joint_target_wait_steps` defaults to `60` and is used to:
      - keep stepping physics after the trajectory ends until the measured joints get closer to the final target
  - Impact:
    - no planning or execution logic changed in this round; this is a visualization-only update
  - Relevant code:
    - waypoint markers:
      - `update_ik_waypoint_visuals(...)`
      - `_ensure_ik_waypoint_marker_actors(...)`
      - `_ensure_ik_waypoint_endpoint_actors(...)`
    - init / stage settling:
      - `apply_robot_init_pose(...)`
      - `execute_single_arm_plan(...)`
      - `execute_dual_arm_plan(...)`
      - `settle_arms_to_targets(...)`
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed pure-mode EE-pose serialization compatibility in `pose_debug.jsonl`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - the newly added `current_left_ee_pose_world_wxyz` / `current_right_ee_pose_world_wxyz` export path in `record_frame(...)` incorrectly assumed that `robot.get_*_ee_pose()` returns a `sapien.Pose`
    - in this robot implementation the API actually returns a 7D list `[x, y, z, qw, qx, qy, qz]`
    - pure-mode batch runs therefore crashed on the first frame write with:
      - `AttributeError: 'list' object has no attribute 'p'`
  - Fix:
    - added `pose_like_to_world_wxyz(...)`
    - made the serializer accept both `sapien.Pose` objects and 7D pose lists
    - switched the relevant head/ee pose exports in `record_frame(...)` to use the shared helper
  - Impact:
    - no change to planning or execution behavior
    - this only fixes pure-mode `pose_debug.jsonl` serialization
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Added CLI parameter `--urdfik_cartesian_interp_auto_step_m` to control the translation-density threshold used by automatic waypoint mode when `--urdfik_cartesian_interp_steps=-1`.
- The old `0.05m` threshold was hardcoded; it is now configurable with the same default `0.05`. Fixed waypoint mode is unchanged.
- `render_hand_retarget_r1_npz_urdfik.py` now prints the active `auto_step_m` in `[ik-trajectory]` and `[ik-waypoints]` logs.
- Added execution-side joint-convergence tuning parameters for the AnyGrasp planner:
  - `--joint_command_scene_steps`
  - `--joint_target_wait_steps`
  - `--joint_target_wait_tol_rad`
- Modified files:
  - `code_painting/plan_anygrasp_keyframes_r1.py`
  - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
- Purpose:
  - address the case where `plan-request` / `plan-solution` already indicate that the planned endpoint is moving backward, but `attempt` still remains stuck at a forward-biased pose for many retries.
  - make the execution path advance more physics scene steps per joint waypoint and wait for final joint convergence before measuring reach error.
- Current diagnosis:
  - the earlier interpretation of `plan_vs_current_fwd_cm` was partially wrong.
  - the correct meaning is: in `plan-solution`, `plan_vs_current_fwd_cm > 0` means the current pose is ahead of the planned endpoint, so the planned endpoint is actually behind the current pose.
  - therefore, in the long-tail retry regime, IK is already commanding backward motion, but the execution layer is not converging far enough to that command.
- Extra docs:
  - `agent-read/2026-03-25_ik_execution_regression/README.zh.md`
  - `agent-read/2026-03-25_ik_execution_regression/README.en.md`

- Added "skip grasp re-execution when the target is the same" logic:
  - when `pregrasp` and `grasp` resolve to the same pose (the typical case is `--approach_offset_m 0.0`)
  - the system no longer replans and re-executes `grasp` after already reaching `pregrasp`
  - instead it reuses the `pregrasp` result and records `grasp_skipped_same_target`
- Why:
  - under `approach_offset_m=0.0`, logs showed that `pregrasp` could already converge
  - but the subsequent redundant `grasp` replanning to the exact same target pulled the end effector away again
- Additional note:
  - this round briefly tried a seeded/unseeded FK post-check scorer inside `urdfik.py`
  - that attempt produced clearly wrong pregrasp poses on the first retry
  - it has been reverted and is not kept

- Added a third correction for `cartesian_interp_ik`:
  - tightened the default position threshold in `urdfik.py` from `0.005m` to `0.001m`
  - tightened the default rotation threshold from `0.05rad` to `0.02rad`
  - stopped threshold relaxation from expanding toward `0.1m`
  - `render_hand_retarget_r1_npz_urdfik.py` now automatically reduces overly dense cartesian waypoint counts based on IK thresholds
- Immediate reason:
  - the `action` logs showed total target motion on the order of a few centimeters to about 9 cm, while `--urdfik_cartesian_interp_steps 30` reduced each translation step to roughly `3mm`
  - the old IK success threshold was `5mm`
  - that allowed the solver to accept many waypoint solves with almost no motion, so the final theoretical endpoint stayed far from the target
- Updated conclusion:
  - for this bug, “more retries” is not the main fix
  - increasing interpolation density can actually make it worse
  - the waypoint resolution must first remain larger than the IK success threshold

- Added a waypoint-level `seeded/unseeded` candidate comparison on top of that:
  - location: `code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - behavior: for each waypoint, try both “use current seed” and “no seed”
  - then use FK-based post-checking against that waypoint's `ee` target and keep the closer candidate
- Purpose:
  - address the remaining `action` failure mode where the interpolation-density fix helped, but the left arm was still being trapped in a local solution basin by the current seed

- Adjusted terminal debug output formatting:
  - `plan-request`
  - `plan-solution`
  - `attempt`
- Changes:
  - dual-arm logs are now printed as separate left/right lines
  - `theory` is shortened to:
    - `forward`
    - `backward`
    - `aligned`
  - `fwd_cm` now uses ANSI color highlighting so sign changes are easier to spot

- Further compressed the end-of-sample terminal summary:
  - it no longer prints the full `statuses_by_arm={...}` dictionary
  - it now uses a short format with:
    - `arms`
    - `arm`
    - `obj`
    - `fXX=cYY`
    - `pre/gr/act`
    - `video`

## 2026-03-25 11:51:21 +08

- Fixed a logging-format regression:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - `colorize_forward_cm()` was accidentally indented inside `short_direction_label()` during the previous log refactor
    - dual-arm `plan-request` logging then crashed with `NameError: name 'colorize_forward_cm' is not defined`
  - Fix:
    - restored `colorize_forward_cm()` to module scope
    - corrected its internal branch indentation so positive/negative/near-zero color cases work again
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

## 2026-03-25 12:08:00 +08

- Added a grasp/action object-collision toggle:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - New flag:
    - `--enable_grasp_action_object_collision 0|1`
  - Behavior:
    - default `0` preserves the original no-collision execution mode
    - when set to `1`, execution objects selected by the executed arm keep collision geometry
    - collision stays disabled during `pregrasp`
    - collision is enabled right before `close_gripper` and remains enabled through `action`
    - object attachment logic, TCP-relative transforms, target generation, and other pose transforms are unchanged
  - Implementation note:
    - stage-local collision enable/disable is implemented by caching and restoring SAPIEN collision groups
    - non-selected objects keep their original behavior
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-25 12:22:00 +08

- Adjusted the default visualization behavior of `plan_anygrasp_keyframes_r1.py`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Changes:
    - restored target-pose axes to be visible by default
    - hid the left/right wrist cameras by default in this planner script so wrist-camera frustums no longer appear in saved videos / viewer output
  - Notes:
    - this affects only the default behavior of `plan_anygrasp_keyframes_r1.py`
    - it does not change the shared base renderer behavior for other scripts
    - it does not affect head-camera or third-person capture itself
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

## 2026-03-25 13:05:00 +08

- Added pure/debug visualization controls to `plan_anygrasp_keyframes_r1.py`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - New flags:
    - `--debug_visualize_targets 0|1`
    - `--viewer_show_camera_frustums 0|1`
  - Changes:
    - restored target-axis rendering to an explicit flag instead of a hardcoded always-on path while keeping the default enabled
    - disabled SAPIEN `ControlWindow.show_camera_linesets` by default in the viewer path
    - after the wrist frustums are hidden, this also removes the remaining zed/third camera frustum lines by default
  - Notes:
    - `pure_scene_output` still controls clean main videos: no overlay text, no candidate grippers, no target axes
    - `viewer_show_camera_frustums=0` now controls viewer camera-line visibility
    - `debug_visualize_targets=1` keeps target axes available for debug runs
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed the batch wrapper so the new pure/debug visualization flags are forwarded:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Changes:
    - added `--debug_visualize_targets` to the batch parser
    - added `--viewer_show_camera_frustums` to the batch parser
    - forwarded both flags in `build_single_command()`
  - Reason:
    - the pure-mode batch command was being rejected by batch-layer argparse before the single-run script could receive the flags
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-25 13:35:00 +08

- Added a minimal "progressive close until contact/stall" path for `--enable_grasp_action_object_collision=1`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Analysis:
    - Previously the flag only re-enabled collision for the selected object during `grasp/action`
    - The gripper still used a one-shot `set_grippers(close)` update and only advanced a very small number of physics steps
    - As a result, the fingers could still visually close through the object even when collision shapes were present
  - Change:
    - Added `close_grippers_progressively_with_collision_stop()`
    - The `close_gripper` stage now closes in small command increments
    - Each increment advances physics, reads gripper joint `qpos`, and checks contacts between the selected object and the executing gripper links
    - Closure stops early once contact is present and gripper joint motion has stalled
    - The original one-shot `renderer.set_grippers(...)` path remains unchanged when the collision flag is disabled
  - Note:
    - This is a minimal fix; it does not change `pregrasp/grasp/action` target construction or the TCP-to-object attachment transform
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed a SAPIEN API compatibility issue in the progressive-close path:
  - Problem:
    - `PhysxArticulationJoint` in the current environment does not provide `get_qpos()`, which caused a crash when entering `close_gripper`
  - Fix:
    - read the full articulation `qpos` from `entity.get_qpos()`
    - then recover each gripper joint's actual position by accumulating offsets over `active_joints` with `joint.get_dof()`
  - Code location:
    - `code_painting/plan_anygrasp_keyframes_r1.py:_get_gripper_joint_positions`

## 2026-03-25 14:05:00 +08

- Added an explicit warning when `grasp` has not reached the target but execution still proceeds to `close_gripper`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Behavior:
    - before `close_gripper`, if the `grasp` stage reports `reached=False`, the script now prints `[warn] grasp_not_reached_before_close ...`
    - this is logging only and does not change the execution logic

- Added an automatic waypoint mode for URDF IK Cartesian interpolation:
  - File:
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - New behavior:
    - when `--urdfik_cartesian_interp_steps > 1`, the original fixed-step mode is preserved
    - when `--urdfik_cartesian_interp_steps -1`, an automatic mode is enabled
  - Automatic mode rule:
    - if absolute translation distance is `<= 5cm`, no intermediate TCP waypoint is added (start and goal only)
    - once translation exceeds `5cm`, one additional intermediate waypoint is added per extra `5cm` bucket
    - examples:
      - `10cm` translation -> `1` intermediate waypoint
      - `15cm` translation -> `2` intermediate waypoints
  - Note:
    - this automatic mode only adjusts waypoint count from TCP translation distance and does not change the IK target or execution semantics

- Recorded the actual execution semantics of `cartesian_interp_ik`:
  - New documents:
    - `agent-read/V1.15_urdfik_cartesian_interp_execution_semantics_ZH.md`
    - `agent-read/V1.15_urdfik_cartesian_interp_execution_semantics.md`
  - Conclusion:
    - intermediate `ee/tcp` waypoints are indeed used for per-waypoint IK solving
    - after the latest fix, the execution stage now also replays the corresponding `joint_waypoints`
    - `cartesian_interp_ik` now affects both IK solving and the final executed path more directly

## 2026-03-25 14:25:00 +08

- Fixed the `urdfik` execution layer so it actually consumes `joint_waypoints`:
  - File:
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - Changes:
    - `execute_plans(...)` no longer performs only an endpoint interpolation from `current_joints` to `target_joints`
    - it now directly consumes `plan["position"]` / `plan["velocity"]`
    - in dual-arm execution, left and right trajectories are interleaved by relative trajectory progress, matching the base renderer semantics
    - `_execute_single_ik_plan(...)` also now prefers replaying the full `plan["position"]` trajectory
  - Effect:
    - the `joint_waypoints` generated by `cartesian_interp_ik` now become part of the real executed path
# 2026-03-25

- Enhanced pure-mode outputs:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Changes:
    - `pure_scene_output=1` no longer generates `debug_selection_preview.mp4`
    - the main planning run now also writes:
      - `head_cam_plan.mp4`
      - `left_wrist_cam_plan.mp4`
      - `right_wrist_cam_plan.mp4`
    - pure mode now auto-enables `pose_debug.jsonl` even without explicitly passing `--save_pose_debug 1`
    - `pose_debug.jsonl` now additionally records:
      - left/right wrist-camera poses
      - left/right TCP / EE poses
      - left/right 6D arm qpos
      - left/right gripper finger-joint qpos
      - object actor poses / replay poses
    - `plan_summary.json` now includes the corresponding video and data paths
  - Documentation:
    - `agent-read/2026-03-25_pure_mode_outputs_ZH.md`
    - `agent-read/2026-03-25_pure_mode_outputs.md`
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py code_painting/render_hand_retarget_r1_npz_urdfik.py`
    - `git diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- Added a `--debug_visualize_ik_waypoints` debug parameter for `--planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik`, rendering intermediate `tcp_waypoints_world` into viewer/debug output.
- The visualization is a “point + local forward-axis” marker and only shows intermediate waypoints; the start and final target remain excluded, with the final target still represented by the existing target axis.
- This change affects debug display only and does not change waypoint generation, IK solving, trajectory execution, or collision behavior.
- `plan_anygrasp_keyframes_r1_batch.py` now forwards the parameter as well.
- Validation: `python -m py_compile` and `git diff --check` are run after this round.
# 2026-03-25

- Fixed `base_occluder` initialization/update missing on the `urdfik` / `ReplayRenderer` path:
  - File:
    - `code_painting/replay_r1_h5.py`
  - Cause:
    - `ReplayRenderer._load_robot()` overrides the base robot-loading flow but did not hook in the later-added `base_occluder` logic
    - as a result, under `plan_anygrasp_keyframes_r1.py --planner_backend urdfik`:
      - the `[base-occluder] ...` debug line was never printed
      - the occluder was not updated using the corrected anchor semantics
  - Change:
    - added the missing calls inside `ReplayRenderer._load_robot()`:
      - `self._base_occluder_link = self._find_robot_link(["base_link"])`
      - `self._update_base_occluder_pose()`
    - this keeps replay / urdfik execution consistent with the base renderer behavior
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/replay_r1_h5.py code_painting/render_hand_retarget_r1_npz.py`

- Further refined the `base_occluder` height semantics:
  - File:
    - `code_painting/render_hand_retarget_r1_npz.py`
  - Cause:
    - the `base_link` plane position is closer to the visible chassis, but its `z` origin is not the user-intuitive "height above ground" reference
    - using the full 3D `base_link` pose directly could place the occluder below the floor or at an obviously wrong height
  - Change:
    - the occluder now uses a mixed anchor:
      - `x/y` still follow the `base_link` planar position
      - `z` is interpreted relative to the renderer root/base world height
      - orientation keeps only the base yaw instead of inheriting the full 3D link pose
    - the debug log now also prints:
      - `anchor_p=...`
      - `root_z=...`
    - this makes the `Z` value in `--base_occluder_local_pos X Y Z` behave like a height-above-ground control
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/render_hand_retarget_r1_npz.py`

- Fixed the `base_occluder` panel appearing far away from the robot base:
  - File:
    - `code_painting/render_hand_retarget_r1_npz.py`
  - Cause:
    - the occluder previously followed the renderer's internal root/base pose rather than the visible chassis-aligned `base_link`
    - under the current R1 setup those frames are offset, so the occluder could appear far from the robot in the viewer
  - Change:
    - the `base_occluder` now anchors to `base_link` when available
    - it falls back to the previous root/base pose only if `base_link` cannot be found
    - the debug log now prints `anchor_link=...` so the active anchor can be verified directly
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/render_hand_retarget_r1_npz.py`

- Added a stage-level run analysis document:
  - `agent-read/2026-03-25_overall_run_analysis_ZH.md`
  - `agent-read/2026-03-25_overall_run_analysis.md`
- Recorded the current conclusions:
  - trajectory shape has improved, but planner/IK endpoints are still frequently wrong
  - unreached `grasp` is a major reason why `close_gripper` sees no contact
  - current gripper contact detection monitors finger-joint `child_link`s rather than the `left/right_gripper_link` base links
- No runtime logic was changed in this round; this was documentation-only analysis.

- Added stronger gripper-close collision debug output and a "solid object" test mode:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - New parameters:
    - `--debug_collision_report 1`
    - `--execution_object_collision_mode solid_bbox`
  - Changes:
    - `close_grippers_progressively_with_collision_stop(...)` can now print, in debug mode:
      - target-object collision-shape summary
      - `left/right_gripper_link` collision-shape summary
      - finger-link collision-shape summary
      - per progressive-close iteration:
        - `finger_contact`
        - `base_contact`
        - `finger_pairs`
        - `base_pairs`
    - Regular `[gripper-close]` output now also includes:
      - `base_contact=...`
    - Execution-object collision now supports two modes:
      - `convex`: keep the existing `add_convex_collision_from_file`
      - `solid_bbox`: derive mesh bounds and use one axis-aligned solid box as collision
    - `solid_bbox` changes only execution collision, not the visual mesh
    - the batch wrapper now forwards both new parameters
  - Notes:
    - the early-stop criterion still uses finger-link contact only, so behavior is unchanged by default
    - `base_contact` is currently debug-only extra output to reveal whether the gripper base touched the object first
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `git diff --check -- code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`

- Recorded one `d_pour_blue_0` collision-debug conclusion:
  - Command:
    - ran with `--execution_object_collision_mode solid_bbox --debug_collision_report 1`
  - Key output:
    - `planned_object_cup(shapes=1,types=PhysxCollisionShapeBox)`
    - `planned_object_bottle(shapes=1,types=PhysxCollisionShapeBox)`
    - `left_gripper_link(shapes=0)`
    - `right_gripper_link(shapes=0)`
    - `left_gripper_finger_link1/2(shapes=0)`
    - `right_gripper_finger_link1/2(shapes=0)`
    - the entire closing phase stayed at `contact=0` and `base_contact=0`
  - Conclusion:
    - object-side collision is active
    - in the current runtime instance, the gripper base and finger links both show `shapes=0` through the current shape-inspection path
    - therefore full closure is more consistent with “no detectable gripper-side collision shape” than with “objects have no collision”

- Added gripper contact monitoring scope parameter:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - New parameter:
    - `--gripper_contact_monitor_mode {fingers,fingers_and_base,all_robot_links}`
  - Notes:
    - `fingers`
      - keeps the original finger-only stop monitoring
    - `fingers_and_base`
      - also monitors `left/right_gripper_link`
    - `all_robot_links`
      - monitors the full robot articulation link set during close-gripper contact checks; mainly intended to debug whether the gripper-link collisions are missing in the current runtime path

## 2026-03-25 23:10:00 +08

- Added a minimal gripper/object collision probe and ran the first validation round:
  - Files:
    - `code_painting/minimal_gripper_collision_probe.py`
    - `agent-read/2026-03-25_minimal_gripper_collision_probe_ZH.md`
    - `agent-read/2026-03-25_minimal_gripper_collision_probe.md`
  - Goal:
    - verify, without AnyGrasp/IK/stage logic, whether the R1 gripper actually generates raw physics contacts with a probe object during `close_gripper`
    - separate “the main-pipeline contact debug helper is blind” from “the physics engine has no robot-object collision at all”
  - Minimal experiments:
    - box probe:
      - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/minimal_gripper_collision_probe.py --arm left --object_kind box --probe_local_offset 0.04 0.0 0.0 --max_iters 20 --settle_steps_per_iter 8`
    - mesh + solid_bbox probe:
      - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/minimal_gripper_collision_probe.py --arm left --object_kind mesh --mesh_path /home/zaijia001/ssd/data/R1/hand/obj_mesh/blue_cup/blue_cup.obj --mesh_collision_mode solid_bbox --probe_local_offset 0.04 0.0 0.0 --max_iters 20 --settle_steps_per_iter 8`
  - New conclusion:
    - the helper/component summary still reports robot links as `shapes=0`
    - however, in the minimal isolation experiment, `scene.get_contacts()` can already stably observe:
      - `left_gripper_finger_link1<->probe_box`
      - `left_gripper_finger_link2<->probe_box`
      - `left_gripper_finger_link1<->probe_mesh`
      - `left_gripper_finger_link2<->probe_mesh`
      - and for the larger mesh case, `left_gripper_link<->probe_mesh`
    - therefore `shapes=0` can no longer be treated as proof that the robot has no collision geometry or that no physics contact exists
    - the full-pipeline `contact=0` symptom now looks more like a monitoring/matching/timing issue, or a mismatch between the object's real close-stage pose and the visual impression from the video
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/minimal_gripper_collision_probe.py`
    - `git diff --check -- code_painting/minimal_gripper_collision_probe.py`

- Added raw target-contact reporting to the main `close_gripper` debug path and completed a reinjection check:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `agent-read/2026-03-25_minimal_gripper_collision_probe_ZH.md`
    - `agent-read/2026-03-25_minimal_gripper_collision_probe.md`
  - Change:
    - when `debug_collision_report=1`, `close_grippers_progressively_with_collision_stop(...)` now additionally prints:
      - `raw_target_contacts`
      - `raw_target_contact_total`
    - the `[gripper-close]` summary now also includes:
      - `raw_target_contact=0|1`
  - Reinjection validation command:
    - `bash code_painting/run_plan_anygrasp_keyframes_r1_batch.sh ... --debug_collision_report 1 --gripper_contact_monitor_mode all_robot_links --enable_viewer 0`
  - New conclusion:
    - in the current `d_pour_blue_0` case, the full close stage reports not only monitor/helper contact as zero, but also raw target contacts as zero throughout
    - therefore the current issue can no longer be explained only as “the monitor under-reports”; it now looks more like the full close stage never generates raw physics contact with the target object at all

- Added target-object pose / collision debug output to the main close stage:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Change:
    - when `debug_collision_report=1`, `[collision-debug-init]` now additionally prints:
      - `target_pose=...`
      - `target_collision_debug=...`
    - `[collision-debug-step]` now also prints, for each close iteration:
      - `target_pose=...`
  - Reinjection conclusion in this round:
    - in the current `d_pour_blue_0` close stage, the target pose remains essentially stable while raw target contact stays zero throughout
    - therefore the issue now looks more like a substantial geometric mismatch between the visual mesh and the `solid_bbox` collision primitive, rather than the object pose being repeatedly reset during close

- Added execution-object collision-bbox visualization:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - New flag:
    - `--debug_visualize_object_collision_bbox 0|1`
  - Behavior:
    - when an execution object uses `solid_bbox` collision, create an additional visual-only bbox actor for it
    - the bbox actor uses the same local `center/half_size` as the collision primitive
    - and follows the object actor pose
  - Purpose:
    - compare the visual mesh and the collision bbox directly in the viewer/videos
    - determine whether the current "interpenetration" is only a visual-mesh effect
  - Validation:
    - `python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - main-pipeline reinjection with `... --debug_visualize_object_collision_bbox 1 ...`

- Added a `convex` comparison experiment and updated the interpretation:
  - Validation command:
    - `... --execution_object_collision_mode convex --debug_collision_report 1 --gripper_contact_monitor_mode all_robot_links --enable_viewer 0`
  - Result:
    - for the current `d_pour_blue_0` case, the close stage still reports zero raw target contacts under `convex`
  - Conclusion:
    - the issue is not only about the `solid_bbox` box approximation
    - even after switching to `convex` mesh collision, the full close stage still produces no raw physics contact

- Added close-stage pose export and an experiment that enables object collision from `pregrasp`:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - New flag:
    - `--grasp_action_object_collision_start_stage {close_gripper,grasp,pregrasp}`
  - New behavior:
    - write `close_stage_snapshot_*.json` before close
    - allow selected execution objects to start participating in collision as early as `grasp` or `pregrasp`
  - Key experiment result:
    - in the `pregrasp + convex` experiment, raw target contacts appear stably in the full pipeline for the first time
    - but the monitor/helper still reports `monitor_contact=0`
  - Updated conclusion:
    - the old default of enabling object collision only at `close_gripper` is indeed too late
    - at the same time, the current monitor/contact matching logic still under-reports existing contacts

- Added execution-object scale overrides so `cup` / `bottle` can be shrunk independently for execution visual and collision geometry:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/README_anygrasp_keyframe_planner.md`
  - New flags:
    - `--execution_object_scale_override NAME=S`
    - `--execution_object_scale_override NAME=SX,SY,SZ`
  - Behavior:
    - scale both the execution visual mesh and collision shape together
    - under `solid_bbox`, scale bbox center / half_size consistently as well
  - Typical usage:
    - `--execution_object_scale_override cup=0.9 --execution_object_scale_override bottle=0.9`
  - Note:
    - to keep the original logic where object collision starts only before `close_gripper`, continue using:
      - `--grasp_action_object_collision_start_stage close_gripper`

- Ran a control experiment with smaller objects while preserving the old `close_gripper`-only collision timing:
  - Output dir:
    - `code_painting/anygrasp_single_scaled_close_only_probe/d_pour_blue_0`
  - Parameters:
    - `--grasp_action_object_collision_start_stage close_gripper`
    - `--execution_object_scale_override cup=0.9`
    - `--execution_object_scale_override bottle=0.9`
  - Result:
    - from close init through iter 20, `raw_target_contact_total=0`
    - final `raw_target_contact=0`
  - Conclusion:
    - shrinking the execution objects to 0.9 alone is not enough to make the old `close_gripper`-only collision-enable logic detect contact

- Ran full-collision experiments with scales 0.8 / 0.5:
  - Output dirs:
    - `code_painting/anygrasp_single_all_collision_scale08/d_pour_blue_0`
    - `code_painting/anygrasp_single_all_collision_scale05/d_pour_blue_0`
  - Common parameters:
    - `--grasp_action_object_collision_start_stage pregrasp`
    - `--execution_object_collision_mode convex`
  - Result:
    - `0.8`: both left/right detect raw target contact stably during close
    - `0.5`: right has raw contact from close init, left starts showing raw contact later in close, and both finish with `raw_target_contact=1`
  - Conclusion:
    - the full-collision output itself looks normal and trustworthy; the real abnormal part is still the under-reporting monitor/contact chain
    - once collision is enabled early enough, shrinking to `0.8` / `0.5` still does not eliminate raw contact

- Analyzed the `0.5 + pregrasp` full-collision result and added a `0.5 + close_gripper + fingers` control experiment:
  - New output dir:
    - `code_painting/anygrasp_single_scale05_close_only_fingers/d_pour_blue_0`
  - Result analysis:
    - `0.5 + pregrasp` looks closer to the desired effect where the gripper closes onto the object and is visually blocked before appearing fully closed
    - but the current close-stop logic still depends on `monitor_contact`, so the final reason may still show `target_reached`
  - New control result:
    - under `close_gripper + fingers + scale=0.5`, left is still `raw_target_contact=0` while right is `raw_target_contact=1`
  - Conclusion:
    - after shrinking to `0.5`, if collision is still enabled only at close, the problem is not fully gone; the more reliable setup still starts collision from `pregrasp`

- Added separate visual/collision scale overrides for execution objects:
  - New flags:
    - `--execution_object_visual_scale_override`
    - `--execution_object_collision_scale_override`
  - Kept for compatibility:
    - `--execution_object_scale_override`
  - Semantics:
    - visual mesh and collision shape can now be scaled independently
    - if both unified and dedicated overrides are provided for the same object, the dedicated override takes priority

- Added a mechanism-level analysis of why enabling collision only after the gripper is already inside the collision volume does not immediately block at that pose:
  - Key points:
    - late-enabled collision does not roll the system back to the first-touch boundary
    - close stopping depends on monitor_contact + stall, not raw contact directly
    - gripper control is coupled at arm level, not independent per finger
    - execution objects are kinematic actors, so they do not get naturally pushed into a clean separating contact state like dynamic bodies would
- 2026-05-07
  - Added Piper gripper orientation rule docs:
    - `agent-read/PIPER_GRIPPER_ORIENTATION_RULES.zh.md`
    - `agent-read/PIPER_GRIPPER_ORIENTATION_RULES.en.md`
  - Contents:
    - Explains why recent debug wrappers default to the left hand
    - Summarizes HaMeR/NPZ gripper local-axis definitions
    - Summarizes the order of `stored_orientation_post_rot_xyz_deg` and `orientation_remap_label`
    - Summarizes head-camera to world, world to Piper base, and gripper target to `link6` / URDFIK transforms
    - Records the current observation: blue `+Z` is closest to approach, green `+Y` is closest to opening, red `+X` is side/normal

- 2026-05-07
  - Added `code_painting/run_piper_retarget_postrot_board_video.sh`
  - Purpose:
    - Reuse the full Piper retarget replay path
    - Scan candidate rotations only through `stored_orientation_post_rot_xyz_deg`
    - Compose each candidate's `zed_replay.mp4` and `third_replay.mp4` into `board_zed.mp4` / `board_third.mp4`
  - Use case:
    - When direct local-axis scans produce too many IK failures, this keeps the visualization closer to the original retarget replay path
  - Validation:
    - `bash -n code_painting/run_piper_retarget_postrot_board_video.sh`
    - A 1-frame `standard` smoke test successfully generated `/tmp/piper_retarget_postrot_board_smoke/board/board_zed.mp4` and `board_third.mp4`

- 2026-05-07
  - Extended the Piper local-axis sweep tool:
    - Added multi-frame video mode via `--video_mode 1`
    - Added `board_all_zed.mp4` and `board_success_zed.mp4`
    - Added `--candidate_mode semantic`, which emits `forward_from_xp/xm/yp/ym/zp/zm` and `open_from_xp/xm/yp/ym/zp/zm` candidates
  - Goal:
    - Let the id0 gripper position move across frames while each candidate uses a fixed orientation mapping
    - Reduce visual noise from the many failing 24-remap candidates
  - Validation:
    - `python3 -m py_compile code_painting/build_piper_local_axis_sweep_board.py`
    - `bash -n code_painting/run_piper_local_axis_sweep_board.sh`
    - A 2-frame smoke test successfully generated `/tmp/piper_axis_video_smoke/board_all_zed.mp4` and `/tmp/piper_axis_video_smoke/board_success_zed.mp4`

- 2026-05-07
  - Added `code_painting/build_piper_local_axis_sweep_board.py` and `code_painting/run_piper_local_axis_sweep_board.sh`
  - Purpose:
    - Keep the current `PiperPika` scene and head-camera calibration fixed
    - Enumerate all valid right-handed local-axis remaps for one arm on one frame
    - Export `board_zed.png`, `board_third.png`, `summary.json`, and `summary.csv`
  - Why:
    - The existing `orientation_sweep` is oriented toward world-frame target orientation scans and does not directly answer what the HaMeR/recomputed local gripper `x/y/z` axes mean
    - The new script annotates each RGB local axis with its robot-relative `forward/left/up` meaning first, so local-axis semantics can be fixed before execution-error debugging
  - Validation:
    - `python3 -m py_compile code_painting/build_piper_local_axis_sweep_board.py`
    - `bash -n code_painting/run_piper_local_axis_sweep_board.sh`

- 2026-05-11
  - Added `code_painting/run_piper_hamer_axes_replay_batch.sh`
  - Purpose:
    - Batch replay `hand_detections_*.npz`
    - Use the finalized HaMeR/NPZ gripper-axis rule: `orientation_remap_label=identity` and `stored_orientation_post_rot_xyz_deg=0 0 0`
    - Replay both hands by default via `ARMS=both`, with `ID_FILTER` support for single IDs, multiple IDs, and ranges
    - With the default `KEEP_ONLY_ZED_THIRD=1`, clean depth/wrist PNGs under `frames/` and keep only zed/third RGB frames
  - Documentation:
    - The D section in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` now keeps only final replay commands
    - Gripper-orientation debug/sweep/historical-remap commands were moved to `/home/zaijia001/ssd/PIPER_GRIPPER_ORIENTATION_DEBUG.zh.md`
  - Validation:
    - `bash -n code_painting/run_piper_hamer_axes_replay_batch.sh`

- 2026-05-11
  - Extended `code_painting/render_hand_retarget_r1_npz.py`
  - New capability:
    - Overlay FoundationPose object tracks in the Piper/HaMeR hand replay scene
    - New `--object_replay_input_dir` flag points to a video-level FoundationPose output directory
    - New `--object_missing_frame_policy hide|hold_last` flag
    - New `--objects` and `--object NAME=/path/to/mesh.obj` flags for object selection and mesh overrides
  - Added batch script:
    - `code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`
    - Automatically matches each `hand_detections_<id>.npz` with the same-ID FoundationPose object directory
  - Documentation:
    - Added section E, "HaMeR hands + FoundationPose objects in one replay", to `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
  - Validation:
    - `python3 -m py_compile code_painting/render_hand_retarget_r1_npz.py`
    - `bash -n code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`
    - A 1-frame smoke test successfully generated `/tmp/piper_hamer_axes_with_objects_smoke/id_0/zed_replay.mp4` and `third_replay.mp4`

- 2026-05-11
  - Added `code_painting/plot_piper_gripper_wrist_object_axis_distances.py`
  - Purpose:
    - Use the same Piper head-camera calibration to transform HaMeR gripper/wrist-retreat points and FoundationPose object poses into world coordinates
    - Plot world-axis distance curves for left hand vs `pear` and right hand vs `star_fruit`
    - Curves include gripper `dx/dy/dz` and wrist-retreat `dx/dy/dz`
  - Documentation:
    - Added D-debug-9 to `/home/zaijia001/ssd/PIPER_GRIPPER_ORIENTATION_DEBUG.zh.md`
  - Validation:
    - `python3 -m py_compile code_painting/plot_piper_gripper_wrist_object_axis_distances.py`
    - Successfully generated `output_piper_replay_hamer_axes_with_objects_all/id_0/gripper_wrist_object_axis_distance_id0.png` for id0

- 2026-05-18
  - Added the 0515/new_table Piper calibration config:
    - `robot_config_PiperPika_agx_dual_table_0515.json`
  - Calibration sources:
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/head_d435_new_table_0515_head_from_wrist.json`
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/left_base_T_right_base_new_table.json`
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/right_wrist_new_table_eye_in_hand.json`
  - Updated default head/base parameters in:
    - `code_painting/run_piper_hamer_axes_replay_batch.sh`
    - `code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`
    - `code_painting/plot_piper_gripper_wrist_object_axis_distances.py`
  - Updated user commands:
    - C/D/E Piper replay commands in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
    - Added C0 to record the files and values that must be changed after the next calibration update
  - Validation:
    - `bash -n` passed for both Piper replay wrappers
    - `python3 -m json.tool robot_config_PiperPika_agx_dual_table_0515.json` passed
    - `python -m py_compile code_painting/plot_piper_gripper_wrist_object_axis_distances.py` passed
    - A 1-frame id0 smoke test started and produced videos; the log confirmed right base `[0.5562, -0.2718, 0.7698]`, but the old id0 target is IK-failing for both arms under the new calibration, so target offset/reachability should be checked next

- 2026-05-18
  - Corrected the 0515/new_table Piper calibration note by adding the previously omitted left wrist camera extrinsic:
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/left_wrist_new_table_eye_in_hand.json`
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/right_wrist_new_table_eye_in_hand.json`
  - Updated `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`:
    - C0 now explicitly lists the four current new_table JSON files: head 0515, left/right wrist, and left_base_T_right_base
    - Documented that the older `head_d435_new_table_head_from_wrist.json` in the same directory is not used for the current replay commands
    - Added the current base/head-camera placement estimate for checking the physical table setup
    - Fixed the section F distance-plot command that had been broken across lines and was not directly copyable
  - Note:
    - The current D/E replay path still consumes only `robot_config + head_camera`; the left/right wrist extrinsics are recorded for future true wrist-camera rendering where separate left/right local poses will be wired in
  - Validation:
    - Read the four new_table JSON files and computed the derived placement relationship successfully

- 2026-05-18
  - Updated section D in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`:
    - Added `D0. place_bread_basket: HaMeR detection results and human-hand gripper visualization`
    - Recorded the HaMeR GPU detection command for `place_bread_basket/harmer_input -> harmer_output`
    - Added commands to inspect `hand_vis_gripper_*.mp4` for direct HaMeR human-hand gripper-point/axis visualization
    - Added single-ID and batch commands that feed `place_bread_basket/harmer_output/hand_detections_*.npz` into the Piper HaMeR axes replay path
  - Validation:
    - Confirmed that `harmer_output` already contains `hand_detections_0..10.npz` and `hand_vis_gripper_0..10.mp4`

- 2026-05-18
  - Added the Piper calibration bundle workflow:
    - `code_painting/build_piper_calibration_bundle.py`
    - `code_painting/visualize_piper_calibration_bundle.py`
    - `/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json`
  - Updated replay entrypoints:
    - `render_hand_retarget_r1_npz.py` now accepts `--piper_calibration_bundle`
    - `run_piper_hamer_axes_replay_batch.sh` now accepts `CALIBRATION_BUNDLE`
    - `run_piper_hamer_axes_with_objects_replay_batch.sh` now accepts `CALIBRATION_BUNDLE`
  - Purpose:
    - Compose the four hand-eye JSON files for head/base/left_wrist/right_wrist into one self-contained calibration JSON
    - During replay, the bundle writes `calibration_bundle_robot_config.json` under the output directory and overrides the head-camera local pos/quat
    - Provides `axes_compare_old_head.png` to visualize base/head-camera axes and compare the older head extrinsic
  - Observation:
    - The previous head parameters differ from the 0515 head by about `0.123 m` translation and `120.57 deg` rotation
    - The older `head_d435_new_table_head_from_wrist.json` in the same directory is closer to the 0515 head: about `0.039 m` translation and `4.00 deg` rotation
  - Validation:
    - Bundle generation passed
    - Axis PNG generation passed
    - `bash -n` passed for both batch wrappers
    - `py_compile` passed for the three Python files
    - A 1-frame `place_bread_basket` id0 smoke test with `CALIBRATION_BUNDLE=...` completed and loaded the bundle correctly; IK still failed on that frame due to target reachability, not bundle loading

- 2026-05-18
  - Added in-scene head-camera visualization:
    - `render_hand_retarget_r1_npz.py` now accepts `--debug_visualize_cameras`
    - Third-person renders can show the head camera as a white body with red/green/blue local xyz axes and a yellow `-Z` optical ray
    - `run_piper_hamer_axes_replay_batch.sh` and `run_piper_hamer_axes_with_objects_replay_batch.sh` now accept `DEBUG_VISUALIZE_CAMERAS/DEBUG_CAMERA_AXIS_LENGTH/DEBUG_CAMERA_AXIS_THICKNESS`
  - Updated `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`:
    - Documented that the SAPIEN viewer requires `DISPLAY/WAYLAND_DISPLAY`; in the current SSH-only session with `DISPLAY=None`, no viewer window will appear
    - Added three-version head-camera marker comparison commands for `place_bread_basket`:
      - earliest manual head parameters + old robot config
      - pre-0515 new_table head bundle
      - 0515 new_table head bundle
  - Validation:
    - A 1-frame `place_bread_basket` id0 smoke test with `DEBUG_VISUALIZE_CAMERAS=1` completed
    - Successfully generated `/tmp/place_bread_basket_camera_marker_smoke/id_0/frames/third_0000.png`
    - `bash -n` passed for both batch wrappers
    - `py_compile` passed for replay and bundle visualization scripts

- 2026-05-18
  - Improved SAPIEN viewer diagnostics:
    - `render_hand_retarget_r1_npz.py` now prints `DISPLAY/WAYLAND_DISPLAY/XDG_SESSION_TYPE` before creating the viewer
    - Successful viewer creation prints `[viewer] interactive viewer created`
    - Viewer creation failures now catch broader exceptions and print the exception type
    - Added `code_painting/probe_sapien_viewer.py` to independently verify a minimal SAPIEN viewer from a VNC terminal
  - Updated `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`:
    - Viewer command now prints the display environment first
    - Viewer command includes `--debug_visualize_cameras 1 --debug_camera_axis_length 0.22`
    - Added a minimal viewer probe command
  - Note:
    - The previous `output_place_bread_basket_piper_viewer_probe` command did not enable `debug_visualize_cameras`, so its third-person image would not show the head-camera marker
  - Validation:
    - `DEBUG_VISUALIZE_CAMERAS=1 DEBUG_CAMERA_AXIS_LENGTH=0.5` generated `/tmp/place_bread_basket_camera_marker_big_smoke/id_0/frames/third_0000.png` with visible head-camera axis markers

- 2026-05-18
  - Identified why the viewer window did not appear:
    - In the VNC terminal, `probe_sapien_viewer.py` creates a viewer successfully when `CUDA_VISIBLE_DEVICES` is not set
    - The hand replay command failed with `Renderer does not support display` when `CUDA_VISIBLE_DEVICES=2` was set
    - Conclusion: the viewer needs access to the GPU driving the VNC/X display, and `CUDA_VISIBLE_DEVICES=2` hides that display GPU from the process
  - Updates:
    - `render_hand_retarget_r1_npz.py` viewer diagnostics now include `CUDA_VISIBLE_DEVICES`
    - The viewer command in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` no longer sets `CUDA_VISIBLE_DEVICES=2`
    - Added a negative-control probe command with `CUDA_VISIBLE_DEVICES=2`
  - Validation:
    - `bash -n code_painting/run_piper_hamer_axes_replay_batch.sh`
    - `python -m py_compile code_painting/render_hand_retarget_r1_npz.py code_painting/probe_sapien_viewer.py`

- 2026-05-18
  - Identified and fixed the root cause of the 0515 head-camera orientation offset:
    - `left_base_T_head_camera` in the hand-eye JSON is a raw/optical camera frame
    - Replay commands using `--camera_cv_axis_mode legacy_r1` require a render/SAPIEN camera pose
    - The correct relation is `T_render = T_raw_optical @ legacy_r1.T`
  - Key validation:
    - The earliest manual head position differs from `head_d435_try2_head_from_wrist.json` raw translation by only `3.3e-7 m`
    - The earliest manual head quaternion differs from the try2 raw rotation by `120.0 deg`
    - The earliest manual head quaternion differs from `try2_raw @ legacy_r1.T` by `0.0 deg`
    - Therefore the observed `120 deg` difference is mainly a camera-axis convention difference, not physical calibration drift
  - Updates:
    - `build_piper_calibration_bundle.py` now stores the raw optical head transform and converts `head_camera.left_base_T_head_camera` to the render/SAPIEN pose used by replay
    - Regenerated `calibration_bundle_piper_new_table_0515.json` and `calibration_bundle_piper_new_table_pre0515.json`
    - Updated the D/E wrappers and `plot_piper_gripper_wrist_object_axis_distances.py` default head quaternion to the 0515 render/SAPIEN quaternion
    - Updated the direct C/D/E commands in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` that spell out the head quaternion
  - Calibration quality observations:
    - 0515 head residual mean/max: `0.607/2.193 deg`, `0.0048/0.0167 m`
    - pre-0515 head residual mean/max: `0.574/1.298 deg`, `0.0059/0.0151 m`
    - left wrist residual mean/max: `0.563/0.985 deg`, `0.0129/0.0193 m`
    - right wrist residual mean/max: `1.795/3.567 deg`, `0.0103/0.0180 m`
  - Validation:
    - `py_compile` passed for `build_piper_calibration_bundle.py`, `render_hand_retarget_r1_npz.py`, and `plot_piper_gripper_wrist_object_axis_distances.py`
    - `bash -n` passed for both Piper HaMeR batch wrappers
    - `json.tool` passed for both new_table calibration bundles
    - `CALIBRATION_BUNDLE=...new_table_0515.json DEBUG_VISUALIZE_CAMERAS=1 MAX_FRAMES=1 ID_FILTER=0` smoke test passed; the bundle loaded correctly and generated `/tmp/place_bread_basket_camera_axis_fixed_smoke/id_0/third_replay.mp4`

- 2026-05-19
  - Added a replay target offset along the gripper-local blue `+Z` approach axis:
    - `render_hand_retarget_r1_npz.py` now accepts `--target_local_forward_retreat_m`
    - Positive values mean `target_position -= distance * local(+Z)`, i.e. retreat opposite the visualized blue approach axis
    - The local retreat is applied after camera-to-world conversion and before the ordinary `target_world_offset_xyz`, so it follows each frame's gripper orientation instead of a fixed world XYZ direction
  - Updated wrappers:
    - `run_piper_hamer_axes_replay_batch.sh` now accepts `TARGET_LOCAL_FORWARD_RETREAT_M`
    - `run_piper_hamer_axes_with_objects_replay_batch.sh` now accepts `TARGET_LOCAL_FORWARD_RETREAT_M`
  - Compatibility fixes:
    - Added the new renderer constructor argument in `build_piper_local_axis_sweep_board.py` and `plot_piper_gripper_wrist_object_axis_distances.py`
  - Validation:
    - `py_compile` passed for `render_hand_retarget_r1_npz.py`, `build_piper_local_axis_sweep_board.py`, and `plot_piper_gripper_wrist_object_axis_distances.py`
    - `bash -n` passed for both Piper HaMeR batch wrappers
    - `TARGET_LOCAL_FORWARD_RETREAT_M=0.05 MAX_FRAMES=1 ID_FILTER=0` smoke test passed and printed `[target-local-retreat] along_local_plus_z_blue_m=0.0500`

- 2026-05-20
  - Fixed a renderer-constructor compatibility issue in FoundationPose multi-object replay:
    - After `HandRetargetR1Renderer.__init__` gained camera-debug and local-blue-axis retreat arguments, `render_object_pose_r1_npz.py` still used the old constructor argument list, causing `run_multi_object_pose_r1_npz_batch.sh` to fail with `missing 4 required positional arguments`
    - Added default values in the `ReplayRenderer(...)` constructor calls in `render_object_pose_r1_npz.py`, `replay_r1_h5.py`, and `minimal_gripper_collision_probe.py`
  - Updated the command library:
    - Added a note around line 182 in C1 of `/home/zaijia001/ssd/RoboTwin/COMMAND_LIBRARY.zh.md` to mark the pick_diverse_bottles command as a Piper 0515 head/base calibrated FoundationPose two-object replay command
  - Validation:
    - `py_compile` passed for `render_object_pose_r1_npz.py`, `replay_r1_h5.py`, `minimal_gripper_collision_probe.py`, and `render_multi_object_pose_r1_npz.py`
    - `bash -n` passed for `run_multi_object_pose_r1_npz_batch.sh`
    - A pick_diverse_bottles id0 smoke test with `--max_frames 1 --skip_existing 0` passed and generated `/tmp/pick_diverse_bottles_foundation_replay_smoke/foundation_input_0/head_cam_replay.mp4` and `multi_object_world_poses.npz`

- 2026-05-20
  - Updated the E2 single-video replay commands in `COMMAND_LIBRARY.zh.md`:
    - Added hand + FoundationPose object replay commands for pick_diverse_bottles, place_bread_basket, and stack_cups
    - All three commands use `--piper_calibration_bundle calibration_bundle_piper_new_table_0515.json`
    - All three commands include `--target_local_forward_retreat_m 0.05` to retreat 5 cm opposite the gripper-local blue `+Z` approach axis
    - E2 now documents the viewer toggle: append `--enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02`
  - Validation:
    - Ran a pick_diverse_bottles id0 smoke test with `--max_frames 1` successfully
    - The log confirmed `[target-local-retreat] along_local_plus_z_blue_m=0.0500`
    - The log confirmed FoundationPose objects `['left_bottle', 'right_bottle']` were loaded

- 2026-05-20
  - Changed the E2.1/E2.2/E2.3 H2O replay commands in `COMMAND_LIBRARY.zh.md` from single id0 runs to batch id0-id10 runs:
    - Use `for ID in $(seq 0 10)`
    - Inputs now use `hand_detections_${ID}.npz`
    - FoundationPose object directories now use `foundation_input_${ID}`
    - Output directories now use `id${ID}_z005`
  - Updated the viewer note:
    - To enable the viewer, first narrow `seq 0 10` to a single ID such as `seq 0 0`
    - Then append `--enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02`
  - Validation:
    - Ran `bash -n` syntax checks for the three batch loop commands; did not run all 33 replays

- 2026-05-20
  - Added section G at the end of `COMMAND_LIBRARY.zh.md`:
    - Added H2O id0-id10 commands for plotting world-axis distances from gripper/wrist-retreat points to FoundationPose object centers
    - Covered pick_diverse_bottles, place_bread_basket, and stack_cups
    - Each ID writes a PNG plus matching CSV under `code_painting/human_object_replay/h2o/.../id${ID}_z005/`
  - Added interpretation notes:
    - Stable same-direction `dz` offsets across tasks/IDs point more toward the head/depth/camera-to-world or replay calibration chain
    - Object-specific or frame-local jumps point more toward FoundationPose pose/depth/mesh estimation
  - Validation:
    - `bash -n` passed for all three id0-id10 loop commands
    - A pick_diverse_bottles id0 `--max_frames 2` smoke test passed and generated `/tmp/pick_diverse_bottles_axis_distance_id0_smoke.png` and `.csv`

- 2026-05-26
  - Added a LeRobot cache episode-subset script:
    - Added `policy/pi0/scripts/subset_lerobot_episodes.py` to copy selected episodes from an existing `/home/zaijia001/.cache/huggingface/lerobot/local/<dataset>` cache.
    - Supports `--episodes '0-24'` and `--episodes '0,1-5,7'`; the parsed ids are deduplicated and sorted by old episode id.
    - Writes to a new `--output-repo-id` and rewrites `episode_index`, `frame_index`, global `index`, `meta/info.json`, `meta/episodes.jsonl`, and `meta/episodes_stats.jsonl` so the subset is numbered continuously as `0..N-1`.
    - Updated `COMMAND_LIBRARY.zh.md` L11 and `agent-read/COMMANDS/pi0_h2o_training_data.*.md` with the 25-episode subset command and output checks.
  - Validation:
    - `uv run python -m py_compile scripts/subset_lerobot_episodes.py` passed.
    - Tested a temporary subset from `local/h2o_pick_diverse_bottles_human_head_pure_action` using `0,1-2,7`; verified `total_episodes=4`, reindexed episodes `0..3`, and present parquet plus three-camera videos, then removed the temporary repo.

- 2026-05-21
  - Updated `plot_piper_gripper_wrist_object_axis_distances.py`:
    - Added `--plot_clip_abs_m`, defaulting to `0.5`
    - PNG plotting now clips values beyond `±plot_clip_abs_m` to the plot boundary so the sub-0.5m trend remains readable
    - CSV output still keeps raw unclipped values, and plot titles report clipping and clipped-value counts
    - Missing FoundationPose object `poses.npz` files no longer abort the whole plot; the script prints a warning and writes NaNs for that side
  - Data observations:
    - There are now 33 H2O task/id CSVs; place_bread_basket id5/id6 are missing `bread/poses.npz`, but the script can still generate the left-side basket curves
    - Treating `|value|>0.5m` as large outliers, the normal-frame overall dz medians are about gripper `+0.150m` and wrist `+0.169m`
  - Validation:
    - `py_compile` passed
    - A pick_diverse_bottles id0 `--max_frames 2` clipped smoke test passed
    - place_bread_basket id5/id6 generated PNG/CSV files despite missing bread tracks

- 2026-05-21
  - Added a raw HaMeR hand point vs FoundationPose object point comparison tool:
    - `code_painting/make_hamer_foundation_point_compare_video.py`
    - Inputs HaMeR `hand_detections_<id>.npz`, `hand_vis_gripper_<id>.mp4`, and FoundationPose object directories
    - Outputs a horizontal comparison video with a HaMeR hand-point panel plus each object's `mesh_overlay.mp4` panel
    - Overlays thumb tip, index tip, thumb/index midpoint, and projected object center
    - Writes a paired CSV with camera-frame `hand_midpoint - object_center` `dx/dy/dz`
  - Updated the H section in `COMMAND_LIBRARY.zh.md`:
    - Recorded the H2O three-task id0-id10 replay CSV statistics from section G
    - Added id0-id10 raw point comparison commands for pick_diverse_bottles, place_bread_basket, and stack_cups
  - Data notes:
    - In inlier frames, overall `abs dz median` is about `15.1cm` for gripper and `17.0cm` for wrist-retreat
    - Signed dz median is about `+15.0cm` for gripper and `+16.9cm` for wrist-retreat
    - pick/place contain a small number of meter-scale z outliers; stack_cups has no `>0.5m` outliers
  - Validation:
    - `py_compile` passed
    - place_bread_basket id0 `--max_frames 5` smoke test succeeded and produced `/tmp/hamer_foundation_point_compare_place_bread_basket_id0.mp4` plus `.csv`

- 2026-05-21
  - Extended `make_hamer_foundation_point_compare_video.py`:
    - Added default distance-curve PNG output named `*_distance.png`
    - The plot shows camera-frame `dx/dy/dz` from the HaMeR thumb/index midpoint to the FoundationPose object center
    - Added `--output_plot` to override the plot path
    - Added `--plot_clip_abs_m`, defaulting to `0.5`, matching section G behavior by clipping only the PNG display while preserving raw CSV values
  - Updated `COMMAND_LIBRARY.zh.md` H2/H6:
    - H2 now documents the video, CSV, and distance-curve PNG outputs
    - H6 now includes a finder command for `*_hamer_foundation_points_distance.png`
  - Validation:
    - `py_compile` passed
    - place_bread_basket id0 `--max_frames 5` smoke test succeeded and produced a video, CSV, and distance-curve PNG

- 2026-05-21
  - Analyzed the newly generated raw HaMeR/FoundationPose CSV files from section H:
    - Found 11 CSVs for `pick_diverse_bottles` id0-id10
    - The inlier overall `abs dz median` in camera-frame `hand_midpoint - object_center` is about `5.1cm`
    - In comparison, the G/H1 world replay statistics for pick have gripper/wrist `abs dz median` about `14.6cm/16.5cm`
    - Conclusion: the 15cm-level z offset is not mainly present in the raw detection points; it is more likely introduced by camera-to-world conversion, `target_world_offset_xyz`, retreat-point definition, and replay coordinate-chain effects
  - Updated `COMMAND_LIBRARY.zh.md`:
    - Added E2.0 before E2.1/E2.2/E2.3 for pure hand replay on the three H2O tasks without loading FoundationPose objects
    - Added the current H raw CSV statistics and comparison against G/H1 world replay statistics after H1
  - Validation:
    - `bash -n` passed for the three E2.0 id0-id10 pure hand replay loop commands

- 2026-05-21
  - Updated `COMMAND_LIBRARY.zh.md` E2.0:
    - Changed the three pure hand replay commands from `--save_png_frames 1` to `--save_png_frames 0`
    - Documented that `--save_png_frames 0` avoids saving per-frame PNG files under `frames/` and keeps only the main replay mp4/npz outputs
    - Added a VS Code-compatible transcode command using `ffmpeg -c:v libx264 -pix_fmt yuv420p -movflags +faststart`
  - Note: raw replay mp4 files may use a codec or pixel format unsupported by VS Code/Chromium; the comparison videos are viewable because the ffmpeg hstack command re-encodes them as H.264/yuv420p
  - Validation:
    - `bash -n` passed for the three E2.0 loop commands and the single ffmpeg transcode command

- 2026-05-21
  - Expanded the Piper H2O debug/generation workflow in `COMMAND_LIBRARY.zh.md`:
    - Added E0 pure replay to create clean zed/third RGB robot videos for later repainting.
    - Added I/J/K to connect SAM hand removal, pure replay repainting, AnyGrasp candidate filtering, AnyGrasp keyframe replay, and repainting.
  - This round only changed command documentation and agent-read logs; no long rendering or repainting jobs were run.
  - Validation:
    - Extracted the new command blocks and `bash -n` passed.
    - Confirmed the repaint and AnyGrasp entry scripts exist.

- 2026-05-21
  - Fixed the section-I SAM repaint command docs so Stage-1 hand removal no longer skips everything just because the full E0 pure robot replay set is incomplete.
  - I1 now uses a dummy robot video; I2 still uses E0 pure robot videos and reports more specific missing paths.
  - Validation: section-I bash blocks passed `bash -n`.

- 2026-05-22
  - Fixed the Stage-1 background input paths for I2/K2 in `COMMAND_LIBRARY.zh.md`.
  - The background file is actually under `stage1_human_inpaint/removed_w_mask_*.mp4`, so the commands now fall back to that path when the top-level `human_hand_bg.mp4` alias is absent.
  - Updated the I1 output check command accordingly.
  - Validation: the I/K2 repaint command blocks passed `bash -n`; sampled id0/id1/id10 across all three tasks and each resolved to an existing Stage-1 background file.

- 2026-05-22
  - Updated the K section in `COMMAND_LIBRARY.zh.md`:
    - Added a K0 manual keyframe review flow before K1.
    - Added a command to generate `hand_keyframes_all.json` from a TSV file.
    - Added a command to rerun AnyGrasp preview summaries using the manual keyframes.
    - Added a bad-id dry-run/move command that records `_rejected_human_ids/rejected_ids.json`.
  - Validation: the K0 bash command blocks passed `bash -n`.

- 2026-05-22
  - Parameterized the video-directory prefix in `run_render_anygrasp_ranked_preview_keyframes_batch.sh`:
    - The default remains `d_pour_blue` for compatibility with the old AnyGrasp flow.
    - `VIDEO_PREFIX=foundation_input` now supports H2O task directories named `foundation_input_<id>`.
  - Updated K0.3/K1 in `COMMAND_LIBRARY.zh.md`:
    - K0.3 uses the batch wrapper to generate manual-keyframe preview summaries for whole tasks.
    - K1 now processes whole tasks instead of only `id0-id10`.
  - Validation: the wrapper and the K0.3/K1 documented command blocks passed `bash -n`.

- 2026-05-22
  - Added a normal D435 head-view record to `COMMAND_LIBRARY.zh.md`:
    - Recorded the headD435 source, RGB/depth topics, `fx/fy/cx/cy`, and `640x480` resolution from `pick_diverse_bottles/origin/episode35/head_d435_rgbd_meta.json` and `harmer_input/params_35.json`.
    - Recorded the real-intrinsics equivalent `fovy_deg=42.499880046655484` and noted that the default replay uses `640x360 + fovy_deg=90`, which makes the rendered view wider and the Piper projection smaller.
    - Added E2.4 D435 pure Piper replay commands and I3 D435 inpainting/repainting commands, using `h2_pure_d435`, `results_repaint_piper_h2_d435`, and `d435` in output filenames.
  - Validation: extracted the D435 replay/repaint documentation commands and checked them with `bash -n`.
  - Further clarified the D435 FOV source:
    - Stated that `42.499880046655484°` comes from the recorded RGB camera_info intrinsics `fy=617.160888671875, height=480`, not from image dimensions alone; `640x480` alone cannot determine FOV.
    - Added the distinction between the official D435 depth FOV `85.2 x 58` and the nominal color camera FOV `H:69 / V:42 / D:77`; this replay aligns `/camera/color/image_raw`, so `fovy_deg=42.499880046655484` remains the command value.
  - Added the I3 D435 repaint anomaly diagnosis:
    - Confirmed that I3 `BG` still points to the I1 `removed_w_mask_rgb_<id>.mp4`; the background path itself was not the direct mistake.
    - Documented the Stage-2 compositing semantics: source pixels are copied from `robot_video` to the target background only inside the robot mask.
    - Compared old/D435 id0 `w_mask/w_box/final` outputs and confirmed that the D435 GroundingDINO/SAM2 mask covers large simulated-background regions, so the final output copies the robot replay background as well.
    - Recorded that both the original human video and D435 replay are `640x480`; the old replay is `640x360`, but Stage-2 resizes it to the target background size.

- 2026-05-23
  - Further updated the I3 D435 repaint diagnosis in `COMMAND_LIBRARY.zh.md`:
    - Added the user-confirmed root cause: with the narrower D435 FOV, some videos do not show the robot in frame 0 or the first few frames, so Stage-2 first-frame detection can select the white simulated background or table as the robot mask and propagate that wrong target.
    - Documented mitigation guidance: make the Stage-2 initialization frame contain a clearly visible robot first; thresholds alone cannot fix a completely robot-free first frame.
    - Recommended starting D435 tuning with more conservative `--robot_box_threshold 0.30~0.40`, `--robot_text_threshold 0.25~0.35`, `--robot_max_mask_area_ratio 0.20~0.35`, and a more explicit robot prompt.
    - Added a SAM3 Stage-2 robot repaint command that directly reuses the I1 background, with a note that the current SAM3 script also initializes from frame 0.
  - Validation:
    - The new I3.2/I3.3 bash commands passed `bash -n`.

## 2026-05-28 (Piper AnyGrasp Gripper Axis Fix And IK Position-First Diagnostic)

- Findings:
  - The D435 preview gripper wireframe uses the AnyGrasp visible gripper frame, with `local +X = rotation_matrix[:, 0]`.
  - Piper reports the visible gripper pose as `R_report = R_link6 @ global_trans_matrix @ delta_matrix`.
  - The current Piper config uses `global_trans_matrix=diag(1,-1,-1)` and `delta_matrix=I`.
  - A hard-coded inverse `global_trans_matrix` was considered, but the correct direct Piper hand replay behavior shows that it must not be enabled by default.
- Code updates:
  - `render_hand_retarget_piper_dual_npz_urdfik.py` now defaults to the original direct Piper hand replay URDFIK convention and exposes `urdfik_apply_global_trans_to_ik` only as a diagnostic comparison switch.
  - `plan_anygrasp_keyframes_r1.py` keeps Piper dual `reach_error_pose_source=ee` in the visible gripper-frame convention instead of converting the target into raw link6 space.
  - `cartesian_interp_ik` partial mode now tries a shorter sub-waypoint between the current pose and the first failed waypoint when waypoint 1 fails.
  - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` now accepts `--ik_max_position_threshold_m` and `--ik_max_rotation_threshold_rad` for direct position-first diagnostics.
  - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --viewer` now defaults to `--viewer_wait_at_end 0`, so batch viewer runs automatically proceed to the next id. Use `--viewer_wait_at_end 1` only when an end-of-id pause is desired.
  - The wrapper now supports `--id_start`, `--id_end`, `--ids`, and `--piper_apply_global_trans_to_ik`.
- Validation:
  - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/render_hand_retarget_piper_dual_npz_urdfik.py code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --max_per_task 1 --tasks pick_diverse_bottles --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --visualize_targets --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --output_root /tmp/axis_fix_pos_first_dryrun`
  - With the default `--ik_max_rotation_threshold_rad 0.12`, the smoke test still failed at Cartesian waypoint 1, showing that static behavior was not caused by step/settle settings.
  - With `--ik_max_rotation_threshold_rad 3.14`, `pick_diverse_bottles id0` printed continuous `[exec-pose]` lines, confirming that waypoint execution works and the main blocker is the orientation constraint in complete-pose IK.
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --max_per_task 1 --tasks pick_diverse_bottles --viewer --viewer_wait_at_end 1 --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --visualize_targets --ik_max_rotation_threshold_rad 3.14 --output_root /tmp/viewer_wait_dryrun`
  - `bash code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --tasks pick_diverse_bottles --id_start 0 --id_end 10 --viewer --visualize_targets --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /tmp/id_filter_dryrun`

- 2026-05-25
  - Further updated I3 in `COMMAND_LIBRARY.zh.md`:
    - Added the I3.4 "visible-frame SAM reinitialization mode" design note; this is documentation only and does not modify the current code.
    - Compared the current SAM2/SAM3 Stage-2 fixed frame-0 initialization against the proposed mode: the new mode would detect the robot frame by frame, initialize SAM only after a valid robot is found, emit empty masks while the robot is absent, and reinitialize when it reappears.
    - Documented the suggested state machine, mask-validity checks, proposed future interface, and separate output root `results_repaint_piper_h2_d435_sam3_visible_reinit` for later A/B comparison against the current SAM outputs.
  - Validation:
    - The proposed I3.4 future command passed `bash -n`.

- 2026-05-26
  - Fixed the Piper AnyGrasp two-keyframe planning entrypoint:
    - `code_painting/plan_anygrasp_keyframes_piper.py` no longer uses the old R1-style single-base Piper wrapper.
    - The entrypoint now reuses `PiperDualReplayRenderer` and `HandRetargetPiperDualURDFIKRenderer`, matching the direct Piper hand-coordinate replay path.
    - Left and right arms keep their independent base poses from `robot_config_PiperPika_agx_dual_table_0515.json`, and IK converts world targets to base coordinates per arm.
  - Updated documentation:
    - Added `COMMAND_LIBRARY.zh.md` section L14 with a Piper AnyGrasp dual-base Cartesian-waypoint debug command.
    - Added `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md`.
  - Validation:
    - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_piper.py` passed.

- 2026-05-28
  - Added D435 AnyGrasp candidate-selection commands:
    - Added `COMMAND_LIBRARY.zh.md` section J0.1 to check six-task AnyGrasp, `foundation_replay_d435`, and HaMeR NPZ availability.
    - Added J1.1 to generate six-task D435 candidate preview/summary outputs from manual keyframes in `hand_keyframes_all.json`.
    - D435 summaries are written to `code_painting/anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json`, keeping them separate from the default-wide `anygrasp_h2o_preview`.
    - The command supports a `place_bread_basket` fallback to `place_bread_basket_output_old_cam`.
    - Updated `agent-read/COMMANDS/pi0_h2o_training_data.zh.md` / `.en.md` accordingly.
  - Validation:
    - Checked six-task `foundation_replay_d435` inputs and several AnyGrasp roots.
    - The new J0.1/J1.1 command blocks passed `bash -n`.

- 2026-05-28
  - Updated the L11.2.4 D435 robot replay `_25ep` subset logic:
    - It no longer blindly selects LeRobot episodes `0-24`.
    - It now reads `processed_data/h2o_<TASK>_pure_d435_visible_reinit-120/episode_*/instructions.json/source_episode_id` and aligns filtering by original id.
    - It excludes original bad ids `0,7,12,29` for `handover_bottle` and `0,1,2,3,4,5,6,22,70` for `pnp_bread`, then fills 25 episodes.
    - Updated `agent-read/COMMANDS/pi0_h2o_training_data.zh.md` / `.en.md` accordingly.
  - Validation:
    - Checked the `source_episode_id` field in the D435 processed data.
    - The updated L11.2.4 command block passed `bash -n`.

- 2026-05-28
  - Added an explanation of where D435 finals come from and which SAM2 fallback batch entrypoint to use:
    - `d435_final` is `results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/<TASK>/id_<ID>_d435/final_repainted.mp4`.
    - It is generated by I3.5 through `batch_visible_reinit_d435_repaint.py`, not by L8.2.
    - This machine does not have `Grounded_SAM_3`, so the I3.5 batch path currently logs `[backend] SAM=sam2, DINO=dino2`.
    - The batch path prints `loading DINO once` and `loading SAM image predictor once`, so it loads checkpoints once and then loops over task/id jobs.
    - Added a SAM2/DINO2 fallback batch command in `COMMAND_LIBRARY.zh.md` I3.5 for first filling each new task to at least 25 final videos.
  - Validation:
    - Checked the checkpoint-loading and `final_repainted.mp4` copy logic in `batch_visible_reinit_d435_repaint.py` and `remove_anything_video_sam3_robot_visible_reinit.py`.
    - The new command block passed `bash -n`.

- 2026-05-28
  - Investigated why the new-three-task D435 robot replay processed-HDF5 counts in `pro5-17` are low:
    - `pro5-17` is/was running the L8.2 conversion stage; it only reads existing `final_repainted.mp4` files and does not create missing D435 repaint outputs.
    - The new three tasks have enough `h2_pure_d435` retarget outputs: `handover_bottle=51`, `pnp_bread=81`, `pnp_tray=51`.
    - The missing parts are Stage-1 BGs and I3.5 D435 repaint finals; current final counts are approximately `handover_bottle=12`, `pnp_bread=2`, `pnp_tray=14`.
    - Added `COMMAND_LIBRARY.zh.md` section I1.1.1 with a command that only fills missing Stage-1 BGs.
    - Added an I3.5 0..80 resume command using `--overwrite 0` to fill missing D435 repaint finals before rerunning L8.2.
    - Updated `agent-read/COMMANDS/pi0_h2o_training_data.zh.md` / `.en.md` accordingly.
  - Validation:
    - Checked `tmux capture-pane -pt pro5-17` and confirmed the skip log is from L8.2 missing `final_repainted.mp4`.
    - The new command blocks were checked with `bash -n`.

- 2026-05-28
  - Fixed and rechecked the Piper D435 AnyGrasp keyframe execution reach issue:
    - `target_pose_for_error(..., ee)` in `plan_anygrasp_keyframes_r1.py` now uses `world_pose_to_base_pose_for_arm/base_pose_to_world_pose_for_arm` for Piper dual arms, avoiding conversion of right-arm targets through the left-arm base.
    - `planned_eval_pose_from_plan()` no longer treats `target_pose_world` as the planned pose; it evaluates target joints through FK and the robot gripper/endlink static transform.
    - In partial diagnostic execution, an arm whose plan failed is held at its current joints while the other arm executes, preventing physics drift.
    - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` now accepts `--reach_error_pose_source` and defaults it to `ee`. The old `tcp` check leaves a fixed approximately 12 cm TCP/EE offset.
  - `pick_diverse_bottles id0` recheck:
    - Under the old `tcp` check, the right-arm grasp showed about `0.125m` error; with `ee` plus the per-arm base fix, the right-arm grasp position error is about `0.0057m`.
    - The episode still fails overall because the left-arm first-keyframe IK/target fails; strict dual synchronization blocks the stage.
  - Documentation:
    - Added L15.7 to `COMMAND_LIBRARY.zh.md` for the current keyframe execution logic, EE reached-check rationale, and separate six-task commands.
    - Updated `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md`.
  - Validation:
    - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_piper.py` passed.
    - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` passed.
    - The six-task wrapper dry-run covered one summary for each task.
    - Reran `pick_diverse_bottles id0` with partial/joint_interp/pose printing and confirmed right-arm EE reach while the failed left arm is held.

- 2026-05-28
  - Added Piper D435 AnyGrasp preview comparison export and viewer target visualization:
    - The planner now copies the reused summary's original D435 preview images and same-path legacy preview images into `<OUT>/source_preview_compare/`.
    - Added `selected_candidate_mapping.json`, recording each frame/arm selected `candidate_idx`, rank, source translation, planner raw pose, planner target pose, and `candidate_target_local_x_offset_m`.
    - The six-task wrapper now supports `--visualize_targets`; viewer debugging disables `pure_scene_output` and shows target axes / active candidate grippers.
  - Validation:
    - `pick_diverse_bottles id0` produced `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/2-pdb/pick_diverse_bottles/foundation_input_0/source_preview_compare/` with frame 38/78 D435 and legacy `orientation/fused` images plus the mapping JSON.
    - `plan_summary.json` records 13 `source_preview_compare` entries.
    - `py_compile` and wrapper `bash -n` passed.

- 2026-05-28
  - Clarified and completed the six-task D435 robot replay data-processing chain:
    - `COMMAND_LIBRARY.zh.md` now states in L6.1 that the command only applies to the default-wide `h2_pure` path; the new three tasks lack both `h2_pure` and default-wide repaint heads, so L6.1 skips all of them.
    - Added L8.2: six-task D435 visible-reinit robot repaint head + `h2_pure_d435` action/wrist to processed HDF5.
    - Added L10.6: six-task `h2o_<TASK>_pure_d435_visible_reinit-120` to LeRobot cache.
    - Added L11.2.4: six-task D435 robot replay `_25ep` subset, packaging, and upload-check commands.
    - Updated `agent-read/COMMANDS/pi0_h2o_training_data.zh.md` / `.en.md` with the D435 workflow.
  - Current path check:
    - The old three tasks have both default-wide `h2_pure` and D435 `h2_pure_d435`.
    - The new three tasks only have D435 `h2_pure_d435`; they do not have default-wide `h2_pure`.
    - The new three tasks currently only have partial D435 visible-reinit repaint outputs, so I1.1/I3.5 must be completed before L8.2/L10.6/L11.2.4.
  - Validation:
    - The new L8.2/L10.6/L11.2.4 command blocks were checked with `bash -n`.

- 2026-05-28
  - Updated the H2O pi0 data-processing command docs:
    - Added `COMMAND_LIBRARY.zh.md` section L11.1.3 for the post-L5.2 -> L10.5 new-three-task human head + D435 action/wrist `_25ep` subset step.
    - Clarified that post-L10.5 subsetting must not use `local/h2o_<TASK>_pure_repaint`; that repo belongs to L6/L6.1 robot replay data.
    - L11.1.3 reads `instructions.json/source_episode_id` from the processed data, excludes original bad ids `0,7,12,29` for `handover_bottle` and `0,1,2,3,4,5,6,22,70` for `pnp_bread`, then fills the first 25 usable episodes.
    - Updated `agent-read/COMMANDS/pi0_h2o_training_data.zh.md` / `.en.md` accordingly.
  - Validation:
    - The check in this round confirmed that naive `0-24` includes original ids `7,12` for `handover_bottle` and original id `22` for `pnp_bread`.
    - The documented command blocks passed `bash -n` syntax checks.

- 2026-05-26
  - Added a Piper hand raw `origin/` review script:
    - `code_painting/review_piper_hand_origin.py`
    - It reads `/home/zaijia001/ssd/data/piper/hand/<TASK>/origin/episode*/camera/color/headD435/*.png` directly, so bad data can be filtered before AnyGrasp / foundation preprocessing.
    - Pressing `b` / `d` moves the current episode directory to the sibling `bad/` directory and records the decision in `<TASK>/origin_bad_review.json`.
  - Moved all `/home/zaijia001/ssd/data/piper/hand/handover_bottle/origin/episode*` directories to `/home/zaijia001/ssd/data/piper/hand/handover_bottle/bad/` as requested. After the move, `origin/` has 0 episode directories and `bad/` has 51.
  - Updated docs:
    - Added `COMMAND_LIBRARY.zh.md` section L13.
    - Added `agent-read/COMMANDS/piper_hand_origin_review.zh.md` / `.en.md`.
  - Validation:
    - `python code_painting/review_piper_hand_origin.py --help` passed in the `RoboTwin_openvla` environment.
    - The default system `python3` lacks `cv2`, so the documented commands explicitly activate `RoboTwin_openvla`.

- 2026-05-25
  - Implemented the I3.4 visible-frame SAM reinitialization mode:
    - Added `/home/zaijia001/ssd/inpainting_sam3_robot/remove_anything_video_sam3_robot_visible_reinit.py` without changing the existing SAM2/SAM3 script interfaces.
    - Added `/home/zaijia001/ssd/inpainting_sam3_robot/batch_visible_reinit_d435_repaint.py`; the batch path loads DINO/SAM checkpoints once and then processes task/id jobs in a loop.
    - Single-video logic: inactive state detects robot frame by frame; a valid candidate initializes SAM; tracking state prompts SAM on the current frame with an expanded bbox from the previous valid frame; invalid/lost masks become empty masks and the script waits for a later valid robot detection to reinitialize.
    - Updated `COMMAND_LIBRARY.zh.md` I3.4 from a design note to implemented-mode documentation, including single-video, batch, and dry-run commands.
  - Validation:
    - `python3 -m py_compile` passed for both new scripts.
    - Both new scripts run `--help` in the `inpainting-sam3-dino3` environment.
    - A batch dry-run for `pick_diverse_bottles id0` resolved the I1 background and D435 robot replay inputs correctly.
    - The I3.4 single-video, batch, and dry-run documentation commands passed `bash -n`.

- 2026-05-25
  - Fixed a GroundingDINO loading compatibility issue in the I3.4 new script under the `inpainting-sam3-dino3` environment:
    - The user hit `AttributeError: 'BertModel' object has no attribute 'get_head_mask'`.
    - Cause: the old GroundingDINO `BertModelWarper` under `Grounded_SAM_2` expects the older transformers `BertModel.get_head_mask` helper, but the current environment's transformers version no longer exposes it.
    - Added a local compatibility patch in `remove_anything_video_sam3_robot_visible_reinit.py`: the new script dynamically adds `get_head_mask` to `transformers.BertModel` at runtime, without editing third-party sources or changing the original SAM2/SAM3 script interfaces.
  - Updated `COMMAND_LIBRARY.zh.md`:
    - Renamed I3.3 to clearly indicate it is the current SAM3-project first-frame initialization path, and noted that the real backend is determined by the `[backend] SAM=...` startup log.
    - Renamed I3.4 to "new logic: visible-frame reinitialization SAM2/SAM3 mode".
    - Explicitly documented the difference among the original SAM2 command, the current SAM3-project command, and the new visible-reinit command.
  - Validation:
    - In the current environment, `BertModel.get_head_mask` is `False` before the patch and `True` after the patch.
    - `VisibleReinitRobotSegmenter` completes DINO/SAM model loading in the `inpainting-sam3-dino3` environment and prints `[ok] model loaded`.
    - Both new scripts pass `py_compile`.

- 2026-05-25
  - Further fixed the old GroundingDINO / new transformers compatibility issue in the `inpainting-sam3-dino3` environment:
    - The user then hit `TypeError: to() received an invalid combination of arguments - got (dtype=torch.device, )`.
    - Cause: the old `BertModelWarper` calls `get_extended_attention_mask(attention_mask, input_shape, device)` using the old transformers API, while the current transformers 5.3.0 expects `dtype` as the third argument.
    - Added a unified `patch_transformers_bert_for_groundingdino()` helper in `/home/zaijia001/ssd/inpainting_sam3_robot/remove_anything_video_sam3_robot.py`, patching both `get_head_mask` and `get_extended_attention_mask`.
    - Updated `/home/zaijia001/ssd/inpainting_sam3_robot/remove_anything_video_sam3_robot_visible_reinit.py` to reuse that unified compatibility helper.
  - Updated `COMMAND_LIBRARY.zh.md`:
    - Added I3.0 as a comparison section listing the original fixed-first-frame SAM2 command, the SAM3-project fixed-first-frame command, and the new visible-frame reinitialization command with their entrypoints, backend behavior, initialization logic, and output roots.
    - Added I3.0.1 original SAM2 single-id command, I3.0.2 SAM3-project single-id command, a true-SAM3 backend template, I3.0.3 new visible-reinit single-id command, and I3.0.4 new visible-reinit batch command.
    - Documented that this machine currently only has `Grounded_SAM_2`, not `Grounded_SAM_3`, so the SAM3-project commands currently fall back to SAM2/DINO2.
  - Validation:
    - The three related scripts passed `py_compile`.
    - A DINO forward smoke test in the `inpainting-sam3-dino3` environment printed `[ok] detect call completed 0`, confirming both compatibility failures are bypassed.
    - The new I3.0 command blocks passed `bash -n`.
## 2026-05-29 (Fix D435 AnyGrasp Viewer First-Keyframe Target Display)

- Fixed execution-preview frame selection in `plan_anygrasp_keyframes_r1.py`:
  - Added `active_frame_by_arm` to `DebugExecutionState`.
  - Dual-arm `pregrasp/grasp` now display each arm's first keyframe target/candidate gripper; `action` switches to each arm's second keyframe.
  - `record_frame()`, `update_candidate_debug_visuals()`, and the debug execution preview now use the per-arm active frame, avoiding the previous left-arm-only `active_frame` behavior that could show the later or wrong frame in the viewer.
  - `pose_debug.jsonl` and `execution_metrics.jsonl` now record `active_frame_by_arm` for viewer-frame auditing.
- Documentation synced:
  - Added L15.14 to `COMMAND_LIBRARY.zh.md` with the cause, fix, and `jq` inspection command.
  - Updated `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` with L15.14.
- Validation:
  - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py` passed.
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` passed.
  - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --tasks pick_diverse_bottles --id_start 0 --id_end 10 ...` resolved 11 D435 summaries for ids 0-10.
  - A non-viewer `pick_diverse_bottles id0 --debug_stop_after_keyframe1` smoke run wrote to `/tmp/anygrasp_active_frame_by_arm_check`; `pose_debug.jsonl` showed `stage=pregrasp active_frame_by_arm={"left": 38, "right": 38}`, confirming that keyframe-1 display state is correct.
## 2026-05-29 (Stack Cups id0 No-Collision Target-Only Debug)

- Added D435/Piper six-task wrapper debug switches:
  - `--disable_execution_collisions`: passes `--enable_grasp_action_object_collision 0` to the planner to rule out grasp/action object collision and contact-stop close logic.
  - `--target_axes_only`: keeps only the active execution target axes and hides candidate gripper axes, selected-keyframe axes, and IK waypoint markers, reducing viewer coordinate-frame clutter.
  - Added passthrough controls for `--debug_candidate_top_k`, `--debug_common_candidate_top_k`, `--debug_visualize_selected_keyframe_axes`, and `--debug_visualize_ik_waypoints`.
- Added `--debug_visualize_selected_keyframe_axes` to `plan_anygrasp_keyframes_r1.py` so execution previews can hide selected-keyframe axis actors.
- Rechecked `stack_cups/foundation_input_0`:
  - The D435 summary uses per-arm keyframes: left `[139, 195]`, right `[51, 106]`.
  - Therefore `active_frame_by_arm={"left": 139, "right": 51}` during pregrasp is expected.
  - The earlier viewer "four coordinate systems" came from simultaneously showing active target axes, selected-keyframe axes, candidate gripper axes, and IK waypoint markers.
- Validation:
  - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py` passed.
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` passed.
  - The `--dry_run --tasks stack_cups --ids 0 --target_axes_only --disable_execution_collisions ...` command resolved correctly.
  - A no-viewer smoke run under `/tmp/stack_cups_id0_no_collision_target_axes_only` confirmed `enable_grasp_action_object_collision=0`, but stack_cups id0 still failed to reach; the current primary issue is not object collision but IK/trajectory execution tracking error and candidate-pose reachability.
## 2026-05-29 (Added Direct Piper Hand Replay Viewer Comparison Command)

- Added L15.16 to `COMMAND_LIBRARY.zh.md`:
  - Provides a `stack_cups id0` viewer command for directly replaying the stored HaMeR NPZ gripper poses.
  - Shows both the target gripper axes and the robot execution.
  - Enables `--debug_mode 1 --debug_post_execute 1` to inspect target EE/TCP pose versus actual execution error.
  - Saves `world_targets_and_status.npz` for later target pose/status inspection.
- Updated `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` and the command changelog.
## 2026-05-29 (Documented Direct Replay Versus AnyGrasp Axis Convention)

- Documented the gripper-frame difference between direct replay and the AnyGrasp planner:
  - Direct Piper hand replay's stored gripper frame uses local `+Z` blue as the approach/forward axis.
  - The AnyGrasp preview wireframe uses `rotation_matrix[:, 0]`, local `+X` red, as the finger-depth direction from palm/back bar to fingertips.
- Updated `plan_anygrasp_keyframes_r1.py` parameter help and diagnostic comments to explicitly distinguish AnyGrasp local `+X` from direct replay local `+Z`.
- Added L15.17 to `COMMAND_LIBRARY.zh.md` with a single-run `stack_cups id0` `--candidate_orientation_remap_label swap_red_blue` comparison command, testing whether AnyGrasp local `+X` should be mapped to direct replay local `+Z`.
- Added `policy/openvla-oft/runs/` to `.gitignore` to keep 96G of run outputs out of git; also ignored `*.bak` / `*.bak2` backup files.

## 2026-05-29 (Replay-Axis AnyGrasp Keyframe Runner)

- Added planner parameters:
  - `--candidate_target_local_z_offset_m`: applies AnyGrasp target compensation along target local `+Z`.
  - `--approach_axis local_x|local_z`: chooses whether pregrasp retreat uses local `+X` or local `+Z`.
- Kept old defaults unchanged:
  - `--candidate_target_local_x_offset_m` retains its existing meaning.
  - `--approach_axis` defaults to `local_x`, preserving the old AnyGrasp planner behavior.
- Added D435 six-task wrapper passthrough parameters:
  - `--candidate_orientation_remap_label`
  - `--candidate_target_local_x_offset_m`
  - `--candidate_target_local_z_offset_m`
  - `--approach_axis`
  - `--approach_offset_m`
- Added the separate entrypoint `code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh`:
  - Forces `swap_red_blue`.
  - Disables local-X compensation and enables `--candidate_target_local_z_offset_m -0.05`.
  - Forces `--approach_axis local_z`.
- Documentation:
  - `COMMAND_LIBRARY.zh.md` now has L15.18 explaining the direct-replay local +Z blue-axis convention versus the AnyGrasp local +X red-axis convention, plus six-task viewer/no-viewer commands.
  - `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` now include L15.18.
- Validation:
  - `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py` passed.
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` passed.
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh` passed.
  - A no-viewer `run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --tasks stack_cups --ids 0 --debug_stop_after_keyframe1 ...` smoke run completed under `/tmp/stack_cups_id0_replay_axes_check/stack_cups/foundation_input_0`; `plan_summary.json` records `candidate_target_local_z_offset_m=-0.05` and `approach_axis=local_z`, and VSCode-compatible `head_cam_plan.mp4` / `third_cam_plan.mp4` were generated. In this debug run, both arms reached keyframe-1 grasp.

## 2026-05-29 (L15.19 Candidate-Stage Gripper/Robot Frame Unification Design Note)

- Documentation-only update as requested; no code was changed.
- Added L15.19 to the end of `COMMAND_LIBRARY.zh.md`:
  - Records why AnyGrasp gripper frame and robot/replay frame should be unified during candidate filtering.
  - Clarifies that viewer axis colors remain red/green/blue for local +X/+Y/+Z.
  - Clarifies that orange in rank previews is the right-hand/candidate gripper color, not an axis color.
  - Records the long-term recommended frame: `robot local +Z = AnyGrasp raw local +X`, `robot local +Y = AnyGrasp raw local +Y`, and `robot local +X = -AnyGrasp raw local +Z`.
  - Adds the current runnable comparison command that writes to `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper`.
- Updated `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md`.

## 2026-05-29 (Implemented L15.19 Robot-Frame Preview And Planner Path)

- Code changes:
  - `render_anygrasp_ranked_preview.py` now supports `--candidate_frame_mode robot_replay` and `--candidate_target_local_z_offset_m`.
  - Robot-frame candidate rotations are saved as `target local +Z = AnyGrasp raw local +X`, `target local +Y = AnyGrasp raw local +Y`, and `target local +X = -AnyGrasp raw local +Z`.
  - 2D preview C-shaped gripper wireframes can use local Z as the fingertip direction.
  - `plan_anygrasp_keyframes_r1.py` adds `--debug_gripper_actor_forward_axis local_z` so viewer/rank-preview C-gripper actors draw their fingertip direction along blue local +Z.
  - `run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` adds `--preview_root` and `--debug_gripper_actor_forward_axis` passthroughs.
- New scripts:
  - `code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh`
  - `code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh`
- Documentation:
  - `COMMAND_LIBRARY.zh.md` now includes L15.19.1 with robot-frame preview generation and viewer_gripper planner commands.
  - `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` were updated with the same command flow.
- Validation:
  - `python3 -m py_compile code_painting/render_anygrasp_ranked_preview.py code_painting/plan_anygrasp_keyframes_r1.py` passed.
  - `bash -n code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh`, `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh`, and `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` passed.
  - `run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh --tasks pick_diverse_bottles --ids 0` generated a robot-frame summary; `summary.json` records `candidate_frame_mode=robot_replay` and `candidate_target_local_z_offset_m=-0.05`.
  - A no-viewer `run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --tasks pick_diverse_bottles --ids 0 --debug_stop_after_keyframe1 ...` smoke run generated `head_cam_plan.mp4` / `third_cam_plan.mp4`. The smoke run confirms the new path executes; the right arm still did not fully reach, so remaining work is IK/candidate reachability tuning rather than entrypoint or frame serialization failure.

## 2026-05-29 (Fixed Robot-Frame Planner Only Running id0)

- Root cause:
  - `run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh` runs over the available robot-frame preview summaries.
  - At the time, `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435_robot_frame` contained only `pick_diverse_bottles/foundation_input_0/summary.json`, so the planner could only run id0.
- Fix:
  - The robot-frame preview wrapper now supports `--max_per_task`, `--id_start`, `--id_end`, `--skip_existing`, and `--source_preview_root`.
  - The robot-frame planner wrapper now auto-generates missing robot-frame summaries for the same task/id/max range before invoking the planner.
  - Added `--skip_preview_generation` for cases where only existing summaries should be used.
- Validation:
  - `bash -n code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh` passed.
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh` passed.
  - `run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --max_per_task 2 --dry_run --tasks pick_diverse_bottles ...` auto-generated the id1 summary and listed id0/id1 in the planner dry-run.
  - `run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --max_per_task 1 --dry_run ...` reached planner dry-run for all six tasks; some tasks start from an id other than 0 because the available D435 preview summary ids determine the order.

## 2026-05-29 (Added Robot-Frame Viewer Commands With Explicit ids)

- Added L15.19.2 to `COMMAND_LIBRARY.zh.md`:
  - `stack_cups id4` viewer debug command.
  - Per-task viewer templates with explicit ids.
  - One command for all six tasks with the same explicit id set.
- Updated `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` with the same explicit-id commands.
- Validation:
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh` passed.
  - `bash -n code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh` passed.
  - `run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --ids 4 --dry_run --tasks stack_cups ...` correctly listed `stack_cups/foundation_input_4/summary.json`.

## 2026-06-02 (Mode O Piper Gripper Axis Audit And Chinese Comments)

- Checked the Piper/Pika gripper orientation definition against the original ALOHA-AgileX setup:
  - The high-level config values `global_trans_matrix=diag(1,-1,-1)`, `delta_matrix=I`, and `grasp_perfect_direction=["front_right","front_left"]` match.
  - The URDF gripper structural axes differ: ALOHA-AgileX's finger-depth/fingertip direction is naturally link6 local `+X`, while Piper/Pika's gripper opening axis is gripper-base local `Z/-Z`.
  - Mode O currently follows the Piper/replay target-frame convention and uses local `+Z` as the approach/forward axis. This matches direct replay and robot-frame AnyGrasp, but it is not the original ALOHA-style local `+X` convention.
- Added Chinese comments to `code_painting/plan_first_frame_foundation_pick_diverse_bottles.py`, covering the file purpose, target generation logic, pose storage order, local `+Z` approach-axis convention, and the difference from ALOHA-style local `+X`.
- Updated `COMMAND_LIBRARY.zh.md` and `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` with the Mode O gripper-orientation audit.
- Validation:
  - `python3 -m py_compile code_painting/plan_first_frame_foundation_pick_diverse_bottles.py` passed.

## 2026-06-02 (Mode O Gripper Frame Visualization And ALOHA-Style Local-X Comparison)

- Added `code_painting/visualize_mode_o_gripper_frame_conventions.py`:
  - Does not run IK or open the SAPIEN viewer.
  - Reads FoundationPose object positions for `pick_diverse_bottles`.
  - Draws `piper_local_z`, `aloha_local_x_y_up`, and `aloha_local_x_z_up` frames at the same grasp target.
  - Writes a PNG and JSON; the JSON records the angle between local X/Y/Z and the physical approach direction.
- Added to `code_painting/plan_first_frame_foundation_pick_diverse_bottles.py`:
  - `--target_frame_convention piper_local_z|aloha_local_x_y_up|aloha_local_x_z_up`
  - `--plan_only`
  - When `aloha_local_x_*` is selected, the planner automatically receives `--approach_axis local_x` and `--debug_gripper_actor_forward_axis local_x`.
- Updated `code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh` to pass through `--target_frame_convention` and `--plan_only`.
- Updated `.gitignore` to allow the new visualization script while still ignoring generated PNG/JSON debug outputs.
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_first_frame_foundation_pick_diverse_bottles.py code_painting/visualize_mode_o_gripper_frame_conventions.py` passed.
  - `bash -n code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh` passed.
  - The visualization script successfully generated PNG/JSON for `pick_diverse_bottles id0 frame0`; the result shows `piper_local_z` local Z is `0deg` from the physical approach direction, while both ALOHA-style local-X variants have local X at `0deg`.
  - `--plan_only --target_frame_convention aloha_local_x_z_up` successfully wrote a summary recording `target_frame_convention=aloha_local_x_z_up` and `planner_approach_axis=local_x`.

## 2026-06-02 (O.0 Piper Data Generation Entry With Original demo_clean Logic)

- Added `envs/pick_diverse_bottles_piper.py`:
  - Inherits the original `pick_diverse_bottles`.
  - Does not modify `envs/pick_diverse_bottles.py`.
  - Preserves the original bottle random sampling, random rotation, left/right regions, `grasp_actor`, lift, and place logic.
- Added `task_config/demo_clean_piper.yml`:
  - Based on `demo_clean.yml`.
  - Sets `embodiment` to `[piper, piper, 0.60]`.
- Added `description/task_instruction/pick_diverse_bottles_piper.json`:
  - Reuses the original `pick_diverse_bottles` instruction template so `collect_data.py` can generate episode instructions after collection.
- Updated `.gitignore` to allow `task_config/demo_clean_piper.yml`.
- Recommended command:
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper 0`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile envs/pick_diverse_bottles_piper.py script/collect_data.py` passed.
  - Dynamic import of `envs.pick_diverse_bottles_piper` in the conda environment passed, and the class MRO shows inheritance from `pick_diverse_bottles`.
  - `task_config/demo_clean_piper.yml` parsed as `embodiment=['piper','piper',0.60]` and `episode_num=50`.
  - `description/task_instruction/pick_diverse_bottles_piper.json` parsed as `seen=50` and `unseen=10`.

## 2026-06-02 (Fixed gen-23 O.0 collect_data Command)

- Original `gen-23` errors:
  - Running `bash collect_data.sh ...` from `~` could not find the script.
  - After entering the repo, the old `demo_clean_piper.yml` used `embodiment: [piper]`, which triggered the dual-arm embodiment path and made RoboTwin look for missing `assets/embodiments/piper/curobo_left.yml`.
  - Later `'Robot' object has no attribute 'left_planner'` messages were secondary errors after planner initialization failed and the task reused a partial `robot` object.
- Fix:
  - Changed `task_config/demo_clean_piper.yml` to `embodiment: [piper, piper, 0.60]`, using two single-arm Piper instances and `curobo.yml`.
  - Updated the O.0 docs to use the full command with conda activation and repo path.
- Validation:
  - Parsed `task_config/demo_clean_piper.yml` as `['piper', 'piper', 0.6]`.
  - A conda-activated `timeout 35s bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper 0` smoke run did not reproduce the `curobo_left.yml` or `left_planner` errors.
  - Cleaned up the temporary `data/pick_diverse_bottles_piper/demo_clean_piper/` trajectory output from the smoke run so the user can start collection from episode 0.

## 2026-06-03 (Switched O.0 To The Calibrated Piper/Pika Embodiment)

- Problem:
  - The user successfully generated `data/pick_diverse_bottles_piper/demo_clean_piper/`, but the rendered robot did not look like the calibrated Piper setup.
  - Inspection showed that the old `demo_clean_piper.yml` no longer used ALOHA-AgileX, but it still loaded RoboTwin's built-in `assets/embodiments/piper/config.yml` and `piper.urdf`, not the calibrated `piper_pika_agx` setup corresponding to `robot_config_PiperPika_agx_dual_table_0515.json`.
- Fix:
  - Added `assets/embodiments/piper_pika_agx/config.yml`, using the calibrated `piper_pika_agx.urdf`, Piper/Pika gripper joints, left/right base poses, `delta_matrix=I`, and `global_trans_matrix=diag(1,-1,-1)`.
  - Added `piper_pika_agx_calibrated` to `task_config/_embodiment_config.yml`.
  - Changed `task_config/demo_clean_piper.yml` to `embodiment: [piper_pika_agx_calibrated, piper_pika_agx_calibrated, 0.0]`.
  - Added `task_config/demo_clean_piper_calibrated.yml`; this is the recommended config for a separate output directory so the old `demo_clean_piper` data is not mixed with calibrated data.
  - Updated `.gitignore` to allow `assets/embodiments/piper_pika_agx/config.yml` and `task_config/demo_clean_piper_calibrated.yml`, while generated data remains ignored.
- New recommended command:
  - `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0`
- Validation:
  - Python/YAML checks confirmed both `demo_clean_piper` and `demo_clean_piper_calibrated` resolve to `assets/embodiments/piper_pika_agx/piper_pika_agx.urdf`, with gripper base `left_joint` and the calibrated left/right base poses.

## 2026-06-03 (gen1-2 O.0 Head-Only Config And Error Diagnosis)

- Inspected `tmux gen1-2`:
  - `No left camera link` / `No right camera link` is a fallback warning from `envs/robot/robot.py` when wrist camera links cannot be found. It is not the direct exception that caused seed collection to fail.
  - The actual failure happens during seed/premotion collection; episode 0 repeatedly failed from seed 421 through seed 730.
  - The main failure types are bottle instability, `Objects is unstable ... 001_bottle`, and `target_pose cannot be None for move action`, where the original scripted demo path cannot produce an executable move target.
- Changes:
  - Set `collect_wrist_camera: false` in `task_config/demo_clean_piper.yml` and `task_config/demo_clean_piper_calibrated.yml`, so collection stores only the head view.
  - Added `task_config/demo_clean_piper_calibrated_viewer.yml` with `render_freq: 1`, `episode_num: 1`, `collect_data: false`, and `collect_wrist_camera: false` for one-episode viewer/head-only debugging.
  - Updated `.gitignore` to allow the new viewer config.
- Conclusion:
  - Disabling wrist saving removes wrist-link data dependency and reduces log noise.
  - If `target_pose cannot be None` continues, the root issue is a mismatch between the original ALOHA-style `pick_diverse_bottles.py` demo planner and the calibrated Piper/Pika geometry/reachability. The next fix should be a Piper/Pika-specific task variant.

## 2026-06-03 (Fixed O.0 Viewer No-Output Command And Added Scene-Only Viewer)

- Rechecked `tmux gen1-2`:
  - The minimal `probe_sapien_viewer.py` can create a viewer, so the current VNC/display session is usable.
  - The previous documented viewer command contained `bash ./script/.update_path.sh ... && python ...`, but this repo has no `script/.update_path.sh`; the redirected failure stopped the `&&` chain, making the command return with no output.
  - Using `script/collect_data.py` directly as a viewer entrypoint is also not appropriate because it continues into the original `play_once` / `grasp_actor` planner, which can still fail with `target_pose cannot be None for move action` on the calibrated Piper/Pika setup.
- Changes:
  - Added `run_collect_piper_calibrated_viewer.sh`, removing the missing `.update_path.sh` dependency, unsetting `CUDA_VISIBLE_DEVICES` for viewer mode, and setting the NVIDIA Vulkan ICD automatically when available.
  - Added `view_pick_diverse_bottles_piper_scene.py` and `run_view_pick_diverse_bottles_piper_scene.sh`; this loads the `pick_diverse_bottles_piper` scene only, skips `play_once` planning, skips unstable seeds automatically, and stops in the SAPIEN viewer.
  - Updated the preferred viewer command to `bash run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50`.

## 2026-06-03 (gen1 Viewer Completion Semantics And No-Viewer Generation Notes)

- Rechecked `tmux gen1`:
  - The scene-only viewer successfully loaded the calibrated Piper/Pika setup; seed 0/1 were skipped due to bottle instability and seed 2 loaded successfully.
  - This viewer entrypoint is for interactive inspection. It stays in the render loop until the user closes the window or presses `Ctrl-C`; it does not run the full demo or generate data automatically.
  - Closing the SAPIEN window previously raised `AttributeError: 'NoneType' object has no attribute 'should_close'`.
  - The no-viewer generation command, `bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0`, starts the calibrated head-only config but still fails during seed search with `Objects is unstable` and `target_pose cannot be None for move action`.
- Changes:
  - Updated `view_pick_diverse_bottles_piper_scene.py` to exit cleanly when the viewer window is closed or becomes `None`.
  - Updated `COMMAND_LIBRARY.zh.md` and the O.0 command docs with the viewer completion semantics, the no-viewer generation command, and the current failure cause.

## 2026-06-03 (Fixed Mode M/N Viewer CUDA Mask Restoration)

- After inspecting `tmux modeln-4`:
  - The Mode N `pnp_tray` command did pass `--viewer`, and the planner attempted to create an interactive viewer.
  - The failure log showed `[viewer] creating interactive viewer ... CUDA_VISIBLE_DEVICES=2`, followed by `Renderer does not support display`.
  - The user's minimal probe can create a SAPIEN viewer in the graphical session after `unset CUDA_VISIBLE_DEVICES`.
- Root cause:
  - The bash wrappers already used `env -u CUDA_VISIBLE_DEVICES` in viewer mode.
  - `plan_keyframes_human_replay.py` and `plan_keyframes_foundation_pose.py` then restored `CUDA_VISIBLE_DEVICES=2` from `--gpu` before invoking the planner.
- Fix:
  - Both Python middle layers now remove `CUDA_VISIBLE_DEVICES` from the planner environment when `--enable_viewer 1` is active.
  - Non-viewer mode still uses `--gpu` for compute GPU selection.
- Documentation:
  - Added Mode M/N viewer notes to `COMMAND_LIBRARY.zh.md`, covering the required `DISPLAY` and unset CUDA mask.
  - Updated `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` with the diagnosis, minimal probe, and validation commands.
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_keyframes_foundation_pose.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_keyframes_human_replay.py` passed.
  - `timeout 60s bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh --gpu 2 --ids 0 --viewer --viewer_wait_at_end 0 --tasks pnp_tray --foundation_pose_retreat_m 0.03 --output_root /tmp/robotwin_viewer_env_probe` passed. In this non-graphical shell `DISPLAY=None`, so the viewer still fell back to offscreen, but the key log changed to `CUDA_VISIBLE_DEVICES=None`.

## 2026-06-03 (O.0 Motion Baseline Runs Calibrated Piper/Pika Follow-Up Motion)

- Rechecked issues:
  - `run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50` is a scene-only viewer. It only loads a stable seed and waits in the SAPIEN viewer; it does not run `play_once` or any follow-up motion.
  - `collect_data.sh pick_diverse_bottles_piper demo_clean_piper_calibrated 0` still enters the original `pick_diverse_bottles.py` `grasp_actor` path. `tmux gen1-1/gen1-2` repeatedly showed `Objects is unstable` and `target_pose cannot be None for move action`, meaning the original ALOHA-style EE grasp target generation is not compatible with the calibrated Piper/Pika setup.
- Changes:
  - Added `envs/pick_diverse_bottles_piper_motion.py`, inheriting the original `pick_diverse_bottles` bottle random sampling, random rotation, left/right placement regions, and stability check, while bypassing the original `choose_grasp_pose/grasp_actor`.
  - The task uses conservative joint-space stages around the calibrated Piper/Pika home pose: approach, lower, close gripper, lift/retract, move outward, and open gripper.
  - Added `task_config/demo_clean_piper_motion.yml` for no-viewer/head-only data saving, plus `task_config/demo_clean_piper_motion_viewer.yml` and `run_pick_diverse_bottles_piper_motion_viewer.sh` for motion viewer checks.
  - Added `description/task_instruction/pick_diverse_bottles_piper_motion.json` so `collect_data.py` can save instruction files.
  - Added Chinese comments to `envs/pick_diverse_bottles_piper_motion.py` clarifying that this is a joint-space motion baseline and does not mean true bottle grasping is solved.
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile envs/pick_diverse_bottles_piper_motion.py view_pick_diverse_bottles_piper_scene.py` passed.
  - `bash -n run_pick_diverse_bottles_piper_motion_viewer.sh run_view_pick_diverse_bottles_piper_scene.sh run_collect_piper_calibrated_viewer.sh` passed.
  - YAML/JSON parsing passed. Both `demo_clean_piper_motion` configs use `piper_pika_agx_calibrated+piper_pika_agx_calibrated` and `collect_wrist_camera: false`.
  - The no-viewer command `timeout 180s bash collect_data.sh pick_diverse_bottles_piper_motion demo_clean_piper_motion 0` succeeded: seed 0/1 were skipped as unstable, seed 2 reached `simulate data episode 0 success`, and 64 head-camera frames plus `episode0.hdf5`, `episode0.mp4`, `episode0.pkl`, and instruction json were saved.
  - In `tmux gen1-1`, `bash run_pick_diverse_bottles_piper_motion_viewer.sh` successfully ran through seed 2 premotion and returned to the shell.

## 2026-06-03 (O.0 Original IK/Planning Path And Piper/Pika TCP Experiment)

- Rechecked issues:
  - The currently working `pick_diverse_bottles_piper_motion` data-generation path does not call the original `pick_diverse_bottles.py` IK/planning path; it directly creates joint-space interpolation.
  - The original task path is `grasp_actor/place_actor -> Action(move target_pose) -> Base_Task.move -> robot.left/right_plan_path -> _trans_from_gripper_to_endlink -> CuroboPlanner.plan_path`.
  - The existing `assets/embodiments/piper_pika_agx/curobo.yml` still points to the old `/assets/embodiments/piper/piper.urdf` and uses the old Piper `link7/link8/joint7/joint8`, which do not match the Pika gripper links and joints in `piper_pika_agx.urdf`.
- Changes:
  - Added the `piper_pika_agx_ik_orig_tcp` embodiment. It keeps the calibrated `piper_pika_agx.urdf` and left/right base poses, but uses the built-in RoboTwin Piper TCP conversion matrices for `delta_matrix/global_trans_matrix`.
  - Added matching `curobo.yml` and `collision_piper_pika.yml` so Curobo also uses `piper_pika_agx.urdf`, `gripper_base_link/gripper_left_link/gripper_right_link`, and `left_joint/right_joint`.
  - Added `task_config/demo_clean_piper_ik_orig_tcp.yml`. The command still uses `pick_diverse_bottles_piper`, so it genuinely enters the original `pick_diverse_bottles.py` `grasp_actor/place_actor` IK/planning path.
- Validation:
  - YAML parsing passed for the new task config, embodiment config, Curobo config, and collision config; `_embodiment_config.yml` resolves `piper_pika_agx_ik_orig_tcp`.
  - `py_compile envs/pick_diverse_bottles_piper.py envs/pick_diverse_bottles_piper_motion.py` passed.
  - `git diff --check` passed.
  - `timeout 120s bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_ik_orig_tcp 0` confirmed `Embodiment Config: piper_pika_agx_ik_orig_tcp+piper_pika_agx_ik_orig_tcp` and the original `pick_diverse_bottles_piper` task, but did not finish an episode within 120 seconds. Failures remained mainly `Objects is unstable` and `target_pose cannot be None for move action`.

## 2026-06-04 (O.0 Command Cleanup And gen1 Error Recheck)

- Rechecked tmux:
  - There is no standalone pane named `gen1`; the actual sessions are `gen1-1` and `gen1-2`.
  - Both panes show the same latest behavior: from seed 72 through 115, failures alternated between `Objects is unstable` and `target_pose cannot be None for move action`, then the user interrupted with `Ctrl-C`.
  - Conclusion: the command was running the original task/IK path, not the tested `pick_diverse_bottles_piper_motion` path. The remaining failure is still that the original `choose_grasp_pose/grasp_actor` cannot reliably generate executable targets for the calibrated Piper/Pika setup.
- Documentation cleanup:
  - Rewrote the O.0 section in `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md`, keeping only four titled commands: no-viewer data generation, motion viewer, scene-only viewer, and original-IK diagnostic.
  - Rewrote the O.0 section in `COMMAND_LIBRARY.zh.md` and removed the duplicate old O.0 head-only/motion section at the file end.
  - `pick_diverse_bottles_piper demo_clean_piper_calibrated` is no longer kept as a recommended command; it is only mentioned as a failing original-`grasp_actor` path.

## 2026-06-04 (O.0 Viewer-Only Planner Skip And Motion Viewer Fix)

- tmux recheck:
  - The original-IK diagnostic command in `gen1-1` / `gen1-2` does enter `envs/pick_diverse_bottles.py -> grasp_actor -> choose_grasp_pose -> CuroboPlanner.plan_batch`.
  - The failures have two causes: `Objects is unstable` is the bottle-settling stability check after random placement; `target_pose cannot be None for move action` means the original grasp-candidate path did not produce an executable target for the calibrated Piper/Pika setup.
  - `No left camera link` / `No right camera link` is only the fallback warning from the current `piper_pika_agx.urdf` lacking wrist camera links. It is not the IK failure root cause; O.0 remains head-only.
- Changes:
  - Added `skip_planner=True` support to `Base_Task.load_robot()` so viewer-only scene loading can avoid Curobo planner initialization.
  - Added a default gripper-only planner placeholder in `Robot` initialization. It only supports initial gripper interpolation, and the normal IK/data-collection path still replaces it with Curobo via `set_planner()`.
  - Added `--show_axes` and `--hold` to `view_pick_diverse_bottles_piper_scene.py`. It now shows RGB axes on the two bottle centers and the left/right place targets by default, and scene-only loading skips both planner and `play_once`.
  - Added `view_pick_diverse_bottles_piper_motion.py`, bypassing `collect_data.py` `seed.txt` progress so each viewer debug run searches for a stable seed and executes `play_once()` once.
  - Updated `run_pick_diverse_bottles_piper_motion_viewer.sh` to call the new motion viewer entrypoint.
- Validation:
  - `python -m py_compile envs/_base_task.py envs/robot/robot.py view_pick_diverse_bottles_piper_scene.py view_pick_diverse_bottles_piper_motion.py` passed.
  - `bash -n run_pick_diverse_bottles_piper_motion_viewer.sh run_view_pick_diverse_bottles_piper_scene.sh` passed.
  - `DISPLAY=:1.0 timeout 90s python view_pick_diverse_bottles_piper_scene.py --seed 0 --max_seed_tries 3 --hold 0` passed: seed 0/1 were skipped as unstable, seed 2 loaded, axes were added, and one frame rendered before exit.
  - `DISPLAY=:1.0 timeout 120s bash run_pick_diverse_bottles_piper_motion_viewer.sh --seed 0 --max_seed_tries 3 --hold 0` passed: seed 0/1 were skipped as unstable, seed 2 loaded, and one `play_once()` finished.

## 2026-06-04 (O.0 Piper Motion Stage Logs And EE Target Axes)

- Rechecked issues:
  - The small white cube in the viewer is only the origin of each axis marker; it is not the Piper base or initial pose.
  - The previous viewer only showed bottle-center and left/right place-target axes; it did not show the staged gripper motion targets.
  - The calibrated Piper/Pika home FK is approximately left `(-0.30,-0.48,0.77)` and right `(0.56,-0.50,0.80)`, which does not match the original ALOHA/AgileX bottle range `y=[0.03,0.23]`.
- Changes:
  - `envs/pick_diverse_bottles_piper_motion.py` now overrides `load_actors()` and uses an O.0 motion-baseline bottle range closer to the current Piper/Pika FK workspace: `left=x[-0.30,-0.18],y[-0.20,-0.10]` and `right=x[0.30,0.46],y[-0.20,-0.10]`.
  - Added `[piper-motion][stage]` logs for `play_once/pregrasp/grasp_lower/close_gripper/lift/move_out/open_gripper`.
  - Added `get_debug_axis_poses()`, computing the current left/right `link6` EE poses and the left/right EE target axes for `pregrasp/grasp_lower/lift/move_out` using the current URDF/SAPIEN FK.
  - `view_pick_diverse_bottles_piper_scene.py` now calls the task's `get_debug_axis_poses()` and adds those EE target axes to the viewer.
- Validation:
  - `python -m py_compile envs/pick_diverse_bottles_piper_motion.py view_pick_diverse_bottles_piper_scene.py view_pick_diverse_bottles_piper_motion.py` passed.
  - `DISPLAY=:1.0 timeout 120s bash run_pick_diverse_bottles_piper_motion_viewer.sh --seed 0 --max_seed_tries 10 --hold 0` passed: seed 0/1 were skipped as unstable, seed 2 loaded, all `[target-axis]` and `[stage]` logs printed, and `play_once()` finished.
- Remaining fact:
  - The current staged EE target FK is still around `y=-0.40~-0.47`, so O.0 motion remains a joint-space visualization baseline. A true bottle-aligned grasp requires a later Piper EE target redesign or a Piper/Pika-compatible IK/grasping strategy.

## 2026-06-08 (Mode N-1 Foundation Target Pose Order And C-Gripper Preview)

- Rechecked issue:
  - Mode N should compose a target from the FoundationPose object world position and the human gripper rotation matrix, but the wrapper wrote `pose_world_wxyz` in an order that did not match the planner's actual parser.
  - The planner consumes poses as `[x, y, z, qw, qx, qy, qz]`; the historical field name still says `pose_world_wxyz`.
  - Before this fix, `plan_keyframes_foundation_pose.py` / `plan_keyframes_human_replay.py` wrote `[qw, qx, qy, qz, x, y, z]`, causing the planner to treat quaternion values as position.
- Changes:
  - Fixed Mode N and Mode M wrapper output for `raw_pose_world_wxyz` / `pose_world_wxyz` to `[x, y, z, qw, qx, qy, qz]`.
  - Added 2D C-shaped gripper projection and RGB local axes to `rank_previews/*.png`; left arm is blue, right arm is orange, X=red, Y=green, Z=blue, and blue local `+Z` is the Mode N forward/retreat axis.
  - Rank preview selected marking now uses `(frame, arm)`, and preview frames include each arm's debug frames.
  - Updated the `# 0608` Mode N command in `COMMAND_LIBRARY.zh.md` to write into `N-1_foundation_pose_viewer`.
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_keyframes_foundation_pose.py code_painting/plan_keyframes_human_replay.py` passed.
  - `bash -n code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh` passed.
  - The `pick_diverse_bottles id2` smoke run generated `N-1_foundation_pose_viewer/pick_diverse_bottles/foundation_input_2/plan_summary_foundation_pose.json` plus `rank_previews/keyframe_000036_rank_1.png` and `rank_previews/keyframe_000053_rank_1.png`.

## 2026-06-09 (Mode N-3 Rank Preview Projection Status And Viewer Debug Overlay)

- Rechecked issue:
  - The user's `N-2_foundation_pose_viewer/pick_diverse_bottles/foundation_input_0/rank_previews` images did not show a visible C-shaped gripper.
  - Rechecking `plan_summary_foundation_pose.json` and the smoke logs confirmed that the new code was running; in this sample the composed targets project behind the head camera. The right-arm keyframe-38 target is approximately `(0.644,-0.343,-0.297)`, about 1.04 m below the right bottle.
- Changes:
  - `rank_previews/*.png` now prints target xyz, object-to-target offset, and `proj=inside/offscreen/behind_camera`.
  - The 2D C-gripper projection now clips lines at image bounds, draws edge markers for offscreen targets, and labels targets behind the camera at the bottom of the image.
  - `run_plan_keyframes_foundation_pose_piper_d435.sh` now supports `--debug_viewer_overlay`, which sets `pure_scene_output=0` and shows target axes plus top-1 C-gripper actors in the viewer/videos.
  - Mode N now passes `--debug_candidate_top_k 1` to the planner by default for viewer debugging.
  - Updated the Mode N section in `COMMAND_LIBRARY.zh.md` to `N-3_foundation_pose_viewer` and added a single `pick_diverse_bottles id1 --viewer --debug_viewer_overlay` demo command.
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_keyframes_foundation_pose.py` passed.
  - `bash -n code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh` passed.
  - The `pick_diverse_bottles id0` smoke run generated `N-3_foundation_pose_viewer_smoke/.../rank_previews/keyframe_000038_rank_1.png`, which explicitly shows `L: proj=behind_camera` and `R: proj=behind_camera`.

## 2026-06-09 (Mode N-4 Foundation Replay Pose Order Fix And Camera Visualization)

- Rechecked issue:
  - Further inspection of `foundation_replay_d435/foundation_input_0/multi_object_world_poses.npz` confirmed that `left_bottle__pose_world_wxyz[:3]` matches `left_bottle__pose_world_matrix[:3,3]`; both are the true object position.
  - Old Mode N read `pose_world_wxyz[4:7]` as object position, which was actually the last three quaternion components. For frame 38, the left bottle true position is approximately `(-0.0395, 0.0943, 0.7323)`, while the old read was `(0.3273, 0.6319, 0.6086)`.
  - Therefore the N-2/N-3 C-gripper target left the view because Mode N read the Foundation pose order incorrectly. The Foundation object was visible and the head-camera image itself was not broken.
- Changes:
  - `plan_keyframes_foundation_pose.py` now reads object position from `pose_world_wxyz[:3]`.
  - `plan_keyframes_foundation_pose.py` and `plan_keyframes_human_replay.py` now parse head-camera pose as `pos=[:3]`, `quat=[3:7]`.
  - The 2D rank-preview projection now converts SAPIEN camera frame to OpenCV frame as `[x, y, z]_cv = [x, -y, -z]_sapien_camera`.
  - `--debug_viewer_overlay` now also enables `--debug_visualize_cameras 1` and `--viewer_show_camera_frustums 1`, so the viewer shows head/third camera axes and frustums.
  - Updated the Mode N commands in `COMMAND_LIBRARY.zh.md` to `N-4_foundation_pose_order_fix` and `N-4_foundation_pose_debug_viewer`.
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_keyframes_foundation_pose.py code_painting/plan_keyframes_human_replay.py` passed.
  - `bash -n code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh` passed.
  - The `pick_diverse_bottles id0` smoke output changed frame-38 targets to left `(-0.058, 0.071, 0.735)` and right `(0.253, 0.095, 0.747)`, about one retreat offset from the true bottle positions.
  - The new `rank_previews/keyframe_000038_rank_1.png` shows `L: proj=inside(...)` and `R: proj=inside(...)`; both the 2D C-gripper wireframe and the 3D C-gripper actor are near the bottles.

## 2026-06-09 (Mode N-5 Retreat Parameters And Interpolation Notes)

- Rechecked issue:
  - Current Mode N does not directly interpolate once from keyframe 0 to keyframe 1. It plans staged motions: pregrasp, grasp, and action. Each stage calls the URDF IK planner.
  - The default `cartesian_interp_ik` mode linearly interpolates TCP position, Slerps TCP orientation, and solves IK waypoint by waypoint.
  - If a waypoint's IK switches from the current wrist/elbow branch to another feasible branch, the execution video can show the end effector dipping or twisting before returning to the target. This is more consistent with an IK branch change than with the keyframe-1 target orientation being reversed.
- Changes:
  - Updated the Mode N block in `COMMAND_LIBRARY.zh.md` to N-5.
  - N-5 commands use `--foundation_pose_retreat_m 0.08 --approach_offset_m 0.07`, corresponding to an 8 cm grasp retreat, a 15 cm total pregrasp retreat, and a 7 cm pregrasp-to-grasp advance.
  - Synchronized `agent-read/COMMANDS/piper_anygrasp_keyframes.zh.md` / `.en.md` and the command changelog.
- Validation:
  - This change only updates documentation and command-parameter guidance; planner code was not changed.

## 2026-06-10 (Mode N-6 id1 Viewer Recheck And Issue Attribution)

- Rechecked issue:
  - The user ran `pick_diverse_bottles id=1 --viewer --debug_viewer_overlay --foundation_pose_retreat_m 0.10 --approach_offset_m 0.07` in `modeln-4`.
  - The output directory was `N-5_pregrasp15_grasp8_debug_viewer/.../foundation_input_1`, but the actual parameters were a 10 cm grasp retreat and a 17 cm total pregrasp retreat. Command docs now use an N-6 directory name to avoid that mismatch.
- Observations:
  - `plan_summary.json` shows pregrasp/grasp position errors around 1-2.4 cm, which is usable.
  - In the action stage, the left arm reached after the third replan with about 2.9 cm position error. The right arm did not reach after the third replan and ended about 38.9 cm away.
  - `pose_debug.jsonl` shows that init-to-pregrasp already has large wrist-joint accumulated motion, about 4.25/4.23 rad on the left/right final wrist joint, despite small net change. The right action stage also shows large accumulated joint motion and branch switching.
- Attribution:
  - Foundation object position and projection are no longer the primary issue.
  - Current Mode N allows `reach_rot_tol_deg=180` and `urdfik_max_rotation_threshold_rad=3.14`, with `candidate_keep_camera_up=0`; poses equivalent under a 180-degree roll about the approach axis are accepted, and wrist-camera-up / roll continuity is not enforced.
  - Current `cartesian_interp_ik` solves IK waypoint by waypoint, and IK can switch to an unseeded or different wrist/elbow branch mid-stage. This explains the visual dip/twist and large roll about the approach axis.
- Changes:
  - Updated only `COMMAND_LIBRARY.zh.md` and agent-read command docs. The recommended command is now N-6: `--foundation_pose_retreat_m 0.10 --approach_offset_m 0.07`, with output root `N-6_pregrasp17_grasp10`.
  - Planner code was not changed in this step; roll/up constraints, >180-degree erroneous-rotation rejection, and IK branch continuity remain unresolved.

## 2026-06-10 (Piper IK Sequential Trajectory V2 And V1-V4 Collection Fix)

- Root causes: all four move segments were planned from home; lift used the wrong x/y reference; execution advanced before PD convergence; place treated bottle targets as gripper targets; and Phase 1/2 could mix unversioned legacy pickles.
- Changes: move segments are planned and executed sequentially from the preceding IK endpoint; lift preserves grasp x/y and orientation and only increases z; place is corrected from the measured bottle-to-EE offset after close; every move holds its endpoint for settling.
- Trajectory: added `piper_ik_cartesian` schema v2, IK version, action names, targets, and shape/finite/nonempty validation. Legacy formats are rejected.
- V3: MotionGen unavailability, exceptions, or planning failure falls back to cubic interpolation to the same valid IK endpoint.
- Viewer/collection: the viewer requires physical success by default; resume recognizes `_succ/_fail.hdf5`; new `demo_piper_ik_seq_v1..v4` configs isolate legacy data.
- Cameras: `third_camera` is now a right-side view and `opposite_top_camera` adds an opposite overhead view. Every RGB camera produces an MP4.
- Validation: V1-V4 viewers physically succeeded at seed 3. V1 completed five episodes with seeds 3/6/10/14/18. V2/V3/V4 each completed a one-episode smoke collection with `_succ.hdf5`, v2 pickle, instructions, and six videos. Rerunning V1 confirmed existing `_succ.hdf5` files are skipped.
- Final checks: `py_compile`, all four YAML configs, the prompt JSON, `bash -n collect_data.sh`, `git diff --check`, and the legacy-pickle rejection test passed.
- Cleanup: removed reproducible V2/V3/V4 smoke configs and temporary outputs after validation; retained the full ignored V1 validation dataset.

## 2026-06-11 (O.1 Foundation OBJ Grasp Fix)

- O.1 now loads the cola/bottle OBJ paths from the NPZ instead of substituting `001_bottle`.
- Corrected the actor-origin versus OBJ-geometric-center error and compute table clearance from the lowest rotated mesh point.
- Bounds-derived cylinder proxy is the default collision; exact convex remains optional.
- Added distance-gated grasp assist for the narrow source meshes and release drives on open or environment close.
- Trajectories bind Foundation input, frame, collision mode, and mesh geometry. Semantic instruction values and the missing task prompt were added.
- Replay skips planner initialization, fixing V3 Phase-2 multi-camera rendering slowdown.
- Validation: V1-V4 viewers reported `physical_success=True`; full V1-V4 collection produced validated replay, `episode0_succ.hdf5`, six videos, and instructions. V3 fallback worked after MotionGen failure.
- Cleanup: removed this run's reproducible O.1 HDF5 files, videos, caches, generated config, and temporary logs after structural validation.

## 2026-06-11 (O.1 No-Teleport Grasp Gate And Keyframe Modes)

- Root cause: old O.1 called `actor.set_pose(settled_pose)` before close, so a bottle tipped during pregrasp/grasp teleported back into the gripper. Full bottle-body cylinder collision also intersected the open-gripper approach path.
- Changed: removed object pose reset; defaulted to a base-only `support_proxy`; after close, validate displacement, rotation, link6 distance, finger-segment projection, and radial distance before attaching a drive at the current object pose.
- O.1.1 reads the first two episode keyframes from `hand_keyframes_all.json` and uses the first frame for Foundation OBJ setup.
- O.1.2 performs pregrasp/grasp/close from the first frame, then loads left/right EE xyz at the second frame from `world_targets_and_status.npz` and replaces lift/place with one action retaining the grasp orientation.
- Trajectory context now binds mode, episode ID, keyframes, annotation/action source, pregrasp distance, and grasp-gate parameters to prevent cross-mode replay.
- Validation: `py_compile` and `bash -n` passed. V1 O.1/O.1.1/O.1.2 viewers and full two-phase collections passed. Full O.1.2 collection also passed on V2/V3/V4. Every collection produced a v2 pickle, validated replay, `episode0_succ.hdf5`, instructions, and six MP4 files.
- Environment boundary: the current non-interactive shell's X11 socket reports `Renderer does not support display` to SAPIEN, so V2-V4 GUI window creation was not repeated there; full offscreen collection covered the same planning and replay logic.
- Cleanup: after inspecting artifact structure, removed six reproducible validation output directories, temporary YAML files, and `/tmp` logs. Existing collected datasets were not modified.

## 2026-06-11 (Mode N-7 Action Orientation And Dual-Replan Freezing)

- Rechecked the `modeln-4` N-6/N-7 outputs and confirmed that Foundation position/projection is no longer the main issue. On `pick_diverse_bottles id=1`, N-6 reaches pregrasp/grasp at about 1-2.4 cm but misses the right action target by about 38.9 cm after the third replan.
- Added `--foundation_pose_action_orientation_source grasp`: the second/action keyframe still uses the second Foundation object xyz, but orientation and retreat direction stay at the first-keyframe grasp orientation, mirroring O.1.2's "action keeps grasp quaternion" behavior.
- Added `--dual_stage_freeze_reached_arms_on_replan`: in dual-stage replanning, once one arm reaches the target, later attempts hold that arm fixed and only compensate unreached arms.
- Rechecked R1 `candidate_keep_camera_up`: it treats local X as forward, while Mode N currently uses local +Z as the approach/retreat axis, so it cannot be copied directly. The new Mode-N-specific `--foundation_pose_keep_top_axis_up` resolves a 180-degree roll about local +Z, but `top_axis=y` worsened id=1, so it is not recommended yet.
- Validation: `py_compile` and `bash -n` passed. `N-7_action_grasp_rot_freeze_smoke` reached action on `pick_diverse_bottles id=1` with left/right errors about 2.78 cm / 2.07 cm. The remaining unresolved issue is that IK still accepts about-170-degree roll-equivalent poses, so orientation error is not yet a strict success signal.

## 2026-06-11 (Mode M Human Replay IK Continuity Fix)

- Root causes: Mode M did not forward seed settings, old IK selection favored low-pose-error unseeded branches, failed Cartesian prefixes could still execute, and the safety gate skipped keyframe 2 after a keyframe-1 grasp miss.
- Changes: fixed the CuRobo seed tensor to `[batch, num_seeds, dof]`; added explicit perturbed seeds and joint-continuity selection; added cubic joint smoothstep; retained the grasp quaternion for action by default; froze reached arms; and returned a nonzero status on execution failure.
- Orientation finding: there is no strict roll constraint about local +Z. Piper's fixed `global_trans_matrix` and target/report frame conventions remain inconsistent, so approximately 178-180 degree rotation errors are not yet a strict success metric. `apply_global_trans_to_ik=1` performed worse.
- Validation: `pick_diverse_bottles` IDs 1 and 2 completed successfully. ID 0 reached pregrasp/grasp, but action ended at approximately 4.39 cm / 6.41 cm left/right error, exceeding the 4 cm tolerance. The old issue was shared IK/execution behavior, not an ID-1-only annotation problem.

## 2026-06-11 (O.1.2 Bounded Retries And Calibrated Wrist Cameras)

- Tmux inspection confirmed that `gen2-10`, `genikv2-11`, `genikv3-12`, and `genikv4-13` had returned to their shells. Historical jobs ended through `Killed`/Ctrl-C rather than remaining active.
- Corrected batch amplification: V1 now runs one rather than ten episodes per ID; all four configs use `max_seed_tries: 3`; the generic collector returns nonzero at the bound instead of looping forever on deterministic failures.
- `collect_foundation_piper_ik.sh` now accepts an optional `run_tag`, creates isolated config/output names, and forces `episode_num: 1`.
- All four Foundation configs enable wrist cameras with distinct extrinsics from `calibration_bundle_piper_new_table_0515.json`, composed from planner gripper poses after optical/render axis conversion.
- Validation: a V1 O.1.2 two-phase collection produced HDF5, instructions, and eight MP4 files. Both wrist videos have 38 frames at 320x240 and moved about 0.37 m / 0.46 m. V4 ID 9 failed three seeds because right-grasp rotation was about 25.6 degrees against a 15-degree limit, then exited at the configured bound.

## 2026-06-11 (Wrist Frame Adapter And Foundation Video Index)

- Reviewed the 0515/new-table wrist hand-eye files against historical calibrations. Translation and orientation trends are stable; the right wrist's roughly 45-degree roll is a persistent physical mount difference.
- Fixed the real TCP versus RoboTwin Pika CAD parent-frame mismatch. Foundation configs now use `urdf_end_link`, while the bundle supplies a translation-only `piper_pika_agx` shell-clearance adapter that preserves calibrated optical axes, lateral signs, and roll.
- Added viewer option `--wrist_preview 1` for a live left/right wrist RGB mosaic.
- Added `script/index_foundation_piper_ik_videos.py` to map each per-ID directory's `episode0_*` videos to aggregate `episode<ID>_*` names. It uses symlinks and rejects existing episodes by default; replacement requires explicit `--replace-episode`.
- Validation: a full V1/O.1.2 two-phase run for `foundation_input_0` succeeded and produced 38-frame left/right wrist videos, HDF5, instructions, and a validated trajectory. Sampled frames show the corresponding bottle and gripper in both views. A `DISPLAY=:1.0` viewer run with `--wrist_preview 1` completed with `physical_success=True`.

## 2026-06-15 (O.1.2.1 Wrist Shell And Roll Correction)

- The `gen1/gen2/genikv2/genikv3/genikv4` tmux panes had all returned to their shells. Earlier batch snippets omitted `RUN_TAG`, produced `o12_v4__failures.log`, and mixed several camera configurations in untagged directories.
- The official Pika gripper URDF has no camera. Official Piper+Pika uses `rpy="0 -1.57 0"` on `joint6_to_gripper_base`, while the current AGX-derived merge uses an identity connection, so visual alignment is not sufficient evidence of frame alignment.
- `envs/camera/camera.py` now supports per-side `forward_offset_m` and `image_roll_deg`. Base YAML values are left `0.125 m/-15 deg` and right `0.11 m/-60 deg`, without changing the original 0515 JSON or IK.
- The viewer exposes four temporary override arguments for live tuning without editing YAML.
- Validation: Python compilation and all four YAML parses passed. V4 ID 0 O.1.2 under tag `wrist_o121_verified_0615_smoke` completed both phases and produced 38-frame left/right wrist MP4s, HDF5, instructions, and six other videos. Sampled frames show no shell occlusion and an upright right view; the final viewer with `--wrist_preview 1` completed with `physical_success=True`.
- Cleanup: removed three reproducible intermediate tuning smoke runs and retained only `wrist_o121_verified_0615_smoke` for result inspection. All data directories are ignored by `data/*`.

## 2026-06-15 (O.1.2.1 Viewer Wrist Debug Recording)

- Added a viewer wrist debug recorder that writes raw left/right and labeled mosaic MP4s from the same render frames, plus JSON camera, tuning, task, and Foundation context.
- `--wrist_debug_tag` refuses to overwrite a non-empty directory. Recording is currently limited to one episode per command.
- Clarified the frame conclusion: transform composition is valid; the missing segment is `link6_T_real_tcp`, not the calibrated 0515 `real_tcp_T_camera`. Tuning is an empirical estimate of missing mechanical extrinsics.
- Validation: V1/O.1.2 ID 0 viewer reported `physical_success=True`. The retained headless result has 511 frames per MP4 at 30 FPS; left/right are 320x240, the mosaic is 640x240, and JSON includes task/config/ID/mode/seed. The earlier recording without context was removed.

## 2026-06-15 (Wrist Debug H.264 And Headless Formal Overrides)

- Root cause: the old recorder used `mp4v`. Files were complete, but VS Code/Chromium codec support is unreliable. Recording now uses FFmpeg `libx264`, `yuv420p`, and `faststart`, matching the formal H.264 output path.
- Converted six existing `mp4v` files under `data/wrist_camera_debug` in place to `h264/avc1/yuv420p` without changing filenames.
- `collect_foundation_piper_ik.sh` now accepts four all-or-none environment overrides and writes them into generated YAML. Missing or invalid values fail immediately.
- Validation: the new three debug videos each contain 511 H.264 frames with front-loaded `moov`. A full headless V1 ID 0 O.1.2 formal collection succeeded; both wrist videos contain 38 H.264 frames and generated YAML contains left `0.125/-15`, right `0.11/-60`.
- Cleanup: removed the duplicate `o121_h264_smoke_0615` output while retaining the converted original debug directories and the formal collection validation result.

## 2026-06-15 (Viewer Wrist/Head Camera Frustums)

- `view_pick_diverse_bottles_piper_ik_motion.py` adds `--show_camera_frustums`, using SAPIEN `show_camera_linesets` and verifying that both wrist cameras and the head camera are loaded.
- Fixed the unused `--hold 1` option; a single episode now retains the final window until it is closed or interrupted.
- `gen1` failures came from two reused non-empty debug tags, followed by a successful new-tag viewer run, then `unset DISPLAY` and an ineffective `set DISPLAY`. The correct recovery is `export DISPLAY=:1.0`.
- Validation: `xdpyinfo` connected to `:1.0`; the V1/O.1.2 ID 0 viewer listed `left_camera/right_camera/head_camera` and finished with `physical_success=True`; `hold=1` retained the viewer and exited via `Ctrl-C` without a traceback.

## 2026-06-16 (Live Piper IK SAPIEN Motion)

- Root cause: custom move, endpoint-settle, and gripper-settle loops in `envs/pick_diverse_bottles_piper_ik.py` called only `_update_render()`. Wrist images updated live, but `viewer.render()` was omitted, so SAPIEN showed only the final hold state.
- Added `_render_execution_step()` to always refresh observation cameras and draw SAPIEN according to `render_freq`; viewer runs now validate and report the number of live motion frames.
- Mode 1 validation: a 1920x1080 `SAPIEN` window existed during execution, rendered 510 live frames, and finished with `physical_success=True`.
- Mode 2 validation: `SAPIEN` and the 640x299 `RoboTwin wrist cameras` window coexisted during execution, rendered 510 live SAPIEN frames, and finished with `physical_success=True`.

## 2026-06-16 (Wrist Camera Forward-Axis Diagnosis)

- Added `script/diagnose_piper_wrist_camera_axes.py`, reusing the `legacy_r1` axis conversion and `piper_pika_agx` adapter from `envs/camera/camera.py` to compute wrist camera forward axes offline.
- Current results: left forward `[0.999974, -0.003184, 0.006511]`, right forward `[0.999622, -0.014664, 0.023248]`, both close to Pika physical `+X` in the gripper frame.
- Finger-opening `Y` plane error is left `-0.182 deg`, right `-0.840 deg`; zeroing only the `Y` component would require tiny gripper-`+Z` yaw corrections of left `+0.182 deg`, right `+0.840 deg`.
- Recorded conclusion: legacy debug blue `+Z` is an IK/debug target-pose convention, not the Pika physical forward axis. Using it directly would incorrectly suggest an about `-89 deg` large rotation.

## 2026-06-16 (Wrist Parent-Yaw Parameters)

- `envs/camera/camera.py` adds `parent_yaw_deg` tuning, rotating wrist camera orientation about gripper/link6 parent `+Z`; this is distinct from optical-axis `image_roll_deg`.
- `view_pick_diverse_bottles_piper_ik_motion.py` adds `--wrist_left_yaw_deg` and `--wrist_right_yaw_deg` for temporary viewer overrides.
- `script/diagnose_piper_wrist_camera_axes.py` now prints ready-to-copy yaw arguments; current values are left `0.182 deg`, right `0.840 deg`.
- Validation: the V1/O.1.2 live SAPIEN plus wrist viewer run with yaw completed 510 live frames, `physical_success=True`, and logs confirmed `parent_yaw_deg` in tuning.
## 2026-06-16 (Wrist Distance And Downward-View Review)

- Current wrist forward offsets are left `0.125m` and right `0.11m`; moving 2cm further along the optical axis uses left `0.145m`, right `0.13m`.
- With nominal tip `[0.12,0,0]`, current camera-to-tip Euclidean distance is about `11.9cm`, and the distance along camera forward is about `6.8cm`; after +2cm offset it becomes about `4.8cm` along forward.
- Raw 0515 and yaw-corrected forward axes are both close to Pika physical `+X`, not visibly pitched down toward the fingers. A direct nominal-tip view would require about `54deg` gripper-`Y` pitch, so practical tuning should try smaller downward pitch steps.
- The right camera center is `Y=-2.74cm`, while the left is `Y=+2.07cm`; the right side is slightly more laterally offset, which yaw/roll cannot fully replace.



## 2026-06-16 (Foundation Gripper Standoff +2cm)

- Changed the default `foundation_grasp_standoff` in `demo_piper_ik_foundation_v1-v4.yml` from `0.085m` to `0.105m`.
- Added `--foundation_grasp_standoff_m` to `view_pick_diverse_bottles_piper_ik_motion.py` for live-viewer grasp-depth overrides.
- Added `FOUNDATION_GRASP_STANDOFF_M` to `collect_foundation_piper_ik.sh`; generated configs now receive the override.
- Corrected the semantics: this is the gripper-base/EE grasp-target standoff from the object center. Wrist-camera `forward_offset_m` only changes camera extrinsics and does not change grasp depth.

Validation: `py_compile` passed; `bash -n collect_foundation_piper_ik.sh` passed; `view_pick_diverse_bottles_piper_ik_motion.py --help` shows `--foundation_grasp_standoff_m`; a minimal V1/O.1.2 headless run with `--foundation_grasp_standoff_m 0.105` completed, logged `grasp_standoff=0.105m`, and reported `physical_success=True`.


## 2026-06-16 (Wrist Raw-Calibration Angle Table And Pitch/Lateral Tuning)

- Rechecked the 0515 wrist calibration: raw/adapter forward axes are already nearly in the gripper `X-Z` plane, with left `plane_err_y=-0.182deg` and right `-0.840deg`.
- The raw calibration is not visibly downward-looking: forward is only left `0.415deg` and right `1.575deg` from gripper `+X`; after current viewer yaw it is left `0.373deg`, right `1.332deg`.
- Added viewer tuning options: `--wrist_left_pitch_deg`, `--wrist_right_pitch_deg`, `--wrist_left_lateral_offset_m`, and `--wrist_right_lateral_offset_m`.
- First recommended trial: left/right pitch `15deg`, right lateral `+0.0067m`.

Validation: `py_compile envs/camera/camera.py view_pick_diverse_bottles_piper_ik_motion.py` passed; viewer `--help` shows the new pitch/lateral options; a minimal V1/O.1.2 headless run with `--wrist_left_pitch_deg 15 --wrist_right_pitch_deg 15 --wrist_right_lateral_offset_m 0.0067` completed, logged `parent_pitch_deg` and `parent_lateral_offset_m` in camera tuning, and reported `physical_success=True`.


## 2026-06-16 (Wrist Gripper-Centerline Correction Note)

- Clarified mirror correction versus centerline correction: right `+0.0067m` only moves right `Y=-2.74cm` toward the mirror of left `-2.07cm`; it does not place the camera at `Y=0`.
- For a gripper-centerline camera, use lateral offsets left `-0.0207m` and right `+0.0274m`.
- Convention: gripper `+X` is wrist-to-tip forward, and `+Y` is finger-opening/lateral direction.
- Validation: a minimal V1/O.1.2 headless run with centerline correction and left/right pitch `15deg` succeeded, logged `parent_lateral_offset_m` in camera tuning, and reported `physical_success=True`.


## 2026-06-16 (O.1.2 Verified Grasp/Wrist V2 And Real-Grasp Debugging)

- Recorded the user-validated viewer parameters: `foundation_grasp_standoff_m=0.14`, wrist forward left/right `0.145/0.13`, pitch `15deg`, and lateral left/right `-0.0207/0.0274`.
- Added Foundation debug overrides to `view_pick_diverse_bottles_piper_ik_motion.py`: collision mode, collision padding, grasp assist, require contact, radial tolerance, and assist max distance.
- When O.1.2 grasp-assist is disabled, the task now still runs grasp-state validation and prints `contacts/projection/radial`, but does not create an object-gripper drive.

Validation: `py_compile` passed; viewer `--help` shows the new Foundation debug options; verified v2 headless completes with `physical_success=True` when using `radial_tolerance=0.08` and `assist_max_distance=0.16`. The default `0.065/0.14` gate fails around left `radial=0.071m` / `ee_distance=0.143m`, so the documented command now includes explicit gate thresholds. The pure-physics tier with `--foundation_grasp_assist 0 --foundation_collision_mode cylinder_proxy` reaches validation and prints contacts/projection/radial; seed 0 currently fails, which indicates pure contact is not yet carrying the object.


## 2026-06-16 (Verified Collection Wrapper And O.2 pnp_tray)

- Added `collect_foundation_piper_ik_verified.sh`, which writes a verified-v2 task config and calls `collect_data.sh`; it supports `pick_diverse_bottles` and `pnp_tray`, plus `DRY_RUN=1`.
- For `pick_diverse_bottles`, collection uses the stable tier A settings: `support_proxy + grasp_assist=true + require_contact=false + standoff=0.14 + radial=0.08 + assist_max_distance=0.16`, and writes the current head/wrist camera parameters.
- Added `envs/pnp_tray_piper_ik_foundation.py`, reusing the Foundation IK base class with left `left_dark_red_cup`, right `right_bottle`, and an open-gripper stage after the action.
- Refactored the Foundation base class to expose object keys, actor IDs, annotation path, hand-target pattern, and `foundation_open_after_action`; O.1 keeps open-after-action disabled by default.
- Added `task_config/demo_pnp_tray_piper_ik_foundation_v1-v4.yml` and `description/task_instruction/pnp_tray_piper_ik_foundation.json`.
- AB/C conclusion: tier A is the stable collection mode; tier B needs full side-body collision to make contact gating meaningful; tier C disables assist and exposes current pregrasp/grasp object-collision issues.

Validation: `py_compile` passed; `DRY_RUN=1` generated configs for both pick_diverse_bottles and pnp_tray; pick_diverse V1/O.1.2 headless succeeded with `standoff=0.14` and relaxed gates; pnp_tray V1/ID0/O.2 headless succeeded with `standoff=0.105`, including `open_after_action=True` and `open_gripper` in the log.


## 2026-06-17 (O.2 pnp_tray Action Target Fix And Pregrasp Avoidance Trial)

- Root cause: the old O.2 reused the O.1.2 second-keyframe EE target. On ID0, that EE target is around `Y=0.266`, farther forward than the Foundation second-keyframe object centers around `Y=0.18`, so the grippers moved too far forward after close.
- Fix: `pnp_tray_piper_ik_foundation` now defaults to `foundation_action_target_source=object_keyframe`; the action gripper target is computed from the second-keyframe OBJ center plus the current grasp-relative offset.
- Added viewer options `--foundation_action_target_source` and `--foundation_pregrasp_clearance_m`; the wrapper supports `FOUNDATION_PREGRASP_CLEARANCE_M` to generate an isolated avoidance-trial config.
- Optional avoidance: `foundation_pregrasp_clearance=0.06m` inserts a lifted waypoint before pregrasp. The default remains `0`, so existing no-avoidance commands are unchanged.

Validation: O.2 V1/ID0 headless with default object-keyframe succeeded. The action target changed from the old EE `Y≈0.266` to gripper `Y≈0.075/0.069`, with object error about `4.2cm/3.3cm`; `foundation_pregrasp_clearance=0.06` succeeded; `0.10` failed because the left cup rotated about `16.3deg`, exceeding the gate.
