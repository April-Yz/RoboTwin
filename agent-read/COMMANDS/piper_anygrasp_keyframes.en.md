# Piper AnyGrasp Two-Keyframe Planning

## Purpose

Run dual-arm Piper execution from the two selected AnyGrasp rank-preview keyframe candidates. The recommended entrypoint is:

```text
code_painting/plan_anygrasp_keyframes_piper.py
```

## Applicable Version

PiperPika AGX dual single-arm setup, recommended config:

```text
robot_config_PiperPika_agx_dual_table_0515.json
```

## Key Implementation

- The entrypoint reuses `PiperDualReplayRenderer`, preserving separate left/right arm base poses.
- URDFIK uses `HandRetargetPiperDualURDFIKRenderer`.
- IK converts world targets with `world_pose_to_base_pose_for_arm(...)` per arm, avoiding the R1 single-base conversion path.
- `cartesian_interp_ik` is recommended because it creates intermediate TCP waypoints and solves IK per waypoint.
- Piper AnyGrasp execution tries pregrasp, grasp, and action stages. If frame 38 grasp is not reached, the script still proceeds to close/action, so frame 78 usually starts from an already wrong robot state. Debug position reachability first with a loose rotation tolerance, then tune gripper orientation.
- `--dual_stage_require_all_plans 1` makes each dual-arm stage execute only when both left/right plans succeed, preventing one arm from moving alone when the other arm's IK fails.
- `--require_keyframe1_reached_before_action 1` skips the second-keyframe action stage when the first-keyframe grasp did not reach tolerance, preventing action from starting before keyframe 1 is actually reached.
- When using `--reuse_preview_frame_mode annotated_json_keyframes`, do not pass `--keyframes 38 78`; the script reads `frame_selection.annotated_keyframes[:2]` from the current id's preview summary.

## Command Format

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && CUDA_VISIBLE_DEVICES=<GPU> conda run -n RoboTwin_bw python code_painting/plan_anygrasp_keyframes_piper.py \
  --anygrasp_dir <ANYGRASP_ID_DIR> \
  --replay_dir <FOUNDATION_REPLAY_ID_DIR> \
  --hand_npz <HAND_DETECTIONS_ID_NPZ> \
  --output_dir <OUTPUT_DIR> \
  --reuse_preview_summary_json <PREVIEW_SUMMARY_JSON> \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --arm auto \
  --execute_both_arms 1 \
  --dual_stage_require_all_plans 1 \
  --require_keyframe1_reached_before_action 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps -1 \
  --urdfik_cartesian_interp_auto_step_m 0.02 \
  --urdfik_max_position_threshold_m 0.02 \
  --urdfik_max_rotation_threshold_rad 0.12 \
  --reach_rot_tol_deg 180 \
  --debug_visualize_ik_waypoints 1 \
  --vscode_compatible_video 1
```

## Important Parameters

- `--reuse_preview_summary_json`: must match the current id. Do not evaluate id0 execution with id2 rank previews.
- In the D435 pipeline, `--reuse_preview_summary_json` must point to `code_painting/anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json`, and `--replay_dir` must point to `foundation_replay_d435/foundation_input_<ID>`; do not fall back to the default wide-FOV `anygrasp_h2o_preview`.
- `--reuse_preview_frame_mode annotated_json_keyframes`: uses manually annotated keyframes.
- `--keyframes`: do not pass it in annotated-preview reuse mode; keyframes come from the preview summary.
- `frame_selection.effective_keyframes_by_arm`: when the annotation JSON has fewer than two global `keyframes`, preview uses `left_keyframes/right_keyframes` plus global `keyframes` to form two effective frames per arm; the planner uses those arm-specific frames when reusing the summary.
- `--dual_stage_require_all_plans 1`: in dual-sync stages, if either arm fails to plan, neither arm executes that stage.
- `--require_keyframe1_reached_before_action 1`: if the first-keyframe grasp misses, the second-keyframe action is skipped.
- `--urdfik_trajectory_mode cartesian_interp_ik`: uses TCP waypoints with per-waypoint IK.
- `--urdfik_cartesian_interp_auto_step_m 0.02`: automatic waypoint spacing. `0.01` is denser but more likely to fail if one intermediate waypoint misses the strict IK threshold.
- `--urdfik_max_position_threshold_m 0.02`: lets Piper URDFIK relax up to 2 cm during debugging, avoiding hard failure on otherwise usable 5 mm-2 cm intermediate errors.
- `--urdfik_max_rotation_threshold_rad 0.12`: lets IK relax internally to about 6.9 degrees; final reached/not-reached is still controlled by `--reach_rot_tol_deg`.
- `--reach_rot_tol_deg 180`: position-first debugging. After position reaches, tighten this to 60/40/20 while tuning candidate orientation.
- `--debug_visualize_ik_waypoints 1`: a 0/1 switch that visualizes intermediate waypoints for reachability debugging.
- `--head_only 0 --third_person_view 1 --vscode_compatible_video 1`: writes both head and third-person videos and transcodes them to H.264/yuv420p/faststart for direct VS Code preview.
- `--candidate_target_local_x_offset_m`: compensates the AnyGrasp target along gripper local X.
- `--approach_offset_m`: pregrasp retreat distance from the grasp pose along local X.

## D435 Version

Run `J1.1` or `J1.2` in `COMMAND_LIBRARY.zh.md` first to create D435 candidate summaries:

```text
code_painting/anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json
```

Downstream planning should use `L15.3`. The important differences from the default-wide L15/L15.2 commands are:

- `--replay_dir` uses `/home/zaijia001/ssd/data/piper/hand/<TASK>/foundation_replay_d435/foundation_input_<ID>`.
- `--reuse_preview_summary_json` uses `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json`.
- Camera arguments use `--image_width 640 --image_height 480 --fovy_deg 42.499880046655484` to match the D435 color replay.
- If the D435 summary is missing, rerun J1.1/J1.2 instead of reusing the default wide-FOV summary.

The current six-task D435 chain is:

```text
AnyGrasp grasps + foundation_replay_d435/head_anygrasp_frames + HaMeR hand_detections
  -> J1.1/J1.2 creates anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json
  -> L15.3 reuses the D435 summary and D435 replay for planning/execution
```

If the J0.1 broad scan shows many `MISS` lines, first check whether `seq 0 120` simply exceeds the real episode count for that task. The current input intersections on this machine are approximately: `pick_diverse_bottles=102`, `place_bread_basket=92`, `stack_cups=47`, `handover_bottle=47`, `pnp_bread=72`, `pnp_tray=51`. The reason `place_bread_basket/stack_cups/handover_bottle/pnp_bread` previously produced no D435 summaries was that the old wrapper only checked global `keyframes` and ignored `left_keyframes/right_keyframes`; the flow now supports per-arm effective keyframes.

The recommended six-task D435 planner entrypoint is:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 1
```

Do not paste the old long command containing `mapfile` directly into zsh; `mapfile` is a bash builtin. For a full run, remove `--max_per_task 1`. For six-task probing, add `--continue_on_error` so one failed id does not stop later tasks.

Run the first 5 ids per task with:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error
```

See L15.5 in `COMMAND_LIBRARY.zh.md` for a single `stack_cups id0` viewer debug command. It uses `unset CUDA_VISIBLE_DEVICES` for SAPIEN viewer and keeps the D435 summary/replay paths. If the viewer probe prints an empty `DISPLAY=` and reports `Renderer does not support display`, run it from a graphical terminal or a correctly forwarded X11/Wayland session.

L15.6 adds unified viewer/no-viewer and first-keyframe debug entrypoints:

```bash
# No viewer: first 5 summaries for each of the six tasks
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v1

# Viewer: run only from a graphical environment with DISPLAY set
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_viewer

# Viewer motion check: joint_interp, closer to the old R1/V7 visual behavior
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --trajectory_mode joint_interp --joint_interp_waypoints 40 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_viewer_joint_interp

# First-keyframe debug: execute only init -> pregrasp -> grasp; do not close the gripper or enter keyframe 2
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --debug_stop_after_keyframe1 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_keyframe1_debug
```

The script now passes `--require_keyframe1_reached_before_close 1` and `--require_keyframe1_reached_before_action 1` by default. If the first-keyframe grasp is not reached, the gripper will not close and keyframe 2 will not run.

Use `--debug_stop_after_keyframe1` to isolate first-keyframe reachability. For `stack_cups id0`, the failure already occurs during first-keyframe Cartesian waypoint IK: pregrasp fails at left waypoint 13/23 and right waypoint 28/48; grasp fails at left waypoint 16/28 and right waypoint 25/45. With `--dual_stage_require_all_plans 1`, any arm plan failure skips the synchronized dual-arm stage, so the video appears to have no waypoint execution. `execute_interp_steps/joint_command_scene_steps/settle_steps/joint_target_wait_steps` only affect execution after the plan is already `Success`; if the log already contains `[plan-fail]`, tune the trajectory mode, candidate target, or IK parameters instead.

The script now defaults to the old R1/V7-style execution cadence: `execute_interp_steps=24`, `joint_command_scene_steps=10`, `settle_steps=30`, and `joint_target_wait_steps=25`. Do not set both `execute_interp_steps` and `joint_command_scene_steps` to thousands; that makes the viewer appear stuck at a waypoint.

Terminal TCP/EE motion check:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 1 --continue_on_error --tasks pick_diverse_bottles --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_posemove_debug
```

`--print_pose_every 5` prints `[exec-pose]` records with left/right TCP and EE world positions. `--allow_partial_dual_stage` is diagnostic only: it allows the right arm to move when its plan succeeds, confirming that the execution chain changes pose. Keep strict dual synchronization for final data generation.

L15.7 records the current D435/Piper fixes:

- The six-task wrapper now defaults to `--reach_error_pose_source ee`. Piper AnyGrasp targets are fed to IK in the wrist/endlink convention; using `tcp` for reached checks leaves a fixed approximately 12 cm TCP/EE offset.
- `target_pose_for_error(..., ee)` now uses each Piper arm's own base transform, avoiding conversion of right-arm targets through the left-arm base.
- Partial diagnostic mode holds the arm whose plan failed while the other arm executes, preventing drift in the physics scene.
- The current keyframe flow is: reuse per-arm effective keyframes from the D435 summary -> first keyframe `pregrasp -> grasp` -> close gripper and run the second-keyframe action only after the first-keyframe grasp is reached.
- `pick_diverse_bottles id0` recheck: the right arm reaches the grasp with about `0.0057m` position error under `ee` checking, so the execution chain and waypoints do move; the overall episode still fails because the left-arm first-keyframe IK/target fails and strict dual sync blocks the stage.
- The wrapper supports `--visualize_targets` for viewer debugging. It displays the target axis and active-frame candidate grippers, and automatically disables `pure_scene_output`.
- The planner copies the reused summary's source best-candidate preview images into `<OUT>/source_preview_compare/` and writes `selected_candidate_mapping.json` with `candidate_idx/rank/raw_pose/target_pose/local_x_offset`.

See L15.7 in `COMMAND_LIBRARY.zh.md` for six separate first-5 commands. Strict-sync example:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pick_diverse_bottles --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/strict-ee
```

Partial diagnostic example:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --tasks pick_diverse_bottles --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/partial-ee
```

Viewer target-visualization example:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --visualize_targets --tasks pick_diverse_bottles --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-partial-ee
```

## Outputs

```text
<OUTPUT_DIR>/plan_summary.json
<OUTPUT_DIR>/rank_previews/
<OUTPUT_DIR>/debug_execution_metrics.jsonl
<OUTPUT_DIR>/pose_debug.jsonl
<OUTPUT_DIR>/head_cam_plan.mp4
<OUTPUT_DIR>/third_cam_plan.mp4
<OUTPUT_DIR>/debug_execution_preview.mp4
```

## Check Command

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('<OUTPUT_DIR>/plan_summary.json')
d = json.load(open(p))
print('execution_success:', d.get('execution_success'))
print('execution_failed:', d.get('execution_failed'))
print('failed_stage_records:', d.get('failed_stage_records'))
print('rank_preview_images:', d.get('rank_preview_images'))
PY
```

## Common Issues

- If `plan_solution_by_arm` is empty, IK did not find a solution; the arm did not plan to a different target.
- `--settle_steps` and `--joint_target_wait_steps` only wait for an existing joint target to settle; they cannot fix IK failure or an unreachable target pose.
- If the right-arm target appears shifted toward the left arm, check that execution is using the Piper dual renderer instead of the old R1 single-base wrapper.
- If the rank preview looks correct but execution fails, first verify that preview summary, AnyGrasp dir, replay dir, and hand npz all refer to the same id.
- If frame 38 only partially executes and frame 78 is worse, inspect `failed_stage_records`: frame 38 grasp missed, then close/action continued from the wrong robot state.

## L15.8 Preview And Planner Mapping Consistency

This pass found a real visualization/execution mismatch. `render_anygrasp_ranked_preview.py` applied `--candidate_target_local_x_offset_m -0.05` to the summary `translation_world`, but the preview image still drew the raw AnyGrasp `translation_cam/rotation_matrix`. The planner reuses the offset world target from the summary, so the planner/rank preview could appear slightly behind the source preview.

Preview drawing now uses the same remapped/post-rotated/local-X-offset camera-frame target pose. The summary additionally records `translation_cam`, `visual_translation_cam`, `translation_world`, `rotation_matrix`, and `visual_rotation_matrix`. `translation_cam` is the raw AnyGrasp candidate; `visual_translation_cam` is the camera-frame pose corresponding to both the drawn preview and the planner target.

The D435 planner must reuse `code_painting/anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json`, not the default wide-FOV `code_painting/anygrasp_h2o_preview/<TASK>/foundation_input_<ID>/summary.json`. For `pick_diverse_bottles id0 frame 38`, the D435 rank1 candidates are left `16` and right `11`; the default wide-FOV preview has different rank1 candidate ids, so the two preview roots should not be compared as if they were the same run.

Regenerate the six-task D435 previews:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && for TASK in pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray; do case "$TASK" in pick_diverse_bottles) LEFT_OBJ=left_bottle; RIGHT_OBJ=right_bottle ;; place_bread_basket) LEFT_OBJ=basket; RIGHT_OBJ=bread ;; stack_cups) LEFT_OBJ=left_light_pink_cup; RIGHT_OBJ=right_dark_red_cup ;; handover_bottle) LEFT_OBJ=right_bottle; RIGHT_OBJ=right_bottle ;; pnp_bread) LEFT_OBJ=left_bread; RIGHT_OBJ=right_bread ;; pnp_tray) LEFT_OBJ=left_dark_red_cup; RIGHT_OBJ=right_bottle ;; esac; ANN=/home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/${TASK}/hand_keyframes_all.json; ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output; [[ -d "$ANY_ROOT" ]] || ANY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/${TASK}_output_old_cam; REPLAY_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/foundation_replay_d435; HAND_ROOT=/home/zaijia001/ssd/data/piper/hand/${TASK}/harmer_output; OUT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/${TASK}; [[ -f "$ANN" ]] || { echo "[skip] task=${TASK} missing annotation $ANN"; continue; }; [[ -d "$ANY_ROOT" ]] || { echo "[skip] task=${TASK} missing ANY_ROOT=$ANY_ROOT"; continue; }; [[ -d "$REPLAY_ROOT" ]] || { echo "[skip] task=${TASK} missing REPLAY_ROOT=$REPLAY_ROOT"; continue; }; VIDEO_PREFIX=foundation_input CUDA_VISIBLE_DEVICES=2 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_batch.sh "$ANY_ROOT" "$REPLAY_ROOT" "$HAND_ROOT" "$OUT_ROOT" --hand_keyframes_json "$ANN" --left_target_object "$LEFT_OBJ" --right_target_object "$RIGHT_OBJ" --anygrasp_score_weight 0.25 --orientation_score_weight 0.75 --max_rotation_distance_deg 90 --candidate_target_local_x_offset_m -0.05 --draw_object_overlay 1 --draw_hand_reference 1 --debug_dump_object_distances 1 --top_k 20 --camera_cv_axis_mode legacy_r1; done
```

Planner/replay without viewer:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/strict-ee-offsetfix
```

Viewer target visualization:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --max_per_task 5 --continue_on_error --viewer --visualize_targets --trajectory_mode joint_interp --joint_interp_waypoints 40 --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-partial-ee-offsetfix
```

The planner writes `<OUT>/source_preview_compare/selected_candidate_mapping.json` with the source summary, raw pose, and offset target pose. A roughly 5 cm difference between `planner_raw_pose_world_wxyz` and `planner_target_pose_world_wxyz` is expected when `--candidate_target_local_x_offset_m -0.05` is enabled.

## L15.9 Copy-Safe Three-Step Run

`COMMAND_LIBRARY.zh.md` now ends with L15.9, which provides three copy-safe command blocks to run in order:

## Mode M/N Viewer CUDA Mask Fix

Mode M `run_plan_keyframes_human_replay_piper_d435.sh` and Mode N `run_plan_keyframes_foundation_pose_piper_d435.sh` viewer commands require both:

- A graphical terminal with a usable display, for example `DISPLAY=:1.0`.
- No `CUDA_VISIBLE_DEVICES=<compute_gpu>` mask while SAPIEN creates the interactive viewer, otherwise the display-driving GPU can be hidden from the renderer.

The failure was caused by the second condition. The bash wrappers already used `env -u CUDA_VISIBLE_DEVICES` in `--viewer` mode, but the Python middle layers `plan_keyframes_human_replay.py` and `plan_keyframes_foundation_pose.py` set `CUDA_VISIBLE_DEVICES=2` again before invoking the planner. The resulting log was:

```text
[viewer] creating interactive viewer ... CUDA_VISIBLE_DEVICES=2
[viewer-warning] failed to create interactive viewer ... Renderer does not support display.
```

The current fix removes `CUDA_VISIBLE_DEVICES` from the planner environment when `--enable_viewer 1` is active. Non-viewer mode still uses `--gpu` to set the compute GPU.

The minimal viewer probe remains:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES; [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; echo "DISPLAY=$DISPLAY CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}" && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/probe_sapien_viewer.py
```

Validation commands:

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_keyframes_foundation_pose.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_keyframes_human_replay.py

timeout 60s bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_foundation_pose_piper_d435.sh --gpu 2 --ids 0 --viewer --viewer_wait_at_end 0 --tasks pnp_tray --foundation_pose_retreat_m 0.03 --output_root /tmp/robotwin_viewer_env_probe
```

In a non-graphical shell the second command is still expected to fall back to offscreen because `DISPLAY=None`, but the important log should now show `CUDA_VISIBLE_DEVICES=None`, proving the Python middle layer no longer restores the CUDA mask.

1. Regenerate the six-task D435 AnyGrasp no-offset previews/summaries into the main `anygrasp_h2o_preview_d435` directory with `bash <<'BASH'`, avoiding zsh line-wrap argument loss.
2. Run the six-task D435 planner/replay without viewer, outputting to `anygrasp_plan_keyframes_piper_d435_v2/strict-ee-offsetfix`.
3. Run the six-task D435 planner/replay with viewer and `--visualize_targets` to display the gripper target.

During step 1, verify that the log prints this replay root:

```text
/home/zaijia001/ssd/data/piper/hand/<TASK>/foundation_replay_d435
```

If `replay_m_obj_pose_d_pour_blue_norobot` appears, the shell paste/line break still prevented the script from receiving the D435 replay argument.

L15.9 preview conventions:

- `orientation_rank.png` sorts only by hand-orientation similarity; the downstream planner currently defaults to `orientation rank1`.
- `fused_rank.png` sorts by `anygrasp_score * 0.25 + orientation_score * 0.75` and is only a reference view.
- `planner_selected_orientation_rank1.png` draws only the current planner default selection: `orientation rank1`.
- Candidate AnyGrasp grippers: left is blue-ish, right is orange-ish.
- Human-hand reference grippers: left is green, right is purple.

## L15.10 Offset -5cm Comparison Command

`COMMAND_LIBRARY.zh.md` now ends with L15.10, which provides an offset -5cm comparison command. It sets `--candidate_target_local_x_offset_m` to `-0.05` and writes to:

```text
code_painting/anygrasp_h2o_preview_d435_offset_minus_5cm_compare/<TASK>/foundation_input_<ID>
```

Use it to compare against the L15.9 main no-offset preview and confirm whether the 5 cm local-X compensation is the source of the visually shifted gripper. This comparison directory does not overwrite the main `anygrasp_h2o_preview_d435` outputs.

If the preview and planner final execution target need to be exactly identical, both sides must use the same `--candidate_target_local_x_offset_m`. The main directory is now documented as the user-requested no-offset source-candidate view.

## L15.11 Cartesian Partial Prefix Execution

`COMMAND_LIBRARY.zh.md` now ends with L15.11, which adds the `--execute_partial_cartesian_plan` diagnostic entrypoint. This flag only applies to `--trajectory_mode cartesian_interp_ik`: if an intermediate Cartesian waypoint IK fails but earlier waypoints solved successfully, the planner returns `status=Partial` and executes the solved prefix up to the last reachable waypoint.

Purpose:

- Avoid a completely static viewer when the whole Cartesian plan is marked `Fail`.
- Inspect how far the arm can move toward the target.
- Use `[plan-partial] failed_waypoint=N/M solved_prefix=K` and `[exec-pose]` logs to locate the unreachable waypoint.

Limits:

- `Partial` executes, but it is not counted as reached.
- If keyframe-1 is not reached, the close/action guards still prevent gripper close and keyframe-2 action.
- `joint_interp` has no Cartesian waypoint prefix, so this behavior does not apply there.

L15.11 also records the position-priority/orientation-priority note: `--reach_rot_tol_deg 180` makes the reached check position-first for debugging, but IK solving still constrains both position and orientation. A true position-priority IK fallback would require a staged IK strategy, such as full pose first, relaxed rotation threshold second, and position-only or orientation-sampling fallback third.

## L15.12 Piper Gripper Axis Fix And Position-First Diagnostic

`COMMAND_LIBRARY.zh.md` now ends with L15.12. It documents the Piper local-axis convention used by the D435 AnyGrasp planner:

```text
Preview gripper frame:
  local +X = approach / forward axis = rotation_matrix[:, 0]

Piper reported visible gripper frame:
  R_report = R_link6 @ global_trans_matrix @ delta_matrix

Current Piper config:
  global_trans_matrix = diag(1, -1, -1)
  delta_matrix = I
```

The URDFIK link6 target must therefore use:

```text
R_link6_target = R_preview_gripper @ inv(global_trans_matrix @ delta_matrix)
```

The previous Piper URDFIK path only removed `delta_matrix`; it did not remove `global_trans_matrix`, so the visible gripper in the viewer could be flipped about local +X relative to the preview. `render_hand_retarget_piper_dual_npz_urdfik.py` now applies this inverse global transform in `_target_tcp_world_to_ee_base()`. For Piper dual, `reach_error_pose_source=ee` also stays in the visible gripper-frame convention instead of converting the target into raw link6 space.

The six-task wrapper now accepts:

```text
--ik_max_position_threshold_m <meters>
--ik_max_rotation_threshold_rad <radians>
```

This is useful for position-first diagnostics. With the default `--ik_max_rotation_threshold_rad 0.12`, `pick_diverse_bottles id0` still failed at Cartesian waypoint 1 because complete-pose IK was rejected by orientation constraints. With `--ik_max_rotation_threshold_rad 3.14`, the smoke test produced continuous `[exec-pose]` output, proving that waypoint execution is active and that the static viewer behavior came from IK rejection rather than `settle_steps` or `joint_target_wait_steps`.

The partial Cartesian mode also tries a shorter sub-waypoint when the first Cartesian waypoint fails. If even that does not move, the current complete-pose constraint is still too strict for the first small step.

Viewer batching has also been adjusted: `--viewer` now defaults to `--viewer_wait_at_end 0`, so the wrapper automatically proceeds to the next id after one id finishes. Add `--viewer_wait_at_end 1` only when you want to pause at the end of each id for inspection. The L15.12 viewer axis-check command now explicitly includes `--ik_max_rotation_threshold_rad 3.14` to avoid the default `0.12rad` strict IK case where the viewer appears completely static.

## L15.13 Viewer Axis Check And Id-Range Commands

`COMMAND_LIBRARY.zh.md` now ends with L15.13, adding per-task viewer commands for ids 0-10. The wrapper now supports:

```text
--id_start <ID>
--id_end <ID>
--ids <ID...>
--piper_apply_global_trans_to_ik <0|1>
```

The default is `--piper_apply_global_trans_to_ik 0`, matching the direct Piper hand replay URDFIK convention. Value `1` is only for comparison diagnostics.

Viewer axis interpretation:

- The preview gripper wireframe local +X is `rotation_matrix[:, 0]`; visually it points from the palm/back bar toward the two fingertips/opening.
- The viewer target axis actor uses red = local +X, green = local +Y, blue = local +Z.
- To check preview-target versus viewer-target consistency, compare the preview finger direction with the viewer red axis first.
- Piper mesh/link Y/Z may look reversed relative to the target axes because `global_trans_matrix=diag(1,-1,-1)` flips Y/Z. Do not diagnose target mismatch from green/blue alone.

## L15.14 Viewer Target Gripper Uses Per-Arm Active Keyframes

The user observed that the L15.13 viewer target gripper looked like the later keyframe and that the first keyframe was not displayed correctly. The root cause was not necessarily the D435 preview candidate. The execution-preview state previously had only one global `active_frame`; in dual-arm mode this was taken from the left arm, and `record_frame()` plus candidate debug actors refreshed from that single frame. When left/right effective keyframes differed, or after the stage switched to action, the viewer could show a target from the wrong frame.

The fix:

- `DebugExecutionState` now has `active_frame_by_arm`.
- `pregrasp/grasp` use each arm's first keyframe, while `action` uses each arm's second keyframe.
- `record_frame()`, `update_candidate_debug_visuals()`, and the debug execution preview display candidates and selected grippers from `active_frame_by_arm`.
- `pose_debug.jsonl` and `execution_metrics.jsonl` now record `active_frame_by_arm`, so the viewer's current displayed frame can be checked with `jq`.

Dry-run check:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --dry_run --tasks pick_diverse_bottles --id_start 0 --id_end 10 --continue_on_error --viewer --visualize_targets --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10
```

After a real run, inspect:

```bash
jq -c '{stage,active_frame,active_frame_by_arm}' /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10/pick_diverse_bottles/foundation_input_0/pose_debug.jsonl | head -n 20
```

If `jq` is not available in the current shell, use:

```bash
head -n 20 /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/viewer-axischeck-id0-10/pick_diverse_bottles/foundation_input_0/pose_debug.jsonl | sed -E 's/.*"active_frame": ([^,]+), "active_frame_by_arm": \{([^}]*)\}, "stage": "([^"]+)".*/stage=\3 active_frame=\1 active_frame_by_arm={\2}/'
```

## L15.15 Stack Cups id0 No-Collision Target-Only Debug

New wrapper parameters:

- `--disable_execution_collisions`: passes `--enable_grasp_action_object_collision 0` to the planner, disabling grasp/action object collision and contact-stop close logic.
- `--target_axes_only`: turns on `--visualize_targets` and hides candidate gripper axes, selected-keyframe axes, and IK waypoint markers.
- `--debug_candidate_top_k`, `--debug_common_candidate_top_k`, `--debug_visualize_selected_keyframe_axes`, and `--debug_visualize_ik_waypoints`: fine-grained debug actor controls.

Why `stack_cups id0` can show several coordinate systems:

- The D435 summary uses per-arm keyframes: the left first keyframe is `139`, while the right first keyframe is `51`.
- The viewer may draw active execution target axes, selected-keyframe axes, candidate gripper axes, and IK waypoint markers at the same time.
- Use `--target_axes_only` when checking the actual execution target axes. The red axis is still target local +X.

Viewer debug command:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --tasks stack_cups --ids 0 --continue_on_error --viewer --target_axes_only --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_v2/stackcups-id0-nocollision-targetaxes
```

No-viewer keyframe-1-only debug:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh --tasks stack_cups --ids 0 --continue_on_error --target_axes_only --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 20 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --piper_apply_global_trans_to_ik 0 --debug_stop_after_keyframe1 --output_root /tmp/stack_cups_id0_no_collision_target_axes_only
```

The local no-viewer smoke test confirmed `enable_grasp_action_object_collision=0`, but `stack_cups id0` still did not reach the target. Pregrasp ended at roughly left `0.386m/147deg` and right `0.337m/82deg`; grasp still failed or returned partial. This indicates the primary issue is not object collision, but IK solution quality and post-command joint tracking/execution error.

## L15.16 Direct Piper Hand Replay Viewer Comparison

`COMMAND_LIBRARY.zh.md` now ends with L15.16, which replays the stored human gripper poses directly through the Piper URDFIK renderer. This does not use AnyGrasp candidate selection; it reads the stored gripper pose fields from `hand_detections_0.npz`.

Important parameters:

- `--debug_visualize_targets 1`: shows the target gripper axes in the viewer for each frame.
- `--debug_mode 1 --debug_post_execute 1`: prints the post-execution target versus actual TCP/EE error per frame.
- `--save_world_targets 1`: writes `world_targets_and_status.npz` for checking target poses and status fields.
- `--enable_viewer 1 --viewer_wait_at_end 1`: opens the viewer and keeps it open at the end.

Start with `stack_cups id0` to compare against the AnyGrasp planner:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_piper_dual_npz_urdfik_main.py --input_npz /home/zaijia001/ssd/data/piper/hand/stack_cups/harmer_output/hand_detections_0.npz --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/direct_replay_debug_piper_d435/stack_cups/id0_viewer_axes --image_width 640 --image_height 480 --fovy_deg 42.499880046655484 --fps 5 --frame_start 0 --frame_end 220 --max_frames 221 --arms both --piper_calibration_bundle /home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json --camera_cv_axis_mode legacy_r1 --require_stored_gripper_pose 1 --pose_source gripper --orientation_remap_label identity --stored_orientation_post_rot_xyz_deg 0 0 0 --target_local_forward_retreat_m 0.05 --target_world_offset_xyz 0 0.1 0.1 --execute_waypoint_scene_steps 5 --execute_settle_scene_steps 20 --urdfik_joint_interp_waypoints 10 --debug_mode 1 --debug_post_execute 1 --debug_frame_limit -1 --debug_visualize_targets 1 --debug_target_axis_length 0.10 --debug_visualize_cameras 0 --save_world_targets 1 --clean_output 0 --overlay_text_enable 1 --save_png_frames 0 --lighting_mode front_no_shadow --enable_viewer 1 --viewer_frame_delay 0.02 --viewer_wait_at_end 1
```

## L15.17 Direct Replay Versus AnyGrasp Axis Convention

Direct Piper hand replay and the AnyGrasp planner currently use different gripper local frames:

- Direct replay's stored gripper frame uses `local +Z` as the approach/forward axis. `--target_local_forward_retreat_m` retreats along the blue axis and prints `along_local_plus_z_blue_m`.
- The AnyGrasp preview wireframe uses `rotation_matrix[:, 0]` as the finger-depth direction from the palm/back bar to the fingertips, i.e. local +X/red.
- Therefore "blue is forward" in direct replay and "red follows the AnyGrasp wireframe finger direction" in AnyGrasp are not contradictory; they are different local-frame conventions.

To test mapping AnyGrasp local +X onto the direct-replay local +Z convention, add this to a single planner run:

```bash
--candidate_orientation_remap_label swap_red_blue
```

`COMMAND_LIBRARY.zh.md` L15.17 includes a full no-viewer `stack_cups id0` comparison command. Generate `pose_debug.jsonl` and videos first, then check whether the blue axis and execution error become closer to direct replay. If the direction is flipped, test `swap_red_blue_keep_green` or an explicit enumerated label.

## L15.18 Replay-Axis AnyGrasp Keyframe Planner

This section adds a separate entrypoint without replacing the older AnyGrasp commands:

```text
code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh
```

Why this exists: direct Piper hand replay uses the stored gripper frame where `local +Z` blue is the approach/forward axis. The raw AnyGrasp candidate wireframe uses `local +X` red as the palm-to-fingertip finger-depth axis. Therefore the old planner's `identity + local-X offset/pregrasp` path is not equivalent to the direct-replay blue-axis convention.

The new wrapper always enables:

```text
--candidate_orientation_remap_label swap_red_blue
--candidate_target_local_x_offset_m 0.0
--candidate_target_local_z_offset_m -0.05
--approach_axis local_z
--approach_offset_m 0.12
```

Meaning: remap the original AnyGrasp local +X onto the execution target local +Z, then apply both the 5 cm target compensation and the pregrasp retreat along local +Z. The old six-task wrapper still defaults to the local-X behavior.

The six-task no-viewer and viewer commands for the first five summaries are recorded in `COMMAND_LIBRARY.zh.md` L15.18. Common single-task id0-10 viewer debug command:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh --gpu 2 --tasks stack_cups --id_start 0 --id_end 10 --continue_on_error --viewer --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/stack_cups_id0_10_viewer
```

Validation record:

- `python3 -m py_compile code_painting/plan_anygrasp_keyframes_r1.py` passed.
- `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_six_tasks.sh` passed.
- `bash -n code_painting/run_plan_anygrasp_keyframes_piper_d435_replay_axes_six_tasks.sh` passed.
- A no-viewer `stack_cups id0 --debug_stop_after_keyframe1` smoke run completed and generated `head_cam_plan.mp4` / `third_cam_plan.mp4`; `plan_summary.json` records `candidate_target_local_z_offset_m=-0.05` and `approach_axis=local_z`, with both arms reaching keyframe-1 grasp.

## L15.19 Design Note: Unify Gripper/Robot Frames During Candidate Filtering

`COMMAND_LIBRARY.zh.md` now has L15.19 at the end. It records the design analysis and the current comparison command only; no code was changed. The older L15.18 `swap_red_blue + local_z` command remains available.

Core analysis:

- Viewer axis colors are still red = local +X, green = local +Y, blue = local +Z.
- Blue/orange grippers in rank previews are left/right or candidate actor colors, not axis colors; orange usually means the right-hand gripper/candidate.
- In the raw AnyGrasp C-shaped gripper frame, local +X is the in-plane palm-to-fingertip direction, local +Y is opening width, and local +Z is the side normal of the C plane.
- A long-term fix should convert the raw AnyGrasp frame into a canonical robot/replay frame during candidate filtering, instead of relying only on planner-stage remapping.

Recommended long-term frame:

```text
robot/replay target local +Z = AnyGrasp raw local +X
robot/replay target local +Y = AnyGrasp raw local +Y
robot/replay target local +X = -AnyGrasp raw local +Z
```

After that change, the blue axis still means local +Z; orange is still only the right-hand C-gripper color; the right-hand orange C-gripper fingertip direction should align with the robot target blue local +Z axis.

The current L15.19 comparison command still uses the L15.18 wrapper and only changes the output root to:

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
```

### L15.19.1 Implemented Robot-Frame Preview + Planner

New code entrypoints:

```text
code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh
code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh
```

Implemented behavior:

- `render_anygrasp_ranked_preview.py` now supports `--candidate_frame_mode robot_replay`.
- In robot-frame summaries, candidate rotations satisfy `target local +Z = AnyGrasp raw local +X`.
- Preview generation supports `--candidate_target_local_z_offset_m`.
- Planner C-gripper actors support `--debug_gripper_actor_forward_axis local_z`, so the C-gripper fingertip direction is drawn along blue local +Z.
- The robot-frame planner wrapper uses `identity + local_z` and no longer relies on planner-stage `swap_red_blue`.
- `run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh` now auto-generates missing robot-frame preview summaries for the current `--tasks`, `--ids`, `--id_start/--id_end`, and `--max_per_task` range before running the planner.
- Add `--skip_preview_generation` when the planner should use only existing summaries.

Generate robot-frame preview explicitly:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_render_anygrasp_ranked_preview_keyframes_d435_robot_frame_six_tasks.sh --gpu 2 --tasks pick_diverse_bottles --ids 0
```

Or run the viewer_gripper planner directly; the wrapper fills missing previews first:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --max_per_task 5 --continue_on_error --viewer --tasks pick_diverse_bottles --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
```

### L15.19.2 Robot-Frame Planner Viewer Commands With Explicit ids

Use `--ids <ID>` to select an episode; multiple episodes can be passed as `--ids 0 4 8`. The wrapper fills missing robot-frame preview summaries before running the planner.

`stack_cups id4`:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --ids 4 --continue_on_error --viewer --tasks stack_cups --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
```

All six tasks with the same id set:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_d435_robot_frame_six_tasks.sh --gpu 2 --ids 0 1 2 3 4 --continue_on_error --viewer --tasks pick_diverse_bottles place_bread_basket stack_cups handover_bottle pnp_bread pnp_tray --visualize_targets --disable_execution_collisions --trajectory_mode cartesian_interp_ik --cartesian_auto_step_m 0.03 --execute_partial_cartesian_plan --allow_partial_dual_stage --print_pose_every 5 --reach_error_pose_source ee --ik_max_rotation_threshold_rad 3.14 --viewer_wait_at_end 0 --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/viewer_gripper
```

## O. First-Frame FoundationPose Direct Strategy Baseline

### O.0 Piper/Pika Data Generation And IK Diagnostic Commands

O.0 now keeps only four reusable commands. Do not treat `pick_diverse_bottles_piper demo_clean_piper_calibrated` as the recommended collection command anymore; it enters the original `grasp_actor` path and currently keeps failing in `tmux gen1-1/gen1-2`.

Latest `tmux gen1-1/gen1-2` result:

```text
seed 72-115: Objects is unstable / target_pose cannot be None for move action
final: user interrupted with Ctrl-C
```

Meaning: the original `pick_diverse_bottles.py` `choose_grasp_pose/grasp_actor` still cannot reliably generate executable grasp targets for the calibrated Piper/Pika setup. This is not a viewer error; `No left camera link` / `No right camera link` is only the fallback warning caused by missing wrist camera links in the current URDF. The current `piper_pika_agx.urdf` has no `left_camera`, `right_camera`, or `camera` link, so O.0 keeps `collect_wrist_camera: false` and saves only the head view.

Kept entrypoints:

```text
envs/pick_diverse_bottles_piper.py
envs/pick_diverse_bottles_piper_motion.py
task_config/demo_clean_piper_motion.yml
task_config/demo_clean_piper_motion_viewer.yml
task_config/demo_clean_piper_ik_orig_tcp.yml
assets/embodiments/piper_pika_agx_ik_orig_tcp/config.yml
assets/embodiments/piper_pika_agx_ik_orig_tcp/curobo.yml
assets/embodiments/piper_pika_agx_ik_orig_tcp/collision_piper_pika.yml
description/task_instruction/pick_diverse_bottles_piper_motion.json
run_pick_diverse_bottles_piper_motion_viewer.sh
run_view_pick_diverse_bottles_piper_scene.sh
```

#### O.0-1 Tested: Generate Head-Only Data Without Viewer

Purpose: generate the usable O.0 comparison dataset. This command uses `pick_diverse_bottles_piper_motion`, keeps the original bottle randomization and stability check, but bypasses the original IK. The arm motion is a calibrated Piper/Pika joint-space motion baseline.

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash collect_data.sh pick_diverse_bottles_piper_motion demo_clean_piper_motion 0
```

Status: tested successfully. Example outputs:

```text
data/pick_diverse_bottles_piper_motion/demo_clean_piper_motion/data/episode0.hdf5
data/pick_diverse_bottles_piper_motion/demo_clean_piper_motion/video/episode0.mp4
data/pick_diverse_bottles_piper_motion/demo_clean_piper_motion/_traj_data/episode0.pkl
data/pick_diverse_bottles_piper_motion/demo_clean_piper_motion/instructions/episode0.json
```

#### O.0-2 Tested: Motion Viewer For The Motion Baseline

Purpose: inspect the O.0 motion baseline in the SAPIEN viewer. This command calls `view_pick_diverse_bottles_piper_motion.py`; each run searches for a stable seed and executes `play_once()` once, so it is not short-circuited by the old `collect_data.py` `seed.txt` progress. `collect_data: false`, so it does not save hdf5 data.

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_pick_diverse_bottles_piper_motion_viewer.sh
```

Headless smoke check:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && DISPLAY=:1.0 timeout 120s bash run_pick_diverse_bottles_piper_motion_viewer.sh --seed 0 --max_seed_tries 3 --hold 0
```

Status: verified on 2026-06-04. Seed 0/1 were skipped as unstable, seed 2 loaded, and `play_once()` finished. Debug axes are shown by default: red/green/blue are local +X/+Y/+Z and the small white cube is the origin. They are currently attached to the two bottle centers and the left/right place targets, not to grasp candidates.

#### O.0-3 Scene Only: Viewer Without Motion

Purpose: inspect only the calibrated Piper/Pika robot, table, random bottles, target axes, and viewer availability. This command does not enter `play_once`, does not execute motion, and does not save data. It now passes `skip_planner=True` to skip Curobo planner initialization, avoiding scene-only viewer stalls in Curobo warmup.

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && bash run_view_pick_diverse_bottles_piper_scene.sh --seed 0 --max_seed_tries 50
```

Headless smoke check:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && DISPLAY=:1.0 timeout 90s python view_pick_diverse_bottles_piper_scene.py --seed 0 --max_seed_tries 3 --hold 0
```

Status: verified on 2026-06-04. Seed 0/1 were skipped as unstable, seed 2 loaded as a stable scene, axes were added, and one frame rendered before exit. After the full viewer command opens the window it stays in the render loop until the SAPIEN window is closed or `Ctrl-C` is pressed.

#### O.0-4 Diagnostic Only: Original IK/Planning Path

Purpose: validate "calibrated Piper/Pika URDF plus the original `pick_diverse_bottles.py` IK/planning logic." This command uses `pick_diverse_bottles_piper`, so it enters the original `grasp_actor/place_actor -> robot.left/right_plan_path -> CuroboPlanner.plan_path` path. The embodiment is `piper_pika_agx_ik_orig_tcp`, which combines the calibrated `piper_pika_agx.urdf` with the built-in RoboTwin Piper TCP conversion.

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && timeout 120s bash collect_data.sh pick_diverse_bottles_piper demo_clean_piper_ik_orig_tcp 0
```

Status: confirmed to enter `Embodiment Config: piper_pika_agx_ik_orig_tcp+piper_pika_agx_ik_orig_tcp` and the original task/IK path, but it did not finish an episode. The latest `tmux gen1-1/gen1-2` logs still show many `Objects is unstable` and `target_pose cannot be None for move action` failures. Use it only to diagnose IK/grasp-candidate issues; it is not the currently recommended data-generation command.

Original IK path:

```text
pick_diverse_bottles.py
-> grasp_actor/place_actor
-> Action(move target_pose)
-> Base_Task.move
-> robot.left_plan_path / robot.right_plan_path
-> robot._trans_from_gripper_to_endlink
-> CuroboPlanner.plan_path
```

`gen-23` error root cause:

- Running `bash collect_data.sh ...` from `~` cannot find the script; run it from `/home/zaijia001/ssd/RoboTwin`.
- The old `demo_clean_piper.yml` used `embodiment: [piper]`, which triggered the dual-arm embodiment path. RoboTwin then tried to load `assets/embodiments/piper/curobo_left.yml`, but the Piper folder only has `curobo.yml`.
- The later `'Robot' object has no attribute 'left_planner'` messages are secondary errors after the first planner initialization failure left a partial `robot` object on the reused task instance.
- The first fix, `embodiment: [piper, piper, 0.60]`, avoids `curobo_left.yml` but still loads the built-in `assets/embodiments/piper/piper.urdf`.
- The calibrated path now uses `piper_pika_agx_calibrated`, loading `assets/embodiments/piper_pika_agx/piper_pika_agx.urdf` and the calibrated base poses.

`gen1-2` error root cause:

- `No left camera link` / `No right camera link` comes from wrist camera link lookup in `envs/robot/robot.py`. The current `piper_pika_agx.urdf` has no `left_camera`, `right_camera`, or generic `camera` link, so the code prints the warning and falls back to the first link. This is not the direct exception that made seed collection fail.
- The actual failure happens during `[Start Seed and Pre Motion Data Collection]`: episode 0 repeatedly failed from seed 421 through seed 730 without finding a usable premotion seed.
- The failures are mainly `Objects is unstable ... 001_bottle`, meaning the bottle fell or moved during initial settling, and `target_pose cannot be None for move action`, meaning the original `pick_diverse_bottles.py` scripted `grasp_actor`/move path did not produce an executable target for the calibrated Piper/Pika setup.
- Disabling wrist saving removes wrist-data dependency and log noise, but if `target_pose cannot be None` continues, the next fix should be a Piper/Pika-specific task variant that changes grasp candidates/object sampling or uses a fixed side-grasp target instead of fully reusing the original ALOHA-AgileX demo planner.

`COMMAND_LIBRARY.zh.md` now ends with Mode O, a simpler `pick_diverse_bottles` comparison experiment. It does not use manual keyframes, human hand orientation, or AnyGrasp candidates. It reads the two bottle world positions from frame 0 of FoundationPose and generates grasp/place targets with the original env task logic.

New entrypoints:

```text
code_painting/plan_first_frame_foundation_pick_diverse_bottles.py
code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh
```

Core behavior:

- Read `foundation_replay_d435/foundation_input_<ID>/multi_object_world_poses.npz`.
- Assign `left_bottle` to the left arm and `right_bottle` to the right arm.
- Set left-arm local `+Z` to world `[+1,0,0]` and right-arm local `+Z` to world `[-1,0,0]`, so both arms attempt an outside-in horizontal side grasp.
- Set the grasp target to the bottle center retreated by `--grasp_surface_retreat_m`, default `0.03m`, opposite the approach axis.
- Let the planner generate pregrasp by retreating along local `+Z` by `--approach_offset_m`, default `0.08m`, matching `envs/pick_diverse_bottles.py`'s `pre_grasp_dis=0.08`.
- Set the action target to the env placement defaults: left `[-0.06,-0.105,1.0]`, right `[0.06,-0.105,1.0]`.
- Reuse the existing Piper planner through `--reuse_plan_summary_json`, including pregrasp/grasp/close/action, IK, video output, and debug output.

Gripper orientation check:

- The Piper/Pika config matches the original ALOHA-AgileX config for `global_trans_matrix=diag(1,-1,-1)`, `delta_matrix=I`, and `grasp_perfect_direction=["front_right","front_left"]`.
- The important difference is the URDF gripper geometry and target-frame convention. ALOHA-AgileX's gripper depth/fingertip direction is naturally link6 local `+X`, with finger opening along local `+/-Y`. Piper/Pika's gripper opening axis is gripper-base local `Z/-Z`, so the structural axes are not identical.
- Mode O currently follows the calibrated Piper/replay pipeline: the planner target frame uses local `+Z` blue as the approach/forward axis. This is consistent with the earlier direct-replay and robot-frame AnyGrasp paths, but it is not the original ALOHA-style local `+X` fingertip-depth convention.
- A strict ALOHA-style local-X comparison would need a separate Mode O branch that writes the approach direction into target local `+X` and invokes the planner with `--approach_axis local_x`.

New validation entrypoint:

```text
code_painting/visualize_mode_o_gripper_frame_conventions.py
```

This script only reads FoundationPose object positions and does not run IK. It draws `piper_local_z`, `aloha_local_x_y_up`, and `aloha_local_x_z_up` frames at the same grasp point, then writes the angle between each local axis and the physical approach direction.

Static visualization command:

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python /home/zaijia001/ssd/RoboTwin/code_painting/visualize_mode_o_gripper_frame_conventions.py --video_id 0 --foundation_frame 0 --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/mode_o_frame_convention_debug
```

Mode O wrapper additions:

- `--target_frame_convention piper_local_z|aloha_local_x_y_up|aloha_local_x_z_up`
- `--plan_only`: write `plan_summary_first_frame_foundation.json` without invoking the planner.

ALOHA-style local-X plan-only comparison:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 0 --plan_only --target_frame_convention aloha_local_x_z_up --output_root /tmp/mode_o_aloha_local_x_plan_only
```

Important note: keys in `multi_object_world_poses.npz` include `pose_world_wxyz`, but the actual array order used by the planner is `[x, y, z, qw, qx, qy, qz]`.

Recommended smoke command:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh --gpu 2 --ids 0 --continue_on_error --output_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/first_frame_foundation_smoke
```

Viewer notes:

- The `Renderer does not support display` warning means the SAPIEN interactive viewer failed to create a window. It is not a target-generation failure.
- In the failed run, viewer creation still printed `CUDA_VISIBLE_DEVICES=2` because the Mode O Python entrypoint restored the environment variable that the wrapper had already unset.
- This is now fixed: in viewer mode the wrapper passes `--gpu -1` to Python, and Python removes `CUDA_VISIBLE_DEVICES` again before invoking the planner.
- If the minimal viewer probe still fails after this fix, the remaining issue is the current `DISPLAY`/Vulkan graphical session. Run from a VNC/graphical terminal that can create a SAPIEN viewer.

Minimal viewer probe:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && cd /home/zaijia001/ssd/RoboTwin && unset CUDA_VISIBLE_DEVICES; [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]] && export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; echo "DISPLAY=$DISPLAY CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}" && conda run -n RoboTwin_bw python /home/zaijia001/ssd/RoboTwin/code_painting/probe_sapien_viewer.py
```

Validation:

- `python -m py_compile code_painting/plan_first_frame_foundation_pick_diverse_bottles.py` passed.
- `bash -n code_painting/run_plan_first_frame_foundation_pick_diverse_bottles_piper_d435.sh` passed.
- `--ids 0 --dry_run` resolved the expected paths.
- The no-viewer `pick_diverse_bottles id0` smoke run generated `plan_summary_first_frame_foundation.json`, `pose_debug.jsonl`, `head_cam_plan.mp4`, and `third_cam_plan.mp4`. In that run both arms reached pregrasp, but grasp did not reach for both arms, so the default safety gate skipped close/action.
