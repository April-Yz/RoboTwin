# Active Memory

Updated: 2026-06-30

Purpose: record high-priority context for the current session and recent data processing. This is not the full command library; reproducible commands still live in `COMMAND_LIBRARY.zh.md` and `agent-read/COMMANDS/`.

## Current Main Data Version

- The current recommended hand-repaint training-data label is `oursv2_piper0515_25ep`.
- Upload tmux session: `upload_oursv2_piper0515_6task`.
- Local zip path: `/home/zaijia001/.cache/huggingface/lerobot/local/robot_oursv2_piper0515_6task_25ep.zip`.
- rclone target: `gdrive:piper/multi/6task/robot_oursv2_piper0515`.
- Zip contents are named like `h2o_<TASK>_oursv2_piper0515_25ep`.

## Frame Convention Notes

- Do not use intermediate versions `h2o_<TASK>_ours_rightcam_m003_color-120` or `h2o_<TASK>_ours_rightcam_m003_color_25ep` directly as final real-robot-consistent data.
- The Piper 0515 real-robot corrected version should be: `/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_ours_rightcam_m003_color_piper0515_25ep`.
- That version should include `meta/piper0515_world_to_base_conversion.json`, convert state/action using the real-robot world-to-base convention, and use gripper scale `0.0967`.

## L16 Stage-2 Repaint Status

- Current right-cam m003 color Stage-2 output root: `/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/stage2_color_rightcam_m003_full_0_120/e0_robot_object`.
- Main script: `/home/zaijia001/ssd/RoboTwin/code_painting/repaint_l16_white_color_debug.py`.
- The current preferred route is color-based white-background removal, not SAM text-prompt white-background removal, because SAM can over-delete near object/background boundaries.
- Frame alignment should follow the robot replay length instead of truncating to the shorter Stage-1 background length.

## Mode N Ablation

- Mode N is the `N. 消融实验：Foundation Pose 物体位置 + 人手朝向` section in `COMMAND_LIBRARY.zh.md`.
- Logic: use FoundationPose object world position plus human hand orientation, without AnyGrasp candidate ranking.
- Keyframe logic: each arm uses up to two effective keyframes; the grasp keyframe uses that frame's Foundation object position and human hand orientation, while the action keyframe uses the second Foundation object position. Current N-7 keeps the grasp orientation for action via `--foundation_pose_action_orientation_source grasp`.
- Current debug uses the N-7 settings: `--foundation_pose_retreat_m 0.10`, `--approach_offset_m 0.07`, `--foundation_pose_action_orientation_source grasp`, `--dual_stage_freeze_reached_arms_on_replan 1`, and `--debug_viewer_overlay`.
- Output root for this 6-task, 5-id-per-task run: `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-7_foundation_pose_humanrot_keyframe_debug_6task5_20260630`.
- Rerun script: `/home/zaijia001/tmp/run_mode_n_n7_keyframe_debug_6task5_20260630.sh`.
- Reruns create tmux sessions: `mode_n_n7_kf_debug_pick_diverse_bottles`, `mode_n_n7_kf_debug_place_bread_basket`, `mode_n_n7_kf_debug_stack_cups`, `mode_n_n7_kf_debug_handover_bottle`, `mode_n_n7_kf_debug_pnp_bread`, `mode_n_n7_kf_debug_pnp_tray`.
- IDs in this run: `pick_diverse_bottles 0-4`, `place_bread_basket 0-4`, `stack_cups 0-4`, `handover_bottle 1-5`, `pnp_bread 7-11`, `pnp_tray 0-4`.
- For each result, inspect first: `foundation_input_<ID>/head_cam_plan.mp4`, `foundation_input_<ID>/debug_execution_preview.mp4`, `foundation_input_<ID>/plan_summary_foundation_pose.json`, and `foundation_input_<ID>/pose_debug.jsonl`.


## Mode N No-Axis Selected25 Pipeline

- Controller tmux: `mode_n_n7_noaxes_selected25_pipeline`.
- Rerun script: `/home/zaijia001/tmp/run_mode_n_n7_noaxes_selected25_pipeline_20260630.sh`.
- Stage1 no-axis planner root: `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/N-7_foundation_pose_humanrot_noaxes_selected25_20260630`.
- Stage2 repaint root: `/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/stage2_color_mode_n_n7_fpose_hrot_noaxes_selected25/e0_robot_object`.
- Data suffix: `mode_n_n7_fpose_hrot_noaxes`; reuses the same 25 ids from `l16_ours_review_first25`.
- Local zip: `/home/zaijia001/.cache/huggingface/lerobot/local/robot_mode_n_n7_fpose_hrot_noaxes_piper0515_6task_25ep.zip`.
- rclone target: `gdrive:piper/multi/6task/robot_mode_n_n7_fpose_hrot_noaxes_piper0515`; the script prints the manual upload command instead of running remote upload from this environment.

## Open Notes

- `make_l16_repaint_montage.py` may still default to the old `L16_human_replay_clean`; reviewing rightcam m003 wrist views needs explicit support for, or a parameter pointing to, the new L16 root.
- The working tree already contains user/history edits. Commits should stage only task-related files and avoid mixing in existing `COMMAND_LIBRARY.zh.md` and command-log edits.

## SKEYP Keyframe Ablation

- `skeyp` uses option B: reuse the `ours` planner-output conversion format instead of creating reinit-style `world_targets_and_status.npz`.
- Controller script: `/home/zaijia001/ssd/RoboTwin/code_painting/run_skeyp_selected25_pipeline.sh`.
- tmux session: `skeyp_selected25_pipeline`.
- Reused selection root: `/home/zaijia001/ssd/RoboTwin/code_painting/l16_ours_review_first25/selections/<TASK>/ours_review_selection.json`.
- Stage-1 output: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_skeyp/stage1`; removes only hands, wrists, watch, and arms while preserving manipulated objects.
- Planner output: `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/skeyp_selected25_rightcam_m003_20260708`.
- Stage-2 output: `/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_skeyp_visible_reinit/e0_robot`; repaints only the robot and does not paste objects again.
- Final data: `/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_skeyp_piper0515_25ep`.
- Local zip: `/home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_piper0515_6task_25ep.zip`.
- Remote upload should be run manually with the `rclone copy ... gdrive:piper/multi/6task/robot_skeyp_piper0515` command printed by the script.

- 2026-07-08 completed one `skeyp_selected25_pipeline` run: Stage-2 reached 25/25 for all six tasks; final zip is `/home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_piper0515_6task_25ep.zip`, validated with 150 parquet files and 6 piper0515 markers.

## SKEYP v2 reinit gripper-only

- Updated: 2026-07-08.
- Purpose: fix the semantic mismatch in `skeyp` v1, which was too close to the `ours` planner-output path, by adding a gripper-only ablation aligned with reinit/pinpointing.
- Controller script: `/home/zaijia001/ssd/RoboTwin/code_painting/run_skeyp_v2_reinit_gripperonly_pipeline.sh`.
- tmux session: `skeyp_v2_reinit_gripperonly_pipeline`.
- Reused selections: `/home/zaijia001/ssd/RoboTwin/code_painting/l16_ours_review_first25/selections/<TASK>/ours_review_selection.json`.
- Stage-1 backgrounds: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_skeyp/v2_reinit_gripperonly/stage1`; prompt is `arms, hands, wrists, watch.`, preserving real objects.
- Reinit retarget symlinks: `/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/skeyp_v2_reinit_gripperonly/h2_pure_d435_selected25`; source is the existing `human_replay/h2_pure_d435`, including `world_targets_and_status.npz` and three replay videos.
- Stage-2 gripper-only outputs: `/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_skeyp_visible_reinit/v2_reinit_gripperonly/e0_gripper`; inspect `w_box_zed_replay_d435.mp4`, `w_mask_zed_replay_d435.mp4`, `mask_zed_replay_d435.mp4`, and `final_repainted.mp4` under each id directory.
- Training-data suffix: `skeyp_reinit_gripperonly`; final local repos look like `/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_skeyp_reinit_gripperonly_piper0515_25ep`.
- Local zip: `/home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_reinit_gripperonly_piper0515_6task_25ep.zip`.
- The current implementation reuses existing reinit-style per-frame trajectories. A stricter "keyframes only to `world_targets_and_status.npz`" version still needs a new reinit-compatible keyframe trajectory generator.

- 2026-07-08 v2 finished: Stage-2 finals, processed HDF5, and Piper0515 LeRobot repos all reached 6 tasks * 25 episodes. Local zip `/home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_reinit_gripperonly_piper0515_6task_25ep.zip` is about 130 MB and contains 150 parquet files plus 6 piper0515 markers.
