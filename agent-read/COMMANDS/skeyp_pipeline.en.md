# SKEYP Keyframe Ablation Pipeline

## Purpose

`skeyp` is an ablation variant that generates robot replay from human hand keyframes only. It reuses the `ours` planner-output data conversion format and does not change the existing `ours`, `reinit`, or Mode N logic.

## Core Logic

- Selected ids: reuse the 25 selected episodes per task from `/home/zaijia001/ssd/RoboTwin/code_painting/l16_ours_review_first25/selections/<TASK>/ours_review_selection.json`.
- Stage 1: inpaint only arms, hands, wrists, and watch; preserve the real manipulated objects.
- Planner: run `run_plan_keyframes_human_replay_piper_d435.sh` to generate interpolated `head_cam_plan.mp4`, wrist camera videos, and `pose_debug.jsonl`.
- Stage 2: repaint only the robot onto the Stage-1 background; objects are not pasted again.
- Data conversion: reuse the `process lerobot subset piper0515` steps in `run_l16_ours_selected_pipeline.sh`, producing LeRobot data aligned to the real Piper 0515 frame convention.

## One-Shot Run

```bash
tmux new-session -d -s skeyp_selected25_pipeline \
  'bash /home/zaijia001/ssd/RoboTwin/code_painting/run_skeyp_selected25_pipeline.sh'
```

## Monitoring

```bash
tmux attach -t skeyp_selected25_pipeline
tail -f /home/zaijia001/tmp/skeyp_selected25_rightcam_m003_20260708_logs/task_stack_cups.log
```

Each task has its own log:

```text
/home/zaijia001/tmp/skeyp_selected25_rightcam_m003_20260708_logs/task_<TASK>.log
```

## Outputs

- Stage-1 hands-only backgrounds: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_skeyp/stage1/<TASK>/id_<ID>/stage1_human_inpaint/removed_w_mask_rgb_<ID>.mp4`
- Keyframe replay: `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/skeyp_selected25_rightcam_m003_20260708/<TASK>/foundation_input_<ID>/`
- Stage-2 robot-only repaint: `/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_skeyp_visible_reinit/e0_robot/<TASK>/id_<ID>_skeyp/final_repainted.mp4`
- Intermediate HDF5: `/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_<TASK>_skeyp-120`
- Local LeRobot repos: `/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_skeyp_piper0515_25ep`
- Local zip: `/home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_piper0515_6task_25ep.zip`

## Manual Upload

```bash
rclone copy /home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_piper0515_6task_25ep.zip \
  gdrive:piper/multi/6task/robot_skeyp_piper0515 \
  -P --drive-chunk-size 64M --transfers 4
```

## Related Code

- Controller: `code_painting/run_skeyp_selected25_pipeline.sh`
- Keyframe planner: `code_painting/run_plan_keyframes_human_replay_piper_d435.sh`
- Planner-output conversion: `policy/pi0/scripts/process_repainted_planner_outputs.py`
- piper0515 frame alignment: `code_painting/convert_lerobot_piper0515_world_to_base.py`

## 2026-07-08 Run Result

- tmux session: `skeyp_selected25_pipeline`, finished.
- Stage-2 `final_repainted.mp4`: 25/25 for all six tasks.
- Local zip: `/home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_piper0515_6task_25ep.zip`, about 191 MB.
- Zip validation: 150 parquet files and 6 `piper0515_world_to_base_conversion.json` markers.
- Each `h2o_<TASK>_skeyp_piper0515_25ep` repo has 25 parquet files and the piper0515 frame-conversion marker.

## SKEYP v2: reinit gripper-only

### Purpose

`skeyp_reinit_gripperonly` is the second ablation version aligned with the reinit/pinpointing path. It does not use planner-output conversion and does not paste object/Foundation replay. Real objects stay in the Stage-1 background, and Stage-2 pastes back only the gripper/end-effector from the reinit-style replay.

### Core Differences

- v1: `plan_keyframes_human_replay.py` writes `pose_debug.jsonl` and `head_cam_plan.mp4`, then conversion uses `process_repainted_planner_outputs.py`.
- v2: reuse reinit-style `h2_pure_d435` files: `world_targets_and_status.npz`, `zed_replay_d435.mp4`, and wrist camera videos, then convert with `process_repainted_headcam_with_wrist.py`.
- v2 Stage-1 prompt remains `arms, hands, wrists, watch.`, preserving the real manipulated objects.
- v2 Stage-2 prompt defaults to `robotic gripper, gripper fingers, end effector, robot hand.`, intended to detect and paste only the gripper area.
- The current implementation reuses existing `h2_pure_d435` trajectories, so it is the reinit-compatible gripper-only version. A stricter "keyframes only to `world_targets_and_status.npz`" variant would need a new reinit-compatible keyframe trajectory generator.

### One-Shot Run

```bash
tmux new-session -d -s skeyp_v2_reinit_gripperonly_pipeline \
  'bash /home/zaijia001/ssd/RoboTwin/code_painting/run_skeyp_v2_reinit_gripperonly_pipeline.sh'
```

### Monitoring

```bash
tmux attach -t skeyp_v2_reinit_gripperonly_pipeline
tail -f /home/zaijia001/tmp/skeyp_v2_reinit_gripperonly_20260708_logs/task_stack_cups.log
```

Each task has a separate log:

```text
/home/zaijia001/tmp/skeyp_v2_reinit_gripperonly_20260708_logs/task_<TASK>.log
```

### Outputs

- Stage-1 hands-only backgrounds: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_skeyp/v2_reinit_gripperonly/stage1/<TASK>/id_<ID>/stage1_human_inpaint/removed_w_mask_rgb_<ID>.mp4`
- Reinit-style retarget symlinks: `/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/skeyp_v2_reinit_gripperonly/h2_pure_d435_selected25/<TASK>/id<ID>_d435_z005/`
- Stage-2 gripper-only repaint: `/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_skeyp_visible_reinit/v2_reinit_gripperonly/e0_gripper/<TASK>/id_<ID>_skeyp_gripper/final_repainted.mp4`
- Stage-2 debug videos: `w_box_zed_replay_d435.mp4`, `w_mask_zed_replay_d435.mp4`, `mask_zed_replay_d435.mp4`, and `visible_reinit_meta.json` in the same directory.
- Intermediate HDF5: `/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/h2o_<TASK>_skeyp_reinit_gripperonly-120`
- Local LeRobot repos: `/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_skeyp_reinit_gripperonly_piper0515_25ep`
- Local zip: `/home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_reinit_gripperonly_piper0515_6task_25ep.zip`

### Manual Upload

```bash
rclone copy /home/zaijia001/.cache/huggingface/lerobot/local/robot_skeyp_reinit_gripperonly_piper0515_6task_25ep.zip \
  gdrive:piper/multi/6task/robot_skeyp_reinit_gripperonly_piper0515 \
  -P --drive-chunk-size 64M --transfers 4
```

### Related Code

- Controller: `code_painting/run_skeyp_v2_reinit_gripperonly_pipeline.sh`
- Stage-1 hands-only inpainting: `/home/zaijia001/ssd/inpainting_sam2_robot/remove_anything_video_sam2.py`
- Stage-2 gripper-only repaint: `/home/zaijia001/ssd/inpainting_sam3_robot/remove_anything_video_sam3_robot_visible_reinit.py`
- Reinit/D435 conversion: `policy/pi0/scripts/process_repainted_headcam_with_wrist.py`
- piper0515 frame alignment: `code_painting/convert_lerobot_piper0515_world_to_base.py`
