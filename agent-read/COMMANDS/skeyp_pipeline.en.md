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
