# OursV2 Data-Count and GraspNet Ablations

## R: 1 Robot + 49 Ours

This experiment changes only the oursv2 data composition. Video generation, planner outputs, state/action layout, and Piper0515 conversion remain unchanged. Each task contains 49 ours episodes so one real-robot episode can be added for a 50-episode task.

Selection priority is accepted -> medium -> unreviewed -> bad. If fewer than 49 unique planner/Stage-2 outputs exist, the earliest accepted rows are repeated. Exact order and provenance are stored under:

~~~text
/home/zaijia001/ssd/RoboTwin/code_painting/l16_oursv2_review_49ep/selections/<TASK>/oursv2_49ep_selection_manifest.json
~~~

| task | accepted | medium | unreviewed | bad | repeated | unique | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| pick_diverse_bottles | 29 | 1 | 19 | 0 | 0 | 49 | 49 |
| place_bread_basket | 31 | 0 | 0 | 12 | 6 | 43 | 49 |
| handover_bottle | 29 | 10 | 0 | 8 | 2 | 47 | 49 |
| pnp_bread | 27 | 6 | 16 | 0 | 0 | 49 | 49 |
| pnp_tray | 26 | 4 | 19 | 0 | 0 | 49 | 49 |
| stack_cups | 7 | 30 | 0 | 4 | 8 | 41 | 49 |

Run:

~~~bash
tmux new-session -d -s oursv2_49ep_pipeline \
  'SKIP_UPLOAD=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_oursv2_49ep_pipeline.sh'
~~~

Final paths:

~~~text
/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_oursv2_piper0515_49ep
/home/zaijia001/.cache/huggingface/lerobot/local/robot_oursv2_piper0515_6task_49ep.zip
gdrive:piper/multi/6task/robot_oursv2_piper0515_49ep
~~~

The final conversion updates both observation.state and action. In each Piper0515 arm base frame, +x is forward, +y is left, and +z is up. Current oursv2_piper0515 data is in the same workspace as the real robot. The older combined 300-episode repo still contains world-frame ours rows.

## S: GraspNet Top-Score Ablation

This uses the same 25 IDs, keyframes, right-wrist camera configuration, Stage-1/Stage-2, training conversion, and Piper0515 alignment as ours. The main changed variable is the highest AnyGrasp score among valid candidates for the expected object.

~~~text
candidate_selection_mode=top_score_auto
candidate_max_rotation_distance_deg=-1
candidate_keep_camera_up=0
enforce_candidate_distance_constraint=0
~~~

Top-score selection does not use a hand-orientation threshold or hand-rotation tie-break. Expected-object matching remains enabled, while the candidate-distance threshold is disabled. Empty keyframes resolve to the nearest non-empty AnyGrasp frame.

~~~bash
tmux new-session -d -s graspnet_selected25_pipeline \
  'SKIP_UPLOAD=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_graspnet_selected25_pipeline.sh'
~~~

~~~text
/home/zaijia001/.cache/huggingface/lerobot/local/h2o_<TASK>_graspnet_piper0515_25ep
/home/zaijia001/.cache/huggingface/lerobot/local/robot_graspnet_piper0515_6task_25ep.zip
gdrive:piper/multi/6task/robot_graspnet_piper0515
~~~

## Local Completion Status (2026-07-10)

- R: all six tasks contain 49 episodes and all six repositories include `piper0515_world_to_base_conversion.json`; the archive is 1.3 GB.
- S: all six tasks contain 25 episodes and all six repositories include `piper0515_world_to_base_conversion.json`; the archive is 351 MB.
- Both pipelines completed with `SKIP_UPLOAD=1`. The rclone destinations are documented, but no data has been exported to external Google Drive yet.
