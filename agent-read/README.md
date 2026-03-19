# RoboTwin Agent Notes

## Current Workflow Map

The active `code_painting/` workflow is best understood as five stages:

1. Upstream hand extraction
2. RoboTwin hand replay
3. Upstream FoundationPose extraction
4. RoboTwin object replay
5. RoboTwin AnyGrasp candidate execution

Only stages 2, 4, and 5 are implemented in this repo. Stages 1 and 3 are upstream producers whose outputs are consumed here.

## Reading Order

- Pipeline index: [V1.7_pipeline_index.md](/home/zaijia001/ssd/RoboTwin/agent-read/V1.7_pipeline_index.md)
- Pipeline index (ZH): [V1.7_pipeline_index_ZH.md](/home/zaijia001/ssd/RoboTwin/agent-read/V1.7_pipeline_index_ZH.md)
- Hand replay pipeline: [V1.7_hand_pipeline.md](/home/zaijia001/ssd/RoboTwin/agent-read/V1.7_hand_pipeline.md)
- Hand replay pipeline (ZH): [V1.7_hand_pipeline_ZH.md](/home/zaijia001/ssd/RoboTwin/agent-read/V1.7_hand_pipeline_ZH.md)
- Object/FoundationPose replay pipeline: [V1.7_object_pipeline.md](/home/zaijia001/ssd/RoboTwin/agent-read/V1.7_object_pipeline.md)
- Object/FoundationPose replay pipeline (ZH): [V1.7_object_pipeline_ZH.md](/home/zaijia001/ssd/RoboTwin/agent-read/V1.7_object_pipeline_ZH.md)
- AnyGrasp candidate and execution pipeline: [V1.7_anygrasp_pipeline.md](/home/zaijia001/ssd/RoboTwin/agent-read/V1.7_anygrasp_pipeline.md)
- AnyGrasp candidate and execution pipeline (ZH): [V1.7_anygrasp_pipeline_ZH.md](/home/zaijia001/ssd/RoboTwin/agent-read/V1.7_anygrasp_pipeline_ZH.md)
- Command log: [V1.8_command_log.md](/home/zaijia001/ssd/RoboTwin/agent-read/V1.8_command_log.md)
- Command log (ZH): [V1.8_command_log_ZH.md](/home/zaijia001/ssd/RoboTwin/agent-read/V1.8_command_log_ZH.md)

## Current State

- `code_painting/run_hand_retarget_r1_npz.sh` and `code_painting/run_hand_retarget_r1_npz_urdfik*.sh` are the main RoboTwin-side entrypoints for replaying hand trajectories from `hand_detections_<id>.npz`.
- `code_painting/run_object_pose_r1_npz.sh` and `code_painting/run_multi_object_pose_r1_npz_batch.sh` are the main RoboTwin-side entrypoints for replaying FoundationPose object trajectories.
- `code_painting/run_plan_anygrasp_keyframes_r1_batch.sh` is the main RoboTwin-side entrypoint for AnyGrasp candidate visualization and execution.
- `urdfik` is the current default backend for the AnyGrasp planning workflow.

## Boundaries

- `hand_detections_<id>.npz` is treated as an upstream input. RoboTwin replays it but does not currently generate it.
- FoundationPose `poses.npz` is also treated as an upstream input. RoboTwin replays it but does not estimate it.
- AnyGrasp inference is assumed to run on another server. RoboTwin consumes the resulting grasp JSON files.

## Important Notes

- The repo currently contains unrelated local changes and many generated outputs. Future edits should continue to avoid committing generated videos, rollouts, logs, tarballs, and batch output folders.
- `code_painting/README_anygrasp_orientation_check.md` records the current orientation-conversion investigation for AnyGrasp.
- `code_painting/replay_r1_h5.py` and `envs/robot/robot.py` were adjusted so `urdfik` workflows do not import `curobo` eagerly during robot construction.
- In dual-arm AnyGrasp execution, selected keyframe target axes are now tracked per `(frame, arm)` so both hands can keep visible selected targets in the headed viewer during synchronized stage execution.
- In execution-mode debug rendering, only the currently active keyframe target pair should remain visible; future keyframe targets are hidden until they become active.
- `code_painting/render_anygrasp_ranked_preview.py` can generate direct AnyGrasp annotated images without starting RoboTwin.
