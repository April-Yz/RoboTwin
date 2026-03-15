# RoboTwin Agent Notes

## Current Focus

This repository currently includes a debug-heavy AnyGrasp keyframe planning workflow under `code_painting/`.

### AnyGrasp keyframe planner

Relevant files:
- `code_painting/plan_anygrasp_keyframes_r1.py`
- `code_painting/plan_anygrasp_keyframes_r1_batch.py`
- `code_painting/run_plan_anygrasp_keyframes_r1_batch.sh`
- `code_painting/README_anygrasp_keyframe_planner.md`

Pipeline summary:
- Read AnyGrasp grasp candidates for two source keyframes.
- Read human hand gripper orientations from `hand_detections_<id>.npz`.
- Rank candidates per arm, select one arm and one candidate per keyframe, then plan `pregrasp -> grasp -> action` in RoboTwin.
- Default planning backend for this workflow is `urdfik`.

Debug outputs currently supported:
- `debug_selection_preview.mp4`: object replay plus persistent keyframe target visualization.
- `debug_execution_preview.mp4`: slow execution preview with persistent target overlays during planning.
- `head_cam_plan.mp4`: main execution video.
- `plan_summary.json`: selected candidates, stage status, and diagnostics.

Candidate visualization semantics:
- Green grippers: common candidate set shown for the active keyframe.
- Blue grippers: arm-specific top-ranked candidates.
- Red gripper: final selected candidate.
- Left-arm candidates carry a black marker above the gripper.
- Right-arm candidates carry a black marker below the gripper.
- Candidate IDs are projected into the debug videos: raw IDs as numbers, arm-ranked IDs as `L<num>` / `R<num>`.

Repository note:
- The repo currently has unrelated local changes and large generated outputs. Any future edits should continue to avoid committing generated videos, rollouts, logs, tarballs, and batch output folders.
