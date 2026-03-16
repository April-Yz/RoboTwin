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
- Candidate colors now encode the role directly: green = all candidates, blue = left-hand top candidates, orange = right-hand top candidates, red = selected candidates.
- Candidate IDs are projected as smaller plain numbers without background boxes or `L/R` prefixes.

Repository note:
- The repo currently has unrelated local changes and large generated outputs. Any future edits should continue to avoid committing generated videos, rollouts, logs, tarballs, and batch output folders.

Orientation investigation note:
- `code_painting/README_anygrasp_orientation_check.md` records the current conclusion that the AnyGrasp path likely missed the fixed local orientation conversion already present in the hand replay path.

Import-chain note:
- `code_painting/replay_r1_h5.py` and `envs/robot/robot.py` were adjusted so `urdfik`-based replay/planning no longer imports `curobo` eagerly during robot construction. This avoids CUDA OOM or import failures in workflows that do not call `set_planner()`.

- The AnyGrasp keyframe planner now exports stable still-image previews for manual candidate selection. For each keyframe it can write rank-specific PNGs (`rank_previews/keyframe_<frame>_rank_<k>.png`) showing the left rank-k candidate in blue and the right rank-k candidate in orange. This is intended for cases where the interactive SAPIEN viewer cannot be created and the workflow must fall back to offscreen-only debugging.

- The AnyGrasp keyframe planner now supports manual candidate overrides with `--manual_candidate FRAME ARM CANDIDATE_IDX`. This is intended for orientation debugging after reviewing `rank_previews/` still images.
