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

- First-keyframe-only manual inspection can be done by running the AnyGrasp planner with `--keyframes 1 1` together with manual candidate overrides. This keeps the workflow in a single-frame debug loop while preserving the existing viewer/video outputs.

- In headed SAPIEN mode, the AnyGrasp planner now keeps keyframe axes and candidate gripper visuals persistent in the viewer for the current active keyframe. This makes orientation debugging possible directly in the simulator window instead of relying on single-frame flashes.

- In headed AnyGrasp debug mode, each displayed left/right candidate now has its own coordinate axis actor. This avoids the previous behavior where only the selected arm kept an axis while the other manual candidate showed only the gripper mesh.

- The AnyGrasp planner now supports an extended reach experiment mode via `--replan_until_reached 1`. This is meant for isolating whether poor execution accuracy comes from too few replanning iterations versus a deeper target-pose / orientation-conversion mismatch. Stage summaries now keep per-attempt error histories.

- The AnyGrasp planner now separates execution-arm reach error from auxiliary supervision. You can choose `--reach_error_pose_source tcp` or `ee`, and the summary keeps supervision errors for the non-executed hand when manual candidates are provided for both sides. This is intended for debugging TCP-vs-wrist mismatches and missing right-hand reporting.

- Added explicit execution-mode metadata for AnyGrasp stage runs: `executed_arms` and `supervision_only_arms` in `plan_summary.json`, plus an `[exec-mode]` runtime log line. This makes it explicit that the non-selected hand in single-arm runs is supervision-only and not actuated.

- Added a focused investigation note for the right-hand non-arrival question at `agent-read/2026-03-17_right_hand_supervision_analysis/README.md`.

- Versioned record for this behavior clarification: `agent-read/V1.4_anygrasp_execution_mode_supervision.md`.

- Added dual-arm synchronized stage execution mode for AnyGrasp planning via `--execute_both_arms` (with `--arm auto`). In each stage both arms are planned/executed together, and transition requires both sides to satisfy reach tolerance.

- Detailed investigation note for this change: `agent-read/2026-03-17_dual_arm_execution/README.md`.

- Versioned note for this feature: `agent-read/V1.5_anygrasp_dual_arm_execution.md`.

- Added init-pose prefix execution support in `plan_anygrasp_keyframes_r1.py`: before moving to keyframe-1, the planner now re-applies renderer init joints and can emit fixed init frames via `--init_prefix_frames N` for downstream trimming.

- Versioned note for this update: `agent-read/V1.6_anygrasp_init_prefix.md`.
