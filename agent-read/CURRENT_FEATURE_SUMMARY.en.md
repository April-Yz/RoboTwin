# Current Feature Summary

## Current main line

- The repository default remains the latest v1.x iteration. Foundation, AnyGrasp, Ours v2, Dense Replay, and Piper IK V3 are separate experiment lines.
- See `README.en.md`, `VERSION_SUMMARY.en.md`, and `ACTIVE_MEMORY.en.md` for the Piper/0515 qualitative workflow and recent data state.

## Added in this change

- Added the read-only `Selection Strategy Audit V4`; it neither invokes a planner nor modifies OursV2, Orientation, Fused, Top-score, or legacy outputs.
- Recovers the actual Top-score candidate from `selected_candidates_by_executed_arm` while exposing both historical raw/legacy semantics and the canonical reconstruction.
- Each keyframe has Selection Pose and Planner Target panels. Distinct resolved frames use their own Foundation images in side-by-side columns.
- The full audit covers 6 tasks, 150 episodes, 461 keyframe images, and 2192 arm-strategy records; see `SELECTION_STRATEGY_AUDIT_V4.en.md`.
- The preceding isolated correction line remains `Dense Replay URDF-match v2`; see `COMMANDS/dense_replay_urdfmatch_v2.en.md`.

## Reading order

1. `README.en.md`
2. `CURRENT_FEATURE_SUMMARY.en.md`
3. `VERSION_SUMMARY.en.md`
4. `SELECTION_STRATEGY_AUDIT_V4.en.md`
5. The task-specific `COMMANDS/*.en.md`
