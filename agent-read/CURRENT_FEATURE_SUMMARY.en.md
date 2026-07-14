# Current Feature Summary

## Current main line

- The repository default remains the latest v1.x iteration. Foundation, AnyGrasp, Ours v2, Dense Replay, and Piper IK V3 are separate experiment lines.
- See `README.en.md`, `VERSION_SUMMARY.en.md`, and `ACTIVE_MEMORY.en.md` for the Piper/0515 qualitative workflow and recent data state.

## Added in this change

- Added the isolated `Dense Replay URDF-match v2` without modifying legacy Dense logic or outputs.
- Fixed the fixed Curobo/SAPIEN `link6` `Ry(-90 deg)` frame mismatch, exact 0.12 m TCP inversion, overwritten interpolation parameter, and lack of measured joint convergence.
- Entry point: `code_painting/run_dense_replay_urdfmatch_v2.sh`.
- Code: `render_hand_retarget_piper_dual_npz_urdfmatch_v2*.py`.
- Diagnosis, commands, and limitations: `COMMANDS/dense_replay_urdfmatch_v2.en.md`.

## Reading order

1. `README.en.md`
2. `CURRENT_FEATURE_SUMMARY.en.md`
3. `VERSION_SUMMARY.en.md`
4. The task-specific `COMMANDS/*.en.md`
