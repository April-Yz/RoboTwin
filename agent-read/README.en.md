# RoboTwin Project Overview

This repository extends RoboTwin simulation, collection, and policy workflows with Piper/Pika dual-arm scenes, Cartesian IK pick-and-place, hand/object replay, and AnyGrasp planning tools.

For current session context and recent data status, read `agent-read/ACTIVE_MEMORY.en.md` first.

## Current Recommended Workflow

- Piper IK dual-bottle task: use `pick_diverse_bottles_piper_ik` with `demo_piper_ik_seq_v1..v4`.
- Foundation OBJ comparison: use `pick_diverse_bottles_piper_ik_foundation` with `demo_piper_ik_foundation_v1..v4`. O.1 uses an explicit frame, O.1.1 sets up from the first annotated keyframe, and O.1.2 replaces lift/place with second-keyframe EE positions. O.2 adds `pnp_tray_piper_ik_foundation`, picking `left_dark_red_cup` with the left arm and `right_bottle` with the right arm, placing by the second-keyframe OBJ center by default, then opening the grippers.
- Default IK: V1. V2 uses cubic interpolation, V3 uses MotionGen with an IK-interpolation fallback, and V4 uses multi-seed IK.
- Data flow: Phase 1 finds stable, physically successful seeds and saves versioned trajectories. Phase 2 validates and replays them in the same seeded scene, producing HDF5, videos, and instructions.
- Dense Replay: preserve legacy results; use the isolated `run_dense_replay_urdfmatch_v2.sh` when the fixed Piper planning/execution offset must be corrected. The six-task batch entry is `run_dense_replay_urdfmatch_v2_batch.sh`; it writes only to `h2_pure_d435_urdfmatch_v2` and resumes through completeness checks. It fixes the Curobo/SAPIEN link6 frame mismatch and TCP/execution convergence, but does not remove robot-unreachable human orientations or silently promote v1 repaints to v2.
- Selection Strategy Audit V4: scripted read-only comparison of OursV2, Orientation, Fused, and the actual Top-score candidate, now with flat filenames, overlap-safe colors/line styles, and agreement/xyz-distance statistics. Fused matches Orientation in 93.75% of comparable pairs and canonical Top in 8.87%; see `SELECTION_STRATEGY_AUDIT_V4.en.md`.
- Piper real control comparison: use the isolated `run_real_control_compare.sh` to compare OursV2, PiperCanonicalTCP-v1, and measured Piper endPose under synchronized real-q and real-`T_B_RTCP` common inputs. This is distinct from the Orientation/Fused/Top-score candidate-strategy comparison under `outputs_canonical_20260715/eepose`; it writes only to `outputs_real_control_compare_20260716/`.
- Cameras: head, front, side, right-side `third_camera`, opposite overhead `opposite_top_camera`, and top-level `third_view`. Foundation configs export distinct 0515 left/right wrist views through a base Pika adapter plus per-side forward/roll tuning, and support a live dual-wrist viewer mosaic.
- Foundation grasping defaults to a base-only `support_proxy` and a post-close geometric state gate. It never uses `set_pose` to teleport a tipped object back into the gripper.
- Recommended Foundation batches use an isolated run tag, one episode per ID, at most three seed attempts, and an outer timeout to avoid old head-only HDF5 reuse or unbounded retries. The current verified wrapper is `collect_foundation_piper_ik_verified.sh`; it writes stable grasp/wrist parameters per task and saves head plus both wrist videos. The wrist debug recorder saves VS Code-compatible H.264 left/right/mosaic MP4s and parameter JSON. The motion viewer draws and verifies wrist/head frustums, refreshes SAPIEN throughout Piper IK execution, and can run concurrently with the live dual-wrist RGB window. `script/diagnose_piper_wrist_camera_axes.py` recomputes wrist camera forward alignment against Pika physical `+X`, legacy debug `+Z`, and finger-opening `Y`, and emits per-side parent-yaw viewer arguments.
- `script/index_foundation_piper_ik_videos.py` safely maps per-directory Foundation ID N outputs to aggregate `episodeN_*` videos. It uses symlinks and rejects existing episodes by default.

## Environment And Entrypoints

- Conda: `RoboTwin_bw`
- Collection: `collect_data.sh`, `script/collect_data.py`
- Piper IK viewer: `view_pick_diverse_bottles_piper_ik_motion.py`
- Task: `envs/pick_diverse_bottles_piper_ik.py`
- IK: `envs/robot/piper_ik.py`

See `agent-read/COMMANDS/piper_ik_cartesian.en.md`, `piper_ik_foundation.en.md`, `piper_canonical_tcp_v1.en.md`, and `selection_strategy_audit_v4.en.md` for commands. See `agent-read/PIPER_CANONICAL_TCP_V1.en.md` for the Real-Piper-TCP frame contract and `agent-read/VERSION_SUMMARY.en.md` for version relationships.

Use `OUTPUTS_REAL_CONTROL_COMPARE_GUIDE.en.md` for real-control outputs. See `PIPER_CANONICAL_REPLAY_METHOD_COMPARE.en.md` for Canonical Orientation/Fused/Top-score/Human Replay plus the Legacy retreat baseline; `run_replay_method_compare.sh` leaves OursV2 unchanged.

Use `run_ik_logic_grid.sh` to compare the same AnyGrasp/Human semantic source through original Legacy/OursV2 input adaptation and Canonical RTCP adaptation. V2 produces a 2x4 D435 video and audits source positions, axes, per-row target contracts, and the Canonical 19 cm link6 inverse. The old same-numeric-`T_W_RTCP` V1 under `outputs_ik_logic_grid_20260716` has invalid input semantics and is retained only as a counterexample. See `PIPER_IK_LOGIC_GRID_COMPARE.en.md` and `COMMANDS/piper_ik_logic_grid.en.md`.

Paper qualitative assets live under `/home/zaijia001/ssd/data/piper/paper_qualitative_assets`. They currently include the Dense-v2 4x5 grid with separate headers and four-strategy left/right candidate images for frames 38/78 of `pick_diverse_bottles/id0`. See `agent-read/COMMANDS/paper_qualitative_assets.en.md` for reproduction.
