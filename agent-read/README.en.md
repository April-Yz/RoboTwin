# RoboTwin Project Overview

This repository extends RoboTwin simulation, collection, and policy workflows with Piper/Pika dual-arm scenes, Cartesian IK pick-and-place, hand/object replay, and AnyGrasp planning tools.

## Current Recommended Workflow

- Piper IK dual-bottle task: use `pick_diverse_bottles_piper_ik` with `demo_piper_ik_seq_v1..v4`.
- Foundation OBJ comparison: use `pick_diverse_bottles_piper_ik_foundation` with `demo_piper_ik_foundation_v1..v4`. O.1 uses an explicit frame, O.1.1 sets up from the first annotated keyframe, and O.1.2 replaces lift/place with second-keyframe EE positions.
- Default IK: V1. V2 uses cubic interpolation, V3 uses MotionGen with an IK-interpolation fallback, and V4 uses multi-seed IK.
- Data flow: Phase 1 finds stable, physically successful seeds and saves versioned trajectories. Phase 2 validates and replays them in the same seeded scene, producing HDF5, videos, and instructions.
- Cameras: head, front, side, right-side `third_camera`, opposite overhead `opposite_top_camera`, and top-level `third_view`.
- Foundation grasping defaults to a base-only `support_proxy` and a post-close geometric state gate. It never uses `set_pose` to teleport a tipped object back into the gripper.

## Environment And Entrypoints

- Conda: `RoboTwin_bw`
- Collection: `collect_data.sh`, `script/collect_data.py`
- Piper IK viewer: `view_pick_diverse_bottles_piper_ik_motion.py`
- Task: `envs/pick_diverse_bottles_piper_ik.py`
- IK: `envs/robot/piper_ik.py`

See `agent-read/COMMANDS/piper_ik_cartesian.en.md` and `piper_ik_foundation.en.md` for commands, and `agent-read/VERSION_SUMMARY.en.md` for version relationships.
