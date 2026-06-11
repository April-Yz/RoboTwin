# Version Summary

## BASELINE

Original RoboTwin/ALOHA workflows and the early Piper joint-space motion baseline. It remains useful for scene, joint, and rendering diagnosis but is not the recommended Piper Cartesian pick-and-place path.

## V1

Base Piper Cartesian IK implementation. It solves IK and uses linear joint interpolation. This is the current default when speed and stability are the priority.

## V2

The unified sequential-trajectory protocol includes three later variants: V2 uses cubic interpolation, V3 uses MotionGen with fallback, and V4 uses multi-seed IK with cubic interpolation. All four IK variants share trajectory schema v2, action ordering, sequential endpoints, lift/place corrections, and camera outputs.

## Current Recommendation

Use `demo_piper_ik_seq_v1` by default. Use V2 for smoother interpolation, V3 for MotionGen experiments, and V4 for multi-seed IK. Old `demo_piper_ik_v*` pickles are incompatible with the current interface.

## O.1 Foundation Variants

`demo_piper_ik_foundation_v1..v4` retain the same IK-version semantics while replacing random RoboTwin bottle assets with positions and source OBJ meshes from Foundation NPZ files. O.1 uses an explicit frame, O.1.1 sets up from the first annotated keyframe, and O.1.2 replaces lift/place with second-keyframe EE xyz. Start with V1. Defaults include a base-only `support_proxy`, no-teleport grasp-state gating, and calibrated 0515 left/right wrist cameras. Pickles require an exact Foundation mode/source/keyframe/action/geometry context match. Batch collection isolates outputs with run tags, runs one episode per ID, and bounds seed retries.
