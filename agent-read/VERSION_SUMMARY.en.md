# Version Summary

## BASELINE

Original RoboTwin/ALOHA workflows and the early Piper joint-space motion baseline. It remains useful for scene, joint, and rendering diagnosis but is not the recommended Piper Cartesian pick-and-place path.

## V1

Base Piper Cartesian IK implementation. It solves IK and uses linear joint interpolation. This is the current default when speed and stability are the priority.

## V2

The unified sequential-trajectory protocol includes three later variants: V2 uses cubic interpolation, V3 uses MotionGen with fallback, and V4 uses multi-seed IK with cubic interpolation. All four IK variants share trajectory schema v2, action ordering, sequential endpoints, lift/place corrections, and camera outputs.

## Current Recommendation

Use `demo_piper_ik_seq_v1` by default. Use V2 for smoother interpolation, V3 for MotionGen experiments, and V4 for multi-seed IK. Old `demo_piper_ik_v*` pickles are incompatible with the current interface.
