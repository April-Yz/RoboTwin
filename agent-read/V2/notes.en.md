# Notes

- Lift preserves grasp x/y and orientation and only increases z.
- Every segment starts from the preceding trajectory endpoint.
- `third_camera` is the right-side view; `opposite_top_camera` is the opposite overhead view.
- Legacy pickles are rejected and cannot be mixed with v2 replay.

## Foundation O.1.1 / O.1.2

- O.1 no longer resets object pose before close. A post-close state gate verifies that the object stayed stable and lies in the two-finger capture region before attaching a drive at the current pose.
- The default base-only `support_proxy` prevents the open gripper from tipping narrow bottle bodies during pregrasp/grasp.
- O.1.1 sets up from the first annotated keyframe. O.1.2 replaces lift/place with one action using second-keyframe EE xyz while retaining the grasp orientation.
- O.1.2 uses distinct 0515 wrist extrinsics with raw `urdf_end_link` as parent and applies per-side simulation tuning after optical/render conversion. Batch collection runs one episode per ID and tries at most three seeds.

## Mode N-7

- Mode N-7 ports the O.1.2 "action keeps grasp orientation" idea into the Foundation Pose + human-orientation path: action position comes from the second-keyframe Foundation object xyz, while orientation and retreat direction come from the first-keyframe grasp.
- Dual-stage replanning can freeze arms that have already reached the target, preventing a later replan from pulling a reached arm away.
- The R1/AnyGrasp camera-up roll constraint uses local X as forward; Mode N uses local +Z as forward, so that constraint cannot be reused directly.

## Mode M-0611

- Human Replay now defaults to cubic joint-space smoothstep and borrows O.1 V4's explicit small perturbations around the current joint seed, selecting the IK result with the smallest joint change.
- Keyframe-2 action uses keyframe-2 position with the keyframe-1 grasp quaternion, while dual replans freeze reached arms.
- `pick_diverse_bottles` IDs 1 and 2 completed successfully; ID 0 only failed the 4 cm action tolerance.
- Piper IK target and EE report transforms still need to be unified before adding a strict roll constraint about the local +Z approach axis.

## O.1.2.1 Wrist Debug Recorder

- The viewer accepts `--wrist_debug_record 1 --wrist_debug_tag <TAG>` and saves raw left/right videos, a labeled mosaic, and context JSON.
- The missing frame-chain segment is `link6_T_real_tcp`; 0515 supplies `real_tcp_T_camera`. Current tuning estimates missing mechanical extrinsics and is not a physical recalibration.
- The debug recorder uses VS Code-compatible H.264/yuv420p/faststart. The formal wrapper accepts four `WRIST_*` environment variables for headless tuning overrides.
- The viewer adds `--show_camera_frustums 1` to explicitly draw and verify `left_camera`, `right_camera`, and `head_camera`, and fixes the previously ineffective `--hold 1` behavior.
- Fixed Piper IK move/settle/gripper loops that updated wrist images without calling `viewer.render()`; live SAPIEN and dual-wrist preview can now run independently or together.
- Added `script/diagnose_piper_wrist_camera_axes.py` to distinguish Pika physical `+X` from the legacy debug `+Z` forward convention; current wrist forward axes are close to physical `+X`, and the tiny opening-plane correction can be emitted as viewer yaw arguments.


## 2026-06-16: Foundation Grasp-Depth Default

O.1/O.1.2 Foundation Piper IK now defaults `foundation_grasp_standoff` to `0.105m`. The old `0.085m` value could make the bottle appear to enter the gripper root; the new value moves the EE/gripper-base target 2cm back so the bottle sits closer to the fingertip closing region. Debug entrypoints: viewer `--foundation_grasp_standoff_m`, collection wrapper `FOUNDATION_GRASP_STANDOFF_M`.


## 2026-06-16: Wrist Downward View And Right-Side Lateral Bias

The 0515 wrist calibration plus `piper_pika_agx` adapter makes camera forward nearly parallel to gripper `+X`, not downward-looking at the fingers. The right camera center is `Y=-2.74cm`, while the left is `Y=+2.07cm`. The viewer now exposes pitch/lateral tuning; start with left/right pitch `15deg` and right lateral `+0.0067m` when inspecting wrist images.

## 2026-07-10 R/S experiments
- R: oursv2 49ep changes only data composition; entrypoint code_painting/run_oursv2_49ep_pipeline.sh.
- S: graspnet uses the same 25 IDs and AnyGrasp top score without hand-orientation constraints.
- Current oursv2_piper0515 is in per-arm base frames; the old 300-episode repo has world-frame ours rows.
