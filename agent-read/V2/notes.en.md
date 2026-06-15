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
