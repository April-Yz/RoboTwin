# Version Summary

## BASELINE

Original RoboTwin/ALOHA workflows and the early Piper joint-space motion baseline. It remains useful for scene, joint, and rendering diagnosis but is not the recommended Piper Cartesian pick-and-place path.

## V1

Base Piper Cartesian IK implementation. It solves IK and uses linear joint interpolation. This is the current default when speed and stability are the priority.

## V2

The unified sequential-trajectory protocol includes three later variants: V2 uses cubic interpolation, V3 uses MotionGen with fallback, and V4 uses multi-seed IK with cubic interpolation. All four IK variants share trajectory schema v2, action ordering, sequential endpoints, lift/place corrections, and camera outputs.

## Current Recommendation

Use `demo_piper_ik_seq_v1` by default. Use V2 for smoother interpolation, V3 for MotionGen experiments, and V4 for multi-seed IK. Old `demo_piper_ik_v*` pickles are incompatible with the current interface.

## Piper TCP/EE IK V3 (Separate Conversion Line)

An isolated Piper TCP/EE IK V3 was added on 2026-07-13 to fix the non-reversible pose-to-link6 IK caused by current OursV2 fields being named EE while actually containing a TCP with a 12 cm offset. This V3 is not the MotionGen V3 above. It uses new renderer/planner/runner entries, defaults to `ours_tcp` semantics, writes under `human_replay_v3/`, and does not modify OursV2. See `agent-read/PIPER_IK_V3.en.md`.

## Dense Replay URDF-match v2 (Isolated Baseline Correction)

The isolated Dense Replay v2 was added on 2026-07-14. It retains Dense's frame-wise human retargeting definition and only fixes legacy planning/execution correspondence: matched `piper_pika_agx` models, the fixed Curobo-to-SAPIEN link6 `Ry(-90 deg)` adapter, exact 0.12 m TCP inversion, restoration of 10 interpolation waypoints, and measured joint convergence. Legacy code and outputs remain intact. See `agent-read/COMMANDS/dense_replay_urdfmatch_v2.en.md`.

The same-day update adds a sequential six-task batch and a separate expanded paper grid. All 424 discovered inputs write under `h2_pure_d435_urdfmatch_v2`, leaving v1 intact. Only raw replay is currently v2; existing Dense Stage-2 repaint/HDF5 artifacts remain v1 and are not matched v2 downstream results.

## Selection Strategy Audit V4 (Read-only Audit)

The independent V4 audit was added on 2026-07-14. It reads existing OursV2 data, hand-keyframe previews, Top-score plan summaries, AnyGrasp JSON, Foundation replay, and the 0515 calibration without invoking a planner. It obtains the actual Top-score candidate from `selected_candidates_by_executed_arm`, preserves historical raw/legacy semantics, and adds an audit-only canonical reconstruction. Its Selection/Planner panels display distinct resolved Foundation frames in separate columns. A same-day update adds flat `id<ID>_keyframe_*` files, overlap-safe line styles, and an independent agreement/position/contribution statistics script. Generated artifacts live under `code_painting/selection_strategy_compare_v4/` and are not versioned. See `agent-read/SELECTION_STRATEGY_AUDIT_V4.en.md`.

## O.1 Foundation Variants

`demo_piper_ik_foundation_v1..v4` retain the same IK-version semantics while replacing random RoboTwin bottle assets with positions and source OBJ meshes from Foundation NPZ files. O.1 uses an explicit frame, O.1.1 sets up from the first annotated keyframe, O.1.2 replaces lift/place with second-keyframe EE xyz, and O.1.2.1 adds per-side wrist forward/roll tuning without changing the original 0515 files. Start with V1. Defaults include a base-only `support_proxy` and no-teleport grasp-state gating. Pickles require an exact Foundation mode/source/keyframe/action/geometry context match. Batch collection isolates outputs with run tags, runs one episode per ID, bounds seed retries, and can index videos by Foundation ID as `episode<ID>`.

### O.1.2.1 Wrist Debug Addition

O.1.2.1 separates the confirmed parent-frame composition error from the still-unmeasured `link6_T_real_tcp`, and adds same-frame left/right/mosaic viewer recording with parameter JSON. It does not change V1-V4 IK semantics or formal collection trajectories.
Debug videos now use H.264/faststart. The formal collection wrapper accepts four `WRIST_*` environment variables for headless camera overrides.
The viewer can validate and draw wrist/head camera linesets with `--show_camera_frustums 1`; `--hold 1` now retains the final window until the user exits.
The 2026-06-16 fix restores per-step `viewer.render()` in the custom Piper IK executor, supporting both live SAPIEN-only and live SAPIEN-plus-wrist-RGB modes.
The same date adds a wrist forward-axis diagnostic script. Current results show camera forward is close to Pika physical `+X`, with less than one degree of opening-plane error; the roughly 90-degree difference to legacy debug `+Z` should not be applied directly as an extrinsic correction. The viewer can now apply this tiny parent-frame yaw through `--wrist_left_yaw_deg` / `--wrist_right_yaw_deg`.


## 2026-06-16: Foundation O.1/O.1.2 Gripper Standoff

The default `foundation_grasp_standoff` for Foundation Piper IK V1-V4 changed from `0.085m` to `0.105m`. This is a default grasp-depth update for O.1/O.1.2: the gripper-base/EE grasp target stays 2cm farther from the bottle center so the object sits closer to the fingertip/scissor region. Interfaces remain compatible and can be overridden with viewer `--foundation_grasp_standoff_m` or collection-wrapper `FOUNDATION_GRASP_STANDOFF_M`.


## 2026-06-16: Wrist Pitch/Lateral Debug Interface

The Foundation Piper IK viewer now exposes temporary wrist-camera `parent_pitch_deg` and `parent_lateral_offset_m` overrides to test the 0515 wrist calibration issue: the cameras are nearly coplanar with the gripper forward plane but not downward-looking. The recommended first trial is left/right pitch `15deg` and right lateral `+0.0067m`. These are viewer/debug parameters and do not change gripper grasp planning.


## 2026-06-16: O.1.2 Verified Grasp/Wrist V2

The recommended O.1.2 viewer baseline now uses `foundation_grasp_standoff_m=0.14`, wrist forward `0.145/0.13`, pitch `15deg`, and lateral `-0.0207/0.0274`. Real-grasp debug options were added so the viewer can switch collision proxies, require two-finger contact, and disable grasp-assist for pure-physics observation.

Update: the verified grasp/wrist v2 command explicitly includes `foundation_capture_radial_tolerance_m=0.08` and `foundation_grasp_assist_max_distance_m=0.16`; the default gate is slightly too strict for `standoff=0.14`.


## 2026-06-16: O.2 pnp_tray Foundation IK

O.2 is a task extension of O.1.2 Foundation IK and does not change V1-V4 IK semantics. The new `pnp_tray_piper_ik_foundation` maps Foundation NPZ objects to left `left_dark_red_cup` and right `right_bottle`, using pnp_tray manual keyframes. The default action target source is the Foundation second-keyframe OBJ center rather than the `h2_pure_d435` EE target. The action order is `pregrasp -> grasp -> close -> object-keyframe action -> open_gripper`.

Start O.2 validation from V1. `pnp_tray` uses `foundation_grasp_standoff=0.105`, because the pick_diverse verified-v2 value `0.14` pushes the left cup on ID0. Optional `foundation_pregrasp_clearance=0.06` can be used for lifted-pregrasp avoidance trials, while it is disabled by default. Formal collection uses `collect_foundation_piper_ik_verified.sh pnp_tray ...` and still saves head and both wrist videos.

## 2026-07-15: PiperCanonicalTCP-v1 (isolated Real-TCP line)

Added isolated `code_painting/piper_canonical_tcp_v1/` without modifying OursV2 or Piper IK V3. It names `L6_SIM`, `L6_URDF`, `RTCP`, and `CGRASP` separately; runtime validation establishes `T_L6SIM_L6URDF=Ry(+pi/2)`, while the server tool remains exactly `T_L6URDF_RTCP=Ry(-1.57)@Tx(0.19)`. It supports corrected same-q joint comparison and Orientation/Fused/Top-score EE-pose comparison. See `PIPER_CANONICAL_TCP_V1.en.md`.

### 2026-07-16: real-control-compare v1.x increment

Added `real_control_compare.v1` under the same isolated directory without promoting a major version or modifying OursV2. It uses synchronized q/endPose from Piper raw episodes for three-chain Joint and EE-pose comparisons and keeps outputs separate from the 2026-07-15 candidate-strategy batch. Both videos plot 0515 world XYZ; local TCP +X/+Y/+Z remain red/green/blue.

### 2026-07-16: Canonical replay-method compare v1.x increment

An isolated Canonical Human Replay wrapper maps human/CGRASP axes to RTCP, forces final retreat to zero, and reuses the Canonical planner. The four-method video compares Orientation/Fused/Top-score/Human Replay; the five-way video appends an explicit 12 cm Legacy retreat baseline. OursV2 and 2026-07-15 outputs remain unchanged.

### 2026-07-16: Legacy / Canonical IK semantic grid v2 (v1.x increment)

OursV2 remains unchanged. The old same-numeric-target V1 is invalid because it omitted Legacy candidate offsets/Human retreat and did not materialize the Human remap. V2 shares one AnyGrasp/Human semantic source: the top row applies the complete original Legacy target adapter and EE reach, while the bottom row converts to Canonical RTCP and inverts the server 19 cm tool. Composition audits source positions, axes, row-specific contracts, and `link6-RTCP=[-0.19,0,0]` instead of requiring numerically identical planner targets.

### 2026-07-16: paper qualitative assets v1.x increment

No method or IK version changes. The Dense-v2 paper grid moves titles out of video content into separate 38 px headers and preserves the former version. A reusable JSON-driven keyframe-candidate exporter now produces eight four-strategy left/right split images and two contact sheets for frames 38/78 of `pick_diverse_bottles/id0`.
