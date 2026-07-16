# Piper Legacy-original / Canonical-RTCP 2x4 Semantic Comparison

## Conclusion

The V1 comparison under `outputs_ik_logic_grid_20260716/` is invalid. It sent the same numeric `T_W_RTCP` to both rows while labeling the top row “original OursV2.” Original OursV2 also applies a `-0.05 m` target offset on candidate local `+Z`; Human Replay uses a `0.14 m @ local human +Z` retreat. Dropping those adapters made the Legacy EE/link6 origin visibly align with the object.

V1 had a second independent error. Human uses `reuse_plan_summary_json`, and that path does not execute `candidate_orientation_remap_label`. The configured `CGRASP_HUMAN -> RTCP` remap was therefore only a label and never changed the stored pose.

V2 does not modify OursV2. It starts from the same semantic candidate/hand center and then applies the correct native adapter for each row:

| | Orientation | Fused | Top-score | Human Replay |
|---|---|---|---|---|
| Top | Legacy original | Legacy original | Legacy original | Legacy original |
| Bottom | Canonical RTCP | Canonical RTCP | Canonical RTCP | Canonical RTCP |

## Actual Input

The shared input is not an identical numeric planner target. It is the same semantic AnyGrasp candidate center or Human hand/gripper center. Numeric planner targets are expected to differ between rows.

### Legacy original (top)

- Orientation/Fused keep CGRASP local axes, apply `-0.05 m @ local +Z` to the target, and derive a `0.12 m @ local +Z` pregrasp.
- Top-score keeps native AnyGrasp axes and uses the same local-Z target offset and pregrasp convention.
- Human Replay exactly restores the original `0.14 m @ local human +Z` retreat recipe. This value also participates in the handover-keyframe adjustment; it is not merely a removable final translation.
- Renderer: `HandRetargetPiperDualURDFIKRenderer` / `robot._trans_from_gripper_to_endlink(...)`.
- Native solver settings: `3.14 rad` rotation threshold, one seed, and EE reach/readback.

### Canonical RTCP (bottom)

- Orientation/Fused preserve the candidate-center origin and apply `R_W_RTCP = R_W_CGRASP @ R_CGRASP_RTCP` once.
- Top-score preserves the candidate-center origin and interprets native axes directly as RTCP.
- Human Replay first uses the same original 0.14 m recipe to construct the same semantic source, then removes the retreat and materializes the CGRASP_HUMAN rotation as RTCP. The final Canonical target retreat is zero.
- Pregrasp: `0.12 m @ local_RTCP +X`.
- Renderer: `PiperCanonicalTCPRenderer`, using the exact inverse of server `T_L6URDF_RTCP = Ry(-1.57) @ Tx(0.19)`.
- The link6 origin derived from an RTCP target obeys `p_L6 = p_RTCP - 0.19 * local_RTCP_X`.
- Native solver settings: `0.12 rad` rotation threshold, 20 seeds, and TCP reach/readback.

V2 is therefore a comparison of two complete native pipelines under a shared semantic source, not a one-variable link6-conversion ablation.

## Audit

`semantic_source_audit.json` checks:

- arm/frame/candidate identity;
- cross-row semantic-source world xyz;
- Orientation/Fused/Human `CGRASP -> RTCP` rotation relations;
- native Top-score axes;
- every cell's target contract;
- Canonical `link6 - RTCP = [-0.19, 0, 0]` in local RTCP coordinates.

Any failure blocks composition.

## Camera Profiles

An older V2 sample mixed two render profiles. The first three AnyGrasp strategies used the `640x360`, `fovy=90deg`, 10 fps wide diagnostic view, while Human Replay used the calibrated-D435 `640x480`, `fovy=42.499880046655484deg`, 5 fps view. Both used the same head pose, but the FOV/resolution mismatch made the first three panels look wider with a smaller robot and invalidated direct projection comparison.

The runner now requires `--camera-profile d435|wide`:

- `d435`: `640x480`, `fovy=42.499880046655484deg`, 5 fps;
- `wide`: `640x360`, `fovy=90deg`, 10 fps.

The compositor verifies profile, width, height, and fps for all eight sources and refuses to compose any mixed set. Wide is retained as a diagnostic view and does not represent physical D435 intrinsics.

## Outputs

Default root:

`code_painting/piper_canonical_tcp_v1/outputs_ik_semantic_grid_v2_20260716/`

Source files and audit metadata are isolated by camera profile. Final videos are flat under `vis/`:

- `vis/<task>_id<id>_vd435.mp4`: all eight panels use D435;
- `vis/<task>_id<id>_vwide.mp4`: all eight panels use wide;
- `_grid_meta/<profile>/<task>/foundation_input_<id>/semantic_source_audit.json`: semantic source, axes, camera profile, target contract, and 19 cm link6 audit;
- `_grid_meta/<profile>/<task>/foundation_input_<id>/legacy_original_vs_canonical_rtcp_2x4_<profile>.manifest.json`: source videos, execution state, and media properties;
- `_sources/<profile>/legacy_original/...` and `_sources/<profile>/canonical_rtcp/...`: eight cells;
- `_superseded/canonical_human_before_shared_source_fix_20260716/`: preserved intermediate bad sample, excluded from the final video.

The current 6x1x2 batch fixes these samples: `pick_diverse_bottles/id0`, `place_bread_basket/id0`, `stack_cups/id0`, `handover_bottle/id1`, `pnp_bread/id7`, and `pnp_tray/id0`. `pnp_bread/id1` lacks the complete input intersection, so the first complete sample, `id7`, is used.

## handover_bottle / foundation_input_1 Validation

- All four strategies pass the semantic audit. Every source-position delta is `0.0 m`; rotation-matrix errors range from zero to `4.2e-16`.
- V2 Legacy Orientation versus historical `viewer_gripper`, and V2 Legacy Top-score versus historical `S_graspnet_topscore...`, have zero maximum delta in candidate identity, raw pose, and planner target. The top row now reproduces original input logic.
- Canonical Human completes the internal handover (`[handover] SUCCESS`). The generic summary still returns failure because of an earlier left-arm action miss, so final `execution_success` alone does not represent the handover state machine.
- Final MP4: `1920x648`, H.264 High, `yuv420p`, 5 fps, 265 frames/53 s. Full decode and middle-frame visual QA pass.

After the D435-profile rerun, all eight sources verify as `640x480 @ 5 fps`. The flat final video is `vis/handover_bottle_id1_vd435.mp4` and passes H.264 High, `yuv420p`, full-decode, and visual checks.

## Why Canonical AnyGrasp Can Look Static

For `handover_bottle/id1` Orientation/Fused, Canonical did not have zero IK solutions: the right plan succeeded and the left plan failed. Canonical uses `dual_stage_require_all_plans=1`, and the log then explicitly skips the entire stage, leaving a static video. It also uses a `0.12 rad` IK rotation threshold, 20 seeds, physical-RTCP reach, and a 10-degree reach rotation tolerance.

Legacy motion does not prove arrival at the same physical RTCP. Legacy AnyGrasp uses its old `-0.05 m @ local +Z` target, old gripper/endlink semantics, a `3.14 rad` IK rotation threshold, 180-degree reach rotation tolerance, and EE reach. Orientation/Fused also allow `dual_stage_require_all_plans=0`. It can therefore execute a partial/loose solution under a different target and acceptance contract. Canonical Human and Legacy Human look more similar because they share the Human semantic source; the Canonical Human handover state machine actually completes.

IK misses can still arise from unreachable candidate orientations, the strict rotation threshold, or dual-arm gates. Those are independent from the input-semantics wiring error fixed here.
