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

## Outputs

Default root:

`code_painting/piper_canonical_tcp_v1/outputs_ik_semantic_grid_v2_20260716/`

Important per-episode files:

- `legacy_original_vs_canonical_rtcp_2x4_d435.mp4`: final 2x4 video;
- `semantic_source_audit.json`: source, axes, target-contract, and 19 cm link6 audit;
- `legacy_original_vs_canonical_rtcp_2x4_d435.manifest.json`: source videos, execution state, and media properties;
- `_sources/legacy_original/...` and `_sources/canonical_rtcp/...`: eight cells;
- `_superseded/canonical_human_before_shared_source_fix_20260716/`: preserved intermediate bad sample, excluded from the final video.

## handover_bottle / foundation_input_1 Validation

- All four strategies pass the semantic audit. Every source-position delta is `0.0 m`; rotation-matrix errors range from zero to `4.2e-16`.
- V2 Legacy Orientation versus historical `viewer_gripper`, and V2 Legacy Top-score versus historical `S_graspnet_topscore...`, have zero maximum delta in candidate identity, raw pose, and planner target. The top row now reproduces original input logic.
- Canonical Human completes the internal handover (`[handover] SUCCESS`). The generic summary still returns failure because of an earlier left-arm action miss, so final `execution_success` alone does not represent the handover state machine.
- Final MP4: `1920x648`, H.264 High, `yuv420p`, 5 fps, 265 frames/53 s. Full decode and middle-frame visual QA pass.

IK misses can still arise from unreachable candidate orientations, the strict rotation threshold, or dual-arm gates. Those are independent from the input-semantics wiring error fixed here.
