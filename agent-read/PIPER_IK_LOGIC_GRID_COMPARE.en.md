# Piper Legacy / Canonical IK 2x4 Comparison

## Purpose

This comparison answers one isolated question: with identical candidate targets, does the Legacy/OursV2 versus PiperCanonicalTCP-v1 `T_W_RTCP -> URDF link6` conversion produce different IK and execution results?

The output is a two-row-by-four-column D435-calibrated simulated view:

| | Orientation | Fused | Top-score | Human Replay |
|---|---|---|---|---|
| Top row | Legacy IK | Legacy IK | Legacy IK | Legacy IK |
| Bottom row | Canonical IK | Canonical IK | Canonical IK | Canonical IK |

The earlier `canonical_vs_legacy_five_method_d435.mp4` is not this full ablation. It contains four Canonical methods plus only one Legacy Human Replay panel, with no Legacy Orientation/Fused/Top-score panels.

## Actual IK Input

All eight cells use the direct `Selection Pose`, normalized to Real TCP in world coordinates: `T_W_RTCP`.

- Orientation: direct Orientation candidate from the robot-frame preview, with one `CGRASP -> RTCP` axis mapping.
- Fused: direct candidate selected by `0.25 x AnyGrasp score + 0.75 x orientation score`, with the same axis mapping.
- Top-score: direct native highest-score AnyGrasp candidate transformed D435 camera -> world; native axes are interpreted as RTCP without the CGRASP mapping.
- Human Replay: direct human-gripper pose transformed D435 camera -> world, followed by one `CGRASP_HUMAN -> RTCP` axis mapping.

Every method forces:

- final target retreat: `0 m`;
- candidate local X/Z offsets: `0 m`;
- pregrasp: derived only after selection, `0.12 m` backward on local RTCP `+X`;
- reach/readback frame: physical TCP;
- position frame: 0515 `WORLD`; orientation axes: local `RTCP`.

The upper `Selection Pose` in `selection_strategy_compare_v4` is the direct-input source. The lower `Planner Target` already includes historical offset, retreat, pregrasp, world/base, and TCP/link6 processing. It is deliberately excluded to prevent duplicate compensation.

## Only Row-specific Semantic Difference

- Legacy row: `HandRetargetPiperDualURDFIKRenderer` using `robot._trans_from_gripper_to_endlink(...)`.
- Canonical row: `PiperCanonicalTCPRenderer` using the exact inverse of Piper server `T_L6URDF_RTCP = Ry(-1.57) @ Tx(0.19)`.

Candidate selection, numeric target, pregrasp, IK thresholds, seeds, trajectory, and reach gates are identical between the two cells of each column. Before composition, `audit_ik_logic_inputs.py` compares candidate arm, frame, index, and `pose_world_wxyz`; any mismatch blocks the grid video.

## Legacy Human Replay 12/14 cm Note

The formal old Human Replay `plan_summary_human_replay.json` records `target_retreat=0.14 m @ local human +Z`. The earlier five-way comparison used `0.12 m` as an explicit ablation and is therefore not an exact parameter replay of the formal old result. Both rows of the new 2x4 force `target_retreat=0`, preventing historical retreat from stacking with renderer TCP/link6 conversion.

## Outputs

Default root:

`code_painting/piper_canonical_tcp_v1/outputs_ik_logic_grid_20260716/`

Important per-episode files:

- `legacy_vs_canonical_ik_logic_2x4_d435.mp4`: main 2x4 video;
- `input_equality_audit.json`: cross-row input equality audit;
- `legacy_vs_canonical_ik_logic_2x4_d435.manifest.json`: source videos, execution status, media properties, and input contract;
- `_sources/legacy/...` and `_sources/canonical/...`: eight independent method directories;
- per-method `input_target_contract.json`: cell input and row-specific conversion.

An IK/reach miss is not itself a runner failure. If a diagnostic video exists, the grid runner preserves that cell and continues. A real pipeline error means no video, failed input audit, or failed composition.

## handover_bottle id1 Smoke Result

- Each column has two targets per row. Arm/frame/candidate index match exactly and every maximum absolute `pose_world_wxyz` delta is `0.0`.
- Orientation and Fused select the same targets on this ID. Neither IK row finds an executable strict solution; left/right q and TCP change are both zero.
- Legacy Top-score moves, with about `434/367 mm` maximum left/right TCP displacement, but minimum grasp position errors remain about `169/168 mm`. Canonical Top-score finds no executable solution and does not move.
- Legacy Human Replay moves, with about `516/613 mm` maximum left/right TCP displacement, but retains about `120 mm` target residual. This directly exposes the legacy 12 cm end-definition mismatch rather than arrival at the same physical RTCP.
- Canonical Human Replay changes only about `19.4 mm`, while target position errors remain hundreds of millimeters. The direct human RTCP orientation is not executable under the strict `0.12 rad` rotation threshold and dual-arm all-plan gate.
- All eight cells finish with `execution_failed=true`; the video remains a failure-mode/frame-semantics diagnostic, not a successful behavior showcase.
- The main video is `1920x648`, H.264, `yuv420p`, 5 fps, 644 frames/128.8 s. Full decode and middle-frame visual inspection pass.
