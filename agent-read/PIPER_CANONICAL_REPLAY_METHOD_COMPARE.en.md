# Canonical Four-method and Legacy Human Replay Comparison

Each sample produces two D435-calibrated simulation-view videos:

- `canonical_four_method_d435.mp4`: Canonical Orientation, Fused, Top-score, and Human Replay.
- `canonical_vs_legacy_five_method_d435.mp4`: those four plus a Legacy OursV2 Human Replay retreat baseline.

The first four use `T_W_RTCP`, server `Ry(-1.57) @ Tx(0.19)`, and Canonical IK. The fifth only exposes legacy local-axis/retreat semantics and is not Canonical.

| Canonical method | Position | Orientation |
|---|---|---|
| Orientation | AnyGrasp candidate origin | Nearest human orientation; `CGRASP -> RTCP`. |
| Fused | AnyGrasp candidate origin | 0.25 native score + 0.75 orientation; `CGRASP -> RTCP`. |
| Top-score | AnyGrasp candidate origin | Highest native score, interpreted with RTCP axes. |
| Human Replay | Human gripper origin | Explicit human/CGRASP local-axis to RTCP mapping. |

All Canonical final-target offsets are zero. `approach_offset_m=0.12` creates only the pregrasp point.

Raw AnyGrasp translation is in the D435 camera frame, not world. It becomes a candidate origin only after D435 camera-to-world extrinsics. Orientation/Fused change local axes without moving the origin; Top-score adds no position offset.

Canonical Human Replay forces `target_retreat_m=0`. The current legacy wrapper also defaults to zero, while the historical link6-compensation ablation explicitly uses `target_retreat_m=0.12` on legacy human local `+Z`. Do not remove Canonical `approach_offset_m=0.12`; only a legacy final-target retreat may need removal.

Outputs live under `code_painting/piper_canonical_tcp_v1/outputs_replay_method_compare_20260716/<task>/foundation_input_<id>/`; `_sources/` is isolated from old outputs. The D435 view is a calibrated simulated head view reconstructed from D435 replay and Piper0515 calibration, not duplicated raw RGB.
