# Current Feature Summary

## Current main line

- The repository default remains the latest v1.x iteration. Foundation, AnyGrasp, OursV2, Dense Replay, Piper IK V3, and PiperCanonicalTCP-v1 are separate experiment lines.
- `PiperCanonicalTCP-v1` is the current Real-Piper-TCP comparison entry and does not modify OursV2.

## Added in this change

- Planner targets, current readback, reach checks, and visualization all use `T_W_RTCP`.
- SAPIEN `L6_SIM` and CuRobo/server `L6_URDF` share an origin but differ by exact local-axis `Ry(+pi/2)`. Adapted same-q FK error is below `7.5e-8 m / 0.000016 deg`.
- The server tool remains literal `T_L6URDF_RTCP = Ry(-1.57) @ Tx(0.19)`. Preview `CGRASP -> RTCP` remapping is a separate transform.
- Corrected same-q OursV2 TCP versus Real TCP has about `70.0001 mm` mean/max distance on both arms: the 12 cm versus 19 cm difference along the shared forward axis. The old 224.6 mm conclusion is invalid.
- EE-pose comparison supports Orientation, Fused, and Top-score. All three strategies plus corrected joint comparison pass media, ffprobe, and visual QA on `pnp_bread/id8/left`.
- A 6 tasks x 5 episodes manifest, isolated batch orchestrator, and tmux commands are ready. Strategy IK misses preserve videos and failure TSV entries without fake SUCCESS markers.
- Selection Strategy Audit V4 and Dense Replay URDF-match v2 remain separate historical lines.

## Reading order

1. `README.en.md`
2. `CURRENT_FEATURE_SUMMARY.en.md`
3. `VERSION_SUMMARY.en.md`
4. `PIPER_CANONICAL_TCP_V1.en.md`
5. `COMMANDS/piper_canonical_tcp_v1.en.md`
6. `SELECTION_STRATEGY_AUDIT_V4.en.md`
