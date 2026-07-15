# Long-lived Decisions

## 2026-07-14: keep the Dense Replay correction isolated

- Preserve the legacy renderer, runner, and paper assets so historical experiments remain reproducible.
- Name the new implementation `Dense Replay URDF-match v2` and write it under the separate `h2_pure_d435_urdfmatch_v2` output root.
- Keep joint order as `joint1..joint6`; fix the constant error with an explicit frame adapter, not joint swapping or manual joint offsets.
- Interpret the HaMeR fingertip midpoint consistently as TCP; link6 is only an internal IK target frame.
- Dense remains the dense-retargeting baseline. Human orientations unreachable by the robot are not presented as an Ours-v2 capability.
- Six-task batch outputs remain under the same isolated v2 root, with episode-level completeness checks for safe resume; no v1 file is created or overwritten.
- v2 raw replay must not be silently mixed with existing v1 Stage-2 repaint/HDF5 artifacts. A paper diagnostic may display them side by side only with an explicit `NOT V2` label; training-data promotion requires rebuilding the full downstream chain under a new identifier.

## 2026-07-14: keep Selection Strategy V4 read-only

- Preserve legacy OursV2, Orientation, Fused, and Top-score algorithms, summaries, and visualizations; V4 never writes corrections back into historical results.
- Treat `plan_summary.json -> selected_candidates_by_executed_arm` as the actual Top-score selection. Legacy rank previews remain only as evidence of the historical mismatch.
- Preserve raw/legacy Top-score and the canonical reconstruction together. Label the canonical pose audit-only rather than presenting it as historical execution.
- When resolved frames differ, use separate Foundation background columns; never silently project a pose onto another frame.
- Do not displace identical Orientation/Fused poses for visualization. Use thick-solid/thin-dashed lines and distinct markers at the same pose so data semantics remain unchanged.
- Use `<TASK>/id<ID>_keyframe_<FRAME>_*` task outputs without an episode directory; keep the old nested output as a separate rollback backup.
- Keep batch PNG/JSON/reports ignored by Git, and version only the two scripts and bilingual documentation.

## 2026-07-15: PiperCanonicalTCP-v1

- Keep OursV2 fully independent; new Real-TCP semantics live only in `piper_canonical_tcp_v1/`.
- Frame names must distinguish `L6_SIM`, `L6_URDF`, `RTCP`, and `CGRASP`, with explicit world/local axis labels.
- Use the runtime same-q exact signed-axis matrix for `T_L6SIM_L6URDF`. Keep literal server `-1.57` and `0.19` instead of substituting an ideal angle.
- Orientation/Fused convert canonical preview axes back to raw/RTCP; Top-score raw source uses identity. The two numerically identical 90-degree matrices retain separate semantics.
- Continue the batch after a strategy IK miss and record failure. Videos may be composed, but no failed strategy receives a SUCCESS marker.
- Version code and tests; keep smoke, batch videos, logs, and large artifacts ignored.
- Canonical generated videos use the VS Code/Chromium-decodable contract: H.264, `yuv420p`, and faststart. OpenCV-readable does not imply browser-compatible; validate temporary format, geometry/frame count, and full decode before replacement.
- Video provenance must distinguish raw/preview D435 input, a simulated head camera driven by D435 calibration, and simulated third/wrist/composed views. A `d435` path component alone does not make every MP4 raw D435 footage.
