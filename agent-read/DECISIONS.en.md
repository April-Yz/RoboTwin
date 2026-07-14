# Long-lived Decisions

## 2026-07-14: keep the Dense Replay correction isolated

- Preserve the legacy renderer, runner, and paper assets so historical experiments remain reproducible.
- Name the new implementation `Dense Replay URDF-match v2` and write it under the separate `h2_pure_d435_urdfmatch_v2` output root.
- Keep joint order as `joint1..joint6`; fix the constant error with an explicit frame adapter, not joint swapping or manual joint offsets.
- Interpret the HaMeR fingertip midpoint consistently as TCP; link6 is only an internal IK target frame.
- Dense remains the dense-retargeting baseline. Human orientations unreachable by the robot are not presented as an Ours-v2 capability.

## 2026-07-14: keep Selection Strategy V4 read-only

- Preserve legacy OursV2, Orientation, Fused, and Top-score algorithms, summaries, and visualizations; V4 never writes corrections back into historical results.
- Treat `plan_summary.json -> selected_candidates_by_executed_arm` as the actual Top-score selection. Legacy rank previews remain only as evidence of the historical mismatch.
- Preserve raw/legacy Top-score and the canonical reconstruction together. Label the canonical pose audit-only rather than presenting it as historical execution.
- When resolved frames differ, use separate Foundation background columns; never silently project a pose onto another frame.
- Keep batch PNG/JSON/reports ignored by Git, and version only the script and bilingual documentation.
