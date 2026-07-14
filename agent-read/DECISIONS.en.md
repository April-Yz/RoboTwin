# Long-lived Decisions

## 2026-07-14: keep the Dense Replay correction isolated

- Preserve the legacy renderer, runner, and paper assets so historical experiments remain reproducible.
- Name the new implementation `Dense Replay URDF-match v2` and write it under the separate `h2_pure_d435_urdfmatch_v2` output root.
- Keep joint order as `joint1..joint6`; fix the constant error with an explicit frame adapter, not joint swapping or manual joint offsets.
- Interpret the HaMeR fingertip midpoint consistently as TCP; link6 is only an internal IK target frame.
- Dense remains the dense-retargeting baseline. Human orientations unreachable by the robot are not presented as an Ours-v2 capability.
