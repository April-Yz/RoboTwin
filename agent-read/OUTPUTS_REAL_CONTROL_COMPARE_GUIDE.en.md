# Real-control Comparison Output Quick Reference

Applies to `code_painting/piper_canonical_tcp_v1/outputs_real_control_compare_20260716/<task>/<episode>/`.

## Files

| File | Meaning |
|---|---|
| `joint_control.mp4` | Feeds the same measured Piper `q1..q6` to all definitions and compares TCP positions. |
| `eepose_control.mp4` | Feeds one measured `T_B_RTCP` to OursV2 and Canonical and compares achieved physical RTCP. |
| `*.manifest.json` | Codec, frames, inputs, and frame semantics for the video. |
| `control_plan.npz` | Synchronized q/endPose, FK/IK results, and IK masks used by the plots. |
| `summary.json` | Frame counts, per-arm IK success, and position-error summaries. |
| `frame_contract.json` | Explicit world/base/link6/RTCP definitions and local-axis labels. |
| `oursv2_renderer.npz` / `canonical_renderer.npz` | Renderer and qpos provenance for each simulation branch. |
| `sim_direct/`, `sim_oursv2/`, `sim_canonical/` | Upper-panel frame caches, not additional real data. |
| `SUCCESS` / `EXIT_CODE` | Complete-success marker / failed return code. |

## Upper Panels

`joint_control.mp4`: real D435 with three TCP definitions, real left wrist, real right wrist, and calibrated RoboTwin driven by the same measured q.

`eepose_control.mp4`: real D435 plus Real-TCP target, real-q simulation reference, OursV2 legacy EE-pose IK, and Canonical server-semantic IK. The axes on the second panel are target axes, not another q-FK curve.

## Curves

Columns are arms, rows are Piper0515 world X/Y/Z, the horizontal axis is synchronized D435 frame, and the red vertical cursor is the current frame.

| Curve | Color | Meaning |
|---|---|---|
| Piper real | Black | Recorded endPose / Real RTCP. |
| OursV2 | Cyan | 12 cm OursV2 TCP in Joint; achieved physical RTCP after legacy IK in EE-pose. Failed IK frames are omitted. |
| Canonical | Magenta | RTCP from server `Ry(-1.57) @ Tx(0.19)` in Joint; achieved physical RTCP after Canonical IK in EE-pose. |

If only black and cyan appear visible, magenta normally covers an almost identical black curve. Check `summary.json` and the manifest.

Local pose axes are `+X` red, `+Y` green, `+Z` blue. Plot coordinates are always Piper0515 world XYZ.

Fast check: inspect `SUCCESS`/`EXIT_CODE`, then `summary.json`, then plot gaps at the red cursor. Use `control_plan.npz` for precise numerical verification.
