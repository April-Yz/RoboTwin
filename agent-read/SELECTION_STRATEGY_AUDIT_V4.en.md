# Selection Strategy Audit V4

## Purpose and boundary

V4 is a read-only audit that places OursV2, Orientation, Fused, and Top-score in one traceable visualization. It does not import or invoke a planner, rerun AnyGrasp, or modify legacy algorithms, summaries, JSON, NPZ, PNG, MP4, or execution results.

Code:

```text
code_painting/render_selection_strategy_compare_v4.py
```

Outputs from this run:

```text
code_painting/selection_strategy_compare_v4/
  audit_report.json
  audit_report.zh.md
  <TASK>/foundation_input_<ID>/
    keyframe_<FRAME>_overlay.png
    keyframe_<FRAME>_metadata.json
```

The generated directory remains covered by `code_painting/*`; `.gitignore` only unignores the V4 script.

## What the four records mean

Let \(\mathcal C\) be the AnyGrasp candidates that survive the existing geometry, object, and arm filters. For candidate \(i\), let \(s_i\) be the raw AnyGrasp score and let its full SO(3) distance from the hand target be:

\[
\theta_i=\cos^{-1}\!\left(\operatorname{clip}\left(
\frac{\operatorname{tr}(R_h^\top R_i)-1}{2},-1,1
\right)\right).
\]

The legacy Orientation/Fused preview first applies an independent hard filter:

\[
\mathcal C_{90}=\{i\in\mathcal C\mid\theta_i\leq90^\circ\}.
\]

Orientation chooses the candidate with the closest full rotation:

\[
i_{\mathrm{ori}}=\arg\min_{i\in\mathcal C_{90}}\theta_i.
\]

The implemented orientation score is:

\[
o_i=\operatorname{clip}\left(1-\frac{\theta_i}{180^\circ},0,1\right).
\]

The six-task wrapper uses the raw, unnormalized AnyGrasp score and the actual weights:

\[
F_i=0.25s_i+0.75o_i,\qquad
i_{\mathrm{fused}}=\arg\max_{i\in\mathcal C_{90}}F_i.
\]

V4 does not rerank Orientation or Fused. It reads the saved robot-frame preview rank-1 records so the audit describes historical artifacts, not a new implementation.

Top-score no longer treats `rank_previews/...rank_1.png` as the executed selection. Its sole authoritative source is:

```text
plan_summary.json
  -> selected_candidates_by_executed_arm
```

V4 therefore records the candidate index, score, and resolved frame actually consumed by the planner.

OursV2 is not a fourth AnyGrasp ranking. It creates a synthetic hand-retarget target directly from HaMeR/D435 hand keyframes; `candidate_idx` and `candidate_score` are `null`. A historical synthetic score of 1 is not an AnyGrasp rank.

## Axis remap

The AnyGrasp raw and canonical robot/replay frames satisfy:

\[
\begin{aligned}
x_{robot}&=-z_{raw},\\
y_{robot}&= y_{raw},\\
z_{robot}&= x_{raw}.
\end{aligned}
\]

The right-multiplication is:

\[
R_{canonical}=R_{raw}
\begin{bmatrix}
0&0&1\\
0&1&0\\
-1&0&0
\end{bmatrix}.
\]

Canonical local Y is the finger-opening axis and local Z is the approach/forward axis. Axis colors are fixed to X red, Y green, and Z blue.

Legacy Top-score omitted this remap but applied its \(-0.05\,\mathrm m\) target offset along raw local Z. V4 preserves and displays:

1. the raw AnyGrasp pose;
2. the canonical Selection Pose;
3. the legacy actual Top-score target;
4. the correctly remapped, audit-only canonical target.

Item 4 is never presented as historical execution.

## Two panels and multiple Foundation frames

Panel A, `Selection Pose`, shows the location and orientation after candidate selection or hand retargeting, before the \(-5\) cm offset, OursV2 retreat, pregrasp, world-to-base transform, or TCP-to-link6 compensation.

Panel B, `Planner Target`, shows the target that the historical chain prepared for IK. Metadata records every stage:

- camera-to-world;
- raw-to-canonical remap;
- local offset/retreat;
- 12 cm pregrasp for the first event only;
- world-to-base;
- TCP-to-link6 translation and rotation delta;
- OursV2 task-specific world adjustment.

The current 0515 bundle has `gripper_bias=0.12`, so the recorded TCP-to-link6 local-X translation is:

\[
0.12-\texttt{gripper\_bias}=0.
\]

If a strategy moves a requested frame to the nearest nonempty resolved frame, V4 never projects that pose onto the requested-frame image. Every distinct resolved frame gets its own:

- `color_<resolved_frame>.png` Foundation replay image;
- head-camera world pose;
- camera intrinsics;
- requested, resolved, and delta labels.

Distinct frames are stitched as side-by-side columns; the two rows are Selection and Planner.

## Colors and line styles

- OursV2: cyan;
- Orientation: magenta;
- Fused: yellow;
- canonical Top-score: red;
- raw/legacy Top-score: dark-red/orange dashed;
- pregrasp path: dashed in the strategy color;
- `L` / `R`: executed arm.

## Full results on 2026-07-14

- Six tasks, 25 episodes per task, and 150/150 episodes audited.
- 461 keyframe comparison images.
- 2192 arm-strategy records.
- Zero audit failures.
- 461 PNG, 462 JSON, and one Markdown artifact, about 166 MB.
- The combined SHA-256 of legacy input summaries was unchanged before and after the full run: `345226256cadb99935a0af49e7a95fdc7f72889d21bcda354819e9def0002bd1`.

Actual Top-score versus legacy rank previews:

- actual arm-frame pairs: 600;
- legacy rank-1 image matches the actual candidate: 78/600;
- actual candidate appears in exported top-N: 204/600;
- actual candidate is absent from exported top-N: 396/600.

For `handover_bottle/foundation_input_1/frame 39/right`, the old rank-1 is candidate 13, while V4 reads actual candidate 0 from the plan summary.

All 600 Top-score raw-to-canonical records rotate by 90 degrees. The mean, minimum, and maximum legacy/canonical target position gap are all:

\[
0.0707106781\,\mathrm m=\sqrt{0.05^2+0.05^2}.
\]

This is the geometric result of applying the same \(-5\) cm offset along two orthogonal local-Z axes, not a new TCP fit error. The rotation gap is also 90 degrees.

Top-score requested/resolved frame delta histogram:

```text
-13:2, -2:2, -1:10, 0:557, +1:11, +2:4, +3:1,
+4:2, +5:3, +6:3, +7:1, +8:2, +9:1, +18:1
```

The maximum absolute shift is 18 frames.

There are 208 record gaps, all caused by an empty historical Orientation/Fused candidate list at the requested frame: 104 each. Counts by task are:

```text
handover_bottle 2
pick_diverse_bottles 14
place_bread_basket 140
pnp_bread 10
pnp_tray 32
stack_cups 10
```

For example, at `place_bread_basket/foundation_input_0/requested 64/left`, OursV2 uses frame 64 and Top-score resolves to frame 63; the left Orientation/Fused lists at frame 64 are empty. V4 creates separate frame-64 and frame-63 Foundation columns and reports both gaps.

## Known incomplete OursV2 episodes

- `handover_bottle/foundation_input_6`: `plan_summary_human_replay.json` and `head_cam_plan.mp4` exist, but final `plan_summary.json` is absent; legacy execution ended with `IndexError: list index out of range`.
- `pnp_tray/foundation_input_35`: the human summary and video exist, but final plan summary is absent; legacy execution ended with `KeyError: 'left_dark_red_cup'`.

Neither episode is present in this run's Top-score selected-25 input root, so the 150-episode batch does not generate overlays for them. The global report separately inspects the existence of the human summary, final summary, video, and stderr, retaining the error tail. If a future Top-score input root includes them, V4 writes `execution_complete=false` and a warning into metadata rather than claiming that legacy execution succeeded.

## Limitations

- The Fused Planner Target is a hypothetical reconstruction from the saved preview pose because the historical executed group was not Fused.
- Orientation/Fused remain missing when the requested frame has no candidate. V4 does not invent nearest-frame resolution and thereby alter the historical strategy.
- V4 audits selection and pose transforms; it does not prove IK, collision, or physical grasp success.
- When overlays are crowded, metadata matrices, candidate indices, and frame provenance are authoritative.

See `agent-read/COMMANDS/selection_strategy_audit_v4.en.md` for reproduction commands.
