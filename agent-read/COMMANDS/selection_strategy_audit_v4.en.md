# Selection Strategy Audit V4 Commands

## Purpose

Read-only comparison of OursV2, Orientation, Fused, and the actual Top-score pose, producing Selection/Planner PNG panels, per-keyframe metadata, and a batch report.

## Non-runnable template

This block explains parameters and is not runnable as written:

```bash
cd /home/zaijia001/ssd/RoboTwin
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  code_painting/render_selection_strategy_compare_v4.py \
  --tasks <TASK> \
  --ids <FOUNDATION_ID> \
  --requested-frames <KEYFRAME> \
  --output-root <NEW_EMPTY_OUTPUT_ROOT>
```

- `--tasks`: one or more tasks; omitting it selects all six tasks.
- `--ids`: one or more Foundation IDs; omitting it selects every discovered ID.
- `--requested-frames`: optional filter for requested frames.
- `--output-root`: use a new empty directory; nonempty roots are rejected by default.
- `--approach-offset-m`: first-event pregrasp audit display, default 0.12 m.
- `--axis-length-m`: rendered coordinate-axis length, default 0.045 m.

Each task directory directly stores `id<ID>_keyframe_<FRAME>_{overlay,metadata}`; there is no intermediate `foundation_input_<ID>/` directory.

## Runnable smoke: corrected Top-score candidate

```bash
cd /home/zaijia001/ssd/RoboTwin
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  code_painting/render_selection_strategy_compare_v4.py \
  --tasks handover_bottle \
  --ids 1 \
  --requested-frames 39 \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/selection_strategy_compare_v4_handover_1_20260714
```

Expected: right Top-score `candidate_idx=0`, not candidate 13 from the legacy rank-1 image.

## Runnable smoke: distinct Foundation-frame mosaic

```bash
cd /home/zaijia001/ssd/RoboTwin
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  code_painting/render_selection_strategy_compare_v4.py \
  --tasks place_bread_basket \
  --ids 0 \
  --requested-frames 64 \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/selection_strategy_compare_v4_place_bread_0_20260714
```

Expected: OursV2 uses Foundation frame 64 and Top-score uses resolved frame 63 in separate side-by-side columns.

## Runnable full batch

The 2026-07-14 full result already exists at `code_painting/selection_strategy_compare_v4/`. Use a new directory for another run:

```bash
cd /home/zaijia001/ssd/RoboTwin
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  code_painting/render_selection_strategy_compare_v4.py \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/selection_strategy_compare_v4_rerun_20260714
```

## Validation

### Non-runnable agreement-statistics template

```bash
cd /home/zaijia001/ssd/RoboTwin
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  code_painting/analyze_selection_strategy_agreement_v4.py \
  --audit-root <AUDIT_OUTPUT_ROOT>
```

### Current full-batch statistics

```bash
cd /home/zaijia001/ssd/RoboTwin
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  code_painting/analyze_selection_strategy_agreement_v4.py \
  --audit-root /home/zaijia001/ssd/RoboTwin/code_painting/selection_strategy_compare_v4
```

This writes `strategy_agreement_stats.json`, `.zh.md`, and `.en.md`. Left and right count separately; results include Fused–Orientation and Fused–canonical-Top candidate agreement, xyz distances, and weighted Fused-score contributions. It also pairs Orientation, Fused, and canonical Top Selection Poses with the OursV2 direct hand-replay Selection Pose, reporting Euclidean world-xyz distance, signed `AnyGrasp - OursV2` components, and separate same-frame versus cross-frame Top results.

### Syntax and output validation

```bash
cd /home/zaijia001/ssd/RoboTwin
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 -m py_compile \
  code_painting/render_selection_strategy_compare_v4.py \
  code_painting/analyze_selection_strategy_agreement_v4.py
```

Inspect `failures`, `record_gaps`, and coverage in `audit_report.json`. Metadata is authoritative when an overlay is crowded.

`--overwrite` only permits explicit regeneration of V4-owned outputs; it never authorizes changes to legacy strategy directories. Prefer a new output root for normal use.
