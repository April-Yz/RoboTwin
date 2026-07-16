# PiperCanonicalTCP-v1 Commands

Working directory: `/home/zaijia001/ssd/RoboTwin`

## Parameter template (documentation only)

```bash
code_painting/piper_canonical_tcp_v1/run_eepose_strategy.sh \
  --task <TASK> --id <EPISODE_ID> \
  --arm <auto|left|right> \
  --strategy <orientation|fused|top_score> \
  --gpu <GPU_ID> --output-root <NEW_OUTPUT_ROOT>
```

`auto` executes the arms selected by preview/task metadata. The runner refuses a non-empty output directory that lacks `SUCCESS`.

## Test and dry run

```bash
cd /home/zaijia001/ssd/RoboTwin
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  tests/test_piper_canonical_tcp_v1.py

code_painting/piper_canonical_tcp_v1/run_eepose_strategy.sh \
  --task pnp_bread --id 8 --arm left --strategy orientation \
  --gpu 0 --output-root code_painting/piper_canonical_tcp_v1/dry_run \
  --dry-run
```

## Verified single episode

```bash
cd /home/zaijia001/ssd/RoboTwin
for STRATEGY in orientation fused top_score; do
  code_painting/piper_canonical_tcp_v1/run_eepose_strategy.sh \
    --task pnp_bread --id 8 --arm left --strategy "$STRATEGY" \
    --gpu 0 \
    --output-root code_painting/piper_canonical_tcp_v1/smoke_all_pass
done
```

```bash
cd /home/zaijia001/ssd/RoboTwin
EP=code_painting/piper_canonical_tcp_v1/smoke_all_pass/pnp_bread/foundation_input_8
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  code_painting/piper_canonical_tcp_v1/compose_strategy_video.py \
  --episode-root "$EP" --output "$EP/eepose/strategy_comparison.mp4"
```

## Same-q joint comparison

```bash
cd /home/zaijia001/ssd/RoboTwin
SRC=code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_de_human_replay_clean_right_cam/pnp_bread/foundation_input_8
OUT=code_painting/piper_canonical_tcp_v1/smoke_all_pass/pnp_bread/foundation_input_8/joint_control
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  code_painting/piper_canonical_tcp_v1/compare_joint_control.py \
  --pose-debug "$SRC/pose_debug.jsonl" \
  --source-video "$SRC/head_cam_plan.mp4" \
  --robot-config robot_config_PiperPika_agx_dual_table_0515.json \
  --output-dir "$OUT" --task pnp_bread --episode-id 8
```

## 6 tasks x 5 episodes in tmux

Parameter template:

```bash
tmux new-session -d -s <SESSION_NAME> \
  "cd /home/zaijia001/ssd/RoboTwin && \
   code_painting/piper_canonical_tcp_v1/run_batch.sh \
     --mode <joint|eepose> --gpu <GPU_ID> \
     2>&1 | tee code_painting/piper_canonical_tcp_v1/outputs/_batch_logs/<LOG>.log"
```

Runnable commands:

```bash
cd /home/zaijia001/ssd/RoboTwin
mkdir -p code_painting/piper_canonical_tcp_v1/outputs_canonical_20260715/_batch_logs

tmux new-session -d -s pcan_v1_joint_6x5 \
  "cd /home/zaijia001/ssd/RoboTwin && \
   code_painting/piper_canonical_tcp_v1/run_batch.sh --mode joint --gpu 0 \
     --output-root code_painting/piper_canonical_tcp_v1/outputs_canonical_20260715 \
   2>&1 | tee code_painting/piper_canonical_tcp_v1/outputs_canonical_20260715/_batch_logs/joint_tmux.log"

tmux new-session -d -s pcan_v1_eepose_6x5 \
  "cd /home/zaijia001/ssd/RoboTwin && \
   code_painting/piper_canonical_tcp_v1/run_batch.sh --mode eepose --gpu 0 \
     --output-root code_painting/piper_canonical_tcp_v1/outputs_canonical_20260715 \
   2>&1 | tee code_painting/piper_canonical_tcp_v1/outputs_canonical_20260715/_batch_logs/eepose_tmux.log"
```

One-shot status read:

```bash
tmux list-sessions | grep 'pcan_v1_'
tmux capture-pane -pt pcan_v1_joint_6x5 -S -30
tmux capture-pane -pt pcan_v1_eepose_6x5 -S -30
```

## VS Code-compatible MP4 audit and transcode

Parameter template (read-only dry run; documentation only):

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  code_painting/piper_canonical_tcp_v1/vscode_video.py \
  --root <OUTPUT_ROOT> --workers <WORKER_COUNT> \
  --manifest <MANIFEST_JSON>
```

Atomically convert the 2026-07-15 canonical outputs:

```bash
cd /home/zaijia001/ssd/RoboTwin
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  code_painting/piper_canonical_tcp_v1/vscode_video.py \
  --root code_painting/piper_canonical_tcp_v1/outputs_canonical_20260715 \
  --apply --workers 4 \
  --manifest code_painting/piper_canonical_tcp_v1/outputs_canonical_20260715/_batch_logs/vscode_transcode_manifest.json
```

The tool converts only non-`h264/yuv420p` MP4 files. A temporary H.264 file must pass ffprobe, geometry/frame-count checks, and full-frame decode before it atomically replaces the original path. The manifest records before/after formats, sizes, and SHA-256 values.

## Real control compare

Parameter template (documentation only; replace placeholders):

```bash
code_painting/piper_canonical_tcp_v1/run_real_control_compare.sh \
  --task <RAW_TASK> --episode <episodeN> --gpu <GPU_ID> \
  --max-frames <0_FOR_FULL_EPISODE> --output-root <NEW_OUTPUT_ROOT>
```

Passing eight-frame smoke:

```bash
cd /home/zaijia001/ssd/RoboTwin
code_painting/piper_canonical_tcp_v1/run_real_control_compare.sh \
  --task handover_bottle --episode episode0 --gpu 2 --max-frames 8 \
  --output-root code_painting/piper_canonical_tcp_v1/outputs_real_control_smoke_20260716
```

Start an isolated tmux for the 30 audited raw episodes (six tasks x five). Each episode automatically produces both joint and EE-pose videos:

```bash
cd /home/zaijia001/ssd/RoboTwin
mkdir -p code_painting/piper_canonical_tcp_v1/outputs_real_control_compare_20260716/_batch_logs
tmux new-session -d -s pcan_realctrl_30 \
  "cd /home/zaijia001/ssd/RoboTwin && \
   code_painting/piper_canonical_tcp_v1/run_real_control_compare_batch.sh --gpu 2 \
     --output-root code_painting/piper_canonical_tcp_v1/outputs_real_control_compare_20260716 \
   2>&1 | tee code_painting/piper_canonical_tcp_v1/outputs_real_control_compare_20260716/_batch_logs/tmux.log"
```

## Canonical Four-method + Legacy Human Replay

Non-runnable parameter template; retreat is deliberately explicit:

```bash
code_painting/piper_canonical_tcp_v1/run_replay_method_compare.sh \
  --task <TASK> --id <FOUNDATION_ID> --arm <auto|left|right> --gpu <GPU> \
  --canonical-root <EXISTING_CANONICAL_ROOT> --output-root <NEW_OUTPUT_ROOT> \
  --legacy-target-retreat-m <0_OR_0.12>
```

Runnable historical 12 cm retreat comparison:

```bash
cd /home/zaijia001/ssd/RoboTwin
code_painting/piper_canonical_tcp_v1/run_replay_method_compare.sh \
  --task handover_bottle --id 1 --arm auto --gpu 2 \
  --canonical-root code_painting/piper_canonical_tcp_v1/outputs_canonical_20260715 \
  --output-root code_painting/piper_canonical_tcp_v1/outputs_replay_method_compare_20260716 \
  --legacy-target-retreat-m 0.12
```

Batch template:

```bash
code_painting/piper_canonical_tcp_v1/run_replay_method_compare_batch.sh \
  --manifest <TASK_ID_ARM_TSV> --gpu <GPU> \
  --canonical-root <EXISTING_CANONICAL_ROOT> --output-root <NEW_OUTPUT_ROOT> \
  --legacy-target-retreat-m <0_OR_0.12>
```
