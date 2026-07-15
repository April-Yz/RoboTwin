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
