# Piper Legacy-original / Canonical-RTCP 2x4 Commands

## Parameter Template (Not Directly Runnable)

```bash
bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh \
  --task <TASK> \
  --id <FOUNDATION_ID> \
  --arm <auto|left|right> \
  --gpu <GPU_INDEX> \
  --output-root <ABSOLUTE_OUTPUT_ROOT>
```

The script runs `legacy_original` and `canonical_rtcp` sequentially. Each row contains `orientation`, `fused`, `top_score`, and `human_replay`. A complete existing `head_cam_plan.mp4` is reused; a nonempty incomplete directory without the head video is rejected.

## Runnable Example

```bash
cd /home/zaijia001/ssd/RoboTwin
bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh \
  --task handover_bottle \
  --id 1 \
  --arm auto \
  --gpu 0 \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/piper_canonical_tcp_v1/outputs_ik_semantic_grid_v2_20260716
```

## tmux Example

```bash
cd /home/zaijia001/ssd/RoboTwin
tmux new-session -d -s pcanonical_semantic_v2_handover_id1 \
  "bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh --task handover_bottle --id 1 --arm auto --gpu 0"
```

## Command and Path Check Only

```bash
cd /home/zaijia001/ssd/RoboTwin
bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh \
  --task handover_bottle \
  --id 1 \
  --arm auto \
  --gpu 0 \
  --output-root /tmp/piper_ik_semantic_grid_v2_dryrun \
  --dry-run
```

Dry-run validates paths and prints all eight commands without launching the planner/simulation or creating an output directory.
