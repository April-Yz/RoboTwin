# Piper Legacy / Canonical IK 2x4 Commands

## Parameter Template (Not Directly Runnable)

```bash
bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh \
  --task <TASK> \
  --id <FOUNDATION_ID> \
  --arm <auto|left|right> \
  --gpu <GPU_INDEX> \
  --output-root <ABSOLUTE_OUTPUT_ROOT>
```

The script runs Legacy and Canonical rows sequentially. Each row contains `orientation`, `fused`, `top_score`, and `human_replay`. An existing `head_cam_plan.mp4` is reused. A nonempty incomplete directory without a head video is rejected to avoid mixing partial old output.

## Runnable handover_bottle id1 Example

```bash
cd /home/zaijia001/ssd/RoboTwin
bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh \
  --task handover_bottle \
  --id 1 \
  --arm auto \
  --gpu 0 \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/piper_canonical_tcp_v1/outputs_ik_logic_grid_20260716
```

## Input-path and Command Check Only

```bash
cd /home/zaijia001/ssd/RoboTwin
bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh \
  --task handover_bottle \
  --id 1 \
  --arm auto \
  --gpu 0 \
  --output-root /tmp/piper_ik_logic_grid_dryrun \
  --dry-run
```

Dry-run only validates paths and prints all eight commands. It creates no output directory and does not launch the planner/simulation.
