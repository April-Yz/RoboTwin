# Piper Legacy-original / Canonical-RTCP 2x4 Commands

## Single-episode Parameter Template (Not Directly Runnable)

```bash
bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh \
  --task <TASK> \
  --id <FOUNDATION_ID> \
  --arm <auto|left|right> \
  --gpu <GPU_INDEX> \
  --camera-profile <d435|wide> \
  --output-root <ABSOLUTE_OUTPUT_ROOT>
```

The script runs `legacy_original` and `canonical_rtcp` sequentially. Each row contains `orientation`, `fused`, `top_score`, and `human_replay`. D435 is fixed to `640x480 / fovy 42.499880046655484deg / 5 fps`; wide is fixed to `640x360 / fovy 90deg / 10 fps`. Complete existing source videos are reused, while a nonempty incomplete directory without the main video is rejected.

## Runnable D435 Single-episode Example

```bash
cd /home/zaijia001/ssd/RoboTwin
bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh \
  --task handover_bottle \
  --id 1 \
  --arm auto \
  --gpu 0 \
  --camera-profile d435 \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/piper_canonical_tcp_v1/outputs_ik_semantic_grid_v2_20260716
```

The result is `vis/handover_bottle_id1_vd435.mp4`.

## Runnable Wide Single-episode Example

```bash
cd /home/zaijia001/ssd/RoboTwin
bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh \
  --task handover_bottle \
  --id 1 \
  --arm auto \
  --gpu 0 \
  --camera-profile wide \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/piper_canonical_tcp_v1/outputs_ik_semantic_grid_v2_20260716
```

The result is `vis/handover_bottle_id1_vwide.mp4`.

## 6x1x2 Batch Template (Not Directly Runnable)

```bash
bash code_painting/piper_canonical_tcp_v1/run_ik_semantic_camera_batch.sh \
  --output-root <ABSOLUTE_OUTPUT_ROOT> \
  --gpu <GPU_INDEX> \
  --profile <d435|wide|both>
```

Fixed samples are `pick_diverse_bottles/id0`, `place_bread_basket/id0`, `stack_cups/id0`, `handover_bottle/id1`, `pnp_bread/id7`, and `pnp_tray/id0`.

## Runnable 6x1x2 tmux Example

```bash
cd /home/zaijia001/ssd/RoboTwin
tmux new-session -d -s pcanonical_camprofiles_6x1x2 \
  "bash code_painting/piper_canonical_tcp_v1/run_ik_semantic_camera_batch.sh --output-root /home/zaijia001/ssd/RoboTwin/code_painting/piper_canonical_tcp_v1/outputs_ik_semantic_grid_v2_20260716 --gpu 0 --profile both > /home/zaijia001/ssd/RoboTwin/code_painting/piper_canonical_tcp_v1/outputs_ik_semantic_grid_v2_20260716/_batch/camera_profiles_batch.log 2>&1"
```

The status table is `_batch/camera_profiles_status.tsv`; `_batch/SUCCESS` is written after completion. The script advances through six D435 samples and six wide samples sequentially without continuous monitoring.

## Command and Path Check Only

```bash
cd /home/zaijia001/ssd/RoboTwin
bash code_painting/piper_canonical_tcp_v1/run_ik_semantic_camera_batch.sh \
  --output-root /tmp/piper_ik_semantic_grid_v2_dryrun \
  --gpu 0 \
  --profile both \
  --dry-run
```

Dry-run validates paths and prints 12 grid / 96 cell commands without launching the planner/simulation or creating an output directory.
