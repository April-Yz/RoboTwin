# Piper Legacy-original / Canonical-RTCP 2×4 命令

## 单集参数模板（不可直接运行）

```bash
bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh \
  --task <TASK> \
  --id <FOUNDATION_ID> \
  --arm <auto|left|right> \
  --gpu <GPU_INDEX> \
  --camera-profile <d435|wide> \
  --output-root <ABSOLUTE_OUTPUT_ROOT>
```

脚本顺序运行 `legacy_original` 和 `canonical_rtcp`，每行四列：`orientation`、`fused`、`top_score`、`human_replay`。`d435` 固定为 `640×480 / fovy 42.499880046655484° / 5 fps`，`wide` 固定为 `640×360 / fovy 90° / 10 fps`。已有完整源视频会复用；非空但无主视频的残缺目录会被拒绝。

## D435 单集可运行示例

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

结果为 `vis/handover_bottle_id1_vd435.mp4`。

## 广角单集可运行示例

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

结果为 `vis/handover_bottle_id1_vwide.mp4`。

## 6×1×2 批处理模板（不可直接运行）

```bash
bash code_painting/piper_canonical_tcp_v1/run_ik_semantic_camera_batch.sh \
  --output-root <ABSOLUTE_OUTPUT_ROOT> \
  --gpu <GPU_INDEX> \
  --profile <d435|wide|both>
```

固定样本为 `pick_diverse_bottles/id0`、`place_bread_basket/id0`、`stack_cups/id0`、`handover_bottle/id1`、`pnp_bread/id7`、`pnp_tray/id0`。

## 6×1×2 tmux 可运行示例

```bash
cd /home/zaijia001/ssd/RoboTwin
tmux new-session -d -s pcanonical_camprofiles_6x1x2 \
  "bash code_painting/piper_canonical_tcp_v1/run_ik_semantic_camera_batch.sh --output-root /home/zaijia001/ssd/RoboTwin/code_painting/piper_canonical_tcp_v1/outputs_ik_semantic_grid_v2_20260716 --gpu 0 --profile both > /home/zaijia001/ssd/RoboTwin/code_painting/piper_canonical_tcp_v1/outputs_ik_semantic_grid_v2_20260716/_batch/camera_profiles_batch.log 2>&1"
```

状态表为 `_batch/camera_profiles_status.tsv`，全部结束后写 `_batch/SUCCESS`。脚本按 D435 六集、wide 六集顺序自动推进，不需要持续监控。

## 只检查命令与路径

```bash
cd /home/zaijia001/ssd/RoboTwin
bash code_painting/piper_canonical_tcp_v1/run_ik_semantic_camera_batch.sh \
  --output-root /tmp/piper_ik_semantic_grid_v2_dryrun \
  --gpu 0 \
  --profile both \
  --dry-run
```

dry-run 验证路径并打印 12 个 grid / 96 个 cell 命令，不启动 planner/仿真，也不创建输出目录。
