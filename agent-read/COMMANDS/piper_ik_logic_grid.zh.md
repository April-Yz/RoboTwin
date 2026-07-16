# Piper Legacy-original / Canonical-RTCP 2×4 命令

## 参数模板（不可直接运行）

```bash
bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh \
  --task <TASK> \
  --id <FOUNDATION_ID> \
  --arm <auto|left|right> \
  --gpu <GPU_INDEX> \
  --output-root <ABSOLUTE_OUTPUT_ROOT>
```

脚本顺序运行 `legacy_original` 和 `canonical_rtcp`，每行四列：`orientation`、`fused`、`top_score`、`human_replay`。已有完整 `head_cam_plan.mp4` 会复用；非空但无主视频的残缺目录会被拒绝。

## 可运行示例

```bash
cd /home/zaijia001/ssd/RoboTwin
bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh \
  --task handover_bottle \
  --id 1 \
  --arm auto \
  --gpu 0 \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/piper_canonical_tcp_v1/outputs_ik_semantic_grid_v2_20260716
```

## tmux 示例

```bash
cd /home/zaijia001/ssd/RoboTwin
tmux new-session -d -s pcanonical_semantic_v2_handover_id1 \
  "bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh --task handover_bottle --id 1 --arm auto --gpu 0"
```

## 只检查命令与路径

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

dry-run 只验证路径并打印 8 条命令，不启动 planner/仿真，也不创建输出目录。
