# Piper Legacy / Canonical IK 2×4 命令

## 参数模板（不可直接运行）

```bash
bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh \
  --task <TASK> \
  --id <FOUNDATION_ID> \
  --arm <auto|left|right> \
  --gpu <GPU_INDEX> \
  --output-root <ABSOLUTE_OUTPUT_ROOT>
```

脚本按顺序运行 Legacy 和 Canonical 两行，每行四列：`orientation`、`fused`、`top_score`、`human_replay`。已有 `head_cam_plan.mp4` 会复用；非空但没有视频的残缺目录会被拒绝，避免混用旧半成品。

## handover_bottle id1 可运行示例

```bash
cd /home/zaijia001/ssd/RoboTwin
bash code_painting/piper_canonical_tcp_v1/run_ik_logic_grid.sh \
  --task handover_bottle \
  --id 1 \
  --arm auto \
  --gpu 0 \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/piper_canonical_tcp_v1/outputs_ik_logic_grid_20260716
```

## 只检查命令和输入路径

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

dry-run 只检查路径并打印八条命令，不创建输出目录，也不启动 planner/仿真。
