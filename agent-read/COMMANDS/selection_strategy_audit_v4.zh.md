# Selection Strategy Audit V4 命令

## 用途

只读比较 OursV2、Orientation、Fused 和真实 Top-score pose，生成 Selection/Planner 双面板 PNG、逐关键帧 metadata 和批量报告。

## 非运行模板

下面用于解释参数，不可原样运行：

```bash
cd /home/zaijia001/ssd/RoboTwin
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  code_painting/render_selection_strategy_compare_v4.py \
  --tasks <TASK> \
  --ids <FOUNDATION_ID> \
  --requested-frames <KEYFRAME> \
  --output-root <NEW_EMPTY_OUTPUT_ROOT>
```

- `--tasks`：一个或多个任务；不传时使用六任务。
- `--ids`：一个或多个 Foundation ID；不传时使用找到的全部 ID。
- `--requested-frames`：可选，只保留指定 requested frame。
- `--output-root`：必须使用新的空目录；默认拒绝覆盖非空目录。
- `--approach-offset-m`：仅控制第一事件 pregrasp 审计显示，默认 0.12 m。
- `--axis-length-m`：坐标轴显示长度，默认 0.045 m。

## 可运行 smoke：Top-score candidate 修正

```bash
cd /home/zaijia001/ssd/RoboTwin
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  code_painting/render_selection_strategy_compare_v4.py \
  --tasks handover_bottle \
  --ids 1 \
  --requested-frames 39 \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/selection_strategy_compare_v4_handover_1_20260714
```

期望：right Top-score `candidate_idx=0`，而不是旧 rank-1 图片中的 13。

## 可运行 smoke：不同 Foundation 帧拼接

```bash
cd /home/zaijia001/ssd/RoboTwin
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  code_painting/render_selection_strategy_compare_v4.py \
  --tasks place_bread_basket \
  --ids 0 \
  --requested-frames 64 \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/selection_strategy_compare_v4_place_bread_0_20260714
```

期望：OursV2 使用 Foundation frame 64，Top-score 使用 resolved frame 63，两个背景横向分栏。

## 可运行全量命令

当前 2026-07-14 全量结果已经位于 `code_painting/selection_strategy_compare_v4/`。再次运行必须使用新的目录：

```bash
cd /home/zaijia001/ssd/RoboTwin
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  code_painting/render_selection_strategy_compare_v4.py \
  --output-root /home/zaijia001/ssd/RoboTwin/code_painting/selection_strategy_compare_v4_rerun_20260714
```

## 校验

```bash
cd /home/zaijia001/ssd/RoboTwin
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 -m py_compile \
  code_painting/render_selection_strategy_compare_v4.py
```

检查 `audit_report.json` 的 `failures`、`record_gaps` 和覆盖计数；图片拥挤时以 metadata 为准。

`--overwrite` 只允许显式重建 V4 自己的输出，绝不授权修改旧策略目录。常规使用优先创建新的 output root。
