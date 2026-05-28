# Piper hand origin 原始数据审核命令

## 用途

在 AnyGrasp / foundation 预处理之前，直接可视化审核 `/home/zaijia001/ssd/data/piper/hand/<TASK>/origin/episode*` 的原始 D435 RGB 帧，并把坏 episode 移到同级 `bad/`。

## 适用版本

当前 Piper hand 原始采集数据审核流程。它不读取 `foundation_input/`、`harmer_input/` 或 AnyGrasp 结果。

## 前置条件

- 使用 `RoboTwin_openvla` conda 环境，里面已有 `cv2`。
- 需要有图形界面用于 OpenCV 窗口显示。
- 原始帧路径应为 `<TASK>/origin/episode*/camera/color/headD435/*.png`。

## 入口脚本

```text
/home/zaijia001/ssd/RoboTwin/code_painting/review_piper_hand_origin.py
```

## 命令格式

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin && python code_painting/review_piper_hand_origin.py --task <TASK> --delay-ms 45
```

## 参数说明

- `--task`：`/home/zaijia001/ssd/data/piper/hand/` 下的任务名。
- `--data-root`：数据根目录，默认 `/home/zaijia001/ssd/data/piper/hand`。
- `--origin-name`：源目录名，默认 `origin`。
- `--bad-name`：坏数据目标目录名，默认 `bad`。
- `--start-id`：从指定 episode 数字编号开始审核。
- `--delay-ms`：自动播放每帧延迟。
- `--frame-stride`：自动播放时的帧步长。
- `--dry-run`：只写审核日志，不移动 episode 目录。

## 交互按键

- `b` / `d`：标为 bad，并把当前 episode 从 `origin/` 移到 `bad/`。
- `g` / `n` / 回车：保留当前 episode 并进入下一个。
- `p`：回到上一个 episode。
- 左右方向键：逐帧查看。
- 空格 / `s`：暂停或继续播放。
- `r`：从第 0 帧重播当前 episode。
- `q` / `Esc`：保存日志并退出。

## 输入

```text
/home/zaijia001/ssd/data/piper/hand/<TASK>/origin/episode*/camera/color/headD435/*.png
```

## 输出

```text
/home/zaijia001/ssd/data/piper/hand/<TASK>/origin_bad_review.json
/home/zaijia001/ssd/data/piper/hand/<TASK>/bad/episode*
```

## 三任务命令

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin && python code_painting/review_piper_hand_origin.py --task pnp_tray --delay-ms 45
```

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin && python code_painting/review_piper_hand_origin.py --task handover_bottle --delay-ms 45
```

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin && python code_painting/review_piper_hand_origin.py --task pnp_bread --delay-ms 45
```

## 相关代码

- `code_painting/review_piper_hand_origin.py`

## 常见问题

- 默认 `python3` 环境可能没有 `cv2`；使用文档中的 `RoboTwin_openvla` 环境运行。
- 标 bad 后目录会被移动，`origin/` 中不再保留该 episode。
