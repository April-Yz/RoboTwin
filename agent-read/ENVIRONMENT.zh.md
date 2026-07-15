# 环境

- 主机：`pine2`
- 项目根目录：`/home/zaijia001/ssd/RoboTwin`
- Conda 初始化：`source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh`
- 运行环境：`RoboTwin_bw`
- 非交互 SSH 不一定加载 `conda`；只读 V4 可直接使用 `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10`。
- Dense Replay 需要可用的 CUDA GPU、SAPIEN/Vulkan、Curobo、FFmpeg/FFprobe。
- 人手输入默认位于 `/home/zaijia001/ssd/data/piper/hand/<TASK>/harmer_output/hand_detections_<ID>.npz`。

短命令：

```bash
cd /home/zaijia001/ssd/RoboTwin && TASK=pick_diverse_bottles ID=0 GPU=3 bash code_painting/run_dense_replay_urdfmatch_v2.sh
```

展开步骤：进入项目目录，确认 NPZ 存在，选择空闲 GPU，通过 wrapper 激活环境并运行，最后检查 metadata、audit 和 MP4。

六任务批处理当前使用单个 GPU 3 进程，避免多个 SAPIEN/Curobo 实例争抢显存：

```bash
tmux capture-pane -pt dense_replay_urdfmatch_v2:0 -S -60
tail -n 30 /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435_urdfmatch_v2/_batch_logs/status.tsv
```

不要在未确认归属前终止 pine2 上长期占用 GPU 的旧进程；批处理失败会写入状态表并继续下一个 episode。

Selection Strategy Audit V4 只读取已有数据，不需要 GPU、SAPIEN 或 planner；需要 NumPy、SciPy 和 OpenCV。命令见 `COMMANDS/selection_strategy_audit_v4.zh.md`。

## PiperCanonicalTCP-v1

- 必须使用 `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10`；pine2 默认 Python 可能没有 SciPy/NumPy。
- EE-pose runner 需要 SAPIEN、CuRobo 和 GPU；当前 batch 使用 GPU0。joint-control 对比只需 Python/OpenCV 和已有 OursV2 `pose_debug.jsonl`。
- 0515 标定为 `calibration_bundle_piper_new_table_0515.json`，机器人配置为 `robot_config_PiperPika_agx_dual_table_0515.json`。
- 2026-07-15 batch 输出只写全新 `code_painting/piper_canonical_tcp_v1/outputs_canonical_20260715/`；默认 `outputs/` 的旧 dry-run 文件保留。tmux 与命令见 `COMMANDS/piper_canonical_tcp_v1.zh.md`。
- VSCode 兼容视频后处理要求系统 `ffmpeg`/`ffprobe`，且 FFmpeg 提供 `libx264`。标准输出契约为 H.264、`yuv420p`、`+faststart`；转码不需要 GPU。
