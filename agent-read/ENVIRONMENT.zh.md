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

Selection Strategy Audit V4 只读取已有数据，不需要 GPU、SAPIEN 或 planner；需要 NumPy、SciPy 和 OpenCV。命令见 `COMMANDS/selection_strategy_audit_v4.zh.md`。
