# 环境

- 主机：`pine2`
- 项目根目录：`/home/zaijia001/ssd/RoboTwin`
- Conda 初始化：`source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh`
- 运行环境：`RoboTwin_bw`
- Dense Replay 需要可用的 CUDA GPU、SAPIEN/Vulkan、Curobo、FFmpeg/FFprobe。
- 人手输入默认位于 `/home/zaijia001/ssd/data/piper/hand/<TASK>/harmer_output/hand_detections_<ID>.npz`。

短命令：

```bash
cd /home/zaijia001/ssd/RoboTwin && TASK=pick_diverse_bottles ID=0 GPU=3 bash code_painting/run_dense_replay_urdfmatch_v2.sh
```

展开步骤：进入项目目录，确认 NPZ 存在，选择空闲 GPU，通过 wrapper 激活环境并运行，最后检查 metadata、audit 和 MP4。
