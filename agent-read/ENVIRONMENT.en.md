# Environment

- Host: `pine2`
- Project root: `/home/zaijia001/ssd/RoboTwin`
- Conda initialization: `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh`
- Runtime environment: `RoboTwin_bw`
- Non-interactive SSH may not load `conda`; the read-only V4 can call `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10` directly.
- Dense Replay requires a usable CUDA GPU, SAPIEN/Vulkan, Curobo, and FFmpeg/FFprobe.
- Human input defaults to `/home/zaijia001/ssd/data/piper/hand/<TASK>/harmer_output/hand_detections_<ID>.npz`.

Short command:

```bash
cd /home/zaijia001/ssd/RoboTwin && TASK=pick_diverse_bottles ID=0 GPU=3 bash code_painting/run_dense_replay_urdfmatch_v2.sh
```

Expanded steps: enter the project root, confirm the NPZ exists, select an available GPU, let the wrapper activate the environment and run, then inspect metadata, audit, and MP4 outputs.

The six-task batch currently uses one GPU-3 process to avoid competing SAPIEN/Curobo instances:

```bash
tmux capture-pane -pt dense_replay_urdfmatch_v2:0 -S -60
tail -n 30 /home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435_urdfmatch_v2/_batch_logs/status.tsv
```

Do not terminate long-lived GPU processes on pine2 without confirming ownership. Per-episode batch failures are recorded in the status table and processing continues.

Selection Strategy Audit V4 only reads existing data and needs no GPU, SAPIEN, or planner. It requires NumPy, SciPy, and OpenCV. See `COMMANDS/selection_strategy_audit_v4.en.md`.
