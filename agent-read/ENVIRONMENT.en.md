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

Selection Strategy Audit V4 only reads existing data and needs no GPU, SAPIEN, or planner. It requires NumPy, SciPy, and OpenCV. See `COMMANDS/selection_strategy_audit_v4.en.md`.
