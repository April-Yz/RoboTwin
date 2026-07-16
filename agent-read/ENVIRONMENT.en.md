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

## PiperCanonicalTCP-v1

- Use `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10`; pine2's default Python may lack SciPy/NumPy.
- The EE-pose runner requires SAPIEN, CuRobo, and a GPU; the current batch uses GPU0. Joint-control comparison only needs Python/OpenCV and an existing OursV2 `pose_debug.jsonl`.
- The 0515 calibration is `calibration_bundle_piper_new_table_0515.json`; robot config is `robot_config_PiperPika_agx_dual_table_0515.json`.
- The 2026-07-15 batch uses new `code_painting/piper_canonical_tcp_v1/outputs_canonical_20260715/`; old dry-run files under default `outputs/` are preserved. See `COMMANDS/piper_canonical_tcp_v1.en.md` for tmux and commands.
- VS Code-compatible video post-processing requires system `ffmpeg`/`ffprobe` with `libx264`. The output contract is H.264, `yuv420p`, and `+faststart`; transcoding does not require a GPU.
- Real control compare also requires synchronized raw-episode `camera/color/myD435`, left/right `arm/jointState`, and left/right `arm/endPose`. The entry reads `/home/zaijia001/ssd/data/piper/hand/vis/.pos_source/<TASK>/<EPISODE>` and never treats foundation replay as measured robot state.
- Dual IK and SAPIEN rendering require RoboTwin_bw/CUDA. If the local system Python lacks SciPy, it can only run syntax checks; mathematical tests must run inside RoboTwin_bw.
- The four/five-way replay comparison also requires `RoboTwin_bw` (NumPy, SciPy, OpenCV) plus system `ffmpeg`/`ffprobe`; the host system Python is not expected to include OpenCV.
- Canonical Human Replay reads `foundation_replay_d435/`, `harmer_output/`, the AnyGrasp directory, and `code_painting/h2o_manual_review/<task>/hand_keyframes_all.json`.

## Paper qualitative assets

- The grid compositor uses pine2's system Python plus system `ffmpeg`/`ffprobe`; output must be H.264/`yuv420p`.
- The keyframe-candidate exporter needs NumPy, SciPy, and OpenCV and must use `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10`. The default `python3` on pine2 lacks `cv2`.
- The asset root is `/home/zaijia001/ssd/data/piper/paper_qualitative_assets`; see `COMMANDS/paper_qualitative_assets.en.md`.
