# Piper IK Cartesian Pick And Place

## Purpose

Provide working V1-V4 motion-viewer, staged trajectory collection, and replay commands for `pick_diverse_bottles_piper_ik`.

## Applicable Version

- Trajectory schema: `piper_ik_cartesian`
- Trajectory version: 2
- Configs: `demo_piper_ik_seq_v1` through `demo_piper_ik_seq_v4`
- Recommended default: V1. V3 falls back to cubic interpolation to the same IK endpoint when MotionGen fails.

## Prerequisites

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw
cd /home/zaijia001/ssd/RoboTwin
```

## Viewer

```bash
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 50 --require_success 1
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v2 --seed 0 --max_seed_tries 50 --require_success 1
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v3 --seed 0 --max_seed_tries 50 --require_success 1
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v4 --seed 0 --max_seed_tries 50 --require_success 1
```

Headless validation:

```bash
unset DISPLAY
SAPIEN_RT_DENOISER=none timeout 180s python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 50 --hold 0 --render_freq 0 --show_axes 0 --require_success 1
```

## Data Collection

```bash
bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v1 0
bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v2 0
bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v3 0
bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v4 0
```

## Parameters

- `ik_version`: V1 linear, V2 cubic, V3 MotionGen with fallback, V4 multi-seed cubic.
- `lift_height`: world-z increase from the grasp target, currently 0.12 m.
- `move_settle_steps`: final IK command hold after each move, currently 80 steps.
- `gripper_settle_steps`: gripper open/close settling duration, currently 120 steps.
- `require_success=1`: the viewer accepts only seeds that pass the physical `check_success()`.

## Inputs And Outputs

- Prompt: `description/task_instruction/pick_diverse_bottles_piper_ik.json`
- Trajectory: `data/pick_diverse_bottles_piper_ik/<config>/_traj_data/episodeN.pkl`
- Dataset: `data/.../<config>/data/episodeN_succ.hdf5`
- Videos: one MP4 per RGB camera, including right-side `third_camera` and opposite overhead `opposite_top_camera`.

## Execution Logic

The fixed action order is `pregrasp -> grasp -> close -> lift -> place -> open`. The four move segments are planned and executed sequentially; each segment starts from the previous trajectory endpoint. Lift preserves grasp x/y and orientation and only increases z. After close, the measured bottle-functional-point to actual-EE x/y offset corrects the place gripper target.

The viewer plans and executes Phase 1 directly. Collection saves that trajectory in Phase 1 and validates/replays it in the same seeded scene during Phase 2. Motion targets and joint paths therefore match, while collection additionally records observations. Pickles without the required schema, version, action names, or valid nonempty paths are rejected.

Phase 2 does not replan, so replay scenes skip IK/MotionGen planner initialization and avoid extra GPU pressure during V3 multi-camera collection.

## Related Code

- `envs/pick_diverse_bottles_piper_ik.py`
- `envs/robot/piper_ik.py`
- `view_pick_diverse_bottles_piper_ik_motion.py`
- `script/collect_data.py`
- `assets/embodiments/piper_pika_agx/config.yml`

## Common Issues

- The old lift/move failures came from planning every segment from home, deriving lift x/y from the wrong reference, insufficient PD settling, and treating bottle targets as gripper targets.
- On a headless host, do not use an invalid `DISPLAY=:1.0`; use `unset DISPLAY --render_freq 0`.
- Do not load old `demo_piper_ik_v*` outputs with new code. Use the isolated `demo_piper_ik_seq_v*` configs.
