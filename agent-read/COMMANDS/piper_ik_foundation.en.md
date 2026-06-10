# Piper IK Foundation OBJ Pick And Place

## Purpose

Use positions and source OBJ meshes from Foundation NPZ files for Piper V1-V4 Cartesian motion viewing and two-phase SAPIEN data collection.

## Prerequisites

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw
cd /home/zaijia001/ssd/RoboTwin
```

## Viewer

```bash
python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --task_config demo_piper_ik_foundation_v1 \
  --ik_version v1 --foundation_id 0 --foundation_frame 0 \
  --seed 0 --max_seed_tries 1 --require_success 1
```

For headless validation, add `unset DISPLAY` and `--hold 0 --render_freq 0 --show_axes 0`.

## Data Collection

```bash
# Fixed input 0
bash collect_data.sh pick_diverse_bottles_piper_ik_foundation demo_piper_ik_foundation_v1 0

# Arguments: IK version, foundation_input ID, frame, GPU ID
bash collect_foundation_piper_ik.sh v1 0 0 0
```

The wrapper creates isolated config and output names by version, ID, and frame. Do not mutate the base YAML with `sed -i` and reuse one trajectory directory.

## Execution Logic

- The NPZ supplies left/right positions, quaternions, and OBJ paths.
- The OBJ-bounds geometric center is the grasp reference; the actor origin is commonly near the bottle base.
- Visual geometry uses the source OBJ; collision defaults to a bounds-derived cylinder proxy.
- Upright orientation is the default. FoundationPose orientation is optional and may be physically unstable.
- The narrow meshes cannot be held reliably by pure contact with the current Pika gripper, so a distance-gated grasp drive is enabled by default and released on open.
- Phase 1 plans sequentially and saves. Phase 2 validates the Foundation source and trajectory schema before replay and does not initialize planners.

## Validated State

On 2026-06-11, V1-V4 viewers and full two-phase collection succeeded with `foundation_input_0/frame0`. V3 fell back from failed MotionGen optimization to the validated IK interpolation path as designed.

## Related Code

- `envs/pick_diverse_bottles_piper_ik_foundation.py`
- `envs/pick_diverse_bottles_piper_ik.py`
- `view_pick_diverse_bottles_piper_ik_motion.py`
- `collect_foundation_piper_ik.sh`
- `task_config/demo_piper_ik_foundation_v1.yml` through `v4.yml`
