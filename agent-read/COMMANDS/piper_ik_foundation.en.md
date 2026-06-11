# Piper IK Foundation OBJ Pick And Place

## Purpose

Use positions and source OBJ meshes from Foundation NPZ files for Piper V1-V4 Cartesian motion viewing and two-phase SAPIEN data collection.

## Prerequisites

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw
cd /home/zaijia001/ssd/RoboTwin
```

## Modes

- `o1`: use an explicit `foundation_frame` and execute `pregrasp -> grasp -> close -> lift -> place -> open`.
- `o1.1`: initialize object poses from the episode's first annotated keyframe in `hand_keyframes_all.json`; retain the O.1 action sequence.
- `o1.2`: use the first keyframe for setup/grasp, then use left/right EE xyz from the second keyframe in `world_targets_and_status.npz` as one `action` replacing lift/place. The grasp orientation is retained.

## Viewer

```bash
python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --task_config demo_piper_ik_foundation_v1 \
  --ik_version v1 --foundation_id 0 --foundation_frame 0 --foundation_mode o1 \
  --seed 0 --max_seed_tries 1 --require_success 1
```

Use `--foundation_mode o1.1` or `--foundation_mode o1.2` for the keyframe modes. The viewer creates a SAPIEN window and requires a working X11/Vulkan display; do not use `unset DISPLAY` for this command.

## Data Collection

```bash
# Fixed input 0
bash collect_data.sh pick_diverse_bottles_piper_ik_foundation demo_piper_ik_foundation_v1 0

# Arguments: IK version, foundation_input ID, frame, GPU ID, mode
bash collect_foundation_piper_ik.sh v1 0 0 0 o1
bash collect_foundation_piper_ik.sh v1 0 0 0 o1.1
bash collect_foundation_piper_ik.sh v1 0 0 0 o1.2
```

The wrapper creates isolated config and output names by version, ID, and frame/mode. O.1.1/O.1.2 override the frame with the first annotation. Do not mutate the base YAML with `sed -i` and reuse one trajectory directory.

## Execution Logic

- The NPZ supplies left/right positions, quaternions, and OBJ paths.
- The OBJ-bounds geometric center is the grasp reference; the actor origin is commonly near the bottle base.
- Visual geometry uses the source OBJ. Collision defaults to a thin `support_proxy` at the OBJ base so the open gripper does not knock the narrow bottle over during approach.
- Upright orientation is the default. FoundationPose orientation is optional and may be physically unstable.
- Close no longer calls `set_pose`; a bottle knocked over during approach is never teleported back into the gripper.
- After close, the task checks object displacement, rotation, link6 distance, projection between both fingers, and radial distance. A drive is attached at the current object pose only after every gate passes.
- The default `support_proxy` has no full bottle-body contacts, so `foundation_grasp_require_contact=false` uses the geometric capture gate. This is controlled, no-teleport grasp assist rather than pure-contact physics.
- Phase 1 plans sequentially and saves. Phase 2 validates the Foundation source and trajectory schema before replay and does not initialize planners.

## Grasp Debugging

Important defaults are `foundation_pregrasp_distance: 0.12`, maximum displacement `0.025 m`, maximum rotation `15 deg`, radial tolerance `0.065 m`, and segment margin `0.15`.

If pregrasp still collides, increase `foundation_pregrasp_distance` and inspect the viewer axes. For grasp-stage collision, adjust `foundation_grasp_standoff` or gripper orientation. Strict-contact experiments can use `cylinder_proxy` or `exact_convex` with `foundation_grasp_require_contact: true`; input 0 currently exposes an actual collision failure in that setup instead of hiding it with a pose reset.

## Validated State

On 2026-06-11 with `foundation_input_0`, V1 viewer and full two-phase collection succeeded for O.1, O.1.1, and O.1.2. Full O.1.2 collection also succeeded with V2/V3/V4. The annotated frames are 38 and 78, and all four O.1.2 backends produced validated replay, `episode0_succ.hdf5`, instructions, and six videos.

## Related Code

- `envs/pick_diverse_bottles_piper_ik_foundation.py`
- `envs/pick_diverse_bottles_piper_ik.py`
- `view_pick_diverse_bottles_piper_ik_motion.py`
- `collect_foundation_piper_ik.sh`
- `task_config/demo_piper_ik_foundation_v1.yml` through `v4.yml`
