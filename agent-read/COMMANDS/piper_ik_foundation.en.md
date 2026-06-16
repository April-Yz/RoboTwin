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
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --seed 0 --max_seed_tries 1 --require_success 1 --wrist_preview 1
```

`--wrist_preview 1` adds a live left/right wrist RGB mosaic. The viewer requires a working X11/Vulkan display; remote desktops can explicitly use `DISPLAY=:1.0`. Do not use `unset DISPLAY`.

## Data Collection

```bash
# Fixed input 0
bash collect_data.sh pick_diverse_bottles_piper_ik_foundation demo_piper_ik_foundation_v1 0

# Arguments: IK version, foundation_input ID, frame, GPU ID, mode, optional run tag
bash collect_foundation_piper_ik.sh v1 0 0 0 o1
bash collect_foundation_piper_ik.sh v1 0 0 0 o1.1
bash collect_foundation_piper_ik.sh v1 0 0 0 o1.2 wrist_o121_verified_0615
```

The wrapper creates isolated config and output names by version, ID, frame/mode, and run tag, and forces one episode per ID. O.1.1/O.1.2 override the frame with the first annotation. Use a new run tag after enabling wrist cameras: an existing HDF5 is skipped, so an old head-only directory is not upgraded in place. Do not mutate the base YAML with `sed -i` and reuse one trajectory directory.

Recommended batch form:

```bash
RUN_TAG=wrist_o121_verified_0615
mkdir -p data/tmp
FAIL_LOG="$PWD/data/tmp/o12_v1_${RUN_TAG}_failures.log"
: > "$FAIL_LOG"
for id in $(seq 0 120); do
  timeout 600s bash collect_foundation_piper_ik.sh v1 "$id" 0 0 o1.2 "$RUN_TAG" \
    || echo "FAIL v1 id=$id status=$?" | tee -a "$FAIL_LOG"
done
```

For V2/V3/V4 use version/GPU pairs `v2/1`, `v3/2`, and `v4/3`. Inputs currently cover IDs 0-101; scanning 102-120 exits quickly for missing NPZ files and records those IDs in the failure log.

## Execution Logic

- The NPZ supplies left/right positions, quaternions, and OBJ paths.
- The OBJ-bounds geometric center is the grasp reference; the actor origin is commonly near the bottle base.
- Visual geometry uses the source OBJ. Collision defaults to a thin `support_proxy` at the OBJ base so the open gripper does not knock the narrow bottle over during approach.
- Upright orientation is the default. FoundationPose orientation is optional and may be physically unstable.
- Close no longer calls `set_pose`; a bottle knocked over during approach is never teleported back into the gripper.
- After close, the task checks object displacement, rotation, link6 distance, projection between both fingers, and radial distance. A drive is attached at the current object pose only after every gate passes.
- The default `support_proxy` has no full bottle-body contacts, so `foundation_grasp_require_contact=false` uses the geometric capture gate. This is controlled, no-teleport grasp assist rather than pure-contact physics.
- Phase 1 plans sequentially and saves. Phase 2 validates the Foundation source and trajectory schema before replay and does not initialize planners.
- Each version tries at most `max_seed_tries: 3` seeds. A deterministic failure returns nonzero instead of looping indefinitely and appearing stuck.

## Wrist Cameras

- All four Foundation configs enable `collect_wrist_camera: true`.
- The left and right cameras load their distinct `left_gripper_T_camera` and `right_gripper_T_camera` transforms from `calibration_bundle_piper_new_table_0515.json`.
- Cameras follow simulation `link6` through `wrist_camera_pose_reference: urdf_end_link`. `wrist_camera_simulation_adapter: piper_pika_agx` handles the base TCP/CAD translation, then per-side `wrist_camera_tuning` handles optical-axis displacement and training-view roll.
- Current tuning is left `0.125 m/-15 deg`, right `0.11 m/-60 deg`. It does not rewrite the 0515 JSON or affect IK. The left/right displacement difference matches the approximately 1.43 cm difference between calibrated X translations.
- `wrist_camera_axis_mode: legacy_r1` converts the calibrated OpenCV optical frame to the SAPIEN render frame.
- Phase 2 automatically exports every observation RGB camera. The new files are `episode0_succ_left_camera.mp4` and `episode0_succ_right_camera.mp4`, and both cameras are also present in HDF5 observations.

## ID-To-Episode Video Index

Each per-ID Foundation directory internally contains `episode0_*`. Map source ID N to `episodeN_*` in an aggregate directory with:

```bash
python script/index_foundation_piper_ik_videos.py \
  --version v4 --mode o1.2 --run-tag wrist_o121_verified_0615 \
  --output-video-dir data/pick_diverse_bottles_piper_ik/demo_piper_ik_foundation_v4_o1_2_wrist_o121_verified_0615/video \
  --method symlink --dry-run
```

Remove `--dry-run` after inspection. The script writes `foundation_episode_index.json`. Existing episodes are conflicts by default; `--replace-episode` deletes that ID's old destination MP4 files and must be used explicitly.

The currently available V4/O.1.2 directories have no run tag, so omit `--run-tag` when indexing those legacy outputs. A dry run against `demo_piper_ik_v4_3/video` found indexable IDs 0-8 and conflicts for existing IDs 0-4; ID 9 has no successful videos.

## Grasp Debugging

Important defaults are `foundation_pregrasp_distance: 0.12`, maximum displacement `0.025 m`, maximum rotation `15 deg`, radial tolerance `0.065 m`, and segment margin `0.15`.

If pregrasp still collides, increase `foundation_pregrasp_distance` and inspect the viewer axes. For grasp-stage collision, adjust `foundation_grasp_standoff` or gripper orientation. Strict-contact experiments can use `cylinder_proxy` or `exact_convex` with `foundation_grasp_require_contact: true`; input 0 currently exposes an actual collision failure in that setup instead of hiding it with a pose reset.

## Validated State

On 2026-06-15, O.1.2 V4 with `foundation_input_0` completed validated replay, `episode0_succ.hdf5`, instructions, and eight videos using the final tuning. Both wrist videos have 38 frames at 320x240; sampled frames confirm that the shell has left the image and the right view is upright. Old untagged tmux outputs were caused by an unset `RUN_TAG` and must not be mixed with this run.

Viewer-only overrides are available as `--wrist_left_forward_offset_m`, `--wrist_right_forward_offset_m`, `--wrist_left_roll_deg`, and `--wrist_right_roll_deg`. The official Pika gripper URDF contains no camera link. Wrist rendering still uses the D435 preset; strict D405 matching requires measured intrinsics and a camera optical-center CAD transform.

The reviewed tmux panes had already returned to the shell; they were not still collecting. Earlier jobs showed `Killed` and Ctrl-C. V1 had been configured for ten episodes per ID, while failed seed search had no bound. V4 ID 9 deterministically exceeded the right-grasp rotation gate at about 25.6 degrees versus a 15-degree limit across multiple seeds. Current configs use one episode per ID, three seed attempts, and the documented outer timeout.

## Related Code

- `envs/pick_diverse_bottles_piper_ik_foundation.py`
- `envs/pick_diverse_bottles_piper_ik.py`
- `view_pick_diverse_bottles_piper_ik_motion.py`
- `collect_foundation_piper_ik.sh`
- `script/index_foundation_piper_ik_videos.py`
- `code_painting/build_piper_calibration_bundle.py`
- `task_config/demo_piper_ik_foundation_v1.yml` through `v4.yml`

## O.1.2.1 Frame Conclusion And Viewer Debug Recording

The required chain is `world_T_link6 @ link6_T_real_tcp @ real_tcp_T_camera @ optical_T_render`. The 0515 calibration only supplies `real_tcp_T_camera`; the missing real TCP, bracket, and optical-center transform is `link6_T_real_tcp`. Transform composition is valid, but an unknown mechanical extrinsic was omitted. Current tuning is an empirical compensation, not a new physical calibration.

Positive forward offset moves along each camera's own viewing axis. In the current render convention, positive roll appears clockwise and negative roll counterclockwise. Add `--wrist_debug_record 1 --wrist_debug_tag <TAG>` to save raw left/right MP4s, a labeled mosaic MP4, and parameter JSON under `data/wrist_camera_debug/<TAG>/`.

Debug MP4s now use `H.264/avc1 + yuv420p + faststart` for direct VS Code playback. For headless debug recording, add `--render_freq 0 --show_axes 0 --wrist_preview 0`. To use the formal collection pipeline, set all four `WRIST_LEFT_FORWARD_OFFSET_M`, `WRIST_RIGHT_FORWARD_OFFSET_M`, `WRIST_LEFT_ROLL_DEG`, and `WRIST_RIGHT_ROLL_DEG` variables before `collect_foundation_piper_ik.sh`. The wrapper writes them into an isolated generated YAML, and Phase 2 produces the formal HDF5 and eight H.264 videos.

```bash
unset DISPLAY
WRIST_LEFT_FORWARD_OFFSET_M=0.125 \
WRIST_RIGHT_FORWARD_OFFSET_M=0.11 \
WRIST_LEFT_ROLL_DEG=-15 \
WRIST_RIGHT_ROLL_DEG=-60 \
timeout 600s bash collect_foundation_piper_ik.sh \
  v1 0 0 0 o1.2 wrist_left125_right110_roll_m15_m60
```

## Viewer With Camera Frustums

```bash
export DISPLAY=:1.0
TAG="o121_v1_viewer_$(date +%Y%m%d_%H%M%S)"
python view_pick_diverse_bottles_piper_ik_motion.py \
  --task_name pick_diverse_bottles_piper_ik_foundation \
  --ik_version v1 --foundation_id 0 --foundation_mode o1.2 \
  --render_freq 1 --show_axes 1 --show_camera_frustums 1 \
  --wrist_preview 1 --hold 1 \
  --wrist_left_forward_offset_m 0.125 --wrist_right_forward_offset_m 0.11 \
  --wrist_left_roll_deg -15 --wrist_right_roll_deg -60 \
  --wrist_debug_record 1 --wrist_debug_tag "$TAG" \
  --max_seed_tries 1 --require_success 1
```

`--show_camera_frustums 1` enables SAPIEN camera frustums and verifies that `left_camera`, `right_camera`, and `head_camera` are present. `--wrist_preview 1` is the separate dual-wrist RGB window, while `--hold 1` keeps the final viewer open. After `unset DISPLAY`, restore the GUI with `export DISPLAY=:1.0`; `set DISPLAY` is ineffective. The timestamped tag avoids `FileExistsError` from an existing non-empty output directory.

### Live Motion Modes

The Piper IK executor now refreshes both observation cameras and `viewer.render()` during move, settle, and gripper steps. Mode 1 uses `--wrist_preview 0` for live SAPIEN robot motion plus camera frustums only. Mode 2 uses `--wrist_preview 1` to show SAPIEN and the dual-wrist RGB window concurrently. Neither mode needs a debug tag; complete commands are in the 2026-06-16 O.1.2.1 addition to `COMMAND_LIBRARY.zh.md`.

### Wrist Forward-Axis Diagnosis

```bash
python script/diagnose_piper_wrist_camera_axes.py
```

The script reuses the axis conversion from `envs/camera/camera.py` and reports the SAPIEN camera `+X` forward direction in the gripper frame, the error against the plane perpendicular to finger-opening `Y`, and angles to both the Pika physical `+X` and legacy debug `+Z` conventions. Current conclusion: the camera forward axis is already close to Pika physical `+X` and almost in the `X-Z` plane perpendicular to finger opening. The roughly 90-degree error against legacy debug `+Z` is a frame-convention mismatch, not a camera-extrinsic correction to apply directly.
