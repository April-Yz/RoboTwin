# Project Understanding

## Main Structure

- `envs/`: tasks, robot control, observations, and HDF5 conversion.
- `assets/embodiments/`: robot URDF, joints, and static-camera configuration.
- `task_config/`: task collection configurations.
- `description/`: task prompts and episode-instruction generation.
- `code_painting/`: hand/object replay, FoundationPose, and AnyGrasp workflows.

## Piper IK Data Flow

Inputs are a seeded dual-bottle scene, bottle functional points, placement targets, and an IK configuration. The task creates four Cartesian move targets: pregrasp, grasp, lift, and place. Each IK segment outputs `position[N,6]` and `velocity[N,6]`, and the next segment starts from the preceding endpoint joint state.

The trajectory pickle stores schema, version, IK version, action names, Cartesian targets, and left/right joint paths. Phase 2 accepts only the current v2 schema and replays it in the same seeded scene. Observations are written to HDF5 and each RGB camera is written to a separate MP4.

## Model And Backend Roles

- V1/V2/V4: Piper-URDF IK using linear, cubic, and multi-seed strategies respectively.
- V3: solves a valid IK endpoint, then tries MotionGen; if MotionGen is unavailable or fails, it uses a cubic path to that endpoint.
- SAPIEN: executes joint commands, simulates contact, renders cameras, and evaluates task success.

## Known Boundaries

- V3 MotionGen optimization may fail in the current scene, but the fallback path has been validated successfully.
- Viewer and collection use the same trajectory logic; collection adds pickle-schema validation and observation recording. The custom Piper IK execution loop must call `viewer.render()` after `_update_render()` for live main-window updates; move, settle, and gripper paths now follow that order. The motion viewer can concurrently show dynamic wrist/static head frustums and dual-wrist RGB.
- `save_all_episodes` is for debugging and should not be used for formal successful-data filtering.

## O.1 Foundation OBJ Data Flow

O.1 reads positions, optional orientations, and source OBJ paths from `multi_object_world_poses.npz`. Trimesh bounds define the geometric center and table clearance. The input-0 meshes are about 6.6 cm wide, and full bottle-body collision causes the current Pika gripper to tip them during pregrasp/grasp approach, so the default uses a base-only `support_proxy`. Close never resets object pose. It validates displacement/rotation, link6 distance, and geometric capture between both fingers before attaching a drive at the current pose.

O.1.1 uses the first annotated keyframe in `hand_keyframes_all.json` for the Foundation OBJ setup. O.1.2 additionally loads left/right EE xyz at the second keyframe from `world_targets_and_status.npz` and replaces lift/place with one action that retains the grasp orientation. Trajectories bind mode, episode ID, keyframes, action source, grasp-gate parameters, and mesh geometry to prevent cross-setting replay.

Phase 2 only consumes validated joint paths and does not create IK or MotionGen planners, avoiding V3 replay GPU pressure and slow multi-camera rendering.

Foundation wrist cameras load distinct gripper-to-camera extrinsics from `calibration_bundle_piper_new_table_0515.json`. The official Pika gripper URDF has no camera link, and official Piper+Pika versus the current AGX-derived merge express the `link6 -> gripper` axes differently. Each camera follows raw `link6`, applies the base `piper_pika_agx` adapter and optical-to-render conversion, then applies per-side `forward_offset_m/image_roll_deg`; current values are left `0.125/-15` and right `0.11/-60`. This final layer normalizes simulation training views without changing calibration files or IK. The strict chain still lacks `link6_T_real_tcp`; 0515 only supplies `real_tcp_T_camera`, so current tuning is an empirical estimate of missing mechanical extrinsics. The debug recorder saves H.264/faststart left/right/mosaic MP4s and context JSON from the same frames; the formal wrapper writes four environment overrides into generated YAML. The existing HDF5/video merge path exports every observation RGB camera automatically.

Foundation collection writes one directory per source ID, with internal episode numbering starting at `episode0`. `script/index_foundation_piper_ik_videos.py` maps source ID N to aggregate `episodeN` files without modifying source data and records the mapping in a manifest. Existing destination episodes are conflicts by default.

The batch wrapper forces one episode per ID and supports isolated run tags. `script/collect_data.py` applies `max_seed_tries` as a hard bound on seed search. This terminates deterministic geometry failures without treating them as successful episodes.
