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
- Viewer and collection use the same trajectory logic; collection adds pickle-schema validation and observation recording.
- `save_all_episodes` is for debugging and should not be used for formal successful-data filtering.
