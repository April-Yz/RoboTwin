# 2026-04-14 Piper AnyGrasp Planner Wrapper

## Goal

Add a new Piper-oriented AnyGrasp planning entry without modifying the original R1 / R1 Pro planner files.

## Added files

- `robot_config_Piper_dual.json`
- `code_painting/plan_anygrasp_keyframes_piper.py`
- `code_painting/plan_anygrasp_keyframes_piper_batch.py`
- `code_painting/run_plan_anygrasp_keyframes_piper_batch.sh`

## Design

The new Piper entry is intentionally a wrapper around the existing R1 planning stack:

- `plan_anygrasp_keyframes_piper.py`
  - imports `plan_anygrasp_keyframes_r1.py`
  - injects a Piper robot config by default
  - swaps the replay renderer to a Piper-specific subclass that looks up Piper camera / wrist links
  - swaps the URDF-IK renderer to use `assets/embodiments/piper/piper.urdf`
  - keeps the original CLI and planner logic as much as possible
- `plan_anygrasp_keyframes_piper_batch.py`
  - reuses the existing batch launcher logic
  - only changes the single-video script target
- `run_plan_anygrasp_keyframes_piper_batch.sh`
  - mirrors the existing R1 shell wrapper
  - activates the same `RoboTwin_bw` conda env

## Robot config note

`robot_config_Piper_dual.json` uses two Piper instances (`dual_arm_embodied=false` with left/right Piper configs) so the existing left/right execution path can still run without rewriting the original dual-arm planner structure.

This is a compatibility-oriented adapter, not a full semantic rewrite of the pipeline.

## Important limitations

- The original planner is deeply R1-oriented.
- Piper support added here is a compatibility wrapper, mainly for:
  - reusing AnyGrasp candidate selection
  - reusing replay / execution staging
  - swapping URDF IK to Piper
- Camera-frame alignment for Piper may still need tuning later:
  - `--camera_cv_axis_mode`
  - `--head_camera_local_quat_wxyz`
  - possibly object/TCP offsets
- The wrapper currently defaults `--head_camera_local_quat_wxyz 1 0 0 0` unless explicitly overridden.

## Example command

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --skip_existing 0 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps -1 \
  --urdfik_cartesian_interp_auto_step_m 0.01
```

## Minimal validation

- `python -m py_compile code_painting/plan_anygrasp_keyframes_piper.py code_painting/plan_anygrasp_keyframes_piper_batch.py`
- `bash -n code_painting/run_plan_anygrasp_keyframes_piper_batch.sh`
- `python code_painting/plan_anygrasp_keyframes_piper.py --help`
