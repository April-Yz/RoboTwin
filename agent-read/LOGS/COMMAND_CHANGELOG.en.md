## 2026-04-27 13:55:00 +08

- Added FoundationPose preparation and execution commands for Piper:
  - `conda run -n hamer-r1 python /home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py ...`
  - `bash /home/zaijia001/FoundationPose/run_piper_star_pear_foundation.sh`
- Added pear+star multi-object detection command (`run_realr1_dino_sam_batch.py` with `--object pear=... --object star=...`).
- Updated `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` with FoundationPose stage and RoboTwin object-replay stage.

## 2026-04-24 15:20:00 +08

- Added cross-project hand-processing command doc:
  - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
- The doc uses a one-line-comment + one-line-command format and includes:
  - Piper -> HaMeR conversion
  - HaMeR single/batch detection
  - `ffplay` visualization checks
  - RoboTwin downstream replay command
- Related entry scripts:
  - `/home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py`
  - `/home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py`

## 2026-04-16 04:10:00 +08

- Updated `code_painting/pika/visualize_calibrated_piper_pika_scene.py`
- Updated `code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`
  - Added arguments:
    - `--viewer 1`
    - `--viewer-camera {overview,head}`
  - These scripts can now open interactive viewer windows.
- Updated command docs:
  - `agent-read/COMMANDS/pika_scene_commands.en.md`
  - `agent-read/COMMANDS/pika_scene_commands.zh.md`

## 2026-04-16 03:55:00 +08

- Updated `agent-read/COMMANDS/pika_scene_commands.en.md`
- Updated `agent-read/COMMANDS/pika_scene_commands.zh.md`
  - added explicit commands for exporting head-cam-only images
  - added explicit note that wrist views are not exported yet by calibrated-scene scripts

## 2026-04-16 03:40:00 +08

- Added `agent-read/COMMANDS/pika_scene_commands.en.md`
- Added `agent-read/COMMANDS/pika_scene_commands.zh.md`
  - one-line-comment + command format
  - includes viewer commands for the manual tabletop scene
  - notes that calibrated scene scripts are currently offscreen-only

## 2026-04-16 03:25:00 +08

- Added `code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`
  - Purpose:
    - reconstruct the calibrated scene using version B alignment
  - Key differences from the previous calibrated scene script:
    - removes the +90deg anchor rotation
    - treats left/right spread as world y
    - rotates the table convention by 90deg
  - Example:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`

## 2026-04-16 03:10:00 +08

- Added `code_painting/pika/visualize_calibrated_piper_pika_scene.py`
  - Purpose:
    - reconstruct a simulated scene from the real calibration bundle
  - Inputs:
    - `robot_config_PiperPika_agx_dual_table.json`
    - `calibration_bundle_try2.json`
  - Outputs:
    - `code_painting/pika/output_calibrated_scene/`
  - Example:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/pika/visualize_calibrated_piper_pika_scene.py`

## 2026-04-16 02:50:00 +08

- Updated `robot_config_PiperPika_agx_dual_table.json`
  - moved the edge-side base position from `y = -0.60` (outside table) to `y = -0.30` (edge-mounted on tabletop)
- Reused `code_painting/visualize_piper_pika_agx_dual_table.py`
  - exported edge-mounted preview:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode oblique --output-dir code_painting/output_piper_pika_agx_dual_table_edge_mount --image-name piper_pika_agx_dual_table_edge_mount.png --video-name piper_pika_agx_dual_table_edge_mount.mp4`

## 2026-04-16 02:35:00 +08

- Updated `robot_config_PiperPika_agx_dual_table.json`
  - corrected base orientation to face the tabletop from the long-edge side
  - quaternion: `[0.70710678, 0.0, 0.0, 0.70710678]`
- Updated `code_painting/visualize_piper_pika_agx_dual_table.py`
  - corrected `--camera-mode oblique` to a proper behind-the-robots viewpoint
  - refreshed top-down output after quaternion fix
  - examples:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode oblique --output-dir code_painting/output_piper_pika_agx_dual_table_oblique_fixed --image-name piper_pika_agx_dual_table_oblique_fixed.png --video-name piper_pika_agx_dual_table_oblique_fixed.mp4`
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode top_down --output-dir code_painting/output_piper_pika_agx_dual_table_topdown_fixed --image-name piper_pika_agx_dual_table_topdown_fixed.png --video-name piper_pika_agx_dual_table_topdown_fixed.mp4`

## 2026-04-16 02:20:00 +08

- Updated `code_painting/visualize_piper_pika_agx_dual_table.py`
  - clarified the one-side long-edge dual-arm layout:
    - bases at x = ±0.30 m
    - base spacing = 0.60 m
  - added camera selection:
    - `--camera-mode top_down`
    - `--camera-mode oblique`
  - top-down example:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode top_down --output-dir code_painting/output_piper_pika_agx_dual_table_topdown --image-name piper_pika_agx_dual_table_topdown.png --video-name piper_pika_agx_dual_table_topdown.mp4`

## 2026-04-16 02:05:00 +08

- Added `code_painting/visualize_piper_pika_agx_dual_table.py`
  - Purpose:
    - preview the newer colored Piper+Pika dual-arm layout on a 120x60x75 cm table
  - Example:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1`
- Added `robot_config_PiperPika_agx_dual_table.json`
  - UR-style symmetric split using `embodiment_dis = 0.60`
  - effective base poses:
    - left `[-0.30, -0.60, 0.75]`
    - right `[0.30, -0.60, 0.75]`

## 2026-04-16 01:45:00 +08

- Added a color-statistics analysis step (diagnosis only, no source modification)
  - compared exported preview PNGs against the Piper DAE diffuse reference color
  - used approximate foreground color statistics and ΔE76 distances for documentation

## 2026-04-16 01:45:00 +08

- Added `code_painting/visualize_agx_arm_sim_source.py`
  - Purpose:
    - preview the new `agx_arm_sim` Piper/Pika source routing
  - Targets:
    - `--target piper_only`
    - `--target pika_only`
    - `--target piper_pika_combo`
    - `--target all`
  - Example:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_agx_arm_sim_source.py --target all --output-root code_painting/output_agx_arm_sim_preview --video-frames 36 --fps 12`

## 2026-04-16 01:30:00 +08

- Updated `code_painting/visualize_original_source_urdfs.py`
  - Added lighting preset argument:
    - `--lighting {bright,dark}`
  - Dark stepwise validation example:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_original_source_urdfs.py --target both --lighting dark --output-root code_painting/output_original_source_urdf_preview_dark --video-frames 36 --fps 12`

## 2026-04-16 01:15:00 +08

- Updated `code_painting/visualize_piper_pika_single.py`
  - Added lighting preset argument:
    - `--lighting {bright,dark}`
  - Example dark-preview command:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --lighting dark --output-dir code_painting/output_piper_pika_preview_dark --image-name piper_pika_dark.png --video-name piper_pika_dark.mp4 --video-frames 36 --fps 12`

## 2026-04-16 01:00:00 +08

- Added original-source URDF preview command
  - Entry point:
    - `code_painting/visualize_original_source_urdfs.py`
  - Purpose:
    - preview the original Piper arm URDF and original Pika gripper URDF directly from the download folders
  - Main arguments:
    - `--target {piper_arm,pika_gripper,both}`
    - `--output-root`
    - `--video-frames`
    - `--fps`
  - Typical usage:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_original_source_urdfs.py --target both --output-root code_painting/output_original_source_urdf_preview --video-frames 30 --fps 12`

## 2026-04-16 00:45:00 +08

- Reused `code_painting/visualize_piper_pika_single.py` to export a DAE-based preview variant
  - Example command:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --output-dir code_painting/output_piper_pika_preview_dae --video-frames 24 --fps 12`
  - Purpose:
    - verify the arm appearance after switching visual meshes from STL to DAE

## 2026-04-16 00:30:00 +08

- Updated `code_painting/visualize_piper_pika_single.py`
  - Added preview-export support:
    - still image export
    - short mp4 export
  - Useful arguments:
    - `--offscreen-only`
    - `--save-image`
    - `--save-video`
    - `--video-frames`
    - `--fps`
    - `--output-dir`
  - Default exported files:
    - `code_painting/output_piper_pika_preview/piper_pika_preview.png`
    - `code_painting/output_piper_pika_preview/piper_pika_preview.mp4`

## 2026-04-16 00:00:00 +08

- Added standalone piper_pika visualization command
  - Entry point:
    - `code_painting/visualize_piper_pika_single.py`
  - Purpose:
    - load and visualize the new assembled `assets/embodiments/piper_pika/piper_pika.urdf`
  - Key arguments:
    - `--urdf`
    - `--offscreen-only`
  - Typical usage:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py`
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1`

## 2026-04-14 12:30:00 +08

- Updated `code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
  - Purpose:
    - prevent the reused R1 batch parser from injecting `robot_config_R1.json` into Piper V2 runs
  - New enforced defaults when absent from user CLI:
    - `--robot_config /home/zaijia001/ssd/RoboTwin/robot_config_Piper_dual_v2.json`
    - `--head_camera_local_quat_wxyz 1.0 0.0 0.0 0.0`
    - `--head_camera_local_pos 0.0 0.0 0.0`
  - Result:
    - printed batch command now points to the Piper V2 robot config instead of the R1 config

## 2026-04-14 12:00:00 +08

- Added true Piper V2 batch planning command: `bash code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh ...`
  - Entry points:
    - `code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh`
    - `code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
    - `code_painting/plan_anygrasp_keyframes_piper_v2.py`
  - Supporting files:
    - `code_painting/replay_piper_dual_h5.py`
    - `code_painting/render_hand_retarget_piper_dual_npz_urdfik.py`
    - `robot_config_Piper_dual_v2.json`
  - Purpose:
    - provide a real Piper-oriented dual-single-arm setup modeled after the existing UR configuration style
    - keep left/right Piper bases separate in viewer/replay/URDFIK execution
  - Key behavior:
    - uses `dual_arm_embodied=false`
    - loads two Piper URDF instances from `assets/embodiments/piper/piper.urdf`
    - separates them with `embodiment_dis=0.80`
    - effective base poses are `[-0.4, -0.65, 0.72]` and `[0.4, -0.65, 0.72]`
  - Typical usage:
    - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh <anygrasp_root> <replay_root> <hand_dir> <output_root> --planner_backend urdfik ...`

## 2026-04-14 00:00:00 +08

- Added Piper batch planning command: `bash code_painting/run_plan_anygrasp_keyframes_piper_batch.sh ...`
  - Entry points:
    - `code_painting/run_plan_anygrasp_keyframes_piper_batch.sh`
    - `code_painting/plan_anygrasp_keyframes_piper_batch.py`
    - `code_painting/plan_anygrasp_keyframes_piper.py`
  - Purpose:
    - provide a Piper-oriented entry while keeping the original R1 planner code unchanged
    - reuse the existing AnyGrasp batch launcher / single-video planner flow
  - Important behavior:
    - defaults to `robot_config_Piper_dual.json`
    - defaults `--head_camera_local_quat_wxyz 1 0 0 0` unless user overrides it
    - when `--planner_backend urdfik`, uses `assets/embodiments/piper/piper.urdf`
  - Related config:
    - `robot_config_Piper_dual.json`
  - Typical usage:
    - `bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_batch.sh <anygrasp_root> <replay_root> <hand_dir> <output_root> --planner_backend urdfik ...`

## 2026-04-03 00:00:00 +08

- Added a smooth-handling note: `agent-read/smooth/README.en.md`
  - Related commands:
    - `bash code_painting/run_plan_anygrasp_keyframes_r1_batch.sh ...`
    - `bash code_painting/run_replay_pose_debug_smooth.sh ...`
    - `bash code_painting/batch_smooth_planner_outputs.sh`
  - Main arguments discussed:
    - `--urdfik_trajectory_mode cartesian_interp_ik`
    - `--urdfik_cartesian_interp_steps`
    - `--urdfik_cartesian_interp_auto_step_m`
    - `--settle_steps`
    - `--joint_target_wait_steps`
    - `--replan_until_reached`
    - `--replan_until_reached_max_attempts`
  - Purpose:
    - explain the difference between planner smoothing, settle/wait behavior, and post-hoc replay smoothing
    - record how to balance exported-video smoothness against final execution accuracy without changing code
  - Expanded this round with:
    - an evaluation of the strengthened waypoint-IK idea: fixed 1 cm EE sampling + previous-solution seed + joint jump-threshold rejection
    - a comparison between that idea and the current `cartesian_interp_ik` implementation
    - a ranked difficulty discussion for several continuity / smoothing improvement options
    - a V7-debug-specific analysis of the tradeoff among try/replan, `joint_target_wait_steps`, final accuracy, and exported-video smoothness
    - a new implemented diagnostic metric: lateral distance from the current point to the target forward axis
    - new output field: `lat_cm`
    - an additional future direction: an object-relative target adapter (`T_obj_hand_demo + Δ_robot`)

## 2026-03-27 00:20:00 +08

- Added raw-planner-v7 wrapper command: `bash run_planner_v7_repaint_review_pi0.sh`
  - Entry point:
    - `run_planner_v7_repaint_review_pi0.sh`
  - Purpose:
    - run repaint directly on `anygrasp_plan_keyframes_realoffset_batch_pure-v7`
    - manually review repaint videos
    - then convert them into pi0 / robotwin processed_data
  - Key paths:
    - planner root: `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v7`
    - repaint root: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_v7`
    - review json: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue_v7/video_review.json`
    - processed_data: `/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-27-planner-v7`

- Added review-driven Step-2+Step-3+Step-4 controller command: `bash run_reviewed_smooth_repaint_pi0_pipeline.sh`
  - Entry point:
    - `run_reviewed_smooth_repaint_pi0_pipeline.sh`
  - Purpose:
    - extract ids labeled `y` from `video_review.json` (or include `m` by mode)
    - batch-run Step 2 smoothing, Step 3 smooth repaint, and Step 4 pi0 processing
  - Key arguments:
    - `TASK_NAME`
    - `REVIEW_JSON`
    - `REVIEW_MODE`
    - `RUN_SMOOTH`
    - `RUN_REPAINT`
    - `RUN_PI0`
    - `DRY_RUN`
    - `EXPERT_DATA_NUM`
    - `PI0_OUTPUT_DIR`
  - Key outputs:
    - `${SMOOTH_OUTPUT_ROOT}/${TASK_NAME}_${idx}/...`
    - `${REPAINT_OUTPUT_ROOT}/id_${idx}_head_cam_arm_gripper_cup_bottle_pad_target/target_with_original_head_cam_plan.mp4`
    - `${PI0_OUTPUT_DIR}/episode_*/episode_*.hdf5`

- Added smooth-bundle command: `bash code_painting/batch_smooth_planner_outputs.sh`
  - Entry point:
    - `code_painting/batch_smooth_planner_outputs.sh`
    - `code_painting/smooth_planner_outputs_from_pose_debug.py`
  - Purpose:
    - remove lingering frames and interpolate key states from the Step-1 planner outputs
  - Key arguments:
    - `INPUT_ROOT`
    - `OUTPUT_ROOT`
    - `INTERP_FACTOR`
    - `FPS`
    - `KEEP_HOVER_FRAMES_EVERY`
    - `DEDUP_POS_THRESH_M`
    - `DEDUP_ROT_THRESH_DEG`
    - `DEDUP_JOINT_THRESH_RAD`
    - `DEDUP_GRIPPER_THRESH`
  - Key outputs:
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/head_cam_plan.mp4`
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/left_wrist_cam_plan.mp4`
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/right_wrist_cam_plan.mp4`
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/pose_debug.jsonl`
    - `${OUTPUT_ROOT}/${TASK_NAME}_${idx}/smooth_summary.json`

## 2026-03-27 00:30:00 +08

- Added a planner-consistent pi0 data-conversion command: `process_repainted_planner_outputs.py`
  - Entry point:
    - `policy/pi0/scripts/process_repainted_planner_outputs.py`
  - Purpose:
    - generate `processed_data` HDF5 episodes from repaint planner head + planner wrist + planner pose_debug
  - Key arguments:
    - `--head-root`
    - `--head-dir-template`
    - `--head-video-name`
    - `--planner-root`
    - `--planner-dir-template`
    - `--left-wrist-video-name`
    - `--right-wrist-video-name`
    - `--pose-debug-name`
    - `--review-json`
    - `--review-mode`
    - `--ids`
    - `--ignore-ids`
  - Typical command:
    - `python scripts/process_repainted_planner_outputs.py d_pour_blue "pour water" 27 --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' --head-video-name target_with_original_head_cam_plan.mp4 --planner-root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3 --planner-dir-template 'd_pour_blue_{id}' --left-wrist-video-name left_wrist_cam_plan.mp4 --right-wrist-video-name right_wrist_cam_plan.mp4 --pose-debug-name pose_debug.jsonl --review-json /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json --review-mode strict --ignore-ids --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-27-planner`

## 2026-03-27 00:00:00 +08

- Added a new pi0 data-conversion command: `process_repainted_headcam_with_wrist.py`
  - Entry point:
    - `policy/pi0/scripts/process_repainted_headcam_with_wrist.py`
  - Purpose:
    - convert the newer head-cam repaint outputs plus retarget wrist replays into `processed_data` HDF5 episodes
  - Key arguments:
    - `--head-root`
    - `--head-dir-template`
    - `--head-video-name`
    - `--retarget-root`
    - `--retarget-dir-template`
    - `--review-json`
    - `--review-mode`
    - `--ids`
    - `--ignore-ids`
  - Typical command:
    - `python scripts/process_repainted_headcam_with_wrist.py d_pour_blue "pour water" 48 --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' --head-video-name target_with_original_head_cam_plan.mp4 --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue --retarget-dir-template 'hand_detections_{id}' --ignore-ids`

## 2026-03-25 19:15:00 +08

- Added visual-only base-occluder parameters
  - Flags:
    - `--base_occluder_enable 0|1`
    - `--base_occluder_local_pos X Y Z`
    - `--base_occluder_half_size HX HY HZ`
    - `--base_occluder_color R G B`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/render_hand_retarget_r1_npz.py`
  - Purpose:
    - add a white occluder that follows the robot base and hides the chassis/base in videos
    - the current implementation is visual-only and creates no collision
  - Usage notes:
    - useful for cleaning up pure/debug videos
    - `local_pos` controls height and local offset
    - `half_size` controls the occluder box dimensions

## 2026-03-25 18:55:00 +08

- R1 planner wrist-camera export semantics updated
  - Behavior:
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - no longer depend on post-export image rotation
    - instead use an R1-specific wrist local pose inside `plan_anygrasp_keyframes_r1.py` that matches `galaxea_sim/robots/r1.py`
    - output size returns to the original landscape `image_width x image_height`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Notes:
    - this override is specific to the R1 planner path and does not change the global default in `render_hand_retarget_r1_npz.py`
    - the goal is to obtain the correct wrist view from the mounted camera pose itself rather than from an export-time image rotation patch

## 2026-03-25 18:35:00 +08

- Planner wrist-video export fine-tuned again
  - Behavior:
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - now use a uniform `90°` CCW rotation before export
    - writer dimensions match the rotated frames
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Notes:
    - this change is based on the user's actual result: the previous `180°` fix still looked like the correct view rotated `90°` CCW
    - no new CLI parameter was added

## 2026-03-25 18:20:00 +08

- Planner wrist-video export corrected again
  - Behavior:
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - no longer use a `90°` rotation; they now use a uniform `180°` in-plane rotation before export
    - output size returns to landscape `image_width x image_height`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Notes:
    - no new CLI parameter was added
    - this correction is based on the user's actual exported result: the previous version produced portrait videos that were still upside down
    - the new fix only changes image-plane export orientation, not the camera mount or planner coordinate conventions

## 2026-03-25 16:45:00 +08

- `--debug_visualize_ik_waypoints 1`
  - Visualization enhancement:
    - now shows start and goal markers in addition to intermediate TCP waypoints
    - both start and goal use red point+forward-axis markers
    - intermediate waypoint markers are smaller so the hands, target axes, and path are easier to inspect together
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Usage:
    - command syntax is unchanged; keep using `--debug_visualize_ik_waypoints 1`
    - this flag only affects viewer/debug visualization and does not change planning or execution logic

## 2026-03-25 17:10:00 +08

- Planner wrist-video orientation correction
  - Behavior:
    - `left_wrist_cam_plan.mp4`
    - `right_wrist_cam_plan.mp4`
    - are now rotated 90 degrees clockwise before being written
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Notes:
    - no new CLI parameter was added
    - this only affects planner wrist-video orientation
    - it does not change world camera pose or planner coordinate definitions

## 2026-03-25 12:08:00 +08

- Added flag: `--enable_grasp_action_object_collision 0|1`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/run_plan_anygrasp_keyframes_r1_batch.sh`
  - Purpose:
    - enable collision blocking for the selected execution object during `close_gripper` and `action`
    - default `0` preserves the previous no-collision execution mode
  - Usage:
    - single-run command: append `--enable_grasp_action_object_collision 1`
    - batch: append `--enable_grasp_action_object_collision 1` to the batch command as well
  - Notes:
    - this flag does not change target-pose construction for `pregrasp/grasp/action`
    - it also does not change the relative transform used when attaching the object to TCP
## 2026-03-25 13:05:00 +08

- Added visualization-mode flags to `plan_anygrasp_keyframes_r1.py`:
  - New flags:
    - `--debug_visualize_targets 0|1`
    - `--viewer_show_camera_frustums 0|1`
  - Usage:
    - `debug_visualize_targets=0` disables target-axis actors globally
    - `viewer_show_camera_frustums=0` disables SAPIEN camera frustum lines in the interactive viewer
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`

- Synced the same visualization-mode flags through `plan_anygrasp_keyframes_r1_batch.py`:
  - Forwarded flags:
    - `--debug_visualize_targets`
    - `--viewer_show_camera_frustums`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-25 13:35:00 +08

- `--enable_grasp_action_object_collision 1`
  - Behavior enhancement:
    - The `close_gripper` stage no longer always closes fully in one shot
    - It now closes progressively and stops early once contact with the selected object is present and gripper joint motion has stalled
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`

## 2026-03-25 14:10:00 +08

- Added parameter: `--urdfik_cartesian_interp_auto_step_m`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Only active when `--urdfik_cartesian_interp_steps=-1`, controlling the translation threshold of automatic waypoint mode.
  - Default:
    - `0.05`
  - Example:
    - `--urdfik_cartesian_interp_steps -1 --urdfik_cartesian_interp_auto_step_m 0.03`
  - Note:
    - Smaller values create denser intermediate waypoints.
  - Usage:
    - Command syntax is unchanged; keep using `--enable_grasp_action_object_collision 1`
    - With the default `0`, the original fast no-collision closing behavior is preserved

## 2026-03-25 14:05:00 +08

- `--urdfik_cartesian_interp_steps`
  - New convention:
    - `-1` enables automatic waypoint mode
  - Automatic rule:
    - no intermediate waypoint when translation is `<= 0.05m`
    - add one intermediate waypoint for each additional `0.05m` bucket beyond that threshold
  - Example:
    - `--urdfik_cartesian_interp_steps -1`
  - Related code:
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`

## 2026-03-25 14:25:00 +08

- `planner_backend=urdfik` + `urdfik_trajectory_mode=cartesian_interp_ik`
  - Behavior fix:
    - the execution layer now truly consumes the full `joint_waypoints` stored in `plan["position"]`
    - it no longer executes only a straight endpoint interpolation from `current_joints` to `target_joints`
  - Related code:
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`
# 2026-03-25

- `--pure_scene_output 1`
  - Behavior update:
    - no longer generates `debug_selection_preview.mp4`
    - now keeps and writes:
      - `head_cam_plan.mp4`
      - `left_wrist_cam_plan.mp4`
      - `right_wrist_cam_plan.mp4`
    - auto-enables `pose_debug.jsonl`
  - Main `pose_debug.jsonl` fields:
    - `record_index`
    - `stage`
    - `active_frame`
    - `current_*_camera_pose_world_wxyz`
    - `current_*_tcp_pose_world_wxyz`
    - `current_*_ee_pose_world_wxyz`
    - `current_*_arm_qpos_rad`
    - `current_*_gripper_joint_qpos_rad`
    - `object_actor_poses`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Reference docs:
    - `agent-read/2026-03-25_pure_mode_outputs_ZH.md`
    - `agent-read/2026-03-25_pure_mode_outputs.md`

- Added command parameter: `--debug_visualize_ik_waypoints`
  - Entry points:
    - `/home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `/home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Show the intermediate TCP/EE smoothing waypoints of `cartesian_interp_ik` in debug/viewer output, helping distinguish “bad waypoint generation” from “bad IK/execution follow-through”.
  - Display:
    - Position point plus local forward-axis marker for each intermediate waypoint.
  - Default:
    - `0`

- Added command parameter: `--debug_collision_report`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Print stronger collision debug information during the progressive `close_gripper` stage.
  - Main output:
    - `[collision-debug-init]`
    - `[collision-debug-step]`
    - regular `[gripper-close]` now also includes `base_contact=...`
  - Debug focus:
    - distinguish `finger_contact` from `base_contact`
    - inspect `finger_pairs` / `base_pairs`
    - inspect collision-shape summaries of the target object, `left/right_gripper_link`, and finger links
  - Default:
    - `0`

- Added command parameter: `--execution_object_collision_mode {convex,solid_bbox}`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Control which collision geometry execution objects use at runtime.
  - Modes:
    - `convex`
      - keeps the current `add_convex_collision_from_file`
    - `solid_bbox`
      - reads mesh bounds
      - creates one axis-aligned solid box collision
  - Notes:
    - affects execution collision only, not the visual mesh
    - if `--replay_objects_ignore_collision 1` and the object is not enabled for grasp/action collision, collision is still omitted
  - Default:
    - `convex`

- Added command parameter: `--gripper_contact_monitor_mode {fingers,fingers_and_base,all_robot_links}`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - Control which robot links are allowed to trigger contact monitoring during `close_gripper`.
  - Modes:
    - `fingers`
      - finger links only
    - `fingers_and_base`
      - finger links plus `left/right_gripper_link`
    - `all_robot_links`
      - all links of the articulation corresponding to the current arm
  - Notes:
    - this changes the monitoring set used by the early-stop logic, not just the printed debug output
    - it is especially useful for debugging cases where finger/base links report `shapes=0` but other links may still carry collision

## 2026-03-25 23:10:00 +08

- Added a minimal collision-probe command:
  - Entry script:
    - `code_painting/minimal_gripper_collision_probe.py`
  - Purpose:
    - verify raw scene contacts between the R1 gripper and either a simple box or a mesh (`solid_bbox` / `convex`) object without going through the AnyGrasp/IK/stage main pipeline
  - Representative commands:
    - box:
      - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/minimal_gripper_collision_probe.py --arm left --object_kind box --probe_local_offset 0.04 0.0 0.0 --max_iters 20 --settle_steps_per_iter 8`
    - mesh:
      - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/minimal_gripper_collision_probe.py --arm left --object_kind mesh --mesh_path /home/zaijia001/ssd/data/R1/hand/obj_mesh/blue_cup/blue_cup.obj --mesh_collision_mode solid_bbox --probe_local_offset 0.04 0.0 0.0 --max_iters 20 --settle_steps_per_iter 8`
  - Key outputs:
    - step-by-step terminal logs of:
      - `qpos`
      - `raw_contact_total`
      - `target_contact_total`
      - `target_contacts`
    - JSON outputs:
      - `code_painting/minimal_gripper_collision_probe/probe_left_box.json`
      - `code_painting/minimal_gripper_collision_probe/probe_left_mesh.json`

- Enhanced `--debug_collision_report 1` output:
  - added raw target-contact reporting during the close stage:
    - `raw_target_contacts`
    - `raw_target_contact_total`
    - `[gripper-close] ... raw_target_contact=0|1`
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Purpose:
    - distinguish “the monitor/helper failed to match a contact” from “raw physics contact does not exist at all”

- Further enhanced `--debug_collision_report 1` output:
  - Added:
    - `target_pose=...`
    - `target_collision_debug=...`
  - Purpose:
    - directly inspect whether the target-object actor pose stays stable during close
    - directly inspect the `solid_bbox` `center/half_size`

- Added flag: `--debug_visualize_object_collision_bbox 0|1`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Purpose:
    - when `execution_object_collision_mode=solid_bbox`, additionally display the object's collision bbox
  - Typical usage:
    - append to the existing command:
      - `--debug_visualize_object_collision_bbox 1`

- Added flag: `--grasp_action_object_collision_start_stage {close_gripper,grasp,pregrasp}`
  - Purpose:
    - control from which stage selected execution objects start participating in collision
  - Typical experiment:
    - `--enable_grasp_action_object_collision 1 --grasp_action_object_collision_start_stage pregrasp --execution_object_collision_mode convex`

- Added pre-close pose export:
  - Output files:
    - `close_stage_snapshot_dual_before_close.json`
    - `close_stage_snapshot_<arm>_before_close.json`

- Added flag: `--execution_object_scale_override NAME=S|SX,SY,SZ`
  - Purpose:
    - independently scale an execution object's visual mesh and collision geometry together
  - Typical examples:
    - `--execution_object_scale_override cup=0.9`
    - `--execution_object_scale_override bottle=0.9`

- Added flags:
  - `--execution_object_visual_scale_override NAME=S|SX,SY,SZ`
  - `--execution_object_collision_scale_override NAME=S|SX,SY,SZ`
  - Purpose: independently control execution-object visual-mesh scale and collision-shape scale
