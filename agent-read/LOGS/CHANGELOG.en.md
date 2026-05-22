# CHANGELOG.en

## 2026-04-29 (Piper orientation guess: images-only + forward/up offset fix)

- Updated `code_painting/run_piper_gripper_standard_pose_guess.sh`:
  - output is now images-only (`zed/third` PNGs + `index.csv` + `world_targets_and_status.npz`), replay mp4 files are removed automatically.
  - added default target offset `target_world_offset_xyz=(0.0, +0.1, +0.1)` to improve IK reachability.
  - `index.csv` now includes `left_status/right_status` to separate IK-failure from orientation-definition issues.
- Validation (frame0):
  - with offset, standard 8 cases are no longer all Fail (e.g. `backward_guess/open_left_right_guess` become reachable).
  - output board ready for manual semantic labeling at `.../output_piper_gripper_standard_pose_guess_check2/board/`.

## 2026-04-29 (Piper gripper standard-orientation guess board tool)

- Added script: `code_painting/run_piper_gripper_standard_pose_guess.sh`
  - Purpose: generate 8 canonical orientation guesses (front/back/left/right + opening-axis up/down/left/right) on a fixed frame.
  - Runs one-frame replay per case and extracts the first `zed_replay.mp4` frame into a single `board/` folder.
  - Produces `board/index.csv` for manual semantic labeling.
- Added script: `code_painting/run_piper_gripper_orientation_guess_board.sh`
  - Purpose: orientation-sweep-based candidate dump for deeper debugging.
- Updated command doc: `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
  - Added D6 with one-command generation of the Piper orientation guess board.

## 2026-04-29 (HaMeR all-zero detection debug)

- Symptom: `detect_hands_realr1.py` completed on GPU under `hamer-r1` but produced `0/0` detections for all frames, and overwrote previous `hand_detections_0.npz`.
- Input-chain validation passed:
  - `pnp_star_pear_hamer_input` contains complete `rgb_0..15.mp4 / params_0..15.json`.
  - Extracted RGB frames clearly show both hands (not blank/black input).
  - `params_0.json` intrinsics are valid (`fx/fy/cx/cy/width/height`).
- Root cause: GPU commands in docs were using `hamer-r1` (CPU-safe env). On Blackwell GPUs this can hit CUDA-arch mismatch behavior and silently degrade per-frame hand inference.
- Fix verified: switching to `hamer-r1-gpu` (with `unset LD_LIBRARY_PATH`) restored detection on `video_id=0` to:
  - Left: `128/128`
  - Right: `128/128`
  - Both: `128/128`
  - Output dir: `pnp_star_pear_hamer_output_dbg_gpuenv`
- Docs updated: `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
  - A2 GPU commands now use `conda run -n hamer-r1-gpu ... --device cuda`
  - Added quick detection-count check and debug baseline command.

## 2026-04-27 (Piper object replay: head-camera link fallback fix)

- Fixed `code_painting/replay_r1_h5.py`:
  - no longer exits when R1-style `zed_link/head_camera` is missing in non-R1 robot configs.
  - added fallback head-camera pose computed from `robot_base_pose + head_camera_local_*`.
- Fixed `code_painting/render_multi_object_pose_r1_npz_batch.py`:
  - added and forwarded `--save_pose_debug` so batch mode accepts the flag.
- Updated command docs:
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md`
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline.md`
  - replay-stage mesh override name normalized to `star_fruit=...` to match object folder naming.

## 2026-04-27 (added stage I/O formats + per-object replay)

- Updated `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md` and its English pair to include:
  - HaMeR stage input/output roots and key format fields
  - FoundationPose stage input/output roots and key format fields
  - FoundationPose object folder naming (`pear`, `star_fruit`)
- Added split replay commands for trajectory/pose inspection:
  - pear-only replay
  - star_fruit-only replay
- Added replay output notes:
  - `head_cam_replay.mp4`
  - `multi_object_world_poses.npz`
  - `pose_debug.jsonl`

## 2026-04-27 (FoundationPose prompt fix + tmux exit root-cause)

- Conclusion: prompt `star` fails Grounding DINO init on this dataset; `star fruit` enters tracking correctly.
- Updated command docs:
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md`
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline.md`
  - `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
- Recorded tmux pane-exit root cause: `set -e` in `source_foundationpose_env.sh` propagates to caller shell and exits pane on non-zero commands.

## 2026-04-27 (pnp_star_pear: FoundationPose pear+star stage completed)

- Added a Piper-specific FoundationPose preparation script:
  - `/home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py`
- Added a dedicated pear+star FoundationPose runner:
  - `/home/zaijia001/FoundationPose/run_piper_star_pear_foundation.sh`
- Updated workflow docs:
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md`
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline.md`
- Added full command chain for "HaMeR -> FoundationPose(pear+star) -> RoboTwin replay".
- Validation:
  - converted all 16 episodes into `pnp_star_pear_foundation_input` (with metric `depth_<id>/*.npy`).

## 2026-04-16 (added viewer support for calibrated scene scripts)

- Updated `code_painting/pika/visualize_calibrated_piper_pika_scene.py`
- Updated `code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`
- Added interactive viewer support:
  - `--viewer 1`
  - `--viewer-camera overview|head`
- This fixes the previous error: `unrecognized arguments: --viewer 1`.
- Updated command-library docs to include calibrated-scene viewer commands.


## 2026-04-24 (Piper hand dataset -> HaMeR -> RoboTwin documentation refresh)

- Added docs:
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline_ZH.md`
  - `agent-read/2026-04-24_piper_hamer_hand_pipeline.md`
- Updated index:
  - `agent-read/README.md` (new links added)
- Documented the full runnable path for the new dataset layout (`episode*/camera/color|depth/headD435`):
  - convert to `rgb_<id>.mp4/depth_<id>.mp4/params_<id>.json`
  - run HaMeR to produce `hand_detections_<id>.npz`
  - visualize with `hand_vis_<id>.mp4`
  - run RoboTwin downstream replay command
- Validation:
  - `video_id=0` produced `hand_vis_0.mp4` and `hand_vis_gripper_0.mp4`

## 2026-04-16 (command-library clarification for headcam/wrist)

- Updated `agent-read/COMMANDS/pika_scene_commands.en.md`
- Updated `agent-read/COMMANDS/pika_scene_commands.zh.md`
- Clarified exactly which commands export head-cam images.
- Clarified that wrist-view export is not implemented yet in the calibrated-scene scripts.


## 2026-04-16 (command-doc update for viewer/headcam clarification)

- Updated `agent-read/COMMANDS/pika_scene_commands.en.md`
- Updated `agent-read/COMMANDS/pika_scene_commands.zh.md`
- Clarified that:
  - calibrated scene scripts do use the calibration bundle for the second robot and head camera
  - the head camera may be hard to notice in the overview because it is currently rendered only as a small marker with RGB axes
  - only the manual tabletop script currently supports interactive viewer mode


## 2026-04-16 (pika command library + base-on-side analysis)

- Added command-library docs under `agent-read/COMMANDS/`:
  - `README.en.md`
  - `README.zh.md`
  - `pika_scene_commands.en.md`
  - `pika_scene_commands.zh.md`
- Added copyable commands for:
  - interactive viewer runs of the manual tabletop scene
  - offscreen re-export of manual tabletop previews
  - calibrated scene export
  - calibrated version-B export
- Analyzed why the version-B bases look attached to the table side:
  - version B rotates the table convention by 90 degrees
  - then reuses the previous manual inset magnitude as world-x inset
  - so the bases are intentionally placed close to the rotated table side face rather than the former front edge


## 2026-04-16 (calibrated scene version B)

- Implemented version B scene alignment under `code_painting/pika/`:
  - removed the earlier +90deg anchor rotation on the first robot
  - kept the calibration left/right spread aligned with world y as much as possible
  - rotated the table convention instead of rotating the anchor robot
- Added script:
  - `code_painting/pika/visualize_calibrated_piper_pika_scene_vb.py`
- Generated outputs:
  - `code_painting/pika/output_calibrated_scene_vb/calibrated_scene_vb_overview.png`
  - `code_painting/pika/output_calibrated_scene_vb/calibrated_scene_vb_overview.mp4`
  - `code_painting/pika/output_calibrated_scene_vb/calibrated_scene_vb_headcam.png`
- Explicitly documented which fields from `robot_config_PiperPika_agx_dual_table.json` are reused and which are intentionally ignored in version B.


## 2026-04-16 (calibrated real-scene reconstruction in code_painting/pika)

- Read real-scene calibration inputs:
  - `CALIBRATION_TRANSFORMS_README.md`
  - `calibration_bundle_try2.json`
- Reconstructed a simulated scene under `code_painting/pika/` using:
  - first robot = current manual tabletop placement from `robot_config_PiperPika_agx_dual_table.json`
  - second robot = `left_base_T_right_base`
  - head camera = `left_base_T_head_camera`
- Added script:
  - `code_painting/pika/visualize_calibrated_piper_pika_scene.py`
- Generated outputs:
  - `code_painting/pika/output_calibrated_scene/calibrated_scene_overview.png`
  - `code_painting/pika/output_calibrated_scene/calibrated_scene_overview.mp4`
  - `code_painting/pika/output_calibrated_scene/calibrated_scene_headcam.png`
- Validation printed derived world poses for left base, right base, and head camera.


## 2026-04-16 (edge-mount base correction)

- Confirmed that the previous dual-table configuration was not mounted on the tabletop:
  - table long-edge line was `y = -0.30`
  - robot bases were at `y = -0.60`
  - so the bases were outside the table even though `z = 0.75` matched tabletop height
- Updated `robot_config_PiperPika_agx_dual_table.json`
  - moved the shared base pose from `y = -0.60` to `y = -0.30`
  - kept the corrected tabletop-facing quaternion
- New validated base poses:
  - left `[-0.30, -0.30, 0.75]`
  - right `[0.30, -0.30, 0.75]`
- Generated edge-mounted preview:
  - `code_painting/output_piper_pika_agx_dual_table_edge_mount/piper_pika_agx_dual_table_edge_mount.png`
  - `code_painting/output_piper_pika_agx_dual_table_edge_mount/piper_pika_agx_dual_table_edge_mount.mp4`


## 2026-04-16 (dual table orientation + oblique camera fix)

- Checked the existing UR-style camera reference in RoboTwin:
  - `code_painting/replay_piper_dual_h5.py` uses a fixed overview/head camera fallback
  - `code_painting/render_hand_retarget_r1_npz.py` builds third-person views from robot forward + world up
- Diagnosed the table-layout orientation issue:
  - the previous config used identity quaternion `[1, 0, 0, 0]`
  - that made the robot front align with `+x`, i.e. parallel to the table long edge
  - this was inconsistent with the intended use of approaching tabletop objects from one long-edge side
- Fixed `robot_config_PiperPika_agx_dual_table.json`
  - updated base quaternion to `[0.70710678, 0.0, 0.0, 0.70710678]`
  - both robots now face `+y`
- Updated `code_painting/visualize_piper_pika_agx_dual_table.py`
  - corrected oblique camera to sit behind the robots and look toward the tabletop
- Generated corrected outputs:
  - `code_painting/output_piper_pika_agx_dual_table_oblique_fixed/piper_pika_agx_dual_table_oblique_fixed.png`
  - `code_painting/output_piper_pika_agx_dual_table_oblique_fixed/piper_pika_agx_dual_table_oblique_fixed.mp4`
  - `code_painting/output_piper_pika_agx_dual_table_topdown_fixed/piper_pika_agx_dual_table_topdown_fixed.png`
  - `code_painting/output_piper_pika_agx_dual_table_topdown_fixed/piper_pika_agx_dual_table_topdown_fixed.mp4`


## 2026-04-16 (dual table top-down camera adjustment)

- Refined the table-edge interpretation for the dual Piper+Pika layout:
  - both robots are on one long-edge side of the table
  - bases are 0.30 m from the two short-side ends
  - base-to-base spacing is 0.60 m
- Updated `code_painting/visualize_piper_pika_agx_dual_table.py`
  - added `--camera-mode {top_down,oblique}`
  - temporary top-down camera is placed above the midpoint between the two bases and looks downward to the tabletop
- Generated temporary top-down outputs:
  - `code_painting/output_piper_pika_agx_dual_table_topdown/piper_pika_agx_dual_table_topdown.png`
  - `code_painting/output_piper_pika_agx_dual_table_topdown/piper_pika_agx_dual_table_topdown.mp4`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1 --camera-mode top_down --output-dir code_painting/output_piper_pika_agx_dual_table_topdown --image-name piper_pika_agx_dual_table_topdown.png --video-name piper_pika_agx_dual_table_topdown.mp4`


## 2026-04-16 (piper_pika_agx dual table layout)

- Added a new colored combo embodiment for the newer Piper+Pika appearance:
  - `assets/embodiments/piper_pika_agx/piper_pika_agx.urdf`
- The new combo uses:
  - Piper arm from the original DAE-based Piper source
  - Pika gripper from `agx_arm_sim` `pika2_gripper.urdf` + DAE meshes
- Added dual-arm layout config for a 120x60x75 cm table:
  - `robot_config_PiperPika_agx_dual_table.json`
- Layout assumption used in this round:
  - symmetric UR-style split
  - shared base pose before split: `[0.0, -0.60, 0.75]`
  - `embodiment_dis = 0.60`
  - effective bases:
    - left `[-0.30, -0.60, 0.75]`
    - right `[0.30, -0.60, 0.75]`
- Added preview script:
  - `code_painting/visualize_piper_pika_agx_dual_table.py`
- Generated outputs:
  - `code_painting/output_piper_pika_agx_dual_table/piper_pika_agx_dual_table.png`
  - `code_painting/output_piper_pika_agx_dual_table/piper_pika_agx_dual_table.mp4`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_agx_dual_table.py --offscreen-only 1`


## 2026-04-16 (agx_arm_sim source inspection)

- Inspected the newly introduced repository:
  - `/home/zaijia001/Downloads/agx_arm_sim`
- Confirmed that:
  - `ros2 launch agx_arm_description display.launch.py arm_type:=piper end_effector:=pika`
  - routes to a **Piper + Pika** combined model
- Found an important repository state issue:
  - `agx_arm_description/agx_arm_urdf/` is empty in the current checkout
  - so the xacro-referenced Piper arm assets are missing locally from this repo snapshot
- Confirmed that the new `pika2_gripper.urdf` uses DAE meshes with embedded color/material data
- Added preview script:
  - `code_painting/visualize_agx_arm_sim_source.py`
- Generated preview outputs:
  - `code_painting/output_agx_arm_sim_preview/piper_only.png`
  - `code_painting/output_agx_arm_sim_preview/piper_only.mp4`
  - `code_painting/output_agx_arm_sim_preview/pika_only.png`
  - `code_painting/output_agx_arm_sim_preview/pika_only.mp4`
  - `code_painting/output_agx_arm_sim_preview/piper_pika_combo.png`
  - `code_painting/output_agx_arm_sim_preview/piper_pika_combo.mp4`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_agx_arm_sim_source.py --target all --output-root code_painting/output_agx_arm_sim_preview --video-frames 36 --fps 12`


## 2026-04-16 (Piper color diagnosis and quantification)

- Added a diagnosis-only documentation pass for the current observation:
  - original Piper alone looks gray
  - original Pika alone looks white
  - combined `piper_pika` looks whitened
- Confirmed source evidence:
  - Piper DAE contains intrinsic dark-gray diffuse color (`0.113725 0.113725 0.113725`)
  - combined `piper_pika.urdf` still contains explicit light-colored URDF material blocks on Piper arm links (`rgba="0.792156862745098 0.819607843137255 0.933333333333333 1"`)
- Added approximate rendered-image color statistics comparing:
  - original Piper bright
  - original Piper dark
  - combined bright
  - combined dark
- Main conclusion:
  - the user's summary is correct: Piper's original gray comes from DAE, and the combined model is very likely whitened by the URDF material override


## 2026-04-16 (dark stepwise validation)

- Added a stepwise dark-lighting validation pass for:
  - original Piper arm
  - original Pika gripper
  - combined `piper_pika`
- Added/updated scripts:
  - `code_painting/visualize_original_source_urdfs.py` now supports `--lighting {bright,dark}`
  - `code_painting/visualize_piper_pika_single.py` already supports `--lighting {bright,dark}`
- Generated outputs:
  - `code_painting/output_original_source_urdf_preview_dark/piper_arm.png`
  - `code_painting/output_original_source_urdf_preview_dark/piper_arm.mp4`
  - `code_painting/output_original_source_urdf_preview_dark/pika_gripper.png`
  - `code_painting/output_original_source_urdf_preview_dark/pika_gripper.mp4`
  - `code_painting/output_piper_pika_preview_dark/piper_pika_dark.png`
  - `code_painting/output_piper_pika_preview_dark/piper_pika_dark.mp4`
- Main conclusion:
  - the combined model whitening is likely caused by explicit light-colored URDF material blocks still present on the Piper arm links inside `assets/embodiments/piper_pika/piper_pika.urdf`, not only by lighting


## 2026-04-16 (piper_pika dark preview)

- Updated `code_painting/visualize_piper_pika_single.py` to support two lighting presets:
  - `bright`
  - `dark`
- Generated a dark-lighting preview for the current combined model (colored Piper arm + white Pika gripper):
  - `code_painting/output_piper_pika_preview_dark/piper_pika_dark.png`
  - `code_painting/output_piper_pika_preview_dark/piper_pika_dark.mp4`
- Purpose:
  - reduce over-bright lighting so the original deeper Piper arm tone is easier to inspect
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --lighting dark --output-dir code_painting/output_piper_pika_preview_dark --image-name piper_pika_dark.png --video-name piper_pika_dark.mp4 --video-frames 36 --fps 12`


## 2026-04-16 (original source URDF preview comparison)

- Added `code_painting/visualize_original_source_urdfs.py` to preview the original source URDFs directly from the download folders.
- Tested sources:
  - `/home/zaijia001/Downloads/agx_arm_urdf/piper/urdf/piper_description.urdf`
  - `/home/zaijia001/Downloads/pika_ros/src/pika_gripper_description/urdf/pika_gripper_description.urdf`
- Generated outputs:
  - `code_painting/output_original_source_urdf_preview/piper_arm.png`
  - `code_painting/output_original_source_urdf_preview/piper_arm.mp4`
  - `code_painting/output_original_source_urdf_preview/pika_gripper.png`
  - `code_painting/output_original_source_urdf_preview/pika_gripper.mp4`
- Purpose:
  - compare the appearance of the original Piper arm and original Pika gripper before further assembly edits
  - verify whether the whiteness comes from the original source assets themselves
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_original_source_urdfs.py --target both --output-root code_painting/output_original_source_urdf_preview --video-frames 30 --fps 12`


## 2026-04-16 (piper_pika DAE color recovery)

- Switched the assembled `assets/embodiments/piper_pika/piper_pika.urdf` arm visual meshes back to DAE to better recover the original Piper arm appearance.
- Added copied DAE assets under:
  - `assets/embodiments/piper_pika/meshes/dae/*`
- Color-source findings:
  - the original Piper arm color appears to be embedded in the Collada DAE files under `/home/zaijia001/Downloads/agx_arm_urdf/piper/meshes/dae/`
  - the checked Pika gripper source tree did not contain DAE/OBJ/MTL/texture assets; only STL meshes plus white URDF material values were found
- Generated new DAE-based preview outputs:
  - `code_painting/output_piper_pika_preview_dae/piper_pika_preview.png`
  - `code_painting/output_piper_pika_preview_dae/piper_pika_preview.mp4`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --output-dir code_painting/output_piper_pika_preview_dae --video-frames 24 --fps 12`


## 2026-04-16 (piper_pika preview export)

- Enhanced `code_painting/visualize_piper_pika_single.py`.
- Added:
  - a fixed preview camera position that clearly sees the whole single arm
  - still image export
  - short preview video export with a small joint-space motion
- Default outputs:
  - `code_painting/output_piper_pika_preview/piper_pika_preview.png`
  - `code_painting/output_piper_pika_preview/piper_pika_preview.mp4`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/visualize_piper_pika_single.py`
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1 --video-frames 24 --fps 12`
  - confirmed both preview files were written successfully


## 2026-04-16

- Added a new standalone assembled URDF under the original assets tree:
  - `assets/embodiments/piper_pika/piper_pika.urdf`
  - `assets/embodiments/piper_pika/meshes/*`
- Source inputs:
  - arm URDF/meshes from `/home/zaijia001/Downloads/agx_arm_urdf/piper/`
  - gripper URDF/meshes from `/home/zaijia001/Downloads/pika_ros/src/pika_gripper_description/`
- Assembly notes:
  - used the existing combined reference `piper_pika_gripper_description.urdf` as the starting point
  - converted package mesh URIs to local relative `meshes/...` paths
  - renamed the robot to `piper_pika`
  - renamed `dummy_link` -> `piper_pika_dummy_link`
  - renamed `gripper_base` -> `pika_gripper_base`
  - copied required arm and gripper meshes into the new embodiment folder
- Added a minimal single-arm visualization script:
  - `code_painting/visualize_piper_pika_single.py`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/visualize_piper_pika_single.py`
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/visualize_piper_pika_single.py --offscreen-only 1`
  - offscreen load confirmed active joints:
    - `joint1 joint2 joint3 joint4 joint5 joint6 joint7 joint8`
- Repository note:
  - `assets/*` is currently ignored by `.gitignore`, so the new embodiment files exist locally but do not show up in `git status`


## 2026-04-14 (Piper V2 batch fix)

- Fixed Piper V2 batch argument pollution in `code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`.
- Root cause:
  - the reused `plan_anygrasp_keyframes_r1_batch.py` parser was still supplying its own default `--robot_config robot_config_R1.json`
  - as a result, batch mode launched the Piper V2 single-video script with an explicit R1 config, so viewer/rendering still showed R1-style behavior
- Fix:
  - inject Piper defaults before calling the reused batch launcher:
    - `--robot_config /home/zaijia001/ssd/RoboTwin/robot_config_Piper_dual_v2.json`
    - `--head_camera_local_quat_wxyz 1.0 0.0 0.0 0.0`
    - `--head_camera_local_pos 0.0 0.0 0.0`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
  - command probe confirmed printed batch command now contains:
    - `--robot_config /home/zaijia001/ssd/RoboTwin/robot_config_Piper_dual_v2.json`
    - `--head_camera_local_quat_wxyz 1.0 0.0 0.0 0.0`


## 2026-04-14

- Added a true Piper V2 dual-arm-style implementation following the existing UR dual-single-arm setup, without modifying any R1 / R1 Pro files.
- Added files:
  - `robot_config_Piper_dual_v2.json`
  - `code_painting/replay_piper_dual_h5.py`
  - `code_painting/render_hand_retarget_piper_dual_npz_urdfik.py`
  - `code_painting/plan_anygrasp_keyframes_piper_v2.py`
  - `code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
  - `code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh`
  - `agent-read/V2.0_piper_dual_ur_style.md`
  - `agent-read/V2.0_piper_dual_ur_style_ZH.md`
- V2 implementation notes:
  - uses `dual_arm_embodied=false` and two independent Piper URDF instances
  - keeps separate left/right base poses instead of collapsing both arms back to one shared root pose
  - uses a dedicated Piper replay renderer and a dedicated Piper URDFIK renderer
  - uses `assets/embodiments/piper/piper.urdf` for both left and right URDF loading and IK
  - validated effective base poses:
    - left = `[-0.4, -0.65, 0.72]`
    - right = `[0.4, -0.65, 0.72]`
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/replay_piper_dual_h5.py code_painting/render_hand_retarget_piper_dual_npz_urdfik.py code_painting/plan_anygrasp_keyframes_piper_v2.py code_painting/plan_anygrasp_keyframes_piper_v2_batch.py`
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_v2_batch.sh`
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/plan_anygrasp_keyframes_piper_v2.py --help`
  - direct config probe confirmed:
    - `left_urdf_path=./assets/embodiments/piper/piper.urdf`
    - `right_urdf_path=./assets/embodiments/piper/piper.urdf`
    - `left_origin=[-0.4, -0.65, 0.72]`
    - `right_origin=[0.4, -0.65, 0.72]`
  - renderer probe confirmed:
    - `[piper-v2-bases] left=[-0.4, -0.65, 0.72] right=[0.4, -0.65, 0.72]`


## 2026-04-14

- Added a Piper-compatible AnyGrasp planner wrapper without modifying the original R1 / R1 Pro planner files.
- Added files:
  - `robot_config_Piper_dual.json`
  - `code_painting/plan_anygrasp_keyframes_piper.py`
  - `code_painting/plan_anygrasp_keyframes_piper_batch.py`
  - `code_painting/run_plan_anygrasp_keyframes_piper_batch.sh`
  - `agent-read/2026-04-14_piper_anygrasp_wrapper.md`
  - `agent-read/2026-04-14_piper_anygrasp_wrapper_ZH.md`
- Implementation notes:
  - keeps the original `plan_anygrasp_keyframes_r1.py` code path untouched
  - injects a Piper robot config at runtime
  - swaps the replay / URDFIK renderer classes to Piper-specific adapters
  - uses `assets/embodiments/piper/piper.urdf` for URDF IK
  - maps the existing left/right execution structure onto two Piper instances for compatibility
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_piper.py code_painting/plan_anygrasp_keyframes_piper_batch.py`
  - `bash -n code_painting/run_plan_anygrasp_keyframes_piper_batch.sh`
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/plan_anygrasp_keyframes_piper.py --help`


## 2026-04-03

- Added smooth-focused documentation:
  - `agent-read/smooth/README.zh.md`
  - `agent-read/smooth/README.en.md`
- Purpose:
  - document the current smooth-related handling in the AnyGrasp keyframe planner
  - explain why large `joint_target_wait_steps` can create jump / teleport-like motion in exported videos
  - summarize practical no-code-change ways to reduce jumps while keeping accuracy, including pros and cons
- Covered topics:
  - current path structure: `init -> pregrasp -> grasp -> action`
  - current offset semantics: `candidate_target_local_x_offset_m=-0.03` and `approach_offset_m=0.08`
  - current smoothing logic: EE/TCP-pose interpolation followed by per-waypoint IK
  - existing post-hoc smooth tools:
    - `code_painting/replay_pose_debug_smooth.py`
    - `code_painting/smooth_planner_outputs_from_pose_debug.py`
    - `code_painting/batch_smooth_planner_outputs.sh`
- Expanded the analysis with new candidate methods:
  - added a dedicated evaluation of the “sample one EE point every 1 cm + use previous IK solution as the seed + reject the whole segment if adjacent joint deltas exceed a threshold” idea
  - clarified its relationship to the current `cartesian_interp_ik` mode: the current code already has waypoint IK + previous-seed chaining; what is missing is denser sampling and explicit jump-threshold rejection
  - added pros/cons and implementation difficulty for several alternatives:
    - fixed-step dense sampling
    - position+rotation dual-threshold sampling
    - joint jump-threshold filtering
    - soft continuity preferences inside IK
    - post-IK joint smoothing
    - extra semantic intermediate poses
    - switching to global trajectory optimization
- Expanded again with V7 debug analysis:
  - explained why accuracy drops without try / replanning: the current system behaves more like a `plan-execute-correct` closed loop than a single-shot open-loop path that fully lands inside tolerance
  - explained why try improves accuracy but makes the video look more segmented: multiple short corrections inside one stage plus settle tails that are not serialized frame by frame
  - added a proposed new diagnostic quantity:
    - lateral distance from the current point to the target forward axis
    - useful for separating front/back error from lateral miss relative to the intended approach line
- Implemented the code change in this round:
  - `code_painting/plan_anygrasp_keyframes_r1.py`
  - added new breakdown fields:
    - `lateral_to_forward_axis_m`
    - `lateral_to_forward_axis_cm`
  - added new terminal output field:
    - `lat_cm`
  - integrated into:
    - single-arm / dual-arm `plan-request`
    - single-arm / dual-arm `plan-solution`
    - single-arm / dual-arm `attempt`
    - single-arm `attempt-supervision`
    - `attempt_history` / supervision-error structures
- Added another analysis-only design direction that preserves the current main workflow:
  - express the human-hand target as an object-relative pose `T_obj_hand_demo`
  - then generate a more executable robot target through a robot-specific correction `Δ_robot`
  - recommend implementing this later as an independent target-adapter layer rather than modifying the IK solver core first
  - recommended progression:
    - constant rigid correction
    - stage-specific correction
    - object-type/size adaptive correction
- Validation:
  - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py`
- Validation:
  - documentation-only task, so no extra code validation command was run

## 2026-03-27

- Added a raw-planner-v7 -> repaint -> review -> pi0 wrapper script:
  - `run_planner_v7_repaint_review_pi0.sh`
  - Purpose:
    - skip smoothing and directly consume `anygrasp_plan_keyframes_realoffset_batch_pure-v7`
    - call the original `batch_head_cam_repaint_with_auto_pad.sh`
    - call `review_repaint_videos.py` for manual filtering
    - call `process_repainted_planner_outputs.py` to generate pi0 / robotwin processed_data
  - Validation:
    - `bash -n run_planner_v7_repaint_review_pi0.sh`

- Added smooth-bundle scripts:
  - `code_painting/smooth_planner_outputs_from_pose_debug.py`
  - `code_painting/batch_smooth_planner_outputs.sh`
  - `run_reviewed_smooth_repaint_pi0_pipeline.sh`
  - Purpose:
    - remove lingering / near-duplicate frames from the Step-1 planner outputs
    - interpolate key states to smooth jumps
    - re-export source-consistent head / left wrist / right wrist / pose_debug outputs
  - Key output root:
    - `code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3_smooth`
  - Validation:
    - `python -m py_compile code_painting/smooth_planner_outputs_from_pose_debug.py`
    - `bash -n code_painting/batch_smooth_planner_outputs.sh`
    - `bash -n run_reviewed_smooth_repaint_pi0_pipeline.sh`
    - `DRY_RUN=1 bash run_reviewed_smooth_repaint_pi0_pipeline.sh`
    - one-sample output: `/tmp/d_pour_blue_0_smooth_bundle`

- Added script:
  - `policy/pi0/scripts/process_repainted_planner_outputs.py`
  - Purpose: process pi0 data using source-consistent planner outputs:
    - repaint planner head
    - planner `left_wrist_cam_plan.mp4`
    - planner `right_wrist_cam_plan.mp4`
    - planner `pose_debug.jsonl`
  - This avoids mixing hand-retarget wrist / `world_targets_and_status.npz` with planner head videos
  - Minimal validation:
    - `python -m py_compile policy/pi0/scripts/process_repainted_planner_outputs.py`
    - one-sample output test: `/tmp/pi0_planner_repaint_test`

- Added analysis notes:
  - `agent-read/2026-03-27_repaint_duration_mismatch_analysis_ZH.md`
  - `agent-read/2026-03-27_repaint_duration_mismatch_analysis.md`
  - Purpose: record why `process_repainted_headcam_with_wrist.py` outputs are much shorter than `head_cam_plan.mp4`
  - Conclusion:
    - the current script truncates by minimum frame count rather than aligning by real duration in seconds
    - the effective length is actually limited by `world_targets_and_status.npz` and the left/right wrist replay streams
    - if a later viewer/export step plays the result at a higher fps, it will further feel like “only about 1 second”
  - No code changes in this round; only recorded findings and root cause

- Added source-consistency analysis notes:
  - `agent-read/2026-03-27_head_source_vs_wrist_source_analysis_ZH.md`
  - `agent-read/2026-03-27_head_source_vs_wrist_source_analysis.md`
  - Conclusion:
    - `batch_head_cam_repaint_with_auto_pad.sh` uses planner-side `head_cam_plan.mp4` as the head source
    - the current pi0 processing step uses hand-retarget `left/right_wrist_replay.mp4` as the wrist source
    - inside the hand-retarget directory, `zed_replay / wrist / world_targets` are basically aligned in length
    - but planner head and hand-retarget wrist are not the same source stream, so matching frame counts should not be expected

- Added `policy/pi0/scripts/process_repainted_headcam_with_wrist.py`
  - Purpose:
    - convert the newer “SAM/repainted head-cam video + left/right wrist replays + world_targets_and_status.npz” into the pi0 `processed_data` HDF5 intermediate format
  - Main capabilities:
    - separate `--head-root` and `--retarget-root`
    - templated episode directory names via:
      - `--head-dir-template`
      - `--retarget-dir-template`
    - support the new head-video filename:
      - `target_with_original_head_cam_plan.mp4`
    - support `--review-json`, which by default processes only videos manually marked `y` / `usable=true`
    - support `--review-mode include_ambiguous` to also include `m` / `ambiguous`
    - keep output compatible with the existing pi0 `processed_data/<task>-<num>/episode_x/*.hdf5` layout
  - Related docs:
    - `agent-read/2026-03-27_pi0_repaint_wrist_to_hdf5_ZH.md`
    - `agent-read/2026-03-27_pi0_repaint_wrist_to_hdf5.md`
  - Validation:
    - `cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && python -m py_compile scripts/process_repainted_headcam_with_wrist.py scripts/process_data_retageted_human.py scripts/process_data_R1.py`

## 2026-03-25

- Added a visual-only base occluder to hide the chassis in rendered videos:
  - Files:
    - `code_painting/render_hand_retarget_r1_npz.py`
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/render_object_pose_r1_npz.py`
  - Motivation:
    - the robot chassis/base is often visible in head and wrist videos and hurts the composition
    - the user wanted a configurable occluder with custom height/size that does not participate in collision
  - Implementation:
    - added a visual-only `base_occluder` actor with no collision shapes
    - the occluder follows the robot base pose
    - exposed CLI flags:
      - `--base_occluder_enable`
      - `--base_occluder_local_pos X Y Z`
      - `--base_occluder_half_size HX HY HZ`
      - `--base_occluder_color R G B`
  - Semantics:
    - `local_pos` is defined in the robot-base frame, so it rotates with the robot heading
    - `half_size` uses the SAPIEN box half-size convention
    - the occluder is visual-only and does not affect collision, IK obstacles, or grasp contact
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_r1_npz.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py /home/zaijia001/ssd/RoboTwin/code_painting/render_object_pose_r1_npz.py`

- Added a first-pose debug log for the base occluder:
  - File:
    - `code_painting/render_hand_retarget_r1_npz.py`
  - Change:
    - when the occluder pose is updated for the first time, print
      - `world_p`
      - `half_size`
      - `color`
  - Purpose:
    - helps confirm that the occluder is really created and whether it lands at the intended position and size

- Corrected the R1 planner wrist-camera mount definition and removed post-export image rotation:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/CAMERA_DEBUG_NOTES_R1.md`
  - Root cause:
    - the R1 planner path had been using a wrist local pose closer to `galaxea_sim/robots/r1_pro.py`
    - but `galaxea_sim/robots/r1.py` only includes `rx=-10°` and does not include the extra local `z=-90°`
    - that is why wrist exports kept needing `90°/180°` image-plane patches and still looked geometrically unnatural
  - Fix:
    - override the wrist local quaternion specifically inside `plan_anygrasp_keyframes_r1.py` so the R1 planner matches `galaxea_sim/robots/r1.py`
    - `rotate_wrist_rgb_for_export(...)` now becomes a pass-through; planner wrist exports no longer rotate the saved image afterward
    - wrist-writer size returns to the original landscape `(image_width, image_height)`
  - Impact:
    - the wrist view now comes from the actual mounted camera pose rather than an export-time image-plane rotation
    - no change to the global default in `render_hand_retarget_r1_npz.py`, so R1 Pro-related paths are not affected
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Corrected planner wrist-video export orientation again:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - User feedback:
    - after the previous `180°` correction, the exported result still looked like the correct view rotated 90 degrees CCW
  - Fix:
    - changed `rotate_wrist_rgb_for_export(...)` from `cv2.ROTATE_180` to `cv2.ROTATE_90_COUNTERCLOCKWISE`
    - changed planner wrist-writer size back to the rotated-frame size `(image_height, image_width)`
  - Current behavior:
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` are exported with a `90°` CCW image-plane rotation
    - output dimensions match the rotated frames
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Corrected planner wrist-video export orientation and frame size:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - the previous round incorrectly treated the wrist videos as needing a `90°` rotation
    - the exported files became portrait, and the user confirmed the image was still upside down
  - Fix:
    - changed `rotate_wrist_rgb_for_export(...)` from `cv2.ROTATE_90_CLOCKWISE` to `cv2.ROTATE_180`
    - changed planner wrist-writer size back from portrait `(image_height, image_width)` to landscape `(image_width, image_height)`
  - Current behavior:
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` stay in landscape format
    - export now applies only an in-plane `180°` correction
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed the orientation of planner-exported wrist videos:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` appeared 90 degrees CCW relative to the expected viewing orientation
  - Investigation result:
    - after comparing `render_hand_retarget_r1_npz.py` and `galaxea_sim/robots/r1_pro.py`, the R1 and R1 Pro wrist-camera mount poses are effectively the same:
      - base quaternion `[0.5, 0.5, -0.5, 0.5]`
      - extra RPY offset `[-10deg, 0, -90deg]`
    - so this round does not change the mounted camera definition; it corrects the exported image plane instead
  - Fix:
    - added `rotate_wrist_rgb_for_export(...)`
    - apply `cv2.ROTATE_90_CLOCKWISE` before writing left/right wrist videos in `record_frame(...)`
    - rotation now happens before overlay/BGR conversion so debug text stays upright as well
  - Impact:
    - no change to world camera pose, planner targets, candidate coordinate conversion, or head videos
    - this only corrects planner wrist-video export orientation
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed the post-rotation writer-size mismatch for wrist videos:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - the previous round rotated planner wrist videos by `90° clockwise`
    - but the `cv2.VideoWriter` instances were still opened with the original `(image_width, image_height)` = `640x360`
    - the rotated frames are actually `360x640`
    - as a result the writer created tiny placeholder mp4 files (around `258B`) without valid video frames
  - Fix:
    - planner wrist writers now use the rotated frame size `(image_height, image_width)`
  - Impact:
    - `head_cam_plan.mp4` keeps its original landscape size
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` now use portrait dimensions matching the rotated wrist frames
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Adjusted URDF-IK waypoint visualization and documented the stage-settling parameters:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Visualization changes:
    - with `--debug_visualize_ik_waypoints 1`, the viewer now shows start and goal markers in addition to intermediate waypoints
    - both start and goal use red point+forward-axis markers
    - intermediate waypoint markers are now smaller to reduce clutter around the hands and targets
  - Analysis notes:
    - `init` still only applies `apply_robot_init_pose(...)` once and advances a short `step_scene(settle_steps)` window; unlike later stages, it does not call `settle_arms_to_targets(...)` to wait for full convergence
    - `--settle_steps` defaults to `4` and is used to:
      - advance a few physics steps after init
      - add a short settle window after each stage trajectory is sent
    - `--joint_target_wait_steps` defaults to `60` and is used to:
      - keep stepping physics after the trajectory ends until the measured joints get closer to the final target
  - Impact:
    - no planning or execution logic changed in this round; this is a visualization-only update
  - Relevant code:
    - waypoint markers:
      - `update_ik_waypoint_visuals(...)`
      - `_ensure_ik_waypoint_marker_actors(...)`
      - `_ensure_ik_waypoint_endpoint_actors(...)`
    - init / stage settling:
      - `apply_robot_init_pose(...)`
      - `execute_single_arm_plan(...)`
      - `execute_dual_arm_plan(...)`
      - `settle_arms_to_targets(...)`
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed pure-mode EE-pose serialization compatibility in `pose_debug.jsonl`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - the newly added `current_left_ee_pose_world_wxyz` / `current_right_ee_pose_world_wxyz` export path in `record_frame(...)` incorrectly assumed that `robot.get_*_ee_pose()` returns a `sapien.Pose`
    - in this robot implementation the API actually returns a 7D list `[x, y, z, qw, qx, qy, qz]`
    - pure-mode batch runs therefore crashed on the first frame write with:
      - `AttributeError: 'list' object has no attribute 'p'`
  - Fix:
    - added `pose_like_to_world_wxyz(...)`
    - made the serializer accept both `sapien.Pose` objects and 7D pose lists
    - switched the relevant head/ee pose exports in `record_frame(...)` to use the shared helper
  - Impact:
    - no change to planning or execution behavior
    - this only fixes pure-mode `pose_debug.jsonl` serialization
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- Added CLI parameter `--urdfik_cartesian_interp_auto_step_m` to control the translation-density threshold used by automatic waypoint mode when `--urdfik_cartesian_interp_steps=-1`.
- The old `0.05m` threshold was hardcoded; it is now configurable with the same default `0.05`. Fixed waypoint mode is unchanged.
- `render_hand_retarget_r1_npz_urdfik.py` now prints the active `auto_step_m` in `[ik-trajectory]` and `[ik-waypoints]` logs.
- Added execution-side joint-convergence tuning parameters for the AnyGrasp planner:
  - `--joint_command_scene_steps`
  - `--joint_target_wait_steps`
  - `--joint_target_wait_tol_rad`
- Modified files:
  - `code_painting/plan_anygrasp_keyframes_r1.py`
  - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
- Purpose:
  - address the case where `plan-request` / `plan-solution` already indicate that the planned endpoint is moving backward, but `attempt` still remains stuck at a forward-biased pose for many retries.
  - make the execution path advance more physics scene steps per joint waypoint and wait for final joint convergence before measuring reach error.
- Current diagnosis:
  - the earlier interpretation of `plan_vs_current_fwd_cm` was partially wrong.
  - the correct meaning is: in `plan-solution`, `plan_vs_current_fwd_cm > 0` means the current pose is ahead of the planned endpoint, so the planned endpoint is actually behind the current pose.
  - therefore, in the long-tail retry regime, IK is already commanding backward motion, but the execution layer is not converging far enough to that command.
- Extra docs:
  - `agent-read/2026-03-25_ik_execution_regression/README.zh.md`
  - `agent-read/2026-03-25_ik_execution_regression/README.en.md`

- Added "skip grasp re-execution when the target is the same" logic:
  - when `pregrasp` and `grasp` resolve to the same pose (the typical case is `--approach_offset_m 0.0`)
  - the system no longer replans and re-executes `grasp` after already reaching `pregrasp`
  - instead it reuses the `pregrasp` result and records `grasp_skipped_same_target`
- Why:
  - under `approach_offset_m=0.0`, logs showed that `pregrasp` could already converge
  - but the subsequent redundant `grasp` replanning to the exact same target pulled the end effector away again
- Additional note:
  - this round briefly tried a seeded/unseeded FK post-check scorer inside `urdfik.py`
  - that attempt produced clearly wrong pregrasp poses on the first retry
  - it has been reverted and is not kept

- Added a third correction for `cartesian_interp_ik`:
  - tightened the default position threshold in `urdfik.py` from `0.005m` to `0.001m`
  - tightened the default rotation threshold from `0.05rad` to `0.02rad`
  - stopped threshold relaxation from expanding toward `0.1m`
  - `render_hand_retarget_r1_npz_urdfik.py` now automatically reduces overly dense cartesian waypoint counts based on IK thresholds
- Immediate reason:
  - the `action` logs showed total target motion on the order of a few centimeters to about 9 cm, while `--urdfik_cartesian_interp_steps 30` reduced each translation step to roughly `3mm`
  - the old IK success threshold was `5mm`
  - that allowed the solver to accept many waypoint solves with almost no motion, so the final theoretical endpoint stayed far from the target
- Updated conclusion:
  - for this bug, “more retries” is not the main fix
  - increasing interpolation density can actually make it worse
  - the waypoint resolution must first remain larger than the IK success threshold

- Added a waypoint-level `seeded/unseeded` candidate comparison on top of that:
  - location: `code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - behavior: for each waypoint, try both “use current seed” and “no seed”
  - then use FK-based post-checking against that waypoint's `ee` target and keep the closer candidate
- Purpose:
  - address the remaining `action` failure mode where the interpolation-density fix helped, but the left arm was still being trapped in a local solution basin by the current seed

- Adjusted terminal debug output formatting:
  - `plan-request`
  - `plan-solution`
  - `attempt`
- Changes:
  - dual-arm logs are now printed as separate left/right lines
  - `theory` is shortened to:
    - `forward`
    - `backward`
    - `aligned`
  - `fwd_cm` now uses ANSI color highlighting so sign changes are easier to spot

- Further compressed the end-of-sample terminal summary:
  - it no longer prints the full `statuses_by_arm={...}` dictionary
  - it now uses a short format with:
    - `arms`
    - `arm`
    - `obj`
    - `fXX=cYY`
    - `pre/gr/act`
    - `video`

## 2026-03-25 11:51:21 +08

- Fixed a logging-format regression:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Problem:
    - `colorize_forward_cm()` was accidentally indented inside `short_direction_label()` during the previous log refactor
    - dual-arm `plan-request` logging then crashed with `NameError: name 'colorize_forward_cm' is not defined`
  - Fix:
    - restored `colorize_forward_cm()` to module scope
    - corrected its internal branch indentation so positive/negative/near-zero color cases work again
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

## 2026-03-25 12:08:00 +08

- Added a grasp/action object-collision toggle:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - New flag:
    - `--enable_grasp_action_object_collision 0|1`
  - Behavior:
    - default `0` preserves the original no-collision execution mode
    - when set to `1`, execution objects selected by the executed arm keep collision geometry
    - collision stays disabled during `pregrasp`
    - collision is enabled right before `close_gripper` and remains enabled through `action`
    - object attachment logic, TCP-relative transforms, target generation, and other pose transforms are unchanged
  - Implementation note:
    - stage-local collision enable/disable is implemented by caching and restoring SAPIEN collision groups
    - non-selected objects keep their original behavior
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-25 12:22:00 +08

- Adjusted the default visualization behavior of `plan_anygrasp_keyframes_r1.py`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Changes:
    - restored target-pose axes to be visible by default
    - hid the left/right wrist cameras by default in this planner script so wrist-camera frustums no longer appear in saved videos / viewer output
  - Notes:
    - this affects only the default behavior of `plan_anygrasp_keyframes_r1.py`
    - it does not change the shared base renderer behavior for other scripts
    - it does not affect head-camera or third-person capture itself
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

## 2026-03-25 13:05:00 +08

- Added pure/debug visualization controls to `plan_anygrasp_keyframes_r1.py`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - New flags:
    - `--debug_visualize_targets 0|1`
    - `--viewer_show_camera_frustums 0|1`
  - Changes:
    - restored target-axis rendering to an explicit flag instead of a hardcoded always-on path while keeping the default enabled
    - disabled SAPIEN `ControlWindow.show_camera_linesets` by default in the viewer path
    - after the wrist frustums are hidden, this also removes the remaining zed/third camera frustum lines by default
  - Notes:
    - `pure_scene_output` still controls clean main videos: no overlay text, no candidate grippers, no target axes
    - `viewer_show_camera_frustums=0` now controls viewer camera-line visibility
    - `debug_visualize_targets=1` keeps target axes available for debug runs
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed the batch wrapper so the new pure/debug visualization flags are forwarded:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - Changes:
    - added `--debug_visualize_targets` to the batch parser
    - added `--viewer_show_camera_frustums` to the batch parser
    - forwarded both flags in `build_single_command()`
  - Reason:
    - the pure-mode batch command was being rejected by batch-layer argparse before the single-run script could receive the flags
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-25 13:35:00 +08

- Added a minimal "progressive close until contact/stall" path for `--enable_grasp_action_object_collision=1`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Analysis:
    - Previously the flag only re-enabled collision for the selected object during `grasp/action`
    - The gripper still used a one-shot `set_grippers(close)` update and only advanced a very small number of physics steps
    - As a result, the fingers could still visually close through the object even when collision shapes were present
  - Change:
    - Added `close_grippers_progressively_with_collision_stop()`
    - The `close_gripper` stage now closes in small command increments
    - Each increment advances physics, reads gripper joint `qpos`, and checks contacts between the selected object and the executing gripper links
    - Closure stops early once contact is present and gripper joint motion has stalled
    - The original one-shot `renderer.set_grippers(...)` path remains unchanged when the collision flag is disabled
  - Note:
    - This is a minimal fix; it does not change `pregrasp/grasp/action` target construction or the TCP-to-object attachment transform
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- Fixed a SAPIEN API compatibility issue in the progressive-close path:
  - Problem:
    - `PhysxArticulationJoint` in the current environment does not provide `get_qpos()`, which caused a crash when entering `close_gripper`
  - Fix:
    - read the full articulation `qpos` from `entity.get_qpos()`
    - then recover each gripper joint's actual position by accumulating offsets over `active_joints` with `joint.get_dof()`
  - Code location:
    - `code_painting/plan_anygrasp_keyframes_r1.py:_get_gripper_joint_positions`

## 2026-03-25 14:05:00 +08

- Added an explicit warning when `grasp` has not reached the target but execution still proceeds to `close_gripper`:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Behavior:
    - before `close_gripper`, if the `grasp` stage reports `reached=False`, the script now prints `[warn] grasp_not_reached_before_close ...`
    - this is logging only and does not change the execution logic

- Added an automatic waypoint mode for URDF IK Cartesian interpolation:
  - File:
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - New behavior:
    - when `--urdfik_cartesian_interp_steps > 1`, the original fixed-step mode is preserved
    - when `--urdfik_cartesian_interp_steps -1`, an automatic mode is enabled
  - Automatic mode rule:
    - if absolute translation distance is `<= 5cm`, no intermediate TCP waypoint is added (start and goal only)
    - once translation exceeds `5cm`, one additional intermediate waypoint is added per extra `5cm` bucket
    - examples:
      - `10cm` translation -> `1` intermediate waypoint
      - `15cm` translation -> `2` intermediate waypoints
  - Note:
    - this automatic mode only adjusts waypoint count from TCP translation distance and does not change the IK target or execution semantics

- Recorded the actual execution semantics of `cartesian_interp_ik`:
  - New documents:
    - `agent-read/V1.15_urdfik_cartesian_interp_execution_semantics_ZH.md`
    - `agent-read/V1.15_urdfik_cartesian_interp_execution_semantics.md`
  - Conclusion:
    - intermediate `ee/tcp` waypoints are indeed used for per-waypoint IK solving
    - after the latest fix, the execution stage now also replays the corresponding `joint_waypoints`
    - `cartesian_interp_ik` now affects both IK solving and the final executed path more directly

## 2026-03-25 14:25:00 +08

- Fixed the `urdfik` execution layer so it actually consumes `joint_waypoints`:
  - File:
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - Changes:
    - `execute_plans(...)` no longer performs only an endpoint interpolation from `current_joints` to `target_joints`
    - it now directly consumes `plan["position"]` / `plan["velocity"]`
    - in dual-arm execution, left and right trajectories are interleaved by relative trajectory progress, matching the base renderer semantics
    - `_execute_single_ik_plan(...)` also now prefers replaying the full `plan["position"]` trajectory
  - Effect:
    - the `joint_waypoints` generated by `cartesian_interp_ik` now become part of the real executed path
# 2026-03-25

- Enhanced pure-mode outputs:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Changes:
    - `pure_scene_output=1` no longer generates `debug_selection_preview.mp4`
    - the main planning run now also writes:
      - `head_cam_plan.mp4`
      - `left_wrist_cam_plan.mp4`
      - `right_wrist_cam_plan.mp4`
    - pure mode now auto-enables `pose_debug.jsonl` even without explicitly passing `--save_pose_debug 1`
    - `pose_debug.jsonl` now additionally records:
      - left/right wrist-camera poses
      - left/right TCP / EE poses
      - left/right 6D arm qpos
      - left/right gripper finger-joint qpos
      - object actor poses / replay poses
    - `plan_summary.json` now includes the corresponding video and data paths
  - Documentation:
    - `agent-read/2026-03-25_pure_mode_outputs_ZH.md`
    - `agent-read/2026-03-25_pure_mode_outputs.md`
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py code_painting/render_hand_retarget_r1_npz_urdfik.py`
    - `git diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- Added a `--debug_visualize_ik_waypoints` debug parameter for `--planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik`, rendering intermediate `tcp_waypoints_world` into viewer/debug output.
- The visualization is a “point + local forward-axis” marker and only shows intermediate waypoints; the start and final target remain excluded, with the final target still represented by the existing target axis.
- This change affects debug display only and does not change waypoint generation, IK solving, trajectory execution, or collision behavior.
- `plan_anygrasp_keyframes_r1_batch.py` now forwards the parameter as well.
- Validation: `python -m py_compile` and `git diff --check` are run after this round.
# 2026-03-25

- Fixed `base_occluder` initialization/update missing on the `urdfik` / `ReplayRenderer` path:
  - File:
    - `code_painting/replay_r1_h5.py`
  - Cause:
    - `ReplayRenderer._load_robot()` overrides the base robot-loading flow but did not hook in the later-added `base_occluder` logic
    - as a result, under `plan_anygrasp_keyframes_r1.py --planner_backend urdfik`:
      - the `[base-occluder] ...` debug line was never printed
      - the occluder was not updated using the corrected anchor semantics
  - Change:
    - added the missing calls inside `ReplayRenderer._load_robot()`:
      - `self._base_occluder_link = self._find_robot_link(["base_link"])`
      - `self._update_base_occluder_pose()`
    - this keeps replay / urdfik execution consistent with the base renderer behavior
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/replay_r1_h5.py code_painting/render_hand_retarget_r1_npz.py`

- Further refined the `base_occluder` height semantics:
  - File:
    - `code_painting/render_hand_retarget_r1_npz.py`
  - Cause:
    - the `base_link` plane position is closer to the visible chassis, but its `z` origin is not the user-intuitive "height above ground" reference
    - using the full 3D `base_link` pose directly could place the occluder below the floor or at an obviously wrong height
  - Change:
    - the occluder now uses a mixed anchor:
      - `x/y` still follow the `base_link` planar position
      - `z` is interpreted relative to the renderer root/base world height
      - orientation keeps only the base yaw instead of inheriting the full 3D link pose
    - the debug log now also prints:
      - `anchor_p=...`
      - `root_z=...`
    - this makes the `Z` value in `--base_occluder_local_pos X Y Z` behave like a height-above-ground control
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/render_hand_retarget_r1_npz.py`

- Fixed the `base_occluder` panel appearing far away from the robot base:
  - File:
    - `code_painting/render_hand_retarget_r1_npz.py`
  - Cause:
    - the occluder previously followed the renderer's internal root/base pose rather than the visible chassis-aligned `base_link`
    - under the current R1 setup those frames are offset, so the occluder could appear far from the robot in the viewer
  - Change:
    - the `base_occluder` now anchors to `base_link` when available
    - it falls back to the previous root/base pose only if `base_link` cannot be found
    - the debug log now prints `anchor_link=...` so the active anchor can be verified directly
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/render_hand_retarget_r1_npz.py`

- Added a stage-level run analysis document:
  - `agent-read/2026-03-25_overall_run_analysis_ZH.md`
  - `agent-read/2026-03-25_overall_run_analysis.md`
- Recorded the current conclusions:
  - trajectory shape has improved, but planner/IK endpoints are still frequently wrong
  - unreached `grasp` is a major reason why `close_gripper` sees no contact
  - current gripper contact detection monitors finger-joint `child_link`s rather than the `left/right_gripper_link` base links
- No runtime logic was changed in this round; this was documentation-only analysis.

- Added stronger gripper-close collision debug output and a "solid object" test mode:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - New parameters:
    - `--debug_collision_report 1`
    - `--execution_object_collision_mode solid_bbox`
  - Changes:
    - `close_grippers_progressively_with_collision_stop(...)` can now print, in debug mode:
      - target-object collision-shape summary
      - `left/right_gripper_link` collision-shape summary
      - finger-link collision-shape summary
      - per progressive-close iteration:
        - `finger_contact`
        - `base_contact`
        - `finger_pairs`
        - `base_pairs`
    - Regular `[gripper-close]` output now also includes:
      - `base_contact=...`
    - Execution-object collision now supports two modes:
      - `convex`: keep the existing `add_convex_collision_from_file`
      - `solid_bbox`: derive mesh bounds and use one axis-aligned solid box as collision
    - `solid_bbox` changes only execution collision, not the visual mesh
    - the batch wrapper now forwards both new parameters
  - Notes:
    - the early-stop criterion still uses finger-link contact only, so behavior is unchanged by default
    - `base_contact` is currently debug-only extra output to reveal whether the gripper base touched the object first
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `git diff --check -- code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`

- Recorded one `d_pour_blue_0` collision-debug conclusion:
  - Command:
    - ran with `--execution_object_collision_mode solid_bbox --debug_collision_report 1`
  - Key output:
    - `planned_object_cup(shapes=1,types=PhysxCollisionShapeBox)`
    - `planned_object_bottle(shapes=1,types=PhysxCollisionShapeBox)`
    - `left_gripper_link(shapes=0)`
    - `right_gripper_link(shapes=0)`
    - `left_gripper_finger_link1/2(shapes=0)`
    - `right_gripper_finger_link1/2(shapes=0)`
    - the entire closing phase stayed at `contact=0` and `base_contact=0`
  - Conclusion:
    - object-side collision is active
    - in the current runtime instance, the gripper base and finger links both show `shapes=0` through the current shape-inspection path
    - therefore full closure is more consistent with “no detectable gripper-side collision shape” than with “objects have no collision”

- Added gripper contact monitoring scope parameter:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - New parameter:
    - `--gripper_contact_monitor_mode {fingers,fingers_and_base,all_robot_links}`
  - Notes:
    - `fingers`
      - keeps the original finger-only stop monitoring
    - `fingers_and_base`
      - also monitors `left/right_gripper_link`
    - `all_robot_links`
      - monitors the full robot articulation link set during close-gripper contact checks; mainly intended to debug whether the gripper-link collisions are missing in the current runtime path

## 2026-03-25 23:10:00 +08

- Added a minimal gripper/object collision probe and ran the first validation round:
  - Files:
    - `code_painting/minimal_gripper_collision_probe.py`
    - `agent-read/2026-03-25_minimal_gripper_collision_probe_ZH.md`
    - `agent-read/2026-03-25_minimal_gripper_collision_probe.md`
  - Goal:
    - verify, without AnyGrasp/IK/stage logic, whether the R1 gripper actually generates raw physics contacts with a probe object during `close_gripper`
    - separate “the main-pipeline contact debug helper is blind” from “the physics engine has no robot-object collision at all”
  - Minimal experiments:
    - box probe:
      - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/minimal_gripper_collision_probe.py --arm left --object_kind box --probe_local_offset 0.04 0.0 0.0 --max_iters 20 --settle_steps_per_iter 8`
    - mesh + solid_bbox probe:
      - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python code_painting/minimal_gripper_collision_probe.py --arm left --object_kind mesh --mesh_path /home/zaijia001/ssd/data/R1/hand/obj_mesh/blue_cup/blue_cup.obj --mesh_collision_mode solid_bbox --probe_local_offset 0.04 0.0 0.0 --max_iters 20 --settle_steps_per_iter 8`
  - New conclusion:
    - the helper/component summary still reports robot links as `shapes=0`
    - however, in the minimal isolation experiment, `scene.get_contacts()` can already stably observe:
      - `left_gripper_finger_link1<->probe_box`
      - `left_gripper_finger_link2<->probe_box`
      - `left_gripper_finger_link1<->probe_mesh`
      - `left_gripper_finger_link2<->probe_mesh`
      - and for the larger mesh case, `left_gripper_link<->probe_mesh`
    - therefore `shapes=0` can no longer be treated as proof that the robot has no collision geometry or that no physics contact exists
    - the full-pipeline `contact=0` symptom now looks more like a monitoring/matching/timing issue, or a mismatch between the object's real close-stage pose and the visual impression from the video
  - Validation:
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/minimal_gripper_collision_probe.py`
    - `git diff --check -- code_painting/minimal_gripper_collision_probe.py`

- Added raw target-contact reporting to the main `close_gripper` debug path and completed a reinjection check:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `agent-read/2026-03-25_minimal_gripper_collision_probe_ZH.md`
    - `agent-read/2026-03-25_minimal_gripper_collision_probe.md`
  - Change:
    - when `debug_collision_report=1`, `close_grippers_progressively_with_collision_stop(...)` now additionally prints:
      - `raw_target_contacts`
      - `raw_target_contact_total`
    - the `[gripper-close]` summary now also includes:
      - `raw_target_contact=0|1`
  - Reinjection validation command:
    - `bash code_painting/run_plan_anygrasp_keyframes_r1_batch.sh ... --debug_collision_report 1 --gripper_contact_monitor_mode all_robot_links --enable_viewer 0`
  - New conclusion:
    - in the current `d_pour_blue_0` case, the full close stage reports not only monitor/helper contact as zero, but also raw target contacts as zero throughout
    - therefore the current issue can no longer be explained only as “the monitor under-reports”; it now looks more like the full close stage never generates raw physics contact with the target object at all

- Added target-object pose / collision debug output to the main close stage:
  - File:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - Change:
    - when `debug_collision_report=1`, `[collision-debug-init]` now additionally prints:
      - `target_pose=...`
      - `target_collision_debug=...`
    - `[collision-debug-step]` now also prints, for each close iteration:
      - `target_pose=...`
  - Reinjection conclusion in this round:
    - in the current `d_pour_blue_0` close stage, the target pose remains essentially stable while raw target contact stays zero throughout
    - therefore the issue now looks more like a substantial geometric mismatch between the visual mesh and the `solid_bbox` collision primitive, rather than the object pose being repeatedly reset during close

- Added execution-object collision-bbox visualization:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - New flag:
    - `--debug_visualize_object_collision_bbox 0|1`
  - Behavior:
    - when an execution object uses `solid_bbox` collision, create an additional visual-only bbox actor for it
    - the bbox actor uses the same local `center/half_size` as the collision primitive
    - and follows the object actor pose
  - Purpose:
    - compare the visual mesh and the collision bbox directly in the viewer/videos
    - determine whether the current "interpenetration" is only a visual-mesh effect
  - Validation:
    - `python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - main-pipeline reinjection with `... --debug_visualize_object_collision_bbox 1 ...`

- Added a `convex` comparison experiment and updated the interpretation:
  - Validation command:
    - `... --execution_object_collision_mode convex --debug_collision_report 1 --gripper_contact_monitor_mode all_robot_links --enable_viewer 0`
  - Result:
    - for the current `d_pour_blue_0` case, the close stage still reports zero raw target contacts under `convex`
  - Conclusion:
    - the issue is not only about the `solid_bbox` box approximation
    - even after switching to `convex` mesh collision, the full close stage still produces no raw physics contact

- Added close-stage pose export and an experiment that enables object collision from `pregrasp`:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - New flag:
    - `--grasp_action_object_collision_start_stage {close_gripper,grasp,pregrasp}`
  - New behavior:
    - write `close_stage_snapshot_*.json` before close
    - allow selected execution objects to start participating in collision as early as `grasp` or `pregrasp`
  - Key experiment result:
    - in the `pregrasp + convex` experiment, raw target contacts appear stably in the full pipeline for the first time
    - but the monitor/helper still reports `monitor_contact=0`
  - Updated conclusion:
    - the old default of enabling object collision only at `close_gripper` is indeed too late
    - at the same time, the current monitor/contact matching logic still under-reports existing contacts

- Added execution-object scale overrides so `cup` / `bottle` can be shrunk independently for execution visual and collision geometry:
  - Files:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/README_anygrasp_keyframe_planner.md`
  - New flags:
    - `--execution_object_scale_override NAME=S`
    - `--execution_object_scale_override NAME=SX,SY,SZ`
  - Behavior:
    - scale both the execution visual mesh and collision shape together
    - under `solid_bbox`, scale bbox center / half_size consistently as well
  - Typical usage:
    - `--execution_object_scale_override cup=0.9 --execution_object_scale_override bottle=0.9`
  - Note:
    - to keep the original logic where object collision starts only before `close_gripper`, continue using:
      - `--grasp_action_object_collision_start_stage close_gripper`

- Ran a control experiment with smaller objects while preserving the old `close_gripper`-only collision timing:
  - Output dir:
    - `code_painting/anygrasp_single_scaled_close_only_probe/d_pour_blue_0`
  - Parameters:
    - `--grasp_action_object_collision_start_stage close_gripper`
    - `--execution_object_scale_override cup=0.9`
    - `--execution_object_scale_override bottle=0.9`
  - Result:
    - from close init through iter 20, `raw_target_contact_total=0`
    - final `raw_target_contact=0`
  - Conclusion:
    - shrinking the execution objects to 0.9 alone is not enough to make the old `close_gripper`-only collision-enable logic detect contact

- Ran full-collision experiments with scales 0.8 / 0.5:
  - Output dirs:
    - `code_painting/anygrasp_single_all_collision_scale08/d_pour_blue_0`
    - `code_painting/anygrasp_single_all_collision_scale05/d_pour_blue_0`
  - Common parameters:
    - `--grasp_action_object_collision_start_stage pregrasp`
    - `--execution_object_collision_mode convex`
  - Result:
    - `0.8`: both left/right detect raw target contact stably during close
    - `0.5`: right has raw contact from close init, left starts showing raw contact later in close, and both finish with `raw_target_contact=1`
  - Conclusion:
    - the full-collision output itself looks normal and trustworthy; the real abnormal part is still the under-reporting monitor/contact chain
    - once collision is enabled early enough, shrinking to `0.8` / `0.5` still does not eliminate raw contact

- Analyzed the `0.5 + pregrasp` full-collision result and added a `0.5 + close_gripper + fingers` control experiment:
  - New output dir:
    - `code_painting/anygrasp_single_scale05_close_only_fingers/d_pour_blue_0`
  - Result analysis:
    - `0.5 + pregrasp` looks closer to the desired effect where the gripper closes onto the object and is visually blocked before appearing fully closed
    - but the current close-stop logic still depends on `monitor_contact`, so the final reason may still show `target_reached`
  - New control result:
    - under `close_gripper + fingers + scale=0.5`, left is still `raw_target_contact=0` while right is `raw_target_contact=1`
  - Conclusion:
    - after shrinking to `0.5`, if collision is still enabled only at close, the problem is not fully gone; the more reliable setup still starts collision from `pregrasp`

- Added separate visual/collision scale overrides for execution objects:
  - New flags:
    - `--execution_object_visual_scale_override`
    - `--execution_object_collision_scale_override`
  - Kept for compatibility:
    - `--execution_object_scale_override`
  - Semantics:
    - visual mesh and collision shape can now be scaled independently
    - if both unified and dedicated overrides are provided for the same object, the dedicated override takes priority

- Added a mechanism-level analysis of why enabling collision only after the gripper is already inside the collision volume does not immediately block at that pose:
  - Key points:
    - late-enabled collision does not roll the system back to the first-touch boundary
    - close stopping depends on monitor_contact + stall, not raw contact directly
    - gripper control is coupled at arm level, not independent per finger
    - execution objects are kinematic actors, so they do not get naturally pushed into a clean separating contact state like dynamic bodies would
- 2026-05-07
  - Added Piper gripper orientation rule docs:
    - `agent-read/PIPER_GRIPPER_ORIENTATION_RULES.zh.md`
    - `agent-read/PIPER_GRIPPER_ORIENTATION_RULES.en.md`
  - Contents:
    - Explains why recent debug wrappers default to the left hand
    - Summarizes HaMeR/NPZ gripper local-axis definitions
    - Summarizes the order of `stored_orientation_post_rot_xyz_deg` and `orientation_remap_label`
    - Summarizes head-camera to world, world to Piper base, and gripper target to `link6` / URDFIK transforms
    - Records the current observation: blue `+Z` is closest to approach, green `+Y` is closest to opening, red `+X` is side/normal

- 2026-05-07
  - Added `code_painting/run_piper_retarget_postrot_board_video.sh`
  - Purpose:
    - Reuse the full Piper retarget replay path
    - Scan candidate rotations only through `stored_orientation_post_rot_xyz_deg`
    - Compose each candidate's `zed_replay.mp4` and `third_replay.mp4` into `board_zed.mp4` / `board_third.mp4`
  - Use case:
    - When direct local-axis scans produce too many IK failures, this keeps the visualization closer to the original retarget replay path
  - Validation:
    - `bash -n code_painting/run_piper_retarget_postrot_board_video.sh`
    - A 1-frame `standard` smoke test successfully generated `/tmp/piper_retarget_postrot_board_smoke/board/board_zed.mp4` and `board_third.mp4`

- 2026-05-07
  - Extended the Piper local-axis sweep tool:
    - Added multi-frame video mode via `--video_mode 1`
    - Added `board_all_zed.mp4` and `board_success_zed.mp4`
    - Added `--candidate_mode semantic`, which emits `forward_from_xp/xm/yp/ym/zp/zm` and `open_from_xp/xm/yp/ym/zp/zm` candidates
  - Goal:
    - Let the id0 gripper position move across frames while each candidate uses a fixed orientation mapping
    - Reduce visual noise from the many failing 24-remap candidates
  - Validation:
    - `python3 -m py_compile code_painting/build_piper_local_axis_sweep_board.py`
    - `bash -n code_painting/run_piper_local_axis_sweep_board.sh`
    - A 2-frame smoke test successfully generated `/tmp/piper_axis_video_smoke/board_all_zed.mp4` and `/tmp/piper_axis_video_smoke/board_success_zed.mp4`

- 2026-05-07
  - Added `code_painting/build_piper_local_axis_sweep_board.py` and `code_painting/run_piper_local_axis_sweep_board.sh`
  - Purpose:
    - Keep the current `PiperPika` scene and head-camera calibration fixed
    - Enumerate all valid right-handed local-axis remaps for one arm on one frame
    - Export `board_zed.png`, `board_third.png`, `summary.json`, and `summary.csv`
  - Why:
    - The existing `orientation_sweep` is oriented toward world-frame target orientation scans and does not directly answer what the HaMeR/recomputed local gripper `x/y/z` axes mean
    - The new script annotates each RGB local axis with its robot-relative `forward/left/up` meaning first, so local-axis semantics can be fixed before execution-error debugging
  - Validation:
    - `python3 -m py_compile code_painting/build_piper_local_axis_sweep_board.py`
    - `bash -n code_painting/run_piper_local_axis_sweep_board.sh`

- 2026-05-11
  - Added `code_painting/run_piper_hamer_axes_replay_batch.sh`
  - Purpose:
    - Batch replay `hand_detections_*.npz`
    - Use the finalized HaMeR/NPZ gripper-axis rule: `orientation_remap_label=identity` and `stored_orientation_post_rot_xyz_deg=0 0 0`
    - Replay both hands by default via `ARMS=both`, with `ID_FILTER` support for single IDs, multiple IDs, and ranges
    - With the default `KEEP_ONLY_ZED_THIRD=1`, clean depth/wrist PNGs under `frames/` and keep only zed/third RGB frames
  - Documentation:
    - The D section in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` now keeps only final replay commands
    - Gripper-orientation debug/sweep/historical-remap commands were moved to `/home/zaijia001/ssd/PIPER_GRIPPER_ORIENTATION_DEBUG.zh.md`
  - Validation:
    - `bash -n code_painting/run_piper_hamer_axes_replay_batch.sh`

- 2026-05-11
  - Extended `code_painting/render_hand_retarget_r1_npz.py`
  - New capability:
    - Overlay FoundationPose object tracks in the Piper/HaMeR hand replay scene
    - New `--object_replay_input_dir` flag points to a video-level FoundationPose output directory
    - New `--object_missing_frame_policy hide|hold_last` flag
    - New `--objects` and `--object NAME=/path/to/mesh.obj` flags for object selection and mesh overrides
  - Added batch script:
    - `code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`
    - Automatically matches each `hand_detections_<id>.npz` with the same-ID FoundationPose object directory
  - Documentation:
    - Added section E, "HaMeR hands + FoundationPose objects in one replay", to `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
  - Validation:
    - `python3 -m py_compile code_painting/render_hand_retarget_r1_npz.py`
    - `bash -n code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`
    - A 1-frame smoke test successfully generated `/tmp/piper_hamer_axes_with_objects_smoke/id_0/zed_replay.mp4` and `third_replay.mp4`

- 2026-05-11
  - Added `code_painting/plot_piper_gripper_wrist_object_axis_distances.py`
  - Purpose:
    - Use the same Piper head-camera calibration to transform HaMeR gripper/wrist-retreat points and FoundationPose object poses into world coordinates
    - Plot world-axis distance curves for left hand vs `pear` and right hand vs `star_fruit`
    - Curves include gripper `dx/dy/dz` and wrist-retreat `dx/dy/dz`
  - Documentation:
    - Added D-debug-9 to `/home/zaijia001/ssd/PIPER_GRIPPER_ORIENTATION_DEBUG.zh.md`
  - Validation:
    - `python3 -m py_compile code_painting/plot_piper_gripper_wrist_object_axis_distances.py`
    - Successfully generated `output_piper_replay_hamer_axes_with_objects_all/id_0/gripper_wrist_object_axis_distance_id0.png` for id0

- 2026-05-18
  - Added the 0515/new_table Piper calibration config:
    - `robot_config_PiperPika_agx_dual_table_0515.json`
  - Calibration sources:
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/head_d435_new_table_0515_head_from_wrist.json`
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/left_base_T_right_base_new_table.json`
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/right_wrist_new_table_eye_in_hand.json`
  - Updated default head/base parameters in:
    - `code_painting/run_piper_hamer_axes_replay_batch.sh`
    - `code_painting/run_piper_hamer_axes_with_objects_replay_batch.sh`
    - `code_painting/plot_piper_gripper_wrist_object_axis_distances.py`
  - Updated user commands:
    - C/D/E Piper replay commands in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`
    - Added C0 to record the files and values that must be changed after the next calibration update
  - Validation:
    - `bash -n` passed for both Piper replay wrappers
    - `python3 -m json.tool robot_config_PiperPika_agx_dual_table_0515.json` passed
    - `python -m py_compile code_painting/plot_piper_gripper_wrist_object_axis_distances.py` passed
    - A 1-frame id0 smoke test started and produced videos; the log confirmed right base `[0.5562, -0.2718, 0.7698]`, but the old id0 target is IK-failing for both arms under the new calibration, so target offset/reachability should be checked next

- 2026-05-18
  - Corrected the 0515/new_table Piper calibration note by adding the previously omitted left wrist camera extrinsic:
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/left_wrist_new_table_eye_in_hand.json`
    - `/home/zaijia001/ssd/data/piper/calibration/handeye/right_wrist_new_table_eye_in_hand.json`
  - Updated `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`:
    - C0 now explicitly lists the four current new_table JSON files: head 0515, left/right wrist, and left_base_T_right_base
    - Documented that the older `head_d435_new_table_head_from_wrist.json` in the same directory is not used for the current replay commands
    - Added the current base/head-camera placement estimate for checking the physical table setup
    - Fixed the section F distance-plot command that had been broken across lines and was not directly copyable
  - Note:
    - The current D/E replay path still consumes only `robot_config + head_camera`; the left/right wrist extrinsics are recorded for future true wrist-camera rendering where separate left/right local poses will be wired in
  - Validation:
    - Read the four new_table JSON files and computed the derived placement relationship successfully

- 2026-05-18
  - Updated section D in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`:
    - Added `D0. place_bread_basket: HaMeR detection results and human-hand gripper visualization`
    - Recorded the HaMeR GPU detection command for `place_bread_basket/harmer_input -> harmer_output`
    - Added commands to inspect `hand_vis_gripper_*.mp4` for direct HaMeR human-hand gripper-point/axis visualization
    - Added single-ID and batch commands that feed `place_bread_basket/harmer_output/hand_detections_*.npz` into the Piper HaMeR axes replay path
  - Validation:
    - Confirmed that `harmer_output` already contains `hand_detections_0..10.npz` and `hand_vis_gripper_0..10.mp4`

- 2026-05-18
  - Added the Piper calibration bundle workflow:
    - `code_painting/build_piper_calibration_bundle.py`
    - `code_painting/visualize_piper_calibration_bundle.py`
    - `/home/zaijia001/ssd/RoboTwin/calibration_bundle_piper_new_table_0515.json`
  - Updated replay entrypoints:
    - `render_hand_retarget_r1_npz.py` now accepts `--piper_calibration_bundle`
    - `run_piper_hamer_axes_replay_batch.sh` now accepts `CALIBRATION_BUNDLE`
    - `run_piper_hamer_axes_with_objects_replay_batch.sh` now accepts `CALIBRATION_BUNDLE`
  - Purpose:
    - Compose the four hand-eye JSON files for head/base/left_wrist/right_wrist into one self-contained calibration JSON
    - During replay, the bundle writes `calibration_bundle_robot_config.json` under the output directory and overrides the head-camera local pos/quat
    - Provides `axes_compare_old_head.png` to visualize base/head-camera axes and compare the older head extrinsic
  - Observation:
    - The previous head parameters differ from the 0515 head by about `0.123 m` translation and `120.57 deg` rotation
    - The older `head_d435_new_table_head_from_wrist.json` in the same directory is closer to the 0515 head: about `0.039 m` translation and `4.00 deg` rotation
  - Validation:
    - Bundle generation passed
    - Axis PNG generation passed
    - `bash -n` passed for both batch wrappers
    - `py_compile` passed for the three Python files
    - A 1-frame `place_bread_basket` id0 smoke test with `CALIBRATION_BUNDLE=...` completed and loaded the bundle correctly; IK still failed on that frame due to target reachability, not bundle loading

- 2026-05-18
  - Added in-scene head-camera visualization:
    - `render_hand_retarget_r1_npz.py` now accepts `--debug_visualize_cameras`
    - Third-person renders can show the head camera as a white body with red/green/blue local xyz axes and a yellow `-Z` optical ray
    - `run_piper_hamer_axes_replay_batch.sh` and `run_piper_hamer_axes_with_objects_replay_batch.sh` now accept `DEBUG_VISUALIZE_CAMERAS/DEBUG_CAMERA_AXIS_LENGTH/DEBUG_CAMERA_AXIS_THICKNESS`
  - Updated `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`:
    - Documented that the SAPIEN viewer requires `DISPLAY/WAYLAND_DISPLAY`; in the current SSH-only session with `DISPLAY=None`, no viewer window will appear
    - Added three-version head-camera marker comparison commands for `place_bread_basket`:
      - earliest manual head parameters + old robot config
      - pre-0515 new_table head bundle
      - 0515 new_table head bundle
  - Validation:
    - A 1-frame `place_bread_basket` id0 smoke test with `DEBUG_VISUALIZE_CAMERAS=1` completed
    - Successfully generated `/tmp/place_bread_basket_camera_marker_smoke/id_0/frames/third_0000.png`
    - `bash -n` passed for both batch wrappers
    - `py_compile` passed for replay and bundle visualization scripts

- 2026-05-18
  - Improved SAPIEN viewer diagnostics:
    - `render_hand_retarget_r1_npz.py` now prints `DISPLAY/WAYLAND_DISPLAY/XDG_SESSION_TYPE` before creating the viewer
    - Successful viewer creation prints `[viewer] interactive viewer created`
    - Viewer creation failures now catch broader exceptions and print the exception type
    - Added `code_painting/probe_sapien_viewer.py` to independently verify a minimal SAPIEN viewer from a VNC terminal
  - Updated `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md`:
    - Viewer command now prints the display environment first
    - Viewer command includes `--debug_visualize_cameras 1 --debug_camera_axis_length 0.22`
    - Added a minimal viewer probe command
  - Note:
    - The previous `output_place_bread_basket_piper_viewer_probe` command did not enable `debug_visualize_cameras`, so its third-person image would not show the head-camera marker
  - Validation:
    - `DEBUG_VISUALIZE_CAMERAS=1 DEBUG_CAMERA_AXIS_LENGTH=0.5` generated `/tmp/place_bread_basket_camera_marker_big_smoke/id_0/frames/third_0000.png` with visible head-camera axis markers

- 2026-05-18
  - Identified why the viewer window did not appear:
    - In the VNC terminal, `probe_sapien_viewer.py` creates a viewer successfully when `CUDA_VISIBLE_DEVICES` is not set
    - The hand replay command failed with `Renderer does not support display` when `CUDA_VISIBLE_DEVICES=2` was set
    - Conclusion: the viewer needs access to the GPU driving the VNC/X display, and `CUDA_VISIBLE_DEVICES=2` hides that display GPU from the process
  - Updates:
    - `render_hand_retarget_r1_npz.py` viewer diagnostics now include `CUDA_VISIBLE_DEVICES`
    - The viewer command in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` no longer sets `CUDA_VISIBLE_DEVICES=2`
    - Added a negative-control probe command with `CUDA_VISIBLE_DEVICES=2`
  - Validation:
    - `bash -n code_painting/run_piper_hamer_axes_replay_batch.sh`
    - `python -m py_compile code_painting/render_hand_retarget_r1_npz.py code_painting/probe_sapien_viewer.py`

- 2026-05-18
  - Identified and fixed the root cause of the 0515 head-camera orientation offset:
    - `left_base_T_head_camera` in the hand-eye JSON is a raw/optical camera frame
    - Replay commands using `--camera_cv_axis_mode legacy_r1` require a render/SAPIEN camera pose
    - The correct relation is `T_render = T_raw_optical @ legacy_r1.T`
  - Key validation:
    - The earliest manual head position differs from `head_d435_try2_head_from_wrist.json` raw translation by only `3.3e-7 m`
    - The earliest manual head quaternion differs from the try2 raw rotation by `120.0 deg`
    - The earliest manual head quaternion differs from `try2_raw @ legacy_r1.T` by `0.0 deg`
    - Therefore the observed `120 deg` difference is mainly a camera-axis convention difference, not physical calibration drift
  - Updates:
    - `build_piper_calibration_bundle.py` now stores the raw optical head transform and converts `head_camera.left_base_T_head_camera` to the render/SAPIEN pose used by replay
    - Regenerated `calibration_bundle_piper_new_table_0515.json` and `calibration_bundle_piper_new_table_pre0515.json`
    - Updated the D/E wrappers and `plot_piper_gripper_wrist_object_axis_distances.py` default head quaternion to the 0515 render/SAPIEN quaternion
    - Updated the direct C/D/E commands in `/home/zaijia001/ssd/COMMAND_LIBRARY.zh.md` that spell out the head quaternion
  - Calibration quality observations:
    - 0515 head residual mean/max: `0.607/2.193 deg`, `0.0048/0.0167 m`
    - pre-0515 head residual mean/max: `0.574/1.298 deg`, `0.0059/0.0151 m`
    - left wrist residual mean/max: `0.563/0.985 deg`, `0.0129/0.0193 m`
    - right wrist residual mean/max: `1.795/3.567 deg`, `0.0103/0.0180 m`
  - Validation:
    - `py_compile` passed for `build_piper_calibration_bundle.py`, `render_hand_retarget_r1_npz.py`, and `plot_piper_gripper_wrist_object_axis_distances.py`
    - `bash -n` passed for both Piper HaMeR batch wrappers
    - `json.tool` passed for both new_table calibration bundles
    - `CALIBRATION_BUNDLE=...new_table_0515.json DEBUG_VISUALIZE_CAMERAS=1 MAX_FRAMES=1 ID_FILTER=0` smoke test passed; the bundle loaded correctly and generated `/tmp/place_bread_basket_camera_axis_fixed_smoke/id_0/third_replay.mp4`

- 2026-05-19
  - Added a replay target offset along the gripper-local blue `+Z` approach axis:
    - `render_hand_retarget_r1_npz.py` now accepts `--target_local_forward_retreat_m`
    - Positive values mean `target_position -= distance * local(+Z)`, i.e. retreat opposite the visualized blue approach axis
    - The local retreat is applied after camera-to-world conversion and before the ordinary `target_world_offset_xyz`, so it follows each frame's gripper orientation instead of a fixed world XYZ direction
  - Updated wrappers:
    - `run_piper_hamer_axes_replay_batch.sh` now accepts `TARGET_LOCAL_FORWARD_RETREAT_M`
    - `run_piper_hamer_axes_with_objects_replay_batch.sh` now accepts `TARGET_LOCAL_FORWARD_RETREAT_M`
  - Compatibility fixes:
    - Added the new renderer constructor argument in `build_piper_local_axis_sweep_board.py` and `plot_piper_gripper_wrist_object_axis_distances.py`
  - Validation:
    - `py_compile` passed for `render_hand_retarget_r1_npz.py`, `build_piper_local_axis_sweep_board.py`, and `plot_piper_gripper_wrist_object_axis_distances.py`
    - `bash -n` passed for both Piper HaMeR batch wrappers
    - `TARGET_LOCAL_FORWARD_RETREAT_M=0.05 MAX_FRAMES=1 ID_FILTER=0` smoke test passed and printed `[target-local-retreat] along_local_plus_z_blue_m=0.0500`

- 2026-05-20
  - Fixed a renderer-constructor compatibility issue in FoundationPose multi-object replay:
    - After `HandRetargetR1Renderer.__init__` gained camera-debug and local-blue-axis retreat arguments, `render_object_pose_r1_npz.py` still used the old constructor argument list, causing `run_multi_object_pose_r1_npz_batch.sh` to fail with `missing 4 required positional arguments`
    - Added default values in the `ReplayRenderer(...)` constructor calls in `render_object_pose_r1_npz.py`, `replay_r1_h5.py`, and `minimal_gripper_collision_probe.py`
  - Updated the command library:
    - Added a note around line 182 in C1 of `/home/zaijia001/ssd/RoboTwin/COMMAND_LIBRARY.zh.md` to mark the pick_diverse_bottles command as a Piper 0515 head/base calibrated FoundationPose two-object replay command
  - Validation:
    - `py_compile` passed for `render_object_pose_r1_npz.py`, `replay_r1_h5.py`, `minimal_gripper_collision_probe.py`, and `render_multi_object_pose_r1_npz.py`
    - `bash -n` passed for `run_multi_object_pose_r1_npz_batch.sh`
    - A pick_diverse_bottles id0 smoke test with `--max_frames 1 --skip_existing 0` passed and generated `/tmp/pick_diverse_bottles_foundation_replay_smoke/foundation_input_0/head_cam_replay.mp4` and `multi_object_world_poses.npz`

- 2026-05-20
  - Updated the E2 single-video replay commands in `COMMAND_LIBRARY.zh.md`:
    - Added hand + FoundationPose object replay commands for pick_diverse_bottles, place_bread_basket, and stack_cups
    - All three commands use `--piper_calibration_bundle calibration_bundle_piper_new_table_0515.json`
    - All three commands include `--target_local_forward_retreat_m 0.05` to retreat 5 cm opposite the gripper-local blue `+Z` approach axis
    - E2 now documents the viewer toggle: append `--enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02`
  - Validation:
    - Ran a pick_diverse_bottles id0 smoke test with `--max_frames 1` successfully
    - The log confirmed `[target-local-retreat] along_local_plus_z_blue_m=0.0500`
    - The log confirmed FoundationPose objects `['left_bottle', 'right_bottle']` were loaded

- 2026-05-20
  - Changed the E2.1/E2.2/E2.3 H2O replay commands in `COMMAND_LIBRARY.zh.md` from single id0 runs to batch id0-id10 runs:
    - Use `for ID in $(seq 0 10)`
    - Inputs now use `hand_detections_${ID}.npz`
    - FoundationPose object directories now use `foundation_input_${ID}`
    - Output directories now use `id${ID}_z005`
  - Updated the viewer note:
    - To enable the viewer, first narrow `seq 0 10` to a single ID such as `seq 0 0`
    - Then append `--enable_viewer 1 --viewer_wait_at_end 1 --viewer_frame_delay 0.02`
  - Validation:
    - Ran `bash -n` syntax checks for the three batch loop commands; did not run all 33 replays

- 2026-05-20
  - Added section G at the end of `COMMAND_LIBRARY.zh.md`:
    - Added H2O id0-id10 commands for plotting world-axis distances from gripper/wrist-retreat points to FoundationPose object centers
    - Covered pick_diverse_bottles, place_bread_basket, and stack_cups
    - Each ID writes a PNG plus matching CSV under `code_painting/human_object_replay/h2o/.../id${ID}_z005/`
  - Added interpretation notes:
    - Stable same-direction `dz` offsets across tasks/IDs point more toward the head/depth/camera-to-world or replay calibration chain
    - Object-specific or frame-local jumps point more toward FoundationPose pose/depth/mesh estimation
  - Validation:
    - `bash -n` passed for all three id0-id10 loop commands
    - A pick_diverse_bottles id0 `--max_frames 2` smoke test passed and generated `/tmp/pick_diverse_bottles_axis_distance_id0_smoke.png` and `.csv`

- 2026-05-21
  - Updated `plot_piper_gripper_wrist_object_axis_distances.py`:
    - Added `--plot_clip_abs_m`, defaulting to `0.5`
    - PNG plotting now clips values beyond `±plot_clip_abs_m` to the plot boundary so the sub-0.5m trend remains readable
    - CSV output still keeps raw unclipped values, and plot titles report clipping and clipped-value counts
    - Missing FoundationPose object `poses.npz` files no longer abort the whole plot; the script prints a warning and writes NaNs for that side
  - Data observations:
    - There are now 33 H2O task/id CSVs; place_bread_basket id5/id6 are missing `bread/poses.npz`, but the script can still generate the left-side basket curves
    - Treating `|value|>0.5m` as large outliers, the normal-frame overall dz medians are about gripper `+0.150m` and wrist `+0.169m`
  - Validation:
    - `py_compile` passed
    - A pick_diverse_bottles id0 `--max_frames 2` clipped smoke test passed
    - place_bread_basket id5/id6 generated PNG/CSV files despite missing bread tracks

- 2026-05-21
  - Added a raw HaMeR hand point vs FoundationPose object point comparison tool:
    - `code_painting/make_hamer_foundation_point_compare_video.py`
    - Inputs HaMeR `hand_detections_<id>.npz`, `hand_vis_gripper_<id>.mp4`, and FoundationPose object directories
    - Outputs a horizontal comparison video with a HaMeR hand-point panel plus each object's `mesh_overlay.mp4` panel
    - Overlays thumb tip, index tip, thumb/index midpoint, and projected object center
    - Writes a paired CSV with camera-frame `hand_midpoint - object_center` `dx/dy/dz`
  - Updated the H section in `COMMAND_LIBRARY.zh.md`:
    - Recorded the H2O three-task id0-id10 replay CSV statistics from section G
    - Added id0-id10 raw point comparison commands for pick_diverse_bottles, place_bread_basket, and stack_cups
  - Data notes:
    - In inlier frames, overall `abs dz median` is about `15.1cm` for gripper and `17.0cm` for wrist-retreat
    - Signed dz median is about `+15.0cm` for gripper and `+16.9cm` for wrist-retreat
    - pick/place contain a small number of meter-scale z outliers; stack_cups has no `>0.5m` outliers
  - Validation:
    - `py_compile` passed
    - place_bread_basket id0 `--max_frames 5` smoke test succeeded and produced `/tmp/hamer_foundation_point_compare_place_bread_basket_id0.mp4` plus `.csv`

- 2026-05-21
  - Extended `make_hamer_foundation_point_compare_video.py`:
    - Added default distance-curve PNG output named `*_distance.png`
    - The plot shows camera-frame `dx/dy/dz` from the HaMeR thumb/index midpoint to the FoundationPose object center
    - Added `--output_plot` to override the plot path
    - Added `--plot_clip_abs_m`, defaulting to `0.5`, matching section G behavior by clipping only the PNG display while preserving raw CSV values
  - Updated `COMMAND_LIBRARY.zh.md` H2/H6:
    - H2 now documents the video, CSV, and distance-curve PNG outputs
    - H6 now includes a finder command for `*_hamer_foundation_points_distance.png`
  - Validation:
    - `py_compile` passed
    - place_bread_basket id0 `--max_frames 5` smoke test succeeded and produced a video, CSV, and distance-curve PNG

- 2026-05-21
  - Analyzed the newly generated raw HaMeR/FoundationPose CSV files from section H:
    - Found 11 CSVs for `pick_diverse_bottles` id0-id10
    - The inlier overall `abs dz median` in camera-frame `hand_midpoint - object_center` is about `5.1cm`
    - In comparison, the G/H1 world replay statistics for pick have gripper/wrist `abs dz median` about `14.6cm/16.5cm`
    - Conclusion: the 15cm-level z offset is not mainly present in the raw detection points; it is more likely introduced by camera-to-world conversion, `target_world_offset_xyz`, retreat-point definition, and replay coordinate-chain effects
  - Updated `COMMAND_LIBRARY.zh.md`:
    - Added E2.0 before E2.1/E2.2/E2.3 for pure hand replay on the three H2O tasks without loading FoundationPose objects
    - Added the current H raw CSV statistics and comparison against G/H1 world replay statistics after H1
  - Validation:
    - `bash -n` passed for the three E2.0 id0-id10 pure hand replay loop commands

- 2026-05-21
  - Updated `COMMAND_LIBRARY.zh.md` E2.0:
    - Changed the three pure hand replay commands from `--save_png_frames 1` to `--save_png_frames 0`
    - Documented that `--save_png_frames 0` avoids saving per-frame PNG files under `frames/` and keeps only the main replay mp4/npz outputs
    - Added a VS Code-compatible transcode command using `ffmpeg -c:v libx264 -pix_fmt yuv420p -movflags +faststart`
  - Note: raw replay mp4 files may use a codec or pixel format unsupported by VS Code/Chromium; the comparison videos are viewable because the ffmpeg hstack command re-encodes them as H.264/yuv420p
  - Validation:
    - `bash -n` passed for the three E2.0 loop commands and the single ffmpeg transcode command

- 2026-05-21
  - Expanded the Piper H2O debug/generation workflow in `COMMAND_LIBRARY.zh.md`:
    - Added E0 pure replay to create clean zed/third RGB robot videos for later repainting.
    - Added I/J/K to connect SAM hand removal, pure replay repainting, AnyGrasp candidate filtering, AnyGrasp keyframe replay, and repainting.
  - This round only changed command documentation and agent-read logs; no long rendering or repainting jobs were run.
  - Validation:
    - Extracted the new command blocks and `bash -n` passed.
    - Confirmed the repaint and AnyGrasp entry scripts exist.

- 2026-05-21
  - Fixed the section-I SAM repaint command docs so Stage-1 hand removal no longer skips everything just because the full E0 pure robot replay set is incomplete.
  - I1 now uses a dummy robot video; I2 still uses E0 pure robot videos and reports more specific missing paths.
  - Validation: section-I bash blocks passed `bash -n`.

- 2026-05-22
  - Fixed the Stage-1 background input paths for I2/K2 in `COMMAND_LIBRARY.zh.md`.
  - The background file is actually under `stage1_human_inpaint/removed_w_mask_*.mp4`, so the commands now fall back to that path when the top-level `human_hand_bg.mp4` alias is absent.
  - Updated the I1 output check command accordingly.
  - Validation: the I/K2 repaint command blocks passed `bash -n`; sampled id0/id1/id10 across all three tasks and each resolved to an existing Stage-1 background file.
