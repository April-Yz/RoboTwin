# 2026-03-27: Convert wrist videos + SAM-repainted head-cam videos into pi0 HDF5 data

## 1. Goal

This note documents the current pipeline:

1. RoboTwin plans and exports `head_cam_plan.mp4`
2. `inpainting_sam2_robot` composites the robot onto the background video and produces a SAM/repainted head-cam video
3. `policy/pi0/scripts/process_repainted_headcam_with_wrist.py` converts:
   - the head-cam video
   - the left/right wrist videos
   - `world_targets_and_status.npz`
   into the pi0 training format `processed_data/<task>-<num>/episode_x/*.hdf5`

When moving to another server, you only need to preserve the input directory layout described here and rerun the command.

---

## 2. Key commands and code inspected in this round

### 2.1 AnyGrasp planning batch runner

Shell entry:
- `code_painting/run_plan_anygrasp_keyframes_r1_batch.sh`

It only does three things:
- enters the RoboTwin root
- activates the `RoboTwin_bw` conda environment
- calls:
  - `code_painting/plan_anygrasp_keyframes_r1_batch.py`

Your command is effectively running:
- AnyGrasp grasp candidates + hand keyframes + object replay
- and exporting per-episode outputs such as:
  - `head_cam_plan.mp4`
  - `plan_summary.json`
  - debug videos

### 2.2 Auto-pad the stage1 background and run head-cam repaint

Shell entry:
- `inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad.sh`

Its logic is:
1. read fps and duration from `head_cam_plan.mp4`
2. use `ffmpeg` to pad the stage1 background video to roughly the same duration
3. call:
   - `remove_anything_video_sam2_robot.py`
4. produce a composited head-cam video such as:
   - `target_with_original_head_cam_plan.mp4`

### 2.3 The old retargeted-human processing script

Old script:
- `policy/pi0/scripts/process_data_retageted_human.py`

Its core method is:
1. read `world_targets_and_status.npz`
2. use:
   - `left_world_targets`
   - `right_world_targets`
   - `left_gripper_value`
   - `right_gripper_value`
   - `left_plan_status`
   - `right_plan_status`
3. convert quaternions `(w,x,y,z)` into Euler angles `(x,y,z)`
4. build a 14D state:
   - left arm `xyz + euler + gripper` = 7D
   - right arm `xyz + euler + gripper` = 7D
5. forward-fill non-`Success` frames
6. read videos:
   - head cam
   - left wrist
   - right wrist
7. align lengths and write:
   - `processed_data/<task>-<num>/episode_x/instructions.json`
   - `processed_data/<task>-<num>/episode_x/episode_x.hdf5`

However, the old script is more suitable for the older directory layout and is not robust enough when a single repaint root mixes folders like `id_0` and `id_0_head_cam_*`.

---

## 3. New script added

New script:
- `policy/pi0/scripts/process_repainted_headcam_with_wrist.py`

It is specifically designed for the newer pipeline with wrist videos plus SAM-repainted head-cam videos.

Compared with the old script, it adds these capabilities:

1. explicitly separates:
   - the head-cam root
   - the retarget root
2. supports templated episode directory names such as:
   - `id_{id}`
   - `id_{id}_head_cam_arm_gripper_cup_bottle_pad_target`
   - `hand_detections_{id}`
3. supports explicit new head-video names such as:
   - `target_with_original_head_cam_plan.mp4`
4. can auto-discover valid episode ids from the directory template
5. keeps the output format fully compatible with the existing pi0 `processed_data`

---

## 4. Input format for the new script

### 4.1 Head-cam repaint root

Example:

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue
```

Episode directories may look like:

```text
id_0_head_cam_arm_gripper_cup_bottle_pad_target/
  target_with_original_head_cam_plan.mp4

id_1_head_cam_arm_gripper_cup_bottle_pad_target/
  target_with_original_head_cam_plan.mp4
```

Then use:
- `--head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target'`
- `--head-video-name target_with_original_head_cam_plan.mp4`

If you instead want the older zed repaint output, the directories may look like:

```text
id_0/
  final_repainted.mp4
```

Then use:
- `--head-dir-template 'id_{id}'`
- `--head-video-name final_repainted.mp4`

### 4.2 Retarget root

Example:

```text
/home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue
```

Each episode directory is typically:

```text
hand_detections_0/
  world_targets_and_status.npz
  left_wrist_replay.mp4
  right_wrist_replay.mp4
  zed_replay.mp4
```

Corresponding argument:
- `--retarget-dir-template 'hand_detections_{id}'`

### 4.3 Important fields in `world_targets_and_status.npz`

The script currently reads:
- `left_world_targets` `(T, 7)`
- `right_world_targets` `(T, 7)`
- `left_gripper_value` `(T,)`
- `right_gripper_value` `(T,)`
- `left_plan_status` `(T,)`
- `right_plan_status` `(T,)`

Semantics:
- `world_targets`: `[x, y, z, qw, qx, qy, qz]`
- `gripper_value`: gripper command/value
- `plan_status == Success` means the frame is directly usable

---

## 5. Output format

The default output directory is:

```text
policy/pi0/processed_data/<task_name>-<expert_data_num>/
```

Each episode contains:

```text
episode_0/
  instructions.json
  episode_0.hdf5
```

### 5.1 `instructions.json`

Example:

```json
{
  "instructions": ["pour water"],
  "source_episode_id": 0
}
```

### 5.2 HDF5 layout

```text
/action                                (N-1, 14) float32
/observations/state                    (N-1, 14) float32
/observations/left_arm_dim             (N-1,) int32
/observations/right_arm_dim            (N-1,) int32
/observations/images/cam_high          (N-1,) JPEG bytes
/observations/images/cam_left_wrist    (N-1,) JPEG bytes
/observations/images/cam_right_wrist   (N-1,) JPEG bytes
```

The 14D state/action layout is:
- left arm: `x y z roll pitch yaw gripper`
- right arm: `x y z roll pitch yaw gripper`

Images are resized by default to:
- `640 x 480`

---

## 6. New processing commands

## 6.0 Review videos manually first (recommended)

First, inspect the repaint videos one by one in a GUI:

```bash
cd /home/zaijia001/ssd/inpainting_sam2_robot
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate inpainting-sam2-r1

python script/review_repaint_videos.py \
  /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue \
  --dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' \
  --video-name target_with_original_head_cam_plan.mp4 \
  --json-path /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json
```

Common keys:
- `y`: usable
- `n`: unusable
- `a/d`: previous/next video
- `j/l`: previous/next frame
- `r`: replay current video
- `space`: play/pause
- `q`: save and quit

That JSON can be passed directly to the processing script below.

## 6.1 Process the new “head-cam repaint + wrist replay” pipeline

Run inside the RoboTwin repo:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

python scripts/process_repainted_headcam_with_wrist.py \
  d_pour_blue \
  "pour water" \
  48 \
  --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue \
  --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' \
  --head-video-name target_with_original_head_cam_plan.mp4 \
  --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue \
  --retarget-dir-template 'hand_detections_{id}' \
  --review-json /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json \
  --ignore-ids
```

Notes:
- if `--review-json` is provided, the script automatically processes only videos marked `y` / `usable=true`
- if `--review-json` is not provided, the script auto-discovers episode ids from `head-root`
- it then looks up wrist videos and `world_targets_and_status.npz` with the same id under `retarget-root`
- `--ignore-ids` can be passed with no values, meaning “ignore nothing in this run”

## 6.2 Only process selected ids

```bash
python scripts/process_repainted_headcam_with_wrist.py \
  d_pour_blue \
  "pour water" \
  3 \
  --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue \
  --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' \
  --head-video-name target_with_original_head_cam_plan.mp4 \
  --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue \
  --retarget-dir-template 'hand_detections_{id}' \
  --ids 0 1 2 \
  --ignore-ids
```

## 6.3 Process the older zed repaint result instead

```bash
python scripts/process_repainted_headcam_with_wrist.py \
  d_pour_blue \
  "pour water" \
  48 \
  --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue \
  --head-dir-template 'id_{id}' \
  --head-video-name final_repainted.mp4 \
  --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue \
  --retarget-dir-template 'hand_detections_{id}' \
  --ignore-ids
```

---

## 7. Alignment rule

The script reads:
- pose length `T_pose`
- head-cam frames `T_head`
- left wrist frames `T_left`
- right wrist frames `T_right`

Then it uses:

```text
usable_len = min(T_pose, T_head, T_left, T_right)
```

After that:
- `state = seq[:usable_len-1]`
- `action = seq[1:usable_len]`
- images are also clipped to `usable_len-1`

So this script does not perform temporal resampling. It only clips all sources to the shortest usable length.

That means:
- if the head-cam video was padded beforehand, it is more likely to align well with wrist/pose lengths
- if one source is much shorter, the final episode length will shrink to that source

---

## 8. Minimum files needed for server migration

You need to keep at least these two groups of directories.

### 8.1 Repainted head-cam directories

```text
results_repaint/<task>/id_<id>_head_cam_.../target_with_original_head_cam_plan.mp4
```

### 8.2 Retarget directories

```text
output_hand_retarget.../<task>/hand_detections_<id>/
  world_targets_and_status.npz
  left_wrist_replay.mp4
  right_wrist_replay.mp4
```

As long as those two groups remain available, the new script can regenerate `processed_data`.

---

## 9. Recommended debugging order

If processing fails, check in this order:

1. confirm the head-cam video exists
   - `target_with_original_head_cam_plan.mp4`
2. confirm the wrist videos exist
   - `left_wrist_replay.mp4`
   - `right_wrist_replay.mp4`
3. confirm `world_targets_and_status.npz` exists
4. confirm the directory templates are correct
   - `id_{id}_head_cam_arm_gripper_cup_bottle_pad_target`
   - `hand_detections_{id}`

---

## 10. Related code locations

- AnyGrasp batch entry:
  - `code_painting/run_plan_anygrasp_keyframes_r1_batch.sh`
- AnyGrasp batch implementation:
  - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
- auto-pad + head-cam repaint:
  - `inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad.sh`
- robot repaint implementation:
  - `inpainting_sam2_robot/remove_anything_video_sam2_robot.py`
- old retargeted-human processor:
  - `policy/pi0/scripts/process_data_retageted_human.py`
- new script added in this round:
  - `policy/pi0/scripts/process_repainted_headcam_with_wrist.py`

---

## 11. Validation in this round

Minimal validation already run:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw
python -m py_compile scripts/process_repainted_headcam_with_wrist.py scripts/process_data_retageted_human.py scripts/process_data_R1.py
```

When producing real data, it is recommended to first run a tiny subset:
- `--ids 0`
- or `--ids 0 1 2`

After confirming `processed_data/.../episode_0/episode_0.hdf5` is generated correctly, run the full batch.
