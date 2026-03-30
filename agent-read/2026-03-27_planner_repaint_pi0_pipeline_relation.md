# 2026-03-27: Relationship among planner -> SAM repaint -> pi0 processing

## 1. Recommended way to understand this pipeline

It is best to view the current pipeline as three consecutive steps.

### Step 1: planner / AnyGrasp generates execution videos and state logs
Entry command:
- `code_painting/run_plan_anygrasp_keyframes_r1_batch.sh`

It outputs:
- `head_cam_plan.mp4`
- `left_wrist_cam_plan.mp4`
- `right_wrist_cam_plan.mp4`
- `pose_debug.jsonl`
- `plan_summary.json`

These files are stored under:

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_<id>/
```

### Step 2: SAM / repaint composites the planner head onto the background
Entry command:
- `inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad.sh`

It reads the Step-1 file:
- `head_cam_plan.mp4`

and produces:
- `target_with_original_head_cam_plan.mp4`

Output directory:

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/id_<id>_head_cam_arm_gripper_cup_bottle_pad_target/
```

### Step 3: pi0 data processing
New recommended entry:
- `policy/pi0/scripts/process_repainted_planner_outputs.py`

It reads:
- the repaint head from Step 2:
  - `target_with_original_head_cam_plan.mp4`
- the planner wrist videos from Step 1:
  - `left_wrist_cam_plan.mp4`
  - `right_wrist_cam_plan.mp4`
- the planner state log from Step 1:
  - `pose_debug.jsonl`

and converts them into:

```text
processed_data/<task>-<num>/episode_x/
  instructions.json
  episode_x.hdf5
```

So the relationship is:

```text
run_plan_anygrasp_keyframes_r1_batch.sh
    ↓
  generate planner head / wrist / pose_debug
    ↓
batch_head_cam_repaint_with_auto_pad.sh
    ↓
  run SAM/repaint on planner head and generate target_with_original_head_cam_plan.mp4
    ↓
process_repainted_planner_outputs.py
    ↓
  use repaint head + planner wrist + planner pose_debug to generate pi0 HDF5
```

---

## 2. Why this new script is now recommended

The previous issue was that:
- head came from planner `head_cam_plan.mp4`
- wrist / pose came from hand-retarget `left/right_wrist_replay.mp4` + `world_targets_and_status.npz`

That mixed two different sources:
- planner stream
- hand-retarget stream

As a result:
- head often had `255~263` frames
- wrist / pose often had only `35~60`
- the final processed length was capped by the shorter wrist / pose streams

The new script `process_repainted_planner_outputs.py` changes this to:
- head: still use the repaint planner head
- wrist: use planner wrist camera videos
- state: use planner `pose_debug.jsonl`

Now all three streams come from the same planner execution chain, so their lengths are aligned.

---

## 3. Step-1 command: planner / AnyGrasp batch

Below is a full example equivalent to the planner stage in your current workflow.

```bash
cd /home/zaijia001/ssd/RoboTwin
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3 \
  --ids 0 \
  --skip_existing 0 \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps 20 \
  --urdfik_cartesian_interp_auto_step_m 0.05 \
  --candidate_selection_mode planner \
  --left_target_object cup \
  --right_target_object bottle \
  --candidate_target_local_x_offset_m -0.05 \
  --approach_offset_m 0.20 \
  --reach_error_pose_source tcp \
  --replan_until_reached 1 \
  --replan_until_reached_max_attempts 3 \
  --save_debug_preview 1 \
  --save_debug_execution_preview 1 \
  --reach_pos_tol_m 0.03 \
  --reach_rot_tol_deg 20 \
  --enable_grasp_action_object_collision 1 \
  --grasp_action_object_collision_start_stage pregrasp \
  --execution_object_collision_mode convex \
  --execution_object_visual_scale_override cup=0.7 \
  --execution_object_collision_scale_override cup=0.3 \
  --execution_object_visual_scale_override bottle=0.7 \
  --execution_object_collision_scale_override bottle=0.3 \
  --debug_visualize_targets 1 \
  --debug_visualize_ik_waypoints 1 \
  --debug_collision_report 1 \
  --debug_visualize_object_collision_bbox 1 \
  --gripper_contact_monitor_mode all_robot_links \
  --pure_scene_output 1 \
  --overlay_text 0 \
  --viewer_show_camera_frustums 0 \
  --viewer_frame_delay 0.02 \
  --viewer_wait_at_end 1 \
  --settle_steps 10 \
  --joint_target_wait_steps 70 \
  --disable_table 1 \
  --base_occluder_enable 0 \
  --enable_viewer 1
```

### Important inputs for this step

- `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results`
  - AnyGrasp candidate grasps
- `/home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot`
  - object replay trajectories
- `/home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis`
  - hand keyframes / hand data
- `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch`
  - preview keyframes and candidate selection results

### Important outputs of this step

For `id=0`:

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_0/
  head_cam_plan.mp4
  left_wrist_cam_plan.mp4
  right_wrist_cam_plan.mp4
  pose_debug.jsonl
  plan_summary.json
```

Where:
- `head_cam_plan.mp4` is the head input for later SAM/repaint
- `left/right_wrist_cam_plan.mp4` are the wrist videos used later by pi0 processing
- `pose_debug.jsonl` is the state sequence used later by pi0 processing

---

## 4. Step-2 command: SAM / repaint planner head

```bash
cd /home/zaijia001/ssd/inpainting_sam2_robot
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate inpainting-sam2-r1

bash /home/zaijia001/ssd/inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad.sh 0
```

### Important inputs for this step

By default, the script reads:

- planner head:
  ```text
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_0/head_cam_plan.mp4
  ```
- stage1 background:
  ```text
  /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/stage1_no-human-obj_d_pour_blue_0/removed_w_mask_rgb_0.mp4
  ```

### Important outputs of this step

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/id_0_head_cam_arm_gripper_cup_bottle_pad_target/
  target_with_original_head_cam_plan.mp4
  w_mask_head_cam_plan.mp4
  w_box_head_cam_plan.mp4
  mask_head_cam_plan.mp4
```

Where:
- `target_with_original_head_cam_plan.mp4`
  - is the head video used later by pi0 processing

---

## 5. Step-3 command: pi0 processing, fully switched to planner-consistent sources

New recommended command:

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

python scripts/process_repainted_planner_outputs.py \
  d_pour_blue \
  "pour water" \
  27 \
  --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue \
  --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' \
  --head-video-name target_with_original_head_cam_plan.mp4 \
  --planner-root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3 \
  --planner-dir-template 'd_pour_blue_{id}' \
  --left-wrist-video-name left_wrist_cam_plan.mp4 \
  --right-wrist-video-name right_wrist_cam_plan.mp4 \
  --pose-debug-name pose_debug.jsonl \
  --review-json /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json \
  --review-mode strict \
  --ignore-ids \
  --output-dir /home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-27-planner
```

### What each argument means in this step

- `--head-root`
  - repaint result root
- `--head-dir-template`
  - repaint episode directory template
- `--head-video-name`
  - repaint head video filename
- `--planner-root`
  - Step-1 planner output root
- `--planner-dir-template`
  - planner episode directory template
- `--left-wrist-video-name`
  - planner left wrist video filename
- `--right-wrist-video-name`
  - planner right wrist video filename
- `--pose-debug-name`
  - planner state log filename
- `--review-json`
  - manual review result; only reviewed usable videos are processed
- `--review-mode strict`
  - strict mode, process only `y`
- `--output-dir`
  - final pi0 HDF5 output directory

### Actual files read by this step

For `id=0`:

#### repaint head

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/id_0_head_cam_arm_gripper_cup_bottle_pad_target/target_with_original_head_cam_plan.mp4
```

#### planner wrist

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_0/left_wrist_cam_plan.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_0/right_wrist_cam_plan.mp4
```

#### planner state

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_0/pose_debug.jsonl
```

---

## 6. How the new script extracts state from planner outputs

New script:
- `policy/pi0/scripts/process_repainted_planner_outputs.py`

State is extracted from these `pose_debug.jsonl` fields:

- `current_left_tcp_pose_world_wxyz`
- `current_right_tcp_pose_world_wxyz`
- `current_left_gripper_command`
- `current_right_gripper_command`

Each frame is converted into a 14D state:

- left arm:
  - `x y z roll pitch yaw gripper`
- right arm:
  - `x y z roll pitch yaw gripper`

So the new state source is no longer:
- `world_targets_and_status.npz`

It is now:
- planner `pose_debug.jsonl`

---

## 7. Why this new scheme is more reasonable

Because all streams are finally source-consistent:

- head: repaint result of planner head
- left wrist: planner left wrist
- right wrist: planner right wrist
- state: planner pose_debug

They all come from the same planner execution trace.

Sampling already confirms that under the planner directory:
- `head_cam_plan.mp4`
- `left_wrist_cam_plan.mp4`
- `right_wrist_cam_plan.mp4`
- `pose_debug.jsonl`

share the same length. For example, `id=0`:

```text
head_cam_plan.mp4         263 frames
left_wrist_cam_plan.mp4   263 frames
right_wrist_cam_plan.mp4  263 frames
pose_debug.jsonl          263 lines
```

So the new script preserves planner execution length instead of being capped by the low-frame-count hand-retarget wrist stream.

---

## 8. Recommended one-line mental model

If you need to explain this pipeline later or move it to another server, the clearest summary is:

```text
run_plan_anygrasp_keyframes_r1_batch.sh
  first generates planner head / wrist / pose_debug
batch_head_cam_repaint_with_auto_pad.sh
  then runs SAM/repaint on the planner head
process_repainted_planner_outputs.py
  finally uses repaint head + planner wrist + planner pose_debug to generate pi0 training data
```

This makes the upstream/downstream relationship explicit and avoids mixing planner streams with hand-retarget streams again.
