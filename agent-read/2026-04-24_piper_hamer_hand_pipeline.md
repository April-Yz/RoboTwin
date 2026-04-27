# 2026-04-24: End-to-end Piper (pnp_star_pear) -> HaMeR -> RoboTwin hand pipeline

## 1. What this document covers

Your new dataset is located at:

- `/home/zaijia001/ssd/data/piper/hand/pnp_star_pear`

Its structure is `episode*/camera/color/headD435/*.png + camera/depth/headD435/*.png|*_meters.npy`,
while the current HaMeR script `detect_hands_realr1.py` expects a flat RealR1-style input:

- `rgb_<id>.mp4`
- `depth_<id>.mp4`
- `params_<id>.json`

So this document fills the full runnable path: HaMeR conversion/detection -> FoundationPose (pear+star) object pose -> RoboTwin replay.

---

## 2. Relation to earlier notes

Previous related docs:

- `/home/zaijia001/ssd/RoboTwin/agent-read/V1.7_hand_pipeline_ZH.md`
- `/home/zaijia001/ssd/RoboTwin/agent-read/2026-03-27_planner_repaint_pi0_pipeline_relation_ZH.md`

This update complements the upstream detection stages by:

1. locating both HaMeR and FoundationPose code locations,
2. adding converters for the Piper dataset layout,
3. providing directly executable commands,
4. adding pear+star multi-object FoundationPose commands,
5. producing downstream-ready outputs (`hand_detections_<id>.npz` and `obj_vis/.../poses.npz`).

---

## 3. Key code locations (confirmed)

- HaMeR project root:
  - `/home/zaijia001/ssd/hamer_r1`
- Hand detection entry:
  - `/home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py`
- Newly added Piper converter (HaMeR):
  - `/home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py`
- Newly added Piper converter (FoundationPose):
  - `/home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py`
- Newly added pear+star wrapper script:
  - `/home/zaijia001/FoundationPose/run_piper_star_pear_foundation.sh`

---

## 4. Data mapping (Piper -> HaMeR input)

### Original Piper layout (per episode)

- RGB: `episodeX/camera/color/headD435/*.png`
- Depth: `episodeX/camera/depth/headD435/*.png` and `*_meters.npy`
- Intrinsics: `episodeX/camera/color/headD435_camera_info.json`

### Converted RealR1-style layout (flat directory)

- `rgb_<id>.mp4`
- `depth_<id>.mp4`
- `params_<id>.json`

`<id>` matches the episode number (`episode0 -> 0`).

---

## 5. Standard commands

### Step A: Convert dataset

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py \
  --input_root /home/zaijia001/ssd/data/piper/hand/pnp_star_pear \
  --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input \
  --use_meta_fps \
  --overwrite
```

### Step B: Run HaMeR on one sequence first (sanity check)

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py \
  --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input \
  --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output \
  --video_ids 0 \
  --device cpu
```

### Step C: Run all episodes

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py \
  --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input \
  --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output \
  --device cpu
```

### Step D: Visualize HaMeR output

```bash
ffplay -autoexit /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output/hand_vis_0.mp4
```

```bash
ffplay -autoexit /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output/hand_vis_gripper_0.mp4
```

### Step E: Prepare metric depth input for FoundationPose (required)

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda run -n hamer-r1 python /home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py \
  --input_root /home/zaijia001/ssd/data/piper/hand/pnp_star_pear \
  --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_input \
  --use_meta_fps \
  --overwrite
```

### Step F: Run FoundationPose for pear + star

```bash
source /home/zaijia001/FoundationPose/source_foundationpose_env.sh
cd /home/zaijia001/FoundationPose

CUDA_VISIBLE_DEVICES=1 python run_realr1_dino_sam_batch.py \
  --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_input \
  --output_root /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis \
  --object pear=/home/zaijia001/ssd/data/R1/hand/obj_mesh/pear/pear.obj \
  --object star=/home/zaijia001/ssd/data/R1/hand/obj_mesh/star/star.obj \
  --save_video 1 \
  --save_mesh_overlay_video 1 \
  --save_bbox_overlay_video 1 \
  --mesh_overlay_alpha 0.45
```

For a quick sanity run, add `--video_ids 0`.

Or use the dedicated wrapper:

```bash
bash /home/zaijia001/FoundationPose/run_piper_star_pear_foundation.sh
```

### Step G: Replay FoundationPose outputs in RoboTwin

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh \
  /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_vis/obj_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_pnp_star_pear_norobot \
  5 \
  --lighting_mode front_no_shadow \
  --hide_robot 1 \
  --save_head_depth 1 \
  --save_anygrasp_frames 1 \
  --object pear=/home/zaijia001/ssd/data/R1/hand/obj_mesh/pear/pear.obj \
  --object star=/home/zaijia001/ssd/data/R1/hand/obj_mesh/star/star.obj
```

---

## 6. Output files

For each `<id>`, HaMeR outputs are:

- `hand_detections_<id>.npz` (main downstream input)
- `hand_detections_<id>.npy`
- `hand_vis_<id>.mp4`
- `hand_vis_gripper_<id>.mp4`

RoboTwin downstream primarily consumes:

- `hand_detections_<id>.npz`

FoundationPose multi-object outputs (per id) include:

- `.../obj_vis/pnp_star_pear_foundation_input_<id>/pear/poses.npz`
- `.../obj_vis/pnp_star_pear_foundation_input_<id>/star/poses.npz`

---

## 7. How to “send” a command

In practice:

1. open a terminal,
2. paste one full command,
3. press Enter.

Recommended order: Step A -> Step B (`video_id=0`) -> Step C (full batch),
then FoundationPose with `--video_ids 0` first, and finally full-batch.

---

## 8. RoboTwin downstream replay command

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz_urdfik.sh \
  /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_pnp_star_pear \
  5 \
  --lighting_mode front_no_shadow
```

---

## 9. Current status

Completed:

- Added converter (HaMeR): `convert_piper_dataset_to_hamer.py`
- Added converter (FoundationPose): `prepare_piper_for_foundationpose.py`
- Added wrapper (pear+star): `run_piper_star_pear_foundation.sh`
- Converted all 16 episodes into:
  - `pnp_star_pear_hamer_input`
  - `pnp_star_pear_foundation_input`
- Generated HaMeR detection + visualization for `video_id=0` (`hand_vis_0.mp4`, `hand_vis_gripper_0.mp4`)
