# 2026-04-24：Piper 新数据（pnp_star_pear）到 HaMeR 再到 RoboTwin 的完整手部流程

## 1. 这份文档解决什么问题

你当前的新数据在：

- `/home/zaijia001/ssd/data/piper/hand/pnp_star_pear`

结构是 `episode*/camera/color/headD435/*.png + camera/depth/headD435/*.png|*_meters.npy`，
而 HaMeR 现有脚本 `detect_hands_realr1.py` 期待的是扁平化的 RealR1 风格输入：

- `rgb_<id>.mp4`
- `depth_<id>.mp4`
- `params_<id>.json`

因此这里补齐一个“HaMeR 转换与检测 -> FoundationPose 双物体位姿（pear+star）-> RoboTwin 回放”的可执行流程。

---

## 2. 与旧文档的关系（你之前维护过的内容）

可参考历史说明：

- `/home/zaijia001/ssd/RoboTwin/agent-read/V1.7_hand_pipeline_ZH.md`
- `/home/zaijia001/ssd/RoboTwin/agent-read/2026-03-27_planner_repaint_pi0_pipeline_relation_ZH.md`

本次文档相当于“上游检测阶段”的补全版，重点是：

1. 明确找到 HaMeR 与 FoundationPose 代码位置；
2. 针对 Piper 新数据结构给出专门转换；
3. 给出能直接跑通并可视化的命令；
4. 补齐双物体（pear + star）位姿检测命令；
5. 把产物对齐到 RoboTwin 下游可消费格式（`hand_detections_<id>.npz` 与 `obj_vis/<scene>/<object>/poses.npz`）。

---

## 3. 本机确认到的关键代码位置

- HaMeR 工程根目录：
  - `/home/zaijia001/ssd/hamer_r1`
- 人手检测主脚本：
  - `/home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py`
- 本次新增的 Piper 结构转换脚本（HaMeR）：
  - `/home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py`
- 本次新增的 Piper 结构转换脚本（FoundationPose）：
  - `/home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py`
- 本次新增的 pear+star 专用运行脚本：
  - `/home/zaijia001/FoundationPose/run_piper_star_pear_foundation.sh`

---

## 4. 数据结构映射（Piper -> HaMeR 输入）

### Piper 原始结构（每个 episode）

- RGB：`episodeX/camera/color/headD435/*.png`
- Depth：`episodeX/camera/depth/headD435/*.png` 与 `*_meters.npy`
- 内参：`episodeX/camera/color/headD435_camera_info.json`

### 转换后的 RealR1 风格结构（扁平目录）

- `rgb_<id>.mp4`
- `depth_<id>.mp4`
- `params_<id>.json`

其中 `<id>` 与 `episode` 数字一致（`episode0 -> 0`）。

---

## 5. 标准执行步骤（可直接复制）

### Step A：先转换数据

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/convert_piper_dataset_to_hamer.py \
  --input_root /home/zaijia001/ssd/data/piper/hand/pnp_star_pear \
  --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input \
  --use_meta_fps \
  --overwrite
```

### Step B：跑 HaMeR 手关节检测（单条先验证）

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py \
  --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input \
  --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output \
  --video_ids 0 \
  --device cpu
```

### Step C：批量跑全部 episode

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda run -n hamer-r1 python /home/zaijia001/ssd/hamer_r1/detect_hands_realr1.py \
  --data_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_input \
  --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output \
  --device cpu
```

### Step D：可视化检查（HaMeR）

```bash
ffplay -autoexit /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output/hand_vis_0.mp4
```

```bash
ffplay -autoexit /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output/hand_vis_gripper_0.mp4
```

### Step E：为 FoundationPose 准备 metric depth 输入（必须）

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda run -n hamer-r1 python /home/zaijia001/FoundationPose/prepare_piper_for_foundationpose.py \
  --input_root /home/zaijia001/ssd/data/piper/hand/pnp_star_pear \
  --output_dir /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_foundation_input \
  --use_meta_fps \
  --overwrite
```

### Step F：FoundationPose 跑双物体位姿（pear + star）

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

只测一个 id（例如 0）时加：`--video_ids 0`

也可以直接用专门脚本：

```bash
bash /home/zaijia001/FoundationPose/run_piper_star_pear_foundation.sh
```

### Step G：FoundationPose 后续在 RoboTwin 回放双物体

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

## 6. 输出产物说明

HaMeR 检测完成后，每个 `<id>` 对应：

- `hand_detections_<id>.npz`（下游主输入）
- `hand_detections_<id>.npy`
- `hand_vis_<id>.mp4`（骨架可视化）
- `hand_vis_gripper_<id>.mp4`（带夹爪坐标轴）

RoboTwin 下游通常直接消费：

- `hand_detections_<id>.npz`

FoundationPose 双物体输出（每个 id）：

- `.../obj_vis/pnp_star_pear_foundation_input_<id>/pear/poses.npz`
- `.../obj_vis/pnp_star_pear_foundation_input_<id>/star/poses.npz`

---

## 7. 一条命令怎么“发送”

这里“发送”的实际操作是：

1. 打开终端；
2. 复制上面的完整命令；
3. 粘贴后回车执行。

建议先按 Step A -> Step B（只跑 `video_ids 0`）验证，再跑 Step C 全量；
FoundationPose 也建议先用 `--video_ids 0` 验证，再去全量。

---

## 8. RoboTwin 对接命令（下游）

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz_urdfik.sh \
  /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_pnp_star_pear \
  5 \
  --lighting_mode front_no_shadow
```

---

## 9. 本次状态

已完成：

- 新增转换脚本（HaMeR）：`convert_piper_dataset_to_hamer.py`
- 新增转换脚本（FoundationPose）：`prepare_piper_for_foundationpose.py`
- 新增运行脚本（pear+star）：`run_piper_star_pear_foundation.sh`
- 已将 `pnp_star_pear` 的 16 个 episode 转换到：
  - `pnp_star_pear_hamer_input`
  - `pnp_star_pear_foundation_input`
- 已在 `video_id=0` 上生成 HaMeR 检测结果和可视化视频（`hand_vis_0.mp4`, `hand_vis_gripper_0.mp4`）
