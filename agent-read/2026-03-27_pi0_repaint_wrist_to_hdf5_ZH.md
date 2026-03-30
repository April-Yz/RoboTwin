# 2026-03-27：wrist 视频 + SAM 后 head cam 视频转 pi0 HDF5 数据

## 1. 目的

这份说明整理了当前这条链路：

1. RoboTwin 规划并导出 `head_cam_plan.mp4`
2. `inpainting_sam2_robot` 把 head cam 视频和背景视频合成，得到 SAM/repaint 后的 head cam 视频
3. `policy/pi0/scripts/process_repainted_headcam_with_wrist.py` 把：
   - head cam 视频
   - 左右 wrist 视频
   - `world_targets_and_status.npz`
   统一转成 pi0 训练使用的 `processed_data/<task>-<num>/episode_x/*.hdf5`

这样迁移到别的服务器时，只需要按这里的输入目录约定准备数据，然后直接跑对应命令即可。

---

## 2. 这次检查过的关键命令和代码

### 2.1 AnyGrasp 规划批处理

入口 shell：
- `code_painting/run_plan_anygrasp_keyframes_r1_batch.sh`

实际只做三件事：
- 进入 RoboTwin 根目录
- 激活 `RoboTwin_bw` conda 环境
- 调用：
  - `code_painting/plan_anygrasp_keyframes_r1_batch.py`

你的命令本质是在跑：
- AnyGrasp 候选抓取 + hand keyframes + object replay
- 输出每个 episode 的：
  - `head_cam_plan.mp4`
  - `plan_summary.json`
  - 调试视频

### 2.2 自动补长 stage1 背景并做 head cam repaint

入口 shell：
- `inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad.sh`

它的处理逻辑是：
1. 读取 `head_cam_plan.mp4` 的 fps / duration
2. 用 `ffmpeg` 把 stage1 背景视频补长成和机器人视频接近相同时长
3. 调用：
   - `remove_anything_video_sam2_robot.py`
4. 得到新的合成 head cam 视频，例如：
   - `target_with_original_head_cam_plan.mp4`

### 2.3 旧的 retargeted-human 处理脚本

旧脚本：
- `policy/pi0/scripts/process_data_retageted_human.py`

它的核心处理方法是：
1. 读取 `world_targets_and_status.npz`
2. 取：
   - `left_world_targets`
   - `right_world_targets`
   - `left_gripper_value`
   - `right_gripper_value`
   - `left_plan_status`
   - `right_plan_status`
3. 把四元数 `(w,x,y,z)` 转成欧拉角 `(x,y,z)`
4. 组合成 14 维状态：
   - 左臂 `xyz + euler + gripper` = 7 维
   - 右臂 `xyz + euler + gripper` = 7 维
5. 对非 `Success` 帧做 forward-fill
6. 读取视频：
   - head cam
   - left wrist
   - right wrist
7. 对齐长度后写成：
   - `processed_data/<task>-<num>/episode_x/instructions.json`
   - `processed_data/<task>-<num>/episode_x/episode_x.hdf5`

但旧脚本更偏向旧目录格式，直接处理现在混合在一个目录里的 `id_0`、`id_0_head_cam_*` 不够稳。

---

## 3. 新增脚本

新增脚本：
- `policy/pi0/scripts/process_repainted_headcam_with_wrist.py`

这个脚本专门支持“新的 wrist 视频 + SAM 后的 head cam 视频”这条链路。

它和旧脚本相比，主要多了这些能力：

1. 显式区分：
   - head cam 根目录
   - retarget 根目录
2. 支持模板化目录名：
   - `id_{id}`
   - `id_{id}_head_cam_arm_gripper_cup_bottle_pad_target`
   - `hand_detections_{id}`
3. 支持直接指定新的 head cam 视频文件名：
   - `target_with_original_head_cam_plan.mp4`
4. 支持自动从目录模板发现可处理的 id
5. 输出格式仍然和 pi0 现有 `processed_data` 一致

---

## 4. 新脚本的输入格式

### 4.1 head cam repaint 根目录

例子：

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue
```

里面的 episode 目录可以是：

```text
id_0_head_cam_arm_gripper_cup_bottle_pad_target/
  target_with_original_head_cam_plan.mp4

id_1_head_cam_arm_gripper_cup_bottle_pad_target/
  target_with_original_head_cam_plan.mp4
```

此时：
- `--head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target'`
- `--head-video-name target_with_original_head_cam_plan.mp4`

如果你要处理旧的 zed repaint 结果，则目录可能是：

```text
id_0/
  final_repainted.mp4
```

此时：
- `--head-dir-template 'id_{id}'`
- `--head-video-name final_repainted.mp4`

### 4.2 retarget 根目录

例子：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue
```

里面每个 episode 目录通常是：

```text
hand_detections_0/
  world_targets_and_status.npz
  left_wrist_replay.mp4
  right_wrist_replay.mp4
  zed_replay.mp4
```

对应参数：
- `--retarget-dir-template 'hand_detections_{id}'`

### 4.3 `world_targets_and_status.npz` 关键字段

当前脚本会读取：
- `left_world_targets` `(T, 7)`
- `right_world_targets` `(T, 7)`
- `left_gripper_value` `(T,)`
- `right_gripper_value` `(T,)`
- `left_plan_status` `(T,)`
- `right_plan_status` `(T,)`

语义：
- `world_targets`: `[x, y, z, qw, qx, qy, qz]`
- `gripper_value`: 夹爪值
- `plan_status == Success` 表示该帧可直接使用

---

## 5. 输出格式

输出目录默认是：

```text
policy/pi0/processed_data/<task_name>-<expert_data_num>/
```

每个 episode 生成：

```text
episode_0/
  instructions.json
  episode_0.hdf5
```

### 5.1 `instructions.json`

示例：

```json
{
  "instructions": ["pour water"],
  "source_episode_id": 0
}
```

### 5.2 HDF5 结构

```text
/action                                (N-1, 14) float32
/observations/state                    (N-1, 14) float32
/observations/left_arm_dim             (N-1,) int32
/observations/right_arm_dim            (N-1,) int32
/observations/images/cam_high          (N-1,) JPEG bytes
/observations/images/cam_left_wrist    (N-1,) JPEG bytes
/observations/images/cam_right_wrist   (N-1,) JPEG bytes
```

状态/动作 14 维语义：
- 左臂：`x y z roll pitch yaw gripper`
- 右臂：`x y z roll pitch yaw gripper`

图像默认 resize 到：
- `640 x 480`

---

## 6. 新的处理命令

## 6.0 先做人眼筛选（推荐）

先在有头界面里逐个检查 repaint 后的视频质量：

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

常用按键：
- `y`：可用
- `n`：不可用
- `a/d`：上一个/下一个视频
- `j/l`：上一帧/下一帧
- `r`：重播当前视频
- `space`：播放/暂停
- `q`：保存并退出

这个 JSON 可以直接给下面的新处理脚本使用。

## 6.1 处理“新 head cam repaint + wrist replay”

在 RoboTwin 仓库里执行：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

python scripts/process_repainted_headcam_with_wrist.py \
  d_pour_blue \
  "pour water" \
  60 \
  --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue \
  --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' \
  --head-video-name target_with_original_head_cam_plan.mp4 \
  --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue \
  --retarget-dir-template 'hand_detections_{id}' \
  --review-json /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json \
  --ignore-ids
```

说明：
- 如果提供了 `--review-json`，脚本会自动只处理 JSON 里标记为 `y` / `usable=true` 的视频
- 如果不提供 `--review-json`，它会自动从 `head-root` 里发现符合模板的 episode id
- 再去 `retarget-root` 找同 id 的 wrist 视频和 `world_targets_and_status.npz`
- `--ignore-ids` 可以空着传，表示本轮不忽略任何 id

## 6.2 只跑指定 id

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

## 6.3 如果要处理旧的 zed repaint 结果

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

## 7. 数据对齐规则

脚本会分别读入：
- pose 序列长度 `T_pose`
- head cam 帧数 `T_head`
- left wrist 帧数 `T_left`
- right wrist 帧数 `T_right`

然后取：

```text
usable_len = min(T_pose, T_head, T_left, T_right)
```

之后：
- `state = seq[:usable_len-1]`
- `action = seq[1:usable_len]`
- 图像也截到 `usable_len-1`

所以这个脚本不会自己做时间重采样，只做“按最短长度裁切对齐”。

这意味着：
- 如果 head cam 被 pad 过，更容易和 wrist / pose 长度对齐
- 如果有一路视频明显更短，最终 episode 长度也会跟着变短

---

## 8. 迁移到别的服务器时最少需要带什么

至少要保留两类目录：

### 8.1 repaint head cam 目录

```text
results_repaint/<task>/id_<id>_head_cam_.../target_with_original_head_cam_plan.mp4
```

### 8.2 retarget 目录

```text
output_hand_retarget.../<task>/hand_detections_<id>/
  world_targets_and_status.npz
  left_wrist_replay.mp4
  right_wrist_replay.mp4
```

只要这两类目录还在，新脚本就能重新生成 `processed_data`。

---

## 9. 推荐排查顺序

如果处理失败，按下面顺序查：

1. 先看 head cam 视频是否存在
   - `target_with_original_head_cam_plan.mp4`
2. 再看 wrist 视频是否存在
   - `left_wrist_replay.mp4`
   - `right_wrist_replay.mp4`
3. 再看 `world_targets_and_status.npz` 是否存在
4. 最后确认目录模板是否写对
   - `id_{id}_head_cam_arm_gripper_cup_bottle_pad_target`
   - `hand_detections_{id}`

---

## 10. 相关代码位置

- AnyGrasp 批处理入口：
  - `code_painting/run_plan_anygrasp_keyframes_r1_batch.sh`
- AnyGrasp 主批处理实现：
  - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
- 自动补长 + head cam repaint：
  - `inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad.sh`
- 机械臂 repaint 主实现：
  - `inpainting_sam2_robot/remove_anything_video_sam2_robot.py`
- 旧的人类 retarget 数据处理脚本：
  - `policy/pi0/scripts/process_data_retageted_human.py`
- 新增的新脚本：
  - `policy/pi0/scripts/process_repainted_headcam_with_wrist.py`

---

## 11. 本次验证

已做的最小验证：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw
python -m py_compile scripts/process_repainted_headcam_with_wrist.py scripts/process_data_retageted_human.py scripts/process_data_R1.py
```

建议你真正落数据时，再先跑一个小样本：
- `--ids 0`
- 或 `--ids 0 1 2`

确认 `processed_data/.../episode_0/episode_0.hdf5` 能正常生成后，再跑全量。
