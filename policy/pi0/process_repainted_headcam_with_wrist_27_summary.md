# `process_repainted_headcam_with_wrist.py` 命令分析

## 结论先说

你给的这条命令不会修改输入目录里的原始视频或 npz 文件，只会读取它们，然后在 `policy/pi0/processed_data/d_pour_blue-27/` 下生成新的训练数据。

它不会对视频做压缩、变速、插帧或帧率重采样。实际做的事情是：逐帧读取视频，统一缩放到 `640x480`，然后和状态序列对齐后，把帧编码成 JPEG 字节串写入 hdf5。由于输出只保留对齐后的最短公共长度，最终序列可能比任一输入视频更短，但这是裁剪，不是加速或压缩时长。

## 这条命令会读取什么

命令：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

python scripts/process_repainted_headcam_with_wrist.py \
  d_pour_blue \
  "pour water" \
  27 \
  --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue \
  --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' \
  --head-video-name target_with_original_head_cam_plan.mp4 \
  --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue \
  --retarget-dir-template 'hand_detections_{id}' \
  --review-json /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json \
  --review-mode strict \
  --ignore-ids
```

它会按 id 找到这些输入：

- head 侧视频：`/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/id_{id}_head_cam_arm_gripper_cup_bottle_pad_target/target_with_original_head_cam_plan.mp4`
- retarget 侧数据：`/home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue/hand_detections_{id}/`
- retarget 侧状态文件：`world_targets_and_status.npz`
- retarget 侧腕部视频：`left_wrist_replay.mp4` 和 `right_wrist_replay.mp4`

## 这条命令怎么筛选样本

### `--review-json`

脚本会先读 `video_review.json`，然后只处理被标记为可用的 id。

在 `--review-mode strict` 下，只有这些会被接受：

- `label == "y"`
- 或 `usable == true`

如果是 `m` / `ambiguous`，严格模式会排除。

### `--ignore-ids`

你这条命令写了 `--ignore-ids` 但没有跟任何数字，所以这次等价于“忽略空列表”，不会额外跳过 id。

### `--expert_data_num = 27`

脚本最多会处理 27 个可用 episode，但实际输出数量可能更少，因为它还会跳过：

- review 不通过的 id
- 缺少 head 视频的 id
- 缺少 wrist 视频的 id
- 缺少 `world_targets_and_status.npz` 的 id
- 读不到视频或视频为空的 id

## 它会不会改视频时长、帧率或速度

不会做这些处理：

- 不会重编码成更短或更长的视频
- 不会改 fps 元数据
- 不会做插帧
- 不会做时间拉伸
- 不会做快放/慢放

它唯一和“时长”有关的动作是裁剪对齐：

1. 先分别读取 head 视频、左腕视频、右腕视频、状态序列的所有帧。
2. 取最短长度 `usable_len = min(T_state, T_head, T_left, T_right)`。
3. 只保留前 `usable_len` 个时间步。
4. 再把 `states = seq[:-1]`、`actions = seq[1:]`，所以最终每个 episode 的有效长度是 `usable_len - 1`。

所以你看到的“视频变短”，是因为不同来源长度不一致后被裁掉了尾部，不是因为脚本把视频加速了。

## 它对图像做了什么

读取视频后，每一帧都会被统一缩放到：

- 宽 `640`
- 高 `480`

然后再用 OpenCV 的 JPEG 编码存入 hdf5。

这意味着：

- 输入视频本身不被改写
- 输出 hdf5 里保存的是 JPEG 字节串，不是原始 mp4
- 压缩发生在“存储格式”层面，而不是“播放时长”层面

## 它对状态数据做了什么

脚本读取 `world_targets_and_status.npz` 里的字段：

- `left_world_targets`
- `right_world_targets`
- `left_gripper_value`
- `right_gripper_value`
- `left_plan_status`
- `right_plan_status`

然后：

1. 把四元数 `wxyz` 转成欧拉角 `xyz`
2. 拼成每只手臂 7 维状态：
   - `x, y, z, roll, pitch, yaw, gripper`
3. 左右手臂拼成 14 维状态
4. 对 `Success` 之外的帧做 forward-fill

所以状态里的坏帧不是丢掉，而是用前一个有效状态向前补齐。

## 最终会生成什么

默认输出目录是：

```text
/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/d_pour_blue-27/
```

每个 episode 目录形如：

```text
episode_0/
  instructions.json
  episode_0.hdf5
```

### `instructions.json`

内容类似：

```json
{
  "instructions": ["pour water"],
  "source_episode_id": 27
}
```

### `episode_0.hdf5`

里面的主要字段是：

```text
/action                                (T-1, 14) float32
/observations/state                    (T-1, 14) float32
/observations/left_arm_dim             (T-1,) int32
/observations/right_arm_dim            (T-1,) int32
/observations/images/cam_high          (T-1,) JPEG bytes
/observations/images/cam_left_wrist    (T-1,) JPEG bytes
/observations/images/cam_right_wrist   (T-1,) JPEG bytes
```

其中：

- `/observations/state[t]` 是时刻 `t` 的状态
- `/action[t]` 是时刻 `t+1` 的状态
- 每只手臂的 7 维顺序是：`x y z roll pitch yaw gripper`
- 左右手臂拼起来一共 14 维

## 这次输出里不包含什么

这条脚本不会额外保存：

- 原始 mp4 的 fps 元数据
- 变速后的视频文件
- 任何中间可视化视频
- 任何额外的压缩摘要文件

它只会保存 pi0 训练需要的中间 episode 格式。

## 你这条命令的实际数据流

可以把它理解成：

1. 用 `video_review.json` 挑出可用 id
2. 去 head root 找 repainted head 视频
3. 去 retarget root 找 wrist 视频和 world pose npz
4. 读完整帧序列并统一缩放
5. 按最短长度裁剪对齐
6. 构造 14 维 state/action
7. 把图像帧 JPEG 化后写入 hdf5
8. 在 `processed_data/d_pour_blue-27/episode_*` 下落盘

## 一句话总结

这条命令是“把已有人工筛过的 repainted head 视频 + retarget wrist 视频 + world_targets_and_status.npz 整理成 pi0 训练集”，它不会改变视频快慢，只会对齐长度、缩放分辨率，并把结果写成新的 hdf5 数据集。