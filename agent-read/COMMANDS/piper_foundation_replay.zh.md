# Piper FoundationPose Replay 命令

## 用途

记录 Piper H2O 任务的 FoundationPose 结果在 RoboTwin 中 replay 的可复用命令入口。

## 适用版本

- 当前推荐标定：`calibration_bundle_piper_new_table_0515.json` 对应的 0515 head D435 / base 标定。
- 当前命令库位置：`/home/zaijia001/ssd/RoboTwin/COMMAND_LIBRARY.zh.md` 的 C1 / C1.2。

## D435 内参版 replay

C1.2 新增六个 H2O 任务的 D435 相机内参版 FoundationPose replay。它复用 C1 的 FoundationPose 输入、物体 mesh、Piper head 外参和 `--camera_cv_axis_mode legacy_r1`，但渲染相机改成与后续机器人 pure replay E2.4 一致：

```text
--image_width 640 --image_height 480 --fovy_deg 42.499880046655484
```

输出目录统一使用：

```text
/home/zaijia001/ssd/data/piper/hand/<TASK>/foundation_replay_d435
```

## 任务覆盖

- `pick_diverse_bottles`
- `place_bread_basket`
- `stack_cups`
- `handover_bottle`
- `pnp_bread`
- `pnp_tray`

## 相关代码

- 批处理 wrapper：`/home/zaijia001/ssd/RoboTwin/code_painting/run_multi_object_pose_r1_npz_batch.sh`
- 批处理 Python：`/home/zaijia001/ssd/RoboTwin/code_painting/render_multi_object_pose_r1_npz_batch.py`
- 单条 replay：`/home/zaijia001/ssd/RoboTwin/code_painting/render_multi_object_pose_r1_npz.py`

## 使用备注

- C1.2 只改变渲染内参和输出目录，不改变 FoundationPose 输入、不改变 mesh、不改变 head 外参。
- D435 FOV 来自 E2.4 记录的人手 RGB camera_info：`fovy = 2 * atan(height / (2 * fy))`。
