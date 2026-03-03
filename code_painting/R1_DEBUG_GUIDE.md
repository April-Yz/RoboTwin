# R1 Debug Guide

## Core Concepts

当前脚本里，“位置”和“朝向”是两条独立链路，不一定来自同一个来源。

### 位置来源

默认位置来自：

- 输入 `npz` 里的 `left/right_gripper_position`
- 或 `left/right_wrist_position_retreat`

如果输入 `npz` 没有这些字段，就会现场调用和
[compute_gripper_pose_from_npz.py](/home/zaijia001/ssd/hamer_r1/compute_gripper_pose_from_npz.py)
一致的逻辑重算，函数是：

- [calc_gripper_pose_from_keypoints](/home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_r1_npz.py#L150)

### 朝向来源

朝向有三种常见来源：

1. `npz` 里存好的 `left/right_gripper_rotation_matrix`
2. 如果 `npz` 没有这些字段，则现场用 `calc_gripper_pose_from_keypoints(...)` 重算
3. 调试时直接忽略人手朝向，改成“机器人当前夹爪朝向 + sweep”

所以：

- 普通 replay：
  默认用“人手/npz 计算出来的朝向”
- `orientation_sweep`：
  位置仍然来自人手/npz
  但朝向会被 sweep 覆盖

## Robot Gripper Pose Convention

在 RoboTwin 当前这套 `robot.py` 里，planner 接收的是“夹爪中心位姿”。

关键规律：

- pose 格式是 `[x, y, z, qw, qx, qy, qz]`
- 世界坐标系里机器人 base 当前“朝前”是 `+Y`
- 机器人夹爪局部 `+X` 基本是“朝前/接近方向”
  依据是 [robot.py](/home/zaijia001/ssd/RoboTwin/envs/robot/robot.py) 里会沿局部 `+X` 做 gripper bias 到 end-link 的换算

所以如果某个 case 看起来“夹爪朝右”，本质上通常是在说：

- 这个 gripper pose 的局部 `+X` 没有朝向任务物体
- 或者人手坐标系 remap 到机器人 gripper 坐标系后，前向轴和你预期不一致

## Debug Modes

### 1. Head Camera Orientation Sweep

用途：

- 调 `zed/head` 相机朝向
- 保存 `zed / third / wrist` 图片

命令：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_camera_sweep \
  5 \
  --camera_sweep_enable 1 \
  --camera_debug_target head \
  --camera_sweep_steps_deg -90 0 90 \
  --debug_mode 1 \
  --debug_frame_limit 1 \
  --enable_viewer 1 \
  --viewer_wait_at_end 1
```

### 2. Wrist Camera Orientation Sweep

用途：

- 调 `left/right wrist camera` 朝向
- 看 wrist 相机视野框是否合理

左手：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_left_wrist_sweep \
  5 \
  --camera_sweep_enable 1 \
  --camera_debug_target left_wrist \
  --camera_sweep_steps_deg -90 0 90 \
  --debug_mode 1 \
  --debug_frame_limit 1 \
  --enable_viewer 1 \
  --viewer_wait_at_end 1
```

右手：

```bash
... --camera_debug_target right_wrist
```

### 3. Use Human-Computed Orientation Directly

用途：

- 使用 `compute_gripper_pose_from_npz.py` 已经写好的朝向
- 不在 replay 里现场重算

要求：

- 输入文件必须是 `*_with_gripper.npz`
- 文件里要有 `left/right_gripper_rotation_matrix`

命令示例：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /path/to/hand_detections_0_with_gripper.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_replay_from_stored_pose \
  5
```

启动时如果日志打印：

- `left: stored_npz`
- `right: stored_npz`

就说明确实在直接使用 `npz` 里的朝向。

### 4. Ignore Human Orientation, Keep Human Position Only

用途：

- 位置继续用人手/npz
- 朝向直接换成机器人当前夹爪朝向
- 用来验证“是不是主要卡在朝向”

命令：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_pose_debug \
  5 \
  --debug_mode 1 \
  --debug_frame_limit 5 \
  --debug_force_orientation current_tcp \
  --enable_viewer 1 \
  --viewer_wait_at_end 1
```

### 5. Orientation Sweep With Robot-Based Orientation

用途：

- 固定人手算出来的位置
- 不直接使用人手朝向
- 改成“机器人当前夹爪朝向 + sweep”

这是目前最接近你最近想要的模式。

双手一起 sweep：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_dual_orientation_sweep \
  5 \
  --orientation_sweep_enable 1 \
  --orientation_sweep_arm both \
  --orientation_sweep_both_mode paired \
  --orientation_sweep_base current_tcp \
  --orientation_sweep_steps_deg -90 0 90 \
  --orientation_sweep_execute 1 \
  --orientation_sweep_reset_each_case 1 \
  --target_world_z_offset 0.0 \
  --debug_mode 1 \
  --debug_frame_limit 1 \
  --enable_viewer 1 \
  --viewer_wait_at_end 1
```

说明：

- 位置：来自人手 / `npz`
- 朝向：来自机器人当前 TCP 朝向，再叠加 `rx/ry/rz`
- 双手是同一个 case 同时规划
- 只要一只手 `Success` 就执行；两只都成功就一起执行

### 6. Dual Orientation Sweep Pair Modes

`--orientation_sweep_both_mode paired`

- 左右手使用同一组 `rx/ry/rz`
- case 数少，适合先找规律

`--orientation_sweep_both_mode cartesian`

- 左右手所有 case 两两组合
- 搜索最全，但非常慢

## Useful Extra Flags

`--viewer_wait_at_end 1`

- 跑完后 viewer 不会自动关

`--viewer_frame_delay 0.005`

- 每帧停一下，便于观察

`--target_world_z_offset 0.15`

- 所有目标世界坐标整体上抬 15cm

`--disable_table 1`

- 不加载桌子

## Current Recommended Workflow

1. 先用 camera sweep 确认 `head / wrist` 相机朝向。
2. 再用 `*_with_gripper.npz` 确认“直接使用人手计算朝向”效果。
3. 如果仍然经常 fail，就切到 `orientation_sweep_base current_tcp`。
4. 如果只有单手 success，优先看位置是否太低，再调 `--target_world_z_offset`。


bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_pos_debug \
  5 \
  --debug_force_orientation current_tcp \
  --target_world_offset_xyz 0 -0.75 -0.24 \
  --debug_mode 1 \
  --debug_frame_limit 5 \
  --enable_viewer 1 \
  --viewer_wait_at_end 1

# 遍历一下z从-0.1到+0.5（5cm一档
```
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_pos_debug_z_sweep \
  5 \
  --debug_force_orientation current_tcp \
  --target_world_offset_xyz 0 0 0 \
  --target_world_offset_z_sweep_enable 1 \
  --target_world_offset_z_sweep_start -0.1 \
  --target_world_offset_z_sweep_end 0.5 \
  --target_world_offset_z_sweep_step 0.05 \
  --debug_mode 1 \
  --debug_frame_limit 5 \
  --enable_viewer 1 \
  --viewer_wait_at_end 1

```