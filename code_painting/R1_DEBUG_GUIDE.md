# R1 Debug Guide

## Core Concepts

当前脚本里，“位置”和“朝向”是两条独立链路。

- 位置：
  来自输入 `npz` 里的 `left/right_gripper_position` 或 `left/right_wrist_position_retreat`
- 朝向：
  来自输入 `npz` 里的 `left/right_gripper_rotation_matrix`
  如果 `npz` 没有这些字段，就现场调用
  [calc_gripper_pose_from_keypoints](/home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_r1_npz.py)
  重算

启动时如果打印：

- `left: stored_npz`
- `right: stored_npz`

说明直接用了 `*_with_gripper.npz` 里的结果。否则就是现场从 keypoints 重算。

## Robot Gripper Convention

RoboTwin 这套 `robot.py` 里，planner 期待的是“夹爪中心 pose”，而不是任意手部坐标系。

关键点：

- pose 格式是 `[x, y, z, qw, qx, qy, qz]`
- 机器人 base 当前朝前是世界 `+Y`
- 机器人夹爪局部 `+X` 才更接近“前向/接近方向”
- [robot.py](/home/zaijia001/ssd/RoboTwin/envs/robot/robot.py) 里会先把 gripper pose 转成 end-link pose，再喂给 planner

所以“人手 gripper 旋转矩阵”和“机器人 TCP 旋转矩阵”不是天然同一套轴定义。

## Camera Debug

### Head Camera Sweep

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

### Wrist Camera Sweep

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

右手把 `left_wrist` 改成 `right_wrist`。

## Position Debug

### 只保留人手位置，朝向固定成当前 TCP

这条最适合先判断“主要是位置错了，还是朝向错了”。

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

日志会打印：

- `world_dxyz`
- `base_dxyz`
- `axis_err[x/y/z]`
- `target_axes x/y/z`
- `current_axes x/y/z`

### 手动整体平移目标位置

```bash
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
```

### Z Offset Sweep

从 `-0.1m` 到 `+0.5m`，每 `0.05m` 一档：

```bash
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

## Orientation Debug

### 直接使用人手朝向

要求输入是 `*_with_gripper.npz`：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /path/to/hand_detections_0_with_gripper.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_replay_from_stored_pose \
  5
```

### 直接定义机器人朝前，无 remap

这条是“不要人手朝向，不做 remap，直接把夹爪朝向固定成机器人朝前”的最直接命令。

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz_urdfik.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_forward_no_remap \
  5 \
  --require_stored_gripper_pose 1 \
  --orientation_remap_label identity \
  --debug_force_orientation wrist_forward \
  --enable_viewer 1 \
  --viewer_wait_at_end 1
```

如果你想“仍然使用 NPZ 朝向，但不做 remap，只保留当前默认的前向修正”，用这条：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz_urdfik.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_npz_no_remap \
  5 \
  --require_stored_gripper_pose 1 \
  --orientation_remap_label identity \
  --stored_orientation_post_rot_xyz_deg 0 180 0 \
  --enable_viewer 1 \
  --viewer_wait_at_end 1
```

当前代码默认已经是：

- `--orientation_remap_label identity`
- `--stored_orientation_post_rot_xyz_deg 0 0 0`

也就是默认不做额外 remap，也不做额外后旋转。

### 用机器人当前 TCP 做朝向基准，再扫欧拉角

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
  --debug_mode 1 \
  --debug_frame_limit 1 \
  --enable_viewer 1 \
  --viewer_wait_at_end 1
```

## Orientation Remap Debug

这是新加的模式，专门用于排查：

- 人手 gripper 旋转矩阵的轴定义
- 机器人 TCP 的轴定义

是不是差了一个固定坐标系 remap。

### 固定一个常量 remap

当前默认基线是：

- `--orientation_remap_label identity`
- `--stored_orientation_post_rot_xyz_deg 0 0 0`

也就是不做额外 remap，也不做额外 post rotation。
当前这个参数会同时作用到左手和右手。

如果你已经从 sweep 里挑出了一个更合理的 label，可以直接固定：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_remap_fixed \
  5 \
  --orientation_remap_label x_from_yp_y_from_xm_z_from_zp \
  --debug_mode 1 \
  --debug_frame_limit 5 \
  --enable_viewer 1 \
  --viewer_wait_at_end 1
```

如果你现在想验证“左右手使用同一个定义，应该大致平行”，直接把 `label` 换成你这次挑出来的候选之一，比如：

```bash
--orientation_remap_label x_from_xm_y_from_zp_z_from_yp
```

或者：

```bash
--orientation_remap_label x_from_xp_y_from_zm_z_from_yp
```

### 自动 Sweep 24 个合法轴重排

这条会枚举所有右手系合法轴 remap：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_orientation_remap_sweep \
  5 \
  --orientation_remap_sweep_enable 1 \
  --orientation_remap_sweep_execute 1 \
  --debug_mode 1 \
  --debug_frame_limit 5 \
  --enable_viewer 1 \
  --viewer_wait_at_end 1
```

输出目录：

- `output_orientation_remap_sweep/orientation_remap_sweep/`

每个 remap 一个子目录，里面有：

- `frame_XXXX_Lsucc_Rfail_zed.png`
- `frame_XXXX_Lsucc_Rfail_third.png`
- `summary.json`

根目录也会有一个总的 `summary.json`。

## Right-Hand X Sweep

这个模式适合你现在这个目标：

- 先固定一个你认为定义正确的 `orientation remap`
- 左右手都共享这个 remap
- 只扫描右手的世界坐标 `x` 偏移

例如，把右手 `x` 从 `+0.50m` 扫到 `-0.30m`，每 `0.05m` 一档：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_right_x_sweep \
  5 \
  --orientation_remap_label x_from_xm_y_from_zp_z_from_yp \
  --right_target_world_offset_x_sweep_enable 1 \
  --right_target_world_offset_x_sweep_start +0.50 \
  --right_target_world_offset_x_sweep_end -0.30 \
  --right_target_world_offset_x_sweep_step 0.05 \
  --debug_mode 1 \
  --debug_frame_limit 5 \
  --enable_viewer 1 \
  --viewer_wait_at_end 1
```

输出目录：

- `output_right_x_sweep/right_target_world_offset_x_sweep/`

每个偏移一层子目录，例如：

- `right_x_m0p10`
- `right_x_m0p15`
- `right_x_m0p20`

里面会有：

- `frame_XXXX_Lsucc_Rfail_zed.png`
- `frame_XXXX_Lsucc_Rfail_third.png`
- `summary.json`

如果你已经知道左手还需要额外位置补偿，也可以继续叠加：

```bash
--left_target_world_offset_xyz DX DY DZ
```

或者右手固定再加额外整体补偿：

```bash
--right_target_world_offset_xyz DX DY DZ
```

如果你已经确认“右手需要先往 `+X` 推一点再看”，最直接的固定 debug 命令是：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_right_x_fixed \
  5 \
  --orientation_remap_label x_from_xm_y_from_zp_z_from_yp \
  --right_target_world_offset_xyz 0.20 0 0 \
  --debug_mode 1 \
  --debug_frame_limit 5 \
  --enable_viewer 1 \
  --viewer_wait_at_end 1
```

## Useful Flags

- `--viewer_wait_at_end 1`
  跑完后不自动关 viewer
- `--viewer_frame_delay 0.005`
  放慢可视化
- `--disable_table 1`
  不加载桌子
- `--target_world_offset_xyz DX DY DZ`
  整体平移目标位置
- `--target_world_z_offset Z`
  只改目标 z

## URDF IK Note

当前 `URDF IK` 脚本使用的是 `R1 pro` 这套本地 URDF：

- [urdfik.py](/home/zaijia001/ssd/RoboTwin/code_painting/urdfik.py)
- 默认路径：`/home/zaijia001/ssd/RoboTwin/galaxea_sim/assets/r1_pro/robot.urdf`

## Recommended Workflow

1. 先用 camera sweep 确认 `head / wrist` 相机朝向。
2. 用 `--debug_force_orientation current_tcp` 判断位置是不是已经大致对了。
3. 如果位置对了但人手朝向看起来明显不对，跑 `orientation_remap_sweep`。
4. 从 remap sweep 里挑一个成功率高、视觉也合理的 label。
5. 再把这个 label 固定到正常 replay、或者配合 `right_target_world_offset_x_sweep` 继续只调右手位置。



bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_right_x_sweep \
  5 \
  --orientation_remap_label x_from_xm_y_from_zp_z_from_yp \
  --right_target_world_offset_x_sweep_enable 1 \
  --right_target_world_offset_x_sweep_start +0.50 \
  --right_target_world_offset_x_sweep_end -0.30 \
  --right_target_world_offset_x_sweep_step 0.05 \
  --debug_mode 1 \
  --debug_frame_limit 3 \
  --enable_viewer 1 \
  --viewer_wait_at_end 1


bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz_urdfik.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_urdfik_forward \
  5 \
  --orientation_remap_label x_from_xm_y_from_zp_z_from_yp \
  --right_target_world_offset_xyz 0.20 0 0 \
  --debug_force_orientation wrist_forward \
  --debug_mode 1 \
  --debug_frame_limit 100 \
  --enable_viewer 1 \
  --viewer_wait_at_end 1

# 直接npz位资
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz_urdfik.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_from_npz_gripper \
  5 \
  --require_stored_gripper_pose 1 \
  --pose_source gripper \
  --orientation_remap_label identity \
  --stored_orientation_post_rot_xyz_deg 0 0 0 \
  --debug_force_orientation none \
  --enable_viewer 1 \
  --right_target_world_offset_xyz 0.0 0 0.2 \
  --viewer_wait_at_end 1

# forward
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz_urdfik.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_forward_no_remap \
  5 \
  --require_stored_gripper_pose 1 \
  --orientation_remap_label identity \
  --debug_force_orientation wrist_forward \
  --right_target_world_offset_xyz 0.0 0 0.2 \
  --enable_viewer 1 \
  --viewer_wait_at_end 1

# 交换detect得到结果的红色和蓝色轴
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_hand_retarget_r1_npz_urdfik.sh \
  /home/zaijia001/ssd/data/R1/hand_vis/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue \
  5 \
  --require_stored_gripper_pose 1 \
  --pose_source gripper \
  --orientation_remap_label swap_red_blue \
  --stored_orientation_post_rot_xyz_deg 0 0 0 \
  --right_target_world_offset_xyz 0.0 0 0.2 \
  --debug_force_orientation none \
  --enable_viewer 1 \
  --viewer_wait_at_end 1


