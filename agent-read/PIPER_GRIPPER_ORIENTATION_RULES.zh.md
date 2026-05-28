# Piper 夹爪朝向规则说明

## 当前结论

在当前 `pnp_star_pear_hamer_output_v2/hand_detections_0.npz` 和 `PiperPika` 标定场景下，图里的 RGB 轴含义可以按下面理解：

- 红色 `+X`：由手指骨架平面叉乘得到的侧向/法向轴，不是主要前进轴。
- 绿色 `+Y`：拇指指尖到食指指尖方向，最接近夹爪开合轴。
- 蓝色 `+Z`：由 `X cross Y` 得到，当前观察上最接近夹爪前进/接近方向。

也就是说，你看到“蓝色轴是前进轴、绿色轴是开合、红色是另一个侧向轴”是符合当前代码定义和可视化结果的。

## 为什么之前默认只 debug 左手

最近新增的几个 debug wrapper 默认用了：

- `ARM=left`
- 或 `ARMS=left`

原因是先减少变量：单臂 IK 更快，失败来源更少，也方便先判断局部轴语义。不是代码只能跑左手。

如果要右手：

```bash
ARM=right ...
```

或：

```bash
ARMS=right ...
```

如果要左右一起：

```bash
ARMS=both ...
```

注意：左右一起跑时，某些候选只要一边不可达，就会让观察更乱。当前建议先左手确认轴，再右手单独验证。

## HaMeR / NPZ 里的 gripper 局部轴

入口逻辑在：

- `code_painting/render_hand_retarget_r1_npz.py`
- `calc_gripper_pose_from_keypoints(...)`

如果 NPZ 里有 `left_gripper_rotation_matrix` / `right_gripper_rotation_matrix`，脚本直接读。否则会用同一套公式从 keypoints 重算。

重算公式：

```text
gripper_position = 0.5 * (thumb_tip + index_tip)

+Y = normalize(thumb_tip - index_tip)
temp = index_joint - index_tip
+X = normalize(cross(temp, +Y))
+Z = normalize(cross(+X, +Y))

rotation_matrix = [ +X  +Y  +Z ]  # 三列分别是局部 x/y/z 轴
retreat_position = gripper_position - retreat_distance * +Z
```

含义：

- `+Y` 是两指之间的方向，所以最像开合轴。
- `+Z` 用来计算 retreat：`retreat = center - d * Z`，所以 `+Z` 从 retreat/wrist 侧指向夹爪中心/接近方向。
- `+X` 是为了补齐右手系的第三根轴，更像手掌/手指平面的法向或侧向。

## 朝向修正顺序

每帧的目标朝向从 `rotation_cam` 开始。

代码路径：

- `remap_target_rotation(...)`

公式：

```text
R_cam_fixed = R_cam * R_post * R_remap
```

其中：

- `R_cam`：HaMeR/NPZ 的 gripper rotation。
- `R_post`：命令行 `--stored_orientation_post_rot_xyz_deg RX RY RZ`。
- `R_remap`：命令行 `--orientation_remap_label` 对应的固定轴重排。

重要点：

- 这里是右乘，所以 `R_post` / `R_remap` 作用在 gripper 局部坐标系上。
- 如果你想把“蓝色 `+Z` 前进轴”映射成机器人期望的某根轴，本质上就是在这里找一个稳定的 `R_post` 或 `R_remap`。

## Head camera 到 world 的转换

代码路径：

- `camera_to_world_pose(...)`

当前命令使用：

```text
--camera_cv_axis_mode legacy_r1
--head_camera_local_pos 0.107882 -0.2693875 0.464396
--head_camera_local_quat_wxyz 0.85401166 0.01255256 0.51885652 -0.0359783
```

公式：

```text
pos_world = p_head_world + R_head_world * C_legacy * pos_cam
R_world = R_head_world * C_legacy * R_cam_fixed
```

`legacy_r1` 矩阵是：

```text
C_legacy =
[[ 0,  0,  1],
 [-1,  0,  0],
 [ 0, -1,  0]]
```

Piper 双臂场景里，head camera 是固定在 left base 上的标定位姿：

```text
world_T_head = world_T_left_base * left_base_T_head_camera
```

## Piper base 和双臂位置

当前配置：

- `robot_config_PiperPika_agx_dual_table.json`
- `dual_arm_embodied=false`
- `embodiment_dis=0.60`

因此左右臂是两个独立 Piper 实例：

```text
left_base  ~= [-0.3, -0.25, 0.75]
right_base ~= [ 0.3, -0.25, 0.75]
base_quat  = [0.70710678, 0, 0, 0.70710678]
```

`PiperDualReplayRenderer` 会分别用 left/right base 把 world target 转回各自 base：

```text
target_base = inv(world_T_arm_base) * target_world
```

## Gripper target 到 Piper link6 / URDFIK 的转换

关键路径：

- `render_hand_retarget_piper_dual_npz_urdfik.py`
- `_target_tcp_world_to_ee_base(...)`
- `envs/robot/robot.py`
- `_trans_from_gripper_to_endlink(...)`

当前 Piper URDFIK 求解的是：

```text
base_link -> link6
```

目标进入 IK 前先执行：

```text
target_pose_base = world_pose_to_base_pose_for_arm(target_world, arm)
target_pose_ee = robot._trans_from_gripper_to_endlink(target_pose_base, arm)
```

当前配置中的关键参数：

```text
gripper_bias = 0.12
delta_matrix = I
global_trans_matrix = diag(1, -1, -1)
```

`_trans_from_gripper_to_endlink(...)` 里：

```text
position += R_gripper * [0.12 - gripper_bias, 0, 0]
R_ee = R_gripper * inv(delta_matrix)
```

因为 `gripper_bias=0.12` 且 `delta_matrix=I`：

```text
position offset = 0
R_ee = R_gripper
```

所以在当前 PiperPika 配置下，送进 URDFIK 的 `link6` 目标基本就是 retarget 产生的目标位姿本身。

## 一个容易混淆的不对称点

`get_left_tcp_pose()` / `get_right_tcp_pose()` 读回 TCP 时，会走 `_trans_endpose(..., is_endpose=True)`：

```text
R_tcp_readback = R_link6 * global_trans_matrix * delta_matrix
```

当前：

```text
global_trans_matrix = diag(1, -1, -1)
delta_matrix = I
```

所以读回的 TCP 朝向和 `link6` 朝向之间存在 `Y/Z` 翻转。

但 `_trans_from_gripper_to_endlink(...)` 当前把目标送进 IK 时没有再乘 `global_trans_matrix` 的逆，只乘了 `inv(delta_matrix)`。

实际影响：

- target axis actor 显示的是 retarget 目标轴。
- URDFIK 接收的是接近同一套目标轴。
- `get_*_tcp_pose()` 的 post-execute debug 读回轴会额外带 `diag(1,-1,-1)`。

因此，看“目标轴颜色语义”时，应优先看 target axis 和 board 图；看“执行误差”时，要意识到读回 TCP frame 可能比目标 frame 多一层 `global_trans_matrix`。

## 对当前调试的建议

当前观察结果说明：

- HaMeR/重算 gripper 的蓝色 `+Z` 更像前进轴。
- 绿色 `+Y` 更像开合轴。
- 机器人/IK 侧不是天然把 `+Z` 当作前进轴；现有历史命令里曾使用 `swap_red_blue_keep_green`，本质上就是把红蓝轴关系调到更接近机器人习惯。

建议调试顺序：

1. 先用 `run_piper_retarget_postrot_board_video.sh` 的 `CASE_MODE=standard` 看完整 retarget 回放里哪个后旋转视觉最合理。
2. 再用 `CASE_MODE=axis90` 扩大到单轴 90/180 度旋转。
3. 最后只对视觉合理的 1-2 个候选跑更长帧数或左右手。
4. 如果候选视觉合理但 success 少，优先调 `TARGET_DY/TARGET_DZ` 和 IK 初始姿态，而不是继续盲扫朝向。

## 常用命令

左手，标准 8 候选：

```bash
GPU=3 FPS=5 FRAME_START=0 MAX_FRAMES=32 ARMS=left CASE_MODE=standard \
TARGET_DY=0.1 TARGET_DZ=0.1 ORIENTATION_REMAP_LABEL=identity \
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_retarget_postrot_board_video.sh \
  /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_retarget_postrot_board_standard
```

右手，标准 8 候选：

```bash
GPU=3 FPS=5 FRAME_START=0 MAX_FRAMES=32 ARMS=right CASE_MODE=standard \
TARGET_DY=0.1 TARGET_DZ=0.1 ORIENTATION_REMAP_LABEL=identity \
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_piper_retarget_postrot_board_video.sh \
  /home/zaijia001/ssd/data/piper/hand/pnp_star_pear_hamer_output_v2/hand_detections_0.npz \
  /home/zaijia001/ssd/RoboTwin/code_painting/output_piper_retarget_postrot_board_standard_right
```

输出：

```text
board/board_zed.mp4
board/board_third.mp4
board/board_zed_frame0000.png
board/board_third_frame0000.png
index.csv
```
