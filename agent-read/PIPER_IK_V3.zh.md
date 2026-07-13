# Piper TCP/EE IK V3

## 目的与隔离范围

Piper IK V3 修复 TCP/EE pose 到 `link6` URDFIK target 的不可逆转换，同时不修改现有 OursV2 数据、renderer、planner 或 runner。V3 使用新文件和独立输出目录。它与旧 `demo_piper_ik_seq_v3` MotionGen 版本不是同一条版本线。

## 根因 1：关节角 FK 为什么和 Real endPose 有偏差

关节角计算出的原始 `link6` 没有 12 cm 级错误。去掉双方 TCP 后，12 个可视化 episode 的平均原点误差是左 `6.13 mm`、右 `6.09 mm`。

大偏差来自两套不同的 TCP：

```text
真机 Piper:
T_base_real_tcp = T_base_link6 * Ry(-pi/2) * Tx(0.19)

当前 Ours:
R_ours = R_link6 * diag(1,-1,-1)
p_ours_tcp = p_link6 + R_ours * [0.12,0,0]
```

真机公式来自：

`/home/piper/pika_ros/src/PikaAnyArm/piper/pika_remote_piper/scripts/forward_inverse_kinematics.py`

其中 `first_matrix=Ry(-1.57)`、`second_matrix=Tx(gripper_xyzrpy[0])`，配置 `gripper_xyzrpy[0]=0.19`。FK 乘这个工具变换，IK 使用同一个 end frame，因此真机转换成对可逆。记录数据拟合的有效长度约为左 `0.18424 m`、右 `0.18095 m`；与配置 0.19 m 的几毫米差异属于 URDF、同步、标定和实际工具长度残差。

## 坐标轴颜色和物理含义

| 颜色 | 局部轴 |
|---|---|
| 红 | `+X` |
| 绿 | `+Y` |
| 蓝 | `+Z` |

颜色只表示当前 frame 的轴名，不保证物理语义相同：

- Real TCP 红色 `+X` 是物理纵向/前进轴。
- 该方向约等于 Ours `-Z`，也就是蓝色 `+Z` 箭头的反方向。
- Ours 历史 12 cm 加在红色 `+X`，它不是 Pika 的物理前进轴。
- 绿色 `+Y` 最接近两指之间的开合轴。
- 开合轴和前进/approach 轴是不同的轴。

全量点积：

```text
Real +X dot Ours +X  ~= 0.001
Real +X dot Ours -Z  ~= 0.99999
```

## 根因 2：V5 的 EE-pose 为什么出错

OursV2 LeRobot 字段名是 `left_ee_*` / `right_ee_*`，但构造脚本实际读取：

```text
current_left_tcp_pose_world_wxyz
current_right_tcp_pose_world_wxyz
```

因此这些字段已经包含 Ours TCP 的 12 cm。旧 target 转换为：

```text
position += R_target * [0.12 - gripper_bias, 0, 0]
R_ee = R_target * inv(delta_matrix)
```

当前 `gripper_bias=0.12`，translation 恰好为零；旧路径也没有撤销 `global_trans_matrix=diag(1,-1,-1)`。结果是带 12 cm 和 Ours orientation remap 的 TCP pose 被直接当成 `link6` target。

V5 的 2118 帧统计：规划结果相对 direct-q link6 平均偏离左 `12.76 cm`、右 `12.65 cm`；离线 IK 成功率左 `89.2%`、右 `90.3%`。

旧 human-replay runner 还把 rotation threshold 放宽到 `pi`，会把方向错 90–180 度的结果判成 success。TCP translation 依赖方向，所以方向错时 12 cm 位置也无法闭合。V3 独立 runner 最大 rotation threshold 改为 `0.12 rad`，reach tolerance 为 `10 deg`；OursV2 原 runner 保持不变。

## V3 转换

默认 `PIPER_IK_V3_TARGET_SEMANTICS=ours_tcp`，因为 current OursV2 的 EE-labelled 字段实际保存 TCP：

```text
p_link6 = p_ours_tcp - R_ours_tcp * [gripper_bias,0,0]
R_link6 = R_ours_tcp * inv(delta_matrix) * inv(global_trans_matrix)
```

还支持：

- `ours_ee`：输入是真正的 `current_*_ee_pose`；不减 12 cm，只撤销 orientation remap。
- `real_piper_tcp`：输入是真机 `endPose` frame；撤销 `Ry(-pi/2) * Tx(tool_length)`。

核心文件：

- `code_painting/piper_ik_v3_transforms.py`
- `code_painting/render_hand_retarget_piper_dual_npz_urdfik_v3.py`
- `code_painting/plan_anygrasp_keyframes_piper_v3.py`
- `code_painting/plan_keyframes_human_replay_v3.py`
- `code_painting/run_plan_keyframes_human_replay_piper_d435_v3.sh`

现有 OursV2 文件没有修改；V3 默认输出到 `human_replay_v3/`。

## 验证结果和边界

- 300 个随机 pose 的 Ours TCP、Ours EE、Real Piper TCP round-trip 全部通过。
- 12 个 V5 episode 共 4236 个 arm pose：最大 position error `9.7e-17 m`，最大 rotation error `7.7e-16 rad`。
- direct-q seeded URDFIK：左右臂均 `2118/2118` 成功；mean position error 左 `1.30e-7 m`、右 `2.51e-7 m`。
- Python compile、shell syntax、两个 V3 `--help` 入口均通过。

`stack_cups id6` human-replay smoke 成功实例化 V3，但没有达到人手目标，因为 HaMeR/人手目标朝向与 link6/TCP 语义仍差约 100 度以上。这是独立的 hand-frame orientation-remap 问题。V3 只修复已知 OursV2 TCP/EE 的可逆转换，并会把这种方向错误判为失败。
