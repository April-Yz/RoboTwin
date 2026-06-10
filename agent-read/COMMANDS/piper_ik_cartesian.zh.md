# Piper IK Cartesian 抓放

## 用途

为 `pick_diverse_bottles_piper_ik` 提供 V1-V4 的真实运动 viewer、分阶段轨迹采集和回放命令。

## 适用版本

- 轨迹 schema：`piper_ik_cartesian`
- 轨迹版本：2
- 配置：`demo_piper_ik_seq_v1` 至 `demo_piper_ik_seq_v4`
- 推荐默认：V1；V3 在 MotionGen 失败时回退到同一 IK 终点的三次插值。

## 前置条件

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw
cd /home/zaijia001/ssd/RoboTwin
```

## Viewer

```bash
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 50 --require_success 1
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v2 --seed 0 --max_seed_tries 50 --require_success 1
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v3 --seed 0 --max_seed_tries 50 --require_success 1
python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v4 --seed 0 --max_seed_tries 50 --require_success 1
```

无窗口验证：

```bash
unset DISPLAY
SAPIEN_RT_DENOISER=none timeout 180s python view_pick_diverse_bottles_piper_ik_motion.py --ik_version v1 --seed 0 --max_seed_tries 50 --hold 0 --render_freq 0 --show_axes 0 --require_success 1
```

## 数据采集

```bash
bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v1 0
bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v2 0
bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v3 0
bash collect_data.sh pick_diverse_bottles_piper_ik demo_piper_ik_seq_v4 0
```

## 参数

- `ik_version`：V1 线性、V2 三次、V3 MotionGen 加回退、V4 多种子三次。
- `lift_height`：从 grasp 目标仅沿世界 z 增加的高度，当前为 0.12 m。
- `move_settle_steps`：每段运动结束后保持最终 IK 指令的步数，当前为 80。
- `gripper_settle_steps`：开合夹爪的稳定步数，当前为 120。
- `require_success=1`：viewer 只有在真实 `check_success()` 通过时才接受 seed。

## 输入与输出

- prompt：`description/task_instruction/pick_diverse_bottles_piper_ik.json`
- 轨迹：`data/pick_diverse_bottles_piper_ik/<config>/_traj_data/episodeN.pkl`
- 数据：`data/.../<config>/data/episodeN_succ.hdf5`
- 视频：每个 RGB camera 一条 MP4，包括右侧 `third_camera` 和对向俯视 `opposite_top_camera`。

## 执行逻辑

动作固定为 `pregrasp -> grasp -> close -> lift -> place -> open`。四段 move 逐段规划和执行；下一段从上一段轨迹末端关节状态开始。lift 保留 grasp 的 x/y 和姿态，只增加 z。close 后根据瓶子功能点与实际末端的 x/y 偏移修正 place 夹爪目标。

viewer 直接规划并执行 Phase 1；采集 Phase 1 保存同一轨迹，Phase 2 在相同 seed 场景中校验并逐点回放。因此动作目标和关节轨迹一致，但采集额外保存观测。旧 pickle 缺少 schema、版本、动作名或有效轨迹时会被拒绝。

## 相关代码

- `envs/pick_diverse_bottles_piper_ik.py`
- `envs/robot/piper_ik.py`
- `view_pick_diverse_bottles_piper_ik_motion.py`
- `script/collect_data.py`
- `assets/embodiments/piper_pika_agx/config.yml`

## 常见问题

- 简单 lift/move 失败的旧根因是所有段都从 home 规划、lift x/y 取错参考、末端 PD 未收敛，以及把瓶子目标误当夹爪目标。
- 无图形环境不要设置无效的 `DISPLAY=:1.0`；使用 `unset DISPLAY --render_freq 0`。
- 不要让新版配置读取旧 `demo_piper_ik_v*` 目录；新版命令使用 `demo_piper_ik_seq_v*`。
