# Dense Replay URDF-match v2

该版本是旧 Dense Replay 的隔离修正版，不修改旧 renderer、runner 或输出。它仍把 HaMeR 的逐帧人手运动做 dense retargeting，但修复规划模型、局部坐标系和执行收敛之间的对应错误。

## 根因

1. 论文素材中的旧视频生成于 2026-05-23；当前 Piper IK URDF 的修复提交在 2026-06-18，因此旧视频仍使用旧模型。
2. 对同一组 `joint1..joint6`，Curobo 与 SAPIEN 的 `link6` 原点一致，但局部轴相差固定旋转：

   ```text
   R_sapien_link6 = R_curobo_link6 @ Ry(-90 deg)
   ```

   旧路径没有应用该 adapter。由于 TCP 位于 link6 局部前方 0.12 m，局部轴差会产生 `0.12 * sqrt(2) = 0.1697 m` 的固定位置偏差。关节顺序本身没有错。
3. 旧构造链把命令行请求的 `urdfik_joint_interp_waypoints=10` 覆盖回默认值 2。
4. 旧执行只等待固定步数，不检查实际关节是否到达目标。
5. HaMeR `gripper_position` 是拇指和食指指尖中点，应按 TCP 解释；旧路径混用了 link6/EE/TCP 语义。

## v2 修改

- IK 和 SAPIEN 都强制使用 `piper_pika_agx`，并断言双臂关节顺序为 `joint1..joint6`。
- 在 TCP↔link6 转换中显式加入 Curobo→SAPIEN 的固定 `Ry(-90 deg)` adapter。
- 严格反演机器人报告 TCP 所使用的 0.12 m bias、`global_trans_matrix` 和 `delta_matrix`。
- 在继承构造完成后恢复请求的 10 个关节插值 waypoint。
- 采用 Ours v2 类似的 joint-continuity、多 seed 和放宽不可达人手姿态的 IK 策略。
- 最多额外等待 240 个仿真步；所有目标关节误差小于 0.01 rad 后提前退出。
- 输出 `dense_replay_v2_metadata.json` 和逐臂逐帧 `execution_audit.jsonl`。

## 命令模板（不可直接运行）

```bash
TASK=<task_name> ID=<episode_id> GPU=<gpu_id> MAX_FRAMES=<n_or_-1> \
OUT_ROOT=<separate_output_root> \
bash code_painting/run_dense_replay_urdfmatch_v2.sh
```

常改参数：`TASK`、`ID`、`GPU`、`MAX_FRAMES`、`OUT_ROOT`。不要把 `OUT_ROOT` 指到旧的 `h2_pure_d435`，以免混淆版本。

## 已验证的完整命令

```bash
cd /home/zaijia001/ssd/RoboTwin
TASK=pick_diverse_bottles ID=0 GPU=3 MAX_FRAMES=-1 \
bash code_painting/run_dense_replay_urdfmatch_v2.sh
```

默认输出：

```text
code_painting/human_replay/h2_pure_d435_urdfmatch_v2/
└── pick_diverse_bottles/id0_d435_z005/
    ├── zed_replay.mp4
    ├── third_replay.mp4
    ├── left_wrist_replay.mp4
    ├── right_wrist_replay.mp4
    ├── dense_replay_v2_metadata.json
    ├── execution_audit.jsonl
    └── dense_replay_urdfmatch_v2_validation.json
```

## 验证结果与限制

`pick_diverse_bottles/id0` 的前 8 帧、双臂共 16 条记录：IK `16/16` 成功；规划 TCP 位置误差平均 5.48 mm；执行 TCP 位置误差平均 4.72 mm、最大 14.50 mm；关节误差全部小于 0.01 rad；Curobo FK 与 SAPIEN TCP 的平均差约 `2.65e-7 m`。

完整 106 帧中，左/右臂分别有 `85/106` 和 `83/106` 个成功目标。168 个成功 arm plan 的平均规划/执行位置误差为 `4.44/4.70 mm`，FK/仿真 TCP 平均差为 `3.16e-7 m`。失败主要集中在缺失、无效或远离工作空间的人手目标；它们被如实保留为 Dense baseline 失败，不做伪造补帧。

人手姿态对 Piper 经常不可达，抽样姿态误差仍约 38°。这属于 Dense Replay 的 action-level cross-embodiment gap；Ours v2 通过 robot-native grasp candidates 和 human-guided grasp selection 避免直接复制该稠密姿态。v2 修复固定坐标/执行误差，但不把 Dense baseline 改造成 Ours v2。
