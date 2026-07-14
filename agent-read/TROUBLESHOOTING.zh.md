# 故障排查

## Dense Replay 出现约 17 cm 固定偏差

- 症状：规划轨迹趋势相似，但整条实际 TCP 曲线固定平移；方向还可能相差约 90°。
- 诊断：对同一关节角比较 Curobo link6 FK、SAPIEN link6 和 SAPIEN TCP。
- 原因：Curobo/SAPIEN link6 局部轴固定相差 `Ry(-90 deg)`，旧路径又在该轴上应用 0.12 m TCP offset。
- 修复：使用隔离的 `run_dense_replay_urdfmatch_v2.sh`；不要改关节顺序。

## 首帧执行位置明显落后

- 症状：规划 FK 接近目标，但仿真实际 TCP 仍相差数厘米。
- 诊断：查看 `execution_audit.jsonl` 的 `joint_metrics_after_execute`。
- 原因：固定仿真步数不足以让驱动关节收敛。
- 修复：v2 最多等待 240 步，并在最大关节误差小于 0.01 rad 后提前停止。

## 位置已对齐但姿态仍差几十度

- 原因：Dense 直接复制人手姿态，而该姿态对 Piper 可能不可达。
- 处理：这是 baseline 限制；需要机器人原生抓取姿态时使用 Ours v2，不应把 rotation threshold 再强行收紧到导致整帧 IK 失败。

## 非交互 SSH 报 `conda: command not found`

- 原因：远端 shell 没有加载 Conda 初始化脚本，不代表环境不存在。
- 处理：先 `source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh`，或直接调用 `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10`。

## Audit V4 拒绝已有输出目录

- 症状：`Refusing to overwrite non-empty output root`。
- 原因：V4 默认保护已有 PNG、metadata 和报告。
- 处理：使用新的 `--output-root`。只有明确要重建 V4 自身产物时才使用 `--overwrite`；该选项也不会授权修改旧策略目录。
