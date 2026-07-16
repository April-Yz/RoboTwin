# Real-control 对比输出速查

适用目录：`code_painting/piper_canonical_tcp_v1/outputs_real_control_compare_20260716/<task>/<episode>/`

## 文件含义

| 文件 | 含义 |
|---|---|
| `joint_control.mp4` | 给三套定义输入同一组 Piper 实测 `q1..q6`，比较各自 TCP 位置。 |
| `eepose_control.mp4` | 给 OursV2 与 Canonical 输入同一个实测 `T_B_RTCP`，比较两套 IK 到达的物理 RTCP。 |
| `*.manifest.json` | 对应视频的编码、帧数、输入和坐标语义。 |
| `control_plan.npz` | 同步 q/endPose、两套 FK/IK 结果及 IK 成功掩码；用于复画曲线。 |
| `summary.json` | 帧数、每侧 IK 成功率和位置误差摘要。 |
| `frame_contract.json` | `world/base/link6/RTCP` 定义和局部轴说明。 |
| `oursv2_renderer.npz` / `canonical_renderer.npz` | 两条仿真支路的 renderer 与 qpos 来源。 |
| `sim_direct/`、`sim_oursv2/`、`sim_canonical/` | 上方仿真画面的逐帧缓存，不是新的真机数据。 |
| `SUCCESS` / `EXIT_CODE` | 完整成功标记 / 失败返回码。 |

## 上方四个画面

`joint_control.mp4`：真机 D435 + 三套 TCP 坐标轴、真机左腕、真机右腕、同一组实测 q 在 0515 标定 RoboTwin 中的直接关节控制。

`eepose_control.mp4`：真机 D435 + Real-TCP 目标、Piper 实测 q 的仿真参考、OursV2 旧 EE-pose IK、Canonical 服务器语义 IK。第二格的坐标轴是目标轴，不是额外一条 q-FK 曲线。

## 下方曲线

每列是一只手，每行依次是 0515 `world X/Y/Z`；横轴是同步 D435 帧号，红色竖线是当前视频帧。

| 曲线 | 颜色 | 含义 |
|---|---|---|
| Piper real | 黑色 | 真机记录的 endPose / Real RTCP。 |
| OursV2 | 青色 | Joint 中为 12 cm OursV2 TCP；EE-pose 中为旧 IK 到达后换算的物理 RTCP。IK 失败帧不画线。 |
| Canonical | 紫色 | Joint 中为服务器 `Ry(-1.57) @ Tx(0.19)` RTCP；EE-pose 中为 Canonical IK 到达的物理 RTCP。 |

只看到黑线和青线时，通常是紫线与黑线几乎完全重合并覆盖黑线，并非数据缺失；以 `summary.json` 和 manifest 为准。

局部坐标轴颜色固定为 `+X` 红、`+Y` 绿、`+Z` 蓝。曲线坐标始终是 0515 world XYZ，两者不要混用。

## 最快查验顺序

1. 看 `SUCCESS` 或 `EXIT_CODE`。
2. 看 `summary.json` 的 IK 成功率和 mean/max position error。
3. 看红色竖线附近是否有曲线断点；断点代表该支路该帧 IK 失败。
4. 精确复核读取 `control_plan.npz`，不要从画面像素估计毫米误差。
