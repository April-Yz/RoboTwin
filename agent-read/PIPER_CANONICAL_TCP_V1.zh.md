# PiperCanonicalTCP-v1

## 目标与隔离边界

`PiperCanonicalTCP-v1` 是独立的 Real-Piper-TCP 规划与对比链，不修改 OursV2、Piper IK V3 或旧结果。代码位于 `code_painting/piper_canonical_tcp_v1/`。planner target、current readback、reach check 和可视化 target 都使用 `T_W_RTCP`。

## Frame contract

`T_A_B` 表示 frame B 在 frame A 中的 pose，`R_A_B` 把 B-frame 局部向量映射到 A-frame。

| 名称 | 语义 |
|---|---|
| `W` | 0515 标定共享 world |
| `B_L/B_R` | 左/右臂 base |
| `CGRASP` | robot-frame preview 保存的 canonical grasp 局部轴 |
| `RTCP` | Piper 服务器 Real TCP；局部 `+X` 是前进/approach |
| `L6_SIM` | SAPIEN raw `link6` actor frame |
| `L6_URDF` | CuRobo URDF IK/FK `link6`，也是服务器工具公式的 link6 |

颜色固定为红 `+X`、绿 `+Y`、蓝 `+Z`。文字必须区分 `world_+X` 与 `local_RTCP_+X`，不能只写 X/Y/Z。

### SAPIEN 与 URDF link6

同一 q 的运行时 FK 显示两者原点一致，但局部轴固定差 90°：

```text
R_L6SIM_L6URDF =
[[ 0, 0, 1],
 [ 0, 1, 0],
 [-1, 0, 0]]
= Ry(+pi/2)  # 精确 signed-axis permutation
```

修正前同-q rotation error 是 90.000°；适配后左右约 `0.000016°`、`0.000006°`，position error 小于 `7.5e-8 m`。

### Piper 服务器工具

服务器字面量保持不变，不能把 `-1.57` 换成 `-pi/2`：

```text
T_L6URDF_RTCP = Ry(-1.57) @ Tx(0.19)
```

完整链为：

```text
T_W_RTCP = T_W_B @ T_B_L6SIM
                    @ T_L6SIM_L6URDF
                    @ T_L6URDF_RTCP

T_B_L6URDF = inv(T_W_B) @ T_W_RTCP @ inv(T_L6URDF_RTCP)
```

### Preview grasp 轴

robot-frame preview 保存：

```text
R_W_CGRASP = R_W_RTCP @ R_RTCP_CGRASP
R_RTCP_CGRASP = [[0,0,1],[0,1,0],[-1,0,0]]
```

Orientation/Fused 从 preview 读取时右乘 `R_CGRASP_RTCP = R_RTCP_CGRASP.T`，runner 标签为 `swap_red_blue_keep_green`。Top-score 在同一 keyframe 重新读取 raw AnyGrasp rotation，因此使用 `identity`。

`R_RTCP_CGRASP` 与 `R_L6SIM_L6URDF` 数值恰好相同，但前者是 candidate-source local-axis conversion，后者是 simulator/URDF model-frame adapter；两者不可合并或省略。

## 对比定义

### same-q joint control

同一 q、同一 raw `L6_SIM` 下比较 OursV2 历史 0.12 m TCP 与服务器 0.19 m Real TCP。修正后的 `pnp_bread/id8` 左右臂 mean/max distance 均约 `0.0700001 m`，即统一前进轴上的 7 cm 差。旧 smoke 的 `0.224641 m` 遗漏 `L6_SIM -> L6_URDF`，不是有效物理结论。

### EE-pose 三策略

- Orientation：preview orientation rank 1；
- Fused：`0.25 * AnyGrasp raw score + 0.75 * orientation score` 的 rank 1；
- Top-score：同 keyframe raw AnyGrasp score 最大候选。

Top-score 没有 orientation 约束，可能选中绕 approach 轴约 180° 翻转的候选。严格 IK miss 是策略结果，应保留视频和 failures TSV，不应强制翻转或放宽阈值。

## Smoke 验证

完整通过 episode：`code_painting/piper_canonical_tcp_v1/smoke_all_pass/pnp_bread/foundation_input_8/`。

- arm 为 left；
- Orientation、Fused、Top-score 的 pregrasp/grasp/action 均 reached；
- 三种 head video、三种 `SUCCESS`、`strategy_comparison.mp4` 齐全；
- corrected same-q joint video 与 `SUCCESS` 齐全；
- 三策略视频为 H.264 640×360，合成图 1440×490，joint 对比 1280×794。

另保留 `smoke_pass/stack_cups/id0`：Orientation/Fused 成功，Top-score 选中 178.79° 翻转候选并严格失败，是策略差异样例。

## 批处理

`batch_manifest.tsv` 固定 6 tasks × 5 episodes = 30 episodes。每集依次运行 Orientation、Fused、Top-score；joint-control comparison 单独运行。2026-07-15 批量输出写到全新 `outputs_canonical_20260715/`，不覆盖默认 `outputs/` 中的旧 dry-run 文件。

tmux 名称：`pcan_v1_joint_6x5`、`pcan_v1_eepose_6x5`。

EE batch 在单策略 IK miss 后继续，记录到 `outputs_canonical_20260715/_batch_logs/eepose_failures.tsv`。只要三种 head video 存在就合成 `strategy_comparison.mp4`，但不会伪造失败策略的 `SUCCESS`。

## 验证

- 6 个数学单元测试通过：服务器字面量/乘法顺序、URDF-link6 round trip、SIM/URDF adapter、preview axis inverse、world/base inverse。
- Python `py_compile` 与 shell `bash -n` 通过。
- dry-run 验证 Orientation/Fused remap 为 `swap_red_blue_keep_green`，Top-score 为 `identity`。
- single-episode 三策略、joint control、ffprobe 与抽帧视觉 QA 通过。

## OursV2 / Canonical / Piper real 三链路控制对比

`outputs_canonical_20260715/eepose/<strategy>` 的“三种”仅指 Orientation、Fused、Top-score 候选选择，全部使用 Canonical IK；它不包含 OursV2 IK，也不是三控制器效果对比。

新入口 `run_real_control_compare.sh` 使用 Piper raw episode 的同步 D435、`jointState` 与 `endPose`：

- Joint control：共同输入为 real q1-q6。Real 曲线读取服务器 endPose；OursV2 曲线使用 `R_B_OTCP=R_B_L6URDF@diag(1,-1,-1)` 并沿局部 +X 加 0.12 m；Canonical 使用 `T_B_RTCP=T_B_L6URDF@Ry(-1.57)@Tx(0.19)`。
- EE-pose control：共同输入为 real `T_B_RTCP`。OursV2 分支忠实复现旧默认，把该数值不经 server inverse-tool 直接当 `T_B_L6URDF` IK 目标，最大旋转阈值 3.14 rad；Canonical 先执行 `T_B_L6URDF=T_B_RTCP@inv(Ry(-1.57)@Tx(0.19))`，最大旋转阈值 0.12 rad。
- 两个 IK 结果都重新按 Canonical 物理 RTCP FK 评价；失败臂只在仿真面板显示红色 direct-q fallback，并从曲线/误差统计排除。
- 图表全部是共享 Piper0515 world XYZ。画面里的红/绿/蓝坐标轴是各 TCP 的局部 +X/+Y/+Z，不是 world 曲线颜色。

8 帧 smoke 位于 `outputs_real_control_smoke_20260716/handover_bottle/episode0/`。Joint Canonical 对 real 左右平均位置误差为 `9.7748/9.5863 mm`，OursV2 为 `215.84/218.77 mm`。EE-pose 两分支均 100% IK 成功；Canonical 物理 RTCP 误差为 `0.0111/0.0039 mm`，OursV2 为 `195.08/195.41 mm`。两支 1920×1080 MP4 均为 H.264/yuv420p/faststart，并完成视觉 QA。

`real_control_manifest.tsv` 固定 30 个已审计 raw episode（6 tasks × 5）。它与 `batch_manifest.tsv` 的 30 个 AnyGrasp foundation IDs 数量相同，但样本语义不同：前者是 Piper raw episode，后者是 foundation input ID，不能按 slot 静默配对。
