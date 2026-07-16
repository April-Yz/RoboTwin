# 故障排查

## PiperCanonicalTCP-v1：MP4 能被 ffmpeg 读取，但 VSCode 无法播放

- 症状：OpenCV/ffmpeg 能完整读取视频，VSCode 预览却黑屏、报不支持或无法开始播放。
- 诊断：运行 `ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt <VIDEO>`；`mpeg4`/`mp4v` 容器本身正常，但 Chromium 通常没有该解码器。
- 原因：旧 planner 只把 head/third 转成 H.264；wrist、debug、joint comparison 和 strategy comparison 仍由 OpenCV 直接写 `mp4v`。
- 修复：运行 `COMMANDS/piper_canonical_tcp_v1.zh.md` 中的 `vscode_video.py --apply`。工具先验证临时文件，再原子替换；不要仅修改扩展名。

## PiperCanonicalTCP-v1：哪些视频是 D435

- `foundation_replay_d435`、AnyGrasp D435 preview 和 `source_preview_compare/*d435*.png` 使用实体 D435 数据/标定。
- `head_cam_plan.mp4` 是用 D435 标定驱动的 SAPIEN head-camera 渲染，不是 D435 原始录像。
- `third_cam_plan.mp4`、左右 wrist MP4 与 debug/comparison MP4 都是仿真或合成画面，不应标作原始 D435 视频。

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

## V2 raw replay 旁边仍显示旧 Dense repaint

- 原因：现有 Stage-2 repaint/HDF5 是从 V1 `h2_pure_d435` 生成的，不会因为 raw replay 修复而自动变成 V2。
- 处理：论文扩展图把该格明确标为 `LEGACY V1 SOURCE / NOT V2`。不要静默配对；需要匹配版本时，从 `h2_pure_d435_urdfmatch_v2` 重新生成 Stage-2，并使用新的 processed/LeRobot 名称。

## 批处理不知道正在跑哪一集

- 查看 `tmux capture-pane -pt dense_replay_urdfmatch_v2:0 -S -60` 和 `_batch_logs/status.tsv`。
- `started` 后没有 `complete` 表示当前仍在执行或异常中断；重启同一命令时，只有同时具备 replay、targets、metadata、audit 和有效帧数的 episode 才会跳过。

## Audit V4 拒绝已有输出目录

- 症状：`Refusing to overwrite non-empty output root`。
- 原因：V4 默认保护已有 PNG、metadata 和报告。
- 处理：使用新的 `--output-root`。只有明确要重建 V4 自身产物时才使用 `--overwrite`；该选项也不会授权修改旧策略目录。

## PiperCanonicalTCP-v1：同-q 位置对但旋转固定差 90°

- 症状：`fk-contract-check` 的 raw SIM/URDF position error 接近零，但 rotation error 约 90°。
- 原因：SAPIEN `L6_SIM` 与 CuRobo/server `L6_URDF` 不是同一个局部轴 frame。
- 处理：确认 readback 包含 `T_L6SIM_L6URDF=Ry(+pi/2)`；适配后 rotation error 应接近 `0.00001°`。不能把这层和服务器 `Ry(-1.57)` 合并。

## Top-score IK 失败但 Orientation/Fused 成功

- 症状：Top target rotation 接近 180°，position 可以接近但严格 rotation IK 不收敛。
- 原因：raw-score 最大候选没有 orientation 约束，可能绕 approach 轴翻转。
- 处理：保留失败视频与 `eepose_failures.tsv`；不要强制翻转、不要把 rotation threshold 放宽到 pi。三种 head video 存在时仍合成策略对比。

## 输出目录非空但没有 SUCCESS

- runner 会拒绝覆盖该目录。使用新的 `--output-root`，或人工审计后保留失败结果；不要删除/覆盖旧 smoke 来伪造通过。

## `outputs_canonical_20260715/eepose` 看不到 OursV2 IK

- 原因：该目录的三项是 Orientation/Fused/Top-score 候选策略，三项都进入 Canonical IK。
- 处理：使用 `run_real_control_compare.sh`；`eepose_control.mp4` 才包含 Piper real reference、OursV2 legacy EE-pose IK 和 Canonical server-semantic IK。

## Real control compare 的 OursV2 EE-pose 约差 19.5 cm

- 原因：共同输入是服务器 `T_B_RTCP`，旧 OursV2 默认把数值原样当 `T_B_L6URDF` 目标，没有撤掉 `Ry(-1.57)@Tx(0.19)`。这不是 IK position threshold 放宽造成的。
- 处理：检查 `summary.json` 的 branch semantics；最终比较必须把两套 q 都按 Canonical RTCP FK 评价。不要在 OursV2 分支偷偷补 server tool，否则不再是旧链路对比。

## 仿真视图里夹爪在画外

- 前若干帧可能因 0515 标定 head camera 的视野而只显示 `offscreen` 箭头；这不代表 FK/曲线缺失。用完整 episode 检查后续帧，并以 world XYZ 曲线和 IK success mask 为数值依据。

## AnyGrasp / Human Replay 位置像被退了 12 cm

- AnyGrasp 原始 translation 是 D435 camera 坐标，必须先转 world；Canonical runner 已执行该变换。
- 区分 `approach_offset_m`（只构造 pregrasp）与 `target_retreat_m`（移动最终目标）。Canonical Human Replay 强制后者为 0；Legacy 12 cm 实验必须显式传入并记录在 manifest。
- 若缺 Canonical Human Replay，检查 `_sources/canonical_human_replay/.../head_cam_plan.mp4`、`EXIT_CODE` 和 `stderr.log`；合成器不会用 Legacy 冒充第四种 Canonical 方法。
- `handover_bottle/id1` 中 Human/CGRASP→RTCP 转换后约有 100° 目标朝向差，严格 `urdfik_max_rotation_threshold_rad=0.12` 会全部 miss；这是 Canonical Human Replay IK/可达性失败，不能通过重新加入 12 cm 最终 retreat 来修复。失败视频和 manifest 仍可用于观察差异。

## 论文候选图导出时报 `No module named 'cv2'`

- 原因：pine2 默认 `python3` 没有 OpenCV；这不表示生成的 PNG 损坏。
- 处理：改用 `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10` 执行导出与验证。

## 超宽 contact sheet 在应用预览中出现黑块

- 先用 OpenCV 统计严格黑色像素，再用 FFmpeg 完整解码各单图和 contact sheet。
- 本轮 frame 78 的源图、4 张单图和 contact sheet 严格黑色像素比例均为 0，且解码通过；黑块属于应用对超宽 PNG 的预览伪影，不是文件内容损坏。
