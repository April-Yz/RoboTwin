# 当前功能摘要

## 当前主线

- 仓库默认版本仍按最新 v1.x 迭代管理；已有 Foundation、AnyGrasp、Ours v2、Dense Replay、Piper IK V3 等独立实验线。
- Piper/0515 定性链路和近期数据状态见 `README.zh.md`、`VERSION_SUMMARY.zh.md` 与 `ACTIVE_MEMORY.zh.md`。

## 本轮新增

- 新增隔离的 `Dense Replay URDF-match v2`，不修改旧 Dense 逻辑或旧输出。
- 修复 Curobo 与 SAPIEN `link6` 固定 `Ry(-90 deg)` 局部轴差、0.12 m TCP 反变换、插值参数被覆盖和执行未按关节收敛的问题。
- 入口：`code_painting/run_dense_replay_urdfmatch_v2.sh`。
- 代码：`render_hand_retarget_piper_dual_npz_urdfmatch_v2*.py`。
- 诊断、命令和限制：`COMMANDS/dense_replay_urdfmatch_v2.zh.md`。

## 读取顺序

1. `README.zh.md`
2. `CURRENT_FEATURE_SUMMARY.zh.md`
3. `VERSION_SUMMARY.zh.md`
4. 与任务对应的 `COMMANDS/*.zh.md`
