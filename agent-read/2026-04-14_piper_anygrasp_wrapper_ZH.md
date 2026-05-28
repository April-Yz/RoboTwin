# 2026-04-14 Piper AnyGrasp 规划包装层

## 目标

在不改原有 R1 / R1 Pro planner 文件的前提下，新增一个面向 Piper 的 AnyGrasp 规划入口。

## 新增文件

- `robot_config_Piper_dual.json`
- `code_painting/plan_anygrasp_keyframes_piper.py`
- `code_painting/plan_anygrasp_keyframes_piper_batch.py`
- `code_painting/run_plan_anygrasp_keyframes_piper_batch.sh`

## 设计思路

新的 Piper 入口本质上是对现有 R1 规划链的包装：

- `plan_anygrasp_keyframes_piper.py`
  - 导入 `plan_anygrasp_keyframes_r1.py`
  - 默认注入 Piper robot config
  - 将 replay renderer 替换为 Piper 专用子类，用于查找 Piper 的 camera / wrist link
  - 将 URDF-IK renderer 替换为使用 `assets/embodiments/piper/piper.urdf`
  - 尽量保留原始 CLI 和 planner 逻辑
- `plan_anygrasp_keyframes_piper_batch.py`
  - 复用现有 batch launcher 逻辑
  - 只替换单视频脚本入口
- `run_plan_anygrasp_keyframes_piper_batch.sh`
  - 对齐现有 R1 shell wrapper
  - 继续使用同一个 `RoboTwin_bw` conda 环境

## Robot config 说明

`robot_config_Piper_dual.json` 采用两个 Piper 实例（`dual_arm_embodied=false`，同时提供 left/right Piper 配置），这样可以在不重写原始双臂 planner 结构的情况下继续复用 left/right 执行路径。

这是一层“兼容性适配”，不是对原流程做完整语义改写。

## 当前限制

- 原 planner 整体仍然是强 R1 导向的。
- 这一版 Piper 支持主要是兼容层，重点是：
  - 复用 AnyGrasp candidate selection
  - 复用 replay / execution staging
  - 将 URDF IK 切到 Piper
- Piper 的相机坐标对齐后续仍可能需要继续调：
  - `--camera_cv_axis_mode`
  - `--head_camera_local_quat_wxyz`
  - 以及可能的 object/TCP offset
- 当前包装层默认在用户未显式指定时注入：
  - `--head_camera_local_quat_wxyz 1 0 0 0`

## 示例命令

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_piper_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --skip_existing 0 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps -1 \
  --urdfik_cartesian_interp_auto_step_m 0.01
```

## 最小验证

- `python -m py_compile code_painting/plan_anygrasp_keyframes_piper.py code_painting/plan_anygrasp_keyframes_piper_batch.py`
- `bash -n code_painting/run_plan_anygrasp_keyframes_piper_batch.sh`
- `python code_painting/plan_anygrasp_keyframes_piper.py --help`
