# CHANGELOG.zh

## 2026-03-25

- 新增 AnyGrasp 执行层关节收敛调试与补偿参数：
  - `--joint_command_scene_steps`
  - `--joint_target_wait_steps`
  - `--joint_target_wait_tol_rad`
- 修改位置：
  - `code_painting/plan_anygrasp_keyframes_r1.py`
  - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
- 改动目的：
  - 解决 `plan-request` / `plan-solution` 已经显示规划终点开始回退，但 `attempt` 仍然长期停留在前方偏差的问题。
  - 让执行阶段在每个 joint waypoint 后推进更多 physics scene step，并在整段轨迹结束后继续等待，直到实际关节更接近最终命令目标再测 reach error。
- 当前分析结论：
  - 先前对 `plan_vs_current_fwd_cm` 的语义有过误读。
  - 正确解释是：在 `plan-solution` 中，`plan_vs_current_fwd_cm > 0` 表示“当前姿态在规划终点前方”，也就是规划终点其实在当前姿态后方。
  - 因此在较后期的长 try 段里，IK 终点已经开始回退，但执行层没有充分收敛到该终点。
- 额外文档：
  - `agent-read/2026-03-25_ik_execution_regression/README.zh.md`
  - `agent-read/2026-03-25_ik_execution_regression/README.en.md`

- 增加“同目标时跳过 grasp 重执行”的逻辑：
  - 当 `pregrasp` 和 `grasp` 目标 pose 实际相同（典型情形是 `--approach_offset_m 0.0`）
  - 不再在到达 `pregrasp` 后，对同一个目标再做一次 `grasp` 重规划和执行
  - 直接把 `grasp` 视为复用 `pregrasp` 结果，并写入 `grasp_skipped_same_target`
- 这样做的原因：
  - 在 `approach_offset_m=0.0` 时，日志显示 `pregrasp` 已可到位
  - 但随后对同一目标再次进入 `grasp` 会把末端重新拉离正确位置
- 另外记录：
  - 本轮曾尝试在 `urdfik.py` 中加入 seeded/unseeded FK 后验评分
  - 该尝试导致 `pregrasp` 第一轮就出现大幅错误姿态
  - 已回退，不保留该修改

- 新增对 `cartesian_interp_ik` 的第三轮修正：
  - 把 `urdfik.py` 默认位置阈值从 `0.005m` 收紧到 `0.001m`
  - 把默认旋转阈值从 `0.05rad` 收紧到 `0.02rad`
  - 阈值放宽不再无上限扩大到 `0.1m`，而是限制在小范围内
  - `render_hand_retarget_r1_npz_urdfik.py` 现在会根据 IK 阈值自动缩减过细的 cartesian waypoint 数
- 直接原因：
  - `action` 阶段日志显示目标总位移大约只有几厘米到 9 厘米，但 `--urdfik_cartesian_interp_steps 30` 会把单步平移切到约 `3mm`
  - 旧的 IK 成功阈值是 `5mm`
  - 这会让求解器在很多 waypoint 上“几乎不动也算成功”，最终整条路径理论终点仍然离目标很远
- 结论更新：
  - 对这个问题，“更多 try”不是主修复手段
  - “继续加大插值步数”反而可能更差
  - 需要先保证单个 waypoint 的目标分辨率高于 IK 成功阈值

- 在此基础上又增加了 waypoint 级别的 `seeded/unseeded` 候选比较：
  - 位置：`code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - 行为：对同一个 waypoint，同时尝试“使用当前 seed”与“无 seed”两种 IK 解
  - 再用 FK 后验比较两者对该 waypoint 的 `ee` 目标误差，保留更接近的一支
- 目的：
  - 解决在 `action` 阶段中，插值分辨率修正后仍存在的“左手被当前 seed 锁在局部解里”的问题

- 调整终端调试输出格式：
  - `plan-request`
  - `plan-solution`
  - `attempt`
- 改动内容：
  - 双臂日志改为左右手分行打印
  - `theory` 从长字符串改为：
    - `forward`
    - `backward`
    - `aligned`
  - `fwd_cm` 增加 ANSI 颜色高亮，正负更容易区分

- 进一步压缩单样本结束时的终端总结：
  - 不再打印完整的 `statuses_by_arm={...}` 大字典
  - 改为短格式：
    - `arms`
    - `arm`
    - `obj`
    - `fXX=cYY`
    - `pre/gr/act`
    - `video`

## 2026-03-25 11:51:21 +08

- 修复日志格式化回归：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 问题：
    - `colorize_forward_cm()` 在前一轮日志重构后被错误缩进到 `short_direction_label()` 作用域内
    - 运行双臂 `plan-request` 时触发 `NameError: name 'colorize_forward_cm' is not defined`
  - 修复：
    - 将 `colorize_forward_cm()` 恢复为模块级函数
    - 同时修正其内部 `if/elif` 缩进，保证正值/负值/近零颜色分支正常工作
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

## 2026-03-25 12:08:00 +08

- 新增 grasp/action 期物体碰撞开关：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - 新参数：
    - `--enable_grasp_action_object_collision 0|1`
  - 行为：
    - 默认 `0`，保持原来的无碰撞模式不变
    - 设为 `1` 时，为被执行臂选中的执行物体保留 collision geometry
    - 在 `pregrasp` 阶段仍关闭这些物体的碰撞
    - 在 `close_gripper` 前开启所选物体碰撞，并保持到 `action` 阶段结束
    - 不修改对象附着逻辑、TCP 相对位姿、目标位姿生成或其它相对变换
  - 实现说明：
    - 通过缓存并恢复 SAPIEN collision groups 做阶段性启停
    - 未被选中的其它物体仍保持原有行为
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-25 12:22:00 +08

- 调整 `plan_anygrasp_keyframes_r1.py` 的默认可视化行为：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 改动：
    - 将目标位姿坐标轴可视化恢复为默认开启
    - 在这条规划脚本里默认隐藏 left/right wrist camera，使保存视频和 viewer 中不再出现 wrist 相机视野框
  - 说明：
    - 这只影响 `plan_anygrasp_keyframes_r1.py` 这条脚本的默认行为
    - 不修改底层通用 renderer 的其它脚本用途
    - 不影响 head camera / third-person 输出本身
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

## 2026-03-25 13:05:00 +08

- 为 `plan_anygrasp_keyframes_r1.py` 增加纯净/调试可视化控制：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 新参数：
    - `--debug_visualize_targets 0|1`
    - `--viewer_show_camera_frustums 0|1`
  - 改动：
    - 将之前硬编码常开的目标坐标轴恢复为显式参数，默认仍为开启
    - 在 viewer 路径里默认关闭 SAPIEN `ControlWindow.show_camera_linesets`
    - 这样 wrist frustum 隐藏后，剩余的 zed/third 相机视野线框也会一起默认关闭
  - 说明：
    - `pure_scene_output` 继续负责主视频去掉文字、候选 gripper、target axis
    - `viewer_show_camera_frustums=0` 负责去掉 viewer 里的相机线框
    - `debug_visualize_targets=1` 可保留 target axis 作为 debug 模式
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- 修复 batch 包装层未透传纯净/调试可视化参数的问题：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
  - 改动：
    - 为 batch 脚本补充 `--debug_visualize_targets`
    - 为 batch 脚本补充 `--viewer_show_camera_frustums`
    - 在 `build_single_command()` 中将这两个参数继续透传给 `plan_anygrasp_keyframes_r1.py`
  - 原因：
    - 之前纯净模式命令在 batch 层被 argparse 拒绝，主脚本参数无法到达
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1_batch.py`

## 2026-03-25 13:35:00 +08

- 为 `--enable_grasp_action_object_collision=1` 增加最小版“渐进闭合直到接触/停滞”逻辑：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 问题分析：
    - 之前该开关只是在 `grasp/action` 阶段重新启用所选物体碰撞
    - 夹爪仍然使用一次性 `set_grippers(close)`，且仅推进极少量物理步
    - 因此即使物体存在 collision shape，手指仍可能视觉上完全闭合并穿过物体
  - 改动：
    - 新增 `close_grippers_progressively_with_collision_stop()`
    - 在 `close_gripper` 阶段按小步命令闭合夹爪
    - 每步推进物理，并读取夹爪关节 `qpos`
    - 同时检查所选物体与当前执行臂夹爪 links 的接触
    - 当“已接触且夹爪关节位移停滞”时提前停止闭合
    - 默认无碰撞模式保持原来的 `renderer.set_grippers(...)` 行为不变
  - 备注：
    - 这是最小修复，不改变 `pregrasp/grasp/action` 目标构造，也不改变物体附着到 TCP 的相对位姿
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
    - `git -C /home/zaijia001/ssd/RoboTwin diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- 修复渐进闭合逻辑中的 SAPIEN API 兼容问题：
  - 问题：
    - `PhysxArticulationJoint` 在当前环境没有 `get_qpos()`，导致批处理在进入 `close_gripper` 时崩溃
  - 修复：
    - 改为从 articulation 的 `entity.get_qpos()` 读取整条 `qpos`
    - 再按 `active_joints` 和 `joint.get_dof()` 累积偏移，提取夹爪 joint 对应的实际关节值
  - 代码位置：
    - `code_painting/plan_anygrasp_keyframes_r1.py:_get_gripper_joint_positions`
