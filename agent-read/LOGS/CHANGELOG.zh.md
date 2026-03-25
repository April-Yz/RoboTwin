# CHANGELOG.zh

## 2026-03-25

- 修复 planner 导出的 wrist 视频朝向：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 问题：
    - `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4` 视觉上相当于期望视角逆时针转了 90 度
  - 排查结论：
    - 对比 `render_hand_retarget_r1_npz.py` 与 `galaxea_sim/robots/r1_pro.py` 后，R1 与 R1 Pro 的 wrist 相机挂载本地姿态本质一致：
      - 基础四元数 `[0.5, 0.5, -0.5, 0.5]`
      - 额外 RPY 偏移 `[-10deg, 0, -90deg]`
    - 因此这次不改相机挂载定义，而是在 planner wrist 视频写出前做图像平面纠正
  - 修复：
    - 新增 `rotate_wrist_rgb_for_export(...)`
    - 在 `record_frame(...)` 写出 left/right wrist 视频前统一做 `cv2.ROTATE_90_CLOCKWISE`
    - 先旋转，再叠字/转 BGR，保证 debug 文本方向也正常
  - 影响：
    - 不改变相机世界位姿、planner target、候选坐标系转换或 head 视频
    - 只修正 planner wrist 视频导出方向
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- 调整 URDF IK waypoint 可视化，并补充对 stage 收敛参数的分析说明：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 可视化改动：
    - `--debug_visualize_ik_waypoints 1` 现在除中间 waypoint 外，也会显示起点和终点 marker
    - 起点与终点统一使用红色 point+forward-axis marker
    - 中间 waypoint marker 尺寸缩小，避免遮挡 viewer 中的手和目标
  - 分析说明：
    - 当前 `init` 仍只执行 `apply_robot_init_pose(...)` 的一次性 joint 命令后短暂 `step_scene(settle_steps)`，不会像 stage 末尾那样再调用 `settle_arms_to_targets(...)` 等待完全收敛
    - `--settle_steps` 默认值为 `4`，其作用是：
      - init 后推进少量物理步
      - 每段 stage 主轨迹发完后，再额外推进若干 physics scene step
    - `--joint_target_wait_steps` 默认值为 `60`，其作用是：
      - 在 stage 轨迹结束后，继续逐步等待实际关节逼近最终 joint target
  - 影响：
    - 本轮不改变规划或执行逻辑，只改 viewer/debug 呈现
  - 相关代码位置：
    - waypoint marker 更新：
      - `update_ik_waypoint_visuals(...)`
      - `_ensure_ik_waypoint_marker_actors(...)`
      - `_ensure_ik_waypoint_endpoint_actors(...)`
    - init 与 stage 收敛：
      - `apply_robot_init_pose(...)`
      - `execute_single_arm_plan(...)`
      - `execute_dual_arm_plan(...)`
      - `settle_arms_to_targets(...)`
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- 修复 pure 模式 `pose_debug.jsonl` 导出时的 EE pose 类型兼容问题：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 问题：
    - `record_frame(...)` 新增的 `current_left_ee_pose_world_wxyz` / `current_right_ee_pose_world_wxyz` 导出逻辑，误把 `robot.get_*_ee_pose()` 当成 `sapien.Pose`
    - 但该机器人接口实际返回的是 7 维列表 `[x, y, z, qw, qx, qy, qz]`
    - 导致 pure 模式批处理在首帧落盘时触发：
      - `AttributeError: 'list' object has no attribute 'p'`
  - 修复：
    - 新增 `pose_like_to_world_wxyz(...)`
    - 统一兼容 `sapien.Pose` 与 7 维 pose 列表两种输入
    - `record_frame(...)` 中的 head/ee pose 导出改为统一走该 helper
  - 影响：
    - 不改变主规划/执行逻辑
    - 只修复 pure 模式下 `pose_debug.jsonl` 的数据序列化
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`

- 新增 CLI 参数 `--urdfik_cartesian_interp_auto_step_m`，用于控制 `--urdfik_cartesian_interp_steps=-1` 时自动 waypoint 模式的平移密度阈值。
- 旧逻辑中 `0.05m` 为硬编码；现在变为参数，默认仍为 `0.05`，固定步数模式不受影响。
- `render_hand_retarget_r1_npz_urdfik.py` 现在会在 `[ik-trajectory]` 与 `[ik-waypoints]` 日志中打印当前 `auto_step_m`。
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

## 2026-03-25 14:05:00 +08

- 为 `grasp` 未到位但仍继续 `close_gripper` 的场景增加显式警告：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 行为：
    - 在进入 `close_gripper` 前，如果 `grasp` 阶段 `reached=False`，打印 `[warn] grasp_not_reached_before_close ...`
    - 仅增加日志，不改变原有执行逻辑

- 为 URDF IK 的 Cartesian waypoint 模式增加自动步数模式：
  - 文件：
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - 新行为：
    - `--urdfik_cartesian_interp_steps > 1` 时，继续保留原有固定步数模式
    - `--urdfik_cartesian_interp_steps -1` 时启用自动模式
  - 自动模式规则：
    - 绝对平移距离 `<= 5cm` 时，不增加中间 TCP waypoint（等价于只保留起点和终点）
    - 平移距离 `> 5cm` 后，每增加一个 `5cm` 档位增加一个中间 waypoint
    - 示例：
      - `10cm` 平移 -> `1` 个中间 waypoint
      - `15cm` 平移 -> `2` 个中间 waypoint
  - 说明：
    - 该自动模式只根据 TCP 平移距离调节 waypoint 数，不改变现有 IK 目标或执行逻辑

- 记录 `cartesian_interp_ik` 的实际执行语义说明：
  - 新文档：
    - `agent-read/V1.15_urdfik_cartesian_interp_execution_semantics_ZH.md`
    - `agent-read/V1.15_urdfik_cartesian_interp_execution_semantics.md`
  - 结论：
    - 当前中间 `ee/tcp` waypoint 确实用于逐点 IK 求解
    - 最新修复后，执行阶段也会逐段消费这些 waypoint 对应的 `joint_waypoints`
    - `cartesian_interp_ik` 现在既影响 IK 解，也更直接影响最终执行轨迹

## 2026-03-25 14:25:00 +08

- 修复 `urdfik` 执行层未真正消费 `joint_waypoints` 的问题：
  - 文件：
    - `code_painting/render_hand_retarget_r1_npz_urdfik.py`
  - 改动：
    - `execute_plans(...)` 不再只从 `current_joints` 到 `target_joints` 做一次端点插值
    - 现在直接消费 `plan["position"]` / `plan["velocity"]`
    - 双臂执行时按左右臂相对轨迹进度交错推进，和基础 renderer 的轨迹执行语义保持一致
    - `_execute_single_ik_plan(...)` 也优先执行完整 `plan["position"]` 轨迹
  - 影响：
    - `cartesian_interp_ik` 生成的 `joint_waypoints` 现在会成为真实执行轨迹的一部分
# 2026-03-25

- pure 模式输出增强：
  - 文件：
    - `code_painting/plan_anygrasp_keyframes_r1.py`
  - 改动：
    - `pure_scene_output=1` 时不再生成 `debug_selection_preview.mp4`
    - 规划主流程现在会同时写出：
      - `head_cam_plan.mp4`
      - `left_wrist_cam_plan.mp4`
      - `right_wrist_cam_plan.mp4`
    - pure 模式会自动启用 `pose_debug.jsonl`，即使未显式传 `--save_pose_debug 1`
    - `pose_debug.jsonl` 现在额外记录：
      - 左右腕部相机 pose
      - 左右 TCP / EE pose
      - 左右臂 6 维 qpos
      - 左右夹爪 finger-joint qpos
      - 物体 actor pose / replay pose
    - `plan_summary.json` 新增对应的视频和数据路径字段
  - 文档：
    - `agent-read/2026-03-25_pure_mode_outputs_ZH.md`
    - `agent-read/2026-03-25_pure_mode_outputs.md`
  - 验证：
    - `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python -m py_compile code_painting/plan_anygrasp_keyframes_r1.py code_painting/plan_anygrasp_keyframes_r1_batch.py code_painting/render_hand_retarget_r1_npz_urdfik.py`
    - `git diff --check -- code_painting/plan_anygrasp_keyframes_r1.py`

- 新增 `--debug_visualize_ik_waypoints` 调试参数，用于在 `--planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik` 下，把中间 `tcp_waypoints_world` 可视化到 viewer/debug 输出中。
- 可视化形式为“小球 + 局部前进轴”，仅显示中间 waypoint，不显示起点和终点；终点继续使用原有 target axis。
- 该改动只影响调试显示，不改变 waypoint 生成、IK 求解、轨迹执行或碰撞逻辑。
- `plan_anygrasp_keyframes_r1_batch.py` 已同步透传该参数。
- 验证：本轮改动后运行 `python -m py_compile` 与 `git diff --check`。
# 2026-03-25

- 新增阶段性运行分析文档：
  - `agent-read/2026-03-25_overall_run_analysis_ZH.md`
  - `agent-read/2026-03-25_overall_run_analysis.md`
- 记录了当前版本的主要结论：
  - 轨迹形状已有改善，但 planner/IK 终点仍经常错误
  - `grasp` 未到位是 `close_gripper` 无接触的重要原因
  - 当前夹爪接触检测监控的是 finger joints 的 `child_link`，而不是 `left/right_gripper_link` 本体
- 本轮未修改运行逻辑，仅补充分析记录。
