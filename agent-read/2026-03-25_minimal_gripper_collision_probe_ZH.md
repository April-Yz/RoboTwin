# 2026-03-25 最小 gripper/object 碰撞探针

## 目的

在不经过 AnyGrasp 候选选择、IK 规划、pregrasp/grasp/action 阶段切换的情况下，单独验证下面这个问题：

- 机器人手指与物体在 `close_gripper` 时是否真的会产生物理接触？
- 当前 `plan_anygrasp_keyframes_r1.py` 里的 contact debug 为何会打印全机器人 `shapes=0` / `contact=0`？

## 新增脚本

- `code_painting/minimal_gripper_collision_probe.py`

## 脚本逻辑

最小实验只保留：

1. 加载 R1 机器人到 SAPIEN scene
2. 不挂 planner，仅保留机器人 articulation 与 gripper drive
3. 创建一个测试物体：
   - 默认：简单 box collision + visual
   - 可选：mesh visual + `convex/solid_bbox` collision
4. 将测试物体直接放到当前 TCP 前方的局部偏移位姿
5. 仅执行单侧 gripper 从 `open -> close`
6. 每一步打印：
   - gripper joint qpos
   - raw scene contact 总数
   - 与目标物体相关的 raw contact pairs
7. 额外打印机器人 link 的 component/collision shape 摘要

## 关键发现

### 1. 当前 helper 读取 robot link collision shape 的结论并不可信

脚本里对 robot links 做 component 级摘要时，仍然得到：

- `links_with_nonzero_component_shapes=0`

这与主流程中看到的：

- `left_gripper_link(shapes=0)`
- `left_gripper_finger_link1(shapes=0)`
- `all_robot_links ... shapes=0`

现象一致。

但下面的 raw contact 实验表明：

- 即使 helper/summary 报 `shapes=0`
- 物理 scene 里仍然可以出现真实的 robot-object contact

因此当前结论应更新为：

> `shapes=0` 更像是“当前 debug helper 读 articulation link collision 的方式不对”，不能直接推出“机器人没有碰撞体”或“物理中没有接触”。

### 2. 最小 box 探针能够稳定观测到手指与物体的 raw contact

命令：

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  code_painting/minimal_gripper_collision_probe.py \
  --arm left \
  --object_kind box \
  --probe_local_offset 0.04 0.0 0.0 \
  --max_iters 20 \
  --settle_steps_per_iter 8
```

观察到：

- 从 `iter=2` 开始出现：
  - `left_gripper_finger_link1<->probe_box`
- 后续还能同时出现：
  - `left_gripper_finger_link1<->probe_box`
  - `left_gripper_finger_link2<->probe_box`

这说明：

- 在最小场景里，robot-object 物理接触是存在的
- finger link 与 box probe 的 raw contacts 能被 `scene.get_contacts()` 抓到

### 3. 最小 mesh+solid_bbox 探针同样能观测到稳定 raw contact

命令：

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  code_painting/minimal_gripper_collision_probe.py \
  --arm left \
  --object_kind mesh \
  --mesh_path /home/zaijia001/ssd/data/R1/hand/obj_mesh/blue_cup/blue_cup.obj \
  --mesh_collision_mode solid_bbox \
  --probe_local_offset 0.04 0.0 0.0 \
  --max_iters 20 \
  --settle_steps_per_iter 8
```

观察到：

- 很早就出现：
  - `left_gripper_finger_link1<->probe_mesh`
- 后续出现：
  - `left_gripper_finger_link2<->probe_mesh`
  - `left_gripper_link<->probe_mesh`
  - `left_realsense_link<->probe_mesh`

这说明：

- 使用与主流程更接近的 `mesh + solid_bbox` 组合时
- 机器人和物体依然能够在最小场景里产生 raw contact

## 对主流程问题的新解释

结合最小探针结果，主流程里“视频看起来穿模，但日志一直 `contact=0`”的问题，已经不能再简单解释为：

- 机器人没有 collision
- 或 `solid_bbox` 物体完全没有碰撞

更合理的新解释是：

### A. 当前主流程中的 contact debug helper 存在 articulation-link 读取盲区

也就是：

- `shapes=0` 不代表真实物理中没有碰撞
- 当前 `_get_actor_collision_shapes(...)` / entity-summary 这条路径对 articulation links 不可靠

### B. 主流程中的 monitor/contact 匹配逻辑与最小实验不一致，或被时序/位姿问题破坏

由于最小实验里 raw contact 明确存在，而主流程里却持续 `monitor_contact=0`，需要继续检查：

1. 主流程 close 阶段的目标物体位姿是否与视频中看到的一致
2. close 阶段 object actor 是否被其他逻辑覆盖/重设
3. 主流程 contact 统计时拿到的 entity identity 是否与 raw contact bodies 对应不上
4. 主流程的 object enable timing 与 visual frame 观察之间是否有时序错位

## 当前最重要的新结论

1. **不能再把 `shapes=0` 直接当作“robot 没有碰撞体”的证据。**
2. **最小隔离实验已经证明：R1 gripper 与 box / mesh(solid_bbox) 物体之间，raw physics contact 是可以发生并被抓到的。**
3. **因此主流程中 `contact=0` 的问题，更像是主流程监控/匹配/时序问题，或目标物体在主流程中的实际位姿与视频观感不一致，而不是物理引擎完全不支持 robot-object 接触。**

## 输出文件

- `code_painting/minimal_gripper_collision_probe/probe_left_box.json`
- `code_painting/minimal_gripper_collision_probe/probe_left_mesh.json`

这些 JSON 保存了逐步闭合时的 raw contacts 与 qpos 演化，可用于后续继续比对主流程。 

## 主流程回灌验证

在给 `plan_anygrasp_keyframes_r1.py` 增加 raw target contact 输出后，重新执行了主流程 close 阶段调试命令（关闭 viewer 与视频导出以加快验证）。

代表命令：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_debug_collision_batch_raw_probe \
  --ids 0 \
  --skip_existing 0 \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps -1 \
  --urdfik_cartesian_interp_auto_step_m 0.05 \
  --candidate_selection_mode planner \
  --left_target_object cup \
  --right_target_object bottle \
  --candidate_target_local_x_offset_m -0.05 \
  --approach_offset_m 0.08 \
  --reach_error_pose_source tcp \
  --replan_until_reached 1 \
  --replan_until_reached_max_attempts 3 \
  --enable_grasp_action_object_collision 1 \
  --execution_object_collision_mode solid_bbox \
  --debug_collision_report 1 \
  --gripper_contact_monitor_mode all_robot_links \
  --enable_viewer 0
```

### 新观察

主流程现在会额外打印：

- `raw_target_contacts=...`
- `raw_target_contact_total=...`
- `[gripper-close] ... raw_target_contact=...`

而这条 case 的 close 阶段结果是：

- init：`raw_target_contacts=['none']`
- iter 1~20：`raw_target_contact_total=0`
- 最终：`raw_target_contact=0`

### 这说明什么

这一步非常关键，因为它把问题进一步缩小了：

1. 最小隔离实验里，raw contacts 能稳定出现
2. 但主流程 close 阶段里，即使直接绕过 monitor helper，`scene.get_contacts()` 仍然对目标物体返回 0 条 raw target contact

因此当前更强的结论是：

> 主流程中不仅 monitor/helper 看不到接触；在当前这条 case 的 close 阶段，raw physics contact 本身也没有发生。

这意味着主矛盾更偏向：

- close 阶段目标物体的真实 pose / 姿态 / 尺寸与我们从视频直觉理解的不一致
- 或目标物体在主流程中的位置已经不在会与 finger/link 发生 raw contact 的区域
- 而不是“physics contact 已存在，只是 debug monitor 没匹配上”

### 相比上一轮结论的更新

上一轮最小实验只能说明：

- 物理引擎和 R1 gripper/object 接触机制本身是可工作的

这轮主流程回灌则进一步说明：

- 对于当前 `d_pour_blue_0` 的 close 阶段，问题不是单纯 contact monitor 漏报
- 因为连 raw target contacts 也没有

所以后续最值得继续查的是：

1. close 阶段物体 actor 的真实世界位姿
2. close 阶段物体 collision 开启后，actor pose 是否仍与视频中看到的位置一致
3. wrist/head 视频里的“穿模观感”是否来自 visual mesh 与真实 collision primitive 的错位
4. candidate / target offset 是否把手实际带到了一个“视觉上靠近、物理上并未进入 target collision 区域”的关系

## close 阶段物体 pose / collision debug 增强

本轮继续在主流程中加入：

- 目标物体 actor 当前 pose
- 目标物体 collision debug 信息（含 `solid_bbox` 的 `center/half_size`）

对应输出示例：

- `target_pose=planned_object_cup(p=[...],q=[...])`
- `target_collision_debug={... 'center': [...], 'half_size': [...]}`

### 针对当前 `d_pour_blue_0` 的直接观察

left hand close 阶段 init：

- cup actor pose：
  - `p=[-0.1836, 0.0499, 0.8385]`
- left gripper base pose：
  - `p=[-0.2086, -0.0007, 0.9410]`
- cup solid bbox：
  - `center=[0.0, 0.053424, 0.0]`
  - `half_size=[0.040001, 0.053424, 0.039616]`

right hand close 阶段 init：

- bottle actor pose：
  - `p=[0.0799, 0.0772, 0.8432]`
- right gripper base pose：
  - `p=[0.0717, 0.0533, 0.9032]`
- bottle solid bbox：
  - `center=[0.0, 0.100586, 0.0]`
  - `half_size=[0.033206, 0.100586, 0.032682]`

### 新结论补充

结合 raw target contact 始终为 0，以及 close 期间 target pose 基本稳定不变，可以得到更强判断：

1. 当前 close 阶段不是“物体被别的逻辑不断重设导致 contact 丢失”这种简单问题
2. 更像是当前目标物体的 visual 外观与其 collision primitive（尤其 `solid_bbox`）存在显著几何偏差
3. 也就是说，视频中看起来像 finger 穿过了物体 mesh，但对 `solid_bbox` 而言，finger 可能并没有真正进入能产生 raw contact 的区域

因此后续最值得做的，不再只是加更多 contact 日志，而是：

- 直接可视化 object collision bbox
- 或打印/导出“finger/base pose 到 object bbox center/extent 的相对关系”

## 物体 collision bbox 可视化

本轮新增参数：

- `--debug_visualize_object_collision_bbox 1`

行为：

- 当 execution object 使用 `solid_bbox` collision 时
- 为对应物体额外创建一个 visual-only bbox actor
- bbox actor 使用与 collision 相同的局部 `center/half_size`
- bbox actor 跟随 execution object 的 world pose 更新

用途：

- 直接在 viewer / 导出视频中对比：
  - 物体 visual mesh
  - 物体 collision bbox
- 判断 wrist/head 视角中的“穿模”究竟是：
  - 穿过 visual mesh
  - 还是也穿过了 collision bbox

### 验证

已用命令回灌执行：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_debug_collision_batch_bbox_probe \
  --ids 0 \
  ... \
  --execution_object_collision_mode solid_bbox \
  --debug_collision_report 1 \
  --debug_visualize_object_collision_bbox 1 \
  --gripper_contact_monitor_mode all_robot_links \
  --enable_viewer 0
```

本轮运行通过，未引入新的执行报错。

### 你下一步最应该观察什么

建议你下一轮亲自看：

- `anygrasp_single_debug_collision_batch_bbox_probe/d_pour_blue_0/head_cam_plan.mp4`
- 如果之后你要开 viewer，也看 wrist / head 里绿色 bbox 与物体 mesh 的相对位置

重点判断：

1. finger 是否只是穿过了物体 mesh，但没有进入绿色 bbox
2. 若 finger 也明显进入了绿色 bbox，但 raw contact 仍为 0，再继续查物理层

## `convex` 对照实验

本轮对主流程执行物体碰撞模式做了直接对照：

- `solid_bbox`
- `convex`

对照命令（本轮执行的是 `convex`）：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_debug_collision_batch_convex_probe \
  --ids 0 \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --candidate_selection_mode planner \
  --left_target_object cup \
  --right_target_object bottle \
  --candidate_target_local_x_offset_m -0.05 \
  --approach_offset_m 0.08 \
  --reach_error_pose_source tcp \
  --enable_grasp_action_object_collision 1 \
  --execution_object_collision_mode convex \
  --debug_collision_report 1 \
  --gripper_contact_monitor_mode all_robot_links \
  --enable_viewer 0
```

### 对照结果

`convex` 模式下，close 阶段结果仍然是：

- init：`raw_target_contacts=['none']`
- iter 1~20：`raw_target_contact_total=0`
- 最终：`raw_target_contact=0`

同时：

- cup target shape 类型变成：`PhysxCollisionShapeConvexMesh`
- bottle target shape 类型变成：`PhysxCollisionShapeConvexMesh`

也就是说：

> 即使把 execution object collision 从 `solid_bbox` 换成 `convex`，当前这条主流程 case 的 close 阶段仍然没有产生 raw target contact。

### 这意味着什么

这个对照非常重要，因为它基本排除了下面这个解释：

- “只是 `solid_bbox` 近似太粗糙 / 太大 / 太保守，才导致 close 阶段 contact 为 0”

当前更强的结论是：

1. 问题不局限于 `solid_bbox` 的盒子近似
2. 即便使用 `convex` mesh collision，主流程 close 阶段仍然没有 raw contact
3. 因此主矛盾更像是：
   - 主流程当前 hand/object 几何关系本身没有真正进入 target collision 区域
   - 或 close 阶段 object-enable / pose / 时序语义仍有更深层问题
4. 单纯缩小 bbox 现在已经不是最高优先级解释

## 当前阶段性判断更新

到目前为止，已经确认：

- 最小隔离实验：
  - box probe 可以产生 raw contact
  - mesh + `solid_bbox` probe 也可以产生 raw contact
- 主流程：
  - `solid_bbox` 无 raw contact
  - `convex` 也无 raw contact

所以当前最值得继续追的方向变成：

1. 主流程 close 阶段手与物体的**真实相对几何关系**
2. 是否需要输出 / 可视化 finger/base 相对 object collision 几何中心和边界的距离
3. 是否应该把 close 开始前那一帧的真实 hand/object pose 复刻到最小探针里复现

## close 起始 pose 导出

本轮新增 close 起始姿态导出：

- dual-arm：
  - `close_stage_snapshot_dual_before_close.json`
- single-arm：
  - `close_stage_snapshot_<arm>_before_close.json`

导出内容包括：

- 当前 arm 的 TCP 世界位姿
- gripper joint qpos
- gripper base pose
- finger link pose
- object actor pose
- object collision mode / collision debug info

### 当前导出文件

在“从一开始就让 selected object 保持碰撞开启”的实验里，已生成：

- `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_all_collision_from_start/d_pour_blue_0/close_stage_snapshot_dual_before_close.json`

## 从一开始就开启物体碰撞的实验

本轮新增参数：

- `--grasp_action_object_collision_start_stage {close_gripper,grasp,pregrasp}`

默认仍是：

- `close_gripper`

新实验使用：

- `pregrasp`

即：

- selected execution objects 从一开始就保持 collision enabled
- 不再等到 `close_gripper` 前才开启

### 实验命令

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_all_collision_from_start \
  --ids 0 \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --candidate_selection_mode planner \
  --left_target_object cup \
  --right_target_object bottle \
  --candidate_target_local_x_offset_m -0.05 \
  --approach_offset_m 0.08 \
  --reach_error_pose_source tcp \
  --enable_grasp_action_object_collision 1 \
  --grasp_action_object_collision_start_stage pregrasp \
  --execution_object_collision_mode convex \
  --debug_collision_report 1 \
  --debug_visualize_object_collision_bbox 1 \
  --gripper_contact_monitor_mode all_robot_links \
  --save_debug_preview 1 \
  --save_debug_execution_preview 1 \
  --enable_viewer 0
```

### 结果

这轮实验与之前最大的不同在于：

- **主流程中终于出现了 raw target contacts**
- 例如 close init 时就能看到：
  - `left_gripper_finger_link1<->planned_object_cup`
  - `right_gripper_finger_link2<->planned_object_bottle`
  - `right_gripper_link<->planned_object_bottle`
  - `right_realsense_link<->planned_object_bottle`
- close 过程中 raw target contacts 持续存在
- `[gripper-close]` 最终也变成：
  - `raw_target_contact=1`

### 但仍有一个重要现象

即使 raw target contacts 已经出现，当前 `monitor_contact` / `base_contact` 仍显示为 0。

这说明：

1. 主流程中“原来完全没有 raw contact”的一个重要原因，确实和“物体碰撞启用时机过晚”有关
2. 但当前 contact monitor 仍然存在匹配问题：
   - raw target contacts 已经存在
   - monitor/helper 却仍然报 0

### 当前新的最强结论

到这一轮为止，可以更有把握地说：

- 如果 selected object 从 `pregrasp` 开始就参与碰撞，主流程里是能够观察到 raw target contacts 的
- 因此之前 `close_gripper` 才开启碰撞，确实会错过一部分已经形成的接触/重叠关系
- 同时，当前 `monitor_contact` 逻辑依然不可靠，因为它在已有 raw contact 的情况下仍输出 0

也就是说，问题现在被拆成了两部分：

1. **碰撞启用时机问题**
   - 旧默认 `close_gripper` 太晚
2. **contact monitor 匹配问题**
   - raw contact 已存在时，monitor 仍可能漏报

## 缩小 cup / bottle，但保留 close_gripper 才开始碰撞的实验

按用户要求，新增了“缩小执行物体，但保留原始 close_gripper 才启用碰撞”的验证。

### 实验设置

- `--grasp_action_object_collision_start_stage close_gripper`
- `--execution_object_collision_mode convex`
- `--execution_object_scale_override cup=0.9`
- `--execution_object_scale_override bottle=0.9`
- `--gripper_contact_monitor_mode all_robot_links`

### 实验命令

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_scaled_close_only_probe \
  --ids 0 \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --candidate_selection_mode planner \
  --left_target_object cup \
  --right_target_object bottle \
  --candidate_target_local_x_offset_m -0.05 \
  --approach_offset_m 0.08 \
  --reach_error_pose_source tcp \
  --enable_grasp_action_object_collision 1 \
  --grasp_action_object_collision_start_stage close_gripper \
  --execution_object_collision_mode convex \
  --execution_object_scale_override cup=0.9 \
  --execution_object_scale_override bottle=0.9 \
  --debug_collision_report 1 \
  --debug_visualize_object_collision_bbox 1 \
  --gripper_contact_monitor_mode all_robot_links \
  --enable_viewer 0
```

### 结果

这轮结果是：**没有检测到**。

close 阶段日志显示：

- init: `raw_target_contacts=['none']`
- iter 1~20: `raw_target_contact_total=0`
- final:
  - `left ... raw_target_contact=0`
  - `right ... raw_target_contact=0`

### 当前结论更新

把 `cup` 和 `bottle` 缩小到 `0.9` 后：

- 如果仍然保留原始逻辑，即 **只在 `close_gripper` 才开始启用碰撞**
- 那么这条样例里仍然 **检测不到 raw target contact**

因此目前可以认为：

- 仅靠把执行物体缩小，并不能让“close 时才启用碰撞”的旧逻辑恢复出接触检测
- 之前能看到 raw contact 的关键因素，仍然主要是：
  - **碰撞启用时机前移到 `pregrasp`**
- 缩放本身至少在 `0.9` 这个幅度下，没有把 `close_gripper`-only 模式救回来

## 关于“全程碰撞”输出是否正常

就目前日志来看，**是正常且有解释力的**，但要注意“正常”不等于“monitor 逻辑正确”。

可以认为这轮输出正常的原因是：

- `pregrasp` 开始启用碰撞后，主流程稳定出现 raw target contacts
- 物体 pose 稳定，没有看到明显的异常爆炸或失控
- 轨迹仍然可执行到 close / action 阶段
- `raw_target_contact` 与 headed 观察到的“确实有物理接触”是一致的

但不正常/未修复的部分仍然是：

- `monitor_contact=0`
- `base_contact=0`

所以更准确的表述是：

- **全程碰撞实验本身是正常、可信的**
- **但 contact monitor 这条统计链仍然漏报**

## 缩小到 0.8 / 0.5 的全程碰撞实验

继续基于：

- `--grasp_action_object_collision_start_stage pregrasp`
- `--execution_object_collision_mode convex`
- `--gripper_contact_monitor_mode all_robot_links`

分别测试：

- `cup=0.8, bottle=0.8`
- `cup=0.5, bottle=0.5`

### 0.8 结果

输出目录：

- `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_all_collision_scale08/d_pour_blue_0`

close 日志显示：

- left init 就有：`left_gripper_finger_link1<->planned_object_cup`
- right init 就有：`right_gripper_finger_link2<->planned_object_bottle`
- close 过程中 raw target contacts 持续存在
- 最终：
  - `left ... raw_target_contact=1`
  - `right ... raw_target_contact=1`

结论：

- 缩小到 `0.8` 后，全程碰撞模式仍然能稳定检测到 raw target contact

### 0.5 结果

输出目录：

- `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_all_collision_scale05/d_pour_blue_0`

close 日志显示：

- left init 时：`raw_target_contacts=['none']`
- right init 时已有：`right_gripper_finger_link1<->planned_object_bottle`
- left 到 close iter 8 左右开始出现 raw contact
- right 在 close 全程都有 raw contact
- 最终：
  - `left ... raw_target_contact=1`
  - `right ... raw_target_contact=1`

结论：

- 即使缩小到 `0.5`，全程碰撞模式下最终仍然能检测到 raw target contact
- 只是 left 的接触出现时机会更晚，不再是 close init 就立刻出现

### 当前缩放实验的结论汇总

在“从 `pregrasp` 开始启用碰撞”的前提下：

- `0.9`：能检测到
- `0.8`：能检测到
- `0.5`：仍然能检测到

也就是说：

- 对这条样例来说，只要碰撞启用时机足够早，物体缩小并不会让 raw contact 消失
- 主要决定因素依然是“启碰撞时机是否前移”，而不是单纯的物体大小

## 0.5 全程碰撞结果分析

用户反馈：在 `0.5 + pregrasp` 的结果里，成功接触时夹爪看起来没有完全闭合；这正是用户期望的效果。

### 结合日志的解释

这轮现象是合理的，且和日志一致：

1. `close_stage_snapshot_dual_before_close.json` 表明 close 开始前两侧 gripper joint qpos 约为：
   - `[0.045001, 0.044999]`
   - 即 close 开始前夹爪还明显是张开的
2. 在 `0.5 + pregrasp` 下：
   - right 从 close init 就已有 raw contact
   - left 在 close 中后段（约 iter 8）才开始出现 raw contact
3. 这意味着：
   - 物体缩小后，至少左手侧不再是一开始就被两指“夹住”
   - 夹爪需要进一步闭合，直到手指真正靠到物体上，接触才开始出现
4. 因此视频里出现“夹爪没有完全闭合、被物体挡住”的视觉效果，是符合这组 raw contact 时序的

### 一个重要细节

当前 close 逻辑的停止条件不是“只要有 raw contact 就立即停”，而是：

- 先通过 `monitor_contact` 检测接触
- 再结合 `qpos_delta <= stall_qpos_tol`
- 满足连续若干步后才以 `contact_stall` 停止

但现在 monitor 仍然漏报，所以这轮日志里最终 reason 仍常常是：

- `target_reached`

这并不表示“物理上完全闭合到了零缝隙”，而更接近：

- 控制命令已经下到了 `cmd=0.0`
- 但实际视觉上仍可能被物体挡住，显得没有完全闭合

所以你的主观观察是合理的：

- **0.5 全程碰撞的结果，确实更接近“手指闭合到物体后被挡住、不完全闭合”的效果**

## 0.5 缩放下：从 close 开始碰撞 + 只计算 finger 的实验

按你的要求，新跑了一轮：

- `--grasp_action_object_collision_start_stage close_gripper`
- `--gripper_contact_monitor_mode fingers`
- `--execution_object_scale_override cup=0.5`
- `--execution_object_scale_override bottle=0.5`

输出目录：

- `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_scale05_close_only_fingers/d_pour_blue_0`

### 结果

close 日志显示：

- left:
  - init: `raw_target_contacts=['none']`
  - iter 1~20: `raw_target_contact_total=0`
  - final: `raw_target_contact=0`
- right:
  - init: `raw_target_contacts=['none']`
  - 从 iter 8 开始出现 `right_gripper_finger_link2<->planned_object_bottle`
  - final: `raw_target_contact=1`

同时 monitor/helper 仍然是：

- `monitor_contact=0`

### 结论

把问题限定为：

- **碰撞只从 close 开始启用**
- **只看 finger**
- **物体缩到 0.5**

则原来的问题 **没有完全消失**：

- 左手仍然检测不到 raw target contact
- 右手可以检测到，但 monitor 仍然漏报

所以更具体地说：

1. 缩小到 `0.5` 后，确实更容易出现“闭合到物体后被挡住、不完全闭合”的视觉效果
2. 但如果碰撞启用时机仍然拖到 `close_gripper`，问题仍然存在：
   - left 仍可能完全检测不到
3. 如果要稳定得到你想要的这种“闭合被物体挡住”的效果，当前更可靠的条件仍是：
   - **从 `pregrasp` 开始启用碰撞**

## 为什么“先进入碰撞体内部，再在 close 阶段启碰撞”会直接闭合，而不是当场卡住

用户提出的关键现象是：

- 在 `pregrasp / grasp` 阶段如果夹爪已经进入物体碰撞体内部
- 但碰撞要到 `close_gripper` 才开启
- 那么开启碰撞后，并不会在当前姿态“立刻卡住”
- 反而常常继续向 `cmd=0.0` 闭合，表现得像“直接闭到底”

结合当前代码和实验结果，这个现象主要由以下几件事共同造成。

### 1. 晚启碰撞不会把系统回退到“首次接触之前”

`close_gripper` 前才把 collision group 打开时，物体和手指已经在一个**带重叠的初始状态**里。

这时物理不会自动做的事情是：

- 不会把手指回退到“刚好接触”的边界位置
- 不会把已经重叠的几何自动投影回一个理想无穿透姿态
- 不会替你重建“从外部接近 -> 首次接触 -> 阻挡”的历史过程

所以晚启碰撞得到的不是“接触临界点”，而是“已经重叠后的当前状态”。

### 2. 当前 close 停止逻辑并不是看 raw contact，而是看 monitor_contact + stall

当前 `close_grippers_progressively_with_collision_stop(...)` 的停止条件核心是：

- `has_contact = _contact_involves_entities(...)`
- 且 `qpos_delta <= stall_qpos_tol`
- 且持续若干步 (`contact_confirm_iters`)
- 才会设为 `reason=contact_stall`

但实验已经反复说明：

- raw target contact 可能已经存在
- `monitor_contact` 仍然是 0

于是结果就变成：

- 控制器继续把夹爪命令往 `target=0.0` 推
- 最终 reason 常常是 `target_reached`
- 看起来就像“直接完全闭合了”

所以这里不是“没有物理接触”就一定会闭合，而是：

- **即使有 raw contact，只要 monitor 没认出来，就不会按 contact_stall 停下来**

### 3. 左右手指不是独立控制，而是同一个 gripper 标量命令驱动一组关节

从 `envs/robot/robot.py` 可见，`set_gripper(gripper_val, arm_tag, ...)` 是：

- 给当前 arm 的 `joints = self.left_gripper / self.right_gripper`
- 对这组 gripper joints 统一设置 drive target

也就是说当前语义更接近：

- **一个 arm 的夹爪由同一个开合标量命令驱动**
- 不是“左指一个命令、右指一个命令”分别独立控制

因此即使现实直觉上你会觉得：

- 一侧手指应该先被挡住
- 另一侧也许还能继续动一点

在当前控制结构里，系统并没有“按单根手指独立停住”的高层逻辑；它只是持续给整组 gripper joints 下同一个 closing target。

### 4. 物体是 kinematic actor，晚启碰撞时也不会像动态物体那样被挤开

执行对象当前是通过 `builder.build_kinematic(...)` 创建的。

这意味着：

- 物体不会像 dynamic rigid body 那样被手指推走、弹开、自动让位
- 晚启碰撞时，如果系统一开始就在重叠状态，求解器也不会把对象主动挤出到一个“合理接触面”位置

于是你看到的就更容易是：

- 物体保持原位
- 控制器继续闭合
- 而不是“在当前姿态瞬间形成一个非常干净的卡住状态”

### 5. 为什么不是“一侧卡住，另一侧继续”

这件事在当前系统里受两个约束：

1. **控制上是耦合的**
   - 一个 arm 只有一个 gripper close/open 标量命令
2. **停止逻辑也不是按单根手指分别判停**
   - 当前是按 arm 级别的 monitor/contact 状态来决定是否停止整只 gripper

所以系统不会自然表现成：

- finger1 停住
- finger2 继续单独推进很多

更可能的表现是：

- 两侧都继续朝共同目标前进
- 若 monitor 没认到接触，就一起继续闭合
- 若某侧 raw contact 较晚出现，也只是体现在 raw log 里更晚，而不会自动变成“单指停住”的控制行为

## 当前最准确的结论

因此，晚启碰撞时“已经在碰撞体内部却仍继续闭合”的根本原因，不是单一的物理 bug，而是这几个机制叠加：

1. **碰撞启用太晚**，系统已经错过首次接触历史
2. **不会自动回退到接触边界**
3. **停止逻辑依赖 monitor_contact，而 monitor 仍在漏报**
4. **夹爪控制是 arm 级耦合的，不是单指独立控制**
5. **物体是 kinematic，不会被自然挤开成一个稳定接触边界**

所以它才会表现成：

- 不是在晚启碰撞瞬间就“优雅卡住”
- 而是常常继续往 `cmd=0.0` 闭合

这也进一步解释了为什么：

- 如果你想得到更自然的“闭合到物体后被挡住”效果
- 当前更可靠的做法仍然是：
  - **从 `pregrasp` 或至少 `grasp` 开始启碰撞**
