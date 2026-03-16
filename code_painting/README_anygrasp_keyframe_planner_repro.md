# AnyGrasp Keyframe Planner 复现说明

这份文档是给不看代码的人用的。目标是说明这套流程现在在做什么、视频里分别能看到什么、当前有哪些处理规则、以及不同规则下怎么运行。

## 1. 这套流程解决什么问题

输入三类数据：
- AnyGrasp 在某个视频上输出的抓取候选
- RoboTwin replay 出来的物体运动结果
- 人手检测 `hand_detections_<id>.npz` 里的左右 gripper 朝向

输出一条两关键帧的机器人规划 demo：
- 关键帧 1: 对准抓取位姿
- 关键帧 2: 对准操作位姿
- 机器人按照 `pregrasp -> grasp -> close_gripper -> action` 执行

---

## 2. 最小输入

### AnyGrasp 结果
例如：
```
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results/d_pour_blue_1/grasps/grasp_000001.json
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results/d_pour_blue_1/grasps/grasp_000022.json
```

### 物体 replay 结果
例如：
```
/home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue/d_pour_blue_1/multi_object_world_poses.npz
```

### 人手朝向结果
例如：
```
/home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/hand_detections_1.npz
```

---

## 3. 输出文件分别表示什么

每个视频目录下会生成：
- `debug_selection_preview.mp4`
- `debug_execution_preview.mp4`
- `head_cam_plan.mp4`
- `plan_summary.json`

### `debug_selection_preview.mp4`
用途：检查关键帧候选有没有选对。

视频里会看到：
- replay 的物体运动
- 关键帧 1 和 22 的目标轴
- 当前关键帧对应的所有候选夹爪
- 候选编号

这个视频主要用来回答：
- AnyGrasp 候选是不是压根就偏了
- 左手和右手各自更可能抓哪一个
- 最终被选中的红色候选是不是合理

### `debug_execution_preview.mp4`
用途：检查机器人是不是先到位再进入下一段。

视频里会看到：
- 机器人慢速执行 `pregrasp -> grasp -> action`
- 当前正在追踪的关键帧目标
- 当前阶段对应的候选抓取可视化
- 候选编号

这个视频主要用来回答：
- 机器人是否真的到达关键帧 1 再去关键帧 2
- 轨迹是不是太快
- 最终选中的那个候选是不是可达

### `head_cam_plan.mp4`
用途：看最终结果，不强调 debug 标记。

### `plan_summary.json`
用途：查文字版结果。

里面重点看：
- `selected_arm`
- `selected_object`
- `selected_candidates`
- `arm_debugs`
- `stages.pregrasp/grasp/action`

---

## 4. 视频里的可视化规则

### 颜色
- 绿色夹爪: 当前关键帧显示出来的全部候选
- 蓝色夹爪: 左右手各自排序后保留下来的 top 候选
- 红色夹爪: 最终被选中的候选

### 左右手区分
因为背景是白色，原来的白色标记不明显，现在改成黑色：
- 左手候选: 夹爪上方一个黑色小标记
- 右手候选: 夹爪下方一个黑色小标记

### 编号
现在每个候选都会画编号：
- 绿色小编号 `12`: 表示原始候选 `candidate_idx=12`
- 蓝色数字：左手排序结果里的候选编号
- 橙色数字：右手排序结果里的候选编号
- 红色数字：最终选中的候选编号

---

## 5. 当前处理规则

这套流程支持两类模式。

### 模式 A: 正常筛选模式
规则：
1. 关键帧 1 和 22 都读取 AnyGrasp 候选
2. 读取人手左右 gripper 朝向
3. 每只手各自对候选做朝向匹配
4. 可选地再叠加物体约束和距离约束
5. 选择总成本更低的一只手
6. 关键帧 1 和 22 用同一只手、同一物体的候选

适合：
- 想直接得到自动选择结果
- 已经确认候选分布基本靠谱

### 模式 B: 放宽约束的 debug 模式
规则：
1. 不强制左手抓 cup / 右手抓 bottle
2. 不强制候选必须靠近物体中心
3. 先把所有候选都画出来
4. 用视频先看 AnyGrasp 候选本身有没有问题

适合：
- 现在这种“先查 bug”阶段
- 怀疑规则把好候选过滤掉了

---

## 6. 最常用命令

### 6.1 单个视频，正常自动筛选
```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes \
  --ids 1 \
  --keyframes 1 22 \
  --lighting_mode front_no_shadow \
  --planner_backend urdfik
```

### 6.2 单个视频，放宽约束做 debug
这个模式更适合先看候选本身：
```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes \
  --ids 1 \
  --keyframes 1 22 \
  --lighting_mode front_no_shadow \
  --planner_backend urdfik \
  --enforce_target_object_constraint 0 \
  --enforce_candidate_distance_constraint 0 \
  --debug_show_all_candidates 1 \
  --debug_candidate_top_k 5 \
  --debug_common_candidate_top_k 0 \
  --save_debug_preview 1 \
  --save_debug_execution_preview 1
```

### 6.3 打开 viewer，边看边调
```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes \
  --ids 1 \
  --keyframes 1 22 \
  --lighting_mode front_no_shadow \
  --planner_backend urdfik \
  --enforce_target_object_constraint 0 \
  --enforce_candidate_distance_constraint 0 \
  --debug_show_all_candidates 1 \
  --debug_candidate_top_k 5 \
  --debug_common_candidate_top_k 0 \
  --save_debug_preview 1 \
  --save_debug_execution_preview 1 \
  --enable_viewer 1 \
  --viewer_frame_delay 0.02 \
  --viewer_wait_at_end 1
```

### 6.4 批处理所有视频
```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes \
  --keyframes 1 22 \
  --lighting_mode front_no_shadow \
  --planner_backend urdfik
```

---

## 7. 现在最推荐的 debug 顺序

1. 先看 `debug_selection_preview.mp4`
   - 重点看绿色、蓝色、橙色、红色数字
   - 判断候选本身是否合理
2. 再看 `debug_execution_preview.mp4`
   - 判断机器人是不是先到位再继续
3. 最后看 `plan_summary.json`
   - 确认最终到底选了哪个 `candidate_idx`

如果你想人工指定候选，最先需要用到的信息就是：
- 关键帧 1 你想选哪个编号
- 关键帧 22 你想选哪个编号
- 对应是左手还是右手

这三个信息现在都能直接从 debug 视频里读出来，不需要回头看源码。
