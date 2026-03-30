# 2026-03-27：`batch_head_cam_repaint_with_auto_pad.sh` 的 head 来源与当前 pi0 处理所用 wrist 来源是否一致

## 1. 用户问题

用户问：

- `batch_head_cam_repaint_with_auto_pad.sh` 这一步里，用的是哪个目录下面的 head 视频？
- 它和当前 `process_repainted_headcam_with_wrist.py` 使用的 wrist 路径是否一致？
- 如果一致，按印象保存帧数应该一样，为什么现在长度差这么多？

本次只做分析，不修改代码。

---

## 2. 直接结论

**不一致。**

更准确地说：

### `batch_head_cam_repaint_with_auto_pad.sh` 里的 head 来源
来自：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_<id>/head_cam_plan.mp4
```

即：
- AnyGrasp / planner 产出的 `head_cam_plan.mp4`
- 是 planner 执行视频，不是 hand-retarget replay 视频

### 当前 `process_repainted_headcam_with_wrist.py` 里的 wrist 来源
来自：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue/hand_detections_<id>/left_wrist_replay.mp4
/home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue/hand_detections_<id>/right_wrist_replay.mp4
```

即：
- hand-retarget pipeline 产出的左右 wrist replay

所以：
- **head 来自 planner pipeline**
- **wrist 来自 hand-retarget pipeline**
- 不是同一套渲染/执行视频流

---

## 3. `batch_head_cam_repaint_with_auto_pad.sh` 实际使用的 head 路径

脚本：
- `inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad.sh`

关键默认参数：

```bash
ROBOT_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3
```

每个 `id` 的机器人输入视频是：

```bash
robot_video="${ROBOT_ROOT}/${TASK_NAME}_${idx}/head_cam_plan.mp4"
```

所以它真正拿来做 repaint 的视频是：

```text
.../anygrasp_plan_keyframes_realoffset_batch_pure-v3/d_pour_blue_<id>/head_cam_plan.mp4
```

这条 head 视频后面被 `remove_anything_video_sam2_robot.py` 当作：
- `--input_video`

并最终生成：

```text
results_repaint/d_pour_blue/id_<id>_head_cam_arm_gripper_cup_bottle_pad_target/target_with_original_head_cam_plan.mp4
```

---

## 4. 当前 pi0 处理脚本实际使用的 wrist 路径

脚本：
- `policy/pi0/scripts/process_repainted_headcam_with_wrist.py`

当前命令里传的是：

```text
--retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue
--retarget-dir-template 'hand_detections_{id}'
```

因此它实际使用：

```text
.../output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue/hand_detections_<id>/left_wrist_replay.mp4
.../output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue/hand_detections_<id>/right_wrist_replay.mp4
.../output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue/hand_detections_<id>/world_targets_and_status.npz
```

---

## 5. 为什么用户会以为“如果一致，帧数应该一样”

这个印象其实对 **hand-retarget pipeline 内部** 是成立的。

我实际检查了 `hand_detections_0`：

```text
left_wrist_replay.mp4   39 frames @ 5 fps
right_wrist_replay.mp4  39 frames @ 5 fps
third_replay.mp4        39 frames @ 5 fps
zed_depth.mp4           39 frames @ 5 fps
zed_replay.mp4          39 frames @ 5 fps
```

也就是说，在 **同一个 hand-retarget 输出目录内部**：
- `zed_replay`
- `left_wrist_replay`
- `right_wrist_replay`
- `world_targets_and_status.npz`

它们的长度是基本一致的。

所以如果当前 head 也来自这个 retarget 目录下的 `zed_replay.mp4`，那你的直觉是对的：
- 它和 wrist / pose 的长度应该大致一致

但现在并不是这样。

---

## 6. 实际长度差异说明了什么

抽样检查若干 `y` episode：

```text
id=0
head_plan   263 frames @ 10 fps
zed_replay   39 frames @ 5 fps
left_wrist   39 frames @ 5 fps
right_wrist  39 frames @ 5 fps

id=8
head_plan   259 frames @ 10 fps
zed_replay   41 frames @ 5 fps
left_wrist   41 frames @ 5 fps
right_wrist  41 frames @ 5 fps

id=10
head_plan   263 frames @ 10 fps
zed_replay   35 frames @ 5 fps
left_wrist   35 frames @ 5 fps
right_wrist  35 frames @ 5 fps
```

这说明：

### planner head 和 retarget wrist 不是同一套时间采样

- `head_cam_plan.mp4`：
  - 来自 planner
  - 帧数很多，通常 `255~263`
  - fps 是 `10`
- `left/right_wrist_replay.mp4`：
  - 来自 hand-retarget
  - 帧数少很多，通常 `35~60`
  - fps 是 `5`

因此它们不但来源不同，连：
- 帧率
- 帧数
- 时间采样策略

都不同。

---

## 7. 为什么 planner head 会比 retarget replay 长很多

从现象上看，planner 的 `head_cam_plan.mp4` 不是简单的一帧对应一个 retarget 源帧。

它更像包含了：
- 插值执行过程
- 多个阶段（pregrasp / grasp / action）
- 执行动画中的中间帧
- 额外等待/稳定帧

而 retarget 目录下的：
- `zed_replay.mp4`
- `left_wrist_replay.mp4`
- `right_wrist_replay.mp4`

更接近“按 retarget 输入序列逐步回放”的结果，所以长度与 `world_targets_and_status.npz` 基本一致。

因此：
- planner head 是**高密度执行视频**
- retarget wrist 是**低密度监督/回放视频**

它们本来就不应该期待帧数一样。

---

## 8. 这对当前 pi0 处理意味着什么

现在这条链路实际上是在混用两套来源：

### 视觉 head
- 来自 planner pipeline 的 repaint 后 head cam

### 视觉 wrist + 控制 pose
- 来自 hand-retarget pipeline 的 wrist replay + world targets

因此 `process_repainted_headcam_with_wrist.py` 里用：

```text
usable_len = min(T_pose, T_head, T_left, T_right)
```

时，`T_head` 基本总是远大于其他三项。

所以最后保留下来的长度，实际上还是由：
- `T_pose`
- `T_left`
- `T_right`

决定，而不是由 planner head 决定。

---

## 9. 最关键的结论

### 一致的部分
在 hand-retarget 目录内部：
- `zed_replay`
- `left_wrist_replay`
- `right_wrist_replay`
- `world_targets_and_status.npz`

长度基本一致。

### 不一致的部分
这次用于 repaint 的 head：
- 不是 `hand_detections_<id>/zed_replay.mp4`
- 而是 planner 目录下的 `head_cam_plan.mp4`

所以你现在看到的长度不匹配，不是异常，而是因为：
- **上一步的 head 和当前用的 wrist 根本不是同源视频流**

---

## 10. 本次检查命令

### 10.1 读取 batch 脚本

```bash
read /home/zaijia001/ssd/inpainting_sam2_robot/script/batch_head_cam_repaint_with_auto_pad.sh
```

### 10.2 抽查 planner head 与 retarget wrist/zed 长度

```bash
cd /home/zaijia001/ssd/RoboTwin
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw
python - <<'PY'
import cv2, json
from pathlib import Path
review=Path('/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json')
d=json.load(open(review))
ids=sorted(int(k) for k,v in d['videos'].items() if v.get('label')=='y' or v.get('usable') is True)
robot_root=Path('/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3')
retarget_root=Path('/home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue')
for i in ids[:12]:
    head=robot_root/f'd_pour_blue_{i}'/'head_cam_plan.mp4'
    lw=retarget_root/f'hand_detections_{i}'/'left_wrist_replay.mp4'
    rw=retarget_root/f'hand_detections_{i}'/'right_wrist_replay.mp4'
    zed=retarget_root/f'hand_detections_{i}'/'zed_replay.mp4'
    def probe(p):
        cap=cv2.VideoCapture(str(p))
        fc=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0)
        cap.release()
        return fc,fps
    print(i,'head_plan',probe(head),'zed_replay',probe(zed),'left_wrist',probe(lw),'right_wrist',probe(rw))
PY
```

### 10.3 检查 retarget 目录内部各视频长度

```bash
cd /home/zaijia001/ssd/RoboTwin
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw
python - <<'PY'
import cv2
from pathlib import Path
base=Path('/home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue/hand_detections_0')
for p in sorted(base.glob('*.mp4')):
    cap=cv2.VideoCapture(str(p))
    print(p.name, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), float(cap.get(cv2.CAP_PROP_FPS) or 0))
    cap.release()
PY
```
