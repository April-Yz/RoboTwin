# 2026-03-27：`process_repainted_headcam_with_wrist.py` 产物时长偏短原因分析

## 1. 现象

用户使用下面这条命令：

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw

python scripts/process_repainted_headcam_with_wrist.py \
  d_pour_blue \
  "pour water" \
  27 \
  --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue \
  --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' \
  --head-video-name target_with_original_head_cam_plan.mp4 \
  --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue \
  --retarget-dir-template 'hand_detections_{id}' \
  --review-json /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json \
  --review-mode strict \
  --ignore-ids
```

观察到：
- 处理后的序列看起来只有约 1 秒
- 明显短于 `head_cam_plan.mp4` 的时长

本次只做分析，不改代码。

---

## 2. 核心结论

问题不是 `review-json` 过滤错了，也不是 `head cam` 文件缺失。

根因主要有两层：

1. `process_repainted_headcam_with_wrist.py` 是按**帧数最短值**裁切，不按真实时间长度对齐。
2. 下游 `processed_data` / HDF5 不保存原视频 fps，后续如果按较高 fps 播放这些帧，就会看起来更短，常见就是“约 1 秒”。

也就是说，**真正限制长度的不是 head cam，而是 retarget wrist / pose 序列长度**。

---

## 3. 这条链路里的对齐规则

脚本当前逻辑是：

```text
usable_len = min(T_pose, T_head, T_left, T_right)
states  = seq[:usable_len-1]
actions = seq[1:usable_len]
images  = frames[:usable_len-1]
```

这里：
- `T_pose` 来自 `world_targets_and_status.npz`
- `T_head` 来自 `target_with_original_head_cam_plan.mp4`
- `T_left` 来自 `left_wrist_replay.mp4`
- `T_right` 来自 `right_wrist_replay.mp4`

所以它不是“把各路视频按秒数同步”，而是“直接取最少的帧数”。

---

## 4. 实际检查结果

本次对 `video_review.json` 中 strict 模式下的 27 个 `y` 视频做了抽查。

### 4.1 `review-json` 统计是正常的

- `y`: 27 个
- `m`: 11 个
- `n`: 18 个

说明这次 `expert_data_num=27` 正好对应 27 个严格可用视频，不是筛选数出了问题。

### 4.2 抽样检查若干 episode

示例：

```text
id=0
head  = 263 frames @ 10 fps
left  = 39  frames @ 5 fps
right = 39  frames @ 5 fps
pose  = 39
usable_len = 39
action_len = 38

id=8
head  = 259 frames @ 10 fps
left  = 41 frames @ 5 fps
right = 41 frames @ 5 fps
pose  = 41
usable_len = 41
action_len = 40

id=10
head  = 263 frames @ 10 fps
left  = 35 frames @ 5 fps
right = 35 frames @ 5 fps
pose  = 35
usable_len = 35
action_len = 34
```

可以看到：
- head cam 通常是 `255~263` 帧，`10 fps`，约 `25~26` 秒
- left/right wrist 通常只有 `34~57` 帧，`5 fps`
- pose 长度和 wrist 基本一致
- 最终 usable_len 被 wrist / pose 卡住，而不是被 head cam 卡住

---

## 5. 为什么会“看起来像 1 秒”

这里有两个时间尺度：

### 5.1 第一层：脚本先把 head cam 截短成几十帧

例如 `id=0`：
- 原 head cam：`263` 帧 @ `10 fps` → `26.3` 秒
- 但脚本只保留前 `39` 帧
- 这 39 帧如果按 head cam 原始 `10 fps` 理解，也只剩 `3.9` 秒

### 5.2 第二层：后续 HDF5 通常不带 fps

`processed_data/.../episode_x.hdf5` 里保存的是：
- `action`
- `state`
- JPEG 编码后的图像序列

它没有显式保存源视频 fps。

因此后续如果有某个可视化/导出流程默认按 `30 fps` 或更高 fps 播放：
- `38` 帧 → `38 / 30 = 1.27` 秒
- `40` 帧 → `1.33` 秒
- `34` 帧 → `1.13` 秒

这就是为什么用户主观上会觉得“每次都只有 1 秒左右”。

所以：
- **短帧数**来自当前脚本的最短帧数裁切策略
- **更短的秒数观感**来自后续查看时使用了较高的默认 fps

---

## 6. 当前产物并不是全都严格只有 1 秒

检查 `processed_data/d_pour_blue-27` 后发现：
- 不是所有 episode 都是 `38` 帧
- 例如有：
  - `38`
  - `40`
  - `55`
  - `59`
  - `77`

但因为这些帧数整体仍然远小于 head cam 原始 `255~263` 帧，所以无论怎么看，都明显短于原始 head cam。

---

## 7. 这说明当前脚本的真实语义

`process_repainted_headcam_with_wrist.py` 当前更接近：

- “把 head cam 当成一段与 wrist / pose 对齐的视觉观察序列”

而不是：

- “完整保留 head cam 的全部时长，再去对齐 wrist / pose”

换句话说，它默认假设：
- 训练样本长度应该由 pose / wrist 那条低帧率控制流决定
- head cam 只是配套图像流

如果你希望：
- 更接近完整 head cam 时长
- 或按真实秒数对齐而不是按帧数最短值裁切

那就已经是**处理策略层**的问题，不是简单参数问题。

---

## 8. 本次结论总结

本次命令得到的序列偏短，主要原因是：

1. `head cam` 很长：大约 `255~263` 帧，`10 fps`
2. `wrist replay` 和 `world_targets` 很短：通常 `34~57` 帧，`5 fps`
3. 脚本当前按 `min(frame_count)` 裁切
4. 所以最终只留下几十帧
5. 如果这些帧后续按 `30 fps` 查看，就会显得只有约 `1~2` 秒

因此当前现象是**符合现有代码逻辑的**，不是某个单独 episode 损坏导致的异常。

---

## 9. 本次检查命令

### 9.1 检查 review-json 的 y/m/n 数量

```bash
cd /home/zaijia001/ssd/inpainting_sam2_robot
python3 - <<'PY'
import json
p='/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json'
with open(p) as f: d=json.load(f)
ys=[int(k) for k,v in d['videos'].items() if v.get('label')=='y' or v.get('usable') is True]
ms=[int(k) for k,v in d['videos'].items() if v.get('label')=='m' or v.get('usable')=='ambiguous']
ns=[int(k) for k,v in d['videos'].items() if v.get('label')=='n' or v.get('usable') is False]
print('y',len(ys),sorted(ys))
print('m',len(ms),sorted(ms))
print('n',len(ns),sorted(ns))
print('summary',d.get('summary'))
PY
```

### 9.2 抽查 head / wrist / pose 长度

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw
python - <<'PY'
import json, cv2, numpy as np
from pathlib import Path
review='/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json'
with open(review) as f:d=json.load(f)
ids=sorted(int(k) for k,v in d['videos'].items() if v.get('label')=='y' or v.get('usable') is True)
head_root=Path('/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue')
retarget_root=Path('/home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue')
for i in ids[:10]:
    head=head_root/f'id_{i}_head_cam_arm_gripper_cup_bottle_pad_target'/'target_with_original_head_cam_plan.mp4'
    lw=retarget_root/f'hand_detections_{i}'/'left_wrist_replay.mp4'
    rw=retarget_root/f'hand_detections_{i}'/'right_wrist_replay.mp4'
    npz=retarget_root/f'hand_detections_{i}'/'world_targets_and_status.npz'
    def probe(p):
        cap=cv2.VideoCapture(str(p))
        fc=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0)
        cap.release()
        return fc,fps
    h=probe(head); l=probe(lw); r=probe(rw)
    data=np.load(npz, allow_pickle=True)
    T=len(data['left_world_targets'])
    print(i,'head',h,'left',l,'right',r,'pose',T,'usable_len',min(h[0],l[0],r[0],T))
PY
```

### 9.3 检查生成后的 HDF5 长度

```bash
cd /home/zaijia001/ssd/RoboTwin/policy/pi0
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_bw
python - <<'PY'
import h5py, glob, os
paths=sorted(glob.glob('processed_data/d_pour_blue-27/episode_*/episode_*.hdf5'))
print('episodes',len(paths))
for p in paths[:5]:
    with h5py.File(p,'r') as f:
        print(os.path.basename(os.path.dirname(p)), 'action', f['action'].shape, 'state', f['observations/state'].shape, 'cam_high', f['observations/images/cam_high'].shape)
PY
```
