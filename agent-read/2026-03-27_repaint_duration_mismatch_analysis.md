# 2026-03-27: Why `process_repainted_headcam_with_wrist.py` outputs look much shorter than head-cam videos

## 1. Observed issue

The user ran:

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

The resulting sequences looked like they were only around 1 second long, much shorter than the original `head_cam_plan.mp4`.

This round only analyzes the cause. No code changes are made.

---

## 2. Main conclusion

The problem is not caused by a bad `review-json` filter and not by missing head-cam files.

The main causes are:

1. `process_repainted_headcam_with_wrist.py` aligns streams by the **minimum frame count**, not by true duration in seconds.
2. The downstream `processed_data` / HDF5 format does not preserve source fps, so if later visualization/export uses a higher default fps, the sequence looks even shorter, often around 1 second.

In other words, the effective length is limited by the retarget wrist / pose streams, not by the head-cam video.

---

## 3. Current alignment rule in the script

The script currently does:

```text
usable_len = min(T_pose, T_head, T_left, T_right)
states  = seq[:usable_len-1]
actions = seq[1:usable_len]
images  = frames[:usable_len-1]
```

Where:
- `T_pose` comes from `world_targets_and_status.npz`
- `T_head` comes from `target_with_original_head_cam_plan.mp4`
- `T_left` comes from `left_wrist_replay.mp4`
- `T_right` comes from `right_wrist_replay.mp4`

So it does **not** align by real time. It simply keeps the smallest frame count.

---

## 4. Actual inspection results

This round sampled the 27 `y` videos selected by the strict review mode.

### 4.1 The `review-json` itself is fine

- `y`: 27
- `m`: 11
- `n`: 18

So `expert_data_num=27` matches the 27 strict-usable videos. The filtering count is not the source of the issue.

### 4.2 Sampled per-episode lengths

Examples:

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

What this shows:
- head cam is usually `255~263` frames at `10 fps`, roughly `25~26` seconds
- left/right wrist are often only `34~57` frames at `5 fps`
- pose length is almost identical to wrist length
- therefore `usable_len` is capped by wrist/pose, not by head cam

---

## 5. Why it can feel like “only ~1 second”

There are two time scales involved.

### 5.1 First: the script already truncates head cam to just a few dozen frames

For `id=0`:
- original head cam: `263` frames @ `10 fps` → `26.3` seconds
- but the script keeps only the first `39` frames
- even if those 39 frames are interpreted at the original head-cam rate, that is only `3.9` seconds

### 5.2 Second: the HDF5 output does not store fps

`processed_data/.../episode_x.hdf5` stores:
- `action`
- `state`
- JPEG-encoded image sequences

It does not explicitly store the original source fps.

So if a later visualization/export step plays those frames at `30 fps` or another higher default fps:
- `38` frames → `38 / 30 = 1.27` seconds
- `40` frames → `1.33` seconds
- `34` frames → `1.13` seconds

That is why the result can subjectively look like it is “only around 1 second”.

So:
- the **small frame count** comes from the current minimum-frame-count truncation strategy
- the **even shorter time impression** comes from later playback using a higher fps

---

## 6. The outputs are not literally all exactly 1 second

Inspection of `processed_data/d_pour_blue-27` shows that not all episodes are `38` frames long.
Examples include:
- `38`
- `40`
- `55`
- `59`
- `77`

However, since these are still far smaller than the original `255~263` head-cam frames, they remain obviously shorter than the original head-cam videos.

---

## 7. What the current script really means semantically

`process_repainted_headcam_with_wrist.py` is currently closer to:

- “use head cam as a visual observation stream aligned to wrist/pose control sequences”

rather than:

- “preserve the full head-cam duration and then align wrist/pose to it”

In other words, the current assumption is:
- the training sample length should be determined by the lower-rate pose / wrist stream
- head cam is an accompanying image stream

If the desired behavior is instead:
- keep much more of the full head-cam duration
- or align by real time rather than by minimum frame count

then that becomes a **processing-strategy change**, not just a simple parameter tweak.

---

## 8. Final summary

The sequence is short mainly because:

1. `head cam` is long: about `255~263` frames at `10 fps`
2. `wrist replay` and `world_targets` are much shorter: often `34~57` frames at `5 fps`
3. the script currently truncates by `min(frame_count)`
4. so only a few dozen frames survive
5. if those frames are later viewed at `30 fps`, they look only `~1–2` seconds long

So the current behavior is consistent with the existing code logic. It is not caused by a single corrupted episode.

---

## 9. Commands used in this inspection

### 9.1 Count y/m/n in review-json

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

### 9.2 Sample head / wrist / pose lengths

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

### 9.3 Inspect generated HDF5 lengths

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
