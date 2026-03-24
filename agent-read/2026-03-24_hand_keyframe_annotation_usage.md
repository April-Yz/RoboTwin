# 2026-03-24 手部关键帧标注工具说明

## 目的

为目录

- `/home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis`

下的 `hand_vis*.mp4` / `hand_vis_gripper*.mp4` 提供一个有头交互式标注脚本，用于逐视频浏览并把关键帧统一维护到同一个 JSON 文件。

当前版本会把文字信息单独显示在图像上方的顶部信息栏，不再直接压在视频主体内容上。

脚本路径：

- `/home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/annotate_hand_keyframes.py`

默认输出 JSON：

- `/home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/hand_keyframes_all.json`

## 运行方式

```bash
python3 /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/annotate_hand_keyframes.py
```

如果需要调慢或调快自动播放速度：

```bash
python3 /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/annotate_hand_keyframes.py --delay-ms 150
```

## 交互按键

- `Space`：把当前帧切换为关键帧 / 取消关键帧
- `Left`：向前退一帧
- `Right`：向后一帧
- `r`：从当前视频第 0 帧重新播放
- `s`：播放 / 暂停切换
- `n`：保存当前视频结果并切换到下一个视频
- `p`：保存当前视频结果并切换到上一个视频
- `q` 或 `Esc`：保存当前结果并退出

你问到“如何切换到下一个视频”，当前实现就是：

- 按 `n`

窗口左上角也会显示这一行提示：

- `Keys: space toggle keyframe | left/right step | r replay | n next | p prev | q quit`

## JSON 维护方式

脚本不是每次新建一个分散的小文件，而是始终维护同一个总文件：

- `hand_keyframes_all.json`

每次发生以下事件时都会立即覆盖更新同一个 JSON：

- 空格新增 / 取消关键帧
- `n` 切到下一个视频
- `p` 切到上一个视频
- `q` / `Esc` 退出
- 当前视频播放到末尾后离开

因此这个 JSON 可以持续累积使用，中途退出后再次运行会继续读取已有标注。

## JSON 结构

当前版本使用可维护结构：

```json
{
  "_meta": {
    "schema_version": 2,
    "video_dir": "...",
    "output_json": "...",
    "current_video": "...",
    "total_videos": 122,
    "annotated_videos": 17,
    "completed_videos": 12,
    "updated_at": "2026-03-24 00:00:00",
    "description": "Incrementally maintained keyframe annotations for hand visualization videos."
  },
  "videos": {
    "hand_vis_0.mp4": {
      "video_path": ".../hand_vis_0.mp4",
      "keyframes": [10, 24, 39],
      "total_frames": 61,
      "fps": 10.0,
      "updated_at": "2026-03-24 00:00:00",
      "status": "done"
    }
  }
}
```

## JSON Schema 说明

### 顶层结构

- 顶层一定是一个 `dict/object`
- 顶层定义两个核心键：
  - `_meta`
  - `videos`

伪类型：

```text
root: {
  "_meta": MetaInfo,
  "videos": Dict[str, VideoAnnotation]
}
```

### `_meta` 字段定义

- `schema_version: int`
  - 当前固定为 `2`
- `video_dir: str`
  - 被标注视频目录的绝对路径
- `output_json: str`
  - 当前 JSON 文件本身的绝对路径
- `current_video: str`
  - 最近一次保存时所在的视频文件名，不含目录
  - 例如 `hand_vis_12.mp4`
- `total_videos: int`
  - 当前扫描到的视频总数
- `annotated_videos: int`
  - `videos` 中已有记录的视频数量
- `completed_videos: int`
  - `videos` 中 `status == "done"` 的数量
- `updated_at: str`
  - 最近一次整体写盘时间，格式 `%Y-%m-%d %H:%M:%S`
- `description: str`
  - 人类可读说明，不建议程序逻辑强依赖

### `videos` 字段定义

- `videos` 是一个字典
- key 是视频文件名，不是绝对路径
- value 是一个 `VideoAnnotation`

伪类型：

```text
videos: Dict[video_name: str, VideoAnnotation]
```

### `VideoAnnotation` 字段定义

每个视频记录都包含：

- `video_path: str`
  - 视频绝对路径
- `keyframes: List[int]`
  - 已选关键帧列表
  - 约束：
    - 0-based 帧号
    - 升序
    - 去重
    - 元素均为整数
- `total_frames: int`
  - 视频总帧数
- `fps: float`
  - 视频帧率
- `updated_at: str`
  - 该视频最后一次被修改的时间
- `status: str`
  - 当前只定义两个枚举值：
    - `done`
    - `in_progress`

### `status` 语义

- `done`
  - 代表用户通过 `n` 离开当前视频
  - 可以理解为这一轮已经确认完成
- `in_progress`
  - 代表视频还处于处理中
  - 例如按 `p` 返回上一个视频、按 `q` 中途退出、或只是刚做过若干关键帧增删

### 推荐解析方式

如果后续代码只想读取每个视频的关键帧，建议固定按下面路径取值：

1. 读取 JSON 顶层对象
2. 访问 `root["videos"]`
3. 遍历每个 `video_name`
4. 读取 `video_info["keyframes"]`

示例：

```python
import json

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

videos = data["videos"]
for video_name, video_info in videos.items():
    keyframes = video_info["keyframes"]
    status = video_info["status"]
```

### 排序与稳定性约定

- `videos` 在写回时按自然顺序排序
  - 例如 `hand_vis_2.mp4` 会排在 `hand_vis_10.mp4` 前面
- `keyframes` 写回时始终保持升序
- 后续代码建议把这些字段视为稳定接口：
  - `_meta.schema_version`
  - `_meta.current_video`
  - `videos`
  - `videos[video_name].keyframes`
  - `videos[video_name].status`

### 向后兼容

- 脚本兼容读取旧版平铺结构：

```json
{
  "hand_vis_0.mp4": {
    "keyframes": [10, 24, 39],
    "video_path": "...",
    "total_frames": 61,
    "fps": 10.0,
    "updated_at": "2026-03-24 00:00:00"
  }
}
```

- 如果读到旧结构，脚本会在下一次保存时自动升级为：
  - 顶层 `_meta`
  - 顶层 `videos`
  - 每个视频记录新增 `status`

### 字段语义速查

- `_meta.current_video` 表示最近一次保存时所在的视频
- `videos[video_name].keyframes` 是该视频当前选中的关键帧
- `status=done` 代表你是通过 `n` 离开该视频的
- `status=in_progress` 代表你只是中途保存、返回上一段、或直接退出

## 兼容性说明

- 当前运行环境如果缺少 `cv2`，脚本会直接提示安装 `opencv-python`。

## 下游集成

这个总 JSON 现在还会被抓取候选预览脚本直接读取：

- [render_anygrasp_ranked_preview.py](/home/zaijia001/ssd/RoboTwin/code_painting/render_anygrasp_ranked_preview.py)

新增了一种自动取帧模式：

- `--frame_selection_mode hand_keyframes_json`
- `--hand_keyframes_json /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/hand_keyframes_all.json`

这个模式下，预览脚本会：

1. 从 `anygrasp_dir` 或 `hand_npz` 推断当前视频 id
2. 到 `hand_keyframes_all.json` 里查找：
   - `videos["hand_vis_<id>.mp4"]["keyframes"]`
3. 自动生成请求帧：
   - 第 `0` 帧
   - 加上该视频里保存的关键帧

例如：

- `hand_vis_1.mp4 -> keyframes=[15, 27]`

则预览脚本会自动使用：

- `0 15 27`

这套模式和旧的手动模式并存：

- 手动模式仍然支持：
  - `--frames 1 22 -10`
- 自动模式则适合“按标注关键帧直接生成三帧最佳候选图”

示例命令：

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/render_anygrasp_ranked_preview.py \
  --anygrasp_dir /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results/d_pour_blue_1 \
  --replay_dir /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot/d_pour_blue_1 \
  --hand_npz /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/hand_detections_1.npz \
  --base_image_dir /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot/d_pour_blue_1/head_anygrasp_frames \
  --base_image_mode raw \
  --output_dir /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview/d_pour_blue_1_keyframes \
  --frame_selection_mode hand_keyframes_json \
  --hand_keyframes_json /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/hand_keyframes_all.json \
  --top_k 1 \
  --left_target_object cup \
  --right_target_object bottle \
  --draw_grasp_boxes 1
```

输出里会额外记录：

- `summary.json -> frame_selection`
- `warnings.json -> frame_selection`

这样后续批处理时可以追溯这一轮三帧到底是：

- 手动 `--frames`
- 还是来自 `hand_keyframes_all.json`

## 验证

本次只做了语法级验证：

```bash
python3 -m py_compile /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis/annotate_hand_keyframes.py
```

由于当前系统 Python 没有 `cv2`，本轮没有实际启动交互窗口做运行时验证。
