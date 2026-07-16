# 论文定性素材：独立标题网格与关键帧候选图

## 用途

维护 `/home/zaijia001/ssd/data/piper/paper_qualitative_assets` 中的论文视频网格和关键帧候选图。两类工具只读取已有视频、Foundation 图像和 Selection Strategy Audit V4 metadata；不会运行 IK、修改 OursV2，或覆盖原始数据。

## 参数模板（不可直接运行）

在候选图 JSON 中显式指定 episode 和关键帧，不应把路径硬编码进 Python：

```json
{
  "metadata_root": "<SELECTION_STRATEGY_COMPARE_V4>",
  "output_root": "<PAPER_ASSET_ROOT>/outputs/keyframe_candidates",
  "strategies": ["orientation", "fused", "top_score", "oursv2"],
  "episodes": [{"task": "<TASK>", "id": 0, "keyframes": [38, 78]}]
}
```

## 可直接运行：分离标题栏的视频网格

```bash
cd /home/zaijia001/ssd/data/piper/paper_qualitative_assets
python3 compose_pipeline_grid.py \
  --config pipeline_grid_expanded_dense_urdfmatch_v2_config.json --dry-run
python3 compose_pipeline_grid.py \
  --config pipeline_grid_expanded_dense_urdfmatch_v2_config.json
```

当前配置保持视频内容为每格 `480x270`，另在上方增加 `38 px` 标题栏；单格总尺寸是 `480x308`，4x5 输出是 `1920x1540`。旧的标题叠加版保存在 `outputs/pipeline_grid_expanded_dense_urdfmatch_v2_title_overlay_v1.mp4`。

## 可直接运行：关键帧候选图

```bash
cd /home/zaijia001/ssd/data/piper/paper_qualitative_assets
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  export_keyframe_candidate_images.py \
  --config keyframe_candidate_config.json --dry-run
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  export_keyframe_candidate_images.py \
  --config keyframe_candidate_config.json --overwrite
```

每张单策略 PNG 含左右两个面板，分别只画左/右手的 Selection Pose；双手事件在总标题标为 `BOTH`。坐标轴遵循 `X=红、Y=绿、Z=蓝`。Orientation/Fused/Top-score 显示所选 AnyGrasp candidate；OursV2 显示 synthetic human-retarget target，并明确标为 `HUMAN TARGET`，不能称为 AnyGrasp candidate。

## 验证

```bash
cd /home/zaijia001/ssd/data/piper/paper_qualitative_assets
python3 -m json.tool outputs/pipeline_grid_expanded_dense_urdfmatch_v2_manifest.json >/dev/null
ffprobe -v error -select_streams v:0 \
  -show_entries 'stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames:format=duration,size' \
  -of json outputs/pipeline_grid_expanded_dense_urdfmatch_v2.mp4
ffmpeg -hide_banner -v error \
  -i outputs/pipeline_grid_expanded_dense_urdfmatch_v2.mp4 -f null -
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 - <<'PY'
import cv2, json
from pathlib import Path
d = json.loads(Path('outputs/keyframe_candidates/manifest.json').read_text())
assert len(d['individual_images']) == 8
assert len(d['contact_sheets']) == 2
for item in d['individual_images']:
    assert cv2.imread(item['output']) is not None
print('candidate images:', len(d['individual_images']))
PY
```

pine2 默认 `python3` 没有 OpenCV；候选图导出和图像验证必须直接使用 `RoboTwin_bw` 的 Python。
