# Paper Qualitative Assets: Separate Headers and Keyframe Candidates

## Purpose

Maintain the paper video grid and keyframe-candidate images under `/home/zaijia001/ssd/data/piper/paper_qualitative_assets`. Both tools only read existing videos, Foundation images, and Selection Strategy Audit V4 metadata. They do not run IK, modify OursV2, or overwrite source data.

## Parameter template (not directly runnable)

Specify the episode and keyframes in JSON instead of hard-coding paths in Python:

```json
{
  "metadata_root": "<SELECTION_STRATEGY_COMPARE_V4>",
  "output_root": "<PAPER_ASSET_ROOT>/outputs/keyframe_candidates",
  "strategies": ["orientation", "fused", "top_score", "oursv2"],
  "episodes": [{"task": "<TASK>", "id": 0, "keyframes": [38, 78]}]
}
```

## Runnable: video grid with separate headers

```bash
cd /home/zaijia001/ssd/data/piper/paper_qualitative_assets
python3 compose_pipeline_grid.py \
  --config pipeline_grid_expanded_dense_urdfmatch_v2_config.json --dry-run
python3 compose_pipeline_grid.py \
  --config pipeline_grid_expanded_dense_urdfmatch_v2_config.json
```

The current config preserves `480x270` video content per cell and adds a separate `38 px` header above it. Each complete cell is `480x308`, so the 4x5 output is `1920x1540`. The former title-overlay version is preserved as `outputs/pipeline_grid_expanded_dense_urdfmatch_v2_title_overlay_v1.mp4`.

## Runnable: keyframe candidate images

```bash
cd /home/zaijia001/ssd/data/piper/paper_qualitative_assets
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  export_keyframe_candidate_images.py \
  --config keyframe_candidate_config.json --dry-run
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10 \
  export_keyframe_candidate_images.py \
  --config keyframe_candidate_config.json --overwrite
```

Each strategy PNG has separate left/right panels, each drawing only that arm's Selection Pose. A dual-arm event is marked `BOTH` in the global header. Axes follow `X=red, Y=green, Z=blue`. Orientation/Fused/Top-score display the selected AnyGrasp candidate. OursV2 displays a synthetic human-retarget target and is explicitly labeled `HUMAN TARGET`; it must not be described as an AnyGrasp candidate.

## Validation

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

The default `python3` on pine2 has no OpenCV. Use the `RoboTwin_bw` Python directly for candidate export and image validation.
