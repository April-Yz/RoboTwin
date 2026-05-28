# Piper Hand Origin Review Commands

## Purpose

Review raw D435 RGB frames under `/home/zaijia001/ssd/data/piper/hand/<TASK>/origin/episode*` before AnyGrasp / foundation preprocessing, and move bad episodes to the sibling `bad/` directory.

## Applicable Version

Current Piper hand raw-data review workflow. It does not read `foundation_input/`, `harmer_input/`, or AnyGrasp outputs.

## Prerequisites

- Use the `RoboTwin_openvla` conda environment, which has `cv2`.
- A graphical session is required for the OpenCV review window.
- Raw frames should live at `<TASK>/origin/episode*/camera/color/headD435/*.png`.

## Entrypoint

```text
/home/zaijia001/ssd/RoboTwin/code_painting/review_piper_hand_origin.py
```

## Command Format

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin && python code_painting/review_piper_hand_origin.py --task <TASK> --delay-ms 45
```

## Parameters

- `--task`: task name under `/home/zaijia001/ssd/data/piper/hand/`.
- `--data-root`: data root, default `/home/zaijia001/ssd/data/piper/hand`.
- `--origin-name`: source split directory name, default `origin`.
- `--bad-name`: bad-data destination directory name, default `bad`.
- `--start-id`: start reviewing from a numeric episode id.
- `--delay-ms`: autoplay delay per frame.
- `--frame-stride`: frame step during autoplay.
- `--dry-run`: write the review log without moving episode directories.

## Interactive Keys

- `b` / `d`: mark as bad and move the current episode from `origin/` to `bad/`.
- `g` / `n` / Enter: keep the current episode and go next.
- `p`: go back to the previous episode.
- Left/right arrows: step by frame.
- Space / `s`: pause or resume.
- `r`: replay the current episode from frame 0.
- `q` / Esc: save the log and quit.

## Inputs

```text
/home/zaijia001/ssd/data/piper/hand/<TASK>/origin/episode*/camera/color/headD435/*.png
```

## Outputs

```text
/home/zaijia001/ssd/data/piper/hand/<TASK>/origin_bad_review.json
/home/zaijia001/ssd/data/piper/hand/<TASK>/bad/episode*
```

## Three-Task Commands

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin && python code_painting/review_piper_hand_origin.py --task pnp_tray --delay-ms 45
```

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin && python code_painting/review_piper_hand_origin.py --task handover_bottle --delay-ms 45
```

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin && python code_painting/review_piper_hand_origin.py --task pnp_bread --delay-ms 45
```

## Related Code

- `code_painting/review_piper_hand_origin.py`

## Common Issues

- The default `python3` environment may not have `cv2`; run with the documented `RoboTwin_openvla` environment.
- After an episode is marked bad, the directory is moved and no longer remains under `origin/`.
