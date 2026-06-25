# H2O pi0 Training Data Conversion Commands

## Purpose

Record the input paths and conversion entrypoints for turning H2O human, pure replay, and AnyGrasp replay data into pi0 training data.

## Main Entrypoints

- Pure replay to pi0 HDF5: `policy/pi0/scripts/process_repainted_headcam_with_wrist.py`
- AnyGrasp planner replay to pi0 HDF5: `policy/pi0/scripts/process_repainted_planner_outputs.py`
- pi0 HDF5 to LeRobot: `policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_R1.py`
- Selected-episode subset from an existing LeRobot cache: `policy/pi0/scripts/subset_lerobot_episodes.py`
- pi0 HDF5 visual review: `policy/pi0/scripts/visualize_processed_hdf5_episode.py`

## Data Sources

- Original human head: `/home/zaijia001/ssd/data/piper/hand/<TASK>/harmer_input/rgb_<ID>.mp4`
- Original human directory: `/home/zaijia001/ssd/data/piper/hand/<TASK>/origin/`
- Manual review filter: `/home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/<TASK>/hand_keyframes_all.json`
- Pure replay repaint head: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/<TASK>/id_<ID>/final_repainted.mp4`
- Pure replay retarget: `/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/<TASK>/id<ID>_z005/`
- D435 pure replay retarget: `/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/<TASK>/id<ID>_d435_z005/`
- AnyGrasp planner: `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/<TASK>/foundation_input_<ID>/`
- AnyGrasp repaint head: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/anygrasp/<TASK>/id_<ID>/final_repainted.mp4`
- L16 planner: `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/<TASK>/foundation_input_<ID>/`
- L16 whitebg repaint head: `/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object/<TASK>/id_<ID>_l16_whitebg_human_object/final_repainted.mp4`
- L16 stack_cups B-variant repaint head: `/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative/stack_cups/id_<ID>_l16_whitebg_human_object/final_repainted.mp4`

## Pure Replay Conversion

`process_repainted_headcam_with_wrist.py` reads the repainted head video and these files from each retarget episode:

- `world_targets_and_status.npz`
- `left_wrist_replay.mp4`
- `right_wrist_replay.mp4`

The runnable template is appended in `COMMAND_LIBRARY.zh.md` section `L2`.

The same script also supports "original human head + pure replay action/wrist": set `--head-root .../harmer_input --head-dir-template '.' --head-video-name 'rgb_{id}.mp4'`, while action/state and wrist videos still come from `h2_pure/<TASK>/id<ID>_z005/`.

All three conversions should pass `--review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/<TASK>/hand_keyframes_all.json`. Episodes with `status=reject/discard/bad` in that JSON are skipped.

## AnyGrasp Conversion

`process_repainted_planner_outputs.py` reads the repainted head video and these files from each planner episode:

- `pose_debug.jsonl`
- `left_wrist_cam_plan.mp4`
- `right_wrist_cam_plan.mp4`

The current check shows `anygrasp_h2o_plan` has `head_cam_plan.mp4` and `pose_debug.jsonl`, but no `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4`. AnyGrasp data therefore cannot yet be fully converted into pi0 three-camera training data until planner replay also saves both wrist plan videos.

The D435 AnyGrasp candidate-preview path does not use the default `foundation_replay`; it uses:

- AnyGrasp: `/home/zaijia001/ssd/data/piper/hand/<TASK>/<TASK>_output/foundation_input_<ID>`, with `place_bread_basket` currently able to fall back to `place_bread_basket_output_old_cam`
- D435 replay: `/home/zaijia001/ssd/data/piper/hand/<TASK>/foundation_replay_d435/foundation_input_<ID>`
- HaMeR: `/home/zaijia001/ssd/data/piper/hand/<TASK>/harmer_output/hand_detections_<ID>.npz`
- Summary output: `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json`

`COMMAND_LIBRARY.zh.md` sections J0.1/J1.1 document the six-task D435 AnyGrasp availability check and human-keyframe-based candidate preview/summary generation commands.

## L16 Whitebg Repaint Conversion

L16 `L16_human_replay_clean/<TASK>/foundation_input_<ID>/` is planner-style output. Each episode uses:

- `pose_debug.jsonl`
- `left_wrist_cam_plan.mp4`
- `right_wrist_cam_plan.mp4`
- `head_cam_plan.mp4`

The current L16 directories do not contain `world_targets_and_status.npz`, so they should not use the D435 pure replay script `process_repainted_headcam_with_wrist.py`. L16 repaint videos should enter the training format through:

```text
I3.6/I3.6.1 repaint final
-> L9.2 process_repainted_planner_outputs.py
-> L10.7 convert_aloha_data_to_lerobot_R1.py
-> L11.2.5 subset_lerobot_episodes.py + zip/rclone
```

Default six-task dataset names:

```text
processed_data/h2o_<TASK>_l16_whitebg_repaint-120
local/h2o_<TASK>_l16_whitebg_repaint
local/h2o_<TASK>_l16_whitebg_repaint_25ep
```

If `stack_cups` uses the green-cup-protect B variant, the separate dataset names are:

```text
processed_data/h2o_stack_cups_l16_whitebg_b_points_negative-120
local/h2o_stack_cups_l16_whitebg_b_points_negative
local/h2o_stack_cups_l16_whitebg_b_points_negative_25ep
```

The runnable commands are documented in `COMMAND_LIBRARY.zh.md` sections L9.2, L10.7, and L11.2.5. Section I3.6.2 records the `stack_cups` B/C debug conclusions and the B-variant Stage-1/Stage-2 output paths.

## Check Commands

```bash
find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan -maxdepth 3 -type f \( -name 'head_cam_plan.mp4' -o -name 'pose_debug.jsonl' -o -name '*wrist*plan*.mp4' \) | sort | head -n 80
find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan -maxdepth 3 -type f -name '*wrist*plan*.mp4' | wc -l
```

## Three-Task Commands and Visualization

`COMMAND_LIBRARY.zh.md` sections `L5` / `L6` / `L8` / `L9` contain separate commands for:

- `pick_diverse_bottles`
- `place_bread_basket`
- `stack_cups`

`L5` is "original human head + pure replay action/wrist"; `L6` is "repaint robot head + pure replay action/wrist"; `L8` is "D435 visible-reinit head + D435 pure replay action/wrist"; `L9` is "AnyGrasp repaint head + planner action/wrist".

Note: `L9` still requires each planner episode to have `left_wrist_cam_plan.mp4` and `right_wrist_cam_plan.mp4`.

## Three Data Pipelines

`COMMAND_LIBRARY.zh.md` section L0 is the current recommended index:

- Human data: L5/L5.1, or L5.2 for the new three tasks; then convert to LeRobot with L10.4 or L10.5.
- Robot replay data: default wide replay uses L6/L6.1; six-task D435 visible-reinit uses I1/I1.1 -> I3.4/I3.5 -> L8.2 -> L10.6 -> L11.2.4.
- AnyGrasp replay data: L9/L9.1, requiring planner episodes with `pose_debug.jsonl` and both wrist plan videos.
- L16 whitebg repaint data: I3.6/I3.6.1 -> L9.2 -> L10.7 -> L11.2.5. If `stack_cups` uses the B variant, use the separate I3.6.2 `e0_robot_object_b_points_negative` output.

Core order:

```text
source videos/trajectories -> processed HDF5 -> LeRobot cache -> 25 episode subset/zip
```

For the new-three-task baseline using currently available D435 action/wrist:

```text
L5.2 -> L10.5 -> L11.1.3
```

For comparison against D435 robot replay on the new three tasks:

```text
I1.1 -> I3.5 -> L8.1 or L8.2 -> L10.6 -> L11.2.4
```

After generation, use `L7` to check:

- Count HDF5 episodes under each `processed_data/<dataset>`.
- Read the first few HDF5 files and print `observations/state`, `action`, and three-camera frame counts.
- Run `visualize_processed_hdf5_episode.py` to make a review mp4 combining `cam_high`, `cam_left_wrist`, and `cam_right_wrist`.

## LeRobot Conversion

`COMMAND_LIBRARY.zh.md` section `L10` records LeRobot conversion commands for the 3 currently usable data modes x 3 tasks:

- L5 data: original human head + pure replay action/wrist.
- L6 data: default wide robot repaint head + pure replay action/wrist.
- L8 data: D435 visible-reinit robot repaint head + D435 pure replay action/wrist.

L9 AnyGrasp mode is not included in the formal 3x3 conversion yet because it still depends on missing planner wrist videos.

The downstream conversion uses `examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --use-wrist --mode video` because the current H2O processed HDF5 stores `/observations/state`, while the normal branch of `convert_aloha_data_to_lerobot_robotwin.py` still reads `/observations/qpos`.

## LeRobot Episode Subsets

When a full LeRobot cache already exists, for example:

```text
/home/zaijia001/.cache/huggingface/lerobot/local/h2o_pick_diverse_bottles_human_head_pure_action
```

use `subset_lerobot_episodes.py` to copy selected episodes without reconverting from HDF5. The script accepts specs such as `0-24` and `0,1-5,7`; it deduplicates, sorts by the old episode id, and reindexes the output to continuous `episode_000000` through `episode_0000NN`.

First 25 episodes:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && uv run python scripts/subset_lerobot_episodes.py --source local/h2o_pick_diverse_bottles_human_head_pure_action --output-repo-id local/h2o_pick_diverse_bottles_human_head_pure_action_25ep --episodes '0-24' --overwrite
```

Arbitrary episode selection:

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && uv run python scripts/subset_lerobot_episodes.py --source local/h2o_pick_diverse_bottles_human_head_pure_action --output-repo-id local/h2o_pick_diverse_bottles_human_head_pure_action_subset --episodes '0,1-5,7' --overwrite
```

Output check:

```bash
ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_pick_diverse_bottles_human_head_pure_action_25ep; python3 - <<'PY'
import json
from pathlib import Path
root = Path("/home/zaijia001/.cache/huggingface/lerobot/local/h2o_pick_diverse_bottles_human_head_pure_action_25ep")
print(json.load(open(root / "meta/info.json"))["total_episodes"])
print((root / "meta/episodes.jsonl").read_text().splitlines()[:30])
print(sorted(p.name for p in (root / "data/chunk-000").glob("episode_*.parquet"))[:30])
PY
find "$ROOT/videos/chunk-000" -maxdepth 2 -type f -name 'episode_*.mp4' | sort | head -n 20
```

### Robot Replay / AnyGrasp 25-Episode Subsets

`COMMAND_LIBRARY.zh.md` section L11.1 now contains 25-episode subset commands for six robot datasets:

- `local/h2o_pick_diverse_bottles_pure_repaint`
- `local/h2o_place_bread_basket_pure_repaint`
- `local/h2o_stack_cups_pure_repaint`
- `local/h2o_pick_diverse_bottles_anygrasp_repaint`
- `local/h2o_place_bread_basket_anygrasp_repaint`
- `local/h2o_stack_cups_anygrasp_repaint`

The run order is: confirm that the full LeRobot caches already exist, generate the `_25ep` subsets, zip `robot_replay_3task_25ep.zip` and `robot_anygrasp_3task_25ep.zip`, then check the upload destination with `rclone --dry-run` before the real upload.

`COMMAND_LIBRARY.zh.md` section L11.2 further covers the six H2O tasks used by the FoundationPose sections:

- `pick_diverse_bottles`
- `place_bread_basket`
- `stack_cups`
- `handover_bottle`
- `pnp_bread`
- `pnp_tray`

L11.2 creates `_25ep` subsets for both `pure_repaint` and `anygrasp_repaint` across all six tasks, then packages them as:

- `robot_replay_6task_25ep.zip`
- `robot_anygrasp_6task_25ep.zip`

Note: L11.2 can only subset LeRobot caches that already exist. The full order is:

1. L5.1: create six-task `human_head_pure_action` processed HDF5.
2. L6.1: create six-task `pure_repaint` processed HDF5.
3. L9.1: create six-task `anygrasp_repaint` processed HDF5.
4. L10.4: convert L5.1/L6.1/L9.1 outputs into `local/h2o_<TASK>_human_head_pure_action`, `local/h2o_<TASK>_pure_repaint`, and `local/h2o_<TASK>_anygrasp_repaint`.
5. L11/L11.2: create `_25ep` subsets from those LeRobot caches.

The six-task prompts follow the complete descriptions in L5.1; do not rely only on the LeRobot converter's `--task`.

Additional note: for the new three tasks `handover_bottle / pnp_bread / pnp_tray`, the current available retarget outputs are under `human_replay/h2_pure_d435/<TASK>/id<ID>_d435_z005`; `human_replay/h2_pure/<TASK>/id<ID>_z005` is absent. L5.1 will therefore skip those tasks. `COMMAND_LIBRARY.zh.md` section L5.2 documents a human-head processing command that uses the existing D435 pure replay action/wrist outputs.

The post-L10.5 subset step must not use `local/h2o_<TASK>_pure_repaint`; `pure_repaint` belongs to L6/L6.1 robot replay data. The source repos for the new-three-task human-head D435 baseline are:

- `local/h2o_handover_bottle_human_head_pure_d435_action`
- `local/h2o_pnp_bread_human_head_pure_d435_action`
- `local/h2o_pnp_tray_human_head_pure_d435_action`

`COMMAND_LIBRARY.zh.md` section L11.1.3 records the dedicated post-L10.5 subset command. It reads `instructions.json/source_episode_id` from the processed data, excludes original bad ids `0,7,12,29` for `handover_bottle` and `0,1,2,3,4,5,6,22,70` for `pnp_bread`, then keeps the first 25 usable episodes and reindexes the output to `0..24`.

For comparison against D435 robot replay, the recommended order for the new three tasks is:

1. L5.2: real human head + D435 action/wrist baseline.
2. I1.1: Stage-1 human-hand removal backgrounds for the new three tasks.
3. I3.5: D435 visible-reinit robot repaint for the new three tasks.
4. L8.1 or L8.2: convert I3.5 outputs into `h2o_<TASK>_pure_d435_visible_reinit-120` processed HDF5.
5. L10.6: convert the D435 robot replay processed HDF5 into `local/h2o_<TASK>_pure_d435_visible_reinit`.
6. L11.2.4: subset the D435 robot replay LeRobot caches to `_25ep` and package them.

Note: L6.1 is the default-wide `h2_pure` path, not the D435 path. The new three tasks currently do not have `human_replay/h2_pure/<TASK>/id<ID>_z005`, and they also do not have default-wide `results_repaint_piper_h2/e0_robot/<TASK>/id_<ID>/final_repainted.mp4`; running L6.1 on those tasks therefore skips all episodes and ends with `No usable episodes were processed`. D435 robot replay should use `h2_pure_d435` together with `results_repaint_piper_h2_d435_sam3_visible_reinit`.

If L8.2 produces only a few episodes for the new three tasks, check the I1.1 Stage-1 BG count and the I3.5 D435 final count before rerunning L8.2. L8.2 only reads:

```text
results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/<TASK>/id_<ID>_d435/final_repainted.mp4
human_replay/h2_pure_d435/<TASK>/id<ID>_d435_z005/*
```

It does not generate missing `final_repainted.mp4` files. `COMMAND_LIBRARY.zh.md` section I1.1.1 records a "fill missing Stage-1 BG only" command, and I3.5 records a 0..80 resume command with `--overwrite 0` for filling missing D435 repaint finals before returning to L8.2.

`d435_final` means:

```text
results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/<TASK>/id_<ID>_d435/final_repainted.mp4
```

It is generated by I3.5 through `batch_visible_reinit_d435_repaint.py`. This machine currently does not have `Grounded_SAM_3`, so the script starts with `[backend] SAM=sam2, DINO=dino2`; it also prints `loading DINO once` and `loading SAM image predictor once`, meaning the batch process loads the checkpoints once and then loops over task/id jobs. I3.5 now also records a SAM2/DINO2 fallback batch command for first filling each new task to at least 25 final videos.

L11.2.4 now subsets D435 robot replay `_25ep` datasets with the same `source_episode_id` alignment used by the human replay path: it reads `processed_data/h2o_<TASK>_pure_d435_visible_reinit-120/episode_*/instructions.json`, excludes original bad ids `0,7,12,29` for `handover_bottle` and `0,1,2,3,4,5,6,22,70` for `pnp_bread`, then fills 25 LeRobot episode indices. The output is still reindexed to `0..24`, but those original bad episodes are not included.

### Task Prompt Source

In the current `convert_aloha_data_to_lerobot_R1.py`, `--task "..."` does not override prompts already stored in processed episodes. The converter reads each episode's `instructions.json`:

```text
/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/<DATASET>/episode_*/instructions.json
```

Recommended order:

1. Set the prompt via `INSTRUCTION="..."` when generating HDF5 processed data with `process_repainted_headcam_with_wrist.py` or `process_repainted_planner_outputs.py`.
2. If processed data already exists but has not been converted to LeRobot, batch-edit `episode_*/instructions.json` first.
3. If the LeRobot cache already exists and only the metadata needs a quick fix, use L12 to replace `meta/tasks.jsonl` and `meta/episodes.jsonl`.

## LeRobot Task Text Fix

In the current `convert_aloha_data_to_lerobot_R1.py`, the `--task` argument is not the main source of the text stored in the dataset frames. The script reads each processed episode's `instructions.json`; therefore, if those files still contain the old text, the generated LeRobot cache will also keep the old task text.

To fix an already generated cache, replace the text in `meta/tasks.jsonl` and `meta/episodes.jsonl`. The parquet files only store `task_index`, not the task string, so they do not need to be edited.

```bash
ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_pick_diverse_bottles_human_head_pure_action; OLD='pick diverse bottles'; NEW='pick up one bottle with one arm, and pick up another bottle with the other arm.'; cp "$ROOT/meta/tasks.jsonl" "$ROOT/meta/tasks.jsonl.bak" && cp "$ROOT/meta/episodes.jsonl" "$ROOT/meta/episodes.jsonl.bak" && OLD="$OLD" NEW="$NEW" perl -0pi -e 's/\Q$ENV{OLD}\E/$ENV{NEW}/g' "$ROOT/meta/tasks.jsonl" "$ROOT/meta/episodes.jsonl"
```

```bash
ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_pick_diverse_bottles_human_head_pure_action; sed -n '1,5p' "$ROOT/meta/tasks.jsonl"; sed -n '1,5p' "$ROOT/meta/episodes.jsonl"
```


## L16 Six-Task Robot+Object Repaint Commands

`COMMAND_LIBRARY.zh.md` sections I3.5.2/I3.5.3 document the robot/object prompt route for L16 Human Replay outputs. This flow does not reuse the old I1 "human hand only" backgrounds. Instead, it writes `results_repaint_piper_h2_l16/stage1_human_object`, removes both human hands and task objects with task-specific prompts, then visible-reinit repaints robot+object pixels from `L16_human_replay_clean/<TASK>/foundation_input_<ID>/head_cam_plan.mp4`.

Recommended order: run I3.5.2 first for five debug IDs per task and inspect `w_box_head_cam_plan.mp4`, `w_mask_head_cam_plan.mp4`, and `final_repainted.mp4`; after confirming no background leakage or object ghosting, run the I3.5.3 full batch command.

`COMMAND_LIBRARY.zh.md` section I3.6 now documents the added white-background SAM plus inverted-mask comparison route. It defaults to the I3.5.2/I3.5.3 human+object Stage-1 background, prompts for the white background in each L16 source video, then uses the saved inverted mask frames to externally compose `final_repainted.mp4`; output length follows the robot/mask frames and shorter Stage-1 backgrounds are sampled proportionally. For `stack_cups`, the Stage-1 prompt removes the generic `cups` term and keeps only `left light pink cup, right dark red cup` to avoid removing the green cup. Its output root is `results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object`; inspect `w_box_head_cam_plan.mp4`, `w_mask_head_cam_plan.mp4`, `mask_head_cam_plan.mp4`, and `mask_head_cam_plan/000000.jpg` for SAM/inversion quality.

## L16 Visualization Montage: HaMeR / Foundation / L16 / Repaint

Added script: `code_painting/make_l16_repaint_montage.py`. It horizontally stitches each task/id's HaMeR hand visualization, Foundation object replay, and L16 `head_cam_plan.mp4`. When Stage-1 inpaint and final repaint outputs already exist, they are automatically appended as the fourth and fifth panels.

Core inputs:

- HaMeR: `/home/zaijia001/ssd/data/piper/hand/<TASK>/harmer_output/hand_vis_gripper_<ID>.mp4`
- Foundation replay: prefers `foundation_replay_d435/foundation_input_<ID>/head_cam_replay.mp4`, falling back to `foundation_replay`
- L16 plan: `code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/<TASK>/foundation_input_<ID>/head_cam_plan.mp4`
- Optional Stage 1: `/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object/<TASK>/id_<ID>/stage1_human_inpaint/removed_w_mask_rgb_<ID>.mp4`
- Optional final repaint: `/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_visible_reinit/e0_robot_object/<TASK>/id_<ID>_l16/final_repainted.mp4`

Output: `code_painting/l16_repaint_montage/<TASK>/id_<ID>/compare_hamer_foundation_l16_repaint_<TASK>_id<ID>.mp4`, with a sibling JSON manifest recording the videos actually used.

Validated command:

```bash
tmux new-session -d -s l16_vis_id0 'cd /home/zaijia001/ssd/RoboTwin && python3 /home/zaijia001/ssd/RoboTwin/code_painting/make_l16_repaint_montage.py --task pick_diverse_bottles --id 0 --overwrite'
```

Batch notes: `pick_diverse_bottles/place_bread_basket/stack_cups/pnp_tray` can use `--ids 0-4`; `handover_bottle` should start with `--ids 1-5`; `pnp_bread` should start with `--ids 7-11` because its L16 outputs do not include id0-id6.


## L16 White-Background Inverted-Mask Per-Task Entry Points

`COMMAND_LIBRARY.zh.md` section I3.6.1 adds two task-level scripts:

- `code_painting/run_l16_stage1_human_object_task.sh`: fill or rerun Stage-1 human+object inpaint for one `TASK`.
- `code_painting/run_l16_whitebg_repaint_task.sh`: run white-background SAM plus inverted-mask repaint for one `TASK`; compose output length follows robot/mask frames, and shorter BG videos are sampled proportionally.

Use these scripts to run the five non-`stack_cups` tasks on separate GPUs. Keep `stack_cups` separate until its Stage-1 prompt/dilation is verified not to remove the green cup.

### stack_cups Green-Cup Protection Debug Entry Point

Added scripts:

- `code_painting/l16_stack_cups_debug_variants.py`
- `code_painting/run_l16_stack_cups_debug_variants.sh`

Purpose: run Stage-1 mask/inpaint debug only for `stack_cups id_0..4` without overwriting formal `stage1_human_object` outputs. The four variants write to:

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/A_protect_dino/stack_cups/id_<ID>/stage1_human_inpaint/
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/B_points_negative/stack_cups/id_<ID>/stage1_human_inpaint/
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/C_hsv_green_protect/stack_cups/id_<ID>/stage1_human_inpaint/
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/D_tight_dino/stack_cups/id_<ID>/stage1_human_inpaint/
```

Variant meanings:

- `A_protect_dino`: create a DINO/SAM2 remove mask, create a `green cup` protect mask, then use `remove - protect`. The current debug result is unstable and this is treated as a wrong route because both remove/protect masks still depend on DINO and inherit large-box errors.
- `B_points_negative`: bypass DINO and use fixed positive points for the two red cups and hands, with the green cup center as a negative point. The current `id_0..4` debug result is usable and this is the preferred route.
- `C_hsv_green_protect`: subtract an HSV green-region protect mask from the DINO/SAM2 remove mask. The current `id_0..4` debug result is also usable and serves as B's fallback.
- `D_tight_dino`: stricter DINO prompt/threshold baseline with no green protection. The current debug result fails and this is treated as a wrong route because DINO can still return a large box covering the green cup.

Inspect these files in each output directory:

```text
w_box_rgb_<ID>.mp4
w_mask_rgb_<ID>.mp4
removed_w_mask_rgb_<ID>.mp4
w_protect_mask_rgb_<ID>.mp4   # A/C only
debug_summary.json
```

Run command:

```bash
tmux new-session -d -s l16_stack_debug_variants_gpu1 'GPU=1 IDS="0 1 2 3 4" MAX_FRAMES=300 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_stack_cups_debug_variants.sh'
```

Run only the B variant for full Stage-1:

```bash
IDS=$(find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/stack_cups -path '*/head_cam_plan.mp4' 2>/dev/null | sed 's#.*/foundation_input_\([0-9]*\)/head_cam_plan.mp4#\1#' | sort -n | tr '\n' ' ')
tmux new-session -d -s l16_stack_B_stage1_gpu1 "IDS=\"$IDS\" GPU=1 VARIANTS=\"B_points_negative\" MAX_FRAMES=300 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_stack_cups_debug_variants.sh"
```

The B Stage-2 job reads the `B_points_negative` backgrounds and writes to a separate output root:

```bash
tmux new-session -d -s l16_stack_B_stage2_after_s1_gpu1 'while tmux has-session -t l16_stack_B_stage1_gpu1 2>/dev/null; do sleep 60; done; TASK=stack_cups GPU=1 OVERWRITE=1 STAGE1=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/B_points_negative OUTROOT=/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_whitebg_repaint_task.sh'
```

B final videos:

```text
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative/stack_cups/id_<ID>_l16_whitebg_human_object/final_repainted.mp4
```
