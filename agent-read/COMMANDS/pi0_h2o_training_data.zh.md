# H2O pi0 训练数据转换命令

## 用途

记录 H2O 人手数据、pure replay 数据、AnyGrasp replay 数据进入 pi0 训练数据格式时的输入路径和转换入口。

## 关键入口

- pure replay 转 pi0 HDF5：`policy/pi0/scripts/process_repainted_headcam_with_wrist.py`
- AnyGrasp planner 转 pi0 HDF5：`policy/pi0/scripts/process_repainted_planner_outputs.py`
- pi0 HDF5 转 LeRobot：`policy/pi0/examples/aloha_real/convert_aloha_data_to_lerobot_R1.py`
- 已生成 LeRobot 数据抽取 episode 子集：`policy/pi0/scripts/subset_lerobot_episodes.py`
- pi0 HDF5 可视化检查：`policy/pi0/scripts/visualize_processed_hdf5_episode.py`

## 数据来源

- 原始人手 head：`/home/zaijia001/ssd/data/piper/hand/<TASK>/harmer_input/rgb_<ID>.mp4`
- 原始人手目录：`/home/zaijia001/ssd/data/piper/hand/<TASK>/origin/`
- 人工标注过滤：`/home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/<TASK>/hand_keyframes_all.json`
- pure replay repaint head：`/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/e0_robot/<TASK>/id_<ID>/final_repainted.mp4`
- pure replay retarget：`/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure/<TASK>/id<ID>_z005/`
- D435 pure replay retarget：`/home/zaijia001/ssd/RoboTwin/code_painting/human_replay/h2_pure_d435/<TASK>/id<ID>_d435_z005/`
- AnyGrasp planner：`/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan/<TASK>/foundation_input_<ID>/`
- AnyGrasp repaint head：`/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2/anygrasp/<TASK>/id_<ID>/final_repainted.mp4`
- L16 planner：`/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/<TASK>/foundation_input_<ID>/`
- L16 whitebg repaint head：`/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object/<TASK>/id_<ID>_l16_whitebg_human_object/final_repainted.mp4`
- L16 stack_cups B 方案 repaint head：`/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative/stack_cups/id_<ID>_l16_whitebg_human_object/final_repainted.mp4`

## pure replay 转换

`process_repainted_headcam_with_wrist.py` 读取 repaint 后 head 视频，并从 retarget episode 中读取：

- `world_targets_and_status.npz`
- `left_wrist_replay.mp4`
- `right_wrist_replay.mp4`

命令模板已经追加到 `COMMAND_LIBRARY.zh.md` 的 `L2`。

同一个脚本也支持“原始人手 head + pure replay action/wrist”：设置 `--head-root .../harmer_input --head-dir-template '.' --head-video-name 'rgb_{id}.mp4'`，action/state 和 wrist 仍然来自 `h2_pure/<TASK>/id<ID>_z005/`。

三类转换都应传 `--review-json /home/zaijia001/ssd/RoboTwin/code_painting/h2o_manual_review/<TASK>/hand_keyframes_all.json`。该 JSON 中 `status=reject/discard/bad` 的 episode 会被跳过。

## AnyGrasp 转换

`process_repainted_planner_outputs.py` 读取 repaint 后 head 视频，并从 planner episode 中读取：

- `pose_debug.jsonl`
- `left_wrist_cam_plan.mp4`
- `right_wrist_cam_plan.mp4`

当前检查显示 `anygrasp_h2o_plan` 下已有 `head_cam_plan.mp4` 和 `pose_debug.jsonl`，但没有 `left_wrist_cam_plan.mp4` / `right_wrist_cam_plan.mp4`。因此 AnyGrasp 数据暂时不能完整转换为 pi0 三相机训练数据，除非先让 planner replay 同步保存左右 wrist plan 视频。

D435 版本的 AnyGrasp 候选预览不走默认 `foundation_replay`，而是使用：

- AnyGrasp：`/home/zaijia001/ssd/data/piper/hand/<TASK>/<TASK>_output/foundation_input_<ID>`，`place_bread_basket` 当前可 fallback 到 `place_bread_basket_output_old_cam`
- D435 replay：`/home/zaijia001/ssd/data/piper/hand/<TASK>/foundation_replay_d435/foundation_input_<ID>`
- HaMeR：`/home/zaijia001/ssd/data/piper/hand/<TASK>/harmer_output/hand_detections_<ID>.npz`
- 输出 summary：`/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_preview_d435/<TASK>/foundation_input_<ID>/summary.json`

`COMMAND_LIBRARY.zh.md` 的 J0.1/J1.1 记录了 6 task D435 AnyGrasp 可用性检查和基于人工关键帧的候选 preview/summary 生成命令。

## L16 whitebg repaint 转换

L16 `L16_human_replay_clean/<TASK>/foundation_input_<ID>/` 是 planner-style 输出，目录里使用：

- `pose_debug.jsonl`
- `left_wrist_cam_plan.mp4`
- `right_wrist_cam_plan.mp4`
- `head_cam_plan.mp4`

当前 L16 目录不包含 `world_targets_and_status.npz`，所以不能走 D435 pure replay 的 `process_repainted_headcam_with_wrist.py`。L16 repaint 视频进入训练格式时应走：

```text
I3.6/I3.6.1 repaint final
-> L9.2 process_repainted_planner_outputs.py
-> L10.7 convert_aloha_data_to_lerobot_R1.py
-> L11.2.5 subset_lerobot_episodes.py + zip/rclone
```

默认六任务数据集名：

```text
processed_data/h2o_<TASK>_l16_whitebg_repaint-120
local/h2o_<TASK>_l16_whitebg_repaint
local/h2o_<TASK>_l16_whitebg_repaint_25ep
```

如果 `stack_cups` 使用绿色杯保护 B 方案，独立数据集名是：

```text
processed_data/h2o_stack_cups_l16_whitebg_b_points_negative-120
local/h2o_stack_cups_l16_whitebg_b_points_negative
local/h2o_stack_cups_l16_whitebg_b_points_negative_25ep
```

这些命令已经记录在 `COMMAND_LIBRARY.zh.md` 的 L9.2、L10.7、L11.2.5。I3.6.2 记录 `stack_cups` B/C debug 结论和 B 方案 Stage-1/Stage-2 输出位置。

## 检查命令

```bash
find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan -maxdepth 3 -type f \( -name 'head_cam_plan.mp4' -o -name 'pose_debug.jsonl' -o -name '*wrist*plan*.mp4' \) | sort | head -n 80
find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_h2o_plan -maxdepth 3 -type f -name '*wrist*plan*.mp4' | wc -l
```

## 三任务命令和可视化

`COMMAND_LIBRARY.zh.md` 的 `L5` / `L6` / `L8` / `L9` 记录了三任务分别运行的命令：

- `pick_diverse_bottles`
- `place_bread_basket`
- `stack_cups`

`L5` 是“原始人手 head + pure replay action/wrist”；`L6` 是“repaint robot head + pure replay action/wrist”；`L8` 是“D435 visible-reinit head + D435 pure replay action/wrist”；`L9` 是“AnyGrasp repaint head + planner action/wrist”。

注意：`L9` 仍要求 planner episode 已补齐 `left_wrist_cam_plan.mp4` 和 `right_wrist_cam_plan.mp4`。

## 三类数据运行顺序

`COMMAND_LIBRARY.zh.md` 的 L0 是当前推荐索引：

- Human 数据：L5/L5.1 或新三任务 L5.2，之后转 LeRobot 用 L10.4 或 L10.5。
- Robot replay 数据：默认广角用 L6/L6.1；D435 visible-reinit 六任务统一用 I1/I1.1 -> I3.4/I3.5 -> L8.2 -> L10.6 -> L11.2.4。
- AnyGrasp replay 数据：L9/L9.1，前提是 planner episode 已有 `pose_debug.jsonl` 和左右 wrist plan 视频。
- L16 whitebg repaint 数据：I3.6/I3.6.1 -> L9.2 -> L10.7 -> L11.2.5。`stack_cups` 如采用 B 方案，使用 I3.6.2 的独立 `e0_robot_object_b_points_negative` 输出。

关键顺序：

```text
源视频/轨迹 -> processed HDF5 -> LeRobot cache -> 25 episode subset/zip
```

新三任务若使用当前已有的 D435 action/wrist baseline：

```text
L5.2 -> L10.5 -> L11.1.3
```

新三任务若要和 D435 robot replay 对比：

```text
I1.1 -> I3.5 -> L8.1 或 L8.2 -> L10.6 -> L11.2.4
```

生成后用 `L7` 检查：

- 统计每个 `processed_data/<dataset>` 下的 HDF5 episode 数量。
- 读取前三个 HDF5，打印 `observations/state`、`action` 和三路相机帧数。
- 调用 `visualize_processed_hdf5_episode.py` 把 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 拼成一个 review mp4。

## LeRobot 转换

`COMMAND_LIBRARY.zh.md` 的 `L10` 记录了 3 种已可用数据模式 x 3 个任务的 LeRobot 转换命令：

- L5 数据：原始人手 head + pure replay action/wrist。
- L6 数据：默认广角 robot repaint head + pure replay action/wrist。
- L8 数据：D435 visible-reinit robot repaint head + D435 pure replay action/wrist。

L9 AnyGrasp 模式暂不纳入 3x3 正式 LeRobot 转换，因为当前仍依赖尚未补齐的 planner wrist 视频。

后续转换入口使用 `examples/aloha_real/convert_aloha_data_to_lerobot_R1.py --use-wrist --mode video`，原因是当前 H2O processed HDF5 写的是 `/observations/state`，而 `convert_aloha_data_to_lerobot_robotwin.py` 的普通分支仍读取 `/observations/qpos`。

## LeRobot episode 子集抽取

如果完整 LeRobot cache 已经生成，例如：

```text
/home/zaijia001/.cache/huggingface/lerobot/local/h2o_pick_diverse_bottles_human_head_pure_action
```

可以用 `subset_lerobot_episodes.py` 直接抽取指定 episode，不需要重新从 HDF5 转换。脚本支持 `0-24`、`0,1-5,7` 这类写法；解析后会去重、按旧 episode id 升序排序，并把输出数据重新编号成连续 `episode_000000` 到 `episode_0000NN`。

前 25 个 episode 示例：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && uv run python scripts/subset_lerobot_episodes.py --source local/h2o_pick_diverse_bottles_human_head_pure_action --output-repo-id local/h2o_pick_diverse_bottles_human_head_pure_action_25ep --episodes '0-24' --overwrite
```

任意 episode 示例：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_openvla && cd /home/zaijia001/ssd/RoboTwin/policy/pi0 && uv run python scripts/subset_lerobot_episodes.py --source local/h2o_pick_diverse_bottles_human_head_pure_action --output-repo-id local/h2o_pick_diverse_bottles_human_head_pure_action_subset --episodes '0,1-5,7' --overwrite
```

检查输出：

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

### Robot replay / AnyGrasp 25 episode 子集

`COMMAND_LIBRARY.zh.md` 的 L11.1 新增了 6 个 robot 数据集的 25 episode 子集命令：

- `local/h2o_pick_diverse_bottles_pure_repaint`
- `local/h2o_place_bread_basket_pure_repaint`
- `local/h2o_stack_cups_pure_repaint`
- `local/h2o_pick_diverse_bottles_anygrasp_repaint`
- `local/h2o_place_bread_basket_anygrasp_repaint`
- `local/h2o_stack_cups_anygrasp_repaint`

运行顺序是：先确认完整 LeRobot cache 已生成，再抽取 `_25ep`，然后分别 zip `robot_replay_3task_25ep.zip` 和 `robot_anygrasp_3task_25ep.zip`，最后先用 `rclone --dry-run` 检查上传目标。

`COMMAND_LIBRARY.zh.md` 的 L11.2 进一步补齐 FoundationPose 章节里的 6 个 H2O task：

- `pick_diverse_bottles`
- `place_bread_basket`
- `stack_cups`
- `handover_bottle`
- `pnp_bread`
- `pnp_tray`

L11.2 会分别为 `pure_repaint` 和 `anygrasp_repaint` 生成 6 task 的 `_25ep` 子集，并打包成：

- `robot_replay_6task_25ep.zip`
- `robot_anygrasp_6task_25ep.zip`

注意：L11.2 只能从已经存在的 LeRobot cache 抽子集。完整顺序是：

1. L5.1：6 task 的 `human_head_pure_action` processed HDF5。
2. L6.1：6 task 的 `pure_repaint` processed HDF5。
3. L9.1：6 task 的 `anygrasp_repaint` processed HDF5。
4. L10.4：把 L5.1/L6.1/L9.1 输出转成 `local/h2o_<TASK>_human_head_pure_action`、`local/h2o_<TASK>_pure_repaint` 和 `local/h2o_<TASK>_anygrasp_repaint`。
5. L11/L11.2：从这些 LeRobot cache 抽取 `_25ep`。

六任务 prompt 统一使用 L5.1 中的完整任务描述；不要只依赖 LeRobot 转换命令的 `--task`。

补充：当前新三任务 `handover_bottle / pnp_bread / pnp_tray` 只检查到 `human_replay/h2_pure_d435/<TASK>/id<ID>_d435_z005`，没有 `human_replay/h2_pure/<TASK>/id<ID>_z005`。因此 L5.1 会在这三个任务上 skip。`COMMAND_LIBRARY.zh.md` 的 L5.2 记录了可直接使用现有 D435 pure replay action/wrist 的 human-head 处理命令。

L10.5 后续不能用 `local/h2o_<TASK>_pure_repaint` 抽取；`pure_repaint` 属于 L6/L6.1 的 robot replay 数据。新三任务 human-head D435 baseline 的源 repo 是：

- `local/h2o_handover_bottle_human_head_pure_d435_action`
- `local/h2o_pnp_bread_human_head_pure_d435_action`
- `local/h2o_pnp_tray_human_head_pure_d435_action`

`COMMAND_LIBRARY.zh.md` 的 L11.1.3 记录了 L10.5 后续专用抽取命令。该命令会读取 processed data 的 `instructions.json/source_episode_id`，排除 `handover_bottle` 原始 id `0,7,12,29` 和 `pnp_bread` 原始 id `0,1,2,3,4,5,6,22,70`，然后补足前 25 个可用 episode 并重新编号为 `0..24`。

若目标是和 D435 robot replay 对比，新三任务当前推荐顺序是：

1. L5.2：真实人手 head + D435 action/wrist baseline。
2. I1.1：新三任务 Stage-1 人手抠除背景。
3. I3.5：新三任务 D435 visible-reinit robot repaint。
4. L8.1 或 L8.2：把 I3.5 输出转成 `h2o_<TASK>_pure_d435_visible_reinit-120` processed HDF5。
5. L10.6：把 D435 robot replay processed HDF5 转成 `local/h2o_<TASK>_pure_d435_visible_reinit`。
6. L11.2.4：从 D435 robot replay LeRobot cache 抽取 `_25ep` 并打包。

注意：L6.1 是默认广角 `h2_pure` 流程，不是 D435 流程。新三任务目前没有 `human_replay/h2_pure/<TASK>/id<ID>_z005`，也没有默认广角 `results_repaint_piper_h2/e0_robot/<TASK>/id_<ID>/final_repainted.mp4`，所以对新三任务运行 L6.1 会全部 skip 并报 `No usable episodes were processed`。D435 robot replay 应使用 `h2_pure_d435` 和 `results_repaint_piper_h2_d435_sam3_visible_reinit`。

如果 L8.2 输出的新三任务 episode 很少，先检查 I1.1 Stage-1 BG 和 I3.5 D435 final 数量，而不是重复跑 L8.2。L8.2 只读取：

```text
results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/<TASK>/id_<ID>_d435/final_repainted.mp4
human_replay/h2_pure_d435/<TASK>/id<ID>_d435_z005/*
```

它不会生成缺失的 `final_repainted.mp4`。`COMMAND_LIBRARY.zh.md` 的 I1.1.1 记录了“只补缺失 Stage-1 BG”的命令，I3.5 记录了 `--overwrite 0` 的 0..80 resume 命令，用于补齐 D435 repaint final 后再回到 L8.2。

`d435_final` 指的是：

```text
results_repaint_piper_h2_d435_sam3_visible_reinit/e0_robot/<TASK>/id_<ID>_d435/final_repainted.mp4
```

它由 I3.5 的 `batch_visible_reinit_d435_repaint.py` 生成。当前机器没有 `Grounded_SAM_3`，所以该脚本启动时实际显示 `[backend] SAM=sam2, DINO=dino2`；同时它会打印 `loading DINO once` 和 `loading SAM image predictor once`，表示批处理只加载一次 checkpoint 后循环处理 task/id。I3.5 现在也记录了“先补到至少 25 个 final”的 SAM2/DINO2 fallback 批处理命令。

L11.2.4 的 D435 robot replay `_25ep` 抽取已经改为和 human replay 一样的 `source_episode_id` 对齐逻辑：读取 `processed_data/h2o_<TASK>_pure_d435_visible_reinit-120/episode_*/instructions.json`，排除 `handover_bottle` 原始 bad id `0,7,12,29` 和 `pnp_bread` 原始 bad id `0,1,2,3,4,5,6,22,70`，再补足 25 个 LeRobot episode index。这样输出仍然重排为 `0..24`，但不会把这些原始坏数据放进去。

### Task prompt 设置位置

当前 `convert_aloha_data_to_lerobot_R1.py --task "..."` 不会覆盖已经写在 processed episode 里的 prompt。该转换脚本实际读取每个 episode 的 `instructions.json`：

```text
/home/zaijia001/ssd/RoboTwin/policy/pi0/processed_data/<DATASET>/episode_*/instructions.json
```

推荐顺序：

1. 生成 HDF5 processed data 时，在 `process_repainted_headcam_with_wrist.py` 或 `process_repainted_planner_outputs.py` 的 `INSTRUCTION="..."` 中设置 prompt。
2. 如果 processed data 已经生成但还没转 LeRobot，先批量替换 `episode_*/instructions.json`。
3. 如果 LeRobot cache 已经生成，只想快速修正 meta，用 L12 替换 `meta/tasks.jsonl` 和 `meta/episodes.jsonl`。

## LeRobot task 文本修正

当前 `convert_aloha_data_to_lerobot_R1.py` 的 `--task` 参数不是最终写入帧的主要来源；脚本会读取每个 processed episode 下的 `instructions.json`，所以如果源 `instructions.json` 还是旧文本，生成的 LeRobot cache 里也会保留旧 task。

如果只想修正已经生成好的 cache，可以直接替换 `meta/tasks.jsonl` 和 `meta/episodes.jsonl`。parquet 里只存 `task_index`，不存 task 文本，所以不用改 parquet。

```bash
ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_pick_diverse_bottles_human_head_pure_action; OLD='pick diverse bottles'; NEW='pick up one bottle with one arm, and pick up another bottle with the other arm.'; cp "$ROOT/meta/tasks.jsonl" "$ROOT/meta/tasks.jsonl.bak" && cp "$ROOT/meta/episodes.jsonl" "$ROOT/meta/episodes.jsonl.bak" && OLD="$OLD" NEW="$NEW" perl -0pi -e 's/\Q$ENV{OLD}\E/$ENV{NEW}/g' "$ROOT/meta/tasks.jsonl" "$ROOT/meta/episodes.jsonl"
```

```bash
ROOT=/home/zaijia001/.cache/huggingface/lerobot/local/h2o_pick_diverse_bottles_human_head_pure_action; sed -n '1,5p' "$ROOT/meta/tasks.jsonl"; sed -n '1,5p' "$ROOT/meta/episodes.jsonl"
```


## L16 六任务机器人+物体 repaint 指令

`COMMAND_LIBRARY.zh.md` 的 I3.5.2/I3.5.3 记录了 L16 Human Replay 输出的 robot/object prompt 路线。该流程不复用旧 I1 的“只抠除人手”背景，而是新建 `results_repaint_piper_h2_l16/stage1_human_object`，用任务物体 prompt 同时抠除人手和真实物体，再把 `L16_human_replay_clean/<TASK>/foundation_input_<ID>/head_cam_plan.mp4` 中的机器人+物体通过 visible-reinit 贴回。

推荐顺序：先运行 I3.5.2 六任务各 5 个 ID debug，检查 `w_box_head_cam_plan.mp4`、`w_mask_head_cam_plan.mp4` 和 `final_repainted.mp4`；确认无背景误贴和物体重影后再运行 I3.5.3 全量批处理。

`COMMAND_LIBRARY.zh.md` 的 I3.6 记录了新增的白色背景 SAM + 反选 mask 对照路线。它默认复用 I3.5.2/I3.5.3 的 human+object Stage-1 背景，先 prompt L16 源视频中的白色背景，再使用保存出的反选 mask 帧外部合成 `final_repainted.mp4`，避免直接在第一帧 prompt 机械臂/物体；合成时输出帧数跟随 robot/mask，较短的 Stage-1 背景按比例采样拉伸。`stack_cups` 的 Stage-1 prompt 去掉泛化 `cups`，只保留 `left light pink cup, right dark red cup`，以免误抠除绿色杯子。输出根目录是 `results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object`；SAM/反选检查重点看 `w_box_head_cam_plan.mp4`、`w_mask_head_cam_plan.mp4`、`mask_head_cam_plan.mp4` 和 `mask_head_cam_plan/000000.jpg`。

## L16 可视化拼接：HaMeR / Foundation / L16 / Repaint

新增脚本：`code_painting/make_l16_repaint_montage.py`。该脚本把每个任务/ID 的 HaMeR 人手可视化、Foundation object replay、L16 `head_cam_plan.mp4`、`left_wrist_cam_plan.mp4`、`right_wrist_cam_plan.mp4`、Stage1 inpaint 和 final repaint 拼成七面板 montage；默认超过 4 个面板会自动折成两行。

核心输入：

- HaMeR：`/home/zaijia001/ssd/data/piper/hand/<TASK>/harmer_output/hand_vis_gripper_<ID>.mp4`
- Foundation replay：优先 `foundation_replay_d435/foundation_input_<ID>/head_cam_replay.mp4`，缺失时回退到 `foundation_replay`
- L16 plan：`code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/<TASK>/foundation_input_<ID>/head_cam_plan.mp4`
- L16 wrist plan：同一目录下的 `left_wrist_cam_plan.mp4` 和 `right_wrist_cam_plan.mp4`，存在才加入
- 可选 Stage1：`/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stage1_human_object/<TASK>/id_<ID>/stage1_human_inpaint/removed_w_mask_rgb_<ID>.mp4`
- 可选 final repaint：`/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_visible_reinit/e0_robot_object/<TASK>/id_<ID>_l16/final_repainted.mp4`

输出：`code_painting/l16_repaint_montage/<TASK>/id_<ID>/compare_hamer_foundation_l16_repaint_<TASK>_id<ID>.mp4`，旁边会写入同名 JSON manifest 记录实际采用的输入视频。

已测试命令：

```bash
tmux new-session -d -s l16_vis_id0 'cd /home/zaijia001/ssd/RoboTwin && python3 /home/zaijia001/ssd/RoboTwin/code_painting/make_l16_repaint_montage.py --task pick_diverse_bottles --id 0 --overwrite'
```

批处理：`pick_diverse_bottles/place_bread_basket/stack_cups/pnp_tray` 可用 `--ids 0-4`；`handover_bottle` 建议从 `--ids 1-5` 开始；`pnp_bread` 建议从 `--ids 7-11` 开始，因为 L16 没有其 id0-id6。


### L16 ours review 与选取后转换

新增脚本：

- `code_painting/review_l16_ours_montages.py`
- `code_painting/run_l16_ours_selected_pipeline.sh`
- `code_painting/annotate_l16_ours_<TASK>.sh` 六个任务级标注入口

推荐流程：

```text
P4 annotate_l16_ours_<TASK>.sh 生成/播放七面板 montage（超过四列自动两行）
-> 每个任务单独按 y/n/m 标记，目标先卡 25 条
-> 写出 l16_ours_review/selections/<TASK>/ours_review_selection.json
-> run_l16_ours_selected_pipeline.sh 读取 JSON
-> h2o_<TASK>_ours-120 processed HDF5
-> local/h2o_<TASK>_ours LeRobot
-> local/h2o_<TASK>_ours_25ep
-> local/h2o_<TASK>_ours_piper0515_25ep（world frame 转左右臂 base frame）
-> robot_ours_piper0515_<TASK_GROUP>_25ep.zip + rclone dry-run/上传
```

任务级脚本默认 `TARGET_COUNT=25`、`OVERWRITE_MONTAGE=1`、`INITIAL_SPEED=1.0`。如果只是继续已有标注、不想重生成 montage，用 `OVERWRITE_MONTAGE=0 bash <script>`。

任务级标注入口：

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_pick_diverse_bottles.sh
bash /home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_place_bread_basket.sh
bash /home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_handover_bottle.sh
bash /home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_pnp_bread.sh
bash /home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_pnp_tray.sh
bash /home/zaijia001/ssd/RoboTwin/code_painting/annotate_l16_ours_stack_cups.sh
```

调速推荐使用 `+`/`-` 或 `=`/`_`；`[`/`]` 仍保留兼容，但如果窗口或播放器把它们解释成逐帧，就改用 `+/-`。其它交互按键：`y` accept、`n` reject、`m` maybe、`u` 清除、空格暂停/继续、`.` 下一条、`,` 上一条、`r` 重播、`s` 保存、`q` 保存退出。

按 JSON 做五任务转换、抽取和打包 dry-run：

```bash
TASKS="pick_diverse_bottles place_bread_basket handover_bottle pnp_bread pnp_tray" TASK_GROUP=5task DRY_RUN=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_ours_selected_pipeline.sh
```

六任务全部完成后把 `TASKS` 加上 `stack_cups`，并设置 `TASK_GROUP=6task`。正式上传时把 `DRY_RUN=1` 改成 `DRY_RUN=0`.


### L16 whitebg mask debug

`code_painting/make_l16_whitebg_mask_debug.py` 用于检查 white-background Stage-2 的 mask 方向。当前 `run_l16_whitebg_repaint_task.sh` 传了 `--invert_mask`，因此 `mask_head_cam_plan/*.jpg` 是已反选的 foreground alpha，不是原始白背景 mask。

stack id0 debug 命令：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python code_painting/make_l16_whitebg_mask_debug.py --task stack_cups --id 0 --max_frames 120
```

输出：`code_painting/l16_whitebg_mask_debug/stack_cups/id_0/whitebg_invert_debug_stack_cups_id0.mp4`。面板 3/4 是保存出的 foreground alpha，面板 5 是反向背景检查，面板 6 是最终 repaint。


## L16 白背景反选按任务并行入口

`COMMAND_LIBRARY.zh.md` 的 I3.6.1 新增了两个任务级脚本：

- `code_painting/run_l16_stage1_human_object_task.sh`：按单个 `TASK` 补/重跑 Stage-1 人手+物体 inpaint。
- `code_painting/run_l16_whitebg_repaint_task.sh`：按单个 `TASK` 执行白背景 SAM + 反选 repaint；合成时输出帧数跟随 robot/mask，短 BG 会按比例采样拉伸。

这两个脚本用于把五个非 `stack_cups` 任务分配到不同 GPU 并行运行。`stack_cups` 仍建议先单独调 Stage-1 prompt/dilation，确认绿色杯子没有被误删后再跑 repaint。

### stack_cups 绿色杯保护 debug 入口

新增脚本：

- `code_painting/l16_stack_cups_debug_variants.py`
- `code_painting/run_l16_stack_cups_debug_variants.sh`

用途：只对 `stack_cups id_0..4` 跑 Stage-1 mask/inpaint debug，不覆盖正式 `stage1_human_object`。四个方案分别输出到：

```text
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/A_protect_dino/stack_cups/id_<ID>/stage1_human_inpaint/
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/B_points_negative/stack_cups/id_<ID>/stage1_human_inpaint/
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/C_hsv_green_protect/stack_cups/id_<ID>/stage1_human_inpaint/
/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/D_tight_dino/stack_cups/id_<ID>/stage1_human_inpaint/
```

四个方案含义：

- `A_protect_dino`：DINO/SAM2 生成 remove mask，再用 `green cup` 生成 protect mask，最终 `remove - protect`。当前 debug 效果不稳定，属于错误路线；remove/protect 都依赖 DINO，大框错误会继续传导。
- `B_points_negative`：不用 DINO，使用固定正点标注两个红杯和双手，并把绿色杯中心作为负点。当前 `id_0..4` debug 可用，优先采用。
- `C_hsv_green_protect`：DINO/SAM2 remove mask 减去 HSV 绿色区域保护 mask。当前 `id_0..4` debug 可用，作为 B 的备选。
- `D_tight_dino`：更严格 prompt/threshold 的 DINO 基线，不做绿色保护。当前 debug 效果不行，属于错误路线；DINO 仍可能给出覆盖绿色杯的大框。

每个目录重点查看：

```text
w_box_rgb_<ID>.mp4
w_mask_rgb_<ID>.mp4
removed_w_mask_rgb_<ID>.mp4
w_protect_mask_rgb_<ID>.mp4   # A/C 才有
debug_summary.json
```

运行命令：

```bash
tmux new-session -d -s l16_stack_debug_variants_gpu1 'GPU=1 IDS="0 1 2 3 4" MAX_FRAMES=300 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_stack_cups_debug_variants.sh'
```

只跑 B 方案全量 Stage-1：

```bash
IDS=$(find /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean/stack_cups -path '*/head_cam_plan.mp4' 2>/dev/null | sed 's#.*/foundation_input_\([0-9]*\)/head_cam_plan.mp4#\1#' | sort -n | tr '\n' ' ')
tmux new-session -d -s l16_stack_B_stage1_gpu1 "IDS=\"$IDS\" GPU=1 VARIANTS=\"B_points_negative\" MAX_FRAMES=300 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_stack_cups_debug_variants.sh"
```

B 方案 Stage-2 读取 `B_points_negative` 背景，输出到独立结果根目录：

```bash
tmux new-session -d -s l16_stack_B_stage2_after_s1_gpu1 'while tmux has-session -t l16_stack_B_stage1_gpu1 2>/dev/null; do sleep 60; done; TASK=stack_cups GPU=1 OVERWRITE=1 STAGE1=/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16/stack_cups_debug_variants/B_points_negative OUTROOT=/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_whitebg_repaint_task.sh'
```

B 方案最终视频：

```text
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative/stack_cups/id_<ID>_l16_whitebg_human_object/final_repainted.mp4
```

### stack_cups Stage-2 红/粉杯保护重合成

`code_painting/recompose_l16_stack_redprotect.py` 是 `stack_cups` 专用的纯后处理 debug 路径。它不重新跑 SAM/GPU，而是读取 I3.6.2 B 方案 Stage-2 已保存的 foreground alpha，再从 L16 源视频按红/粉颜色规则把左右红杯 OR 回 alpha，输出到独立目录：

```text
/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative_redprotect/stack_cups/id_<ID>_l16_whitebg_human_object/
```

运行 `id_0..4`：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python code_painting/recompose_l16_stack_redprotect.py --ids 0-4 --overwrite
```

重点查看 `redprotect_debug_stack_cups_id<ID>.mp4`、`red_protect_overlay.mp4`、`final_repainted.mp4` 和 `redprotect_manifest.json`。这个方案是反选之后的红/粉杯保护，不是 negative text prompt；当前 SAM3 robot 脚本没有负文本 prompt 参数。

Piper0515 坐标转换说明：`process_repainted_planner_outputs.py` 写出的 L16/ours pose 来自 `current_*_tcp_pose_world_wxyz`，属于标定 world frame；真机器人 robot 数据使用每个机械臂自己的 base/cartesian frame。`run_l16_ours_selected_pipeline.sh` 默认在 `subset` 和 `zip` 之间调用 `code_painting/convert_lerobot_piper0515_world_to_base.py`，输出 `local/h2o_<TASK>_ours_piper0515_25ep`。该步骤只修改 LeRobot parquet 里的 `observation.state` 和 `action`，视频和 episode 顺序不变。

只补转换和打包：

```bash
STEPS="piper0515 zip" TASKS="pick_diverse_bottles place_bread_basket handover_bottle pnp_bread pnp_tray stack_cups" TASK_GROUP=6task REVIEW_ROOT=/home/zaijia001/ssd/RoboTwin/code_painting/l16_ours_review_first25 DRY_RUN=1 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_l16_ours_selected_pipeline.sh
```

### L16 Stage-2 SAM 参数化与颜色去白 debug

`run_l16_whitebg_repaint_task.sh` 走 `inpainting_sam3_robot/remove_anything_video_sam3_robot.py`，在 `inpainting-sam3-dino3` 环境中优先使用 SAM3/DINO3。Stage-2 现在可通过环境变量指定 `WHITE_PROMPT`、`MASK_IDX`、`BOX_THRESHOLD`、`TEXT_THRESHOLD`、`MAX_MASK_AREA_RATIO`、`EXCLUDE_BOTTOM_RATIO`、`DILATE_KERNEL_SIZE`、`ERODE_KERNEL_SIZE`、`COMPOSITE_ERODE`、`BLEND_ALPHA_SIGMA`。

若 SAM3/DINO3 白背景框误选严重，可用 `code_painting/repaint_l16_white_color_debug.py` 绕开 SAM，直接按 HSV/RGB 白色阈值生成白背景 mask，再反选成 foreground alpha。默认 `--border-only 1` 只去掉边界连通白背景；若要移除所有白色可设 `--border-only 0`，但更容易吃掉浅色机械臂或物体。

代表命令（`--max-frames 120` 只用于快速预览，会截短输出；全长对齐时删掉它）：

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw && cd /home/zaijia001/ssd/RoboTwin && python code_painting/repaint_l16_white_color_debug.py --task pick_diverse_bottles --ids "0 1 2 3 4" --out-root /home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/stage2_debug_color/e0_robot_object --max-frames 120 --overwrite
```

颜色路线全量输出默认跟随 L16 robot 视频帧数；Stage-1 背景按比例采样对齐，和原 Stage-2 compose 一致。
