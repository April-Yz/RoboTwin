# Piper TCP/EE IK V3 Commands

V3 is an isolated entry and does not modify or overwrite OursV2. Current OursV2 EE-labelled fields actually store the Ours TCP, so `ours_tcp` is the default.

## Parameter template (documentation only; not directly runnable)

```bash
PIPER_IK_V3_TARGET_SEMANTICS=<ours_tcp|ours_ee|real_piper_tcp> \
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435_v3.sh \
  --gpu <GPU_ID> \
  --tasks <TASK> \
  --ids <EPISODE_ID...> \
  --output_root <V3_OUTPUT_ROOT>
```

- `ours_tcp`: current OursV2; remove 12 cm and invert `delta/global_trans`.
- `ours_ee`: only for poses actually built from `current_*_ee_pose`; do not remove 12 cm.
- `real_piper_tcp`: consume the real recorded endPose frame; remove `Ry(-pi/2) * Tx(0.19)`.
- The V3 runner caps rotation acceptance at `0.12 rad`; it does not accept the old runner's 90–180 degree false-success solutions.

## Fully runnable example

```bash
PIPER_IK_V3_TARGET_SEMANTICS=ours_tcp \
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435_v3.sh \
  --gpu 2 \
  --tasks stack_cups \
  --ids 6 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/ik_v3_runs
```

Without `--output_root`, V3 writes to:

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/human_replay_v3/<TASK>/foundation_input_<ID>
```

## Transform-only unit test

```bash
cd /home/zaijia001/ssd/RoboTwin
PYTHONPATH=code_painting \
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  tests/test_piper_ik_v3_transforms.py
```

## Direct V3 planner entry

For custom `plan_summary` or AnyGrasp inputs, start with:

```bash
PIPER_IK_V3_TARGET_SEMANTICS=ours_tcp \
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_piper_v3.py \
  --help
```
