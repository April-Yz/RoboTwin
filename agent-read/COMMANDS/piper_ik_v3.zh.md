# Piper TCP/EE IK V3 命令

V3 是独立入口，不修改或覆盖现有 OursV2。当前 OursV2 的 EE-labelled 字段实际保存 Ours TCP，因此默认使用 `ours_tcp`。

## 参数模板（说明用，不可直接运行）

```bash
PIPER_IK_V3_TARGET_SEMANTICS=<ours_tcp|ours_ee|real_piper_tcp> \
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435_v3.sh \
  --gpu <GPU_ID> \
  --tasks <TASK> \
  --ids <EPISODE_ID...> \
  --output_root <V3_OUTPUT_ROOT>
```

- `ours_tcp`：当前 OursV2；减去 12 cm 并撤销 `delta/global_trans`。
- `ours_ee`：仅用于真正从 `current_*_ee_pose` 构建的 pose；不减 12 cm。
- `real_piper_tcp`：用于真机 `endPose` frame；撤销 `Ry(-pi/2) * Tx(0.19)`。
- V3 runner 默认最大 rotation threshold 为 `0.12 rad`，不会接受旧 runner 中约 90–180 度的伪成功解。

## 可直接运行示例

```bash
PIPER_IK_V3_TARGET_SEMANTICS=ours_tcp \
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_keyframes_human_replay_piper_d435_v3.sh \
  --gpu 2 \
  --tasks stack_cups \
  --ids 6 \
  --output_root /home/zaijia001/ssd/RoboTwin/code_painting/ik_v3_runs
```

不传 `--output_root` 时，默认写到：

```text
/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/human_replay_v3/<TASK>/foundation_input_<ID>
```

## 只做转换单元测试

```bash
cd /home/zaijia001/ssd/RoboTwin
PYTHONPATH=code_painting \
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  tests/test_piper_ik_v3_transforms.py
```

## 直接调用 planner V3

需要自定义 `plan_summary` 或 AnyGrasp 输入时，使用：

```bash
PIPER_IK_V3_TARGET_SEMANTICS=ours_tcp \
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  /home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_piper_v3.py \
  --help
```
