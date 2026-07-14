# Dense Replay URDF-match v2

This is an isolated correction of the legacy Dense Replay. It does not modify the legacy renderer, runner, or outputs. It still performs dense retargeting of frame-wise HaMeR motion, while fixing mismatches among the planning model, local frames, and execution convergence.

## Root causes

1. The legacy qualitative video was generated on 2026-05-23, while the current Piper IK URDF correction was committed on 2026-06-18. The video therefore predates the model correction.
2. For the same `joint1..joint6`, Curobo and SAPIEN agree on the `link6` origin, but their local axes differ by a fixed rotation:

   ```text
   R_sapien_link6 = R_curobo_link6 @ Ry(-90 deg)
   ```

   The legacy path omitted this adapter. Since the TCP is 0.12 m in front of link6, the frame mismatch creates a fixed `0.12 * sqrt(2) = 0.1697 m` position offset. The joint order itself is correct.
3. The legacy constructor chain overwrote the requested `urdfik_joint_interp_waypoints=10` with its default value of 2.
4. Legacy execution waited a fixed number of steps without checking measured joint convergence.
5. HaMeR `gripper_position` is the midpoint of thumb and index fingertips and must be interpreted as a TCP; the legacy path mixed link6, EE, and TCP semantics.

## v2 changes

- Force both IK and SAPIEN to use `piper_pika_agx`, and assert the `joint1..joint6` order for both arms.
- Apply the fixed Curobo-to-SAPIEN `Ry(-90 deg)` adapter in TCP↔link6 conversion.
- Exactly invert the 0.12 m TCP bias, `global_trans_matrix`, and `delta_matrix` used by robot TCP reporting.
- Restore the requested 10 joint interpolation waypoints after inherited initialization.
- Use an Ours-v2-like joint-continuity, multi-seed IK policy while relaxing robot-unreachable human orientation.
- Wait for at most 240 additional simulation steps, exiting early when every target joint is within 0.01 rad.
- Emit `dense_replay_v2_metadata.json` and per-arm, per-frame `execution_audit.jsonl`.

## Command template (not directly runnable)

```bash
TASK=<task_name> ID=<episode_id> GPU=<gpu_id> MAX_FRAMES=<n_or_-1> \
OUT_ROOT=<separate_output_root> \
bash code_painting/run_dense_replay_urdfmatch_v2.sh
```

Common parameters are `TASK`, `ID`, `GPU`, `MAX_FRAMES`, and `OUT_ROOT`. Do not point `OUT_ROOT` at the legacy `h2_pure_d435` directory, to avoid mixing versions.

## Verified runnable command

```bash
cd /home/zaijia001/ssd/RoboTwin
TASK=pick_diverse_bottles ID=0 GPU=3 MAX_FRAMES=-1 \
bash code_painting/run_dense_replay_urdfmatch_v2.sh
```

Default output:

```text
code_painting/human_replay/h2_pure_d435_urdfmatch_v2/
└── pick_diverse_bottles/id0_d435_z005/
    ├── zed_replay.mp4
    ├── third_replay.mp4
    ├── left_wrist_replay.mp4
    ├── right_wrist_replay.mp4
    ├── dense_replay_v2_metadata.json
    ├── execution_audit.jsonl
    └── dense_replay_urdfmatch_v2_validation.json
```

## Validation and limitation

For the first 8 frames of `pick_diverse_bottles/id0`, all 16 arm records solved IK. Mean planned TCP position error was 5.48 mm; mean executed TCP position error was 4.72 mm with a 14.50 mm maximum; every final joint error was below 0.01 rad; mean Curobo-FK versus SAPIEN-TCP discrepancy was about `2.65e-7 m`.

Across all 106 frames, the left/right arms have `85/106` and `83/106` successful targets. For the 168 successful arm plans, mean planned/executed position errors are `4.44/4.70 mm`, and mean FK-versus-simulated-TCP discrepancy is `3.16e-7 m`. Failures are mainly missing, invalid, or far outside-workspace human targets. They remain explicit Dense-baseline failures and are not replaced with fabricated frames.

Human orientations are frequently unreachable by Piper, and the sampled orientation error remains about 38 degrees. This is an action-level cross-embodiment gap of Dense Replay. Ours v2 avoids copying this dense orientation by using robot-native grasp candidates and human-guided grasp selection. v2 fixes the coordinate and execution offsets without turning the Dense baseline into Ours v2.
