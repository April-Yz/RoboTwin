# Troubleshooting

## Dense Replay has an approximately 17 cm fixed offset

- Symptom: planned and executed trends look similar, but the entire actual TCP curve is translated; orientation may also differ by about 90 degrees.
- Diagnosis: compare Curobo link6 FK, SAPIEN link6, and SAPIEN TCP for the same joint vector.
- Cause: Curobo and SAPIEN link6 local axes differ by fixed `Ry(-90 deg)`, while the legacy path applies the 0.12 m TCP offset in the wrong frame.
- Fix: use the isolated `run_dense_replay_urdfmatch_v2.sh`; do not reorder the joints.

## First executed frame visibly lags the plan

- Symptom: planned FK is close to the target, but the simulated TCP remains several centimeters away.
- Diagnosis: inspect `joint_metrics_after_execute` in `execution_audit.jsonl`.
- Cause: a fixed small number of simulation steps is insufficient for actuator convergence.
- Fix: v2 waits for at most 240 steps and exits early once the maximum joint error is below 0.01 rad.

## Position aligns but orientation remains tens of degrees away

- Cause: Dense directly copies a human orientation that may be unreachable by Piper.
- Handling: this is a baseline limitation. Use Ours v2 for robot-native grasp orientation; do not force a tighter rotation threshold that would turn the whole frame into an IK failure.

## Non-interactive SSH reports `conda: command not found`

- Cause: the remote shell did not load Conda initialization; this does not mean the environment is absent.
- Handling: source `/home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh`, or call `/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python3.10` directly.

## Audit V4 refuses an existing output root

- Symptom: `Refusing to overwrite non-empty output root`.
- Cause: V4 protects existing PNG, metadata, and reports by default.
- Handling: choose a new `--output-root`. Use `--overwrite` only when deliberately rebuilding V4-owned artifacts; it never authorizes changes to legacy strategy directories.
