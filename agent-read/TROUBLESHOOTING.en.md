# Troubleshooting

## PiperCanonicalTCP-v1: ffmpeg reads an MP4 but VS Code cannot play it

- Symptom: OpenCV/ffmpeg decodes the complete video, while VS Code preview is blank, unsupported, or never starts.
- Diagnosis: run `ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt <VIDEO>`. An `mpeg4`/`mp4v` file may be structurally valid even though Chromium lacks the decoder.
- Cause: the legacy planner transcoded only head/third videos to H.264. Wrist, debug, joint-comparison, and strategy-comparison files remained OpenCV `mp4v` outputs.
- Fix: run `vscode_video.py --apply` from `COMMANDS/piper_canonical_tcp_v1.en.md`. It validates a temporary file before atomic replacement; renaming the extension is not a fix.

## PiperCanonicalTCP-v1: which videos are D435

- `foundation_replay_d435`, AnyGrasp D435 preview, and `source_preview_compare/*d435*.png` use physical D435 data/calibration.
- `head_cam_plan.mp4` is a SAPIEN head-camera render driven by the D435 calibration, not raw D435 footage.
- `third_cam_plan.mp4`, left/right wrist MP4 files, and debug/comparison MP4 files are simulated or composed views and must not be labeled as raw D435 video.

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

## A legacy Dense repaint still appears beside v2 raw replay

- Cause: existing Stage-2 repaint/HDF5 artifacts were generated from v1 `h2_pure_d435`; fixing raw replay cannot silently turn them into v2.
- Handling: the expanded paper grid labels that tile `LEGACY V1 SOURCE / NOT V2`. Do not present it as a matched pair. Regenerate Stage-2 from `h2_pure_d435_urdfmatch_v2` and use a new processed/LeRobot name when a matched v2 dataset is needed.

## The active batch episode is unclear

- Inspect `tmux capture-pane -pt dense_replay_urdfmatch_v2:0 -S -60` and `_batch_logs/status.tsv`.
- A `started` row without `complete` means the episode is still running or was interrupted. On restart, an episode is skipped only when replay, targets, metadata, audit, and a valid frame count all exist.

## Audit V4 refuses an existing output root

- Symptom: `Refusing to overwrite non-empty output root`.
- Cause: V4 protects existing PNG, metadata, and reports by default.
- Handling: choose a new `--output-root`. Use `--overwrite` only when deliberately rebuilding V4-owned artifacts; it never authorizes changes to legacy strategy directories.

## PiperCanonicalTCP-v1: same-q position matches but rotation differs by 90 degrees

- Symptom: `fk-contract-check` reports near-zero raw SIM/URDF position error but about 90-degree rotation error.
- Cause: SAPIEN `L6_SIM` and CuRobo/server `L6_URDF` are different local-axis frames.
- Handling: ensure readback includes `T_L6SIM_L6URDF=Ry(+pi/2)`; adapted rotation error should be near `0.00001 deg`. Never merge this with server `Ry(-1.57)`.

## Top-score IK fails while Orientation/Fused succeed

- Symptom: Top target rotation is near 180 degrees; position is close but strict rotation IK does not converge.
- Cause: maximum raw score has no orientation constraint and may be flipped around the approach axis.
- Handling: preserve the failure video and `eepose_failures.tsv`. Do not force-flip or relax rotation acceptance to pi. Compose the strategy comparison whenever all three head videos exist.

## Non-empty output directory without SUCCESS

- The runner refuses to overwrite it. Use a new `--output-root`, or retain the audited failure result; never delete/overwrite an old smoke to manufacture a pass.

## OursV2 IK is absent under `outputs_canonical_20260715/eepose`

- Cause: those three entries are Orientation/Fused/Top-score candidate strategies, and all three feed Canonical IK.
- Handling: run `run_real_control_compare.sh`. Its `eepose_control.mp4` contains the Piper-real reference, OursV2 legacy EE-pose IK, and Canonical server-semantic IK.

## Real-control OursV2 EE-pose is about 19.5 cm away

- Cause: the common input is server `T_B_RTCP`, while the legacy OursV2 default sends the numeric pose unchanged as a `T_B_L6URDF` target without removing `Ry(-1.57)@Tx(0.19)`. This is not caused by a relaxed IK position threshold.
- Handling: inspect branch semantics in `summary.json`. Evaluate both q traces through Canonical physical-RTCP FK. Quietly adding the server tool to the OursV2 branch would stop being a legacy-chain comparison.

## A simulated gripper is offscreen

- Early frames may show only an `offscreen` arrow because of the calibrated 0515 head-camera field of view. This does not mean FK or curves are missing. Inspect later frames in the full episode and use world-XYZ curves plus IK success masks for numerical conclusions.
