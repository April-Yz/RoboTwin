# PiperCanonicalTCP-v1

## Goal and isolation boundary

`PiperCanonicalTCP-v1` is an isolated Real-Piper-TCP planning and comparison line. It does not modify OursV2, Piper IK V3, or legacy results. Code lives in `code_painting/piper_canonical_tcp_v1/`. Planner targets, current readback, reach checks, and visualized targets all use `T_W_RTCP`.

## Frame contract

`T_A_B` is the pose of frame B expressed in frame A. `R_A_B` maps a local-B vector into frame A.

| Name | Meaning |
|---|---|
| `W` | shared Piper0515 world |
| `B_L/B_R` | left/right arm base |
| `CGRASP` | canonical grasp local axes stored by robot-frame preview |
| `RTCP` | Piper server Real TCP; local `+X` is forward/approach |
| `L6_SIM` | raw SAPIEN `link6` actor frame |
| `L6_URDF` | CuRobo URDF IK/FK `link6`, also used by the server tool formula |

Colors are red `+X`, green `+Y`, and blue `+Z`. Text must distinguish `world_+X` from `local_RTCP_+X` rather than using bare X/Y/Z labels.

### SAPIEN versus URDF link6

Runtime same-q FK shows identical origins but a fixed 90-degree local-axis difference:

```text
R_L6SIM_L6URDF =
[[ 0, 0, 1],
 [ 0, 1, 0],
 [-1, 0, 0]]
= Ry(+pi/2)  # exact signed-axis permutation
```

Before adaptation, same-q rotation error is 90.000 degrees. After adaptation it is about `0.000016 deg` left and `0.000006 deg` right, with position error below `7.5e-8 m`.

### Piper server tool

The server literals remain exact; do not replace `-1.57` with `-pi/2`:

```text
T_L6URDF_RTCP = Ry(-1.57) @ Tx(0.19)
```

The complete chain is:

```text
T_W_RTCP = T_W_B @ T_B_L6SIM
                    @ T_L6SIM_L6URDF
                    @ T_L6URDF_RTCP

T_B_L6URDF = inv(T_W_B) @ T_W_RTCP @ inv(T_L6URDF_RTCP)
```

### Preview grasp axes

Robot-frame preview stores:

```text
R_W_CGRASP = R_W_RTCP @ R_RTCP_CGRASP
R_RTCP_CGRASP = [[0,0,1],[0,1,0],[-1,0,0]]
```

Orientation/Fused right-multiply by `R_CGRASP_RTCP = R_RTCP_CGRASP.T`, implemented as `swap_red_blue_keep_green`. Top-score recomputes raw AnyGrasp rotations at the same keyframes and uses `identity`.

`R_RTCP_CGRASP` and `R_L6SIM_L6URDF` happen to share a numeric matrix, but the first is a candidate-source local-axis conversion and the second is a simulator/URDF model-frame adapter. They must not be merged or omitted.

## Comparison definitions

### Same-q joint control

For the same q and raw `L6_SIM`, compare the historical OursV2 0.12 m TCP against the server's 0.19 m Real TCP. In corrected `pnp_bread/id8`, both arms have mean/max distance near `0.0700001 m`: the 7 cm difference along the now-shared forward axis. The old `0.224641 m` smoke result omitted `L6_SIM -> L6_URDF` and is not a valid physical conclusion.

### Three EE-pose strategies

- Orientation: preview orientation rank 1;
- Fused: rank 1 from `0.25 * raw AnyGrasp score + 0.75 * orientation score`;
- Top-score: maximum raw AnyGrasp score at the same keyframe.

Top-score has no orientation constraint and may select a grasp flipped nearly 180 degrees around the approach axis. A strict IK miss is a strategy outcome. Preserve its video and failure TSV entry rather than forcing a flip or relaxing thresholds.

## Smoke validation

The fully passing episode is `code_painting/piper_canonical_tcp_v1/smoke_all_pass/pnp_bread/foundation_input_8/`.

- arm is left;
- Orientation, Fused, and Top-score all reached pregrasp, grasp, and action;
- all three head videos, three `SUCCESS` markers, and `strategy_comparison.mp4` exist;
- corrected same-q joint video and `SUCCESS` exist;
- strategy videos are H.264 640x360, composite is 1440x490, and joint comparison is 1280x794.

`smoke_pass/stack_cups/id0` is retained as a strategy-difference example: Orientation/Fused succeed while Top-score selects a 178.79-degree flipped candidate and strictly fails.

## Batch

`batch_manifest.tsv` fixes 6 tasks x 5 episodes = 30 episodes. Each episode runs Orientation, Fused, and Top-score sequentially; joint-control comparison runs separately. The 2026-07-15 batch uses new `outputs_canonical_20260715/` and does not overwrite old dry-run files under default `outputs/`.

tmux names: `pcan_v1_joint_6x5` and `pcan_v1_eepose_6x5`.

The EE batch continues after a strategy IK miss and records it in `outputs_canonical_20260715/_batch_logs/eepose_failures.tsv`. It composes `strategy_comparison.mp4` whenever all three head videos exist, but never creates a fake `SUCCESS` for a failed strategy.

## Validation

- Six mathematical unit tests pass: server literals/order, URDF-link6 round trip, SIM/URDF adapter, preview-axis inverse, and world/base inverse.
- Python `py_compile` and shell `bash -n` pass.
- Dry runs confirm `swap_red_blue_keep_green` for Orientation/Fused and `identity` for Top-score.
- Single-episode three-strategy planning, joint control, ffprobe, and visual frame QA pass.

## OursV2 / Canonical / Piper-real three-chain control comparison

The three entries under `outputs_canonical_20260715/eepose/<strategy>` are only Orientation, Fused, and Top-score candidate-selection strategies, all executed through Canonical IK. They do not include OursV2 IK and are not a three-controller effect comparison.

The new `run_real_control_compare.sh` consumes synchronized D435, `jointState`, and `endPose` data from a Piper raw episode:

- Joint control: the common input is real q1-q6. The Real curve reads server endPose; OursV2 uses `R_B_OTCP=R_B_L6URDF@diag(1,-1,-1)` plus 0.12 m along local +X; Canonical uses `T_B_RTCP=T_B_L6URDF@Ry(-1.57)@Tx(0.19)`.
- EE-pose control: the common input is real `T_B_RTCP`. The OursV2 branch faithfully preserves the legacy default by sending those numeric values directly as the `T_B_L6URDF` IK target without the server inverse tool, using a 3.14-rad maximum rotation threshold. Canonical first applies `T_B_L6URDF=T_B_RTCP@inv(Ry(-1.57)@Tx(0.19))` and uses 0.12 rad.
- Both planned-q results are re-evaluated as physical Canonical RTCP. A failed arm shows a red direct-q visual fallback and is excluded from curves and error statistics.
- Every chart uses shared Piper0515 world XYZ. Red/green/blue pose axes in the image are local TCP +X/+Y/+Z and are not the world-curve colors.

The eight-frame smoke is under `outputs_real_control_smoke_20260716/handover_bottle/episode0/`. Joint Canonical mean position error versus real is `9.7748/9.5863 mm` left/right, while OursV2 is `215.84/218.77 mm`. Both EE-pose branches achieve 100% IK success; Canonical physical-RTCP error is `0.0111/0.0039 mm`, versus `195.08/195.41 mm` for OursV2. Both 1920x1080 MP4s are H.264/yuv420p/faststart and pass visual QA.

`real_control_manifest.tsv` fixes 30 audited raw episodes (six tasks x five). It has the same count as the 30 AnyGrasp foundation IDs in `batch_manifest.tsv`, but not the same sample semantics: one is a Piper raw episode and the other is a foundation input ID. Slots must not be silently paired.
