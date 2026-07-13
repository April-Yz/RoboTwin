# Piper TCP/EE IK V3

## Goal and isolation

Piper IK V3 makes TCP/EE pose to `link6` URDFIK conversion reversible without modifying the current OursV2 datasets, renderer, planner, or runner. V3 uses new files and separate outputs. It is not the legacy `demo_piper_ik_seq_v3` MotionGen line.

## Root cause 1: q-FK versus real endPose

The q-derived raw `link6` does not have a 12 cm error. After removing each side's TCP, the mean origin error over the 12 visualization episodes is `6.13 mm` left and `6.09 mm` right.

The large raw-pose delta comes from different TCP definitions:

```text
Real Piper:
T_base_real_tcp = T_base_link6 * Ry(-pi/2) * Tx(0.19)

Current Ours:
R_ours = R_link6 * diag(1,-1,-1)
p_ours_tcp = p_link6 + R_ours * [0.12,0,0]
```

The real formula was verified in:

`/home/piper/pika_ros/src/PikaAnyArm/piper/pika_remote_piper/scripts/forward_inverse_kinematics.py`

It uses `first_matrix=Ry(-1.57)`, `second_matrix=Tx(gripper_xyzrpy[0])`, and configured `gripper_xyzrpy[0]=0.19`. FK applies this tool transform and IK uses the same end frame, so the real pair is reversible. Fitted effective recorded lengths are `0.18424 m` left and `0.18095 m` right; the millimeter difference from 0.19 m is residual URDF, synchronization, calibration, and physical-tool error.

## Axis colors and physical meaning

| Color | Local axis |
|---|---|
| red | `+X` |
| green | `+Y` |
| blue | `+Z` |

Color denotes the current frame's axis name, not a shared physical role:

- Real TCP red `+X` is the physical longitudinal/forward axis.
- It is approximately Ours `-Z`, opposite the blue `+Z` arrow.
- The historical Ours 12 cm is applied along red `+X`, which is not Pika physical forward.
- Green `+Y` is closest to the lateral finger-opening axis.
- Opening and forward/approach are different axes.

Full-data dot products:

```text
Real +X dot Ours +X  ~= 0.001
Real +X dot Ours -Z  ~= 0.99999
```

## Root cause 2: why V5 EE-pose planning fails

OursV2 LeRobot fields are named `left_ee_*` / `right_ee_*`, but the builder reads:

```text
current_left_tcp_pose_world_wxyz
current_right_tcp_pose_world_wxyz
```

Those fields already contain the 12 cm Ours TCP. The old target conversion is:

```text
position += R_target * [0.12 - gripper_bias, 0, 0]
R_ee = R_target * inv(delta_matrix)
```

With `gripper_bias=0.12`, translation is zero. The old path also does not remove `global_trans_matrix=diag(1,-1,-1)`. A TCP pose containing both the 12 cm offset and the Ours orientation remap is therefore sent directly to `link6` IK.

Across 2118 V5 frames, planned versus direct-q link6 differs by `12.76 cm` left and `12.65 cm` right; offline IK succeeds on `89.2%` left and `90.3%` right.

The old human-replay runner also relaxes rotation acceptance to `pi`, reporting 90–180 degree orientation errors as successes. Since TCP translation depends on orientation, the 12 cm inverse cannot close when orientation is wrong. The isolated V3 runner caps rotation acceptance at `0.12 rad` and reach tolerance at `10 deg`; the OursV2 runner remains unchanged.

## V3 conversion

The default is `PIPER_IK_V3_TARGET_SEMANTICS=ours_tcp`, because current OursV2 EE-labelled fields actually store TCP:

```text
p_link6 = p_ours_tcp - R_ours_tcp * [gripper_bias,0,0]
R_link6 = R_ours_tcp * inv(delta_matrix) * inv(global_trans_matrix)
```

Other supported semantics:

- `ours_ee`: input is an actual `current_*_ee_pose`; remove the orientation remap but no 12 cm translation.
- `real_piper_tcp`: input uses the real endPose frame; remove `Ry(-pi/2) * Tx(tool_length)`.

Core files:

- `code_painting/piper_ik_v3_transforms.py`
- `code_painting/render_hand_retarget_piper_dual_npz_urdfik_v3.py`
- `code_painting/plan_anygrasp_keyframes_piper_v3.py`
- `code_painting/plan_keyframes_human_replay_v3.py`
- `code_painting/run_plan_keyframes_human_replay_piper_d435_v3.sh`

No existing OursV2 file is modified. V3 writes to `human_replay_v3/` by default.

## Validation and scope boundary

- 300 random Ours TCP, Ours EE, and real Piper TCP round trips passed.
- 4236 arm poses from the 12 V5 episodes: max position error `9.7e-17 m`; max rotation error `7.7e-16 rad`.
- Direct-q-seeded URDFIK: `2118/2118` successes for both arms; mean position error `1.30e-7 m` left and `2.51e-7 m` right.
- Python compilation, shell syntax, and both V3 `--help` entries passed.

The `stack_cups id6` human-replay smoke instantiated V3 successfully but did not reach the hand targets because the HaMeR/hand target orientation remains over 100 degrees away from link6/TCP semantics. That is a separate hand-frame orientation-remap issue. V3 fixes the known OursV2 TCP/EE inverse and rejects such orientation errors.
