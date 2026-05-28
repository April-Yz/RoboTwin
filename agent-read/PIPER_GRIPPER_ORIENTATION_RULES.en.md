# Piper Gripper Orientation Rules

## Current Takeaway

For the current `pnp_star_pear_hamer_output_v2/hand_detections_0.npz` and calibrated `PiperPika` scene, the RGB axes in the debug boards can be read as:

- Red `+X`: a side/normal axis derived from the finger skeleton plane, not the main approach axis.
- Green `+Y`: thumb-tip to index-tip direction, closest to the gripper opening axis.
- Blue `+Z`: computed by `X cross Y`, currently closest to the gripper approach direction.

So your observation that blue is the forward/approach axis, green is the opening axis, and red is the remaining side axis matches the current code and visualization.

## Why Recent Debug Runs Defaulted To Left Hand

The recent debug wrappers default to:

- `ARM=left`
- or `ARMS=left`

This was only to reduce variables while debugging axis semantics. Single-arm IK is faster and easier to interpret. The tools are not left-hand only.

Use:

```bash
ARM=right ...
```

or:

```bash
ARMS=right ...
```

For both arms:

```bash
ARMS=both ...
```

Running both arms can make the board harder to read because one unreachable side may dominate the status. The recommended flow is left first, then right separately.

## HaMeR / NPZ Gripper Local Axes

Main code:

- `code_painting/render_hand_retarget_r1_npz.py`
- `calc_gripper_pose_from_keypoints(...)`

If the NPZ contains `left_gripper_rotation_matrix` / `right_gripper_rotation_matrix`, the script uses those directly. Otherwise it recomputes the same convention from keypoints.

Formula:

```text
gripper_position = 0.5 * (thumb_tip + index_tip)

+Y = normalize(thumb_tip - index_tip)
temp = index_joint - index_tip
+X = normalize(cross(temp, +Y))
+Z = normalize(cross(+X, +Y))

rotation_matrix = [ +X  +Y  +Z ]  # columns are local x/y/z axes
retreat_position = gripper_position - retreat_distance * +Z
```

Interpretation:

- `+Y` is the fingertip-to-fingertip direction, so it is closest to the opening axis.
- `+Z` is used for retreat: `retreat = center - d * Z`, so `+Z` points from the retreat/wrist side toward the gripper center/approach side.
- `+X` completes the right-handed frame and is closer to a side/normal axis.

## Orientation Fix Order

Each frame starts from `rotation_cam`.

Code:

- `remap_target_rotation(...)`

Formula:

```text
R_cam_fixed = R_cam * R_post * R_remap
```

Where:

- `R_cam`: HaMeR/NPZ gripper rotation.
- `R_post`: command-line `--stored_orientation_post_rot_xyz_deg RX RY RZ`.
- `R_remap`: fixed axis remap from `--orientation_remap_label`.

This is right multiplication, so `R_post` and `R_remap` act in the local gripper frame.

## Head Camera To World

Code:

- `camera_to_world_pose(...)`

Current calibrated command values:

```text
--camera_cv_axis_mode legacy_r1
--head_camera_local_pos 0.107882 -0.2693875 0.464396
--head_camera_local_quat_wxyz 0.85401166 0.01255256 0.51885652 -0.0359783
```

Formula:

```text
pos_world = p_head_world + R_head_world * C_legacy * pos_cam
R_world = R_head_world * C_legacy * R_cam_fixed
```

`legacy_r1` matrix:

```text
C_legacy =
[[ 0,  0,  1],
 [-1,  0,  0],
 [ 0, -1,  0]]
```

In the Piper dual-arm scene, the head camera is fixed relative to the left base:

```text
world_T_head = world_T_left_base * left_base_T_head_camera
```

## Piper Base And Dual-Arm Placement

Current config:

- `robot_config_PiperPika_agx_dual_table.json`
- `dual_arm_embodied=false`
- `embodiment_dis=0.60`

The scene loads two independent Piper instances:

```text
left_base  ~= [-0.3, -0.25, 0.75]
right_base ~= [ 0.3, -0.25, 0.75]
base_quat  = [0.70710678, 0, 0, 0.70710678]
```

Each target is converted from world to the corresponding arm base:

```text
target_base = inv(world_T_arm_base) * target_world
```

## Gripper Target To Piper Link6 / URDFIK

Key code:

- `render_hand_retarget_piper_dual_npz_urdfik.py`
- `_target_tcp_world_to_ee_base(...)`
- `envs/robot/robot.py`
- `_trans_from_gripper_to_endlink(...)`

Piper URDFIK solves:

```text
base_link -> link6
```

Before IK:

```text
target_pose_base = world_pose_to_base_pose_for_arm(target_world, arm)
target_pose_ee = robot._trans_from_gripper_to_endlink(target_pose_base, arm)
```

Current config values:

```text
gripper_bias = 0.12
delta_matrix = I
global_trans_matrix = diag(1, -1, -1)
```

Inside `_trans_from_gripper_to_endlink(...)`:

```text
position += R_gripper * [0.12 - gripper_bias, 0, 0]
R_ee = R_gripper * inv(delta_matrix)
```

With the current config:

```text
position offset = 0
R_ee = R_gripper
```

So in the current `PiperPika` setup, the pose sent to URDFIK is effectively the retarget target pose itself.

## Important Frame Asymmetry

`get_left_tcp_pose()` / `get_right_tcp_pose()` read back TCP through `_trans_endpose(..., is_endpose=True)`:

```text
R_tcp_readback = R_link6 * global_trans_matrix * delta_matrix
```

Currently:

```text
global_trans_matrix = diag(1, -1, -1)
delta_matrix = I
```

So the readback TCP frame has flipped local Y/Z relative to `link6`.

However, `_trans_from_gripper_to_endlink(...)` currently applies only `inv(delta_matrix)` when feeding the IK target, not the inverse of `global_trans_matrix`.

Practical impact:

- The target axis actor shows the retarget target axes.
- URDFIK receives nearly the same target axes.
- Post-execution debug from `get_*_tcp_pose()` may include an extra `diag(1,-1,-1)` frame conversion.

Use target-axis boards for axis semantics. Treat post-execution TCP frame comparisons with this frame asymmetry in mind.

## Debug Recommendation

Current observations imply:

- HaMeR/recomputed gripper blue `+Z` is closest to approach direction.
- Green `+Y` is closest to opening direction.
- Robot/IK conventions do not naturally assume that `+Z` is the approach axis.

Recommended sequence:

1. Use `run_piper_retarget_postrot_board_video.sh` with `CASE_MODE=standard`.
2. Expand to `CASE_MODE=axis90`.
3. Run longer sequences or right-hand checks only for visually plausible candidates.
4. If a candidate looks right but succeeds rarely, tune `TARGET_DY/TARGET_DZ` and the IK initial state before continuing to scan more rotations.
