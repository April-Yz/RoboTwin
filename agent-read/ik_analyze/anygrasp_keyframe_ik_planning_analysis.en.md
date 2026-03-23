# RoboTwin AnyGrasp Keyframe IK / Planning Analysis

This document explains the IK and planning logic used by the AnyGrasp keyframe pipeline.

## 1. Scope and call chain

Relevant files:

- `/home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh`
- `/home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1_batch.py`
- `/home/zaijia001/ssd/RoboTwin/code_painting/plan_anygrasp_keyframes_r1.py`
- `/home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_r1_npz.py`
- `/home/zaijia001/ssd/RoboTwin/code_painting/render_hand_retarget_r1_npz_urdfik.py`
- `/home/zaijia001/ssd/RoboTwin/code_painting/urdfik.py`
- `/home/zaijia001/ssd/RoboTwin/envs/robot/robot.py`
- `/home/zaijia001/ssd/RoboTwin/envs/robot/planner.py`

Actual call chain:

1. `run_plan_anygrasp_keyframes_r1_batch.sh`
2. `plan_anygrasp_keyframes_r1_batch.py`
3. `plan_anygrasp_keyframes_r1.py`
4. build renderer according to `--planner_backend`
5. select AnyGrasp candidates for the two keyframes
6. execute `pregrasp -> grasp -> close gripper -> action`

There are **2 backends**, but **3 practical planning modes**:

1. Backend A old mode: `--planner_backend urdfik --urdfik_trajectory_mode joint_interp`
2. Backend A new mode: `--planner_backend urdfik --urdfik_trajectory_mode cartesian_interp_ik`
3. Backend B: `--planner_backend curobo`

## 2. Computation modes

### Mode A1: `planner_backend=urdfik`, `urdfik_trajectory_mode=joint_interp`

Implemented by:

- `render_hand_retarget_r1_npz_urdfik.py`
- `urdfik.py`

Important conclusion:

- The file name says `urdfik`, but the solver is actually **cuRobo IKSolver**, not a classical analytical URDF IK package.
- This mode does **not** use cuRobo MotionGen trajectory optimization.
- It solves **one final target joint configuration** only.
- Then it executes that result with **joint-space linear interpolation**.

This is the original behavior.

### Mode A2: `planner_backend=urdfik`, `urdfik_trajectory_mode=cartesian_interp_ik`

Implemented by:

- `render_hand_retarget_r1_npz_urdfik.py`
- `urdfik.py`

Important conclusion:

- This mode still uses the same local cuRobo IK core.
- But it does **not** jump directly from current TCP pose to final TCP pose before interpolating in joint space.
- Instead it:
  1. interpolates TCP waypoints in Cartesian space
  2. converts each TCP waypoint to the internal end-link target
  3. solves IK waypoint by waypoint
  4. seeds each waypoint IK with the previous solved joint state
  5. builds a multi-point joint trajectory from those waypoint IK results

So this is still **direct IK**, not full planning, but it is much closer to "follow this EE path".

### Mode B: `planner_backend=curobo`

Implemented by:

- `render_hand_retarget_r1_npz.py`
- `robot.py`
- `planner.py`

Important conclusion:

- This mode uses **cuRobo MotionGen** for full path planning.
- It plans from current joint state to the target end-effector pose and returns a **trajectory** (`position`, `velocity`), not just one IK solution.

## 3. Candidate selection before IK / planning

The script first selects which AnyGrasp pose should be executed.

Per-frame ranking prefers:

1. smaller orientation mismatch against the hand reference
2. smaller distance to the nearest object
3. larger AnyGrasp score

Then the two-keyframe pair is chosen with a same-object constraint.

So there are two optimization layers:

1. candidate selection
2. IK / planning for the selected target pose

## 4. Shared target-pose semantics

The selected AnyGrasp target is treated as a **TCP / gripper-center target**, not directly the wrist end-link target.

Before IK or planning, the code converts:

- world TCP target
- to robot-base frame
- then from gripper/TCP frame to end-link frame

So the solver is not solving the raw AnyGrasp pose directly. It solves the corresponding wrist/end-link pose after the built-in TCP-to-endlink transform.

## 5. What exactly happens in each mode?

### 5.1 Mode A1: single endpoint IK + joint interpolation

Pipeline:

1. read current arm joints
2. concatenate `torso_qpos + current_arm_joints` as the IK seed
3. convert target TCP pose to target EE pose in base frame
4. call cuRobo `IKSolver.solve_batch`
5. if successful, extract the final arm joint solution
6. build a 2-point joint trajectory: `[current_arm, target_arm]`
7. densify it later with `--execute_interp_steps`
8. execute the densified joint trajectory

The executed TCP path is therefore only an indirect consequence of joint interpolation.

### 5.2 Mode A2: Cartesian TCP interpolation + waypoint IK

Pipeline:

1. read current TCP pose
2. interpolate from current TCP pose to final TCP pose in Cartesian space
   - position: linear interpolation
   - orientation: quaternion slerp
3. for each interpolated waypoint:
   - convert waypoint TCP target to EE target in base frame
   - solve IK with the previous solution as the seed
4. stack all solved arm joint waypoints into a joint trajectory
5. optionally densify again with `--execute_interp_steps`
6. execute the resulting joint sequence

New parameter:

- `--urdfik_cartesian_interp_steps N`

Interpretation:

- this is still **local waypoint IK**, not full motion planning
- but the desired end-effector path is explicitly defined before IK

Why A2 is usually more reasonable than A1 when EE-path smoothness matters:

- if start and end Cartesian poses are close but joint branches are ambiguous, joint interpolation can produce unintuitive wrist motion
- Cartesian waypoint interpolation constrains the desired TCP path directly
- reseeding each waypoint from the previous IK result usually keeps the motion on one local branch more consistently

Main caveat:

- because each waypoint still relies on local IK and there is still no collision-aware planner, the chain can fail at an intermediate waypoint even when the final pose alone is solvable

### 5.3 Mode B: full cuRobo MotionGen planning

Pipeline:

1. target TCP pose is passed into `robot.left_plan_path()` / `robot.right_plan_path()`
2. robot converts TCP target into end-link pose
3. `CuroboPlanner.plan_path(...)` converts world target to base frame and applies frame bias
4. `MotionGen.plan_single(...)` generates a trajectory from current state to the goal pose
5. the returned trajectory is executed point by point
6. if error is still too large, the system replans from the updated current state

This is much closer to a real planner than either A1 or A2.

## 6. Which one is more reasonable?

Short answer:

- If the real requirement is "I want the end-effector path itself to be smoother and more intuitive", then **A2 is more reasonable than A1**.
- If the real requirement is "I want a reliable collision-aware planner", then **B is still the better direction**.

Practical interpretation:

- A1 optimizes for simplicity and speed.
- A2 is the better compromise when you want to stay on the direct-IK backend but avoid obviously strange Cartesian motion.
- B is the right answer when planning quality matters more than raw speed or simplicity.

## 7. Reach check and replanning

After each stage execution, the code computes pose error and can replan from the updated current state.

So the overall system is still:

- plan
- execute
- measure
- optionally replan

The new A2 mode only changes **how Backend A constructs the trajectory inside one planning attempt**.

## 8. Explicit commands

### 8.1 Backend A old mode

```bash
CUDA_VISIBLE_DEVICES=3 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_debug \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes \
  --ids 1 \
  --keyframes 1 22 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode joint_interp
```

### 8.2 Backend A new mode

```bash
CUDA_VISIBLE_DEVICES=3 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_debug \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes \
  --ids 1 \
  --keyframes 1 22 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps 8
```

### 8.3 Backend B

```bash
CUDA_VISIBLE_DEVICES=3 bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_debug \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes \
  --ids 1 \
  --keyframes 1 22 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend curobo
```

## 9. Bottom line

1. Backend A old mode is still available and unchanged in semantics.
2. Backend A new mode makes the TCP path explicit before IK, which is usually a better match for the current debugging goal.
3. Backend B is still the better answer when true planning quality matters more than just smoother IK execution.

## 10. Selected-keyframe camera-up rule

For `--candidate_keep_camera_up 1`, the current behavior is now sequential instead of independent per keyframe:

1. keyframe 1 still resolves the roll ambiguity by comparing:
   - the original pose
   - the pose rolled by `180` degrees around local `+X`
   and keeping the more upward-facing one.
2. later keyframes still compare only those same two equivalent roll states,
   but they now choose the variant whose rotation change from the previous selected keyframe is smaller.
3. if the difference is nearly tied, the more upward-facing variant is preferred.

This change is intended to stop keyframe 22 from making a full extra roll around the forward axis when keyframe 1 already established a good camera-up baseline.
