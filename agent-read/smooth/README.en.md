# Smooth handling notes (AnyGrasp keyframe planner)

## 1. Background

Your current common pipeline is this command family:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_realoffset_batch_pure-v3 \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --urdfik_cartesian_interp_steps -1 \
  --urdfik_cartesian_interp_auto_step_m 0.05 \
  --candidate_selection_mode planner \
  --left_target_object cup \
  --right_target_object bottle \
  --candidate_target_local_x_offset_m -0.03 \
  --approach_offset_m 0.08 \
  --reach_error_pose_source tcp \
  --replan_until_reached 1 \
  --replan_until_reached_max_attempts 3 \
  --save_debug_preview 1 \
  --save_debug_execution_preview 0 \
  --reach_pos_tol_m 0.03 \
  --reach_rot_tol_deg 20 \
  --enable_grasp_action_object_collision 1 \
  --pure_scene_output 1 \
  --overlay_text 0 \
  --debug_visualize_targets 0 \
  --viewer_show_camera_frustums 0 \
  --settle_steps 20 \
  --joint_target_wait_steps 70 \
  --enable_viewer 0
```

The current issue is:

- large `joint_target_wait_steps`: better convergence and better final accuracy
- but the exported video can show visible stage-end jumps / teleport-like motion
- small `joint_target_wait_steps`: cleaner video, but less precise final execution

The core tension is:

- **execution accuracy needs extra physics settling**
- **but the main exported video does not serialize each settle-loop step frame by frame**

So the viewer / internal physics can still be continuous while the saved video looks abrupt.

---

## 2. Summary of the current smooth-related handling

### 2.1 Task-level path structure

Based on the existing docs and parameters, the current execution chain can be summarized as:

1. known `init` pose
2. known cup-grasp pose
3. known final pour / action-end pose
4. organized as:
   - `init`
   - `pregrasp`
   - `grasp`
   - `action`

Where:

- `pregrasp` is not manually annotated as an extra frame
- it is derived from the `grasp` pose
- at a high level it can be remembered as:
  - **`init -> pregrasp (retreat about 10 cm along the grasp direction) -> grasp -> action`**
- more precisely, the currently common parameters are:
  - `--candidate_target_local_x_offset_m -0.03`
  - `--approach_offset_m 0.08`
- together they can be interpreted as:
  - first shift the final grasp target backward by `3 cm` along the gripper local `+X` opposite direction
  - then generate an additional `pregrasp` pose another `8 cm` behind that grasp pose
  - at the macro level this matches your mental model of “retreat about 10 cm along the grasp direction before entering the grasp”

### 2.2 Current smooth / trajectory-generation semantics

The system is not just doing a direct two-point joint-space connection.
Instead it does:

1. define stage target EE/TCP poses:
   - `init`
   - `pregrasp`
   - `grasp`
   - `action`
2. interpolate in **EE pose / TCP pose space** between adjacent stages
3. solve IK **for each interpolated waypoint**
4. execute the resulting joint trajectory
5. if the stage-end error is still above threshold, do extra waiting / replanning

With the current parameters:

- `--planner_backend urdfik`
- `--urdfik_trajectory_mode cartesian_interp_ik`
- `--urdfik_cartesian_interp_steps -1`
- `--urdfik_cartesian_interp_auto_step_m 0.05`

this means:

- interpolate in EE/TCP space
- choose waypoint density adaptively by path length
- solve IK waypoint by waypoint
- i.e. **smooth first in Cartesian space, then map to joint space**

### 2.3 Existing “smooth” tools and their roles

There are at least two kinds of smooth handling in the project.

#### A. Planner/execution-side smoothing before or during execution

Mainly controlled by:

- `--urdfik_trajectory_mode cartesian_interp_ik`
- `--urdfik_cartesian_interp_steps`
- `--urdfik_cartesian_interp_auto_step_m`
- and execution-side waiting parameters:
  - `--settle_steps`
  - `--joint_target_wait_steps`

This affects:

- how the robot moves toward targets
- waypoint density
- whether extra convergence waiting happens

#### B. Post-execution replay / bundle smoothing

Existing scripts:

- `code_painting/replay_pose_debug_smooth.py`
- `code_painting/smooth_planner_outputs_from_pose_debug.py`
- `code_painting/batch_smooth_planner_outputs.sh`

These do:

- read saved `pose_debug.jsonl`
- remove near-duplicate hover frames
- interpolate between recorded states
- re-export smoother videos / pose bundles

Important note:

- this kind of smoothing improves **presentation quality**
- it does not improve the original run's execution accuracy
- and it does not change the original plan success rate

---

## 3. Why larger `joint_target_wait_steps` tends to create stronger visible jumps

The mechanism is:

1. the planned waypoint trajectory has finished
2. the robot is still not fully at target
3. the system enters the settle / wait loop
4. physics keeps advancing inside that loop
5. but the main export usually **does not record every settle step as a frame**
6. in the end it records a later “already converged” result frame

So the effect becomes:

- real execution: continuous gradual motion toward the target
- exported video: looks like the robot suddenly snaps close to the target

Therefore:

- the larger `joint_target_wait_steps` is
- the more visible motion is allowed to happen inside the non-frame-by-frame settle section
- the stronger the saved-video jump often becomes

---

## 4. Best methods to try without modifying code

All methods below only involve:

- parameter tuning
- workflow changes
- export / post-processing changes

No planner source modification is required.

### Method 1: lower `joint_target_wait_steps` while increasing path density

Idea:

- avoid leaving too much motion to the settle loop
- move more motion into the already planned and recorded waypoint section

Try:

- reduce `--joint_target_wait_steps`
- while also reducing `--urdfik_cartesian_interp_auto_step_m`
  - e.g. from `0.05` to `0.03` or `0.02`
- or switch to a denser fixed `--urdfik_cartesian_interp_steps`

Pros:

- most consistent with the current system design
- no code changes
- often directly reduces stage-end snap motion

Cons:

- execution time can increase
- a small settle jump may still remain
- very dense IK waypoints can introduce local jitter or more solver cost

Best for:

- cases where the main issue is video appearance, not total failure to reach
- when you can trade runtime for more even motion

### Method 2: keep `joint_target_wait_steps` small-to-medium and rely more on `replan_until_reached`

Idea:

- do not spend too long settling in one stage
- if still off-target, replan from the current pose
- use multiple short corrections instead of one long invisible settle

You are already using:

- `--replan_until_reached 1`
- `--replan_until_reached_max_attempts 3`

Suggested direction:

- keep `joint_target_wait_steps` moderate
- let final correction happen more through “next replans” than through one long settle tail

Pros:

- easier to avoid one large end-stage jump
- usually keeps accuracy better than removing waiting completely
- matches the current pipeline well

Cons:

- frequent replans can introduce more stage boundaries
- total runtime may increase
- some cases may look like short pause-and-go corrections

Best for:

- cases where reach errors are occasional, not catastrophic
- when you want controlled short corrections instead of long waiting tails

### Method 3: separate the roles of `settle_steps` and `joint_target_wait_steps`

Idea:

- `settle_steps` behaves like a coarse extra stabilization time
- `joint_target_wait_steps` is more about final joint-error convergence
- do not set both very large at the same time

Try:

- keep one as the main waiting source
- reduce the other
- test pairs such as:
  - low `settle_steps` + medium `joint_target_wait_steps`
  - medium `settle_steps` + low `joint_target_wait_steps`

Pros:

- helps diagnose whether the jump mainly comes from immediate post-trajectory instability or from slow final convergence
- useful for localizing which wait phase hurts video appearance most

Cons:

- still empirical tuning
- does not remove the structural issue that settle phases are not fully serialized into the main video

Best for:

- quick diagnosis with minimal workflow changes

### Method 4: prioritize pure-scene output, then run post-hoc smooth replay

Idea:

- let the original execution focus on success rate and accuracy
- let video appearance be handled afterward

Existing tools:

- `code_painting/replay_pose_debug_smooth.py`
- `code_painting/smooth_planner_outputs_from_pose_debug.py`

Recommended usage:

1. keep the more reliable accuracy-oriented planner parameters
2. export `pose_debug.jsonl`
3. run replay smooth / bundle smooth afterward

Pros:

- preserves original execution accuracy
- most reliable for presentation videos
- existing tools already available

Cons:

- smooths the presentation, not the original control process itself
- if a downstream task depends on the raw head/wrist timing semantics, source differences must be tracked carefully

Best for:

- cases where the main goal is a smoother exported video
- when execution success is more important than the first-pass video appearance

### Method 5: add more semantic intermediate targets and split long motions into shorter segments

Idea:

- current structure is `init -> pregrasp -> grasp -> action`
- if `grasp -> action` is long or has large orientation change
- even with Cartesian interpolation, IK / dynamics may still leave a lot of residual motion to the end
- workflow-level segmentation can reduce that

Conceptually split into:

- `init`
- `pregrasp`
- `grasp`
- `lift / stabilize`
- `pre-pour`
- `pour-end`

Even with the same planner backend, giving more intermediate targets often reduces per-segment residual error.

Pros:

- often the best way to improve both accuracy and appearance together
- shorter segments are easier for IK and execution
- closer to the real manipulation structure of complex dual-arm tasks

Cons:

- requires extra intermediate keypoints / keyframes
- not a one-parameter fix
- workflow becomes more complex

Best for:

- long `grasp -> action` motions with large translation or orientation change
- cases where you can afford more annotation or rule-generated intermediate poses

### Method 6: separate “evaluation parameters” from “presentation parameters”

Idea:

- use one conservative parameter set for judging real success / accuracy
- use another smoother parameter set for generating presentation videos

For example:

- evaluation run: larger `joint_target_wait_steps`
- presentation run: smaller `joint_target_wait_steps` + denser waypoint path

Pros:

- easiest way to satisfy both scientific evaluation and presentation quality
- does not force one run configuration to optimize two conflicting goals

Cons:

- requires managing two result tracks
- the presentation run may differ slightly from the evaluation run near the end states

Best for:

- workflows that need both rigorous evaluation and clean demo videos

---

## 5. My most practical recommendation right now

If you want to **avoid code changes entirely**, I would prioritize the following.

### Option A: keep execution robust, use post-processing for appearance

This is the most practical current choice.

Suggested approach:

1. keep the current robust reach strategy:
   - `replan_until_reached=1`
   - `replan_until_reached_max_attempts=3`
2. do not push `joint_target_wait_steps` too high
   - try a medium value first, not a very large one
3. if the video still jumps, use:
   - `replay_pose_debug_smooth.py`
   - or `smooth_planner_outputs_from_pose_debug.py`
   for post-processing

Why:

- this is the lowest-risk path within the existing project structure
- it cleanly decouples “accuracy / success” from “video appearance”

### Option B: reduce wait, increase path density

If you want the main exported video itself to look more natural, first try:

- lower `joint_target_wait_steps`
- lower `urdfik_cartesian_interp_auto_step_m`

Goal:

- move visible motion out of the settle phase and back into the recorded waypoint phase

This is the parameter sweep I would try first.

### Option C: add intermediate targets for long action segments

If `grasp -> action` is especially long or complicated, tuning wait and interpolation density may not be enough.
In that case the most effective lever is often not more wait, but:

- add 1 to 2 semantic intermediate poses
- split the long segment into shorter ones

That often improves all three at once:

- final accuracy
- pose stability
- video appearance

---

## 6. Suggested experiment order (no-code-change version)

### Group 1: sweep only wait

Fix:

- `urdfik_cartesian_interp_auto_step_m=0.05`
- everything else unchanged

Compare:

- `joint_target_wait_steps = 10 / 20 / 40 / 70`

Observe:

- final reach error
- whether stage-end jumps become obvious
- which stage is worst

### Group 2: with smaller wait, sweep path density

Fix one small or medium value of:

- `joint_target_wait_steps`

Then compare:

- `urdfik_cartesian_interp_auto_step_m = 0.05 / 0.03 / 0.02`

Observe:

- does the video become more continuous
- does runtime increase too much
- do you see local jitter / repeated correction

### Group 3: if needed, switch to post-hoc smoothing

If Groups 1 and 2 show that:

- good accuracy still requires relatively large wait
- and large wait necessarily causes visible saved-video jumps

then the practical conclusion is usually:

- **raw run for accuracy**
- **smooth replay for appearance**

That is usually more efficient than over-optimizing one conflicting parameter.

---

## 7. Analysis of your proposed method: dense 1 cm sampling + previous-solution seed + jump-threshold rejection

Your proposed method can be summarized as:

```python
Pose1 ---------------------------------> Pose3
sample one EE pose every 1 cm
IK(p1, seed=q0) -> IK(p2, seed=q1) -> ... -> IK(pN, seed=qN-1)
if adjacent joint changes exceed a threshold, reject the whole segment and replan
```

### 7.1 Relation to the current `urdfik_cartesian_interp_ik` mode

This is **not a completely different route** from the current implementation.
It is better understood as:

- **partially already present**
- **partially still missing as explicit hard constraints**

What already exists now:

1. the system already interpolates in EE/TCP Cartesian space rather than only doing a final joint interpolation
2. it already solves IK **waypoint by waypoint**
3. it already uses the **previous waypoint solution as the seed** for the next waypoint

So the core ideas already present in the current system are:

- `Cartesian waypoint interpolation`
- `IK waypoint-by-waypoint`
- `previous solution as seed`

The main new ingredients in your proposal, compared with the current code, are:

1. **much denser sampling**
   - e.g. explicit `eef_step = 0.01 m`
2. **stronger continuity enforcement**
   - not just “prefer previous seed”
   - but explicitly check `|q_i - q_{i-1}|`
3. **reject the whole segment on a jump**
   - the current logic is more about waypoint IK success/failure
   - your idea is: even if every waypoint solves, the segment should still be rejected if the joint branch changes too abruptly

So the right interpretation is:

- **a stricter / strengthened version of the current A2 (`cartesian_interp_ik`) path**

### 7.2 Advantages of this method

#### Advantage 1: it is the most compatible with the current execution chain

Because the system already does:

- Cartesian interpolation
- waypoint IK
- previous-seed chaining

this would not require throwing away the current structure.
It would mostly mean:

- extending the current A2 path with stricter constraints and denser sampling

That is the biggest practical advantage.

#### Advantage 2: it can strongly reduce IK-branch jumps

If every waypoint uses the previous solution as the seed, and the sampling is dense enough:

- the solver is more likely to stay on one local continuous branch
- it is less likely to suddenly switch to a very different joint configuration

This directly targets the “EE path looks smooth, but the joints suddenly flip the wrist/elbow branch” type of problem.

#### Advantage 3: it can reject bad trajectories earlier in planning

If you add checks such as:

- adjacent-waypoint joint-delta threshold
- approximate max single-step velocity / acceleration threshold

then many trajectories that are technically solvable but visually unnatural can be rejected before execution.

#### Advantage 4: it may reduce dependence on large end-stage waiting

If the trajectory itself is more continuous and easier for the arm to track:

- the executor is more likely to follow it directly
- less visible motion is left for `joint_target_wait_steps` to clean up afterward

That is directly relevant to your video-jump issue.

### 7.3 Disadvantages of this method

#### Disadvantage 1: runtime cost will increase

For example:

- 20 cm path
- one waypoint every 1 cm

means about 20 IK waypoints for a single segment. With dual arms and multiple stages, the cost grows noticeably.

Compared with the current coarser auto-step mode, this means:

- longer planning time
- lower batch throughput

#### Disadvantage 2: it can fail earlier on locally difficult regions

Dense sampling + jump threshold is stricter by design.
That means:

- some trajectories that are still physically usable may now be rejected
- even if they only contain a small but acceptable joint-branch adjustment

This can lead to:

- lower success rate
- more replanning / candidate fallback

#### Disadvantage 3: a good jump threshold is hard to set globally

Different segments have different expected motion ranges:

- `init -> pregrasp`
- `pregrasp -> grasp`
- `grasp -> action`

So the acceptable joint-delta range may not be the same across all segments.

If the threshold is too strict:

- valid trajectories are rejected

If it is too loose:

- real branch jumps slip through

#### Disadvantage 4: it is still not full trajectory optimization

Even with dense sampling + continuous seeding + jump checks, the method is still fundamentally:

- **local waypoint IK**

rather than:

- a global trajectory optimizer with smoothness, velocity, acceleration, and collision constraints

So it can greatly improve continuity, but it is not the theoretically strongest formulation.

### 7.4 Implementation difficulty relative to the current method

#### Difficulty: medium

Why:

- it is not a from-scratch planner rewrite
- the existing `cartesian_interp_ik` mode is already very close to this design
- the main additions would be:
  1. a more explicit Cartesian sampling policy (e.g. fixed `eef_step=1 cm`)
  2. joint continuity metrics
  3. segment rejection / retry logic after threshold violation
  4. summary/debug reporting for why a segment was rejected

In other words:

- **much easier than replacing the stack with a global trajectory optimizer**
- **but clearly more involved than simple parameter tuning**

If I classify it by difficulty, I would rate:

- low: only tune `auto_step_m / wait_steps / replan`
- **medium: add dense sampling + jump-threshold continuity checks on top of current A2**
- high: move to real trajectory optimization with smoothness constraints

---

## 8. Other new methods worth considering while keeping the current `run_plan_anygrasp_keyframes_r1_batch.sh` workflow

### Method A: fixed-step dense sampling (1 cm) instead of the current coarser auto-step

Essence:

- still the current `cartesian_interp_ik`
- but with a more stable, explicit fixed waypoint spacing

Pros:

- most direct
- matches your current idea best
- easy to reason about and debug

Cons:

- can become slow on long paths
- if orientation changes are large but translation is small, pure 1 cm spacing may still be insufficient

Implementation difficulty: **low to medium**

### Method B: dual-threshold sampling for both position and rotation

Essence:

- do not sample only by translation distance
- also sample by maximum orientation change
- for example:
  - at least one waypoint every 1 cm or every 5°

Pros:

- more principled than pure `1 cm`
- especially suitable for tasks like pouring where orientation changes matter a lot

Cons:

- more complicated sampling rules
- more parameters to tune

Implementation difficulty: **medium**

### Method C: add a joint-space continuity filter after waypoint IK

Essence:

- first solve the whole waypoint-IK segment as usual
- then check:
  - adjacent `q` deltas
  - per-joint max change
  - overall norm change
- reject the segment if too large

Pros:

- directly matches your jump-threshold idea
- most direct defense against branch jumps
- relatively small intrusion into the current structure

Cons:

- can only detect and reject, not automatically repair
- if the candidate or initial seed is bad, failures may become frequent

Implementation difficulty: **medium**

### Method D: add soft smoothness preferences / costs inside waypoint IK

Essence:

- when solving each waypoint, do not only minimize pose error
- also prefer solutions that:
  - stay close to previous `q`
  - stay away from joint limits
  - remain near a comfortable nominal posture

Pros:

- softer and more graceful than hard rejection thresholds
- more likely to automatically choose a continuous branch instead of jumping first and rejecting later

Cons:

- depends heavily on how much control the underlying IK solver exposes
- if the solver API is restrictive, implementation can become awkward

Implementation difficulty: **medium to high**

### Method E: waypoint IK first, then joint-trajectory smoothing / shortcutting

Essence:

- first obtain `q1...qN`
- then post-process in joint space with:
  - smoothing filters
  - shortcutting
  - velocity / acceleration clipping

Pros:

- conceptually clear
- can be decoupled from the current pipeline
- easy to compare in debugging

Cons:

- after smoothing, the TCP path will drift away from the original Cartesian path
- without FK / error re-checking, grasp precision may be harmed

Implementation difficulty: **medium**

### Method F: split long actions into more semantic intermediate poses

Essence:

- do not change the solver first
- reduce the per-segment difficulty itself

Pros:

- often very effective with the current system
- frequently one of the highest-return strategies
- especially friendly for long dual-arm motions

Cons:

- needs a source of intermediate poses
- increases workflow complexity

Implementation difficulty: **low to medium** (algorithmically low, data/workflow-wise medium)

### Method G: switch to real trajectory optimization / motion generation

Essence:

- no longer only waypoint IK
- directly optimize a global constrained trajectory

Pros:

- theoretically the most complete
- easiest place to integrate smoothness, speed limits, collisions, and similar constraints in a unified way

Cons:

- biggest code and system change
- not a small extension of the current urdfik route inside `run_plan_anygrasp_keyframes_r1_batch.sh`
- high debugging cost

Implementation difficulty: **high**

---

## 9. Difficulty ranking and recommendation order under the current IK method

Given that:

- current A2 already is Cartesian interpolation + waypoint IK + previous-seed chaining
- what is missing today is mainly stricter continuity control and clearer rejection logic

I would rank the next steps like this.

### Best first candidates

1. **fixed-step or dual-threshold dense sampling**
2. **adjacent-waypoint joint-jump threshold detection and rejection**
3. **if needed, add more semantic intermediate poses**

Why:

- these are the most compatible with the current structure
- they have the best expected return-vs-complexity ratio

### Second tier

4. **soft continuity preferences inside IK**
5. **joint smoothing after waypoint IK, plus FK error re-checking**

Why:

- promising
- but more uncertain in implementation detail than the first group

### Leave for later

6. **switch to true global trajectory optimization**

Why:

- highest cost
- not the best first move for the current “video jump + continuity” issue

---

## 10. Extra analysis for your V7 debug command

The key characteristics of your current V7 command are:

- `urdfik_cartesian_interp_auto_step_m=0.01`
- `execute_interp_steps=24`
- `joint_command_scene_steps=4`
- `settle_steps=30`
- `joint_target_wait_steps=15`
- `joint_target_wait_tol_rad=0.01`
- `replan_until_reached=1`
- `replan_until_reached_max_attempts=3`
- `execute_both_arms=1`
- `debug_visualize_ik_waypoints=1`

So this is already a configuration with relatively dense waypoint sampling and stronger emphasis on execution tracking, yet you still feel that the motion is not smooth enough, and that without `try` / replanning the final pose is not accurate enough.

### 10.1 Why accuracy drops when you do not use try / replanning

This usually does not mean the target pose suddenly became wrong.
The deeper issue is that in the current system, “planning to the target” and “ending up at the target after real execution” are not the same thing.

There are at least several error sources in the current pipeline:

1. **waypoint IK is still local solving**
   - solving every waypoint does not guarantee that real execution will land exactly on the final pose

2. **the execution layer has tracking lag**
   - `joint_command_scene_steps=4` still gives only limited physics time to each joint command
   - especially with dual arms, collisions, and attached objects, the drives may not fully catch up

3. **stage completion is judged by reach tolerance, not by the theoretical planned endpoint**
   - even if the FK endpoint of the planned joints looks close to the target
   - the real executed `attempt` may still violate `reach_pos_tol_m / reach_rot_tol_deg`

4. **dual-arm synchronization and collision constraints amplify residual error**
   - you enabled:
     - `execute_both_arms=1`
     - `grasp_action_object_collision_start_stage pregrasp`
     - `execution_object_collision_mode convex`
     - `gripper_contact_monitor_mode all_robot_links`
   - this makes the real execution more conservative and more likely to keep residual error than a simpler single-arm case

So the real reason “without try it is not accurate enough” is usually:

- **one plan + one execution pass only gets the system near the target**
- **the last few centimeters / degrees still need either extra convergence or another correction pass from the current real state**

### 10.2 Why try / replanning improves accuracy but hurts visual smoothness

Because try / replanning is effectively doing this:

1. execute one segment
2. measure the real post-execution error
3. if the segment is still outside tolerance, replan again from the current real pose

It improves accuracy because:

- the second or third pass starts closer to the target
- it only has to correct the residual error
- that is often more robust than forcing one long segment to land perfectly in one shot

But it also makes the exported video less smooth for two reasons.

#### Reason A: you get a segmented “short correction” rhythm within one stage

That means:

- move a bit
- stop
- decide not reached
- move again
- stop again

Even if each individual short segment is smooth, the full stage looks segmented.

#### Reason B: each try can still end with a small unrecorded settle tail

Even with `joint_target_wait_steps=15`:

- if some visible motion still happens inside the settle loop
- the main exported video can still present that as a small end-stage snap

So the try mechanism is not necessarily “wrong”.
It is better understood as:

- **good for final reach accuracy**
- **not inherently ideal for producing the visually smoothest single-shot video**

### 10.3 What your feeling “maybe try has a problem” more precisely means

A more precise statement is not:

- `try` itself is buggy

but rather:

- in the current system, `try` is acting as the execution-error compensator
- which indicates that the previous plan+execute pass is still not sufficient to bring the real system inside tolerance in one shot

So if you observe:

- without try: it often remains slightly off
- with try: it gradually converges into tolerance

that usually means the main issue lies in:

1. **insufficient execution tracking**
2. **a waypoint trajectory that is theoretically valid but not easy enough for the real executor to follow perfectly**
3. **dual-arm / collision / object constraints making the last bit of error hard to eliminate in one pass**

So the existence of try is not surprising. It actually means the system is closer to:

- a **plan-execute-correct** loop

than to:

- a single open-loop trajectory that is already accurate enough by itself

### 10.4 How to balance “execution accuracy” and “non-jumpy video” without code changes

If you insist on not modifying code, the most realistic balancing strategies are:

#### Option A: keep try, but reduce how much visible motion is left for settle inside each try

Idea:

- keep `replan_until_reached`
- but try to make more of each correction happen inside the recorded waypoint section
- and less inside the unrecorded wait / settle tail

Direction:

- keep `joint_target_wait_steps` moderate
- keep waypoint density reasonably high
- avoid using large `joint_target_wait_steps` as the main way to grind out accuracy

This is the most realistic compromise.

#### Option B: split long motions into more intermediate poses instead of relying on try at the tail of a long segment

If the main residual error is concentrated in:

- `grasp -> action`

then try can look “problematic” simply because:

- the segment itself is too long
- the orientation change is too large
- the residual error accumulates near the end

So the most natural improvement is often:

- add semantic intermediate poses
- reduce the per-segment span

That usually helps both:

- final accuracy
- visual continuity

#### Option C: let raw execution optimize for accuracy, and let replay / bundle smoothing optimize for appearance

If you finally confirm that:

- without try, accuracy is insufficient
- with try, accuracy is good but the video is less natural

then the most engineering-realistic conclusion is:

- **raw execution serves reach accuracy**
- **the presentation video is produced by replay / bundle smoothing**

This is not “avoiding the problem”; it acknowledges that:

- the current main-video recording mechanism is not naturally good at expressing the fine continuous motion inside settle / correction loops

---

## 11. The new diagnostic quantity you want: distance to the target forward axis

You already have:

- `dx/dy/dz/dist`
- `rot`
- `fwd_rot`
- `fwd_cm`

Where:

- `fwd_cm` is the position error projected onto the **target gripper forward axis** (local `+X`)
- it answers:
  - how far ahead / behind the current pose is along the target forward axis

But it does **not** answer:

- how far the current point is from the target forward axis line itself
- i.e. how much lateral bias exists relative to the intended approach line

That extra metric is very reasonable.
It can be defined as follows.

### 11.1 Proposed metric

Given:

- target position: `p_target`
- current position: `p_current`
- target gripper forward-axis unit vector: `a_target` (currently local `+X`)

First define the position error:

- `e = p_current - p_target`

Then decompose it into two parts:

1. **axial error (already present conceptually)**
   - `e_parallel = dot(e, a_target)`
   - this is essentially the current `fwd_cm` idea

2. **lateral distance to the target forward-axis line (the new quantity you want)**
   - `e_perp = || e - dot(e, a_target) * a_target ||`

Possible names:

- `lateral_to_target_forward_axis_cm`
- `target_forward_axis_offset_cm`
- Chinese equivalent:
  - distance to target forward axis
  - lateral offset from target forward axis

### 11.2 What this metric tells you

It separates “not reached” into two categories:

1. **axial misalignment**
   - too far ahead / too far behind
   - use `fwd_cm`

2. **lateral deviation from the approach line**
   - the gripper is not staying on the line it should approach along
   - use `e_perp`

This is especially useful for grasp debugging because many cases look like:

- `dist` is not large
- `fwd_cm` is also not large
- but the gripper is still sliding past the intended forward axis instead of centering onto it

That failure mode would be exposed immediately by `e_perp`.

### 11.3 Relation to existing metrics

If the position error vector is `e`, then:

- `dist = ||e||`
- `fwd_cm = dot(e, a_target)`
- `axis_lateral_cm = ||e - dot(e, a_target)a_target||`

So the relation is:

- `dist^2 = fwd^2 + lateral^2`

That means:

- `dist` = total position error
- `fwd_cm` = axial component
- `axis_lateral_cm` = lateral component

All three together are much more informative than only `dist` or only `fwd_cm`.

### 11.4 Most helpful printing format

This is now implemented and printed as:

- `dist`
- `fwd_cm`
- `lat_cm`
- `fwd_rot`

Where:

- `lat_cm`
  - is the lateral distance from the current point to the target forward-axis line, in centimeters
  - a larger value means the gripper is drifting farther away from the intended approach line

This metric is now included in:

- single-arm `plan-request`
- single-arm `plan-solution`
- single-arm `attempt`
- single-arm `attempt-supervision`
- dual-arm `plan-request`
- dual-arm `plan-solution`
- dual-arm `attempt`
- and the stored `attempt_history` / supervision-error structures

So you can immediately distinguish:

- front/back miss
- lateral miss from the intended axis
- orientation mismatch

### 11.5 Implementation difficulty estimate

I would rate this additional diagnostic metric as:

- **low difficulty**

because it does not require changing the planning logic itself.
It only needs:

- the current pose (already available)
- the target forward axis (already available conceptually)
- one more decomposition of the position error vector

So this is essentially:

- a high-value, low-intrusion diagnostic metric

---

## 12. Current conclusion

If you want to try new methods while **keeping the current `run_plan_anygrasp_keyframes_r1_batch.sh` main workflow**, the most natural direction is:

1. keep the current:
   - `init -> pregrasp -> grasp -> action`
   - EE/TCP pose interpolation
   - waypoint-by-waypoint IK
2. strengthen it with:
   - **denser sampling**
   - **explicit joint continuity constraints**
   - **segment rejection and replanning when thresholds are violated**

So your proposal is not a separate planner family from the current system. It is better described as:

- **one of the most reasonable upgrade directions for the current `cartesian_interp_ik` mode**

In addition, for your V7 debug observations, my current judgment is:

- **if it is not accurate enough without try, that means the current system really does need post-execution correction loops;**
- **if larger `joint_target_wait_steps` improves accuracy but makes the video jumpy, that is because more visible motion is being left inside settle phases that are not serialized frame by frame;**
- **to balance both goals, the most realistic path is not “just keep increasing wait”, but to reduce tail residuals, reduce the span of long segments, and add diagnostics that explain the error structure better — especially the lateral offset from the target forward axis.**

If I had to compress it to one line:

- **start with denser sampling + jump-threshold rejection + the new lateral-to-forward-axis diagnostic;**
- **that is far more realistic than full global trajectory optimization, and it targets the real continuity problem much more directly than tuning wait parameters alone.**

---

## 13. How to construct an object-relative transform that keeps hand-object geometry but improves robot executability

Your goal can be summarized as:

1. keep the current main workflow and all existing features working
2. do not replace the current `run_plan_anygrasp_keyframes_r1_batch.sh` pipeline
3. when implemented later, integrate it through **new functions or a new file**, not by rewriting the current path
4. because human-hand demonstrations and robot manipulation spaces are heterogeneous, add an **object-relative target transform layer** so the robot plans toward a more executable and smoother target family

This is a very reasonable direction. It is importantly different from “directly changing the IK solver.”
The key idea is not to modify the solver first, but to:

- **project the human-demonstration target into a robot-friendly target family before IK**

### 13.1 The most natural modeling view: work in the object frame

Right now you already have:

- object pose in world frame: `T_world_obj`
- human-hand / grasp target pose in world frame: `T_world_hand`

The most natural first step is not to hack the world-frame pose directly, but to move into the object frame:

- `T_obj_hand = inv(T_world_obj) @ T_world_hand`

This quantity represents:

- **the hand pose relative to the object**

If you want to “keep the left/right hand and object relative relationship unchanged,” this is the structure that should be preserved conceptually — not the absolute world pose.

### 13.2 Core construction: object-relative pose + robot executability correction

I recommend splitting the new target construction into two layers.

#### Layer 1: preserve the human semantic relation

First extract from the demonstration:

- left hand relative to cup: `T_cup_left_hand_demo`
- right hand relative to bottle: `T_bottle_right_hand_demo`

This preserves:

- human manipulation intent
- contact-region semantics
- the functional relation between hand and object during grasping / pouring

#### Layer 2: add a robot-specific correction transform

Then do **not** feed this object-relative demo pose directly to the robot.
Instead introduce an additional correction:

- `Δ_left_robot`
- `Δ_right_robot`

and define:

- `T_cup_left_robot_target = T_cup_left_hand_demo @ Δ_left_robot`
- `T_bottle_right_robot_target = T_bottle_right_hand_demo @ Δ_right_robot`

Finally convert back to the world frame:

- `T_world_left_target = T_world_cup @ T_cup_left_robot_target`
- `T_world_right_target = T_world_bottle @ T_bottle_right_robot_target`

This `Δ_robot` is the right place to put the “overall coordinate transform” you are looking for.

### 13.3 Why this is better than directly shifting world-frame targets

Because your real problem is usually not:

- the world-frame target is off by a small absolute translation

The deeper issue is:

- human hand morphology and robot gripper kinematics differ
- a human-comfortable wrist pose is not necessarily robot-comfortable
- but the **functional relation to the object** is what you actually want to preserve

So the right principle is:

- **preserve object-relative task semantics**
- **add a robot-specific correction in the object frame**
- **do not just add ad hoc world-frame offsets**

---

## 14. How should `Δ_robot` be constructed?

I recommend designing it from simple to advanced.

### 14.1 Simplest deployable version: constant rigid correction

That means each arm gets a fixed SE(3) correction:

- left: `Δ_left_robot = [R_left, t_left]`
- right: `Δ_right_robot = [R_right, t_right]`

Where:

- `t`: a small translation in the object-local frame
- `R`: a small rotation in the object-local frame

Interpretation:

- the human hand may sit closer to the object wall, with more inward wrist posture
- the robot gripper may need to stay slightly farther back, more upright, or closer to a preferred grasp orientation
- so the robot always applies a fixed correction for that object/task family

Pros:

- most stable
- easiest to validate first
- can be integrated later as a new function/file without breaking the current chain

Cons:

- limited generalization
- may not fit different object sizes, stages, or motion phases equally well

Implementation difficulty: **low**

### 14.2 Better version: stage-specific correction

Different stages often need different corrections:

- `pregrasp`
- `grasp`
- `action`

So you can define:

- `Δ_left_pregrasp`
- `Δ_left_grasp`
- `Δ_left_action`
- `Δ_right_pregrasp`
- `Δ_right_grasp`
- `Δ_right_action`

This is usually more reasonable than one correction for everything.

For example:

- `pregrasp`: farther from the object, more conservative orientation
- `grasp`: align with the contact region
- `action`: prioritize reachability and stable pouring motion

Pros:

- better matches the actual task structure
- easier to balance success rate and smoothness than one fixed correction

Cons:

- more parameters
- more tuning work

Implementation difficulty: **low to medium**

### 14.3 More adaptive version: depend on object type or size

If later you find that:

- cup / bottle / bowl
- large cup / small cup
- tall bottle / short bottle

need different corrections, then let `Δ_robot` depend on:

- object category
- bounding-box size
- stage

For example:

- `Δ = f(object_type, bbox_size, stage, arm)`

This `f` does not have to be a learned model; it can still be a rule-based function.

Pros:

- stronger adaptability

Cons:

- rules can grow quickly
- more maintenance/documentation cost

Implementation difficulty: **medium**

### 14.4 Strongest but most complex version: local executability projection

A more advanced idea is:

- start from `T_obj_hand_demo`
- search for a small local correction `Δ` around it
- choose a new target that is:
  - easier for IK to solve
  - smoother / more continuous
  - farther from joint limits
  - less likely to cause branch jumps

This can be written as a local optimization:

- minimize:
  - `pose_deviation_cost(Δ)`
  - `ik_difficulty_cost`
  - `joint_jump_cost`
  - `distance_to_nominal_cost`
- subject to:
  - correction size remains small
  - grasp / task semantics remain valid

This is essentially:

- **project the human-hand target onto a robot-executable target manifold**

Pros:

- most aligned with your real goal
- can improve success rate and smoothness simultaneously

Cons:

- highest complexity
- effectively becomes a new local optimization module

Implementation difficulty: **high**

---

## 15. What relative relations should actually be preserved?

You said “keep the relative position between the robot left/right hands and the object unchanged.”
That needs to be handled carefully.

- **I do not recommend preserving the full 6D relative pose as a hard equality constraint.**
- It is better to preserve the functionally important parts and relax the robot-hard parts.

### 15.1 What is most worth preserving for grasping / pouring

Prefer to preserve:

1. approach direction / contact normal
2. relative placement around the intended contact region
3. relative height along the object's main axis
4. for pouring, the relation between cup opening / bottle opening orientations

### 15.2 What should be relaxed first

Prefer to relax:

1. roll around the gripper forward axis
2. small lateral translations
3. small retreat / advance offsets along the forward axis

Why:

- these often matter less to the task semantics
- but matter a lot to robot reachability and motion smoothness

In other words, the right goal is not:

- “copy the human hand-object pose exactly”

but rather:

- “preserve task semantics while applying the smallest robotization correction needed”

---

## 16. The safest module structure for later implementation

You explicitly want to keep all existing code/features working and add the new behavior through new functions or files.

Under that requirement, the safest structure is:

### 16.1 Add an independent target-adapter layer

For example, later you could add:

- new file:
  - `code_painting/object_relative_target_adapter.py`
- new functions:
  - `adapt_demo_hand_pose_in_object_frame(...)`
  - `build_robot_target_from_object_relative_pose(...)`
  - `apply_robot_morphology_delta(...)`

Then, right before the current planner consumes a target pose, insert one extra layer:

- raw demo / candidate target
- -> object-frame relative pose
- -> robot-corrected relative pose
- -> world-frame robot target pose
- -> existing `plan_path(...)`

This is attractive because:

- the old logic can remain intact
- the old flags can keep working
- the new behavior can be isolated behind new flags

### 16.2 Do not start by modifying the IK solver core

Because what you really want to add is not first:

- a new numerical solver trick

but rather:

- **a target-construction policy**

If you inject this semantics directly into `urdfik.py`, you will tightly couple:

- solver internals
- task semantics
- morphology adaptation logic

That will make debugging and A/B comparison harder.

So I strongly recommend:

- **add a target adapter before the solver**
- **do not start by modifying the solver core**

---

## 17. The most practical first version

If the question is “what is the best first construction?”, I would recommend this:

### Version S1: object-relative constant correction

For each arm and each stage:

1. compute:
   - `T_obj_hand_demo`
2. apply a small constant correction in the object frame:
   - translation:
     - retreat a bit along the target forward axis
     - small adjustment along normal / tangent directions
   - rotation:
     - reduce unnecessary wrist roll
     - move the gripper closer to a robot-preferred natural posture
3. transform back to world frame
4. run the existing `cartesian_interp_ik`

This fits your current project state best because:

- it minimally intrudes on the existing functionality
- it is conceptually aligned with `candidate_target_local_x_offset_m` and `approach_offset_m`, but more principled
- it can later be extended to stage-specific correction or local optimization projection

---

## 18. Final recommendation for this direction

If we stay at the analysis stage and do not modify code yet, my recommendation is:

1. **do not add a hard “full hand-object relative pose must remain unchanged” constraint inside IK**
   - that is too rigid and will often make the robot harder to solve
2. **instead, express the demo target in the object frame, then add a robot-specific correction**
3. **preserve the functionally important geometry, and relax roll + small translations first**
4. **add an independent target-adapter layer instead of editing the solver first**
5. **the best first version is an object-relative constant or stage-specific correction**

If I had to compress it to one line:

- **the right construction is not “feed the human-hand pose directly into IK and hard-constrain it unchanged,” but “convert the hand pose into the object frame, apply a robotization correction `Δ_robot`, project it onto a more executable target pose, and then keep using the existing planner/IK chain.”**
