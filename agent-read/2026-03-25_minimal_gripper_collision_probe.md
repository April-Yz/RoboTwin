# 2026-03-25 Minimal gripper/object collision probe

## Goal

This note isolates the following question without going through AnyGrasp candidate selection, IK planning, or the `pregrasp/grasp/action` pipeline:

- Does the R1 gripper actually generate physical robot-object contacts during `close_gripper`?
- Why does the current debug path in `plan_anygrasp_keyframes_r1.py` keep printing robot-link `shapes=0` / `contact=0`?

## Added script

- `code_painting/minimal_gripper_collision_probe.py`

## Script behavior

The minimal experiment keeps only:

1. load the R1 robot into a SAPIEN scene
2. skip planner attachment; keep only the articulation and gripper drives
3. create a probe object:
   - default: simple box visual + box collision
   - optional: mesh visual + `convex/solid_bbox` collision
4. place the probe directly at a local offset in front of the current TCP
5. close one gripper from `open -> close`
6. print at every step:
   - gripper joint qpos
   - total raw scene contacts
   - raw contact pairs involving the target object
7. also print a component/collision summary for robot links

## Key findings

### 1. The current helper-based robot-link collision summary is not reliable

The script still reports:

- `links_with_nonzero_component_shapes=0`

which matches the main-pipeline debug output:

- `left_gripper_link(shapes=0)`
- `left_gripper_finger_link1(shapes=0)`
- `all_robot_links ... shapes=0`

However, the raw-contact experiments below show that:

- even when the helper/summary says `shapes=0`
- real robot-object contacts can still exist in the physics scene

So the updated interpretation is:

> `shapes=0` is more likely a limitation of the current debug helper for articulation links, not proof that the robot has no collision geometry or that no physics contact exists.

### 2. The minimal box probe produces stable raw finger-object contacts

Command:

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  code_painting/minimal_gripper_collision_probe.py \
  --arm left \
  --object_kind box \
  --probe_local_offset 0.04 0.0 0.0 \
  --max_iters 20 \
  --settle_steps_per_iter 8
```

Observed contacts:

- from `iter=2` onward:
  - `left_gripper_finger_link1<->probe_box`
- later also:
  - `left_gripper_finger_link1<->probe_box`
  - `left_gripper_finger_link2<->probe_box`

This means:

- in a minimal scene, robot-object physical contact does exist
- raw contacts between finger links and a box probe can be captured by `scene.get_contacts()`

### 3. The minimal mesh + `solid_bbox` probe also produces stable raw contacts

Command:

```bash
/home/zaijia001/ssd/miniconda3/envs/RoboTwin_bw/bin/python \
  code_painting/minimal_gripper_collision_probe.py \
  --arm left \
  --object_kind mesh \
  --mesh_path /home/zaijia001/ssd/data/R1/hand/obj_mesh/blue_cup/blue_cup.obj \
  --mesh_collision_mode solid_bbox \
  --probe_local_offset 0.04 0.0 0.0 \
  --max_iters 20 \
  --settle_steps_per_iter 8
```

Observed contacts include:

- early:
  - `left_gripper_finger_link1<->probe_mesh`
- later:
  - `left_gripper_finger_link2<->probe_mesh`
  - `left_gripper_link<->probe_mesh`
  - `left_realsense_link<->probe_mesh`

This means:

- even with the more main-pipeline-like `mesh + solid_bbox` setup
- the robot and object can still produce raw contacts in the minimal scene

## Updated interpretation for the main pipeline

Given the minimal probe result, the main-pipeline symptom

- “the video looks interpenetrated, but the log stays at `contact=0`”

can no longer be explained simply by:

- the robot having no collision geometry
- or `solid_bbox` objects having no collision at all

A more plausible updated explanation is:

### A. The current helper-based contact debug path has a blind spot for articulation links

That is:

- `shapes=0` does not imply that real physics collision is absent
- the current `_get_actor_collision_shapes(...)` / entity summary path is unreliable for articulation links

### B. The main-pipeline monitor/contact matching is inconsistent with the raw-physics result, or is broken by timing/pose issues

Since raw contacts clearly exist in the minimal probe but remain `monitor_contact=0` in the full pipeline, the next checks should focus on:

1. whether the target object pose during `close_gripper` is really the same as what the video shows
2. whether the object actor is overwritten/reset by other logic during close
3. whether the entity identity used by the monitor path does not match the raw-contact bodies
4. whether there is a timing mismatch between collision enable, object pose updates, and the rendered frames

## Current most important conclusion

1. **`shapes=0` should no longer be treated as evidence that the robot has no collision geometry.**
2. **The minimal isolation experiment proves that raw physics contacts between the R1 gripper and both a box probe and a mesh(`solid_bbox`) probe can occur and be captured.**
3. **Therefore, the full-pipeline `contact=0` symptom is now more likely a main-pipeline monitoring/matching/timing problem, or a mismatch between the object's actual close-stage pose and the visual impression from the video, rather than a complete lack of robot-object contact support in physics.**

## Output files

- `code_painting/minimal_gripper_collision_probe/probe_left_box.json`
- `code_painting/minimal_gripper_collision_probe/probe_left_mesh.json`

These JSON files store step-by-step raw contacts and qpos evolution for later comparison against the full planner pipeline.

## Main-pipeline reinjection check

After adding raw target-contact printing to `plan_anygrasp_keyframes_r1.py`, I reran the main close-stage debug command (with viewer/video export disabled for speed).

Representative command:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_debug_collision_batch_raw_probe \
  --ids 0 \
  --skip_existing 0 \
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
  --candidate_target_local_x_offset_m -0.05 \
  --approach_offset_m 0.08 \
  --reach_error_pose_source tcp \
  --replan_until_reached 1 \
  --replan_until_reached_max_attempts 3 \
  --enable_grasp_action_object_collision 1 \
  --execution_object_collision_mode solid_bbox \
  --debug_collision_report 1 \
  --gripper_contact_monitor_mode all_robot_links \
  --enable_viewer 0
```

### New observation

The main pipeline now additionally prints:

- `raw_target_contacts=...`
- `raw_target_contact_total=...`
- `[gripper-close] ... raw_target_contact=...`

For this case, the close stage shows:

- init: `raw_target_contacts=['none']`
- iter 1~20: `raw_target_contact_total=0`
- final: `raw_target_contact=0`

### What this means

This sharply narrows the issue further:

1. In the minimal isolation experiment, raw contacts can be observed stably.
2. But in the full pipeline close stage, even after bypassing the monitor helper and looking directly at `scene.get_contacts()`, there are still zero raw target contacts.

So the stronger current conclusion is:

> In this `d_pour_blue_0` close stage, the problem is not merely that the monitor/helper fails to recognize an existing contact; raw physics contact with the target object is not happening at all.

This shifts the primary suspicion toward:

- the true close-stage object pose/orientation/extent being different from what the video impression suggests
- or the object being outside the region that would generate raw finger/link contact in the main pipeline
- rather than “physics contact exists but the debug monitor fails to match it”

### Updated conclusion relative to the previous round

The previous minimal probe only established that:

- the physics engine and the R1 gripper/object contact mechanism can work in principle

This reinjection check now additionally establishes that:

- for the current `d_pour_blue_0` close stage, the issue is not just monitor under-reporting
- because even raw target contacts are zero

So the next most valuable checks are:

1. the true world pose of the object actor during close
2. whether the actor pose remains consistent with the visually perceived location after object collision is enabled
3. whether the visual “interpenetration” impression comes from visual meshes being misaligned with the actual collision primitive
4. whether candidate/target offsets are driving the hand into a relationship that looks close in the video but never actually enters the target collision volume

## Added close-stage object pose / collision debug

This round further adds to the main pipeline:

- the current target-object actor pose
- the target-object collision debug information (including `solid_bbox` `center/half_size`)

Representative output now includes:

- `target_pose=planned_object_cup(p=[...],q=[...])`
- `target_collision_debug={... 'center': [...], 'half_size': [...]}`

### Direct observations for the current `d_pour_blue_0`

Left-hand close-stage init:

- cup actor pose:
  - `p=[-0.1836, 0.0499, 0.8385]`
- left gripper-base pose:
  - `p=[-0.2086, -0.0007, 0.9410]`
- cup solid bbox:
  - `center=[0.0, 0.053424, 0.0]`
  - `half_size=[0.040001, 0.053424, 0.039616]`

Right-hand close-stage init:

- bottle actor pose:
  - `p=[0.0799, 0.0772, 0.8432]`
- right gripper-base pose:
  - `p=[0.0717, 0.0533, 0.9032]`
- bottle solid bbox:
  - `center=[0.0, 0.100586, 0.0]`
  - `half_size=[0.033206, 0.100586, 0.032682]`

### Additional conclusion

Combining the fact that raw target contact stays zero with the fact that the target pose remains essentially stable during close, we can now make a stronger statement:

1. the current close stage is not a simple case of “some other logic keeps resetting the object and destroys contact”
2. it looks more like there is a substantial geometric mismatch between the object's visual appearance and its collision primitive (especially under `solid_bbox`)
3. in other words, the video may look like the fingers pass through the object mesh, while relative to the actual `solid_bbox`, the fingers may never enter the region that would generate raw contacts

So the next most valuable step is no longer just more contact logging, but rather:

- directly visualizing the object collision bbox
- or printing/exporting the finger/base pose relative to the object bbox center/extents

## Object collision-bbox visualization

This round adds a new flag:

- `--debug_visualize_object_collision_bbox 1`

Behavior:

- when an execution object uses `solid_bbox` collision
- create an extra visual-only bbox actor for that object
- the bbox actor uses the same local `center/half_size` as the collision primitive
- the bbox actor follows the execution object's world pose

Purpose:

- directly compare in the viewer / exported videos:
  - the object's visual mesh
  - the object's collision bbox
- determine whether the apparent wrist/head-view interpenetration is:
  - through the visual mesh only
  - or also through the collision bbox

### Validation

The pipeline was reinjected with:

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_debug_collision_batch_bbox_probe \
  --ids 0 \
  ... \
  --execution_object_collision_mode solid_bbox \
  --debug_collision_report 1 \
  --debug_visualize_object_collision_bbox 1 \
  --gripper_contact_monitor_mode all_robot_links \
  --enable_viewer 0
```

The run completed successfully and did not introduce a new execution error.

### What you should inspect next

In the next round, you should directly inspect:

- `anygrasp_single_debug_collision_batch_bbox_probe/d_pour_blue_0/head_cam_plan.mp4`
- and, if you rerun with the viewer enabled, also compare the green bbox against the object mesh in wrist/head views

The key question is:

1. whether the finger only appears to pass through the visual mesh but never enters the green bbox
2. or whether the finger visibly enters the green bbox as well while raw contact still stays zero, in which case we should continue debugging at the physics/contact layer

## `convex` comparison experiment

This round directly compared the main-pipeline execution-object collision modes:

- `solid_bbox`
- `convex`

Representative command used in this round (`convex`):

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_debug_collision_batch_convex_probe \
  --ids 0 \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --candidate_selection_mode planner \
  --left_target_object cup \
  --right_target_object bottle \
  --candidate_target_local_x_offset_m -0.05 \
  --approach_offset_m 0.08 \
  --reach_error_pose_source tcp \
  --enable_grasp_action_object_collision 1 \
  --execution_object_collision_mode convex \
  --debug_collision_report 1 \
  --gripper_contact_monitor_mode all_robot_links \
  --enable_viewer 0
```

### Comparison result

Under `convex`, the close stage still reports:

- init: `raw_target_contacts=['none']`
- iter 1~20: `raw_target_contact_total=0`
- final: `raw_target_contact=0`

At the same time:

- the cup target shape type becomes `PhysxCollisionShapeConvexMesh`
- the bottle target shape type becomes `PhysxCollisionShapeConvexMesh`

In other words:

> even after switching the execution-object collision from `solid_bbox` to `convex`, the current main-pipeline close stage still produces no raw target contact.

### What this means

This is an important control because it largely rules out the explanation that:

- “the issue is only due to the `solid_bbox` approximation being too coarse / too large / too conservative”

The stronger current conclusion is:

1. the issue is not limited to the `solid_bbox` box approximation
2. even with `convex` mesh collision, the full close stage still produces no raw contact
3. therefore the main contradiction is more likely that:
   - the current hand/object geometric relationship in the full pipeline never truly enters the target collision region
   - or there is still a deeper close-stage object-enable / pose / timing issue
4. simply shrinking the bbox is no longer the highest-priority explanation

## Updated current assessment

At this point, the following is established:

- minimal isolation experiment:
  - the box probe can generate raw contact
  - the mesh + `solid_bbox` probe can also generate raw contact
- full pipeline:
  - `solid_bbox` produces no raw contact
  - `convex` also produces no raw contact

So the next most valuable directions are now:

1. the **true relative geometry** of the hand and object during the full close stage
2. whether we should output / visualize the finger/base pose relative to the object collision center and extents
3. whether we should replay the exact hand/object pose from the beginning of the full close stage inside the minimal probe

## Close-stage pose export

This round adds close-stage pose export:

- dual-arm:
  - `close_stage_snapshot_dual_before_close.json`
- single-arm:
  - `close_stage_snapshot_<arm>_before_close.json`

The exported content includes:

- current arm TCP world pose
- gripper joint qpos
- gripper-base pose
- finger-link poses
- object actor pose
- object collision mode / collision debug info

### Current exported file

In the experiment where the selected objects keep collision enabled from the beginning, the following file was generated:

- `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_all_collision_from_start/d_pour_blue_0/close_stage_snapshot_dual_before_close.json`

## Experiment: keep object collision enabled from the beginning

This round adds a new flag:

- `--grasp_action_object_collision_start_stage {close_gripper,grasp,pregrasp}`

The default remains:

- `close_gripper`

The new experiment uses:

- `pregrasp`

which means:

- selected execution objects keep collision enabled from the beginning
- instead of waiting until right before `close_gripper`

### Experiment command

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_all_collision_from_start \
  --ids 0 \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --candidate_selection_mode planner \
  --left_target_object cup \
  --right_target_object bottle \
  --candidate_target_local_x_offset_m -0.05 \
  --approach_offset_m 0.08 \
  --reach_error_pose_source tcp \
  --enable_grasp_action_object_collision 1 \
  --grasp_action_object_collision_start_stage pregrasp \
  --execution_object_collision_mode convex \
  --debug_collision_report 1 \
  --debug_visualize_object_collision_bbox 1 \
  --gripper_contact_monitor_mode all_robot_links \
  --save_debug_preview 1 \
  --save_debug_execution_preview 1 \
  --enable_viewer 0
```

### Result

The major change in this round is:

- **raw target contacts finally appear in the full pipeline**
- for example already at close init we now see:
  - `left_gripper_finger_link1<->planned_object_cup`
  - `right_gripper_finger_link2<->planned_object_bottle`
  - `right_gripper_link<->planned_object_bottle`
  - `right_realsense_link<->planned_object_bottle`
- raw target contacts remain present during close
- the final `[gripper-close]` now also reports:
  - `raw_target_contact=1`

### But an important issue remains

Even when raw target contacts are already present, the current `monitor_contact` / `base_contact` values still remain zero.

This implies:

1. one important reason why the earlier full pipeline had no raw contact at all is indeed that object collision was enabled too late
2. but the current contact monitor still has a matching problem:
   - raw target contacts already exist
   - yet the monitor/helper still reports zero

### Stronger current conclusion

At this point, we can say with more confidence:

- if selected objects participate in collision from `pregrasp`, raw target contacts do appear in the full pipeline
- therefore enabling collision only at `close_gripper` really does miss part of the already formed contact / overlap relationship
- at the same time, the current `monitor_contact` logic remains unreliable, because it can still output zero even when raw contacts already exist

So the problem is now split into two parts:

1. **collision-enable timing problem**
   - the old `close_gripper` default is too late
2. **contact-monitor matching problem**
   - even when raw contact exists, the monitor can still under-report it

## Experiment: shrink cup / bottle but keep collision starting only at close_gripper

As requested, I also ran a validation where the execution objects were shrunk while preserving the original behavior of enabling collision only at `close_gripper`.

### Setup

- `--grasp_action_object_collision_start_stage close_gripper`
- `--execution_object_collision_mode convex`
- `--execution_object_scale_override cup=0.9`
- `--execution_object_scale_override bottle=0.9`
- `--gripper_contact_monitor_mode all_robot_links`

### Command

```bash
bash /home/zaijia001/ssd/RoboTwin/code_painting/run_plan_anygrasp_keyframes_r1_batch.sh \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_batch_results \
  /home/zaijia001/ssd/RoboTwin/code_painting/replay_m_obj_pose_d_pour_blue_norobot \
  /home/zaijia001/ssd/data/R1/gt_depth_vis/d_pour_blue/hand_vis \
  /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_scaled_close_only_probe \
  --ids 0 \
  --reuse_preview_summary_root /home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_direct_preview_keyframes_batch \
  --reuse_preview_frame_mode annotated_json_keyframes \
  --reuse_preview_candidate_group orientation \
  --reuse_preview_top_rank 1 \
  --arm auto \
  --execute_both_arms 1 \
  --planner_backend urdfik \
  --urdfik_trajectory_mode cartesian_interp_ik \
  --candidate_selection_mode planner \
  --left_target_object cup \
  --right_target_object bottle \
  --candidate_target_local_x_offset_m -0.05 \
  --approach_offset_m 0.08 \
  --reach_error_pose_source tcp \
  --enable_grasp_action_object_collision 1 \
  --grasp_action_object_collision_start_stage close_gripper \
  --execution_object_collision_mode convex \
  --execution_object_scale_override cup=0.9 \
  --execution_object_scale_override bottle=0.9 \
  --debug_collision_report 1 \
  --debug_visualize_object_collision_bbox 1 \
  --gripper_contact_monitor_mode all_robot_links \
  --enable_viewer 0
```

### Result

This run still shows **no detected contact**.

The close-stage logs show:

- init: `raw_target_contacts=['none']`
- iter 1~20: `raw_target_contact_total=0`
- final:
  - `left ... raw_target_contact=0`
  - `right ... raw_target_contact=0`

### Updated conclusion

After shrinking `cup` and `bottle` to `0.9`:

- if we still preserve the original behavior where collision is enabled **only at `close_gripper`**
- then this sample still shows **no raw target contact**

So at this point it is reasonable to say:

- shrinking the execution objects alone is not enough to recover contact detection under the old `close_gripper`-only logic
- the key factor behind the earlier successful raw contacts still appears to be:
  - **moving the collision-enable timing earlier to `pregrasp`**
- at least at scale `0.9`, scaling does not rescue the `close_gripper`-only mode

## Is the "collision enabled from the beginning" output normal?

Based on the current logs, **yes, it is normal and informative**, with one important caveat: “normal” does not mean the contact-monitor logic is correct.

Why this full-collision output can be considered normal:

- once collision is enabled from `pregrasp`, stable raw target contacts appear in the full pipeline
- object poses remain stable; there is no obvious blow-up or unstable simulation behavior
- the trajectory still proceeds through close / action
- `raw_target_contact` is consistent with the headed observation that there really is physical contact

What is still not fixed is:

- `monitor_contact=0`
- `base_contact=0`

So the more precise statement is:

- **the full-collision experiment itself looks normal and trustworthy**
- **but the contact-monitor reporting chain still under-reports**

## Full-collision experiments with scale 0.8 / 0.5

Using:

- `--grasp_action_object_collision_start_stage pregrasp`
- `--execution_object_collision_mode convex`
- `--gripper_contact_monitor_mode all_robot_links`

I ran two more scales:

- `cup=0.8, bottle=0.8`
- `cup=0.5, bottle=0.5`

### Scale 0.8 result

Output dir:

- `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_all_collision_scale08/d_pour_blue_0`

The close logs show:

- left already has `left_gripper_finger_link1<->planned_object_cup` at init
- right already has `right_gripper_finger_link2<->planned_object_bottle` at init
- raw target contacts remain present during close
- final:
  - `left ... raw_target_contact=1`
  - `right ... raw_target_contact=1`

Conclusion:

- at scale `0.8`, the full-collision mode still detects raw target contact stably

### Scale 0.5 result

Output dir:

- `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_all_collision_scale05/d_pour_blue_0`

The close logs show:

- left init: `raw_target_contacts=['none']`
- right init already has `right_gripper_finger_link1<->planned_object_bottle`
- left starts showing raw contact only around close iter 8
- right has raw contact throughout close
- final:
  - `left ... raw_target_contact=1`
  - `right ... raw_target_contact=1`

Conclusion:

- even at scale `0.5`, the full-collision mode still eventually detects raw target contact
- only the left-side contact appears later instead of immediately at close init

### Current summary across scales

Under collision enabled from `pregrasp`:

- `0.9`: contact detected
- `0.8`: contact detected
- `0.5`: contact still detected

So for this sample:

- as long as collision is enabled early enough, shrinking the objects does not eliminate raw contact
- the dominant factor still appears to be collision-enable timing, not object size alone

## Analysis of the 0.5 full-collision result

You pointed out that in the `0.5 + pregrasp` result, the successful grasp/contact case appears to stop without the gripper fully closing, and that this is actually the effect you want.

### Interpretation from the logs

This observation is reasonable and matches the logs:

1. `close_stage_snapshot_dual_before_close.json` shows that right before close, both grippers still have joint qpos around:
   - `[0.045001, 0.044999]`
   - i.e. the gripper is still clearly open at close start
2. Under `0.5 + pregrasp`:
   - right already has raw contact at close init
   - left starts showing raw contact only later in close (around iter 8)
3. This means:
   - once the object is shrunk, at least on the left side it is no longer immediately trapped between both fingers at close init
   - the fingers must close further before they actually touch the object
4. Therefore the visual effect of “the gripper does not fully close because it gets blocked by the object” is consistent with this raw-contact timing

### One important detail

The current close logic does **not** stop as soon as raw contact appears. It stops only when:

- `monitor_contact` says there is contact
- and `qpos_delta <= stall_qpos_tol`
- for enough consecutive iterations, triggering `contact_stall`

But the monitor still under-reports, so the final reason in this run is often still:

- `target_reached`

That does **not** necessarily mean the fingers physically achieved a perfect fully closed zero-gap state. It is closer to:

- the command target was driven to `cmd=0.0`
- while visually the object can still keep the fingers from appearing fully closed

So your observation is reasonable:

- **the 0.5 full-collision result really does look closer to the desired effect where the fingers close onto the object and get blocked before appearing fully closed**

## Experiment: start collision only at close + finger-only monitoring, with scale 0.5

As requested, I ran a new test with:

- `--grasp_action_object_collision_start_stage close_gripper`
- `--gripper_contact_monitor_mode fingers`
- `--execution_object_scale_override cup=0.5`
- `--execution_object_scale_override bottle=0.5`

Output dir:

- `/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_single_scale05_close_only_fingers/d_pour_blue_0`

### Result

The close logs show:

- left:
  - init: `raw_target_contacts=['none']`
  - iter 1~20: `raw_target_contact_total=0`
  - final: `raw_target_contact=0`
- right:
  - init: `raw_target_contacts=['none']`
  - starting around iter 8, `right_gripper_finger_link2<->planned_object_bottle` appears
  - final: `raw_target_contact=1`

At the same time, the monitor/helper still reports:

- `monitor_contact=0`

### Conclusion

Under the stricter condition:

- **collision starts only at close**
- **monitor only fingers**
- **objects scaled to 0.5**

The original problem is **not fully gone**:

- the left side still detects no raw target contact at all
- the right side does detect raw contact, but the monitor still under-reports it

So more specifically:

1. shrinking to `0.5` really does make the visual effect closer to “the gripper closes onto the object and is blocked before fully closing”
2. but if collision enabling is still delayed until `close_gripper`, the issue still remains:
   - the left side can still completely miss contact
3. if the goal is to reliably obtain this “blocked before fully closed” effect, the more reliable condition remains:
   - **enable collision from `pregrasp`**

## Why does the gripper often keep closing if collision is enabled only after it is already inside the object?

The key phenomenon you pointed out is:

- during `pregrasp / grasp`, the gripper may already be inside the object's collision volume
- collision is enabled only at `close_gripper`
- once collision is enabled, the system does **not** immediately get stuck at that pose
- instead it often keeps driving toward `cmd=0.0`, i.e. it looks like it just closes through

From the current code and experiments, this behavior is caused by several mechanisms together.

### 1. Enabling collision late does not rewind the system to the first-touch boundary

When collision groups are enabled only right before `close_gripper`, the object and fingers are already in an **overlapping initial state**.

What the physics does **not** automatically do in that case:

- it does not roll the fingers back to the first-touch boundary
- it does not project the already-overlapping geometry back to an ideal non-penetrating pose
- it does not reconstruct the missing history of “approach from outside -> first contact -> blockage”

So late-enabled collision does not give you a clean contact threshold. It gives you the current already-overlapping state.

### 2. The current close-stop logic does not stop on raw contact; it stops on monitor_contact + stall

The stop condition inside `close_grippers_progressively_with_collision_stop(...)` is basically:

- `has_contact = _contact_involves_entities(...)`
- and `qpos_delta <= stall_qpos_tol`
- for enough consecutive iterations (`contact_confirm_iters`)
- then stop with `reason=contact_stall`

But the experiments repeatedly show:

- raw target contact may already exist
- while `monitor_contact` still stays zero

So what happens is:

- the controller keeps driving the gripper toward `target=0.0`
- the final reason often remains `target_reached`
- visually this looks like “it just fully closed anyway”

So the issue is not simply “no physical contact exists.” It is:

- **even if raw contact exists, if the monitor does not recognize it, the logic will not stop on contact_stall**

### 3. The two fingers are not independently controlled; one arm uses one gripper scalar command

From `envs/robot/robot.py`, `set_gripper(gripper_val, arm_tag, ...)`:

- takes `self.left_gripper / self.right_gripper`
- and assigns drive targets to that whole joint group together

So semantically this is closer to:

- **one scalar open/close command per arm**
- not “one command per finger”

Therefore, even though intuitively you might expect:

- one finger gets blocked first
- the other finger might still move more

under the current controller there is no higher-level logic that independently stops one finger while letting the other continue freely. The system keeps issuing one common closing target to the whole gripper joint group.

### 4. The execution objects are kinematic actors, so late-enabled overlap is not resolved like a dynamic push-away event

The execution objects are built with `builder.build_kinematic(...)`.

That means:

- the object does not get pushed away like a dynamic rigid body would
- if collision is enabled late while already overlapping, the solver does not naturally move the object out to a “clean contact boundary” for you

So what you are more likely to observe is:

- the object stays where it is
- the controller keeps trying to close
- instead of instantly forming a neat blocked state at that late-enabled pose

### 5. Why not “one side stuck, the other side still moving”?

This is constrained by two things in the current system:

1. **control is coupled**
   - one scalar gripper command per arm
2. **stopping is not decided per individual finger**
   - it is decided at arm/gripper level using the current monitor/contact logic

So the natural outcome is not:

- finger1 stops
- finger2 keeps advancing independently for a long time

What is more likely is:

- both keep moving toward the same commanded target
- if the monitor does not recognize contact, both keep being driven closed
- if raw contact appears later only on one side, that shows up in the raw logs, but does not automatically become a per-finger stop behavior

## Most accurate current conclusion

So the root cause of “already inside the collision volume, yet still keeps closing when collision is enabled late” is not a single physics bug. It is the combination of:

1. **collision enabled too late**, so the first-contact history is already lost
2. **no automatic rollback to the first-touch boundary**
3. **stop logic depends on monitor_contact, which still under-reports**
4. **gripper control is arm-level coupled, not per-finger independent**
5. **the object is kinematic, so it does not naturally get pushed into a clean separating configuration**

That is why the behavior is often:

- not “gracefully blocked exactly at the late-enabled pose”
- but instead “it keeps driving toward `cmd=0.0`”

And this also explains why, if the goal is a natural “close until the object blocks the fingers” effect, the more reliable setup is still:

- **enable collision from `pregrasp`, or at least from `grasp`**
