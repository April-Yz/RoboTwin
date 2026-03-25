## 2026-03-25 12:08:00 +08

- Added flag: `--enable_grasp_action_object_collision 0|1`
  - Entry points:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
    - `code_painting/plan_anygrasp_keyframes_r1_batch.py`
    - `code_painting/run_plan_anygrasp_keyframes_r1_batch.sh`
  - Purpose:
    - enable collision blocking for the selected execution object during `close_gripper` and `action`
    - default `0` preserves the previous no-collision execution mode
  - Usage:
    - single-run command: append `--enable_grasp_action_object_collision 1`
    - batch: append `--enable_grasp_action_object_collision 1` to the batch command as well
  - Notes:
    - this flag does not change target-pose construction for `pregrasp/grasp/action`
    - it also does not change the relative transform used when attaching the object to TCP
## 2026-03-25 13:05:00 +08

- Added visualization-mode flags to `plan_anygrasp_keyframes_r1.py`:
  - New flags:
    - `--debug_visualize_targets 0|1`
    - `--viewer_show_camera_frustums 0|1`
  - Usage:
    - `debug_visualize_targets=0` disables target-axis actors globally
    - `viewer_show_camera_frustums=0` disables SAPIEN camera frustum lines in the interactive viewer
  - Related code:
    - `code_painting/plan_anygrasp_keyframes_r1.py`
