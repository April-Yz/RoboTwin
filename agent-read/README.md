# RoboTwin-lingbot Agent Notes

## Project Overview

This worktree is a LingBot-VA-specific RoboTwin workspace created from the upstream `baseline` branch.

- Source repository: `/home/zaijia001/ssd/RoboTwin`
- Worktree path: `/home/zaijia001/vam/RoboTwin-lingbot`
- Branch: `RoboTwin-lingbot`
- Conda environment: `RoboTwin-lingbot`

## Intended Use

This copy isolates RoboTwin-side evaluation changes needed by LingBot-VA without modifying the original shared RoboTwin checkout or its existing conda environments.

## LingBot-VA Compatibility Notes

- `script/requirements.txt` is aligned with LingBot-VA's RoboTwin setup notes for `huggingface_hub==0.36.2`.
- `script/_install.sh` already uses `pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation`, which matches the LingBot-VA README requirement.
- The environment was cloned from the existing `RoboTwin` conda environment into `RoboTwin-lingbot` and should be treated as the only editable RoboTwin environment for LingBot-VA work in this session.
- On this machine, upstream `curobo` cannot be used as-is for evaluation because the local NVCC toolchain is CUDA 12.1 while the GPU is Blackwell (`sm_120`). `envs/robot/robot.py` and `envs/robot/planner.py` now fall back to `MplibPlanner` when `CuroboPlanner` import or build fails.
- On March 16, 2026, the `RoboTwin-lingbot` environment was rebuilt so its editable `nvidia_curobo` install now points to the local worktree copy under `/home/zaijia001/vam/RoboTwin-lingbot/envs/curobo/src` instead of `/home/zaijia001/ssd/RoboTwin/curobo/src`.
- That rebuild used the `RoboTwin-lingbot` conda environment plus this worktree only; it did not modify `/home/zaijia001/ssd/RoboTwin` source files or any other RoboTwin conda environment.
- The detailed rebuild/debug record is in `agent-read/curobo-rebuild-v1.md`.
- The fallback normalizes embodiment planner declarations like `"curobo"` to `"mplib_RRT"` and converts MPLib `TOPP` parameterization exceptions into ordinary planning failures so eval can continue instead of aborting the whole process.
- `envs/camera/camera.py` now has a CPU farthest-point fallback when `pytorch3d` is unavailable, so RGB-based LingBot online runs no longer hard-exit on this host.

## Current Assumptions

- RoboTwin assets can be reused from `/home/zaijia001/ssd/RoboTwin/assets` via local links in this worktree.
- LingBot-VA should point its RoboTwin client code at `/home/zaijia001/vam/RoboTwin-lingbot`.
- The worktree now also provides `task_config/demo_clean_large_d435.yml`, which keeps the original `demo_clean` behavior but switches both head and wrist cameras to `Large_D435` (`640x480`).
- `click_bell` has been smoke-tested end-to-end against the LingBot-VA websocket server with `test_num=1`, producing a successful run and result artifacts under ignored output directories.
- On March 16, 2026, the LingBot action-only DSRL entry in the separate `lingbot-va` repo completed one full `click_bell` RoboTwin online episode against this worktree, emitted SAC metrics, and exited cleanly with zero task successes.
- On March 16, 2026, this worktree also completed the first `place_can_basket` post-train LingBot smoke eval result against `checkpoint_step_5000`. The result was `0/1`, but the server-client-task pipeline completed and wrote metrics plus rollout artifacts instead of aborting during setup or planning.
- `envs/_base_task.py` now converts missing grasp poses into ordinary planning failures instead of constructing `move` actions with `target_pose=None`.
- `envs/place_can_basket.py` now fills `self.info["info"]` during `setup_demo(...)`, which keeps prompt generation available even when expert pre-check is disabled for smoke debugging.
- A March 16, 2026 environment-level validation confirmed:
  - `CUROBO_FILE` resolves to `/home/zaijia001/vam/RoboTwin-lingbot/envs/curobo/src/curobo/__init__.py`
  - `CuroboPlanner_is_none` is now `False` in the normal `RoboTwin-lingbot` environment context
- A local `/home/zaijia001/vam/RoboTwin-lingbot/AGENTS.md` file now records the standing scope-protection, debug-log, and command-sync rules for this isolated worktree.
