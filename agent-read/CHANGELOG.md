# Changelog

## 2026-03-14

- Created the `RoboTwin-lingbot` worktree from the upstream `baseline` branch.
- Created the `RoboTwin-lingbot` conda environment by cloning the existing `RoboTwin` environment.
- Updated `script/requirements.txt` to use `huggingface_hub==0.36.2` for LingBot-VA compatibility.
- Reused the original RoboTwin asset directories via links inside this worktree.
- Documented the new isolated RoboTwin workspace in `agent-read/`.
- Added a RoboTwin-side fallback from optional `curobo` planners to `MPLib` so Blackwell GPUs without a matching Curobo build can still run LingBot evaluation.
- Normalized fallback planner types to `mplib_RRT`, aligned the fallback planner interfaces with the existing robot wrapper, and downgraded MPLib `TOPP` parameterization runtime errors into normal planning failures.
- Ran a minimal LingBot evaluation smoke test on `click_bell` with `test_num=1`; the run completed successfully and its generated outputs are now ignored via `.gitignore`.
- Added a CPU farthest-point fallback in `envs/camera/camera.py` so missing `pytorch3d` no longer terminates RGB-based LingBot runs.
- Ignored the local `eval-test-decoder/` output directory in `.gitignore`.
- Confirmed this worktree now supports a full March 16, 2026 LingBot action-only DSRL online `click_bell` episode from the separate `lingbot-va` repo; the run completed end-to-end and logged SAC metrics, although task success remained `0/1`.
- Ignored the local `results_regression_eval/` directory after the March 16, 2026 original LingBot eval regression smoke run (`click_bell 1/1`).
- Added `task_config/demo_clean_large_d435.yml` so LingBot action-only training can target the `Large_D435` camera layout (`640x480`) without overwriting the original `demo_clean.yml`.
- Updated `envs/_base_task.py` so missing grasp poses are treated as normal planning failures instead of raising `target_pose cannot be None for move action`.
- Updated `envs/place_can_basket.py` so `setup_demo(...)` writes prompt substitutions into `self.info["info"]` before `play_once()`.
- Completed a March 16, 2026 `place_can_basket` post-train smoke eval against LingBot `checkpoint_step_5000`; the first result was `0/1`, and generated outputs are now intended to stay untracked.
- Removed the old `nvidia_curobo` editable install that pointed into `/home/zaijia001/ssd/RoboTwin/curobo/src` from the `RoboTwin-lingbot` conda environment.
- Reinstalled `nvidia_curobo` from the local worktree copy at `/home/zaijia001/vam/RoboTwin-lingbot/envs/curobo` using CUDA 12.4 plus `TORCH_CUDA_ARCH_LIST=9.0+PTX`.
- Saved the raw rebuild log to `agent-read/debug-logs/curobo-rebuild-2026-03-16.log` and documented the rebuild in `agent-read/curobo-rebuild-v1.md`.
- Verified in the normal `RoboTwin-lingbot` environment context that:
  - `CUROBO_FILE` now resolves to the local worktree copy
  - `CuroboPlanner_is_none` is `False`
- Added a local `AGENTS.md` so this worktree now carries explicit in-repo rules for scope protection, debug logging, and command documentation synchronization.
