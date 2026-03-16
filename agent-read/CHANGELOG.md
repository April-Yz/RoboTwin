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
