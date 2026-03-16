# Curobo Rebuild Record V1

## Scope And Safety Boundary

This rebuild was intentionally limited to:

- the conda environment `RoboTwin-lingbot`
- the worktree `/home/zaijia001/vam/RoboTwin-lingbot`

It did **not** modify:

- `/home/zaijia001/ssd/RoboTwin` source files
- any other RoboTwin conda environment

## Why A Rebuild Was Needed

Before this rebuild, the active editable install inside `RoboTwin-lingbot` pointed to:

- `/home/zaijia001/ssd/RoboTwin/curobo/src`

That broke the intended isolation and also kept dragging the client back to the shared SSD checkout.

At the same time, the previous failure mode mixed two separate issues:

1. the old editable install referenced the shared SSD RoboTwin Curobo source
2. local Curobo JIT builds were using `/home/zaijia001/ssd/cuda-12.1/bin/nvcc`, which cannot compile for the Blackwell `compute_120` target

## Local Runtime Facts

Inside `RoboTwin-lingbot`, the runtime stack is:

- PyTorch: `2.9.0+cu128`
- torch CUDA runtime tag: `12.8`

But the installed CUDA compilers on disk are:

- `/home/zaijia001/ssd/cuda-12.1`
- `/home/zaijia001/ssd/cuda-12.4`

There is no local CUDA 12.8 toolkit directory in `/home/zaijia001/ssd/`.

So the rebuild strategy was:

- stop using the shared SSD editable source
- reinstall from the local worktree copy at `/home/zaijia001/vam/RoboTwin-lingbot/envs/curobo`
- compile with CUDA 12.4 instead of 12.1
- avoid `compute_120` build failure by setting `TORCH_CUDA_ARCH_LIST=9.0+PTX`

## Exact Rebuild Approach

The rebuild used:

- local source: `/home/zaijia001/vam/RoboTwin-lingbot/envs/curobo`
- compiler: `/home/zaijia001/ssd/cuda-12.4/bin/nvcc`
- editable install into the `RoboTwin-lingbot` conda env
- local build/cache directories:
  - `/home/zaijia001/vam/RoboTwin-lingbot/.torch_extensions`
  - `/home/zaijia001/vam/RoboTwin-lingbot/.tmp_build`

The raw build log is saved at:

- `/home/zaijia001/vam/RoboTwin-lingbot/agent-read/debug-logs/curobo-rebuild-2026-03-16.log`

## What Changed

Before:

- `site-packages/__editable__.nvidia_curobo-0.0.0.pth`
  pointed to `/home/zaijia001/ssd/RoboTwin/curobo/src`

After:

- `site-packages/__editable__.nvidia_curobo-0.7.7.post1.dev5.pth`
  points to `/home/zaijia001/vam/RoboTwin-lingbot/envs/curobo/src`

## Verification

Verified in the `RoboTwin-lingbot` environment with the repository root set to this worktree:

- `CUROBO_FILE /home/zaijia001/vam/RoboTwin-lingbot/envs/curobo/src/curobo/__init__.py`
- `CuroboPlanner_is_none False`

This confirms:

- Curobo is no longer imported from `/home/zaijia001/ssd/RoboTwin/curobo/src`
- `planner.py` no longer falls back at import time just because the Curobo package is missing or ABI-broken

## Important Caveat

This rebuild proves that the environment is now isolated and that Curobo imports successfully in the normal `RoboTwin-lingbot` environment context.

It does **not** yet prove that every RoboTwin task will now fully prefer Curobo in all motion-planning paths or that runtime performance has returned to the original fastest state. Those questions require additional task-level evaluation.

## Practical Conclusion

As of March 16, 2026:

- the `RoboTwin-lingbot` conda environment no longer depends on the shared SSD Curobo editable install
- the Curobo package now resolves to the local worktree copy
- the rebuild was contained to the approved environment and repository scope
- further evaluation can now focus on planner behavior and task speed, rather than installation contamination
