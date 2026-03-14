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

## Current Assumptions

- RoboTwin assets can be reused from `/home/zaijia001/ssd/RoboTwin/assets` via local links in this worktree.
- LingBot-VA should point its RoboTwin client code at `/home/zaijia001/vam/RoboTwin-lingbot`.
