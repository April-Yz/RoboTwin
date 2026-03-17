# RoboTwin-lingbot Environment Setup For Agents

## 1. Goal

Build a RoboTwin environment that is compatible with LingBot-VA evaluation in this workspace.

## 2. Scope And Paths

- Repo root: `RoboTwin-lingbot`
- Expected conda env name: `RoboTwin-lingbot`
- Install entry scripts:
  - `script/_install.sh`
  - `script/_download_assets.sh`

## 3. System Prerequisites

Install Vulkan runtime before Python deps (required by RoboTwin rendering stack):

```bash
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
```

## 4. Conda Environment

Create and activate a clean environment (Python 3.10 is expected by current scripts):

```bash
conda create -n RoboTwin-lingbot python=3.10 -y
conda activate RoboTwin-lingbot
```

## 5. Python Dependency Install

From repo root:

```bash
bash script/_install.sh
```

What this script currently does:

- Installs `script/requirements.txt`
- Installs `pytorch3d` from GitHub with `--no-build-isolation`
- Applies local compatibility patches to installed `sapien` and `mplib`
- Clones and installs `envs/curobo` editable package

## 6. Asset Download

After install, download required assets:

```bash
bash script/_download_assets.sh
```

If this workspace already links assets from another local RoboTwin checkout, keep those links and skip duplicate downloads.

## 7. Minimal Validation

Run quick checks from repo root:

```bash
python -c "import sapien, mplib, gymnasium; print('core deps ok')"
python script/test_render.py
```

## 8. Known Local Compatibility Notes

- On this machine, Curobo may fail for Blackwell if toolchain/CUDA do not match; local fallback to MPLib is already integrated in this worktree.
- If `pytorch3d` is unavailable, camera code contains a CPU fallback in this workspace.
- For LingBot integration, point client-side `ROBOTWIN_ROOT` to this repo path.

## 9. Interop With lingbot-va

Recommended pairing in this workspace:

- RoboTwin side: this repo + `RoboTwin-lingbot` env
- Model side: `lingbot-va` repo + `lingbot-va` env

Keep both repos synchronized when launch commands or eval flags change.
