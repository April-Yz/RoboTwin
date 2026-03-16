# Project Agent Rules

These rules apply to all work inside `/home/zaijia001/vam/RoboTwin-lingbot`.

## Scope Protection

1. Allowed edit scope for this project line of work:
   - `/home/zaijia001/vam/RoboTwin-lingbot`
   - `/home/zaijia001/vam/lingbot-va`
   - conda envs `RoboTwin-lingbot` and `lingbot-va`
2. Do not modify `/home/zaijia001/ssd/RoboTwin` or any other RoboTwin checkout unless the user explicitly approves it again.
3. Prefer fixes inside this worktree or its conda env over shared or system-wide changes.

## Debug Record Rules

1. Every meaningful environment, planner, renderer, or eval issue should be written into `agent-read/`.
2. Each debug record should capture failure symptom, cause, workaround, and final status.
3. Update `agent-read/README.md` and `agent-read/CHANGELOG.md` when debug outcomes change how the project should be run.

## Command Documentation Rules

1. When launch commands or eval flags change, update the LingBot-side bilingual command index in the same turn.
2. If a command relies on a specific GPU or renderer workaround, document it explicitly.
