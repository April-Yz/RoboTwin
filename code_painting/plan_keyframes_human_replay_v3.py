#!/usr/bin/env python3
"""Human-replay target builder routed to the isolated Piper IK V3 planner."""

from __future__ import annotations

import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(PROJECT_ROOT)
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import plan_keyframes_human_replay as base


def main() -> None:
    base.PLANNER_SCRIPT = THIS_DIR / "plan_anygrasp_keyframes_piper_v3.py"
    print(
        "[human-replay-v3] isolated=1 "
        f"planner={base.PLANNER_SCRIPT} "
        f"target_semantics={os.environ.get('PIPER_IK_V3_TARGET_SEMANTICS', 'ours_tcp')}"
    )
    base.main()


if __name__ == "__main__":
    main()
