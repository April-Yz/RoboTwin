#!/usr/bin/env python3
"""Run Human Replay targets through the isolated PiperCanonicalTCP-v1 planner."""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
CODE_PAINTING = THIS_DIR.parent
for path in (THIS_DIR, CODE_PAINTING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_keyframes_human_replay as human_replay  # noqa: E402


FORCED_FLAGS = {
    "--candidate_orientation_remap_label": "swap_red_blue_keep_green",
    "--target_retreat_m": "0",
    "--approach_axis": "local_x",
    "--piper_urdfik_apply_global_trans_to_ik": "0",
    "--reach_error_pose_source": "tcp",
    "--debug_gripper_actor_forward_axis": "local_x",
}


def force_scalar_flags(argv: list[str], forced: dict[str, str]) -> list[str]:
    """Remove user copies of scalar flags, then append the Canonical contract."""
    output: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in forced:
            if index + 1 >= len(argv):
                raise ValueError(f"Missing value for {token}")
            index += 2
            continue
        output.append(token)
        index += 1
    for flag, value in forced.items():
        output.extend((flag, value))
    return output


def main() -> None:
    human_replay.PLANNER_SCRIPT = THIS_DIR / "planner.py"
    sys.argv = [sys.argv[0], *force_scalar_flags(sys.argv[1:], FORCED_FLAGS)]
    print(
        "[piper-canonical-human-replay] source=T_W_CGRASP_HUMAN "
        "axis_remap=swap_red_blue_keep_green planner_target=T_W_RTCP "
        "final_retreat_m=0 pregrasp_axis=local_RTCP_+X"
    )
    human_replay.main()


if __name__ == "__main__":
    main()
