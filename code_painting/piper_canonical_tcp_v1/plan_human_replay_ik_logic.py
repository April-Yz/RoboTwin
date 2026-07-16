#!/usr/bin/env python3
"""Run one shared Human Replay RTCP target through Legacy or Canonical IK."""

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


def pop_ik_logic(argv: list[str]) -> tuple[str, list[str]]:
    output: list[str] = []
    value: str | None = None
    index = 0
    while index < len(argv):
        if argv[index] == "--ik-logic":
            if index + 1 >= len(argv):
                raise ValueError("Missing value for --ik-logic")
            value = argv[index + 1]
            index += 2
            continue
        output.append(argv[index])
        index += 1
    if value not in {"legacy", "canonical"}:
        raise ValueError("--ik-logic must be legacy or canonical")
    return value, output


def force_scalar_flags(argv: list[str]) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in FORCED_FLAGS:
            if index + 1 >= len(argv):
                raise ValueError(f"Missing value for {token}")
            index += 2
            continue
        output.append(token)
        index += 1
    for flag, value in FORCED_FLAGS.items():
        output.extend((flag, value))
    return output


def main() -> None:
    ik_logic, forwarded = pop_ik_logic(sys.argv[1:])
    if ik_logic == "canonical":
        planner = THIS_DIR / "planner.py"
        conversion = "Piper server RTCP -> URDF link6: inverse(Ry(-1.57) @ Tx(0.19))"
    else:
        planner = CODE_PAINTING / "plan_anygrasp_keyframes_piper.py"
        conversion = "legacy robot._trans_from_gripper_to_endlink"
    human_replay.PLANNER_SCRIPT = planner
    sys.argv = [sys.argv[0], *force_scalar_flags(forwarded)]
    print(
        f"[ik-logic-human-replay] ik_logic={ik_logic} source=T_W_CGRASP_HUMAN "
        "axis_remap=swap_red_blue_keep_green planner_input=T_W_RTCP "
        f"final_retreat_m=0 pregrasp=0.12m@local_RTCP_+X conversion={conversion}"
    )
    human_replay.main()


if __name__ == "__main__":
    main()
