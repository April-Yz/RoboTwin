#!/usr/bin/env python3
"""Write the explicit shared-input contract for one IK-logic grid cell."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--ik-logic", choices=("legacy", "canonical"), required=True)
    parser.add_argument(
        "--strategy",
        choices=("orientation", "fused", "top_score", "human_replay"),
        required=True,
    )
    parser.add_argument("--source-semantics", required=True)
    parser.add_argument("--orientation-remap", required=True)
    parser.add_argument("--planner-entry", type=Path, required=True)
    args = parser.parse_args()

    if args.ik_logic == "canonical":
        renderer = "PiperCanonicalTCPRenderer"
        target_to_link6 = "inverse(Ry(-1.57) @ Tx(0.19)); Piper server RTCP semantics"
    else:
        renderer = "HandRetargetPiperDualURDFIKRenderer"
        target_to_link6 = "robot._trans_from_gripper_to_endlink; Legacy/OursV2 semantics"
    payload = {
        "schema": "piper_canonical_tcp_v1.ik_logic_grid_input.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "episode_id": str(args.episode_id),
        "ik_logic": args.ik_logic,
        "strategy": args.strategy,
        "shared_input_contract": {
            "source": args.source_semantics,
            "direct_selection_pose": True,
            "planner_input_pose": "T_W_RTCP",
            "position_frame": "WORLD",
            "orientation_frame": "local_RTCP",
            "orientation_remap": args.orientation_remap,
            "final_target_retreat_m": 0.0,
            "candidate_target_local_x_offset_m": 0.0,
            "candidate_target_local_z_offset_m": 0.0,
            "pregrasp": {
                "derived_after_selection": True,
                "axis": "local_RTCP_+X",
                "offset_m": 0.12,
            },
            "explicitly_not_used_as_input": (
                "selection_strategy_compare_v4 lower Planner Target, because it already "
                "contains historical offsets/retreat/pregrasp/TCP-to-link6 semantics"
            ),
        },
        "row_specific_conversion": {
            "planner_entry": str(args.planner_entry),
            "renderer": renderer,
            "T_W_RTCP_to_urdf_link6": target_to_link6,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
