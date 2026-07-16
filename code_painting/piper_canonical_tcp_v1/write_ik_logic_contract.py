#!/usr/bin/env python3
"""Write the semantic-source and row-specific target contract for one grid cell."""

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
    parser.add_argument("--target-local-z-offset-m", type=float, default=0.0)
    parser.add_argument("--target-retreat-m", type=float, default=0.0)
    parser.add_argument("--approach-axis", choices=("local_x", "local_z"), required=True)
    args = parser.parse_args()

    if args.ik_logic == "canonical":
        renderer = "PiperCanonicalTCPRenderer"
        target_to_link6 = "inverse(Ry(-1.57) @ Tx(0.19)); Piper server RTCP semantics"
    else:
        renderer = "HandRetargetPiperDualURDFIKRenderer"
        target_to_link6 = "robot._trans_from_gripper_to_endlink; Legacy/OursV2 semantics"
    payload = {
        "schema": "piper_canonical_tcp_v1.ik_semantic_grid_input.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "episode_id": str(args.episode_id),
        "ik_logic": args.ik_logic,
        "strategy": args.strategy,
        "semantic_source_contract": {
            "source": args.source_semantics,
            "same_candidate_or_hand_center_across_rows": True,
            "position_frame": "WORLD",
            "note": (
                "The two rows share a semantic grasp/hand center, not an identical "
                "numeric planner target. Each row applies its native pose semantics."
            ),
            "explicitly_not_used_as_input": (
                "selection_strategy_compare_v4 lower Planner Target as a common pose; "
                "it already embeds Legacy offsets/retreat semantics"
            ),
        },
        "row_specific_target_contract": {
            "orientation_remap": args.orientation_remap,
            "target_retreat_m": args.target_retreat_m,
            "candidate_target_local_x_offset_m": 0.0,
            "candidate_target_local_z_offset_m": args.target_local_z_offset_m,
            "pregrasp": {
                "derived_after_selection": True,
                "axis": args.approach_axis,
                "offset_m": 0.12,
            },
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
