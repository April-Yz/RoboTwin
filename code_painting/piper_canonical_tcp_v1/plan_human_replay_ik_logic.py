#!/usr/bin/env python3
"""Adapt one Human Replay center to Legacy-original or Canonical-RTCP IK."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
CODE_PAINTING = THIS_DIR.parent
for path in (THIS_DIR, CODE_PAINTING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_keyframes_human_replay as human_replay  # noqa: E402
from frame_contract import (  # noqa: E402
    R_CGRASP_RTCP,
    matrix_to_pose_wxyz,
    pose_wxyz_to_matrix,
)


COMMON_FORCED_FLAGS = {
    "--piper_urdfik_apply_global_trans_to_ik": "0",
}

LOGIC_FORCED_FLAGS = {
    "legacy": {
        "--candidate_orientation_remap_label": "identity",
        "--target_retreat_m": "0.14",
        "--approach_axis": "local_z",
        "--reach_error_pose_source": "ee",
        "--debug_gripper_actor_forward_axis": "local_z",
    },
    "canonical": {
        # The generated pose is materialized as RTCP below.  A reuse-plan-summary
        # path does not apply candidate_orientation_remap_label a second time.
        "--candidate_orientation_remap_label": "identity",
        # Build both rows from the original 0.14m Human Replay recipe because
        # that value also participates in its handover-keyframe adjustment.
        # The adapter below then removes the retreat before materializing RTCP.
        "--target_retreat_m": "0.14",
        "--approach_axis": "local_x",
        "--reach_error_pose_source": "tcp",
        "--debug_gripper_actor_forward_axis": "local_x",
    },
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


def force_scalar_flags(argv: list[str], ik_logic: str) -> list[str]:
    forced = {**COMMON_FORCED_FLAGS, **LOGIC_FORCED_FLAGS[ik_logic]}
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


def _shift_local_z(pose_wxyz: np.ndarray, distance_m: float) -> np.ndarray:
    transform = pose_wxyz_to_matrix(pose_wxyz)
    transform[:3, 3] += transform[:3, 2] * float(distance_m)
    return matrix_to_pose_wxyz(transform)


def _canonical_rtcp_pose(source_cgrasp_wxyz: np.ndarray) -> np.ndarray:
    transform = pose_wxyz_to_matrix(source_cgrasp_wxyz)
    transform[:3, :3] = transform[:3, :3] @ R_CGRASP_RTCP
    return matrix_to_pose_wxyz(transform)


def _candidate_dicts(summary: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[Any] = [summary.get("selected_candidates", [])]
    containers.extend((summary.get("selected_candidates_by_executed_arm") or {}).values())
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for container in containers:
        for entry in container or []:
            if not isinstance(entry, dict) or id(entry) in seen:
                continue
            seen.add(id(entry))
            output.append(entry)
    return output


def install_summary_adapter(ik_logic: str) -> None:
    original = human_replay.build_plan_summary

    def adapted_build_plan_summary(*args, **kwargs):
        summary = original(*args, **kwargs)
        retreat_m = float(summary.get("human_replay_target_retreat_m", 0.0))
        for entry in _candidate_dicts(summary):
            planner_pose = np.asarray(entry["pose_world_wxyz"], dtype=np.float64).reshape(7)
            source_pose = _shift_local_z(planner_pose, retreat_m)
            entry["semantic_source_selection_pose_world_wxyz"] = source_pose.tolist()
            entry["semantic_source_frame"] = "T_W_CGRASP_HUMAN_CENTER"
            if ik_logic == "canonical":
                rtcp_pose = _canonical_rtcp_pose(source_pose)
                entry["legacy_or_raw_pose_before_rtcp_adapter_wxyz"] = planner_pose.tolist()
                entry["raw_pose_world_wxyz"] = rtcp_pose.tolist()
                entry["pose_world_wxyz"] = rtcp_pose.tolist()
                entry["canonical_rtcp_pose_world_wxyz"] = rtcp_pose.tolist()
                entry["orientation_adapter"] = "R_W_RTCP = R_W_CGRASP_HUMAN @ R_CGRASP_RTCP"
        summary["ik_logic_semantic_adapter"] = {
            "logic": ik_logic,
            "source": "T_W_CGRASP_HUMAN_CENTER",
            "target": "legacy planner target after 0.14m local +Z retreat"
            if ik_logic == "legacy"
            else "T_W_RTCP with source origin preserved and axes materialized",
            "orientation_remap_materialized": ik_logic == "canonical",
            "source_builder_retreat_m": retreat_m,
            "final_target_retreat_m": retreat_m if ik_logic == "legacy" else 0.0,
        }
        return summary

    human_replay.build_plan_summary = adapted_build_plan_summary


def main() -> None:
    ik_logic, forwarded = pop_ik_logic(sys.argv[1:])
    install_summary_adapter(ik_logic)
    if ik_logic == "canonical":
        planner = THIS_DIR / "planner.py"
        conversion = "materialized CGRASP_HUMAN->RTCP; inverse(Ry(-1.57)@Tx(0.19))"
    else:
        planner = CODE_PAINTING / "plan_anygrasp_keyframes_piper.py"
        conversion = "original 0.14m local-human-Z retreat; legacy gripper_to_endlink"
    human_replay.PLANNER_SCRIPT = planner
    sys.argv = [sys.argv[0], *force_scalar_flags(forwarded, ik_logic)]
    print(
        f"[ik-semantic-human-replay] ik_logic={ik_logic} "
        f"source=T_W_CGRASP_HUMAN_CENTER conversion={conversion}"
    )
    human_replay.main()


if __name__ == "__main__":
    main()
