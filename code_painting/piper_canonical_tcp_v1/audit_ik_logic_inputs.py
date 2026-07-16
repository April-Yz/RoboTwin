#!/usr/bin/env python3
"""Audit a shared semantic source with native Legacy and Canonical adapters."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from frame_contract import (  # noqa: E402
    R_CGRASP_RTCP,
    SERVER_TOOL_LENGTH_M,
    pose_wxyz_to_matrix,
)


STRATEGIES = ("orientation", "fused", "top_score", "human_replay")
POSITION_TOL_M = 1e-9
ROTATION_TOL = 1e-8


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_entries(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for entry in summary.get("selected_candidates", []) if isinstance(entry, dict)]


def source_pose(entry: dict[str, Any], strategy: str) -> np.ndarray:
    key = "semantic_source_selection_pose_world_wxyz" if strategy == "human_replay" else "raw_pose_world_wxyz"
    return np.asarray(entry[key], dtype=np.float64).reshape(7)


def identity(entry: dict[str, Any], strategy: str) -> tuple[Any, ...]:
    common = (str(entry.get("arm")), int(entry.get("source_frame", -1)))
    if strategy == "human_replay":
        return common
    return (*common, int(entry.get("candidate_idx", -1)))


def rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def compare_strategy(
    strategy: str,
    legacy_summary: dict[str, Any],
    canonical_summary: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    legacy_entries = selected_entries(legacy_summary)
    canonical_entries = selected_entries(canonical_summary)
    if len(legacy_entries) != len(canonical_entries):
        return False, {
            "reason": "candidate_count",
            "legacy": len(legacy_entries),
            "canonical": len(canonical_entries),
        }

    rows: list[dict[str, Any]] = []
    for index, (legacy_entry, canonical_entry) in enumerate(zip(legacy_entries, canonical_entries)):
        legacy_pose = source_pose(legacy_entry, strategy)
        canonical_pose = source_pose(canonical_entry, strategy)
        legacy_tf = pose_wxyz_to_matrix(legacy_pose)
        canonical_tf = pose_wxyz_to_matrix(canonical_pose)
        expected_source_rotation = legacy_tf[:3, :3]
        if strategy in {"orientation", "fused"}:
            expected_source_rotation = expected_source_rotation @ R_CGRASP_RTCP
        position_delta_m = float(np.linalg.norm(legacy_tf[:3, 3] - canonical_tf[:3, 3]))
        rotation_max_abs = rotation_error(expected_source_rotation, canonical_tf[:3, :3])
        identity_match = identity(legacy_entry, strategy) == identity(canonical_entry, strategy)

        canonical_target = pose_wxyz_to_matrix(
            np.asarray(canonical_entry["pose_world_wxyz"], dtype=np.float64).reshape(7)
        )
        expected_target_rotation = legacy_tf[:3, :3]
        if strategy in {"orientation", "fused", "human_replay"}:
            expected_target_rotation = expected_target_rotation @ R_CGRASP_RTCP
        target_position_delta_m = float(
            np.linalg.norm(legacy_tf[:3, 3] - canonical_target[:3, 3])
        )
        target_rotation_max_abs = rotation_error(
            expected_target_rotation, canonical_target[:3, :3]
        )
        canonical_link6_position = (
            canonical_target[:3, 3] - SERVER_TOOL_LENGTH_M * canonical_target[:3, 0]
        )
        local_delta = canonical_target[:3, :3].T @ (
            canonical_link6_position - canonical_target[:3, 3]
        )
        entry_ok = bool(
            identity_match
            and position_delta_m <= POSITION_TOL_M
            and rotation_max_abs <= ROTATION_TOL
            and target_position_delta_m <= POSITION_TOL_M
            and target_rotation_max_abs <= ROTATION_TOL
            and np.max(
                np.abs(local_delta - np.array([-SERVER_TOOL_LENGTH_M, 0.0, 0.0]))
            )
            <= 1e-9
        )
        rows.append(
            {
                "index": index,
                "identity": list(identity(legacy_entry, strategy)),
                "identity_match": identity_match,
                "source_position_delta_m": position_delta_m,
                "source_rotation_relation": (
                    "R_canonical = R_legacy @ R_CGRASP_RTCP"
                    if strategy in {"orientation", "fused"}
                    else "same semantic source axes"
                ),
                "source_rotation_max_abs_error": rotation_max_abs,
                "canonical_target_relation": (
                    "R_W_RTCP = R_W_CGRASP @ R_CGRASP_RTCP"
                    if strategy in {"orientation", "fused", "human_replay"}
                    else "native Top-score axes interpreted as RTCP"
                ),
                "canonical_target_position_delta_m": target_position_delta_m,
                "canonical_target_rotation_max_abs_error": target_rotation_max_abs,
                "canonical_target_world_xyz_m": canonical_target[:3, 3].tolist(),
                "canonical_derived_link6_world_xyz_m": canonical_link6_position.tolist(),
                "canonical_link6_minus_rtcp_in_local_rtcp_m": local_delta.tolist(),
                "ok": entry_ok,
            }
        )
    ok = all(row["ok"] for row in rows)
    return ok, {"candidate_count": len(rows), "entries": rows}


def contract_ok(path: Path, logic: str, strategy: str) -> tuple[bool, dict[str, Any]]:
    if not path.is_file():
        return False, {"reason": "missing_contract", "path": str(path)}
    data = load_json(path)
    target = data.get("row_specific_target_contract", {})
    if logic == "canonical":
        expected = {
            "target_retreat_m": 0.0,
            "candidate_target_local_z_offset_m": 0.0,
            "approach_axis": "local_x",
        }
    elif strategy == "human_replay":
        expected = {
            "target_retreat_m": 0.14,
            "candidate_target_local_z_offset_m": 0.0,
            "approach_axis": "local_z",
        }
    else:
        expected = {
            "target_retreat_m": 0.0,
            "candidate_target_local_z_offset_m": -0.05,
            "approach_axis": "local_z",
        }
    actual = {
        "target_retreat_m": float(target.get("target_retreat_m", 0.0)),
        "candidate_target_local_z_offset_m": float(target.get("candidate_target_local_z_offset_m", 0.0)),
        "approach_axis": target.get("pregrasp", {}).get("axis"),
    }
    ok = actual == expected
    return ok, {"expected": expected, "actual": actual, "path": str(path.resolve())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "schema": "piper_canonical_tcp_v1.ik_semantic_source_audit.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "episode_id": str(args.id),
        "contract": (
            "same semantic AnyGrasp/human center; Legacy keeps native historical target "
            "adaptation while Canonical interprets that center as Piper RTCP"
        ),
        "tolerances": {"position_m": POSITION_TOL_M, "rotation_matrix_max_abs": ROTATION_TOL},
        "strategies": {},
    }
    all_sources_ok = True
    all_contracts_ok = True
    for strategy in STRATEGIES:
        rel = Path(args.task) / f"foundation_input_{args.id}" / "eepose" / strategy
        summary_name = "plan_summary_human_replay.json" if strategy == "human_replay" else "plan_summary.json"
        legacy_summary_path = args.legacy_root / rel / summary_name
        canonical_summary_path = args.canonical_root / rel / summary_name
        if not legacy_summary_path.is_file() or not canonical_summary_path.is_file():
            source_detail = {
                "ok": False,
                "reason": "missing_plan_summary",
                "legacy": str(legacy_summary_path),
                "canonical": str(canonical_summary_path),
            }
        else:
            source_ok, source_detail = compare_strategy(
                strategy,
                load_json(legacy_summary_path),
                load_json(canonical_summary_path),
            )
            source_detail["ok"] = source_ok
        legacy_contract_ok, legacy_contract = contract_ok(
            args.legacy_root / rel / "input_target_contract.json", "legacy", strategy
        )
        canonical_contract_ok, canonical_contract = contract_ok(
            args.canonical_root / rel / "input_target_contract.json", "canonical", strategy
        )
        result["strategies"][strategy] = {
            "semantic_source": source_detail,
            "contracts": {
                "legacy": {"ok": legacy_contract_ok, **legacy_contract},
                "canonical": {"ok": canonical_contract_ok, **canonical_contract},
            },
        }
        all_sources_ok = all_sources_ok and bool(source_detail["ok"])
        all_contracts_ok = all_contracts_ok and legacy_contract_ok and canonical_contract_ok
    result["all_semantic_sources_identical"] = all_sources_ok
    result["all_contracts_valid"] = all_contracts_ok
    result["all_ok"] = all_sources_ok and all_contracts_ok
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_ok": result["all_ok"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if result["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
