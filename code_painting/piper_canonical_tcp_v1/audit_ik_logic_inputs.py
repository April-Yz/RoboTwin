#!/usr/bin/env python3
"""Verify each Legacy/Canonical column selected exactly the same numeric targets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


STRATEGIES = ("orientation", "fused", "top_score", "human_replay")


def signatures(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("selected_candidates", [])
    return [
        {
            "arm": str(entry.get("arm")),
            "source_frame": int(entry.get("source_frame", -1)),
            "candidate_idx": int(entry.get("candidate_idx", -1)),
            "pose_world_wxyz": [float(value) for value in entry["pose_world_wxyz"]],
        }
        for entry in entries
    ]


def compare(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    if len(left) != len(right):
        return False, {"reason": "candidate_count", "legacy": len(left), "canonical": len(right)}
    max_abs = 0.0
    rows = []
    for index, (legacy, canonical) in enumerate(zip(left, right)):
        identity_ok = all(legacy[key] == canonical[key] for key in ("arm", "source_frame", "candidate_idx"))
        delta = np.asarray(legacy["pose_world_wxyz"]) - np.asarray(canonical["pose_world_wxyz"])
        row_max = float(np.max(np.abs(delta)))
        max_abs = max(max_abs, row_max)
        rows.append({"index": index, "identity_match": identity_ok, "max_abs_pose_delta": row_max})
    ok = all(row["identity_match"] for row in rows) and max_abs <= 1e-9
    return ok, {"candidate_count": len(left), "max_abs_pose_delta": max_abs, "entries": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, Any] = {
        "schema": "piper_canonical_tcp_v1.ik_logic_input_audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "episode_id": str(args.id),
        "contract": "same selected candidate identity and pose_world_wxyz per column; tolerance=1e-9",
        "strategies": {},
    }
    all_ok = True
    for strategy in STRATEGIES:
        rel = Path(args.task) / f"foundation_input_{args.id}" / "eepose" / strategy / "plan_summary.json"
        legacy_path = args.legacy_root / rel
        canonical_path = args.canonical_root / rel
        if not legacy_path.is_file() or not canonical_path.is_file():
            detail = {"ok": False, "reason": "missing_plan_summary", "legacy": str(legacy_path), "canonical": str(canonical_path)}
        else:
            ok, detail = compare(signatures(legacy_path), signatures(canonical_path))
            detail["ok"] = ok
        result["strategies"][strategy] = detail
        all_ok = all_ok and bool(detail["ok"])
    result["all_inputs_identical"] = all_ok
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_inputs_identical": all_ok, "output": str(args.output)}, ensure_ascii=False))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
