#!/usr/bin/env python3
"""Replace the generic renderer label with branch-specific provenance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SEMANTICS = {
    "oursv2": (
        "Piper real T_B_RTCP numeric pose -> unchanged URDF link6 IK target; "
        "legacy target_retreat=0/apply_global_trans_to_ik=0"
    ),
    "canonical": (
        "Piper real T_B_RTCP -> inverse Ry(-1.57)@Tx(0.19) -> URDF link6 IK target"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--branch", choices=tuple(SEMANTICS), required=True)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload["generic_renderer_original_qpos_label"] = payload.get("qpos")
    payload["schema"] = "piper_canonical_tcp_v1.real_control_sim_manifest.v1"
    payload["control_branch"] = args.branch
    payload["qpos"] = SEMANTICS[args.branch]
    payload["evaluation_semantics"] = (
        "planned q is evaluated as physical Piper RTCP using "
        "T_B_L6URDF @ Ry(-1.57) @ Tx(0.19)"
    )
    temporary = args.manifest.with_name(f".{args.manifest.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.manifest)
    print(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
