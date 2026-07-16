#!/usr/bin/env python3
"""Compose a 2x4 D435 video: original Legacy semantics over Canonical RTCP."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vscode_video import ensure_vscode_mp4, probe_mp4


CELL_W, CELL_H, FOOTER_H = 480, 270, 108
OUTPUT_SIZE = (CELL_W * 4, CELL_H * 2 + FOOTER_H)
STRATEGIES = (
    ("orientation", "Orientation"),
    ("fused", "Fused"),
    ("top_score", "Top-score"),
    ("human_replay", "Human Replay"),
)


def open_video(path: Path) -> tuple[cv2.VideoCapture, dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    info = {
        "path": str(path.resolve()),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
        "frame_count": int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)),
        "width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)),
        "height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)),
    }
    if min(info["fps"], info["frame_count"], info["width"], info["height"]) <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata: {info}")
    info["duration_s"] = info["frame_count"] / info["fps"]
    return cap, info


def letterbox(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(CELL_W / width, CELL_H / height)
    target_w, target_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((CELL_H, CELL_W, 3), (18, 21, 26), np.uint8)
    x0, y0 = (CELL_W - target_w) // 2, (CELL_H - target_h) // 2
    canvas[y0 : y0 + target_h, x0 : x0 + target_w] = resized
    return canvas


def annotate(panel: np.ndarray, row: str, method: str) -> np.ndarray:
    image = panel.copy()
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (CELL_W, 54), (12, 15, 20), -1)
    cv2.addWeighted(overlay, 0.84, image, 0.16, 0, image)
    color = (113, 204, 255) if row == "LEGACY ORIGINAL" else (134, 239, 172)
    cv2.putText(image, f"{row} | {method}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
    if row == "LEGACY ORIGINAL":
        detail = "native OursV2 target adapter"
    else:
        detail = "source center -> Piper RTCP"
    cv2.putText(image, detail, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (225, 229, 235), 1, cv2.LINE_AA)
    return image


def read_frame(cap: cv2.VideoCapture, info: dict[str, Any], time_s: float, last: np.ndarray | None) -> np.ndarray:
    index = min(info["frame_count"] - 1, max(0, int(round(time_s * info["fps"]))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if ok:
        return frame
    if last is not None:
        return last
    raise RuntimeError(f"Cannot decode first frame: {info['path']}")


def status(video: Path) -> dict[str, Any]:
    root = video.parent
    summary_path = root / "plan_summary.json"
    output: dict[str, Any] = {
        "success_marker": (root / "SUCCESS").is_file(),
        "exit_code": (root / "EXIT_CODE").read_text().strip() if (root / "EXIT_CODE").is_file() else None,
    }
    if summary_path.is_file():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        output["execution_success"] = data.get("execution_success")
        output["execution_failed"] = data.get("execution_failed")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--camera-profile", choices=("d435", "wide"), required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    audit = json.loads(args.input_audit.read_text(encoding="utf-8"))
    if not audit.get("all_ok"):
        raise ValueError(f"Semantic-source audit failed; refusing misleading grid: {args.input_audit}")
    if audit.get("camera_profile") != args.camera_profile:
        raise ValueError(
            f"Camera profile mismatch: audit={audit.get('camera_profile')} requested={args.camera_profile}"
        )
    if args.output.exists() and not args.force:
        print(f"[skip-existing] {args.output}")
        return 0

    sources: list[tuple[str, str, Path]] = []
    for logic, root in (("LEGACY ORIGINAL", args.legacy_root), ("CANONICAL RTCP", args.canonical_root)):
        for key, label in STRATEGIES:
            path = root / args.task / f"foundation_input_{args.id}" / "eepose" / key / "head_cam_plan.mp4"
            if not path.is_file():
                raise FileNotFoundError(path)
            sources.append((logic, label, path))

    caps: list[cv2.VideoCapture] = []
    infos: list[dict[str, Any]] = []
    try:
        for _, _, source in sources:
            cap, info = open_video(source)
            caps.append(cap); infos.append(info)
        expected = (
            {"width": 640, "height": 480, "fps": 5.0, "fovy_deg": 42.499880046655484}
            if args.camera_profile == "d435"
            else {"width": 640, "height": 360, "fps": 10.0, "fovy_deg": 90.0}
        )
        for info in infos:
            if info["width"] != expected["width"] or info["height"] != expected["height"]:
                raise ValueError(f"Mixed camera resolution for {args.camera_profile}: {info}")
            if abs(info["fps"] - expected["fps"]) > 1e-6:
                raise ValueError(f"Mixed camera fps for {args.camera_profile}: {info}")
        output_fps = min(info["fps"] for info in infos)
        duration_s = max(info["duration_s"] for info in infos)
        output_frames = max(1, math.ceil(duration_s * output_fps))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, OUTPUT_SIZE)
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video: {args.output}")
        last: list[np.ndarray | None] = [None] * len(caps)
        for frame_index in range(output_frames):
            time_s = frame_index / output_fps
            canvas = np.full((OUTPUT_SIZE[1], OUTPUT_SIZE[0], 3), (31, 37, 45), np.uint8)
            for index, ((logic, label, _), cap, info) in enumerate(zip(sources, caps, infos)):
                frame = read_frame(cap, info, time_s, last[index]); last[index] = frame
                row, col = divmod(index, 4)
                panel = annotate(letterbox(frame), logic, label)
                canvas[row * CELL_H : (row + 1) * CELL_H, col * CELL_W : (col + 1) * CELL_W] = panel
            footer_y = CELL_H * 2
            camera_label = (
                "D435 CALIBRATED | 640x480 | fovy=42.50deg"
                if args.camera_profile == "d435"
                else "WIDE DIAGNOSTIC | 640x360 | fovy=90.00deg"
            )
            cv2.putText(canvas, f"{camera_label} | shared semantic source: same AnyGrasp / human grasp center", (18, footer_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 238, 242), 1, cv2.LINE_AA)
            cv2.putText(canvas, "top: native Legacy offsets/retreat | bottom: source center = RTCP; link6 = RTCP - 0.19m @ local +X", (18, footer_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (205, 216, 228), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"frame {frame_index + 1}/{output_frames}", (OUTPUT_SIZE[0] - 210, footer_y + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (178, 190, 204), 1, cv2.LINE_AA)
            writer.write(canvas)
        writer.release()
    finally:
        for cap in caps:
            cap.release()

    ensure_vscode_mp4(args.output)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "schema": "piper_canonical_tcp_v1.ik_semantic_grid.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "layout": "2 rows x 4 columns",
        "camera": {
            "profile": args.camera_profile,
            "image_width": expected["width"],
            "image_height": expected["height"],
            "fps": expected["fps"],
            "fovy_deg": expected["fovy_deg"],
            "all_eight_sources_verified_same_profile": True,
        },
        "rows": {
            "top": "Original Legacy/OursV2 candidate-to-planner-target semantics",
            "bottom": "PiperCanonicalTCP-v1: semantic grasp center interpreted as server RTCP",
        },
        "columns": [key for key, _ in STRATEGIES],
        "shared_semantic_source": {
            "input": "same AnyGrasp candidate or Human hand/gripper center",
            "not_identical_numeric_target": True,
            "legacy_adapter": "Orientation/Fused/Top: -0.05m local +Z; Human: 0.14m local +Z retreat",
            "canonical_adapter": "origin preserved as RTCP; CGRASP axes remapped once where required; no final retreat",
            "canonical_link6": "RTCP - 0.19m along local RTCP +X after inverse Ry(-1.57) @ Tx(0.19)",
        },
        "input_audit": audit,
        "sources": [
            {"row": logic, "method": label, "video": str(path.resolve()), "status": status(path), "probe": probe_mp4(path)}
            for logic, label, path in sources
        ],
        "output": {"path": str(args.output.resolve()), "probe": probe_mp4(args.output)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[success] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
