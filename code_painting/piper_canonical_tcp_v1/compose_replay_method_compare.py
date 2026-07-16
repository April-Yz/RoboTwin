#!/usr/bin/env python3
"""Compose three Canonical AnyGrasp strategies and one legacy Human Replay video."""

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


SCHEMA = "piper_canonical_tcp_v1.replay_method_compare.v1"
CELL_W = 640
CELL_H = 360
FOOTER_H = 92
CANONICAL_OUTPUT_SIZE = (CELL_W * 2, CELL_H * 2 + FOOTER_H)
FULL_OUTPUT_SIZE = (CELL_W * 3, CELL_H * 2 + FOOTER_H)


def method_semantics(
    canonical_approach_offset_m: float,
    legacy_approach_offset_m: float,
    legacy_target_retreat_m: float,
) -> list[dict[str, Any]]:
    """Return the explicit frame/offset contract shown by the four panels."""
    canonical_common = {
        "planner_target": "T_W_RTCP",
        "position_frame": "world",
        "final_target_position_offset_m": 0.0,
        "pregrasp_axis": "local_RTCP_+X",
        "pregrasp_offset_m": float(canonical_approach_offset_m),
        "camera": "D435-calibrated simulated head view",
    }
    return [
        {
            "key": "orientation",
            "label": "Canonical Orientation",
            "source": "robot-frame preview T_W_CGRASP",
            "selection": "best orientation candidate",
            "axis_conversion": "R_W_RTCP = R_W_CGRASP @ R_CGRASP_RTCP",
            **canonical_common,
        },
        {
            "key": "fused",
            "label": "Canonical Fused",
            "source": "robot-frame preview T_W_CGRASP",
            "selection": "0.25 AnyGrasp score + 0.75 orientation score",
            "axis_conversion": "R_W_RTCP = R_W_CGRASP @ R_CGRASP_RTCP",
            **canonical_common,
        },
        {
            "key": "top_score",
            "label": "Canonical Top-score",
            "source": "raw AnyGrasp candidate transformed D435 camera -> world",
            "selection": "highest native AnyGrasp score",
            "axis_conversion": "raw candidate axes interpreted as RTCP",
            **canonical_common,
        },
        {
            "key": "canonical_human_replay",
            "label": "Canonical Human Replay",
            "source": "human hand/gripper pose transformed D435 camera -> world",
            "axis_conversion": "R_W_RTCP = R_W_CGRASP_HUMAN @ R_CGRASP_RTCP",
            "planner_target": "T_W_RTCP",
            "position_frame": "world",
            "final_target_position_offset_m": 0.0,
            "pregrasp_axis": "local_RTCP_+X",
            "pregrasp_offset_m": float(canonical_approach_offset_m),
            "camera": "D435-calibrated simulated head view",
        },
        {
            "key": "legacy_human_replay",
            "label": "Legacy Human Replay",
            "source": "human hand/gripper pose transformed D435 camera -> world",
            "planner_target": "legacy numeric pose target (not Canonical T_W_RTCP contract)",
            "position_frame": "world",
            "final_target_position_offset_m": -float(legacy_target_retreat_m),
            "final_target_offset_axis": "local_human_gripper_+Z",
            "pregrasp_axis": "local_human_gripper_+Z",
            "pregrasp_offset_m": float(legacy_approach_offset_m),
            "camera": "D435-calibrated simulated head view",
        },
    ]


def _open_video(path: Path) -> tuple[cv2.VideoCapture, dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0))
    if fps <= 0.0 or frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(
            f"Invalid video metadata: path={path} fps={fps} frames={frames} size={width}x{height}"
        )
    return cap, {
        "path": str(path.resolve()),
        "fps": fps,
        "frame_count": frames,
        "width": width,
        "height": height,
        "duration_s": frames / fps,
    }


def _letterbox(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(CELL_W / width, CELL_H / height)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
    canvas = np.full((CELL_H, CELL_W, 3), (18, 21, 26), dtype=np.uint8)
    x0 = (CELL_W - target[0]) // 2
    y0 = (CELL_H - target[1]) // 2
    canvas[y0 : y0 + target[1], x0 : x0 + target[0]] = resized
    return canvas


def _annotate(panel: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    image = panel.copy()
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (CELL_W, 58), (16, 19, 24), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0.0, image)
    cv2.putText(
        image,
        title,
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        subtitle,
        (14, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (201, 212, 224),
        1,
        cv2.LINE_AA,
    )
    return image


def _read_frame(
    cap: cv2.VideoCapture,
    info: dict[str, Any],
    time_s: float,
    last: np.ndarray | None,
) -> np.ndarray:
    source_index = min(
        info["frame_count"] - 1,
        max(0, int(round(time_s * info["fps"]))),
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
    ok, frame = cap.read()
    if ok:
        return frame
    if last is not None:
        return last
    raise RuntimeError(f"Cannot decode first frame: {info['path']}")


def _status(path: Path) -> dict[str, Any]:
    summary = path.parent / "plan_summary.json"
    result: dict[str, Any] = {
        "success_marker": (path.parent / "SUCCESS").is_file(),
        "exit_code": (path.parent / "EXIT_CODE").read_text().strip()
        if (path.parent / "EXIT_CODE").is_file()
        else None,
    }
    if summary.is_file():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            result["execution_success"] = data.get("execution_success")
            result["execution_failed"] = data.get("execution_failed")
        except (OSError, json.JSONDecodeError) as exc:
            result["summary_error"] = str(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-episode-root", type=Path, required=True)
    parser.add_argument("--canonical-human-root", type=Path, required=True)
    parser.add_argument("--legacy-human-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canonical-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--canonical-approach-offset-m", type=float, default=0.12)
    parser.add_argument("--legacy-approach-offset-m", type=float, default=0.12)
    parser.add_argument("--legacy-target-retreat-m", type=float, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    canonical = args.canonical_episode_root.resolve()
    canonical_human = args.canonical_human_root.resolve()
    legacy = args.legacy_human_root.resolve()
    sources = [
        canonical / "eepose/orientation/head_cam_plan.mp4",
        canonical / "eepose/fused/head_cam_plan.mp4",
        canonical / "eepose/top_score/head_cam_plan.mp4",
        canonical_human / "head_cam_plan.mp4",
        legacy / "head_cam_plan.mp4",
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing method videos: " + ", ".join(missing))
    if args.output.exists() and not args.force:
        print(f"[skip-existing] {args.output}")
        return 0

    semantics = method_semantics(
        args.canonical_approach_offset_m,
        args.legacy_approach_offset_m,
        args.legacy_target_retreat_m,
    )
    caps: list[cv2.VideoCapture] = []
    infos: list[dict[str, Any]] = []
    try:
        for source in sources:
            cap, info = _open_video(source)
            caps.append(cap)
            infos.append(info)
        output_fps = min(info["fps"] for info in infos)
        duration_s = max(info["duration_s"] for info in infos)
        output_frames = max(1, int(math.ceil(duration_s * output_fps)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.canonical_output.parent.mkdir(parents=True, exist_ok=True)
        full_writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            FULL_OUTPUT_SIZE,
        )
        canonical_writer = cv2.VideoWriter(
            str(args.canonical_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            CANONICAL_OUTPUT_SIZE,
        )
        if not full_writer.isOpened() or not canonical_writer.isOpened():
            raise RuntimeError(f"Cannot create video: {args.output}")
        last: list[np.ndarray | None] = [None] * len(caps)
        subtitles = (
            "AnyGrasp origin | CGRASP axes -> RTCP",
            "AnyGrasp origin | 0.25 score + 0.75 rot",
            "AnyGrasp origin | raw top score, RTCP axes",
            "Human pose origin | CGRASP axes -> RTCP",
            f"local +Z | final retreat={args.legacy_target_retreat_m:.3f} m",
        )
        for frame_index in range(output_frames):
            time_s = frame_index / output_fps
            panels = []
            for index, (cap, info, semantic, subtitle) in enumerate(
                zip(caps, infos, semantics, subtitles)
            ):
                frame = _read_frame(cap, info, time_s, last[index])
                last[index] = frame
                panels.append(
                    _annotate(_letterbox(frame), semantic["label"], subtitle)
                )
            full_canvas = np.full(
                (FULL_OUTPUT_SIZE[1], FULL_OUTPUT_SIZE[0], 3),
                (31, 37, 45),
                dtype=np.uint8,
            )
            full_canvas[0:CELL_H, 0:CELL_W] = panels[0]
            full_canvas[0:CELL_H, CELL_W : CELL_W * 2] = panels[1]
            full_canvas[0:CELL_H, CELL_W * 2 : CELL_W * 3] = panels[2]
            full_canvas[CELL_H : CELL_H * 2, 320:960] = panels[3]
            full_canvas[CELL_H : CELL_H * 2, 960:1600] = panels[4]
            canonical_canvas = np.full(
                (CANONICAL_OUTPUT_SIZE[1], CANONICAL_OUTPUT_SIZE[0], 3),
                (31, 37, 45),
                dtype=np.uint8,
            )
            canonical_canvas[0:CELL_H, 0:CELL_W] = panels[0]
            canonical_canvas[0:CELL_H, CELL_W : CELL_W * 2] = panels[1]
            canonical_canvas[CELL_H : CELL_H * 2, 0:CELL_W] = panels[2]
            canonical_canvas[CELL_H : CELL_H * 2, CELL_W : CELL_W * 2] = panels[3]
            footer_y = CELL_H * 2
            line1 = (
                "D435-calibrated simulated view | AnyGrasp translation: "
                "D435 camera -> WORLD (raw camera xyz is not world xyz)"
            )
            line2 = (
                "Canonical final offset=0; pregrasp=0.12m local_RTCP +X | "
                f"Legacy final retreat={args.legacy_target_retreat_m:.3f}m local human +Z"
            )
            for canvas in (full_canvas, canonical_canvas):
                cv2.putText(canvas, line1, (16, footer_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 238, 242), 1, cv2.LINE_AA)
                cv2.putText(canvas, line2, (16, footer_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (205, 214, 224), 1, cv2.LINE_AA)
            full_writer.write(full_canvas)
            canonical_writer.write(canonical_canvas)
        full_writer.release()
        canonical_writer.release()
    finally:
        for cap in caps:
            cap.release()

    ensure_vscode_mp4(args.output)
    ensure_vscode_mp4(args.canonical_output)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    payload = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "canonical_four_method": {
                "path": str(args.canonical_output.resolve()),
                "video": probe_mp4(args.canonical_output.resolve()),
            },
            "canonical_plus_legacy_five_method": {
                "path": str(args.output.resolve()),
                "video": probe_mp4(args.output.resolve()),
            },
        },
        "synchronization": {
            "rule": "time-based resampling; shorter methods hold their final frame",
            "fps": output_fps,
            "frame_count": output_frames,
            "duration_s": output_frames / output_fps,
        },
        "important_note": (
            "The first four panels share Canonical RTCP semantics. The fifth is an "
            "explicit legacy baseline and is not a fifth Canonical selector."
        ),
        "methods": [
            {
                **semantic,
                "video": info,
                "run_status": _status(path),
            }
            for semantic, info, path in zip(semantics, infos, sources)
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
