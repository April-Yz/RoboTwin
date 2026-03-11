#!/usr/bin/env python3
"""Batch planner for AnyGrasp-driven two-keyframe RoboTwin demos."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


THIS_DIR = Path(__file__).resolve().parent
SINGLE_SCRIPT = THIS_DIR / "plan_anygrasp_keyframes_r1.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch plan RoboTwin demos from AnyGrasp replay outputs.")
    parser.add_argument("--anygrasp_root", type=Path, required=True, help="Root containing per-video AnyGrasp dirs like d_pour_blue_1.")
    parser.add_argument("--replay_root", type=Path, required=True, help="Root containing per-video replay dirs like d_pour_blue_1.")
    parser.add_argument("--hand_dir", type=Path, required=True, help="Directory containing hand_detections_<id>.npz files.")
    parser.add_argument("--output_root", type=Path, required=True, help="Root for per-video planned demo outputs.")
    parser.add_argument("--ids", type=str, nargs="*", default=None, help="Optional subset ids like 1 4 22.")
    parser.add_argument("--keyframes", type=int, nargs=2, default=[1, 22], metavar=("GRASP_FRAME", "ACTION_FRAME"))
    parser.add_argument("--arm", choices=["auto", "left", "right"], default="auto")
    parser.add_argument("--skip_existing", type=int, default=1)
    parser.add_argument("--continue_on_error", type=int, default=1)
    parser.add_argument("--robot_config", type=Path, default=(THIS_DIR.parent / "robot_config_R1.json"))
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--fovy_deg", type=float, default=90.0)
    parser.add_argument("--torso_qpos", type=float, nargs=4, default=[0.25, -0.4, -0.85, 0.0])
    parser.add_argument("--robot_base_pose", type=float, nargs=7, default=None, metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"))
    parser.add_argument("--open_gripper", type=float, default=1.0)
    parser.add_argument("--close_gripper", type=float, default=0.0)
    parser.add_argument("--approach_offset_m", type=float, default=0.08)
    parser.add_argument("--head_only", type=int, default=1)
    parser.add_argument("--overlay_text", type=int, default=1)
    parser.add_argument("--third_person_view", type=int, default=0)
    parser.add_argument("--enable_viewer", type=int, default=0)
    parser.add_argument("--viewer_frame_delay", type=float, default=0.0)
    parser.add_argument("--viewer_wait_at_end", type=int, default=0)
    parser.add_argument("--disable_table", type=int, default=1)
    parser.add_argument("--lighting_mode", choices=["default", "front", "front_no_shadow"], default="front_no_shadow")
    parser.add_argument("--camera_cv_axis_mode", type=str, default="legacy_r1")
    parser.add_argument("--head_camera_local_pos", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--head_camera_local_quat_wxyz", type=float, nargs=4, default=[1.0, 1.0, -1.0, 1.0])
    return parser.parse_args()


def trailing_id(name: str) -> Optional[int]:
    match = re.search(r"(\d+)$", name)
    return None if match is None else int(match.group(1))


def discover_anygrasp_dirs(anygrasp_root: Path) -> List[Path]:
    root = anygrasp_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"anygrasp_root is not a directory: {root}")
    dirs = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if (child / "grasps").is_dir():
            dirs.append(child)
    return sorted(
        dirs,
        key=lambda p: (
            trailing_id(p.name) is None,
            trailing_id(p.name) if trailing_id(p.name) is not None else sys.maxsize,
            p.name,
        ),
    )


def filter_dirs(video_dirs: List[Path], ids: Optional[List[str]]) -> List[Path]:
    if not ids:
        return video_dirs
    wanted = {str(item) for item in ids}
    return [video_dir for video_dir in video_dirs if video_dir.name in wanted or str(trailing_id(video_dir.name)) in wanted]


def build_single_command(args: argparse.Namespace, anygrasp_dir: Path, replay_dir: Path, hand_npz: Path, output_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(SINGLE_SCRIPT),
        "--anygrasp_dir",
        str(anygrasp_dir.resolve()),
        "--replay_dir",
        str(replay_dir.resolve()),
        "--hand_npz",
        str(hand_npz.resolve()),
        "--output_dir",
        str(output_dir.resolve()),
        "--keyframes",
        *(str(v) for v in args.keyframes),
        "--arm",
        str(args.arm),
        "--robot_config",
        str(args.robot_config.resolve()),
        "--image_width",
        str(args.image_width),
        "--image_height",
        str(args.image_height),
        "--fps",
        str(args.fps),
        "--fovy_deg",
        str(args.fovy_deg),
        "--torso_qpos",
        *(str(v) for v in args.torso_qpos),
        "--open_gripper",
        str(args.open_gripper),
        "--close_gripper",
        str(args.close_gripper),
        "--approach_offset_m",
        str(args.approach_offset_m),
        "--head_only",
        str(args.head_only),
        "--overlay_text",
        str(args.overlay_text),
        "--third_person_view",
        str(args.third_person_view),
        "--enable_viewer",
        str(args.enable_viewer),
        "--viewer_frame_delay",
        str(args.viewer_frame_delay),
        "--viewer_wait_at_end",
        str(args.viewer_wait_at_end),
        "--disable_table",
        str(args.disable_table),
        "--lighting_mode",
        str(args.lighting_mode),
        "--camera_cv_axis_mode",
        str(args.camera_cv_axis_mode),
        "--head_camera_local_pos",
        *(str(v) for v in args.head_camera_local_pos),
        "--head_camera_local_quat_wxyz",
        *(str(v) for v in args.head_camera_local_quat_wxyz),
    ]
    if args.robot_base_pose is not None:
        cmd.extend(["--robot_base_pose", *(str(v) for v in args.robot_base_pose)])
    return cmd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    args.anygrasp_root = args.anygrasp_root.resolve()
    args.replay_root = args.replay_root.resolve()
    args.hand_dir = args.hand_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.robot_config = args.robot_config.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if not args.hand_dir.is_dir():
        raise NotADirectoryError(f"hand_dir not found: {args.hand_dir}")
    if not args.robot_config.is_file():
        raise FileNotFoundError(f"robot_config not found: {args.robot_config}")

    all_dirs = discover_anygrasp_dirs(args.anygrasp_root)
    selected_dirs = filter_dirs(all_dirs, args.ids)
    if not selected_dirs:
        raise RuntimeError(f"No AnyGrasp directories matched under {args.anygrasp_root}")

    logging.info("Found %d AnyGrasp dirs, selected %d", len(all_dirs), len(selected_dirs))
    failures = []
    successes = []
    for anygrasp_dir in selected_dirs:
        video_name = anygrasp_dir.name
        video_id = trailing_id(video_name)
        if video_id is None:
            failures.append((video_name, "invalid_video_id"))
            logging.error("Skipping %s because trailing id is missing", video_name)
            if not bool(args.continue_on_error):
                break
            continue

        replay_dir = args.replay_root / video_name
        hand_npz = args.hand_dir / f"hand_detections_{video_id}.npz"
        output_dir = args.output_root / video_name
        summary_path = output_dir / "plan_summary.json"
        if bool(args.skip_existing) and summary_path.is_file():
            logging.info("Skipping %s because %s exists", video_name, summary_path)
            continue

        if not replay_dir.is_dir():
            failures.append((video_name, f"missing_replay_dir:{replay_dir}"))
            logging.error("Missing replay dir for %s: %s", video_name, replay_dir)
            if not bool(args.continue_on_error):
                break
            continue
        if not hand_npz.is_file():
            failures.append((video_name, f"missing_hand_npz:{hand_npz}"))
            logging.error("Missing hand npz for %s: %s", video_name, hand_npz)
            if not bool(args.continue_on_error):
                break
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_single_command(args, anygrasp_dir, replay_dir, hand_npz, output_dir)
        logging.info("Running %s", video_name)
        logging.info("Command: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            successes.append(video_name)
        except subprocess.CalledProcessError as exc:
            failures.append((video_name, f"exit_code:{exc.returncode}"))
            logging.error("Failed %s (exit_code=%s)", video_name, exc.returncode)
            if not bool(args.continue_on_error):
                break

    batch_summary = {
        "anygrasp_root": str(args.anygrasp_root),
        "replay_root": str(args.replay_root),
        "hand_dir": str(args.hand_dir),
        "output_root": str(args.output_root),
        "keyframes": [int(v) for v in args.keyframes],
        "selected_ids": list(args.ids) if args.ids else None,
        "successes": successes,
        "failures": failures,
    }
    summary_path = args.output_root / "batch_plan_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2)

    if failures:
        raise SystemExit(f"Batch planning completed with failures: {failures}")
    logging.info("Batch planning completed successfully.")


if __name__ == "__main__":
    main()
