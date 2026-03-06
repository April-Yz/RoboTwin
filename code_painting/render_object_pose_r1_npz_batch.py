#!/usr/bin/env python3
"""Batch replay FoundationPose poses.npz folders with head-only clean video output."""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


THIS_DIR = Path(__file__).resolve().parent
SINGLE_SCRIPT = THIS_DIR / "render_object_pose_r1_npz.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch replay FoundationPose output folders. Default mode is head-only video without overlay text."
    )
    parser.add_argument("--input_root", type=Path, required=True, help="Directory containing FoundationPose result folders.")
    parser.add_argument("--mesh_file", type=Path, required=True, help="Object mesh file (e.g., bottle.obj).")
    parser.add_argument("--output_root", type=Path, required=True, help="Root output directory for per-sequence replay results.")
    parser.add_argument("--ids", type=str, nargs="*", default=None, help="Optional subset IDs (e.g., 5 7 9). Matches trailing folder id or exact folder name.")
    parser.add_argument("--skip_existing", type=int, default=1, help="Skip when output_dir/head_cam_replay.mp4 already exists.")
    parser.add_argument("--continue_on_error", type=int, default=1, help="Continue processing other folders after failure.")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--fovy_deg", type=float, default=90.0)
    parser.add_argument("--robot_config", type=Path, default=(THIS_DIR.parent / "robot_config_R1.json"))
    parser.add_argument("--torso_qpos", type=float, nargs=4, default=[0.25, -0.4, -0.85, 0.0])
    parser.add_argument("--robot_base_pose", type=float, nargs=7, default=None, metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"))
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=-1)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--save_png_frames", type=int, default=0)
    parser.add_argument("--enable_viewer", type=int, default=0)
    parser.add_argument("--viewer_frame_delay", type=float, default=0.0)
    parser.add_argument("--viewer_wait_at_end", type=int, default=0)
    parser.add_argument("--disable_table", type=int, default=1)
    parser.add_argument("--need_topp", type=int, default=0)
    parser.add_argument("--camera_cv_axis_mode", type=str, default="legacy_r1")
    parser.add_argument("--head_camera_local_pos", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--head_camera_local_quat_wxyz", type=float, nargs=4, default=[1.0, 1.0, -1.0, 1.0])
    parser.add_argument("--object_actor_name", type=str, default="tracked_object")
    parser.add_argument("--head_only", type=int, default=1, help="Default 1: only save head_cam replay.")
    parser.add_argument("--overlay_text", type=int, default=0, help="Default 0: no left-top parameter text.")
    parser.add_argument("--third_person_view", type=int, default=0, help="Used only when --head_only 0.")
    return parser.parse_args()


def discover_pose_dirs(input_root: Path) -> List[Path]:
    root = input_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"input_root is not a directory: {root}")

    pose_dirs: List[Path] = []
    if (root / "poses.npz").is_file():
        pose_dirs.append(root)

    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "poses.npz").is_file():
            pose_dirs.append(child)
    return pose_dirs


def trailing_id(folder_name: str) -> Optional[str]:
    match = re.search(r"(\d+)$", folder_name)
    if match is None:
        return None
    return match.group(1)


def filter_pose_dirs_by_ids(pose_dirs: Iterable[Path], ids: Optional[List[str]]) -> List[Path]:
    if not ids:
        return list(pose_dirs)
    wanted = {str(x) for x in ids}
    selected: List[Path] = []
    for d in pose_dirs:
        name = d.name
        tid = trailing_id(name)
        if name in wanted or (tid is not None and tid in wanted):
            selected.append(d)
    return selected


def build_single_command(args: argparse.Namespace, pose_dir: Path, output_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(SINGLE_SCRIPT),
        "--input_npz",
        str((pose_dir / "poses.npz").resolve()),
        "--mesh_file",
        str(args.mesh_file.resolve()),
        "--output_dir",
        str(output_dir.resolve()),
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
        "--frame_start",
        str(args.frame_start),
        "--frame_end",
        str(args.frame_end),
        "--frame_stride",
        str(args.frame_stride),
        "--max_frames",
        str(args.max_frames),
        "--head_only",
        str(args.head_only),
        "--overlay_text",
        str(args.overlay_text),
        "--third_person_view",
        str(args.third_person_view),
        "--save_png_frames",
        str(args.save_png_frames),
        "--enable_viewer",
        str(args.enable_viewer),
        "--viewer_frame_delay",
        str(args.viewer_frame_delay),
        "--viewer_wait_at_end",
        str(args.viewer_wait_at_end),
        "--disable_table",
        str(args.disable_table),
        "--need_topp",
        str(args.need_topp),
        "--camera_cv_axis_mode",
        str(args.camera_cv_axis_mode),
        "--head_camera_local_pos",
        *(str(v) for v in args.head_camera_local_pos),
        "--head_camera_local_quat_wxyz",
        *(str(v) for v in args.head_camera_local_quat_wxyz),
        "--object_actor_name",
        str(args.object_actor_name),
    ]
    if args.robot_base_pose is not None:
        cmd.extend(["--robot_base_pose", *(str(v) for v in args.robot_base_pose)])
    return cmd


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.mesh_file = args.mesh_file.resolve()
    args.output_root = args.output_root.resolve()
    args.robot_config = args.robot_config.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    if not args.mesh_file.is_file():
        raise FileNotFoundError(f"Mesh file not found: {args.mesh_file}")
    if not args.robot_config.is_file():
        raise FileNotFoundError(f"Robot config not found: {args.robot_config}")

    all_pose_dirs = discover_pose_dirs(args.input_root)
    if not all_pose_dirs:
        raise RuntimeError(f"No poses.npz found under {args.input_root}")

    selected_pose_dirs = filter_pose_dirs_by_ids(all_pose_dirs, args.ids)
    if not selected_pose_dirs:
        raise RuntimeError(f"No folders matched --ids under {args.input_root}: {args.ids}")

    logging.info("Found %d pose folders, selected %d", len(all_pose_dirs), len(selected_pose_dirs))
    logging.info("Selected folders: %s", [p.name for p in selected_pose_dirs])

    failures = []
    for pose_dir in selected_pose_dirs:
        output_dir = args.output_root / pose_dir.name
        head_video = output_dir / "head_cam_replay.mp4"
        if bool(args.skip_existing) and head_video.is_file():
            logging.info("Skipping %s because %s exists", pose_dir.name, head_video)
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_single_command(args, pose_dir=pose_dir, output_dir=output_dir)
        logging.info("Running %s -> %s", pose_dir.name, output_dir)
        logging.info("Command: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append((pose_dir.name, exc.returncode))
            logging.error("Failed %s (exit_code=%s)", pose_dir.name, exc.returncode)
            if not bool(args.continue_on_error):
                break

    if failures:
        raise SystemExit(f"Batch completed with failures: {failures}")
    logging.info("Batch replay completed successfully.")


if __name__ == "__main__":
    main()
