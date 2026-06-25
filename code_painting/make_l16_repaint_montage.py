#!/usr/bin/env python3
"""Build side-by-side L16 visualization montages.

Panels:
1. HaMeR hand_vis_gripper
2. Foundation object replay
3. L16 robot head_cam_plan
4. Optional Stage-1 human/object inpaint
5. Optional final SAM3 repaint
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_TASKS = (
    "pick_diverse_bottles",
    "place_bread_basket",
    "stack_cups",
    "handover_bottle",
    "pnp_bread",
    "pnp_tray",
)

HAND_ROOT = Path("/home/zaijia001/ssd/data/piper/hand")
L16_ROOT = Path(
    "/home/zaijia001/ssd/RoboTwin/code_painting/"
    "anygrasp_plan_keyframes_piper_d435_replay_axes/L16_human_replay_clean"
)
STAGE1_ROOT = Path("/home/zaijia001/ssd/inpainting_sam2_robot/results_repaint_piper_h2_l16")
FINAL_ROOT = Path("/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_visible_reinit")
DEFAULT_OUTPUT_ROOT = Path("/home/zaijia001/ssd/RoboTwin/code_painting/l16_repaint_montage")


@dataclass(frozen=True)
class Panel:
    label: str
    path: Path
    required: bool


def parse_id_spec(spec: str) -> list[int]:
    ids: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            ids.extend(range(start, end + step, step))
        else:
            ids.append(int(chunk))
    return sorted(dict.fromkeys(ids))


def parse_tasks(args: argparse.Namespace) -> list[str]:
    if args.task:
        tasks = args.task
    elif args.tasks:
        tasks = args.tasks
    else:
        tasks = list(DEFAULT_TASKS)
    unknown = [task for task in tasks if task not in DEFAULT_TASKS]
    if unknown:
        raise SystemExit(f"Unknown task(s): {', '.join(unknown)}")
    return tasks


def ffprobe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    try:
        value = float(out)
    except ValueError:
        return 0.0
    return value if value > 0.0 else 0.0


def drawtext_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return next(iter(paths))


def resolve_panels(task: str, video_id: int, args: argparse.Namespace) -> list[Panel]:
    hand_dir = HAND_ROOT / task
    foundation_candidates = (
        hand_dir / "foundation_replay_d435" / f"foundation_input_{video_id}" / "head_cam_replay.mp4",
        hand_dir / "foundation_replay" / f"foundation_input_{video_id}" / "head_cam_replay.mp4",
    )
    panels = [
        Panel(
            "HaMeR gripper",
            hand_dir / "harmer_output" / f"hand_vis_gripper_{video_id}.mp4",
            True,
        ),
        Panel("Foundation object", first_existing(foundation_candidates), True),
        Panel(
            "L16 robot plan",
            L16_ROOT / task / f"foundation_input_{video_id}" / "head_cam_plan.mp4",
            True,
        ),
    ]
    if args.include_optional:
        final_task_dir = args.final_root / args.final_task_subdir_template.format(task=task, id=video_id)
        final_episode_dir = final_task_dir / args.final_episode_dir_template.format(task=task, id=video_id)
        panels.extend(
            [
                Panel(
                    "Stage1 inpaint",
                    args.stage1_root
                    / "stage1_human_object"
                    / task
                    / f"id_{video_id}"
                    / "stage1_human_inpaint"
                    / f"removed_w_mask_rgb_{video_id}.mp4",
                    False,
                ),
                Panel(
                    args.final_label,
                    final_episode_dir / args.final_video_name.format(task=task, id=video_id),
                    False,
                ),
            ]
        )
    return panels


def build_filter(panels: list[Panel], target_duration: float, args: argparse.Namespace) -> str:
    filter_parts: list[str] = []
    outputs: list[str] = []
    for index, panel in enumerate(panels):
        source_duration = ffprobe_duration(panel.path)
        ratio = target_duration / source_duration if source_duration > 0 and target_duration > 0 else 1.0
        label = drawtext_escape(f"{index + 1}. {panel.label}")
        out_label = f"v{index}"
        filter_parts.append(
            (
                f"[{index}:v]"
                f"setpts={ratio:.8f}*PTS,"
                f"fps={args.fps},"
                f"scale={args.panel_width}:{args.panel_height}:force_original_aspect_ratio=decrease,"
                f"pad={args.panel_width}:{args.panel_height}:(ow-iw)/2:(oh-ih)/2:black,"
                "setsar=1,"
                f"drawtext=text='{label}':x=12:y=12:fontsize=22:fontcolor=lime:"
                "box=1:boxcolor=black@0.55"
                f"[{out_label}]"
            )
        )
        outputs.append(f"[{out_label}]")
    filter_parts.append("".join(outputs) + f"hstack=inputs={len(outputs)}:shortest=1[v]")
    return ";".join(filter_parts)


def make_montage(task: str, video_id: int, args: argparse.Namespace) -> bool:
    all_panels = resolve_panels(task, video_id, args)
    missing_required = [panel for panel in all_panels if panel.required and not panel.path.is_file()]
    if missing_required:
        print(f"[skip] {task} id={video_id}: missing required video(s)")
        for panel in missing_required:
            print(f"  - {panel.label}: {panel.path}")
        return False

    panels = [panel for panel in all_panels if panel.required or panel.path.is_file()]
    target_duration = ffprobe_duration(panels[2].path if len(panels) >= 3 else panels[0].path)
    output_dir = args.output_root / task / f"id_{video_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / f"compare_hamer_foundation_l16_repaint_{task}_id{video_id}.mp4"
    manifest_path = output_video.with_suffix(".json")
    if output_video.exists() and not args.overwrite:
        print(f"[exists] {output_video}")
        return True

    cmd = ["ffmpeg", "-y" if args.overwrite else "-n"]
    for panel in panels:
        cmd.extend(["-i", str(panel.path)])
    cmd.extend(
        [
            "-filter_complex",
            build_filter(panels, target_duration, args),
            "-map",
            "[v]",
            "-an",
        ]
    )
    if args.max_duration_sec > 0:
        cmd.extend(["-t", str(args.max_duration_sec)])
    cmd.extend(["-r", str(args.fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_video)])

    print(f"[run] {task} id={video_id} -> {output_video}")
    subprocess.run(cmd, check=True)
    manifest = {
        "task": task,
        "id": video_id,
        "output_video": str(output_video),
        "target_duration_sec": target_duration,
        "panels": [{"label": panel.label, "path": str(panel.path)} for panel in panels],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ok] {output_video}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", choices=DEFAULT_TASKS, help="Single/batch task; can repeat.")
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, help="Batch task list.")
    parser.add_argument("--id", dest="single_id", type=int, help="Single id.")
    parser.add_argument("--ids", default=None, help="Batch ids, for example 0 or 0-4 or 0,2,7-10.")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage1_root", type=Path, default=STAGE1_ROOT)
    parser.add_argument("--final_root", type=Path, default=FINAL_ROOT)
    parser.add_argument("--final_task_subdir_template", default="e0_robot_object/{task}")
    parser.add_argument("--final_episode_dir_template", default="id_{id}_l16")
    parser.add_argument("--final_video_name", default="final_repainted.mp4")
    parser.add_argument("--final_label", default="Final repaint")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--panel_width", type=int, default=426)
    parser.add_argument("--panel_height", type=int, default=320)
    parser.add_argument("--max_duration_sec", type=float, default=0.0, help="0 means no output trim.")
    parser.add_argument("--include_optional", type=int, choices=(0, 1), default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.stage1_root = args.stage1_root.expanduser().resolve()
    args.final_root = args.final_root.expanduser().resolve()
    tasks = parse_tasks(args)
    if args.single_id is not None:
        ids = [args.single_id]
    elif args.ids:
        ids = parse_id_spec(args.ids)
    else:
        ids = [0]
    if not ids:
        raise SystemExit("No ids selected.")

    made = 0
    skipped = 0
    for task in tasks:
        for video_id in ids:
            try:
                if make_montage(task, video_id, args):
                    made += 1
                else:
                    skipped += 1
            except subprocess.CalledProcessError as exc:
                skipped += 1
                print(f"[error] {task} id={video_id}: ffmpeg failed with code {exc.returncode}", file=sys.stderr)
            except Exception as exc:  # Keep batch runs moving across tasks.
                skipped += 1
                print(f"[error] {task} id={video_id}: {exc}", file=sys.stderr)

    print(f"[summary] made_or_existing={made} skipped_or_failed={skipped}")
    return 0 if made > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
