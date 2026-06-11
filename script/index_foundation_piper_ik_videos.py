#!/usr/bin/env python3
"""Index per-ID Foundation collection videos as episode<ID> files."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path(
    "/home/zaijia001/ssd/RoboTwin/data/pick_diverse_bottles_piper_ik_foundation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create episode<ID> video aliases from Foundation per-ID outputs. "
            "The source data remains unchanged."
        )
    )
    parser.add_argument("--version", choices=["v1", "v2", "v3", "v4"], required=True)
    parser.add_argument("--mode", choices=["o1", "o1.1", "o1.2"], default="o1.2")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-video-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["symlink", "copy"], default="symlink")
    parser.add_argument(
        "--replace-episode",
        action="store_true",
        help="Remove existing episode<ID> MP4 files before writing that ID.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_pattern(version: str, mode: str, run_tag: str) -> re.Pattern[str]:
    mode_tag = mode.replace(".", "_")
    if mode == "o1":
        stem = rf"demo_piper_ik_foundation_{version}_id(?P<id>\d+)_frame\d+"
    else:
        stem = rf"demo_piper_ik_foundation_{version}_{mode_tag}_id(?P<id>\d+)"
    if run_tag:
        stem += rf"_{re.escape(run_tag)}"
    return re.compile(rf"^{stem}$")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_video_dir.expanduser().resolve()
    pattern = source_pattern(args.version, args.mode, args.run_tag)

    sources = []
    for directory in source_root.iterdir():
        if not directory.is_dir():
            continue
        match = pattern.fullmatch(directory.name)
        if match:
            sources.append((int(match.group("id")), directory))
    sources.sort()
    if not sources:
        raise FileNotFoundError(
            f"No Foundation outputs matched {pattern.pattern!r} under {source_root}"
        )

    manifest = {
        "schema": "foundation_piper_ik_video_index.v1",
        "version": args.version,
        "mode": args.mode,
        "run_tag": args.run_tag,
        "method": args.method,
        "episodes": {},
    }
    planned = []
    conflicts = []
    for foundation_id, source_dir in sources:
        source_videos = sorted((source_dir / "video").glob("episode0_*.mp4"))
        if not source_videos:
            print(f"[skip] id={foundation_id}: no episode0 videos in {source_dir / 'video'}")
            continue
        existing = sorted(output_dir.glob(f"episode{foundation_id}_*.mp4"))
        if existing and not args.replace_episode:
            names = ", ".join(path.name for path in existing[:4])
            message = (
                f"Episode {foundation_id} already exists in {output_dir}: {names}. "
                "Use a new output directory or pass --replace-episode explicitly."
            )
            if args.dry_run:
                conflicts.append(message)
                print(f"[conflict] {message}")
                continue
            raise FileExistsError(message)
        mapped = []
        for source in source_videos:
            destination = output_dir / source.name.replace(
                "episode0_", f"episode{foundation_id}_", 1
            )
            planned.append((foundation_id, source.resolve(), destination, existing))
            mapped.append(destination.name)
        manifest["episodes"][str(foundation_id)] = {
            "source_dir": str(source_dir.resolve()),
            "videos": mapped,
        }

    if args.dry_run:
        for foundation_id, source, destination, _ in planned:
            print(f"[dry-run] id={foundation_id}: {source} -> {destination}")
        print(
            f"[dry-run] indexable episodes={len(manifest['episodes'])} "
            f"conflicts={len(conflicts)}"
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    cleared_ids = set()
    for foundation_id, source, destination, existing in planned:
        if args.replace_episode and foundation_id not in cleared_ids:
            for old_path in existing:
                old_path.unlink()
            cleared_ids.add(foundation_id)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        if args.method == "symlink":
            destination.symlink_to(os.path.relpath(source, destination.parent))
        else:
            shutil.copy2(source, destination)
        print(f"[index] id={foundation_id}: {destination.name}")

    manifest_path = output_dir / "foundation_episode_index.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[index] manifest={manifest_path}")


if __name__ == "__main__":
    main()
