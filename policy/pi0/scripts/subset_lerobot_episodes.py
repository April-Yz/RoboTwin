#!/usr/bin/env python3
"""Create a re-indexed LeRobot v2 subset from selected episodes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME


def parse_episode_spec(spec: str, allow_duplicates: bool = False) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid decreasing episode range: {part}")
            values = range(start, end + 1)
        else:
            values = [int(part)]
        for value in values:
            if value < 0:
                raise ValueError(f"Episode index must be non-negative: {value}")
            if allow_duplicates:
                selected.append(value)
            else:
                seen.add(value)
    if allow_duplicates:
        if not selected:
            raise ValueError("No episodes selected.")
        return selected
    if not seen:
        raise ValueError("No episodes selected.")
    return sorted(seen)


def resolve_dataset_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    cache_path = HF_LEROBOT_HOME / value
    if cache_path.exists():
        return cache_path
    raise FileNotFoundError(f"Dataset not found as path or repo id: {value}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def update_count_stat(stats: dict, key: str, new_value: int) -> None:
    if key not in stats:
        return
    item = stats[key]
    item["min"] = [new_value]
    item["max"] = [new_value]
    item["mean"] = [float(new_value)]
    item["std"] = [0.0]


def update_index_stat(stats: dict, key: str, start: int, length: int) -> None:
    if key not in stats:
        return
    end = start + length - 1
    mean = (start + end) / 2.0
    variance = ((length * length) - 1) / 12.0 if length > 0 else 0.0
    item = stats[key]
    item["min"] = [int(start)]
    item["max"] = [int(end)]
    item["mean"] = [float(mean)]
    item["std"] = [float(variance**0.5)]
    item["count"] = [int(length)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subset a local LeRobot v2 dataset and re-index selected episodes.")
    parser.add_argument("--source", required=True, help="Source dataset path or repo id under HF_LEROBOT_HOME.")
    parser.add_argument("--output-repo-id", required=True, help="Destination repo id under HF_LEROBOT_HOME, e.g. local/name_25ep.")
    parser.add_argument("--episodes", required=True, help="Episode spec, e.g. '0,1-5,7' or '0-24'.")
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Preserve input order and repeated episode indexes. Default remains sorted unique episodes.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove the destination dataset if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = resolve_dataset_path(args.source)
    output_dir = HF_LEROBOT_HOME / args.output_repo_id
    selected = parse_episode_spec(args.episodes, allow_duplicates=args.allow_duplicates)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)

    src_meta = source_dir / "meta"
    src_data = source_dir / "data" / "chunk-000"
    src_videos = source_dir / "videos" / "chunk-000"
    if not src_meta.is_dir() or not src_data.is_dir():
        raise FileNotFoundError(f"Source dataset is missing expected LeRobot v2 folders: {source_dir}")

    out_meta = output_dir / "meta"
    out_data = output_dir / "data" / "chunk-000"
    out_videos = output_dir / "videos" / "chunk-000"
    out_meta.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)
    out_videos.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_meta / "tasks.jsonl", out_meta / "tasks.jsonl")
    info = json.loads((src_meta / "info.json").read_text(encoding="utf-8"))
    episodes_by_index = {int(row["episode_index"]): row for row in load_jsonl(src_meta / "episodes.jsonl")}
    stats_by_index = {int(row["episode_index"]): row for row in load_jsonl(src_meta / "episodes_stats.jsonl")}

    new_episode_rows: list[dict] = []
    new_stats_rows: list[dict] = []
    total_frames = 0

    video_dirs = [p for p in src_videos.iterdir() if p.is_dir()] if src_videos.is_dir() else []
    for new_idx, old_idx in enumerate(selected):
        src_parquet = src_data / f"episode_{old_idx:06d}.parquet"
        if not src_parquet.is_file():
            raise FileNotFoundError(f"Missing source parquet for episode {old_idx}: {src_parquet}")

        df = pd.read_parquet(src_parquet)
        length = int(len(df))
        df["episode_index"] = new_idx
        df["frame_index"] = range(length)
        df["index"] = range(total_frames, total_frames + length)
        df.to_parquet(out_data / f"episode_{new_idx:06d}.parquet", index=False)

        old_episode_row = dict(episodes_by_index.get(old_idx, {"tasks": []}))
        old_episode_row["episode_index"] = new_idx
        old_episode_row["length"] = length
        new_episode_rows.append(old_episode_row)

        if old_idx in stats_by_index:
            stats_row = json.loads(json.dumps(stats_by_index[old_idx]))
            stats_row["episode_index"] = new_idx
            stats = stats_row.get("stats", {})
            update_count_stat(stats, "episode_index", new_idx)
            update_index_stat(stats, "frame_index", 0, length)
            update_index_stat(stats, "index", total_frames, length)
            new_stats_rows.append(stats_row)

        for video_dir in video_dirs:
            src_video = video_dir / f"episode_{old_idx:06d}.mp4"
            if not src_video.is_file():
                continue
            dst_dir = out_videos / video_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video, dst_dir / f"episode_{new_idx:06d}.mp4")

        print(f"[copy] old_episode={old_idx} -> new_episode={new_idx} frames={length}")
        total_frames += length

    write_jsonl(out_meta / "episodes.jsonl", new_episode_rows)
    write_jsonl(out_meta / "episodes_stats.jsonl", new_stats_rows)

    info["total_episodes"] = len(selected)
    info["total_frames"] = total_frames
    info["total_chunks"] = 1
    camera_count = len(video_dirs)
    info["total_videos"] = len(selected) * camera_count
    info["splits"] = {"train": f"0:{len(selected)}"}
    (out_meta / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")

    print(f"[done] source={source_dir}")
    print(f"[done] output={output_dir}")
    print(f"[done] selected={selected}")
    print(f"[done] total_episodes={len(selected)} total_frames={total_frames} total_videos={info['total_videos']}")


if __name__ == "__main__":
    main()
