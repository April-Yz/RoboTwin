#!/usr/bin/env python3
"""Audit and atomically transcode MP4 files for VS Code/Chromium playback."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


SCHEMA = "piper_canonical_tcp_v1.vscode_video.v1"


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required executable is unavailable: {name}")
    return path


def probe_mp4(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            _tool("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,profile,width,height,nb_frames,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed: {path}")
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream: {path}")
    return streams[0]


def is_vscode_compatible(info: dict[str, Any]) -> bool:
    return info.get("codec_name") == "h264" and info.get("pix_fmt") == "yuv420p"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_error(path: Path) -> str | None:
    result = subprocess.run(
        [
            _tool("ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return None
    return result.stderr.strip() or f"ffmpeg decode failed with exit {result.returncode}"


def _matching_geometry(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if (before.get("width"), before.get("height")) != (after.get("width"), after.get("height")):
        return False
    before_frames = before.get("nb_frames")
    after_frames = after.get("nb_frames")
    if before_frames not in (None, "N/A") and after_frames not in (None, "N/A"):
        return str(before_frames) == str(after_frames)
    return True


def process_mp4(path: Path, *, apply: bool) -> dict[str, Any]:
    path = path.resolve()
    record: dict[str, Any] = {"path": str(path), "apply": apply}
    tmp_path = path.with_name(f".{path.stem}.vscode_tmp{path.suffix}")
    try:
        before = probe_mp4(path)
        record["before"] = before
        record["size_before_bytes"] = path.stat().st_size
        if is_vscode_compatible(before):
            record["status"] = "compatible"
            return record
        if not apply:
            record["status"] = "would_convert"
            return record

        record["sha256_before"] = _sha256(path)
        tmp_path.unlink(missing_ok=True)
        result = subprocess.run(
            [
                _tool("ffmpeg"),
                "-y",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-c:a",
                "aac",
                str(tmp_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not tmp_path.is_file() or tmp_path.stat().st_size <= 0:
            raise RuntimeError(result.stderr.strip() or "ffmpeg produced no output")
        after = probe_mp4(tmp_path)
        if not is_vscode_compatible(after):
            raise RuntimeError(f"Unexpected target format: {after}")
        if not _matching_geometry(before, after):
            raise RuntimeError(f"Frame geometry/count changed: before={before}, after={after}")
        decode_error = _decode_error(tmp_path)
        if decode_error is not None:
            raise RuntimeError(decode_error)

        record["after"] = after
        record["size_after_bytes"] = tmp_path.stat().st_size
        record["sha256_after"] = _sha256(tmp_path)
        os.chmod(tmp_path, path.stat().st_mode)
        os.replace(tmp_path, path)
        record["status"] = "converted"
        return record
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        record["status"] = "failed"
        record["error"] = str(exc)
        return record


def ensure_vscode_mp4(path: Path) -> None:
    result = process_mp4(path, apply=True)
    if result["status"] == "failed":
        raise RuntimeError(f"VS Code MP4 transcode failed for {path}: {result['error']}")
    print(f"[video] {result['status']} h264/yuv420p: {path}")


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Atomically replace incompatible MP4 files after validation.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(
        path for path in root.rglob("*.mp4")
        if ".vscode_tmp" not in path.name
    )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(process_mp4, path, apply=args.apply): path for path in files}
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            record["path"] = str(Path(record["path"]).relative_to(root))
            results.append(record)
            print(f"[{index}/{len(files)}] {record['status']} {record['path']}", flush=True)

    results.sort(key=lambda item: item["path"])
    counts: dict[str, int] = {}
    for record in results:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    payload = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "apply": args.apply,
        "summary": {"total": len(results), **counts},
        "results": results,
    }
    manifest = args.manifest or root / "vscode_transcode_manifest.json"
    _write_manifest(manifest.resolve(), payload)
    print(json.dumps(payload["summary"], sort_keys=True))
    print(f"manifest={manifest.resolve()}")
    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
