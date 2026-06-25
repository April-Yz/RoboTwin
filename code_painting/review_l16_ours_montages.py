#!/usr/bin/env python3
"""Review L16 ours montage videos and write selection JSON files.

The per-task JSON is compatible with process_repainted_planner_outputs.py
--review-json in strict mode: accepted rows get status=usable, rejected rows
get status=reject, and maybe rows use label=m without status.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit("OpenCV is required. Run in RoboTwin_bw, for example: source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh && conda activate RoboTwin_bw") from exc

TASKS = (
    "pick_diverse_bottles",
    "place_bread_basket",
    "handover_bottle",
    "pnp_bread",
    "pnp_tray",
    "stack_cups",
)

REPO_ROOT = Path("/home/zaijia001/ssd/RoboTwin")
MONTAGE_SCRIPT = REPO_ROOT / "code_painting" / "make_l16_repaint_montage.py"
DEFAULT_REVIEW_ROOT = REPO_ROOT / "code_painting" / "l16_ours_review"
DEFAULT_FINAL_ROOT = Path(
    "/home/zaijia001/ssd/inpainting_sam3_robot/"
    "results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object"
)
DEFAULT_STACK_FINAL_ROOT = Path(
    "/home/zaijia001/ssd/inpainting_sam3_robot/"
    "results_repaint_piper_h2_l16_whitebg_invert/e0_robot_object_b_points_negative"
)


def parse_id_spec(spec: str) -> list[int]:
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            ids.extend(range(start, end + step, step))
        else:
            ids.append(int(part))
    return sorted(dict.fromkeys(ids))


def parse_id_from_episode_dir(path: Path) -> int | None:
    name = path.name
    if not name.startswith("id_"):
        return None
    rest = name[3:]
    digits = []
    for ch in rest:
        if not ch.isdigit():
            break
        digits.append(ch)
    if not digits:
        return None
    return int("".join(digits))


def final_root_for_task(task: str, args: argparse.Namespace) -> Path:
    if task == "stack_cups" and args.stack_final_root:
        stack_root = args.stack_final_root.expanduser().resolve()
        if (stack_root / task).is_dir():
            return stack_root
    return args.final_root.expanduser().resolve()


def final_video_path(task: str, video_id: int, args: argparse.Namespace) -> Path:
    episode_dir = args.final_episode_dir_template.format(task=task, id=video_id)
    return final_root_for_task(task, args) / task / episode_dir / args.final_video_name


def discover_ids(task: str, args: argparse.Namespace) -> list[int]:
    task_root = final_root_for_task(task, args) / task
    if not task_root.is_dir():
        return []
    ids: list[int] = []
    for video in task_root.glob(f"*/{args.final_video_name}"):
        episode_id = parse_id_from_episode_dir(video.parent)
        if episode_id is not None:
            ids.append(episode_id)
    return sorted(set(ids))


def montage_path(task: str, video_id: int, args: argparse.Namespace) -> Path:
    return (
        args.review_root
        / "montages"
        / task
        / f"id_{video_id}"
        / f"compare_hamer_foundation_l16_repaint_{task}_id{video_id}.mp4"
    )


def make_montage(task: str, video_id: int, args: argparse.Namespace) -> Path | None:
    out_video = montage_path(task, video_id, args)
    if out_video.is_file() and not args.overwrite_montage:
        return out_video
    final_root = final_root_for_task(task, args)
    cmd = [
        sys.executable,
        str(MONTAGE_SCRIPT),
        "--task",
        task,
        "--id",
        str(video_id),
        "--output_root",
        str(args.review_root / "montages"),
        "--final_root",
        str(final_root),
        "--final_task_subdir_template",
        "{task}",
        "--final_episode_dir_template",
        args.final_episode_dir_template,
        "--final_video_name",
        args.final_video_name,
        "--final_label",
        args.final_label,
        "--max_duration_sec",
        str(args.max_duration_sec),
        "--include_optional",
        "1",
    ]
    if args.overwrite_montage:
        cmd.append("--overwrite")
    print("[montage]", task, video_id)
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0 or not out_video.is_file():
        print(f"[skip] montage failed or missing: {task} id={video_id}")
        return None
    return out_video


def load_task_review(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"meta": {}, "videos": {}}


def task_review_path(task: str, args: argparse.Namespace) -> Path:
    return args.review_root / "selections" / task / "ours_review_selection.json"


def status_from_entry(entry: dict[str, Any]) -> str:
    status = str(entry.get("status", "")).lower()
    if status in {"usable", "good", "accept", "accepted"} or entry.get("label") == "y" or entry.get("usable") is True:
        return "Y"
    if status in {"reject", "discard", "bad"} or entry.get("label") == "n" or entry.get("usable") is False:
        return "N"
    if entry.get("label") == "m" or entry.get("usable") == "ambiguous":
        return "M"
    return "-"


def task_status_counts(task: str, entries: list[dict[str, Any]], review_payload: dict[str, Any]) -> dict[str, int]:
    counts = {"Y": 0, "N": 0, "M": 0, "-": 0, "total": 0}
    videos = review_payload.get("videos", {})
    for entry in entries:
        if entry["task"] != task:
            continue
        counts["total"] += 1
        status = status_from_entry(videos.get(f"id_{entry['id']}", {}))
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_reviews(reviews: dict[str, dict[str, Any]], args: argparse.Namespace) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S %z")
    combined: dict[str, Any] = {"meta": {"updated_at": now, "kind": "l16_ours_review"}, "tasks": {}}
    for task, payload in reviews.items():
        payload.setdefault("meta", {})
        payload["meta"].update({"task": task, "updated_at": now, "kind": "l16_ours_review"})
        path = task_review_path(task, args)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        combined["tasks"][task] = payload
    combined_path = args.review_root / "selections" / "ours_review_selection_all.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def set_label(entry: dict[str, Any], label: str) -> None:
    entry["reviewed"] = True
    entry["label"] = label
    entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    entry.pop("status", None)
    if label == "y":
        entry["status"] = "usable"
        entry["usable"] = True
    elif label == "n":
        entry["status"] = "reject"
        entry["usable"] = False
    elif label == "m":
        entry["usable"] = "ambiguous"


def clear_label(entry: dict[str, Any]) -> None:
    for key in ("label", "status", "usable"):
        entry.pop(key, None)
    entry["reviewed"] = False
    entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")


def overlay(frame, text_lines: list[str]) -> None:
    h, w = frame.shape[:2]
    pad = 8
    line_h = 24
    box_h = pad * 2 + line_h * len(text_lines)
    y0 = max(0, h - box_h)
    cv2.rectangle(frame, (0, y0), (w, h), (0, 0, 0), -1)
    for i, line in enumerate(text_lines):
        cv2.putText(
            frame,
            line,
            (12, y0 + pad + 18 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


def review_entries(entries: list[dict[str, Any]], reviews: dict[str, dict[str, Any]], args: argparse.Namespace) -> None:
    if not entries:
        print("[error] no reviewable montage videos")
        return
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    idx = 0
    speed = args.initial_speed
    playing = True
    cap = None
    frame_index = 0

    def open_current() -> None:
        nonlocal cap, frame_index
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(str(entries[idx]["montage_video"]))
        frame_index = 0

    open_current()
    while True:
        entry_info = entries[idx]
        task = entry_info["task"]
        video_id = entry_info["id"]
        key_name = f"id_{video_id}"
        review_payload = reviews[task]
        item = review_payload["videos"].setdefault(key_name, {})
        item.update(
            {
                "task": task,
                "id": video_id,
                "montage_video": str(entry_info["montage_video"]),
                "final_video": str(entry_info["final_video"]),
            }
        )

        ok, frame = cap.read() if cap is not None else (False, None)
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_index = 0
            ok, frame = cap.read()
            if not ok:
                print(f"[error] failed to read {entry_info['montage_video']}")
                key = ord(".")
            else:
                playing = False
                key = -1
        else:
            frame_index += 1
            key = -1

        if ok:
            status = status_from_entry(item)
            fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            counts = task_status_counts(task, entries, review_payload)
            accepted = counts["Y"]
            remaining = max(0, args.target_count - accepted)
            target_text = f"/{args.target_count}" if args.target_count > 0 else ""
            lines = [
                f"{idx + 1}/{len(entries)} task={task} id={video_id} status={status} speed={speed:.2f}x frame={frame_index}/{total_frames}",
                f"task stats: accepted={accepted}{target_text} remaining={remaining} maybe={counts['M']} reject={counts['N']} unreviewed={counts['-']} total={counts['total']}",
                "keys: y accept | n reject | m maybe | u clear | space play/pause | . next | , prev | ]/[ speed | r replay | s save | q quit",
            ]
            overlay(frame, lines)
            cv2.imshow(args.window_name, frame)
            delay = 1 if playing else 0
            if playing:
                delay = max(1, int(1000 / max(1e-6, fps * speed)))
            key = cv2.waitKeyEx(delay)
        else:
            key = cv2.waitKeyEx(0)

        if key in (-1, 255):
            continue
        if key in (ord("q"), 27):
            write_reviews(reviews, args)
            break
        if key == ord("s"):
            write_reviews(reviews, args)
            print(f"[saved] {args.review_root / 'selections'}")
            continue
        if key == ord(" "):
            playing = not playing
            continue
        if key == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_index = 0
            playing = True
            continue
        if key == ord("]"):
            speed = min(8.0, speed * 1.5)
            continue
        if key == ord("["):
            speed = max(0.125, speed / 1.5)
            continue
        if key == ord("y"):
            set_label(item, "y")
            write_reviews(reviews, args)
            idx = min(len(entries) - 1, idx + 1)
            open_current()
            continue
        if key == ord("n"):
            set_label(item, "n")
            write_reviews(reviews, args)
            idx = min(len(entries) - 1, idx + 1)
            open_current()
            continue
        if key == ord("m"):
            set_label(item, "m")
            write_reviews(reviews, args)
            idx = min(len(entries) - 1, idx + 1)
            open_current()
            continue
        if key == ord("u"):
            clear_label(item)
            write_reviews(reviews, args)
            continue
        if key in (ord("."), ord("l"), 65363, 2555904):
            idx = min(len(entries) - 1, idx + 1)
            open_current()
            continue
        if key in (ord(","), ord("h"), 65361, 2424832):
            idx = max(0, idx - 1)
            open_current()
            continue

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--ids", default=None, help="Optional id spec such as 0-24 or 0,3,8. Default discovers final_repainted outputs.")
    parser.add_argument("--review_root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--final_root", type=Path, default=DEFAULT_FINAL_ROOT)
    parser.add_argument("--stack_final_root", type=Path, default=DEFAULT_STACK_FINAL_ROOT)
    parser.add_argument("--final_episode_dir_template", default="id_{id}_l16_whitebg_human_object")
    parser.add_argument("--final_video_name", default="final_repainted.mp4")
    parser.add_argument("--final_label", default="Ours repaint")
    parser.add_argument("--max_duration_sec", type=float, default=0.0)
    parser.add_argument("--make_montages", type=int, choices=(0, 1), default=1)
    parser.add_argument("--overwrite_montage", action="store_true")
    parser.add_argument("--initial_speed", type=float, default=1.0)
    parser.add_argument("--target_count", type=int, default=25, help="Accepted episode target shown in the overlay.")
    parser.add_argument("--window_name", default="L16 ours review")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.review_root = args.review_root.expanduser().resolve()
    args.final_root = args.final_root.expanduser().resolve()
    args.stack_final_root = args.stack_final_root.expanduser().resolve() if args.stack_final_root else None

    entries: list[dict[str, Any]] = []
    reviews: dict[str, dict[str, Any]] = {}
    for task in args.tasks:
        ids = parse_id_spec(args.ids) if args.ids else discover_ids(task, args)
        if not ids:
            print(f"[skip] {task}: no final videos discovered")
            continue
        reviews[task] = load_task_review(task_review_path(task, args))
        reviews[task].setdefault("videos", {})
        for video_id in ids:
            final_video = final_video_path(task, video_id, args)
            if not final_video.is_file():
                print(f"[skip] {task} id={video_id}: missing final {final_video}")
                continue
            if args.make_montages:
                video = make_montage(task, video_id, args)
            else:
                video = montage_path(task, video_id, args)
            if video and video.is_file():
                entries.append({"task": task, "id": video_id, "montage_video": video, "final_video": final_video})

    entries.sort(key=lambda item: (TASKS.index(item["task"]), item["id"]))
    print(f"[review] entries={len(entries)} review_root={args.review_root}")
    write_reviews(reviews, args)
    for task in args.tasks:
        if task not in reviews:
            continue
        counts = task_status_counts(task, entries, reviews[task])
        accepted = counts["Y"]
        remaining = max(0, args.target_count - accepted)
        target_text = f"/{args.target_count}" if args.target_count > 0 else ""
        print(
            f"[task-summary] {task}: accepted={accepted}{target_text} remaining={remaining} "
            f"maybe={counts['M']} reject={counts['N']} unreviewed={counts['-']} total={counts['total']}"
        )
    review_entries(entries, reviews, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
