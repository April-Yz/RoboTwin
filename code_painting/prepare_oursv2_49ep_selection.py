#!/usr/bin/env python3
"""Build traceable 49-episode selections for the ours v2 data ablation."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TASKS = (
    "pick_diverse_bottles", "place_bread_basket", "handover_bottle",
    "pnp_bread", "pnp_tray", "stack_cups",
)
REVIEW = Path("/home/zaijia001/ssd/RoboTwin/code_painting/l16_ours_review")
STACK_REVIEW = Path("/home/zaijia001/ssd/RoboTwin/code_painting/l16_ours_review_rightcam_m003_color")
HEAD = Path("/home/zaijia001/ssd/inpainting_sam3_robot/results_repaint_piper_h2_l16_whitebg_invert/stage2_color_rightcam_m003_full_0_120/e0_robot_object")
PLANNER = Path("/home/zaijia001/ssd/RoboTwin/code_painting/anygrasp_plan_keyframes_piper_d435_replay_axes/L16_de_human_replay_clean_right_cam")
OUTPUT = Path("/home/zaijia001/ssd/RoboTwin/code_painting/l16_oursv2_review_49ep")


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select ours v2 in y -> m -> unreviewed -> n order.")
    p.add_argument("--review-root", type=Path, default=REVIEW)
    p.add_argument("--stack-review-root", type=Path, default=STACK_REVIEW)
    p.add_argument("--head-root", type=Path, default=HEAD)
    p.add_argument("--planner-root", type=Path, default=PLANNER)
    p.add_argument("--output-root", type=Path, default=OUTPUT)
    p.add_argument("--target-count", type=int, default=49)
    p.add_argument("--max-source-id", type=int, default=119)
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--allow-repeat", action="store_true")
    p.add_argument("--print-processed-spec", action="store_true")
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--processed-root", type=Path)
    return p.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def category(item: dict | None) -> str:
    if not item:
        return "unreviewed"
    label = str(item.get("label", "")).lower()
    status = str(item.get("status", "")).lower()
    usable = item.get("usable")
    if label == "y" or status in {"usable", "good", "accept", "accepted"} or usable is True:
        return "accepted"
    if label == "m" or usable == "ambiguous" or status in {"medium", "maybe", "ambiguous"}:
        return "medium"
    if label == "n" or status in {"reject", "discard", "bad"} or usable is False:
        return "bad"
    return "unreviewed"


def file_ids(root: Path, task: str, pattern: str, regex: str) -> set[int]:
    rx = re.compile(regex)
    result = set()
    for path in (root / task).glob(pattern):
        match = rx.search(str(path))
        if match:
            result.add(int(match.group(1)))
    return result


def available(args: argparse.Namespace, task: str) -> list[int]:
    head = file_ids(args.head_root, task, "id_*/final_repainted.mp4",
                    r"/id_(\d+)_l16_white_color_human_object/")
    planner = file_ids(args.planner_root, task, "foundation_input_*/pose_debug.jsonl",
                       r"/foundation_input_(\d+)/")
    return sorted(value for value in head & planner if value <= args.max_source_id)


def review_path(args: argparse.Namespace, task: str) -> Path:
    root = args.stack_review_root if task == "stack_cups" else args.review_root
    return root / "selections" / task / "ours_review_selection.json"


def build(args: argparse.Namespace, task: str) -> dict:
    source_review = review_path(args, task)
    source_videos = load(source_review).get("videos", {})
    ids = available(args, task)
    items = {source_id: dict(source_videos.get(f"id_{source_id}", {})) for source_id in ids}
    buckets = {name: [] for name in ("accepted", "medium", "unreviewed", "bad")}
    for source_id in ids:
        buckets[category(items[source_id])].append(source_id)

    selected = [
        source_id
        for name in ("accepted", "medium", "unreviewed", "bad")
        for source_id in buckets[name]
    ][:args.target_count]
    repeated = []
    if len(selected) < args.target_count:
        if not args.allow_repeat:
            raise RuntimeError(f"{task}: only {len(selected)} unique rows; use --allow-repeat")
        pool = buckets["accepted"] or buckets["medium"] or buckets["unreviewed"] or buckets["bad"]
        if not pool:
            raise RuntimeError(f"{task}: no available rows")
        while len(selected) < args.target_count:
            source_id = pool[len(repeated) % len(pool)]
            selected.append(source_id)
            repeated.append(source_id)

    unique = list(dict.fromkeys(selected))
    counts = Counter(category(items[source_id]) for source_id in unique)
    occurrences = Counter()
    rows = []
    for rank, source_id in enumerate(selected):
        occurrences[source_id] += 1
        rows.append({
            "rank": rank, "source_id": source_id,
            "source_category": category(items[source_id]),
            "occurrence": occurrences[source_id],
            "is_repeat": occurrences[source_id] > 1,
        })

    generated = {}
    for source_id in unique:
        item = dict(items[source_id])
        item.update({
            "task": task, "id": source_id, "reviewed": True, "label": "y",
            "status": "usable", "usable": True,
            "oursv2_49ep_source_category": category(items[source_id]),
        })
        generated[f"id_{source_id}"] = item

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    out = args.output_root / "selections" / task
    out.mkdir(parents=True, exist_ok=True)
    generated_review = {
        "meta": {
            "task": task, "updated_at": now,
            "kind": "l16_oursv2_49ep_generated_review",
            "source_review_json": str(source_review),
            "priority": ["accepted", "medium", "unreviewed", "bad", "repeat_accepted"],
        },
        "videos": generated,
    }
    (out / "ours_review_selection.json").write_text(
        json.dumps(generated_review, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "task": task, "created_at": now, "target_count": args.target_count,
        "available_count": len(ids), "source_review_json": str(source_review),
        "head_root": str(args.head_root), "planner_root": str(args.planner_root),
        "available_buckets": buckets,
        "composition": {
            "accepted_unique": counts["accepted"], "medium_unique": counts["medium"],
            "unreviewed_unique": counts["unreviewed"], "bad_unique": counts["bad"],
            "repeated": len(repeated), "unique_total": len(unique),
            "final_total": len(selected),
        },
        "selected_source_ids": selected, "unique_selected_source_ids": unique,
        "selection_rows": rows,
    }
    (out / "oursv2_49ep_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def summary(args: argparse.Namespace, manifests: list[dict]) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "oursv2_49ep_selection_summary.json").write_text(
        json.dumps({"kind": "oursv2_49ep_selection_summary", "tasks": manifests},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# OursV2 49ep selection composition", "",
        "Priority: accepted (y) -> medium (m) -> unreviewed -> bad (n) -> repeat accepted only when unique planner outputs are insufficient.", "",
        "| task | accepted | medium | unreviewed | bad | repeated | unique | total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for manifest in manifests:
        c = manifest["composition"]
        lines.append(
            f"| {manifest['task']} | {c['accepted_unique']} | {c['medium_unique']} | "
            f"{c['unreviewed_unique']} | {c['bad_unique']} | {c['repeated']} | "
            f"{c['unique_total']} | {c['final_total']} |"
        )
    output = "\n".join(lines) + "\n"
    (args.output_root / "oursv2_49ep_selection_summary.md").write_text(output, encoding="utf-8")
    print(output, end="")


def processed_spec(args: argparse.Namespace) -> None:
    if not args.task or not args.processed_root:
        raise SystemExit("--print-processed-spec requires --task and --processed-root")
    manifest = load(args.output_root / "selections" / args.task / "oursv2_49ep_selection_manifest.json")
    mapping = {}
    for path in args.processed_root.glob("episode_*/instructions.json"):
        payload = load(path)
        mapping[int(payload["source_episode_id"])] = int(path.parent.name.split("_")[-1])
    missing = sorted(set(manifest["selected_source_ids"]) - set(mapping))
    if missing:
        raise RuntimeError(f"{args.task}: selected source ids missing after processing: {missing}")
    print(",".join(str(mapping[source_id]) for source_id in manifest["selected_source_ids"]))


def main() -> None:
    args = args_parser()
    if args.print_processed_spec:
        processed_spec(args)
    else:
        summary(args, [build(args, task) for task in args.tasks])


if __name__ == "__main__":
    main()
