#!/usr/bin/env python3
"""Measure V4 AnyGrasp strategy agreement and selected-position differences."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_AUDIT_ROOT = Path(
    "/home/zaijia001/ssd/RoboTwin/code_painting/selection_strategy_compare_v4"
)
STRATEGIES = ("oursv2", "orientation", "fused", "top_score")
ARMS = ("left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-zh-md", type=Path, default=None)
    parser.add_argument("--output-en-md", type=Path, default=None)
    parser.add_argument("--position-equal-tol-m", type=float, default=1e-9)
    parser.add_argument("--rotation-equal-tol-deg", type=float, default=1e-4)
    return parser.parse_args()


def metadata_paths(root: Path) -> List[Path]:
    paths = set(root.glob("*/id*_keyframe_*_metadata.json"))
    paths.update(root.glob("*/foundation_input_*/keyframe_*_metadata.json"))
    return sorted(paths)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def record_key(record: Mapping[str, Any]) -> Tuple[str, int, str, int, int]:
    return (
        str(record["task"]),
        int(record["episode_id"]),
        str(record["arm"]),
        int(record["event_index"]),
        int(record["requested_frame"]),
    )


def position(record: Mapping[str, Any]) -> np.ndarray:
    payload = record["selection_pose"]
    frame = str(payload.get("frame", "world"))
    return np.asarray(payload[f"position_{frame}_m"], dtype=np.float64).reshape(3)


def rotation(record: Mapping[str, Any]) -> np.ndarray:
    payload = record["selection_pose"]
    frame = str(payload.get("frame", "world"))
    return np.asarray(payload[f"rotation_{frame}"], dtype=np.float64).reshape(3, 3)


def rotation_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = np.asarray(a, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        b, dtype=np.float64
    ).reshape(3, 3)
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def number_summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def signed_number_summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "mean": None,
            "median": None,
            "std": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5.0)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def signed_xyz_summary(vectors: Sequence[Sequence[float]]) -> Dict[str, Any]:
    array = np.asarray(vectors, dtype=np.float64)
    if array.size == 0:
        array = np.empty((0, 3), dtype=np.float64)
    else:
        array = array.reshape(-1, 3)
    return {
        "definition": "anygrasp_selection_world_xyz - oursv2_human_replay_world_xyz",
        "x": signed_number_summary(array[:, 0].tolist()),
        "y": signed_number_summary(array[:, 1].tolist()),
        "z": signed_number_summary(array[:, 2].tolist()),
    }


def rate(count: int, total: int) -> float | None:
    return None if total == 0 else float(count) / float(total)


def compare_pair(
    left: Mapping[Tuple[str, int, str, int, int], Mapping[str, Any]],
    right: Mapping[Tuple[str, int, str, int, int], Mapping[str, Any]],
    *, position_tol_m: float,
    rotation_tol_deg: float,
    first_label: str,
    second_label: str,
) -> Dict[str, Any]:
    keys = sorted(set(left) & set(right))
    rows: List[Dict[str, Any]] = []
    for key in keys:
        a, b = left[key], right[key]
        same_frame = int(a["resolved_frame"]) == int(b["resolved_frame"])
        same_candidate = (
            same_frame
            and a.get("candidate_idx") is not None
            and b.get("candidate_idx") is not None
            and int(a["candidate_idx"]) == int(b["candidate_idx"])
        )
        position_difference = float(np.linalg.norm(position(a) - position(b)))
        rotation_difference = rotation_distance_deg(rotation(a), rotation(b))
        rows.append(
            {
                "key": key,
                "arm": key[2],
                "same_resolved_frame": same_frame,
                "same_candidate": same_candidate,
                "position_difference_m": position_difference,
                "rotation_difference_deg": rotation_difference,
                "same_pose": position_difference <= position_tol_m
                and rotation_difference <= rotation_tol_deg,
                "first_resolved_frame": int(a["resolved_frame"]),
                "second_resolved_frame": int(b["resolved_frame"]),
                "first_candidate_idx": a.get("candidate_idx"),
                "second_candidate_idx": b.get("candidate_idx"),
                "first_position_world_m": position(a).tolist(),
                "second_position_world_m": position(b).tolist(),
            }
        )

    def summarize(subset: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        selected = list(subset)
        same_frame_rows = [row for row in selected if row["same_resolved_frame"]]
        nonzero_position = [
            float(row["position_difference_m"])
            for row in selected
            if float(row["position_difference_m"]) > position_tol_m
        ]
        same_candidate_count = sum(bool(row["same_candidate"]) for row in selected)
        same_pose_count = sum(bool(row["same_pose"]) for row in selected)
        return {
            "paired_count": len(selected),
            "same_resolved_frame_count": len(same_frame_rows),
            "different_resolved_frame_count": len(selected) - len(same_frame_rows),
            "same_candidate_count": same_candidate_count,
            "same_candidate_rate_of_all_pairs": rate(same_candidate_count, len(selected)),
            "same_candidate_rate_of_same_frame_pairs": rate(
                same_candidate_count, len(same_frame_rows)
            ),
            "same_pose_count": same_pose_count,
            "same_pose_rate": rate(same_pose_count, len(selected)),
            "position_difference_m": number_summary(
                [float(row["position_difference_m"]) for row in selected]
            ),
            "position_difference_nonzero_m": number_summary(nonzero_position),
            "position_difference_same_frame_m": number_summary(
                [
                    float(row["position_difference_m"])
                    for row in same_frame_rows
                ]
            ),
            "rotation_difference_deg": number_summary(
                [float(row["rotation_difference_deg"]) for row in selected]
            ),
        }

    largest = []
    for row in sorted(rows, key=lambda item: float(item["position_difference_m"]), reverse=True)[:20]:
        task, episode, arm, event_index, requested_frame = row["key"]
        largest.append({
            "first_strategy": first_label,
            "second_strategy": second_label,
            "task": task,
            "episode_id": int(episode),
            "arm": arm,
            "event_index": int(event_index),
            "requested_frame": int(requested_frame),
            "first_resolved_frame": int(row["first_resolved_frame"]),
            "second_resolved_frame": int(row["second_resolved_frame"]),
            "first_candidate_idx": row["first_candidate_idx"],
            "second_candidate_idx": row["second_candidate_idx"],
            "first_position_world_m": row["first_position_world_m"],
            "second_position_world_m": row["second_position_world_m"],
            "position_difference_m": float(row["position_difference_m"]),
            "rotation_difference_deg": float(row["rotation_difference_deg"]),
        })
    return {
        "all": summarize(rows),
        "by_arm": {
            arm: summarize(row for row in rows if row["arm"] == arm) for arm in ARMS
        },
        "largest_position_differences": largest,
    }


def compare_anygrasp_to_oursv2(
    anygrasp: Mapping[Tuple[str, int, str, int, int], Mapping[str, Any]],
    oursv2: Mapping[Tuple[str, int, str, int, int], Mapping[str, Any]],
    *,
    anygrasp_label: str,
) -> Dict[str, Any]:
    """Compare canonical AnyGrasp selection positions with raw OursV2 hand replay."""
    keys = sorted(set(anygrasp) & set(oursv2))
    rows: List[Dict[str, Any]] = []
    for key in keys:
        grasp_record, ours_record = anygrasp[key], oursv2[key]
        grasp_position = position(grasp_record)
        ours_position = position(ours_record)
        delta = grasp_position - ours_position
        rows.append(
            {
                "key": key,
                "arm": key[2],
                "same_resolved_frame": int(grasp_record["resolved_frame"])
                == int(ours_record["resolved_frame"]),
                "anygrasp_resolved_frame": int(grasp_record["resolved_frame"]),
                "oursv2_resolved_frame": int(ours_record["resolved_frame"]),
                "anygrasp_candidate_idx": grasp_record.get("candidate_idx"),
                "anygrasp_position_world_m": grasp_position.tolist(),
                "oursv2_human_replay_position_world_m": ours_position.tolist(),
                "anygrasp_minus_oursv2_world_xyz_m": delta.tolist(),
                "position_difference_m": float(np.linalg.norm(delta)),
            }
        )

    def summarize(subset: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        selected = list(subset)
        same_frame_rows = [row for row in selected if row["same_resolved_frame"]]
        cross_frame_rows = [row for row in selected if not row["same_resolved_frame"]]

        def distances(items: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
            return number_summary(
                [float(row["position_difference_m"]) for row in items]
            )

        def deltas(items: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
            return signed_xyz_summary(
                [row["anygrasp_minus_oursv2_world_xyz_m"] for row in items]
            )

        return {
            "paired_count": len(selected),
            "same_resolved_frame_count": len(same_frame_rows),
            "different_resolved_frame_count": len(cross_frame_rows),
            "position_difference_m": distances(selected),
            "position_difference_same_frame_m": distances(same_frame_rows),
            "position_difference_cross_frame_m": distances(cross_frame_rows),
            "signed_delta_world_xyz_m": deltas(selected),
            "signed_delta_same_frame_world_xyz_m": deltas(same_frame_rows),
            "signed_delta_cross_frame_world_xyz_m": deltas(cross_frame_rows),
        }

    largest = []
    for row in sorted(
        rows,
        key=lambda item: float(item["position_difference_m"]),
        reverse=True,
    )[:20]:
        task, episode, arm, event_index, requested_frame = row["key"]
        largest.append(
            {
                "anygrasp_strategy": anygrasp_label,
                "task": task,
                "episode_id": int(episode),
                "arm": arm,
                "event_index": int(event_index),
                "requested_frame": int(requested_frame),
                "anygrasp_resolved_frame": int(row["anygrasp_resolved_frame"]),
                "oursv2_resolved_frame": int(row["oursv2_resolved_frame"]),
                "same_resolved_frame": bool(row["same_resolved_frame"]),
                "anygrasp_candidate_idx": row["anygrasp_candidate_idx"],
                "anygrasp_position_world_m": row["anygrasp_position_world_m"],
                "oursv2_human_replay_position_world_m": row[
                    "oursv2_human_replay_position_world_m"
                ],
                "anygrasp_minus_oursv2_world_xyz_m": row[
                    "anygrasp_minus_oursv2_world_xyz_m"
                ],
                "position_difference_m": float(row["position_difference_m"]),
            }
        )
    return {
        "definition": (
            "canonical AnyGrasp Selection Pose minus OursV2 direct human-hand "
            "replay Selection Pose, both expressed in world coordinates"
        ),
        "all": summarize(rows),
        "by_arm": {
            arm: summarize(row for row in rows if row["arm"] == arm) for arm in ARMS
        },
        "largest_position_differences": largest,
    }


def fused_contribution_stats(
    records: Mapping[Tuple[str, int, str, int, int], Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = []
    for key, record in records.items():
        metrics = record.get("selection_metrics", {})
        raw_score = float(metrics["anygrasp_score_raw"])
        orientation_score = float(metrics["orientation_score"])
        stored_fused_score = float(metrics["fused_score"])
        raw_contribution = 0.25 * raw_score
        orientation_contribution = 0.75 * orientation_score
        reconstructed = raw_contribution + orientation_contribution
        share = None if reconstructed == 0.0 else orientation_contribution / reconstructed
        rows.append({
            "arm": key[2],
            "raw_score": raw_score,
            "orientation_score": orientation_score,
            "raw_contribution": raw_contribution,
            "orientation_contribution": orientation_contribution,
            "orientation_share": share,
            "fused_score_residual": abs(stored_fused_score - reconstructed),
            "orientation_contribution_greater": orientation_contribution > raw_contribution,
        })

    def summarize(selected: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        values = list(selected)
        return {
            "count": len(values),
            "raw_anygrasp_score": number_summary([float(row["raw_score"]) for row in values]),
            "orientation_score": number_summary([float(row["orientation_score"]) for row in values]),
            "weighted_raw_contribution": number_summary([float(row["raw_contribution"]) for row in values]),
            "weighted_orientation_contribution": number_summary([float(row["orientation_contribution"]) for row in values]),
            "orientation_share_of_fused_score": number_summary([
                float(row["orientation_share"]) for row in values if row["orientation_share"] is not None
            ]),
            "orientation_contribution_greater_count": sum(
                bool(row["orientation_contribution_greater"]) for row in values
            ),
            "max_formula_residual": max(
                (float(row["fused_score_residual"]) for row in values), default=0.0
            ),
        }

    return {
        "all": summarize(rows),
        "by_arm": {
            arm: summarize(row for row in rows if row["arm"] == arm) for arm in ARMS
        },
    }


def finite(value: Any) -> Any:
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(finite(dict(payload)), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def mm(value: float | None) -> str:
    return "n/a" if value is None else f"{1000.0 * value:.3f} mm"


def mean_xyz_mm(summary: Mapping[str, Any]) -> str:
    return "[" + ", ".join(mm(summary[axis]["mean"]) for axis in "xyz") + "]"


def write_markdown_zh(path: Path, report: Mapping[str, Any]) -> None:
    orientation = report["comparisons"]["fused_vs_orientation"]
    top = report["comparisons"]["fused_vs_top_canonical"]
    to_ours = report["anygrasp_vs_oursv2_position"]
    contribution = report["fused_score_contributions"]["all"]
    lines = [
        "# Selection Strategy Audit V4 一致率统计",
        "",
        "左右手分别计数；candidate 一致要求 `resolved_frame` 和 `candidate_idx` 同时一致。",
        "",
        "## Fused vs Orientation",
        "",
        f"- 配对：`{orientation['all']['paired_count']}`",
        f"- 同 candidate：`{orientation['all']['same_candidate_count']}` / `{orientation['all']['paired_count']}` = `{percent(orientation['all']['same_candidate_rate_of_all_pairs'])}`",
        f"- position 平均差：`{mm(orientation['all']['position_difference_m']['mean'])}`",
        f"- 非零 position 平均差：`{mm(orientation['all']['position_difference_nonzero_m']['mean'])}`",
        "",
        "## Fused vs Top canonical",
        "",
        f"- 配对：`{top['all']['paired_count']}`",
        f"- 同 resolved frame：`{top['all']['same_resolved_frame_count']}`",
        f"- 同 candidate：`{top['all']['same_candidate_count']}` / `{top['all']['paired_count']}` = `{percent(top['all']['same_candidate_rate_of_all_pairs'])}`",
        f"- 在同帧配对中同 candidate：`{percent(top['all']['same_candidate_rate_of_same_frame_pairs'])}`",
        f"- position 平均差（全部）：`{mm(top['all']['position_difference_m']['mean'])}`",
        f"- position 平均差（同帧）：`{mm(top['all']['position_difference_same_frame_m']['mean'])}`",
        f"- 最大 position 差样本：`{top['largest_position_differences'][0]['task']}/id{top['largest_position_differences'][0]['episode_id']}`，`{top['largest_position_differences'][0]['arm']}`，`{mm(top['largest_position_differences'][0]['position_difference_m'])}`",
        "",
        "## AnyGrasp 候选位置 vs OursV2 人手直接回放",
        "",
        "这里比较 canonical Selection Pose；差向量定义为 `AnyGrasp world xyz - OursV2 human-hand replay world xyz`。不比较 retreat、pre-grasp 或 TCP 补偿后的控制点。",
        "",
        "| AnyGrasp 策略 | 配对 | 同帧 | 距离均值 | 中位数 | P95 | 同帧距离均值 | 平均有符号 ΔXYZ |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        *[
            "| {label} | {pairs} | {same} | {mean} | {median} | {p95} | {same_mean} | {xyz} |".format(
                label=label,
                pairs=to_ours[key]["all"]["paired_count"],
                same=to_ours[key]["all"]["same_resolved_frame_count"],
                mean=mm(to_ours[key]["all"]["position_difference_m"]["mean"]),
                median=mm(to_ours[key]["all"]["position_difference_m"]["median"]),
                p95=mm(to_ours[key]["all"]["position_difference_m"]["p95"]),
                same_mean=mm(
                    to_ours[key]["all"]["position_difference_same_frame_m"]["mean"]
                ),
                xyz=mean_xyz_mm(to_ours[key]["all"]["signed_delta_world_xyz_m"]),
            )
            for key, label in (
                ("orientation", "Orientation"),
                ("fused", "Fused"),
                ("top_canonical", "Top canonical"),
            )
        ],
        "",
        "Top canonical 如果发生历史候选最近帧回退，`全部`统计会同时包含人手/物体随时间运动；判断纯候选空间偏差时优先看同帧统计。左右手明细、跨帧统计和最大差异样本见 JSON。",
        "",
        "## 权重解释",
        "",
        "历史六任务 Fused 使用 `0.25 * raw AnyGrasp score + 0.75 * orientation score`，并先执行 `theta <= 90 deg` 硬过滤。是否偏向 rotation 应同时看固定权重和 Fused 与 Orientation/Top 的实际一致率。",
        f"本批次 Fused 候选的平均加权 raw-score contribution 为 `{contribution['weighted_raw_contribution']['mean']:.6f}`，平均 orientation contribution 为 `{contribution['weighted_orientation_contribution']['mean']:.6f}`；orientation 在最终 Fused score 中的平均占比为 `{percent(contribution['orientation_share_of_fused_score']['mean'])}`。",
        "",
        "左右手明细和 min/median/p95/max 见同目录 JSON。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_en(path: Path, report: Mapping[str, Any]) -> None:
    orientation = report["comparisons"]["fused_vs_orientation"]
    top = report["comparisons"]["fused_vs_top_canonical"]
    to_ours = report["anygrasp_vs_oursv2_position"]
    contribution = report["fused_score_contributions"]["all"]
    lines = [
        "# Selection Strategy Audit V4 Agreement Statistics",
        "",
        "Left and right arms count separately. Candidate equality requires both `resolved_frame` and `candidate_idx` to match.",
        "",
        "## Fused vs Orientation",
        "",
        f"- Pairs: `{orientation['all']['paired_count']}`",
        f"- Same candidate: `{orientation['all']['same_candidate_count']}` / `{orientation['all']['paired_count']}` = `{percent(orientation['all']['same_candidate_rate_of_all_pairs'])}`",
        f"- Mean position difference: `{mm(orientation['all']['position_difference_m']['mean'])}`",
        f"- Mean nonzero position difference: `{mm(orientation['all']['position_difference_nonzero_m']['mean'])}`",
        "",
        "## Fused vs Top canonical",
        "",
        f"- Pairs: `{top['all']['paired_count']}`",
        f"- Same resolved frame: `{top['all']['same_resolved_frame_count']}`",
        f"- Same candidate: `{top['all']['same_candidate_count']}` / `{top['all']['paired_count']}` = `{percent(top['all']['same_candidate_rate_of_all_pairs'])}`",
        f"- Same candidate among same-frame pairs: `{percent(top['all']['same_candidate_rate_of_same_frame_pairs'])}`",
        f"- Mean position difference (all): `{mm(top['all']['position_difference_m']['mean'])}`",
        f"- Mean position difference (same frame): `{mm(top['all']['position_difference_same_frame_m']['mean'])}`",
        f"- Largest position-gap sample: `{top['largest_position_differences'][0]['task']}/id{top['largest_position_differences'][0]['episode_id']}`, `{top['largest_position_differences'][0]['arm']}`, `{mm(top['largest_position_differences'][0]['position_difference_m'])}`",
        "",
        "## AnyGrasp candidate position vs OursV2 direct hand replay",
        "",
        "This compares canonical Selection Poses. The signed delta is `AnyGrasp world xyz - OursV2 human-hand replay world xyz`; retreat, pre-grasp, and TCP-adjusted control points are excluded.",
        "",
        "| AnyGrasp strategy | Pairs | Same frame | Mean distance | Median | P95 | Same-frame mean | Mean signed ΔXYZ |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        *[
            "| {label} | {pairs} | {same} | {mean} | {median} | {p95} | {same_mean} | {xyz} |".format(
                label=label,
                pairs=to_ours[key]["all"]["paired_count"],
                same=to_ours[key]["all"]["same_resolved_frame_count"],
                mean=mm(to_ours[key]["all"]["position_difference_m"]["mean"]),
                median=mm(to_ours[key]["all"]["position_difference_m"]["median"]),
                p95=mm(to_ours[key]["all"]["position_difference_m"]["p95"]),
                same_mean=mm(
                    to_ours[key]["all"]["position_difference_same_frame_m"]["mean"]
                ),
                xyz=mean_xyz_mm(to_ours[key]["all"]["signed_delta_world_xyz_m"]),
            )
            for key, label in (
                ("orientation", "Orientation"),
                ("fused", "Fused"),
                ("top_canonical", "Top canonical"),
            )
        ],
        "",
        "When Top canonical uses a nearest-frame fallback, the all-pairs statistic also includes hand/object motion over time. Prefer the same-frame statistic when interpreting the pure spatial candidate offset. See the JSON for arm-specific, cross-frame, and largest-gap results.",
        "",
        "## Weight interpretation",
        "",
        "The historical six-task Fused score is `0.25 * raw AnyGrasp score + 0.75 * orientation score`, after the independent `theta <= 90 deg` hard filter. Assess rotation bias using both these fixed weights and empirical agreement with Orientation/Top.",
        f"For selected Fused candidates, the mean weighted raw-score contribution is `{contribution['weighted_raw_contribution']['mean']:.6f}` and the mean orientation contribution is `{contribution['weighted_orientation_contribution']['mean']:.6f}`. Orientation accounts for `{percent(contribution['orientation_share_of_fused_score']['mean'])}` of the final Fused score on average.",
        "",
        "See the JSON for left/right breakdowns and min/median/p95/max statistics.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    paths = metadata_paths(args.audit_root)
    if not paths:
        raise FileNotFoundError(f"No V4 keyframe metadata under {args.audit_root}")

    records: Dict[str, Dict[Tuple[str, int, str, int, int], Mapping[str, Any]]] = {
        strategy: {} for strategy in STRATEGIES
    }
    task_counter = Counter()
    duplicate_keys: List[Dict[str, Any]] = []
    for path in paths:
        metadata = load_json(path)
        task_counter[str(metadata["task"])] += 1
        for record in metadata.get("records", []):
            strategy = str(record.get("strategy"))
            if strategy not in records:
                continue
            key = record_key(record)
            if key in records[strategy]:
                duplicate_keys.append(
                    {"strategy": strategy, "key": list(key), "metadata": str(path)}
                )
                continue
            records[strategy][key] = record
    if duplicate_keys:
        raise ValueError(f"Duplicate strategy records: {duplicate_keys[:5]}")

    comparisons = {
        "fused_vs_orientation": compare_pair(
            records["fused"], records["orientation"],
            position_tol_m=args.position_equal_tol_m,
            rotation_tol_deg=args.rotation_equal_tol_deg,
            first_label="fused",
            second_label="orientation",
        ),
        "fused_vs_top_canonical": compare_pair(
            records["fused"], records["top_score"],
            position_tol_m=args.position_equal_tol_m,
            rotation_tol_deg=args.rotation_equal_tol_deg,
            first_label="fused",
            second_label="top_canonical",
        ),
    }
    anygrasp_vs_oursv2_position = {
        "orientation": compare_anygrasp_to_oursv2(
            records["orientation"],
            records["oursv2"],
            anygrasp_label="orientation",
        ),
        "fused": compare_anygrasp_to_oursv2(
            records["fused"],
            records["oursv2"],
            anygrasp_label="fused",
        ),
        "top_canonical": compare_anygrasp_to_oursv2(
            records["top_score"],
            records["oursv2"],
            anygrasp_label="top_canonical",
        ),
    }
    report = {
        "schema": "selection_strategy_audit_v4.agreement.v2",
        "audit_root": str(args.audit_root),
        "counting_unit": "one arm-event; left and right count separately",
        "candidate_equality": "same resolved_frame and same candidate_idx",
        "pose_source": "canonical Selection Pose in world coordinates",
        "fused_formula": {
            "hard_filter": "theta_deg <= 90",
            "orientation_score": "clip(1 - theta_deg / 180, 0, 1)",
            "score": "0.25 * raw_anygrasp_score + 0.75 * orientation_score",
            "raw_anygrasp_score_normalized": False,
        },
        "coverage": {
            "metadata_files": len(paths),
            "metadata_files_by_task": dict(sorted(task_counter.items())),
            "strategy_records": {
                strategy: len(values) for strategy, values in records.items()
            },
        },
        "tolerances": {
            "position_equal_m": float(args.position_equal_tol_m),
            "rotation_equal_deg": float(args.rotation_equal_tol_deg),
        },
        "comparisons": comparisons,
        "anygrasp_vs_oursv2_position": anygrasp_vs_oursv2_position,
        "fused_score_contributions": fused_contribution_stats(records["fused"]),
    }
    output_json = args.output_json or args.audit_root / "strategy_agreement_stats.json"
    output_zh = args.output_zh_md or args.audit_root / "strategy_agreement_stats.zh.md"
    output_en = args.output_en_md or args.audit_root / "strategy_agreement_stats.en.md"
    write_json(output_json, report)
    write_markdown_zh(output_zh, report)
    write_markdown_en(output_en, report)
    return report


def main() -> None:
    args = parse_args()
    report = run(args)
    orientation = report["comparisons"]["fused_vs_orientation"]["all"]
    top = report["comparisons"]["fused_vs_top_canonical"]["all"]
    print(
        json.dumps(
            {
                "coverage": report["coverage"],
                "fused_score_contributions": report["fused_score_contributions"]["all"],
                "fused_vs_orientation": orientation,
                "fused_vs_top_canonical": top,
                "anygrasp_vs_oursv2_position": {
                    key: value["all"]
                    for key, value in report[
                        "anygrasp_vs_oursv2_position"
                    ].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
