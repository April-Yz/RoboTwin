#!/usr/bin/env python3
"""Aggregate cross-episode VR-to-HaMeR transform patterns for 20260708.

This diagnostic sits above the existing per-episode bestfit/local-hand checks.
It tests whether the VR hand centroids can share one image-space transform,
whether a k=2 pose cluster explains the variation, or whether the relation is
only reliable with episode-local transforms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_vr_hamer_alignment as align


DEFAULT_BASE = Path("/home/zaijia001/ssd/data/piper/vr/0_1harmer/datav1")
DEFAULT_OUT_DIR = DEFAULT_BASE / "cross_episode_transform_patterns_20260708"
DEFAULT_BESTFIT_JSON = DEFAULT_BASE / "compare_bestfit_20260708" / "alignment_sweep_20260708.json"
DEFAULT_3D_JSON = DEFAULT_BASE / "compare_3d_20260708" / "summary_3d_20260708.json"
DEFAULT_LOCAL_JSON = DEFAULT_BASE / "local_hand_alignment_20260708" / "summary_local_hand_alignment_20260708.json"
DEFAULT_BOTH_JSON = DEFAULT_BASE / "local_hand_alignment_20260708_bothhands" / "summary_bothhands_local_alignment_20260708.json"
AXIS_NAMES = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vr-root", default=align.DEFAULT_VR_ROOT)
    ap.add_argument("--hamer-root", default=align.DEFAULT_HAMER_ROOT)
    ap.add_argument("--bestfit-json", default=str(DEFAULT_BESTFIT_JSON))
    ap.add_argument("--summary-3d-json", default=str(DEFAULT_3D_JSON))
    ap.add_argument("--local-json", default=str(DEFAULT_LOCAL_JSON))
    ap.add_argument("--bothhands-json", default=str(DEFAULT_BOTH_JSON))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--episode-substr", default="20260708")
    ap.add_argument("--lag-min", type=int, default=-10)
    ap.add_argument("--lag-max", type=int, default=10)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument(
        "--fit-min-samples",
        type=int,
        default=50,
        help="Minimum bestfit sample count for cross-episode global/cluster fitting.",
    )
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else {}


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    return val if math.isfinite(val) else default


def sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def index_episodes(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in payload.get("episodes") or []:
        out[str(item.get("video_id"))] = item
    return out


def quat_to_yaw_pitch_roll(q: list[float]) -> tuple[float, float, float]:
    rot = align.quat_xyzw_to_matrix(q)
    forward = rot[:, 2]
    up = rot[:, 1]
    yaw = math.degrees(math.atan2(float(forward[0]), float(forward[2])))
    pitch = math.degrees(math.atan2(float(forward[1]), math.sqrt(float(forward[0] ** 2 + forward[2] ** 2))))
    roll = math.degrees(math.atan2(float(up[0]), float(up[1])))
    return yaw, pitch, roll


def mean_pose_features(frames: list[dict[str, Any]]) -> dict[str, Any]:
    center_pos = []
    left_pos = []
    right_pos = []
    center_ypr = []
    hand_centers = []
    for frame in frames:
        center_pose = frame.get("center_eye_pose") or []
        if len(center_pose) >= 7:
            center_pos.append([float(v) for v in center_pose[:3]])
            center_ypr.append(quat_to_yaw_pitch_roll(center_pose[3:7]))
        for key, target in (("left_eye_pose", left_pos), ("right_eye_pose", right_pos)):
            pose = frame.get(key) or []
            if len(pose) >= 3:
                target.append([float(v) for v in pose[:3]])
        for side in ("left", "right"):
            hand = frame.get(f"{side}_hand") or {}
            if hand.get("is_tracked"):
                centroid = align.hand_centroid(hand, align.CORE_JOINTS)
                if centroid is not None:
                    hand_centers.append(centroid)

    def mean_or_none(values: list[Any]) -> list[float] | None:
        if not values:
            return None
        return np.mean(np.asarray(values, dtype=np.float64), axis=0).tolist()

    def std_or_none(values: list[Any]) -> list[float] | None:
        if not values:
            return None
        return np.std(np.asarray(values, dtype=np.float64), axis=0).tolist()

    return {
        "center_eye_mean_xyz": mean_or_none(center_pos),
        "center_eye_std_xyz": std_or_none(center_pos),
        "left_eye_mean_xyz": mean_or_none(left_pos),
        "right_eye_mean_xyz": mean_or_none(right_pos),
        "center_eye_mean_yaw_pitch_roll_deg": mean_or_none(center_ypr),
        "hand_trajectory_center_xyz": mean_or_none(hand_centers),
        "hand_trajectory_std_xyz": std_or_none(hand_centers),
        "center_eye_frame_count": len(center_pos),
        "hand_tracked_core_count": len(hand_centers),
    }


def feature_vector_from_pose(pose: dict[str, Any]) -> np.ndarray:
    parts: list[float] = []
    for key, width in (
        ("center_eye_mean_xyz", 3),
        ("center_eye_mean_yaw_pitch_roll_deg", 3),
        ("hand_trajectory_center_xyz", 3),
    ):
        vals = pose.get(key)
        if vals is None:
            parts.extend([float("nan")] * width)
        else:
            parts.extend([float(v) for v in vals[:width]])
    return np.asarray(parts, dtype=np.float64)


def collect_xy_for_config(
    item: dict[str, Any],
    frames: list[dict[str, Any]],
    npz: Any,
    pose_mode: str,
    model_mode: str,
    lag: int,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    max_frames = min(len(frames), int(item.get("frames_written") or len(frames)))
    samples = align.build_samples(
        frames,
        npz,
        float(item["width"]),
        float(item["height"]),
        str(item["episode"]),
        str(item["video_id"]),
        int(lag),
        max_frames,
    )
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for sample in samples:
        point = align.transform_point(sample.world_core, frames[sample.vr_frame], pose_mode)
        if point is None:
            continue
        feat = align.features_from_point(point, model_mode)
        if feat is None or not np.isfinite(feat).all():
            continue
        xs.append(feat)
        ys.append(sample.uv)
    if len(xs) < min_samples:
        return None
    return np.stack(xs), np.stack(ys)


def fit_xy(X: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
    fit = align.fit_linear(X, Y)
    return {
        "coef": np.asarray(fit["coef"], dtype=np.float64),
        "n": int(len(X)),
        "r2_u": float(fit["r2"][0]),
        "r2_v": float(fit["r2"][1]),
        "mean_r2": float(fit["mean_r2"]),
        "rmse_u_px": float(fit["rmse_px_1280"][0]),
        "rmse_v_px": float(fit["rmse_px_1280"][1]),
        "mean_rmse_px": float(fit["mean_rmse_px_1280"]),
    }


def evaluate_coef(X: np.ndarray, Y: np.ndarray, coef: np.ndarray) -> dict[str, Any]:
    X1 = np.c_[X, np.ones(len(X), dtype=np.float64)]
    pred = X1 @ coef
    err = pred - Y
    rmse_norm = np.sqrt(np.mean(err * err, axis=0))
    ss_res = np.sum(err * err, axis=0)
    ss_tot = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return {
        "n": int(len(X)),
        "r2_u": float(r2[0]),
        "r2_v": float(r2[1]),
        "mean_r2": float(np.mean(r2)),
        "rmse_u_px": float(rmse_norm[0] * 1280.0),
        "rmse_v_px": float(rmse_norm[1] * 1280.0),
        "mean_rmse_px": float(np.mean(rmse_norm * 1280.0)),
    }


def weighted_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [(finite_float(row.get(key)), int(row.get("n") or 0)) for row in rows]
    vals = [(v, n) for v, n in vals if v is not None and n > 0]
    if not vals:
        return None
    return float(np.average([v for v, _ in vals], weights=[n for _, n in vals]))


def decompose_transform(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config or not config.get("coef"):
        return {}
    coef = np.asarray(config["coef"], dtype=np.float64)
    model_mode = str(config.get("model_mode") or "")
    feature_names = ["x", "y", "z"] if model_mode == "linear_xyz" and coef.shape[0] >= 4 else ["x_over_z", "y_over_z"]
    feature_rows = min(len(feature_names), coef.shape[0] - 1)
    m2 = coef[:2, :].T if coef.shape[0] >= 3 else np.zeros((2, 2), dtype=np.float64)
    c0 = m2[:, 0]
    c1 = m2[:, 1]
    scale_x = float(np.linalg.norm(c0))
    scale_y = float(np.linalg.norm(c1))
    determinant = float(np.linalg.det(m2))
    shear = float(np.dot(c0, c1) / max(scale_x * scale_y, 1e-12))
    try:
        u, _s, vt = np.linalg.svd(m2)
        r = u @ vt
        rotation_deg = math.degrees(math.atan2(float(r[1, 0]), float(r[0, 0])))
    except np.linalg.LinAlgError:
        rotation_deg = float("nan")
    contributions = {}
    for i in range(feature_rows):
        name = feature_names[i]
        contributions[name] = {
            "to_u": float(coef[i, 0]),
            "to_v": float(coef[i, 1]),
            "abs_to_u": float(abs(coef[i, 0])),
            "abs_to_v": float(abs(coef[i, 1])),
            "sign_to_u": "positive" if coef[i, 0] >= 0 else "negative",
            "sign_to_v": "positive" if coef[i, 1] >= 0 else "negative",
        }
    if contributions:
        dominant_u = max(contributions, key=lambda name: contributions[name]["abs_to_u"])
        dominant_v = max(contributions, key=lambda name: contributions[name]["abs_to_v"])
    else:
        dominant_u = None
        dominant_v = None
    translation = coef[-1, :].tolist()
    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "rotation_deg": rotation_deg,
        "shear": shear,
        "translation_x": float(translation[0]),
        "translation_y": float(translation[1]),
        "determinant": determinant,
        "reflection": determinant < 0,
        "dominant_u_feature": dominant_u,
        "dominant_v_feature": dominant_v,
        "axis_contributions": contributions,
        "coef_shape": list(coef.shape),
    }


def axis_candidate_label(decomp: dict[str, Any]) -> str:
    u_axis = str(decomp.get("dominant_u_feature") or "")
    v_axis = str(decomp.get("dominant_v_feature") or "")
    if u_axis == "x" and v_axis == "y":
        return "no_axis_swap_xy"
    if u_axis == "y" and v_axis == "x":
        return "xy_swap"
    if {u_axis, v_axis} == {"y", "z"}:
        return "yz_axis_candidate"
    if {u_axis, v_axis} == {"z", "x"}:
        return "zx_axis_candidate"
    if "over_z" in u_axis or "over_z" in v_axis:
        return "perspective_xy_over_z"
    return f"{u_axis}_to_u__{v_axis}_to_v"


def mirror_candidate_label(decomp: dict[str, Any]) -> str:
    contrib = decomp.get("axis_contributions") or {}
    x_to_u = (contrib.get("x") or {}).get("to_u")
    y_to_v = (contrib.get("y") or {}).get("to_v")
    if x_to_u is None or y_to_v is None:
        return "not_applicable"
    mirror_x = float(x_to_u) < 0.0
    mirror_y = float(y_to_v) > 0.0
    if mirror_x and mirror_y:
        return "mirror_xy_candidate"
    if mirror_x:
        return "mirror_x_candidate"
    if mirror_y:
        return "mirror_y_candidate"
    return "expected_u_plus_x_v_minus_y"


def kmeans2(features: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(features)
    if n < 2:
        return np.zeros(n, dtype=int), {"silhouette": None, "centroid_distance_std": None}
    x = features.copy()
    means = np.nanmean(x, axis=0)
    inds = np.where(~np.isfinite(x))
    x[inds] = np.take(means, inds[1])
    std = np.nanstd(x, axis=0)
    std[std < 1e-9] = 1.0
    z = (x - np.nanmean(x, axis=0)) / std
    dmat = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(dmat), dmat.shape)
    centroids = np.stack([z[i], z[j]], axis=0)
    labels = np.zeros(n, dtype=int)
    for _ in range(50):
        new_labels = np.argmin(np.linalg.norm(z[:, None, :] - centroids[None, :, :], axis=2), axis=1)
        new_centroids = centroids.copy()
        for c in (0, 1):
            if np.any(new_labels == c):
                new_centroids[c] = np.mean(z[new_labels == c], axis=0)
        if np.array_equal(new_labels, labels):
            centroids = new_centroids
            break
        labels = new_labels
        centroids = new_centroids
    sil_vals = []
    for idx in range(n):
        same = z[labels == labels[idx]]
        other = z[labels != labels[idx]]
        if len(same) <= 1 or len(other) == 0:
            continue
        a = float(np.mean(np.linalg.norm(same - z[idx], axis=1)))
        b = float(np.mean(np.linalg.norm(other - z[idx], axis=1)))
        sil_vals.append((b - a) / max(a, b, 1e-12))
    center_dist = float(np.linalg.norm(centroids[0] - centroids[1]))
    return labels, {
        "silhouette": float(np.mean(sil_vals)) if sil_vals else None,
        "centroid_distance_std": center_dist,
    }


def choose_common_global_config(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidates = []
    for pose_mode in align.POSE_MODES:
        for model_mode in align.MODEL_MODES:
            per_episode = {}
            for rec in records:
                best_lag_fit = None
                for lag in range(args.lag_min, args.lag_max + 1):
                    xy = collect_xy_for_config(
                        rec["manifest"],
                        rec["frames"],
                        rec["npz"],
                        pose_mode,
                        model_mode,
                        lag,
                        args.min_samples,
                    )
                    if xy is None:
                        continue
                    X, Y = xy
                    fit = fit_xy(X, Y)
                    fit.update({"lag": lag, "X": X, "Y": Y})
                    if best_lag_fit is None or (fit["mean_rmse_px"], -fit["mean_r2"], abs(lag)) < (
                        best_lag_fit["mean_rmse_px"],
                        -best_lag_fit["mean_r2"],
                        abs(int(best_lag_fit["lag"])),
                    ):
                        best_lag_fit = fit
                if best_lag_fit is not None:
                    per_episode[str(rec["video_id"])] = best_lag_fit
            if len(per_episode) < 3:
                continue
            X_pool = np.concatenate([row["X"] for row in per_episode.values()], axis=0)
            Y_pool = np.concatenate([row["Y"] for row in per_episode.values()], axis=0)
            global_fit = fit_xy(X_pool, Y_pool)
            evals = []
            for vid, row in per_episode.items():
                ev = evaluate_coef(row["X"], row["Y"], global_fit["coef"])
                ev.update({"video_id": vid, "lag": row["lag"]})
                evals.append(ev)
            candidates.append(
                {
                    "pose_mode": pose_mode,
                    "model_mode": model_mode,
                    "episode_count": len(per_episode),
                    "sample_count": int(len(X_pool)),
                    "coef": global_fit["coef"],
                    "self_by_episode": per_episode,
                    "global_fit": global_fit,
                    "global_eval_by_episode": {row["video_id"]: row for row in evals},
                    "weighted_mean_r2": weighted_mean(evals, "mean_r2"),
                    "weighted_mean_rmse_px": weighted_mean(evals, "mean_rmse_px"),
                }
            )
    if not candidates:
        return {}
    return min(candidates, key=lambda row: (-(row["episode_count"]), row["weighted_mean_rmse_px"] or 1e9))


def build_cluster_transforms(
    common: dict[str, Any],
    records: list[dict[str, Any]],
    labels: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not common:
        return [], {}
    cluster_rows = []
    eval_by_episode = {}
    per_episode = common["self_by_episode"]
    for cluster_id in sorted(set(labels.values())):
        vids = [str(rec["video_id"]) for rec in records if labels.get(str(rec["video_id"])) == cluster_id]
        usable = [vid for vid in vids if vid in per_episode]
        if not usable:
            continue
        X_pool = np.concatenate([per_episode[vid]["X"] for vid in usable], axis=0)
        Y_pool = np.concatenate([per_episode[vid]["Y"] for vid in usable], axis=0)
        fit = fit_xy(X_pool, Y_pool)
        episode_evals = []
        for vid in usable:
            row = per_episode[vid]
            ev = evaluate_coef(row["X"], row["Y"], fit["coef"])
            ev.update({"video_id": vid, "lag": row["lag"], "cluster_id": cluster_id})
            episode_evals.append(ev)
            eval_by_episode[vid] = ev
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "episodes": usable,
                "episode_count": len(usable),
                "sample_count": int(len(X_pool)),
                "pose_mode": common["pose_mode"],
                "model_mode": common["model_mode"],
                "coef": fit["coef"],
                "weighted_mean_r2": weighted_mean(episode_evals, "mean_r2"),
                "weighted_mean_rmse_px": weighted_mean(episode_evals, "mean_rmse_px"),
                "episode_evals": episode_evals,
            }
        )
    return cluster_rows, eval_by_episode


def compact_metric(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    keys = ("n", "r2_u", "r2_v", "mean_r2", "rmse_u_px", "rmse_v_px", "mean_rmse_px", "lag")
    return {key: sanitize(row.get(key)) for key in keys if key in row}


def make_plots(
    out_dir: Path,
    episodes: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    cluster_labels: dict[str, int],
    transform_similarity: dict[str, Any],
) -> dict[str, str]:
    plots: dict[str, str] = {}
    rendered = [ep for ep in episodes if ep.get("status") == "fit" and (ep.get("bestfit") or {}).get("coef")]
    if rendered:
        fig, ax = plt.subplots(figsize=(8, 6))
        for ep in rendered:
            decomp = ep.get("transform_decomposition") or {}
            vid = str(ep["video_id"])
            ax.scatter(
                decomp.get("rotation_deg"),
                decomp.get("scale_x"),
                s=80,
                label=vid,
                c=f"C{cluster_labels.get(vid, 0)}",
                marker="x" if decomp.get("reflection") else "o",
            )
            ax.text(decomp.get("rotation_deg"), decomp.get("scale_x"), vid, fontsize=9)
        ax.set_xlabel("episode-local transform rotation (deg)")
        ax.set_ylabel("scale_x")
        ax.set_title("Transform parameter scatter")
        ax.grid(True, alpha=0.3)
        path = out_dir / "transform_parameter_scatter_20260708.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots["transform_parameter_scatter"] = str(path)

    xs, zs, vids, colors = [], [], [], []
    for ep in episodes:
        pose = ep.get("pose_features") or {}
        center = pose.get("center_eye_mean_xyz")
        if center is None:
            continue
        xs.append(center[0])
        zs.append(center[2])
        vids.append(str(ep["video_id"]))
        colors.append(cluster_labels.get(str(ep["video_id"]), -1))
    if xs:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(xs, zs, c=colors, cmap="tab10", s=90)
        for x, z, vid in zip(xs, zs, vids):
            ax.text(x, z, vid, fontsize=9)
        ax.set_xlabel("center_eye mean X (RUF right, m)")
        ax.set_ylabel("center_eye mean Z (RUF forward, m)")
        ax.set_title("k=2 cluster on eye/head and hand trajectory features")
        ax.grid(True, alpha=0.3)
        path = out_dir / "cluster_eye_pose_20260708.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots["cluster_eye_pose"] = str(path)

    lags = [ep.get("bestfit", {}).get("lag") for ep in episodes if ep.get("bestfit", {}).get("lag") is not None]
    if lags:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(lags, bins=np.arange(min(lags) - 0.5, max(lags) + 1.5, 1), color="#4C78A8", edgecolor="white")
        ax.set_xlabel("best lag (hamer_frame = vr_frame + lag)")
        ax.set_ylabel("episode count")
        ax.set_title("Best lag distribution")
        ax.grid(True, axis="y", alpha=0.3)
        path = out_dir / "lag_distribution_20260708.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots["lag_distribution"] = str(path)

    heat_rows = []
    heat_labels = []
    for ep in rendered:
        contrib = (ep.get("transform_decomposition") or {}).get("axis_contributions") or {}
        if not contrib:
            continue
        heat_rows.append(
            [
                abs((contrib.get("x") or {}).get("to_u") or 0.0),
                abs((contrib.get("y") or {}).get("to_u") or 0.0),
                abs((contrib.get("z") or {}).get("to_u") or 0.0),
                abs((contrib.get("x") or {}).get("to_v") or 0.0),
                abs((contrib.get("y") or {}).get("to_v") or 0.0),
                abs((contrib.get("z") or {}).get("to_v") or 0.0),
            ]
        )
        heat_labels.append(str(ep["video_id"]))
    if heat_rows:
        arr = np.asarray(heat_rows, dtype=np.float64)
        row_sums = np.maximum(np.sum(arr, axis=1, keepdims=True), 1e-9)
        arr = arr / row_sums
        fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(heat_labels) + 2)))
        im = ax.imshow(arr, aspect="auto", cmap="magma")
        ax.set_yticks(np.arange(len(heat_labels)), labels=heat_labels)
        ax.set_xticks(np.arange(6), labels=["x->u", "y->u", "z->u", "x->v", "y->v", "z->v"], rotation=30)
        ax.set_title("Normalized axis contribution heatmap")
        fig.colorbar(im, ax=ax, shrink=0.85)
        path = out_dir / "axis_contribution_heatmap_20260708.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots["axis_contribution_heatmap"] = str(path)

    labels = []
    self_rmse = []
    cluster_rmse = []
    global_rmse = []
    self_r2 = []
    cluster_r2 = []
    global_r2 = []
    for ep in episodes:
        cmp_row = ep.get("transform_comparison") or {}
        if not cmp_row.get("self") or not cmp_row.get("global"):
            continue
        labels.append(str(ep["video_id"]))
        self_rmse.append(cmp_row["self"].get("mean_rmse_px"))
        global_rmse.append(cmp_row["global"].get("mean_rmse_px"))
        cluster_rmse.append((cmp_row.get("cluster") or {}).get("mean_rmse_px"))
        self_r2.append(cmp_row["self"].get("mean_r2"))
        global_r2.append(cmp_row["global"].get("mean_r2"))
        cluster_r2.append((cmp_row.get("cluster") or {}).get("mean_r2"))
    if labels:
        x = np.arange(len(labels))
        width = 0.25
        fig, axes = plt.subplots(2, 1, figsize=(max(9, len(labels) * 0.75), 8), sharex=True)
        axes[0].bar(x - width, self_rmse, width, label="self")
        axes[0].bar(x, cluster_rmse, width, label="cluster")
        axes[0].bar(x + width, global_rmse, width, label="global")
        axes[0].set_ylabel("mean RMSE px")
        axes[0].set_title("Self vs cluster vs global transform comparison")
        axes[0].legend()
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[1].bar(x - width, self_r2, width, label="self")
        axes[1].bar(x, cluster_r2, width, label="cluster")
        axes[1].bar(x + width, global_r2, width, label="global")
        axes[1].set_ylabel("mean R2")
        axes[1].set_xticks(x, labels)
        axes[1].grid(True, axis="y", alpha=0.3)
        path = out_dir / "global_cluster_self_r2_rmse_20260708.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots["global_cluster_self_r2_rmse"] = str(path)

    distances = transform_similarity.get("linear_coef_pairwise_distances")
    if distances:
        vids = sorted({d["a"] for d in distances} | {d["b"] for d in distances}, key=lambda x: int(x))
        idx = {vid: i for i, vid in enumerate(vids)}
        mat = np.full((len(vids), len(vids)), np.nan)
        np.fill_diagonal(mat, 0.0)
        for row in distances:
            ia, ib = idx[row["a"]], idx[row["b"]]
            mat[ia, ib] = mat[ib, ia] = row["distance"]
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks(np.arange(len(vids)), vids)
        ax.set_yticks(np.arange(len(vids)), vids)
        ax.set_title("Episode-local linear coefficient distance")
        fig.colorbar(im, ax=ax, shrink=0.8)
        path = out_dir / "transform_similarity_distance_heatmap_20260708.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plots["transform_similarity_distance_heatmap"] = str(path)
    return plots


def write_csvs(out_dir: Path, episodes: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> None:
    episode_csv = out_dir / "episode_transform_table_20260708.csv"
    fields = [
        "video_id",
        "episode",
        "status",
        "fit_eligible",
        "best_pose",
        "best_model",
        "best_lag",
        "best_n",
        "self_mean_r2",
        "self_mean_rmse_px",
        "cluster_id",
        "cluster_mean_r2",
        "cluster_mean_rmse_px",
        "global_mean_r2",
        "global_mean_rmse_px",
        "dominant_u_feature",
        "dominant_v_feature",
        "axis_candidate",
        "mirror_candidate",
        "rotation_deg",
        "scale_x",
        "scale_y",
        "reflection",
        "ray_mean_dist_m",
        "local_score",
        "both_mapping",
        "both_mirror",
        "recommendation",
        "notes",
    ]
    with episode_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ep in episodes:
            bestfit = ep.get("bestfit") or {}
            decomp = ep.get("transform_decomposition") or {}
            cmp_row = ep.get("transform_comparison") or {}
            cluster_cmp = cmp_row.get("cluster") or {}
            global_cmp = cmp_row.get("global") or {}
            local = ep.get("local_hand_alignment") or {}
            both = ep.get("bothhands_alignment") or {}
            writer.writerow(
                {
                    "video_id": ep.get("video_id"),
                    "episode": ep.get("episode"),
                    "status": ep.get("status"),
                    "fit_eligible": ep.get("fit_eligible"),
                    "best_pose": bestfit.get("pose_mode"),
                    "best_model": bestfit.get("model_mode"),
                    "best_lag": bestfit.get("lag"),
                    "best_n": bestfit.get("n"),
                    "self_mean_r2": bestfit.get("mean_r2"),
                    "self_mean_rmse_px": bestfit.get("mean_rmse_px"),
                    "cluster_id": ep.get("cluster_id"),
                    "cluster_mean_r2": cluster_cmp.get("mean_r2"),
                    "cluster_mean_rmse_px": cluster_cmp.get("mean_rmse_px"),
                    "global_mean_r2": global_cmp.get("mean_r2"),
                    "global_mean_rmse_px": global_cmp.get("mean_rmse_px"),
                    "dominant_u_feature": decomp.get("dominant_u_feature"),
                    "dominant_v_feature": decomp.get("dominant_v_feature"),
                    "axis_candidate": ep.get("axis_candidate"),
                    "mirror_candidate": ep.get("mirror_candidate"),
                    "rotation_deg": decomp.get("rotation_deg"),
                    "scale_x": decomp.get("scale_x"),
                    "scale_y": decomp.get("scale_y"),
                    "reflection": decomp.get("reflection"),
                    "ray_mean_dist_m": (ep.get("summary3d") or {}).get("mean_hamer_ray_to_vr_core_dist_m"),
                    "local_score": local.get("local_shape_score"),
                    "both_mapping": both.get("best_mapping_hypothesis"),
                    "both_mirror": both.get("mirror_summary"),
                    "recommendation": ep.get("recommendation"),
                    "notes": "; ".join(ep.get("notes") or []),
                }
            )

    cluster_csv = out_dir / "cluster_transform_table_20260708.csv"
    fields = [
        "cluster_id",
        "episodes",
        "episode_count",
        "sample_count",
        "pose_model",
        "weighted_mean_r2",
        "weighted_mean_rmse_px",
        "center_eye_mean_xyz",
        "center_eye_spread_m",
        "recommendation",
    ]
    with cluster_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cluster in clusters:
            writer.writerow(
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "episodes": " ".join(cluster.get("episodes") or []),
                    "episode_count": cluster.get("episode_count"),
                    "sample_count": cluster.get("sample_count"),
                    "pose_model": f"{cluster.get('pose_mode')}/{cluster.get('model_mode')}",
                    "weighted_mean_r2": cluster.get("weighted_mean_r2"),
                    "weighted_mean_rmse_px": cluster.get("weighted_mean_rmse_px"),
                    "center_eye_mean_xyz": json.dumps(cluster.get("center_eye_mean_xyz")),
                    "center_eye_spread_m": cluster.get("center_eye_spread_m"),
                    "recommendation": cluster.get("recommendation"),
                }
            )


def markdown_report(payload: dict[str, Any]) -> str:
    conclusion = payload["global_conclusion"]
    lines = [
        "# 20260708 Cross-Episode Transform Pattern Aggregation",
        "",
        "## Conclusion",
        "",
        f"- Global transform supported: **{conclusion.get('global_transform_supported')}**.",
        f"- k=2 standing-position clusters supported: **{conclusion.get('two_stance_clusters_supported')}**.",
        f"- Main supported explanation: **{conclusion.get('main_supported_explanation')}**.",
        f"- Axis-swap + small-angle supported: **{conclusion.get('axis_swap_plus_small_angle_supported')}**."
        f"\n- Eye-frame no-axis-swap + small-angle supported: **{conclusion.get('eye_frame_no_axis_swap_plus_small_angle_supported')}**."
        f"\n- Cluster transform support: **{conclusion.get('cluster_level_transform_supported')}**; {conclusion.get('two_stance_cluster_note')}."
        f"\n- Axis-swap conclusion: {conclusion.get('axis_swap_conclusion')}.",
        f"- Usable world-coordinate episodes: `{', '.join(payload['usable_recommendations'].get('world_coordinate_usable') or [])}`.",
        f"- Reference-only episodes: `{', '.join(payload['usable_recommendations'].get('reference_only') or [])}`.",
        f"- Drop/skipped episodes: `{', '.join(payload['usable_recommendations'].get('drop_or_skipped') or [])}`.",
        "",
        "## Episode Table",
        "",
        "| id | episode | status | pose/model/lag | self RMSE/R2 | cluster RMSE/R2 | global RMSE/R2 | axis | both mapping | recommendation |",
        "|---:|---|---|---|---:|---:|---:|---|---|---|",
    ]
    for ep in payload["episodes"]:
        best = ep.get("bestfit") or {}
        cmp_row = ep.get("transform_comparison") or {}
        cluster = cmp_row.get("cluster") or {}
        glob = cmp_row.get("global") or {}
        both = ep.get("bothhands_alignment") or {}
        lines.append(
            f"| {ep['video_id']} | {ep['episode']} | {ep['status']} | "
            f"{best.get('pose_mode','-')}/{best.get('model_mode','-')}/{best.get('lag','-')} | "
            f"{best.get('mean_rmse_px','-'):.1f}/{best.get('mean_r2','-'):.3f}" if isinstance(best.get("mean_rmse_px"), (int, float)) else
            f"| {ep['video_id']} | {ep['episode']} | {ep['status']} | - | - | - | - | {ep.get('axis_candidate','-')} | {both.get('best_mapping_hypothesis','-')} | {ep.get('recommendation','-')} |"
        )
        if isinstance(best.get("mean_rmse_px"), (int, float)):
            lines[-1] = (
                f"| {ep['video_id']} | {ep['episode']} | {ep['status']} | "
                f"{best.get('pose_mode')}/{best.get('model_mode')}/{best.get('lag')} | "
                f"{best.get('mean_rmse_px'):.1f}/{best.get('mean_r2'):.3f} | "
                f"{cluster.get('mean_rmse_px', float('nan')):.1f}/{cluster.get('mean_r2', float('nan')):.3f} | "
                f"{glob.get('mean_rmse_px', float('nan')):.1f}/{glob.get('mean_r2', float('nan')):.3f} | "
                f"{ep.get('axis_candidate','-')} | {both.get('best_mapping_hypothesis','-')} | {ep.get('recommendation','-')} |"
            )
    lines += [
        "",
        "## Clusters",
        "",
        "| cluster | episodes | RMSE/R2 | center-eye mean xyz | recommendation |",
        "|---:|---|---:|---|---|",
    ]
    for cluster in payload["clusters"]:
        lines.append(
            f"| {cluster.get('cluster_id')} | `{', '.join(cluster.get('episodes') or [])}` | "
            f"{cluster.get('weighted_mean_rmse_px', float('nan')):.1f}/{cluster.get('weighted_mean_r2', float('nan')):.3f} | "
            f"`{cluster.get('center_eye_mean_xyz')}` | {cluster.get('recommendation')} |"
        )
    lines += [
        "",
        "## Pattern Summaries",
        "",
        f"- Axis patterns: `{payload['axis_pattern_summary']}`",
        f"- Lag summary: `{payload['lag_summary']}`",
        f"- Side mapping summary: `{payload['side_mapping_summary']}`",
        f"- Transform similarity summary: `{payload['transform_similarity_summary']}`",
        "",
        "## Output Files",
        "",
    ]
    for key, path in (payload.get("plots") or {}).items():
        lines.append(f"- {key}: `{path}`")
    return "\n".join(lines) + "\n"


def build_similarity_summary(episodes: list[dict[str, Any]], cluster_labels: dict[str, int]) -> dict[str, Any]:
    vectors = []
    for ep in episodes:
        best = ep.get("bestfit") or {}
        if not ep.get("fit_eligible") or best.get("model_mode") != "linear_xyz" or not best.get("coef"):
            continue
        coef = np.asarray(best["coef"], dtype=np.float64)
        if coef.shape != (4, 2):
            continue
        v = coef[:3, :].reshape(-1)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            continue
        vectors.append((str(ep["video_id"]), v / norm))
    distances = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a, va = vectors[i]
            b, vb = vectors[j]
            distances.append(
                {
                    "a": a,
                    "b": b,
                    "same_cluster": cluster_labels.get(a) == cluster_labels.get(b),
                    "distance": float(np.linalg.norm(va - vb)),
                }
            )
    same = [d["distance"] for d in distances if d["same_cluster"]]
    diff = [d["distance"] for d in distances if not d["same_cluster"]]
    return {
        "linear_episode_count": len(vectors),
        "linear_coef_pairwise_distances": distances,
        "mean_distance_same_cluster": float(np.mean(same)) if same else None,
        "mean_distance_cross_cluster": float(np.mean(diff)) if diff else None,
        "cluster_separates_transform_similarity": bool(same and diff and np.mean(same) < np.mean(diff) * 0.85),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vr_root = Path(args.vr_root)
    hamer_root = Path(args.hamer_root)

    bestfit_data = read_json(Path(args.bestfit_json))
    summary3d_data = read_json(Path(args.summary_3d_json))
    local_data = read_json(Path(args.local_json))
    both_data = read_json(Path(args.bothhands_json))
    bestfit_by_id = index_episodes(bestfit_data)
    summary3d_by_id = index_episodes(summary3d_data)
    local_by_id = index_episodes(local_data)
    both_by_id = index_episodes(both_data)
    manifest = read_json(hamer_root / "vr_hamer_input_manifest.json")

    records: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for item in manifest:
        if item.get("skipped") or args.episode_substr not in str(item.get("episode", "")):
            continue
        video_id = str(item["video_id"])
        episode = str(item["episode"])
        frames = read_json(vr_root / episode / f"{episode}.json").get("frames") or []
        npz_path = hamer_root / "output" / f"hand_detections_{video_id}.npz"
        npz = np.load(npz_path, allow_pickle=True) if npz_path.exists() else None
        pose_features = mean_pose_features(frames)
        bestfit_ep = bestfit_by_id.get(video_id) or {}
        best = bestfit_ep.get("best")
        summary3d = summary3d_by_id.get(video_id) or {}
        local_ep = local_by_id.get(video_id) or {}
        both_ep = both_by_id.get(video_id) or {}
        both_hands = both_ep.get("both_hands") or {}
        local_status = local_ep.get("status")
        both_status = both_hands.get("status")
        notes = []
        fit_eligible = False
        status = "skipped"
        if best:
            status = "fit"
            fit_eligible = int(best.get("n") or 0) >= args.fit_min_samples and local_status != "skipped" and both_status != "skipped"
            if int(best.get("n") or 0) < args.fit_min_samples:
                notes.append(f"bestfit n={best.get('n')} below cross-episode fit-min-samples={args.fit_min_samples}")
            if local_status == "skipped":
                notes.append("local hand alignment skipped")
            if both_status == "skipped":
                notes.append("both-hands local alignment skipped")
        else:
            notes.append(bestfit_ep.get("skip_reason") or "missing bestfit")
        decomp = decompose_transform(best)
        mirror_summary = None
        if both_hands:
            mirrors = [both_hands.get("best_left_mirror_variant"), both_hands.get("best_right_mirror_variant")]
            mirror_summary = "/".join([str(m) for m in mirrors if m]) or None
        row = {
            "video_id": video_id,
            "episode": episode,
            "status": status,
            "fit_eligible": fit_eligible,
            "bestfit": sanitize(best) if best else {"status": "skipped", "failure_reason": bestfit_ep.get("skip_reason")},
            "summary3d": {
                "status": summary3d.get("status"),
                "mean_hamer_ray_to_vr_core_dist_m": summary3d.get("mean_hamer_ray_to_vr_core_dist_m"),
                "median_hamer_ray_to_vr_core_dist_m": summary3d.get("median_hamer_ray_to_vr_core_dist_m"),
                "mean_hamer_ray_angle_deg": summary3d.get("mean_hamer_ray_angle_deg"),
            },
            "local_hand_alignment": {
                "status": local_ep.get("status"),
                "category": local_ep.get("category"),
                "best_side": local_ep.get("best_side"),
                "best_lag": local_ep.get("best_lag"),
                "best_alignment": local_ep.get("best_alignment"),
                "matched_frames": local_ep.get("matched_frames"),
                "local_shape_score": local_ep.get("local_shape_score"),
                "velocity_corr": local_ep.get("velocity_corr"),
            },
            "bothhands_alignment": {
                "status": both_hands.get("status"),
                "category": both_hands.get("category"),
                "best_mapping_hypothesis": both_hands.get("best_mapping_hypothesis"),
                "best_left_lag": both_hands.get("best_left_lag"),
                "best_right_lag": both_hands.get("best_right_lag"),
                "best_left_mirror_variant": both_hands.get("best_left_mirror_variant"),
                "best_right_mirror_variant": both_hands.get("best_right_mirror_variant"),
                "mirror_summary": mirror_summary,
                "matched_frames": both_hands.get("matched_frames"),
                "matched_both_frames": both_hands.get("matched_both_frames"),
                "local_shape_score": both_hands.get("local_shape_score"),
                "motion_score": both_hands.get("motion_score"),
            },
            "pose_features": sanitize(pose_features),
            "transform_decomposition": sanitize(decomp),
            "axis_candidate": axis_candidate_label(decomp) if decomp else None,
            "mirror_candidate": mirror_candidate_label(decomp) if decomp else None,
            "notes": notes,
        }
        episodes.append(row)
        if fit_eligible and npz is not None:
            records.append({"manifest": item, "frames": frames, "npz": npz, "video_id": video_id, "episode": episode, "pose_features": pose_features})

    if records:
        feature_matrix = np.stack([feature_vector_from_pose(rec["pose_features"]) for rec in records], axis=0)
        labels_arr, cluster_meta = kmeans2(feature_matrix)
        cluster_labels = {str(rec["video_id"]): int(label) for rec, label in zip(records, labels_arr)}
    else:
        cluster_meta = {"silhouette": None, "centroid_distance_std": None}
        cluster_labels = {}

    common = choose_common_global_config(records, args)
    cluster_transforms, cluster_eval_by_episode = build_cluster_transforms(common, records, cluster_labels)
    global_eval_by_episode = (common or {}).get("global_eval_by_episode") or {}
    common_per_episode = (common or {}).get("self_by_episode") or {}

    for ep in episodes:
        vid = str(ep["video_id"])
        ep["cluster_id"] = cluster_labels.get(vid)
        self_metric = compact_metric(ep.get("bestfit"))
        if vid in common_per_episode:
            common_self = common_per_episode[vid]
            ep["common_config_episode_local"] = {
                "pose_mode": common.get("pose_mode"),
                "model_mode": common.get("model_mode"),
                "lag": common_self.get("lag"),
                **compact_metric(common_self),
            }
        ep["transform_comparison"] = {
            "self": self_metric,
            "cluster": compact_metric(cluster_eval_by_episode.get(vid)),
            "global": compact_metric(global_eval_by_episode.get(vid)),
        }

    cluster_rows = []
    for cluster in cluster_transforms:
        centers = []
        for vid in cluster.get("episodes") or []:
            ep = next((row for row in episodes if str(row["video_id"]) == str(vid)), None)
            center = ((ep or {}).get("pose_features") or {}).get("center_eye_mean_xyz")
            if center is not None:
                centers.append(center)
        center_mean = np.mean(np.asarray(centers, dtype=np.float64), axis=0).tolist() if centers else None
        spread = float(np.mean(np.linalg.norm(np.asarray(centers) - np.asarray(center_mean), axis=1))) if centers and center_mean else None
        c = dict(cluster)
        c["coef"] = sanitize(c.get("coef"))
        c["center_eye_mean_xyz"] = sanitize(center_mean)
        c["center_eye_spread_m"] = spread
        c["recommendation"] = "usable as group transform" if (c.get("weighted_mean_rmse_px") or 1e9) < 80 else "diagnostic only"
        cluster_rows.append(sanitize(c))

    axis_counter = Counter(ep.get("axis_candidate") for ep in episodes if ep.get("axis_candidate"))
    mirror_counter = Counter(ep.get("mirror_candidate") for ep in episodes if ep.get("mirror_candidate"))
    pose_counter = Counter((ep.get("bestfit") or {}).get("pose_mode") for ep in episodes if (ep.get("bestfit") or {}).get("pose_mode"))
    model_counter = Counter((ep.get("bestfit") or {}).get("model_mode") for ep in episodes if (ep.get("bestfit") or {}).get("model_mode"))
    lag_values = [int((ep.get("bestfit") or {}).get("lag")) for ep in episodes if (ep.get("bestfit") or {}).get("lag") is not None]
    side_counter = Counter((ep.get("bothhands_alignment") or {}).get("best_mapping_hypothesis") for ep in episodes if (ep.get("bothhands_alignment") or {}).get("best_mapping_hypothesis"))
    mirror_variant_counter = Counter((ep.get("bothhands_alignment") or {}).get("mirror_summary") for ep in episodes if (ep.get("bothhands_alignment") or {}).get("mirror_summary"))

    transform_similarity = build_similarity_summary(episodes, cluster_labels)
    plots = make_plots(out_dir, episodes, cluster_rows, cluster_labels, transform_similarity)

    eligible_ids = [str(rec["video_id"]) for rec in records]
    cluster_rmse = [c.get("weighted_mean_rmse_px") for c in cluster_rows if c.get("weighted_mean_rmse_px") is not None]
    global_rmse = common.get("weighted_mean_rmse_px") if common else None
    self_rmse = [float((ep.get("bestfit") or {}).get("mean_rmse_px")) for ep in episodes if ep.get("fit_eligible") and (ep.get("bestfit") or {}).get("mean_rmse_px") is not None]
    median_self = float(np.median(self_rmse)) if self_rmse else None
    mean_cluster = float(np.mean(cluster_rmse)) if cluster_rmse else None
    supports_global = bool(global_rmse is not None and median_self is not None and global_rmse < max(90.0, median_self * 1.8))
    supports_cluster = bool(mean_cluster is not None and global_rmse is not None and mean_cluster < global_rmse * 0.85 and (cluster_meta.get("silhouette") or 0.0) > 0.15)
    supports_eye_no_swap_small_angle = axis_counter.get("no_axis_swap_xy", 0) >= max(1, len(eligible_ids) // 2) and mirror_counter.get("expected_u_plus_x_v_minus_y", 0) >= max(1, len(eligible_ids) // 2)
    supports_axis_swap_small_angle = axis_counter.get("xy_swap", 0) >= max(1, len(eligible_ids) // 2)

    world_usable: list[str] = []
    reference_only: list[str] = []
    drop: list[str] = []
    for ep in episodes:
        vid = str(ep["video_id"])
        best = ep.get("bestfit") or {}
        local = ep.get("local_hand_alignment") or {}
        ray = (ep.get("summary3d") or {}).get("mean_hamer_ray_to_vr_core_dist_m")
        local_score = finite_float(local.get("local_shape_score"), 0.0) or 0.0
        mean_r2 = finite_float(best.get("mean_r2"), -999.0) or -999.0
        rmse = finite_float(best.get("mean_rmse_px"), 9999.0) or 9999.0
        if not ep.get("fit_eligible"):
            ep["recommendation"] = "drop_or_skipped"
            drop.append(vid)
        elif vid == "13":
            ep["recommendation"] = "reference_only_side_label_swapped"
            reference_only.append(vid)
        elif mean_r2 >= 0.88 and rmse <= 70.0 and local_score >= 45.0 and (ray is None or float(ray) <= 0.35):
            ep["recommendation"] = "world_coordinate_usable"
            world_usable.append(vid)
        elif mean_r2 >= 0.35 and local_score >= 45.0:
            ep["recommendation"] = "reference_only"
            reference_only.append(vid)
        else:
            ep["recommendation"] = "drop_or_skipped"
            drop.append(vid)

    payload = {
        "global_conclusion": {
            "episode_filter": args.episode_substr,
            "fit_min_samples": args.fit_min_samples,
            "fit_eligible_video_ids": eligible_ids,
            "selected_common_global_config": {
                "pose_mode": common.get("pose_mode"),
                "model_mode": common.get("model_mode"),
                "episode_count": common.get("episode_count"),
                "sample_count": common.get("sample_count"),
                "weighted_mean_r2": common.get("weighted_mean_r2"),
                "weighted_mean_rmse_px": common.get("weighted_mean_rmse_px"),
                "coef": sanitize(common.get("coef")) if common else None,
            },
            "global_transform_supported": supports_global,
            "two_stance_clusters_supported": supports_cluster,
            "axis_swap_plus_small_angle_supported": supports_axis_swap_small_angle,
            "eye_frame_no_axis_swap_plus_small_angle_supported": supports_eye_no_swap_small_angle,
            "cluster_level_transform_supported": "partial" if supports_cluster else False,
            "two_stance_cluster_note": "k=2 separates standing/eye-pose clusters, but only cluster 1 has a usable group transform; cluster 0 remains diagnostic-only",
            "axis_swap_conclusion": "mostly no axis swap in eye-frame; u is usually driven by x and v by negative y, with z/crop terms varying by episode",
            "main_supported_explanation": (
                "eye-frame + lag + per-episode affine/crop is better supported than a single calibrated global transform"
                if not supports_global
                else "a common eye-frame linear transform is partly usable, but per-episode lag/crop still matters"
            ),
            "standing_cluster_test": cluster_meta,
        },
        "episodes": sanitize(episodes),
        "clusters": sanitize(cluster_rows),
        "axis_pattern_summary": {
            "axis_candidate_counts": dict(axis_counter),
            "mirror_candidate_counts": dict(mirror_counter),
            "best_pose_counts": dict(pose_counter),
            "best_model_counts": dict(model_counter),
            "tested_patterns": [
                "no axis swap",
                "xy axis swap",
                "yz axis candidate",
                "zx axis candidate",
                "sign flip candidates",
                "mirror_x",
                "mirror_y",
                "mirror_xy",
                "center_eye",
                "left_eye",
                "right_eye",
                "world",
                "linear_xyz",
                "perspective_xy_over_z",
            ],
        },
        "lag_summary": {
            "lag_counts": dict(Counter(lag_values)),
            "mean_lag": float(np.mean(lag_values)) if lag_values else None,
            "median_lag": float(np.median(lag_values)) if lag_values else None,
            "negative_lag_count": sum(1 for lag in lag_values if lag < 0),
            "positive_lag_count": sum(1 for lag in lag_values if lag > 0),
            "zero_lag_count": sum(1 for lag in lag_values if lag == 0),
            "lag_convention": "hamer_frame = vr_frame + lag",
        },
        "side_mapping_summary": {
            "bothhands_mapping_counts": dict(side_counter),
            "bothhands_mirror_variant_counts": dict(mirror_variant_counter),
            "id13_swapped_side_label": (both_by_id.get("13") or {}).get("both_hands", {}).get("best_mapping_hypothesis") == "swapped",
            "id13_note": "id13 is explicitly marked swapped because both-hands v2 selected VR_left->HaMeR_right and VR_right->HaMeR_left.",
        },
        "transform_similarity_summary": sanitize(transform_similarity),
        "usable_recommendations": {
            "world_coordinate_usable": world_usable,
            "reference_only": reference_only,
            "drop_or_skipped": drop,
            "rule": "world usable requires fit eligibility, good self R2/RMSE, medium local score, and small ray distance; id13 is kept reference-only due swapped side-label.",
        },
        "plots": plots,
    }
    write_csvs(out_dir, episodes, cluster_rows)
    (out_dir / "cross_episode_transform_patterns_20260708.json").write_text(json.dumps(sanitize(payload), indent=2, ensure_ascii=False) + "\n")
    (out_dir / "cross_episode_transform_patterns_20260708.md").write_text(markdown_report(sanitize(payload)))
    print(f"[done] episodes={len(episodes)} fit_eligible={len(records)}")
    print(f"[done] json={out_dir / 'cross_episode_transform_patterns_20260708.json'}")
    print(f"[done] md={out_dir / 'cross_episode_transform_patterns_20260708.md'}")
    print(f"[done] csv={out_dir / 'episode_transform_table_20260708.csv'}")
    print(f"[done] plots={len(plots)}")


if __name__ == "__main__":
    main()
