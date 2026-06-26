#!/usr/bin/env python3
"""Convert Piper0515 LeRobot pose vectors from world frame to arm base frame."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_LEROBOT_LOCAL = Path("/home/zaijia001/.cache/huggingface/lerobot/local")
DEFAULT_ROBOT_CONFIG = Path("/home/zaijia001/ssd/RoboTwin/robot_config_PiperPika_agx_dual_table_0515.json")
STATE_COL = "observation.state"
ACTION_COL = "action"
MARKER_NAME = "piper0515_world_to_base_conversion.json"


def resolve_repo(value: str, lerobot_local: Path) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    if value.startswith("local/"):
        return (lerobot_local / value.split("/", 1)[1]).resolve()
    return (lerobot_local / value).resolve()


def default_output_repo_id(source: str, source_path: Path) -> str:
    name = source.split("/", 1)[1] if source.startswith("local/") else source_path.name
    if name.endswith("_25ep"):
        name = f"{name[:-5]}_piper0515_25ep"
    else:
        name = f"{name}_piper0515"
    return f"local/{name}"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError(f"zero-norm quaternion: {quat}")
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def euler_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    """Match scipy Rotation.from_euler("xyz", euler).as_matrix()."""
    x, y, z = [float(v) for v in euler]
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    """Inverse of euler_xyz_to_matrix, with angles normalized to [-pi, pi]."""
    r = np.asarray(matrix, dtype=np.float64)
    sy = float(np.clip(-r[2, 0], -1.0, 1.0))
    y = math.asin(sy)
    cy = math.cos(y)
    if abs(cy) > 1e-8:
        x = math.atan2(r[2, 1], r[2, 2])
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = 0.0
        z = math.atan2(-r[0, 1], r[1, 1])
    return np.array([x, y, z], dtype=np.float64)


def load_base_poses(robot_config: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(robot_config.read_text(encoding="utf-8"))
    config = payload.get("left_embodiment_config", payload)
    poses = np.asarray(config["robot_pose"], dtype=np.float64)
    if poses.shape != (2, 7):
        raise ValueError(f"expected robot_pose shape (2, 7), got {poses.shape}")
    left_t = poses[0, :3]
    right_t = poses[1, :3]
    left_r = quat_wxyz_to_matrix(poses[0, 3:7])
    right_r = quat_wxyz_to_matrix(poses[1, 3:7])
    return left_t, left_r, right_t, right_r


def transform_arm_state(
    arm_state: np.ndarray,
    base_t: np.ndarray,
    base_r: np.ndarray,
    gripper_scale: float | None,
) -> np.ndarray:
    arm_state = np.asarray(arm_state, dtype=np.float64)
    if arm_state.shape != (7,):
        raise ValueError(f"expected arm state shape (7,), got {arm_state.shape}")
    out = np.empty(7, dtype=np.float64)
    out[:3] = base_r.T @ (arm_state[:3] - base_t)
    out[3:6] = matrix_to_euler_xyz(base_r.T @ euler_xyz_to_matrix(arm_state[3:6]))
    out[6] = arm_state[6] if gripper_scale is None else arm_state[6] * gripper_scale
    return out


def transform_pose_array(
    values: np.ndarray,
    left_t: np.ndarray,
    left_r: np.ndarray,
    right_t: np.ndarray,
    right_r: np.ndarray,
    gripper_scale: float | None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 14:
        raise ValueError(f"expected pose array shape (N, 14), got {values.shape}")
    out = np.empty_like(values, dtype=np.float64)
    for idx, row in enumerate(values):
        out[idx, :7] = transform_arm_state(row[:7], left_t, left_r, gripper_scale)
        out[idx, 7:14] = transform_arm_state(row[7:14], right_t, right_r, gripper_scale)
    return out.astype(np.float32)


def stack_vector_column(series: pd.Series, label: str) -> np.ndarray:
    rows = [np.asarray(item, dtype=np.float64) for item in series.to_list()]
    if not rows:
        raise ValueError(f"{label}: empty vector column")
    arr = np.stack(rows, axis=0)
    if arr.ndim != 2 or arr.shape[1] != 14:
        raise ValueError(f"{label}: expected shape (N, 14), got {arr.shape}")
    return arr


def vector_stats(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": arr.min(axis=0).astype(float).tolist(),
        "max": arr.max(axis=0).astype(float).tolist(),
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0).astype(float).tolist(),
        "count": [int(arr.shape[0])],
    }


def summarize_bbox(label: str, values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float64)
    left = arr[:, :3]
    right = arr[:, 7:10]
    return (
        f"{label}: "
        f"L {np.round(left.min(axis=0), 3).tolist()}->{np.round(left.max(axis=0), 3).tolist()} "
        f"R {np.round(right.min(axis=0), 3).tolist()}->{np.round(right.max(axis=0), 3).tolist()}"
    )


def convert_repo(
    source_dir: Path,
    output_dir: Path,
    robot_config: Path,
    gripper_scale: float | None,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if not (source_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"not a LeRobot repo or missing meta/info.json: {source_dir}")
    if (source_dir / "meta" / MARKER_NAME).exists():
        raise RuntimeError(f"source already has {MARKER_NAME}; refusing to convert it again: {source_dir}")

    left_t, left_r, right_t, right_r = load_base_poses(robot_config)
    parquet_files = sorted((source_dir / "data" / "chunk-000").glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet episodes under {source_dir / 'data' / 'chunk-000'}")

    target_dir = source_dir if dry_run else output_dir
    if not dry_run:
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(f"output exists: {output_dir}; pass --overwrite to replace it")
            shutil.rmtree(output_dir)
        shutil.copytree(source_dir, output_dir)

    stats_updates: dict[int, dict[str, dict]] = {}
    total_frames = 0
    first_before: np.ndarray | None = None
    first_after: np.ndarray | None = None

    for src_parquet in parquet_files:
        episode_index = int(src_parquet.stem.split("_")[-1])
        df = pd.read_parquet(src_parquet)
        if STATE_COL not in df or ACTION_COL not in df:
            raise KeyError(f"{src_parquet}: missing {STATE_COL!r} or {ACTION_COL!r}")

        state_before = stack_vector_column(df[STATE_COL], f"{src_parquet}:{STATE_COL}")
        action_before = stack_vector_column(df[ACTION_COL], f"{src_parquet}:{ACTION_COL}")
        state_after = transform_pose_array(state_before, left_t, left_r, right_t, right_r, gripper_scale)
        action_after = transform_pose_array(action_before, left_t, left_r, right_t, right_r, gripper_scale)

        if first_before is None:
            first_before = state_before
            first_after = state_after

        total_frames += int(len(df))
        stats_updates[episode_index] = {
            STATE_COL: vector_stats(state_after),
            ACTION_COL: vector_stats(action_after),
        }

        if not dry_run:
            df[STATE_COL] = list(state_after)
            df[ACTION_COL] = list(action_after)
            dst_parquet = target_dir / "data" / "chunk-000" / src_parquet.name
            df.to_parquet(dst_parquet, index=False)
        print(f"[convert] episode={episode_index:06d} frames={len(df)}")

    if not dry_run:
        stats_path = target_dir / "meta" / "episodes_stats.jsonl"
        stats_rows = load_jsonl(stats_path)
        for row in stats_rows:
            episode_index = int(row["episode_index"])
            update = stats_updates.get(episode_index)
            if not update:
                continue
            row.setdefault("stats", {})[STATE_COL] = update[STATE_COL]
            row.setdefault("stats", {})[ACTION_COL] = update[ACTION_COL]
        write_jsonl(stats_path, stats_rows)

        marker = {
            "schema": "piper0515_world_to_base_conversion.v1",
            "converted_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_repo": str(source_dir),
            "robot_config": str(robot_config),
            "left_base_position_world": left_t.tolist(),
            "right_base_position_world": right_t.tolist(),
            "gripper_scale": gripper_scale,
            "state_action_layout": "left xyz/rpy/gripper + right xyz/rpy/gripper",
            "position_transform": "p_base = R_base.T @ (p_world - t_base)",
            "orientation_transform": "R_base_frame = R_base.T @ R_world; rpy uses scipy-compatible xyz convention",
        }
        (target_dir / "meta" / MARKER_NAME).write_text(json.dumps(marker, indent=2), encoding="utf-8")

    if first_before is not None and first_after is not None:
        print(summarize_bbox("[bbox before first episode]", first_before))
        print(summarize_bbox("[bbox after  first episode]", first_after))
    print(f"[done] source={source_dir}")
    print(f"[done] output={output_dir if not dry_run else '(dry-run only)'}")
    print(f"[done] total_episodes={len(parquet_files)} total_frames={total_frames}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a local LeRobot repo whose state/action poses are in Piper0515 world frame "
            "into per-arm base-frame poses that match the real robot cartesian datasets."
        )
    )
    parser.add_argument("--source", "--source-repo-id", required=True, help="Source repo id/path, e.g. local/h2o_task_ours_25ep.")
    parser.add_argument("--output-repo-id", help="Destination repo id under the LeRobot local cache.")
    parser.add_argument("--output", help="Destination repo path. Overrides --output-repo-id.")
    parser.add_argument("--lerobot-local", type=Path, default=DEFAULT_LEROBOT_LOCAL)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--gripper-scale", type=float, default=0.0967, help="Map [0,1] gripper command to real-robot width scale.")
    parser.add_argument("--no-gripper-scale", action="store_true", help="Keep the original gripper values.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the destination repo if it already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Read and summarize conversion without writing output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = resolve_repo(args.source, args.lerobot_local)
    if args.output:
        output_dir = Path(args.output).expanduser().resolve()
    else:
        output_repo_id = args.output_repo_id or default_output_repo_id(args.source, source_dir)
        output_dir = resolve_repo(output_repo_id, args.lerobot_local)
    gripper_scale = None if args.no_gripper_scale else float(args.gripper_scale)
    convert_repo(
        source_dir=source_dir,
        output_dir=output_dir,
        robot_config=args.robot_config.expanduser().resolve(),
        gripper_scale=gripper_scale,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
