#!/usr/bin/env python3
"""
Joint statistics from converted HDF5 files.

Targets:
- Read first frame joint values for robot initialization.
- Compute stats on first N frames from one file or a folder of files.

Examples:
  python3 h5_joint_stats.py /path/to/demo.h5
  python3 h5_joint_stats.py /projects/zaijia001/R1/h5/pick/pnp_selected_55 --batch --frames 5
  python3 h5_joint_stats.py /projects/zaijia001/R1/h5/pick/pnp_selected_55 --batch --frames 10 --include-action --output-json ./h5_joint_stats_pnp.json
# 批量统计 + 众数近似
python3 h5_joint_stats.py /projects/zaijia001/R1/h5/pick/pnp_selected_55 --batch --frames 10 --mode-bins 80 --output-json ./h5_joint_stats_mode.json
python3 h5_joint_stats.py /projects/zaijia001/R1/h5/pnp_apple_star/select --batch --frames 10 --mode-bins 80 --output-json ./h5_joint_stats_mode_pnp_star_apple.json

# 先确保 h5 里已有 eef_pos/eef_euler（需要先跑过 h52eepose.py）
python3 RoboTwin/process/h5_joint_stats.py /projects/zaijia001/R1/h5/pnp_apple_star/select \
  --batch --frames 10000 --mode-bins 80 --include-eepose \
  --output-json /projects/zaijia001/RoboTwin/process/h5_joint_stats_mode_pnp_star_apple_eepose.json

  """
"""
[obs_arm_left_pos]
  first_frame_across_files count: 55
    mean: [-0.082882,  1.917741, -0.637424,  1.699203, -0.187354, -1.663718,
 -2.57754 ]
    std:  [0.091584, 0.093582, 0.127308, 0.201771, 0.195278, 0.188905, 0.277898]
  all_first_n_frames_across_files count: 550
    mean: [-0.081473,  1.930972, -0.646281,  1.698322, -0.091176, -1.662621,
 -2.578177]
    std:  [0.088492, 0.09323 , 0.130511, 0.20252 , 0.261126, 0.188759, 0.27715 ]
  file_mean_first_n_across_files count: 55
    mean: [-0.081473,  1.930972, -0.646281,  1.698322, -0.091176, -1.662621,
 -2.578177]
    std:  [0.087663, 0.091174, 0.129774, 0.202485, 0.235844, 0.188533, 0.277035]

[obs_arm_right_pos]
  first_frame_across_files count: 55
    mean: [ 0.107083,  1.640758, -0.549625, -1.713764,  0.165354,  1.61646 ,
 -2.73241 ]
    std:  [0.082599, 0.089122, 0.123678, 0.124111, 0.136794, 0.100812, 0.072353]
  all_first_n_frames_across_files count: 550
    mean: [ 0.108551,  1.605133, -0.552453, -1.713475,  0.104448,  1.617929,
 -2.732391]
    std:  [0.081543, 0.101498, 0.125507, 0.123985, 0.199071, 0.101361, 0.072355]
  file_mean_first_n_across_files count: 55
    mean: [ 0.108551,  1.605133, -0.552453, -1.713475,  0.104448,  1.617929,
 -2.732391]
    std:  [0.081436, 0.094759, 0.12481 , 0.123976, 0.1797  , 0.101128, 0.072355]

[obs_gripper_left_pos]
  first_frame_across_files count: 55
    mean: [91.686791]
    std:  [9.500392]
  all_first_n_frames_across_files count: 550
    mean: [91.707966]
    std:  [9.475887]
  file_mean_first_n_across_files count: 55
    mean: [91.707966]
    std:  [9.472252]

[obs_gripper_right_pos]
  first_frame_across_files count: 55
    mean: [96.988814]
    std:  [2.51103]
  all_first_n_frames_across_files count: 550
    mean: [96.988124]
    std:  [2.511267]
  file_mean_first_n_across_files count: 55
    mean: [96.988124]
    std:  [2.511264]

Saved JSON: h5_joint_stats_pnp.json"""


import argparse
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np


DEFAULT_DATASET_MAP = {
    "obs_arm_left_pos": "obs/arm_left/joint_pos",
    "obs_arm_right_pos": "obs/arm_right/joint_pos",
    "obs_gripper_left_pos": "obs/gripper_left/joint_pos",
    "obs_gripper_right_pos": "obs/gripper_right/joint_pos",
}

ACTION_DATASET_MAP = {
    "action_arm_left_pos": "action/arm_left/joint_pos",
    "action_arm_right_pos": "action/arm_right/joint_pos",
    "action_gripper_left_pos": "action/gripper_left/joint_pos",
    "action_gripper_right_pos": "action/gripper_right/joint_pos",
    "action_gripper_left_cmd": "action/gripper_left/commanded_pos",
    "action_gripper_right_cmd": "action/gripper_right/commanded_pos",
}

OBS_EEPOSE_DATASET_MAP = {
    "obs_arm_left_eef_pos": "obs/arm_left/eef_pos",
    "obs_arm_left_eef_euler": "obs/arm_left/eef_euler",
    "obs_arm_right_eef_pos": "obs/arm_right/eef_pos",
    "obs_arm_right_eef_euler": "obs/arm_right/eef_euler",
}

ACTION_EEPOSE_DATASET_MAP = {
    "action_arm_left_eef_pos": "action/arm_left/eef_pos",
    "action_arm_left_eef_euler": "action/arm_left/eef_euler",
    "action_arm_right_eef_pos": "action/arm_right/eef_pos",
    "action_arm_right_eef_euler": "action/arm_right/eef_euler",
}


def to_frame_matrix(arr: np.ndarray) -> np.ndarray:
    """Convert dataset to shape (num_frames, feature_dim)."""
    if arr.ndim == 0:
        return arr.reshape(1, 1).astype(np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1).astype(np.float64)
    return arr.reshape(arr.shape[0], -1).astype(np.float64)


def compute_mode_like(samples: np.ndarray, mode_bins: int) -> Dict[str, List]:
    mode_values: List[float] = []
    mode_methods: List[str] = []
    mode_support: List[int] = []
    mode_intervals: List[List[float]] = []

    for dim in range(samples.shape[1]):
        col = samples[:, dim]
        unique_vals, counts = np.unique(col, return_counts=True)
        max_count_idx = int(np.argmax(counts))
        max_count = int(counts[max_count_idx])

        # If repeated values are clear enough, use exact mode.
        # Otherwise (continuous joints), use histogram-based mode approximation.
        if max_count > 1 and unique_vals.size <= max(20, int(0.5 * col.size)):
            val = float(unique_vals[max_count_idx])
            mode_values.append(val)
            mode_methods.append("exact")
            mode_support.append(max_count)
            mode_intervals.append([val, val])
            continue

        if np.allclose(col, col[0]):
            val = float(col[0])
            mode_values.append(val)
            mode_methods.append("constant")
            mode_support.append(int(col.size))
            mode_intervals.append([val, val])
            continue

        hist, edges = np.histogram(col, bins=max(2, int(mode_bins)))
        idx = int(np.argmax(hist))
        left = float(edges[idx])
        right = float(edges[idx + 1])
        mode_values.append(0.5 * (left + right))
        mode_methods.append("hist")
        mode_support.append(int(hist[idx]))
        mode_intervals.append([left, right])

    return {
        "mode_like": mode_values,
        "mode_method": mode_methods,
        "mode_support": mode_support,
        "mode_interval": mode_intervals,
    }


def compute_stats(samples: np.ndarray, mode_bins: int) -> Dict[str, List[float]]:
    mode_info = compute_mode_like(samples, mode_bins)
    percentiles = np.percentile(samples, [5, 25, 50, 75, 95], axis=0)

    return {
        "count": int(samples.shape[0]),
        "mean": samples.mean(axis=0).tolist(),
        "std": samples.std(axis=0).tolist(),
        "min": samples.min(axis=0).tolist(),
        "max": samples.max(axis=0).tolist(),
        "p05": percentiles[0].tolist(),
        "p25": percentiles[1].tolist(),
        "p50": percentiles[2].tolist(),
        "p75": percentiles[3].tolist(),
        "p95": percentiles[4].tolist(),
        "mode_like": mode_info["mode_like"],
        "mode_method": mode_info["mode_method"],
        "mode_support": mode_info["mode_support"],
        "mode_interval": mode_info["mode_interval"],
    }


def normalize_vectors(vectors: List[np.ndarray], key: str, label: str) -> List[np.ndarray]:
    if not vectors:
        return vectors

    expected_dim = vectors[0].size
    valid = []
    dropped = 0
    for x in vectors:
        if x.size == expected_dim:
            valid.append(x)
        else:
            dropped += 1

    if dropped > 0:
        print(f"    Warning: dropped {dropped} mismatched samples for {key} in {label}.")
    return valid


def extract_h5_first_n(
    h5_path: Path,
    dataset_map: Dict[str, str],
    frames_per_key: int,
) -> Dict:
    result = {
        "file": str(h5_path),
        "used_paths": {},
        "missing_paths": {},
        "errors": [],
        "frames": {},
    }

    if frames_per_key < 1:
        result["errors"].append("frames_per_key must be >= 1")
        return result

    try:
        with h5py.File(h5_path, "r") as f:
            for key, ds_path in dataset_map.items():
                if ds_path not in f:
                    result["missing_paths"][key] = ds_path
                    continue

                arr_all = np.asarray(f[ds_path])
                frame_mat_all = to_frame_matrix(arr_all)
                if frame_mat_all.shape[0] == 0:
                    result["missing_paths"][key] = f"{ds_path} (empty)"
                    continue

                n = min(frames_per_key, frame_mat_all.shape[0])
                frame_mat = frame_mat_all[:n]

                result["used_paths"][key] = ds_path
                result["frames"][key] = frame_mat

    except Exception as exc:
        result["errors"].append(str(exc))

    return result


def summarize_file(extract_result: Dict, frames_per_key: int, mode_bins: int) -> Dict:
    summary = {
        "file": extract_result["file"],
        "used_paths": extract_result["used_paths"],
        "missing_paths": extract_result["missing_paths"],
        "errors": extract_result["errors"],
        "keys": {},
    }

    for key, frame_mat in extract_result["frames"].items():
        first = frame_mat[0]
        first_n_stats = compute_stats(frame_mat[:frames_per_key], mode_bins)

        summary["keys"][key] = {
            "collected_frames": int(frame_mat.shape[0]),
            "feature_dim": int(frame_mat.shape[1]),
            "first_frame": first.tolist(),
            "stats_first_n": first_n_stats,
        }

    return summary


def summarize_batch(
    extract_results: List[Dict],
    selected_keys: List[str],
    frames_per_key: int,
    mode_bins: int,
) -> Dict:
    batch_summary = {
        "num_files": len(extract_results),
        "frames_per_key": frames_per_key,
        "keys": {},
    }

    for key in selected_keys:
        first_frame_samples: List[np.ndarray] = []
        all_first_n_samples: List[np.ndarray] = []
        file_mean_first_n_samples: List[np.ndarray] = []

        for res in extract_results:
            frame_mat = res["frames"].get(key)
            if frame_mat is None or frame_mat.shape[0] == 0:
                continue

            selected = frame_mat[:frames_per_key]
            first_frame_samples.append(selected[0])
            all_first_n_samples.extend(selected)
            file_mean_first_n_samples.append(selected.mean(axis=0))

        first_frame_samples = normalize_vectors(first_frame_samples, key, "batch_first")
        all_first_n_samples = normalize_vectors(all_first_n_samples, key, "batch_all_first_n")
        file_mean_first_n_samples = normalize_vectors(file_mean_first_n_samples, key, "batch_file_mean")

        key_summary = {
            "first_frame_across_files": None,
            "all_first_n_frames_across_files": None,
            "file_mean_first_n_across_files": None,
        }

        if first_frame_samples:
            key_summary["first_frame_across_files"] = compute_stats(np.stack(first_frame_samples, axis=0), mode_bins)
        if all_first_n_samples:
            key_summary["all_first_n_frames_across_files"] = compute_stats(np.stack(all_first_n_samples, axis=0), mode_bins)
        if file_mean_first_n_samples:
            key_summary["file_mean_first_n_across_files"] = compute_stats(np.stack(file_mean_first_n_samples, axis=0), mode_bins)

        batch_summary["keys"][key] = key_summary

    return batch_summary


def print_stats_block(stats: Dict, indent: str = "    "):
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    mode_like = np.array(stats["mode_like"])
    p05 = np.array(stats["p05"])
    p50 = np.array(stats["p50"])
    p95 = np.array(stats["p95"])
    mode_support = np.array(stats["mode_support"])

    mode_methods = stats["mode_method"]
    unique_methods = sorted(set(mode_methods))
    method_desc = unique_methods[0] if len(unique_methods) == 1 else "mixed"

    print(f"{indent}mean:      {np.array2string(mean, precision=6, separator=', ')}")
    print(f"{indent}std:       {np.array2string(std, precision=6, separator=', ')}")
    print(f"{indent}mode_like: {np.array2string(mode_like, precision=6, separator=', ')}")
    print(f"{indent}mode_method: {method_desc}")
    print(f"{indent}mode_support: {np.array2string(mode_support, separator=', ')}")
    print(f"{indent}p05:       {np.array2string(p05, precision=6, separator=', ')}")
    print(f"{indent}p50:       {np.array2string(p50, precision=6, separator=', ')}")
    print(f"{indent}p95:       {np.array2string(p95, precision=6, separator=', ')}")


def print_file_summary(summary: Dict):
    print(f"\n>>> File: {Path(summary['file']).name}")

    if summary["errors"]:
        for err in summary["errors"]:
            print(f"  Error: {err}")

    if summary["used_paths"]:
        print("  Dataset selection:")
        for key in sorted(summary["used_paths"].keys()):
            print(f"    {key:28s} <- {summary['used_paths'][key]}")

    if summary["missing_paths"]:
        print("  Missing dataset:")
        for key in sorted(summary["missing_paths"].keys()):
            print(f"    {key:28s} -> {summary['missing_paths'][key]}")

    if not summary["keys"]:
        return

    print("  Joint stats:")
    for key in sorted(summary["keys"].keys()):
        info = summary["keys"][key]
        print(f"    {key}")
        print(f"      collected_frames: {info['collected_frames']}")
        print(f"      feature_dim: {info['feature_dim']}")
        print(f"      first_frame: {np.array2string(np.array(info['first_frame']), precision=6, separator=', ')}")
        print("      stats(first_n):")
        print_stats_block(info["stats_first_n"], indent="        ")


def print_batch_brief(summary: Dict):
    parts = []
    for key in sorted(summary["keys"].keys()):
        parts.append(f"{key}:{summary['keys'][key]['collected_frames']}")
    print(f"  collected -> {', '.join(parts)}")
    if summary["errors"]:
        print(f"  errors -> {summary['errors']}")


def print_batch_summary(batch_summary: Dict):
    print("\n=== Batch Statistics ===")
    print(f"files: {batch_summary['num_files']}")
    print(f"frames_per_key: {batch_summary['frames_per_key']}")

    for key in sorted(batch_summary["keys"].keys()):
        print(f"\n[{key}]")
        key_stats = batch_summary["keys"][key]

        first_stats = key_stats["first_frame_across_files"]
        if first_stats is None:
            print("  first_frame_across_files: no data")
        else:
            print(f"  first_frame_across_files count: {first_stats['count']}")
            print_stats_block(first_stats, indent="    ")

        all_stats = key_stats["all_first_n_frames_across_files"]
        if all_stats is None:
            print("  all_first_n_frames_across_files: no data")
        else:
            print(f"  all_first_n_frames_across_files count: {all_stats['count']}")
            print_stats_block(all_stats, indent="    ")

        file_mean_stats = key_stats["file_mean_first_n_across_files"]
        if file_mean_stats is None:
            print("  file_mean_first_n_across_files: no data")
        else:
            print(f"  file_mean_first_n_across_files count: {file_mean_stats['count']}")
            print_stats_block(file_mean_stats, indent="    ")


def collect_h5_files(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix == ".h5":
        return [input_path]
    if not input_path.is_dir():
        return []
    return sorted(input_path.glob("*.h5"))


def parse_args():
    parser = argparse.ArgumentParser(description="Statistics of joint-related data from HDF5 files")
    parser.add_argument("input", type=str, help="Input .h5 file or directory with .h5 files")
    parser.add_argument("--batch", "-b", action="store_true", help="Batch mode for directory input")
    parser.add_argument("--frames", "-n", type=int, default=5, help="Number of first frames used for stats")
    parser.add_argument("--mode-bins", type=int, default=50, help="Histogram bins for mode-like estimation on continuous values")
    parser.add_argument("--include-action", action="store_true", help="Include action datasets")
    parser.add_argument("--include-eepose", action="store_true", help="Include eef_pos and eef_euler datasets (obs, and action if --include-action)")
    parser.add_argument("--verbose-per-file", action="store_true", help="Print detailed stats for each file in batch mode")
    parser.add_argument("--output-json", type=str, default="", help="Optional JSON output path")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    dataset_map = dict(DEFAULT_DATASET_MAP)
    if args.include_eepose:
        dataset_map.update(OBS_EEPOSE_DATASET_MAP)
    if args.include_action:
        dataset_map.update(ACTION_DATASET_MAP)
        if args.include_eepose:
            dataset_map.update(ACTION_EEPOSE_DATASET_MAP)

    selected_keys = list(dataset_map.keys())

    if input_path.is_dir() or args.batch:
        h5_files = collect_h5_files(input_path)
        if not h5_files:
            print(f"No .h5 files found in: {input_path}")
            return

        print(f"Found {len(h5_files)} h5 files in {input_path}")

        extract_results: List[Dict] = []
        file_summaries: List[Dict] = []

        for idx, h5_path in enumerate(h5_files, start=1):
            print(f"\n[{idx}/{len(h5_files)}] {h5_path.name}")
            extract_result = extract_h5_first_n(h5_path, dataset_map, args.frames)
            summary = summarize_file(extract_result, args.frames, args.mode_bins)

            extract_results.append(extract_result)
            file_summaries.append(summary)

            if args.verbose_per_file:
                print_file_summary(summary)
            else:
                print_batch_brief(summary)

        batch_summary = summarize_batch(extract_results, selected_keys, args.frames, args.mode_bins)
        print_batch_summary(batch_summary)

        if args.output_json:
            payload = {
                "mode": "batch",
                "selected_keys": selected_keys,
                "mode_bins": args.mode_bins,
                "file_summaries": file_summaries,
                "batch_summary": batch_summary,
            }
            out = Path(args.output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\nSaved JSON: {out}")
        return

    if not (input_path.is_file() and input_path.suffix == ".h5"):
        print(f"Input is neither a .h5 file nor a directory: {input_path}")
        return

    extract_result = extract_h5_first_n(input_path, dataset_map, args.frames)
    summary = summarize_file(extract_result, args.frames, args.mode_bins)
    print_file_summary(summary)

    if args.output_json:
        payload = {
            "mode": "single",
            "selected_keys": selected_keys,
            "mode_bins": args.mode_bins,
            "file_summary": summary,
        }
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON: {out}")


if __name__ == "__main__":
    main()
