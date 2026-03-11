#!/usr/bin/env python3
"""
Extract initial joint values from ROS bag files (rosbags backend).

Features:
1. Read the first frame of joint topics from a single .bag.
2. Statistics mode for a folder of .bag files (first N frames).
3. Topic priority fallback compatible with bag2h5_lz4.py.

Examples:
  python3 bag_first_joint_stats.py /projects/zaijia001/R1/pour/demo.bag
  python3 bag_first_joint_stats.py /projects/zaijia001/R1/stack_cup --batch --frames 5
  python3 bag_first_joint_stats.py /projects/zaijia001/R1/stack_cup --batch --frames 10 --output-json ./init_joint_stats.json
  python3 bag_first_joint_stats.py /projects/zaijia001/R1/pour/demo.bag --include-action --frames 3

  stack_cup:
  [arm_left_pos]
  first_frame_across_bags count: 48
    mean: [-0.11325,  1.87971, -0.62249,  1.4235 , -0.29925, -1.27673, -2.65068]
    std:  [0.102  , 0.11877, 0.13167, 0.17691, 0.19769, 0.21345, 0.15731]
  bag_mean_first_n_across_bags count: 48
    mean: [-0.10447,  1.85093, -0.68848,  1.42318, -0.30142, -1.24984, -2.65068]
    std:  [0.09957, 0.10142, 0.14724, 0.17865, 0.19005, 0.21351, 0.15731]
  all_first_n_frames_across_bags count: 480
    mean: [-0.10447,  1.85093, -0.68848,  1.42318, -0.30142, -1.24984, -2.65068]
    std:  [0.10063, 0.10947, 0.15873, 0.17868, 0.19261, 0.21537, 0.15731]

[arm_right_pos]
  first_frame_across_bags count: 48
    mean: [ 0.1607 ,  1.66766, -0.49318, -1.40258,  0.15746,  1.43493, -2.72583]
    std:  [0.10268, 0.09816, 0.0866 , 0.11041, 0.13341, 0.16545, 0.09419]
  bag_mean_first_n_across_bags count: 48
    mean: [ 0.16059,  1.54847, -0.5234 , -1.40134,  0.16981,  1.43446, -2.72583]
    std:  [0.1023 , 0.1153 , 0.1068 , 0.11203, 0.13391, 0.16605, 0.09418]
  all_first_n_frames_across_bags count: 480
    mean: [ 0.16059,  1.54847, -0.5234 , -1.40134,  0.16981,  1.43446, -2.72583]
    std:  [0.10238, 0.15256, 0.11649, 0.11212, 0.13673, 0.16606, 0.09418]

[gripper_left_pos]
  first_frame_across_bags count: 48
    mean: [94.1529]
    std:  [5.44321]
  bag_mean_first_n_across_bags count: 48
    mean: [94.15305]
    std:  [5.44326]
  all_first_n_frames_across_bags count: 480
    mean: [94.15305]
    std:  [5.44326]

[gripper_right_pos]
  first_frame_across_bags count: 48
    mean: [96.76491]
    std:  [3.24274]
  bag_mean_first_n_across_bags count: 48
    mean: [96.76465]
    std:  [3.24251]
  all_first_n_frames_across_bags count: 480
    mean: [96.76465]
    std:  [3.24251]
  
  """

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import get_typestore, Stores
except ImportError:
    print("Error: Library 'rosbags' not found.")
    print("Please run: pip install rosbags")
    raise SystemExit(1)


TOPIC_MAP_PRIORITY = {
    'arm_left_pos': ['/hdas/feedback_arm_left', '/hdas/feedback_arm_left_low'],
    'arm_right_pos': ['/hdas/feedback_arm_right', '/hdas/feedback_arm_right_low'],
    'gripper_left_pos': ['/hdas/feedback_gripper_left', '/hdas/feedback_gripper_left_low'],
    'gripper_right_pos': ['/hdas/feedback_gripper_right', '/hdas/feedback_gripper_right_low'],
    'action_arm_left_pos': ['/motion_target/target_joint_state_arm_left', '/motion_target/target_joint_state_arm_left_low'],
    'action_arm_right_pos': ['/motion_target/target_joint_state_arm_right', '/motion_target/target_joint_state_arm_right_low'],
    'action_gripper_left_pos_cmd': ['/motion_control/position_control_gripper_left', '/motion_control/position_control_gripper_left_low'],
    'action_gripper_right_pos_cmd': ['/motion_control/position_control_gripper_right', '/motion_control/position_control_gripper_right_low'],
}

DEFAULT_KEYS = [
    'arm_left_pos',
    'arm_right_pos',
    'gripper_left_pos',
    'gripper_right_pos',
]

ACTION_KEYS = [
    'action_arm_left_pos',
    'action_arm_right_pos',
    'action_gripper_left_pos_cmd',
    'action_gripper_right_pos_cmd',
]


def _as_1d_float_array(msg) -> Optional[np.ndarray]:
    if hasattr(msg, 'position'):
        arr = np.asarray(msg.position, dtype=np.float64).reshape(-1)
        return arr

    if hasattr(msg, 'data'):
        data = msg.data
        if isinstance(data, (list, tuple, np.ndarray)):
            return np.asarray(data, dtype=np.float64).reshape(-1)
        return np.asarray([data], dtype=np.float64)

    return None


def _select_topic_map(reader: Reader, requested_keys: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    available_topics = {conn.topic for conn in reader.connections}
    topic_to_key: Dict[str, str] = {}
    used_topics: Dict[str, str] = {}

    for key in requested_keys:
        for topic in TOPIC_MAP_PRIORITY[key]:
            if topic in available_topics:
                topic_to_key[topic] = key
                used_topics[key] = topic
                break

    return topic_to_key, used_topics


def _normalize_samples(samples: List[np.ndarray], key: str, bag_name: str) -> List[np.ndarray]:
    if not samples:
        return samples

    expected_dim = samples[0].size
    normalized: List[np.ndarray] = []
    dropped = 0

    for x in samples:
        if x.size != expected_dim:
            dropped += 1
            continue
        normalized.append(x)

    if dropped > 0:
        print(f"    Warning: dropped {dropped} mismatched frames for {key} in {bag_name}.")
    return normalized


def extract_first_n_frames(
    bag_path: Path,
    requested_keys: List[str],
    frames_per_key: int,
    typestore,
) -> Dict:
    result = {
        'bag': str(bag_path),
        'used_topics': {},
        'frames': {k: [] for k in requested_keys},
        'errors': [],
    }

    if frames_per_key < 1:
        result['errors'].append('frames_per_key must be >= 1')
        return result

    try:
        with Reader(bag_path) as reader:
            topic_to_key, used_topics = _select_topic_map(reader, requested_keys)
            result['used_topics'] = used_topics

            if not topic_to_key:
                result['errors'].append('No requested joint topics found in bag.')
                return result

            connections = [c for c in reader.connections if c.topic in topic_to_key]
            finished_keys = set()

            for connection, _, rawdata in reader.messages(connections=connections):
                key = topic_to_key[connection.topic]

                if key in finished_keys:
                    continue

                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                arr = _as_1d_float_array(msg)
                if arr is None:
                    continue

                result['frames'][key].append(arr)

                if len(result['frames'][key]) >= frames_per_key:
                    finished_keys.add(key)

                if len(finished_keys) == len(used_topics):
                    break

        for key in requested_keys:
            result['frames'][key] = _normalize_samples(result['frames'][key], key, bag_path.name)

    except Exception as exc:
        result['errors'].append(str(exc))

    return result


def _compute_stats(samples: List[np.ndarray]) -> Optional[Dict[str, List[float]]]:
    if not samples:
        return None

    arr = np.stack(samples, axis=0)
    return {
        'count': int(arr.shape[0]),
        'mean': arr.mean(axis=0).tolist(),
        'std': arr.std(axis=0).tolist(),
        'min': arr.min(axis=0).tolist(),
        'max': arr.max(axis=0).tolist(),
    }


def build_bag_summary(extract_result: Dict, frames_per_key: int) -> Dict:
    summary = {
        'bag': extract_result['bag'],
        'used_topics': extract_result['used_topics'],
        'errors': extract_result['errors'],
        'keys': {},
    }

    for key, frames in extract_result['frames'].items():
        key_info = {
            'collected_frames': len(frames),
            'requested_frames': frames_per_key,
            'first_frame': frames[0].tolist() if frames else None,
            'stats_first_n': _compute_stats(frames[:frames_per_key]) if frames else None,
        }
        summary['keys'][key] = key_info

    return summary


def print_single_summary(summary: Dict):
    print(f"\n>>> Bag: {Path(summary['bag']).name}")

    if summary['errors']:
        for err in summary['errors']:
            print(f"  Error: {err}")

    if summary['used_topics']:
        print("  Topic selection:")
        for key in sorted(summary['used_topics'].keys()):
            topic = summary['used_topics'][key]
            freq = "LOW" if "_low" in topic else "HIGH"
            print(f"    {key:32s} <- {topic} [{freq}]")

    print("  Joint values:")
    for key in sorted(summary['keys'].keys()):
        info = summary['keys'][key]
        print(f"    {key}")
        print(f"      collected: {info['collected_frames']}/{info['requested_frames']}")
        if info['first_frame'] is None:
            print("      first_frame: None")
            continue

        first = np.array(info['first_frame'])
        print(f"      first_frame: {np.array2string(first, precision=5, separator=', ')}")

        stats = info['stats_first_n']
        if stats is not None:
            mean = np.array(stats['mean'])
            std = np.array(stats['std'])
            print(f"      mean(first_n): {np.array2string(mean, precision=5, separator=', ')}")
            print(f"      std(first_n):  {np.array2string(std, precision=5, separator=', ')}")


def summarize_batch(
    extract_results: List[Dict],
    requested_keys: List[str],
    frames_per_key: int,
) -> Dict:
    aggregate = {
        'num_bags': len(extract_results),
        'frames_per_key': frames_per_key,
        'keys': {},
    }

    for key in requested_keys:
        first_frame_samples: List[np.ndarray] = []
        all_frame_samples: List[np.ndarray] = []

        # Build first-frame aggregate directly from first-frame samples.
        bag_mean_samples: List[np.ndarray] = []
        for extract_result in extract_results:
            frames = extract_result['frames'].get(key, [])
            if not frames:
                continue
            first_frame_samples.append(frames[0])
            selected_frames = frames[:frames_per_key]
            all_frame_samples.extend(selected_frames)
            bag_mean_samples.append(np.stack(selected_frames, axis=0).mean(axis=0))

        first_frame_samples = _normalize_samples(first_frame_samples, key, 'batch_first_frame')
        all_frame_samples = _normalize_samples(all_frame_samples, key, 'batch_all_first_n')
        bag_mean_samples = _normalize_samples(bag_mean_samples, key, 'batch_mean_first_n')

        first_stats = _compute_stats(first_frame_samples)
        all_first_n_stats = _compute_stats(all_frame_samples)
        mean_stats = _compute_stats(bag_mean_samples)

        aggregate['keys'][key] = {
            'first_frame_across_bags': first_stats,
            'all_first_n_frames_across_bags': all_first_n_stats,
            'bag_mean_first_n_across_bags': mean_stats,
        }

    return aggregate


def print_batch_summary(batch_summary: Dict):
    print("\n=== Batch Statistics ===")
    print(f"bags: {batch_summary['num_bags']}")
    print(f"frames_per_key: {batch_summary['frames_per_key']}")

    for key in sorted(batch_summary['keys'].keys()):
        print(f"\n[{key}]")
        key_stats = batch_summary['keys'][key]

        first_stats = key_stats['first_frame_across_bags']
        if first_stats is None:
            print("  first_frame_across_bags: no data")
        else:
            print(f"  first_frame_across_bags count: {first_stats['count']}")
            print(f"    mean: {np.array2string(np.array(first_stats['mean']), precision=5, separator=', ')}")
            print(f"    std:  {np.array2string(np.array(first_stats['std']), precision=5, separator=', ')}")

        mean_stats = key_stats['bag_mean_first_n_across_bags']
        if mean_stats is None:
            print("  bag_mean_first_n_across_bags: no data")
        else:
            print(f"  bag_mean_first_n_across_bags count: {mean_stats['count']}")
            print(f"    mean: {np.array2string(np.array(mean_stats['mean']), precision=5, separator=', ')}")
            print(f"    std:  {np.array2string(np.array(mean_stats['std']), precision=5, separator=', ')}")

        all_first_n_stats = key_stats['all_first_n_frames_across_bags']
        if all_first_n_stats is None:
            print("  all_first_n_frames_across_bags: no data")
        else:
            print(f"  all_first_n_frames_across_bags count: {all_first_n_stats['count']}")
            print(f"    mean: {np.array2string(np.array(all_first_n_stats['mean']), precision=5, separator=', ')}")
            print(f"    std:  {np.array2string(np.array(all_first_n_stats['std']), precision=5, separator=', ')}")


def print_batch_bag_brief(summary: Dict):
    key_parts = []
    for key in sorted(summary['keys'].keys()):
        collected = summary['keys'][key]['collected_frames']
        key_parts.append(f"{key}:{collected}")
    print(f"  collected -> {', '.join(key_parts)}")
    if summary['errors']:
        print(f"  errors -> {summary['errors']}")


def collect_bag_files(input_path: Path, include_fixed: bool) -> List[Path]:
    if input_path.is_file() and input_path.suffix == '.bag':
        return [input_path]

    if not input_path.is_dir():
        return []

    files = sorted(input_path.glob('*.bag'))
    if include_fixed:
        return files

    return [f for f in files if not f.name.startswith('fixed_')]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Read first joint frames from ROS bag(s) and compute initialization statistics.'
    )
    parser.add_argument('input', type=str, help='Input .bag file or directory containing .bag files')
    parser.add_argument('--batch', '-b', action='store_true', help='Batch mode for directory input')
    parser.add_argument('--frames', '-n', type=int, default=5, help='Number of initial frames to use per key')
    parser.add_argument('--include-action', action='store_true', help='Include action topics in addition to feedback topics')
    parser.add_argument('--include-fixed', action='store_true', help='Include fixed_*.bag in batch mode')
    parser.add_argument('--verbose-per-bag', action='store_true', help='Print detailed per-bag values in batch mode')
    parser.add_argument('--output-json', type=str, default='', help='Optional path to save JSON results')
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    requested_keys = list(DEFAULT_KEYS)
    if args.include_action:
        requested_keys.extend(ACTION_KEYS)

    typestore = get_typestore(Stores.ROS1_NOETIC)

    if input_path.is_dir() or args.batch:
        bag_files = collect_bag_files(input_path, include_fixed=args.include_fixed)
        if not bag_files:
            print(f'No .bag files found in: {input_path}')
            return

        print(f'Found {len(bag_files)} bag files in {input_path}')

        bag_summaries: List[Dict] = []
        all_results: List[Dict] = []

        for idx, bag_path in enumerate(bag_files, start=1):
            print(f"\n[{idx}/{len(bag_files)}] {bag_path.name}")
            extract_result = extract_first_n_frames(bag_path, requested_keys, args.frames, typestore)
            summary = build_bag_summary(extract_result, args.frames)
            if args.verbose_per_bag:
                print_single_summary(summary)
            else:
                print_batch_bag_brief(summary)
            bag_summaries.append(summary)
            all_results.append(extract_result)

        batch_summary = summarize_batch(all_results, requested_keys, args.frames)
        print_batch_summary(batch_summary)

        if args.output_json:
            payload = {
                'mode': 'batch',
                'requested_keys': requested_keys,
                'bag_summaries': bag_summaries,
                'batch_summary': batch_summary,
            }
            out = Path(args.output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open('w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\nSaved JSON: {out}")
        return

    if not (input_path.is_file() and input_path.suffix == '.bag'):
        print(f'Input is neither a .bag file nor a directory: {input_path}')
        return

    extract_result = extract_first_n_frames(input_path, requested_keys, args.frames, typestore)
    summary = build_bag_summary(extract_result, args.frames)
    print_single_summary(summary)

    if args.output_json:
        payload = {
            'mode': 'single',
            'requested_keys': requested_keys,
            'bag_summary': summary,
        }
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON: {out}")


if __name__ == '__main__':
    main()
