"""
Convert retargeted human data plus repainted videos into the same intermediate
episode HDF5 format produced by process_data_R1.py.

Example:
python scripts/process_data_retageted_human.py d_pour_blue "pour water" 48 \
  --repaint-dir /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue \
  --retarget-dir /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_debug/d_pour_blue \
  --ignore-ids 2 7 9

Notes:
1. This script expects world_targets_and_status.npz plus
   left_wrist_replay.mp4/right_wrist_replay.mp4 for each episode.
2. Output structure matches process_data_R1.py:
   processed_data/<task>-<num>/episode_x/instructions.json
   processed_data/<task>-<num>/episode_x/episode_x.hdf5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation


WORLD_TARGETS_NAME = "world_targets_and_status.npz"
LEFT_WRIST_VIDEO_NAME = "left_wrist_replay.mp4"
RIGHT_WRIST_VIDEO_NAME = "right_wrist_replay.mp4"
EPISODE_ID_RE = re.compile(r"(\d+)$")


def quaternion_to_euler(quat_wxyz: np.ndarray, order: str = "xyz") -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    if quat_wxyz.ndim == 1:
        quat_wxyz = quat_wxyz.reshape(1, 4)
        squeeze = True
    else:
        squeeze = False

    norms = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    quat_wxyz = quat_wxyz / np.clip(norms, 1e-8, None)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    euler = Rotation.from_quat(quat_xyzw).as_euler(order, degrees=False)
    return euler[0] if squeeze else euler


def images_encoding(imgs: list[np.ndarray]) -> tuple[list[bytes], int]:
    encoded = []
    max_len = 0
    for img in imgs:
        success, encoded_image = cv2.imencode(".jpg", img)
        if not success:
            raise RuntimeError("Failed to encode image as JPEG.")
        jpeg_data = encoded_image.tobytes()
        encoded.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    return encoded, max_len


def extract_episode_id(path: Path) -> int:
    match = EPISODE_ID_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse episode id from: {path}")
    return int(match.group(1))


def resolve_world_targets_path(episode_dir: Path, retarget_dir: Path | None) -> Path | None:
    episode_id = extract_episode_id(episode_dir)

    candidates: list[Path] = []
    if retarget_dir is not None:
        candidates.extend(
            [
                retarget_dir / f"hand_detections_{episode_id}" / WORLD_TARGETS_NAME,
                retarget_dir / f"id_{episode_id}" / WORLD_TARGETS_NAME,
            ]
        )

    meta_path = episode_dir / "pipeline_meta.json"
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        robot_video = meta.get("robot_video")
        if robot_video:
            candidates.append(Path(robot_video).with_name(WORLD_TARGETS_NAME))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_video_frames(video_path: Path, image_size: tuple[int, int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width, height = image_size
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (width, height))
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")
    return frames


def load_required_video_frames(video_path: Path, image_size: tuple[int, int]) -> list[np.ndarray]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Required video is missing: {video_path}")
    return load_video_frames(video_path, image_size=image_size)


def forward_fill_segments(values: np.ndarray, valid_mask: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"{label}: expected 2D array, got {values.shape}")
    if values.shape[0] != valid_mask.shape[0]:
        raise ValueError(f"{label}: values and mask length mismatch")
    if not valid_mask.any():
        raise ValueError(f"{label}: no valid frames found")

    first_valid = int(np.flatnonzero(valid_mask)[0])
    filled = values.copy()
    filled[:first_valid] = filled[first_valid]
    last = filled[first_valid].copy()
    for idx in range(first_valid, filled.shape[0]):
        if valid_mask[idx]:
            last = filled[idx].copy()
        else:
            filled[idx] = last
    return filled


def build_pose_sequence(world_targets_path: Path) -> np.ndarray:
    data = np.load(world_targets_path, allow_pickle=True)

    left_world = np.asarray(data["left_world_targets"], dtype=np.float64)
    right_world = np.asarray(data["right_world_targets"], dtype=np.float64)
    left_gripper = np.asarray(data["left_gripper_value"], dtype=np.float64).reshape(-1, 1)
    right_gripper = np.asarray(data["right_gripper_value"], dtype=np.float64).reshape(-1, 1)
    left_status = np.asarray(data["left_plan_status"]).astype(str)
    right_status = np.asarray(data["right_plan_status"]).astype(str)

    left_state = np.concatenate(
        [
            left_world[:, :3],
            quaternion_to_euler(left_world[:, 3:7]),
            left_gripper,
        ],
        axis=1,
    )
    right_state = np.concatenate(
        [
            right_world[:, :3],
            quaternion_to_euler(right_world[:, 3:7]),
            right_gripper,
        ],
        axis=1,
    )

    left_valid = (left_status == "Success") & np.isfinite(left_world).all(axis=1) & np.isfinite(left_gripper[:, 0])
    right_valid = (right_status == "Success") & np.isfinite(right_world).all(axis=1) & np.isfinite(right_gripper[:, 0])

    left_filled = forward_fill_segments(left_state, left_valid, "left arm")
    right_filled = forward_fill_segments(right_state, right_valid, "right arm")
    return np.concatenate([left_filled, right_filled], axis=1).astype(np.float32)


def save_episode(
    save_dir: Path,
    episode_idx: int,
    instructions: str,
    state_action_seq: np.ndarray,
    cam_high_frames: list[np.ndarray],
    cam_left_wrist_frames: list[np.ndarray],
    cam_right_wrist_frames: list[np.ndarray],
) -> None:
    if state_action_seq.shape[0] < 2:
        raise ValueError("Need at least 2 frames to build state/action pairs.")
    if len(cam_high_frames) < 2 or len(cam_left_wrist_frames) < 2 or len(cam_right_wrist_frames) < 2:
        raise ValueError("Need at least 2 frames for cam_high/cam_left_wrist/cam_right_wrist.")

    usable_len = min(
        state_action_seq.shape[0],
        len(cam_high_frames),
        len(cam_left_wrist_frames),
        len(cam_right_wrist_frames),
    )
    if usable_len < 2:
        raise ValueError("Not enough aligned frames after clipping.")

    state_action_seq = state_action_seq[:usable_len]
    cam_high_frames = cam_high_frames[:usable_len]
    cam_left_wrist_frames = cam_left_wrist_frames[:usable_len]
    cam_right_wrist_frames = cam_right_wrist_frames[:usable_len]

    states = state_action_seq[:-1]
    actions = state_action_seq[1:]
    cam_high = cam_high_frames[:-1]
    cam_left_wrist = cam_left_wrist_frames[:-1]
    cam_right_wrist = cam_right_wrist_frames[:-1]

    episode_dir = save_dir / f"episode_{episode_idx}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    with (episode_dir / "instructions.json").open("w", encoding="utf-8") as f:
        json.dump({"instructions": [instructions]}, f, indent=2)

    cam_high_enc, len_high = images_encoding(cam_high)
    cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
    cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
    with h5py.File(episode_dir / f"episode_{episode_idx}.hdf5", "w") as f:
        f.create_dataset("action", data=actions)
        obs = f.create_group("observations")
        obs.create_dataset("state", data=states)
        obs.create_dataset("left_arm_dim", data=np.full(len(actions), 7, dtype=np.int32))
        obs.create_dataset("right_arm_dim", data=np.full(len(actions), 7, dtype=np.int32))
        images = obs.create_group("images")
        images.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
        images.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")
        images.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")


def process_dataset(
    repaint_dir: Path,
    retarget_dir: Path | None,
    save_dir: Path,
    instructions: str,
    episode_num: int,
    video_name: str,
    image_size: tuple[int, int],
    ignore_ids: set[int],
) -> int:
    repaint_episodes = sorted((p for p in repaint_dir.iterdir() if p.is_dir()), key=extract_episode_id)
    if not repaint_episodes:
        raise FileNotFoundError(f"No episode directories found in {repaint_dir}")

    save_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    missing_world_targets: list[Path] = []
    for episode_dir in repaint_episodes:
        if processed >= episode_num:
            break

        episode_id = extract_episode_id(episode_dir)
        if episode_id in ignore_ids:
            print(f"[skip] ignored episode id={episode_id}: {episode_dir}")
            continue

        world_targets_path = resolve_world_targets_path(episode_dir, retarget_dir)
        if world_targets_path is None:
            missing_world_targets.append(episode_dir)
            print(f"[skip] missing {WORLD_TARGETS_NAME}: {episode_dir}")
            continue

        video_path = episode_dir / video_name
        if not video_path.is_file():
            print(f"[skip] missing video {video_name}: {episode_dir}")
            continue

        print(f"[process] episode_dir={episode_dir}")
        print(f"          world_targets={world_targets_path}")

        state_action_seq = build_pose_sequence(world_targets_path)
        cam_high_frames = load_required_video_frames(video_path, image_size=image_size)
        retarget_episode_dir = world_targets_path.parent
        cam_left_wrist_frames = load_required_video_frames(
            retarget_episode_dir / LEFT_WRIST_VIDEO_NAME,
            image_size=image_size,
        )
        cam_right_wrist_frames = load_required_video_frames(
            retarget_episode_dir / RIGHT_WRIST_VIDEO_NAME,
            image_size=image_size,
        )
        save_episode(
            save_dir=save_dir,
            episode_idx=processed,
            instructions=instructions,
            state_action_seq=state_action_seq,
            cam_high_frames=cam_high_frames,
            cam_left_wrist_frames=cam_left_wrist_frames,
            cam_right_wrist_frames=cam_right_wrist_frames,
        )
        processed += 1

    if processed == 0 and missing_world_targets:
        raise RuntimeError(
            "No usable episodes were processed. The repaint results exist, but "
            f"{WORLD_TARGETS_NAME} is missing. Re-run retargeting to make sure each "
            "episode keeps world_targets_and_status.npz and the corresponding wrist "
            "replay videos."
        )
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process retargeted human data into pi0 intermediate HDF5 episodes.")
    parser.add_argument("task_name", type=str, help="Task name used in processed_data/<task>-<num>.")
    parser.add_argument("instructions", type=str, help="Single instruction string for all episodes.")
    parser.add_argument("expert_data_num", type=int, help="Maximum number of episodes to process.")
    parser.add_argument(
        "--repaint-dir",
        type=Path,
        required=True,
        help="Task directory under results_repaint, e.g. .../results_repaint/d_pour_blue",
    )
    parser.add_argument(
        "--retarget-dir",
        type=Path,
        default=None,
        help="Optional task directory containing hand_detections_<id>/world_targets_and_status.npz",
    )
    parser.add_argument(
        "--video-name",
        type=str,
        default="final_repainted.mp4",
        help="Video file name inside each repaint episode directory.",
    )
    parser.add_argument(
        "--ignore-ids",
        type=int,
        nargs="*",
        required=True,
        help="Episode ids to skip. This flag is mandatory every run; pass '--ignore-ids' with no values for an empty list.",
    )
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(f"processed_data/{args.task_name}-{args.expert_data_num}")
    processed = process_dataset(
        repaint_dir=args.repaint_dir,
        retarget_dir=args.retarget_dir,
        save_dir=save_dir,
        instructions=args.instructions,
        episode_num=args.expert_data_num,
        video_name=args.video_name,
        image_size=(args.image_width, args.image_height),
        ignore_ids=set(args.ignore_ids),
    )
    print(f"[done] processed episodes: {processed}")
    print(f"[done] output dir: {save_dir}")


if __name__ == "__main__":
    main()
