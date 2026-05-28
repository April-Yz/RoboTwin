"""
Convert SAM-repainted head-camera videos plus retargeted wrist videos into the
same intermediate episode HDF5 format used by pi0 processed_data.

Typical usage for the new head-cam repaint pipeline:
python scripts/process_repainted_headcam_with_wrist.py d_pour_blue "pour water" 48 \
  --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue \
  --head-dir-template 'id_{id}_head_cam_arm_gripper_cup_bottle_pad_target' \
  --head-video-name target_with_original_head_cam_plan.mp4 \
  --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue \
  --retarget-dir-template 'hand_detections_{id}' \
  --review-json /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue/video_review.json \
  --review-mode strict \
  --ignore-ids

Classic zed-repaint pipeline is also supported:
python scripts/process_repainted_headcam_with_wrist.py d_pour_blue "pour water" 48 \
  --head-root /home/zaijia001/ssd/inpainting_sam2_robot/results_repaint/d_pour_blue \
  --head-dir-template 'id_{id}' \
  --head-video-name final_repainted.mp4 \
  --retarget-root /home/zaijia001/ssd/RoboTwin/code_painting/output_hand_retarget_swap_red_blue_keep_green_no_offset_pool_clean/d_pour_blue \
  --retarget-dir-template 'hand_detections_{id}' \
  --ignore-ids
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation


WORLD_TARGETS_NAME = "world_targets_and_status.npz"
LEFT_WRIST_VIDEO_NAME = "left_wrist_replay.mp4"
RIGHT_WRIST_VIDEO_NAME = "right_wrist_replay.mp4"
ID_FROM_DIGITS_RE = re.compile(r"(\d+)")
DISCARD_STATUSES = {"reject", "discard", "bad"}


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


def parse_id_from_name(name: str) -> int | None:
    matches = ID_FROM_DIGITS_RE.findall(Path(name).stem)
    if not matches:
        return None
    return int(matches[-1])


def format_template(value: str, episode_id: int) -> str:
    return value.format(id=episode_id)


def discover_ids(root: Path, template: str) -> list[int]:
    ids: set[int] = set()
    for child in root.iterdir():
        if not child.is_dir():
            continue
        episode_id = parse_id_from_name(child.name)
        if episode_id is None:
            continue
        expected_name = template.format(id=episode_id)
        if child.name == expected_name:
            ids.add(episode_id)
    return sorted(ids)


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


def build_arm_state(
    world_targets: np.ndarray,
    gripper_value: np.ndarray,
    status: np.ndarray,
    label: str,
) -> np.ndarray:
    world_targets = np.asarray(world_targets, dtype=np.float64)
    gripper_value = np.asarray(gripper_value, dtype=np.float64).reshape(-1, 1)
    status = np.asarray(status).astype(str)
    quat_norm = np.linalg.norm(world_targets[:, 3:7], axis=1)
    valid_mask = (
        (status == "Success")
        & np.isfinite(world_targets).all(axis=1)
        & np.isfinite(gripper_value[:, 0])
        & (quat_norm > 1e-8)
    )
    if not valid_mask.any():
        status_counts = {item: int((status == item).sum()) for item in sorted(set(status.tolist()))}
        zero_quat_count = int((quat_norm <= 1e-8).sum())
        raise ValueError(f"{label}: no valid Success frames found; statuses={status_counts}, zero_quat_count={zero_quat_count}")

    state = np.zeros((world_targets.shape[0], 7), dtype=np.float64)
    state[valid_mask, :3] = world_targets[valid_mask, :3]
    state[valid_mask, 3:6] = quaternion_to_euler(world_targets[valid_mask, 3:7])
    state[valid_mask, 6:7] = gripper_value[valid_mask]
    return forward_fill_segments(state, valid_mask, label)


def build_pose_sequence(world_targets_path: Path) -> np.ndarray:
    data = np.load(world_targets_path, allow_pickle=True)

    left_world = np.asarray(data["left_world_targets"], dtype=np.float64)
    right_world = np.asarray(data["right_world_targets"], dtype=np.float64)
    left_gripper = np.asarray(data["left_gripper_value"], dtype=np.float64).reshape(-1, 1)
    right_gripper = np.asarray(data["right_gripper_value"], dtype=np.float64).reshape(-1, 1)
    left_status = np.asarray(data["left_plan_status"]).astype(str)
    right_status = np.asarray(data["right_plan_status"]).astype(str)

    left_filled = build_arm_state(left_world, left_gripper, left_status, "left arm")
    right_filled = build_arm_state(right_world, right_gripper, right_status, "right arm")
    return np.concatenate([left_filled, right_filled], axis=1).astype(np.float32)


def save_episode(
    save_dir: Path,
    episode_idx: int,
    source_episode_id: int,
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
        json.dump(
            {
                "instructions": [instructions],
                "source_episode_id": source_episode_id,
            },
            f,
            indent=2,
        )

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


def load_usable_ids_from_review_json(review_json: Path, review_mode: str = "strict") -> list[int]:
    with review_json.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    allow_ambiguous = review_mode == "include_ambiguous"

    if isinstance(payload, dict) and "videos" in payload and isinstance(payload["videos"], dict):
        usable_ids: list[int] = []
        for raw_id, item in payload["videos"].items():
            episode_id = parse_id_from_name(str(raw_id))
            if episode_id is None:
                continue
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            usable = item.get("usable")
            reviewed = item.get("reviewed")
            status = str(item.get("status", "")).lower()
            if status in DISCARD_STATUSES:
                continue
            if status:
                usable_ids.append(episode_id)
                continue
            is_yes = label == "y" or usable is True or (reviewed and str(label).lower() in {"usable", "yes", "y"})
            is_ambiguous = label == "m" or usable == "ambiguous" or str(label).lower() in {"m", "maybe", "ambiguous"}
            if is_yes or (allow_ambiguous and is_ambiguous):
                usable_ids.append(episode_id)
        return sorted(set(usable_ids))

    if isinstance(payload, dict):
        usable_ids = []
        for raw_id, usable in payload.items():
            try:
                episode_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if usable is True or (allow_ambiguous and usable == "ambiguous"):
                usable_ids.append(episode_id)
        return sorted(set(usable_ids))

    raise ValueError(f"Unsupported review JSON format: {review_json}")


def process_dataset(
    head_root: Path,
    head_dir_template: str,
    head_video_name: str,
    retarget_root: Path,
    retarget_dir_template: str,
    world_targets_name: str,
    left_wrist_video_name: str,
    right_wrist_video_name: str,
    save_dir: Path,
    instructions: str,
    episode_num: int,
    image_size: tuple[int, int],
    ignore_ids: set[int],
    explicit_ids: list[int] | None,
    review_json: Path | None,
    review_mode: str,
) -> int:
    reviewed_usable_ids = load_usable_ids_from_review_json(review_json, review_mode=review_mode) if review_json else None

    if explicit_ids:
        candidate_ids = explicit_ids
        if reviewed_usable_ids is not None:
            reviewed_set = set(reviewed_usable_ids)
            candidate_ids = [episode_id for episode_id in candidate_ids if episode_id in reviewed_set]
    elif reviewed_usable_ids is not None:
        candidate_ids = reviewed_usable_ids
    else:
        candidate_ids = discover_ids(head_root, head_dir_template)
        if not candidate_ids:
            raise FileNotFoundError(
                f"No episode directories matched template '{head_dir_template}' under {head_root}"
            )

    save_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for episode_id in candidate_ids:
        if processed >= episode_num:
            break
        if episode_id in ignore_ids:
            print(f"[skip] ignored episode id={episode_id}")
            continue

        head_episode_dir = head_root / format_template(head_dir_template, episode_id)
        retarget_episode_dir = retarget_root / format_template(retarget_dir_template, episode_id)
        world_targets_path = retarget_episode_dir / world_targets_name
        head_video_path = head_episode_dir / format_template(head_video_name, episode_id)
        left_wrist_path = retarget_episode_dir / left_wrist_video_name
        right_wrist_path = retarget_episode_dir / right_wrist_video_name

        missing = [
            path
            for path in [head_episode_dir, retarget_episode_dir, world_targets_path, head_video_path, left_wrist_path, right_wrist_path]
            if not path.exists()
        ]
        if missing:
            print(f"[skip] episode id={episode_id} missing paths:")
            for path in missing:
                print(f"       - {path}")
            continue

        print(f"[process] id={episode_id}")
        print(f"          head={head_video_path}")
        print(f"          left_wrist={left_wrist_path}")
        print(f"          right_wrist={right_wrist_path}")
        print(f"          world_targets={world_targets_path}")

        state_action_seq = build_pose_sequence(world_targets_path)
        cam_high_frames = load_required_video_frames(head_video_path, image_size=image_size)
        cam_left_wrist_frames = load_required_video_frames(left_wrist_path, image_size=image_size)
        cam_right_wrist_frames = load_required_video_frames(right_wrist_path, image_size=image_size)

        save_episode(
            save_dir=save_dir,
            episode_idx=processed,
            source_episode_id=episode_id,
            instructions=instructions,
            state_action_seq=state_action_seq,
            cam_high_frames=cam_high_frames,
            cam_left_wrist_frames=cam_left_wrist_frames,
            cam_right_wrist_frames=cam_right_wrist_frames,
        )
        processed += 1

    if processed == 0:
        raise RuntimeError("No usable episodes were processed. Check directory templates and video names.")
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process SAM-repainted head-camera videos and wrist replays into pi0 intermediate HDF5 episodes."
    )
    parser.add_argument("task_name", type=str, help="Task name used in processed_data/<task>-<num>.")
    parser.add_argument("instructions", type=str, help="Single instruction string for all episodes.")
    parser.add_argument("expert_data_num", type=int, help="Maximum number of episodes to process.")
    parser.add_argument("--head-root", type=Path, required=True, help="Root containing repaint episode directories.")
    parser.add_argument(
        "--head-dir-template",
        type=str,
        default="id_{id}",
        help="Episode directory naming template under --head-root. Example: id_{id} or id_{id}_head_cam_arm_gripper_cup_bottle_pad_target",
    )
    parser.add_argument(
        "--head-video-name",
        type=str,
        default="final_repainted.mp4",
        help="Head-camera video file inside each repaint episode directory.",
    )
    parser.add_argument(
        "--retarget-root",
        type=Path,
        required=True,
        help="Root containing retarget episode directories with world_targets_and_status.npz and wrist videos.",
    )
    parser.add_argument(
        "--retarget-dir-template",
        type=str,
        default="hand_detections_{id}",
        help="Episode directory naming template under --retarget-root.",
    )
    parser.add_argument("--world-targets-name", type=str, default=WORLD_TARGETS_NAME)
    parser.add_argument("--left-wrist-video-name", type=str, default=LEFT_WRIST_VIDEO_NAME)
    parser.add_argument("--right-wrist-video-name", type=str, default=RIGHT_WRIST_VIDEO_NAME)
    parser.add_argument(
        "--ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit episode ids. If omitted, ids are discovered from --head-root using --head-dir-template, or from --review-json when that flag is provided.",
    )
    parser.add_argument(
        "--review-json",
        type=Path,
        default=None,
        help="Optional review JSON produced by review_repaint_videos.py. When provided, only reviewed ids allowed by --review-mode are processed automatically.",
    )
    parser.add_argument(
        "--review-mode",
        type=str,
        choices=["strict", "include_ambiguous"],
        default="strict",
        help="How to use review-json labels. strict: only y/usable=true. include_ambiguous: also include m/ambiguous.",
    )
    parser.add_argument(
        "--ignore-ids",
        type=int,
        nargs="*",
        default=[],
        help="Episode ids to skip.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: processed_data/<task>-<expert_data_num>",
    )
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = args.output_dir or Path(f"processed_data/{args.task_name}-{args.expert_data_num}")
    processed = process_dataset(
        head_root=args.head_root,
        head_dir_template=args.head_dir_template,
        head_video_name=args.head_video_name,
        retarget_root=args.retarget_root,
        retarget_dir_template=args.retarget_dir_template,
        world_targets_name=args.world_targets_name,
        left_wrist_video_name=args.left_wrist_video_name,
        right_wrist_video_name=args.right_wrist_video_name,
        save_dir=save_dir,
        instructions=args.instructions,
        episode_num=args.expert_data_num,
        image_size=(args.image_width, args.image_height),
        ignore_ids=set(args.ignore_ids),
        explicit_ids=args.ids,
        review_json=args.review_json,
        review_mode=args.review_mode,
    )
    print(f"[done] processed episodes: {processed}")
    print(f"[done] output dir: {save_dir}")


if __name__ == "__main__":
    main()
