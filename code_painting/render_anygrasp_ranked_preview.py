#!/usr/bin/env python3
"""Render raw AnyGrasp result images with arm-specific candidate numbering, without RoboTwin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate AnyGrasp result images directly, without RoboTwin replay.")
    parser.add_argument("--anygrasp_dir", type=Path, required=True, help="Per-video AnyGrasp result dir.")
    parser.add_argument("--hand_npz", type=Path, required=True, help="hand_detections_<id>.npz used to distinguish left/right orientation similarity.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save annotated preview images.")
    parser.add_argument("--frames", type=int, nargs="+", required=True, help="Source frame ids, e.g. 1 21.")
    parser.add_argument("--top_k", type=int, default=20, help="How many score-ranked candidates to annotate for each arm.")
    parser.add_argument("--font_scale", type=float, default=0.35)
    parser.add_argument("--line_height", type=int, default=18)
    parser.add_argument("--panel_width", type=int, default=340)
    return parser.parse_args()


def load_hand_data(hand_npz: Path) -> Dict[str, np.ndarray]:
    data = np.load(str(hand_npz), allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def load_grasp_json(anygrasp_dir: Path, frame: int) -> Dict:
    grasp_path = anygrasp_dir / "grasps" / f"grasp_{int(frame):06d}.json"
    if not grasp_path.is_file():
        raise FileNotFoundError(f"Missing AnyGrasp JSON: {grasp_path}")
    with grasp_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_vis_image(anygrasp_dir: Path, frame: int, width: int, height: int) -> np.ndarray:
    vis_path = anygrasp_dir / "vis" / f"grasp_result_{int(frame):06d}.png"
    if vis_path.is_file():
        image = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    return np.full((height, width, 3), 255, dtype=np.uint8)


def nearest_valid_hand_frame(frame: int, hand_valid: np.ndarray) -> Optional[int]:
    valid_indices = np.where(np.asarray(hand_valid, dtype=bool))[0]
    if valid_indices.size == 0:
        return None
    return int(valid_indices[np.argmin(np.abs(valid_indices - int(frame)))])


def rotation_distance_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rot_a = np.asarray(rot_a, dtype=np.float64).reshape(3, 3)
    rot_b = np.asarray(rot_b, dtype=np.float64).reshape(3, 3)
    trace_value = float(np.trace(rot_a.T @ rot_b))
    cos_theta = np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def project_camera_point(point_cam: np.ndarray, camera_info: Dict) -> Optional[Tuple[int, int]]:
    point_cam = np.asarray(point_cam, dtype=np.float64).reshape(3)
    z = float(point_cam[2])
    if not np.isfinite(z) or z <= 1e-6:
        return None
    fx = float(camera_info["fx"])
    fy = float(camera_info["fy"])
    cx = float(camera_info["cx"])
    cy = float(camera_info["cy"])
    u = int(round(fx * float(point_cam[0]) / z + cx))
    v = int(round(fy * float(point_cam[1]) / z + cy))
    width = int(camera_info["width"])
    height = int(camera_info["height"])
    if u < -32 or u >= width + 32 or v < -32 or v >= height + 32:
        return None
    return u, v


def draw_tiny_label(image: np.ndarray, text: str, xy: Tuple[int, int], color: Tuple[int, int, int], font_scale: float) -> None:
    cv2.putText(
        image,
        text,
        (int(xy[0]), int(xy[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(font_scale),
        color,
        1,
        cv2.LINE_AA,
    )


def build_arm_rank_table(
    grasps: Sequence[Dict],
    arm_name: str,
    hand_data: Dict[str, np.ndarray],
    frame: int,
) -> Tuple[List[Dict], int]:
    hand_valid = np.asarray(hand_data[f"{arm_name}_gripper_valid"], dtype=bool)
    hand_rotations = np.asarray(hand_data[f"{arm_name}_gripper_rotation_matrix"], dtype=np.float64)
    ref_frame = int(frame) if frame < hand_valid.shape[0] and bool(hand_valid[frame]) else nearest_valid_hand_frame(int(frame), hand_valid)
    if ref_frame is None:
        raise RuntimeError(f"No valid {arm_name} hand rotation exists for frame {frame}.")
    ref_rot = np.asarray(hand_rotations[ref_frame], dtype=np.float64)
    ranked: List[Dict] = []
    for candidate_idx, grasp in enumerate(grasps):
        ranked.append(
            {
                "candidate_idx": int(candidate_idx),
                "score": float(grasp["score"]),
                "translation": np.asarray(grasp["translation"], dtype=np.float64).reshape(3),
                "rotation_matrix": np.asarray(grasp["rotation_matrix"], dtype=np.float64).reshape(3, 3),
                "rotation_distance_deg": rotation_distance_deg(ref_rot, grasp["rotation_matrix"]),
                "width": float(grasp.get("width", 0.0)),
                "depth": float(grasp.get("depth", 0.0)),
            }
        )
    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["score_rank"] = int(rank)
    return ranked, int(ref_frame)


def annotate_arm_preview(
    base_image: np.ndarray,
    arm_name: str,
    ranked: Sequence[Dict],
    camera_info: Dict,
    top_k: int,
    font_scale: float,
    panel_width: int,
    line_height: int,
) -> np.ndarray:
    color = (255, 100, 0) if arm_name == "left" else (0, 140, 255)
    title = "LEFT" if arm_name == "left" else "RIGHT"
    image = base_image.copy()
    panel = np.full((image.shape[0], panel_width, 3), 255, dtype=np.uint8)
    draw_tiny_label(panel, f"{title} score rank", (10, 24), color, 0.55)
    draw_tiny_label(panel, "label=candidate_idx", (10, 44), (0, 0, 0), 0.38)

    capped = list(ranked if int(top_k) < 0 else ranked[: max(int(top_k), 0)])
    for item in capped:
        pixel = project_camera_point(item["translation"], camera_info)
        if pixel is None:
            continue
        offset = (4, -4) if arm_name == "left" else (4, 10)
        draw_tiny_label(
            image,
            str(int(item["candidate_idx"])),
            (pixel[0] + offset[0], pixel[1] + offset[1]),
            color,
            font_scale,
        )

    for row_idx, item in enumerate(capped, start=0):
        y = 70 + row_idx * int(line_height)
        if y >= panel.shape[0] - 8:
            break
        line = (
            f"#{int(item['score_rank']):02d} "
            f"idx={int(item['candidate_idx']):02d} "
            f"s={float(item['score']):.3f} "
            f"rot={float(item['rotation_distance_deg']):.1f}"
        )
        draw_tiny_label(panel, line, (10, y), color if row_idx < 3 else (0, 0, 0), 0.36)

    return np.concatenate([image, panel], axis=1)


def main() -> None:
    args = parse_args()
    args.anygrasp_dir = args.anygrasp_dir.resolve()
    args.hand_npz = args.hand_npz.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hand_data = load_hand_data(args.hand_npz)
    summary = []
    for frame in [int(v) for v in args.frames]:
        payload = load_grasp_json(args.anygrasp_dir, frame)
        camera_info = dict(payload["camera"])
        grasps = list(payload.get("grasps", []))
        base_image = load_vis_image(args.anygrasp_dir, frame, int(camera_info["width"]), int(camera_info["height"]))

        left_ranked, left_ref_frame = build_arm_rank_table(grasps, "left", hand_data, frame)
        right_ranked, right_ref_frame = build_arm_rank_table(grasps, "right", hand_data, frame)

        left_image = annotate_arm_preview(
            base_image=base_image,
            arm_name="left",
            ranked=left_ranked,
            camera_info=camera_info,
            top_k=int(args.top_k),
            font_scale=float(args.font_scale),
            panel_width=int(args.panel_width),
            line_height=int(args.line_height),
        )
        right_image = annotate_arm_preview(
            base_image=base_image,
            arm_name="right",
            ranked=right_ranked,
            camera_info=camera_info,
            top_k=int(args.top_k),
            font_scale=float(args.font_scale),
            panel_width=int(args.panel_width),
            line_height=int(args.line_height),
        )
        combined = np.concatenate([left_image, right_image], axis=0)

        left_path = args.output_dir / f"frame_{frame:06d}_left_score_rank.png"
        right_path = args.output_dir / f"frame_{frame:06d}_right_score_rank.png"
        combined_path = args.output_dir / f"frame_{frame:06d}_left_right_score_rank.png"
        if not cv2.imwrite(str(left_path), left_image):
            raise RuntimeError(f"Failed to write {left_path}")
        if not cv2.imwrite(str(right_path), right_image):
            raise RuntimeError(f"Failed to write {right_path}")
        if not cv2.imwrite(str(combined_path), combined):
            raise RuntimeError(f"Failed to write {combined_path}")

        summary.append(
            {
                "frame": int(frame),
                "left_reference_hand_frame": int(left_ref_frame),
                "right_reference_hand_frame": int(right_ref_frame),
                "left_image": str(left_path),
                "right_image": str(right_path),
                "combined_image": str(combined_path),
                "top_k": int(args.top_k),
            }
        )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"frames": summary}, f, indent=2)
    print(f"[done] wrote {len(summary)} frame previews to {args.output_dir}")


if __name__ == "__main__":
    main()
