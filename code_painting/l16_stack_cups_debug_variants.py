#!/usr/bin/env python3
"""Run stack_cups Stage-1 mask debug variants for a small set of IDs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio
import imageio.v2 as iio
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert


def add_sam2_paths(sam2_root: Path) -> None:
    sys.path.insert(0, str(sam2_root))
    sys.path.insert(0, str(sam2_root / "Grounded_SAM_2"))


@dataclass(frozen=True)
class DinoPrompt:
    text: str
    box_threshold: float
    text_threshold: float
    dilate: int


class StackCupDebugRunner:
    def __init__(self, sam2_root: Path, device: str, sttn_ckpt: Path):
        add_sam2_paths(sam2_root)

        from grounding_dino.groundingdino.util.inference import load_model
        from sam2.build_sam import build_sam2_video_predictor
        from sttn_video_inpaint import build_sttn_model

        grounded = sam2_root / "Grounded_SAM_2"
        self.sam2_root = sam2_root
        self.device = device
        self.grounding_model = load_model(
            model_config_path=str(
                grounded
                / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            ),
            model_checkpoint_path=str(
                grounded / "gdino_checkpoints/groundingdino_swint_ogc.pth"
            ),
            device=device,
        )
        self.video_predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            str(grounded / "checkpoints/sam2.1_hiera_large.pt"),
            device=device,
        )
        self.inpainter = build_sttn_model(model_type="sttn", ckpt_p=str(sttn_ckpt))

    def dino_boxes(self, first_frame: Path, prompt: DinoPrompt) -> tuple[np.ndarray, list]:
        from grounding_dino.groundingdino.util.inference import load_image, predict

        image_source, image_dino = load_image(str(first_frame))
        h, w, _ = image_source.shape
        boxes, confs, labels = predict(
            model=self.grounding_model,
            image=image_dino,
            caption=prompt.text,
            box_threshold=prompt.box_threshold,
            text_threshold=prompt.text_threshold,
        )
        if boxes.numel() == 0:
            return np.zeros((0, 4), dtype=np.float32), []
        boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        boxes_xyxy = box_convert(boxes, "cxcywh", "xyxy").cpu().numpy().astype(np.float32)
        return boxes_xyxy, [str(x) for x in labels]

    def masks_from_boxes(
        self, frames: list[np.ndarray], boxes_xyxy: np.ndarray, tmp_root: Path
    ) -> list[np.ndarray]:
        if len(boxes_xyxy) == 0:
            return [np.zeros(frame.shape[:2], dtype=bool) for frame in frames]
        tmp_dir = self.write_tmp_frames(frames, tmp_root)
        try:
            state = self.video_predictor.init_state(video_path=str(tmp_dir))
            for obj_id, box in enumerate(boxes_xyxy, start=1):
                self.video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=box,
                )
            masks = self.propagate_union_masks(state, frames)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return masks

    def masks_from_points(
        self,
        frames: list[np.ndarray],
        point_objects: list[tuple[list[tuple[float, float]], list[int]]],
        tmp_root: Path,
    ) -> list[np.ndarray]:
        tmp_dir = self.write_tmp_frames(frames, tmp_root)
        try:
            state = self.video_predictor.init_state(video_path=str(tmp_dir))
            for obj_id, (points, labels) in enumerate(point_objects, start=1):
                self.video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=np.asarray(points, dtype=np.float32),
                    labels=np.asarray(labels, dtype=np.int32),
                )
            masks = self.propagate_union_masks(state, frames)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return masks

    def propagate_union_masks(self, state, frames: list[np.ndarray]) -> list[np.ndarray]:
        out = [np.zeros(frame.shape[:2], dtype=bool) for frame in frames]
        for frame_idx, obj_ids, mask_logits in self.video_predictor.propagate_in_video(state):
            if frame_idx >= len(out):
                continue
            parts = [(mask_logits[i] > 0).cpu().numpy().squeeze() for i in range(len(obj_ids))]
            if parts:
                out[frame_idx] = np.any(parts, axis=0)
        return out

    @staticmethod
    def write_tmp_frames(frames: list[np.ndarray], tmp_root: Path) -> Path:
        tmp_dir = Path(tempfile.mkdtemp(prefix="stack_sam2_frames_", dir=tmp_root))
        for idx, frame in enumerate(frames):
            cv2.imwrite(
                str(tmp_dir / f"{idx:05d}.jpg"),
                cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
        return tmp_dir

    def inpaint(self, frames: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
        from sttn_video_inpaint import inpaint_video_with_builded_sttn

        pil_frames = [Image.fromarray(frame[:, :, :3]) for frame in frames]
        pil_masks = [Image.fromarray((mask.astype(np.uint8) * 255)) for mask in masks]
        return [
            np.asarray(frame)
            for frame in inpaint_video_with_builded_sttn(
                self.inpainter, pil_frames, pil_masks, device=self.device
            )
        ]


def read_video(video_path: Path, max_frames: int) -> tuple[list[np.ndarray], float]:
    frames = iio.mimread(video_path, memtest=False)
    fps = imageio.v3.immeta(video_path, exclude_applied=False).get("fps", 5)
    return [frame[:, :, :3] for frame in frames[:max_frames]], float(fps)


def dilate_masks(masks: list[np.ndarray], kernel_size: int) -> list[np.ndarray]:
    if kernel_size <= 0:
        return [mask.astype(bool) for mask in masks]
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return [cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool) for mask in masks]


def hsv_green_masks(frames: list[np.ndarray], dilate: int) -> list[np.ndarray]:
    masks = []
    kernel_close = np.ones((5, 5), np.uint8)
    kernel_dilate = np.ones((dilate, dilate), np.uint8) if dilate > 0 else None
    for frame in frames:
        hsv = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([35, 40, 35]), np.array([88, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        if kernel_dilate is not None:
            mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        masks.append(mask.astype(bool))
    return masks


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    if not mask.any():
        return (0, 0, 0, 0)
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return (int(x), int(y), int(w), int(h))


def overlay_masks(frames: list[np.ndarray], masks: list[np.ndarray], color=(255, 0, 0)) -> list[np.ndarray]:
    out = []
    color_arr = np.asarray(color, dtype=np.float32)
    for frame, mask in zip(frames, masks):
        img = frame[:, :, :3].astype(np.float32).copy()
        img[mask] = img[mask] * 0.45 + color_arr * 0.55
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def box_video(frames: list[np.ndarray], masks: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for frame, mask in zip(frames, masks):
        img = frame[:, :, :3].copy()
        x, y, w, h = bbox_from_mask(mask)
        if w > 0 and h > 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        out.append(img)
    return out


def write_variant_outputs(
    out_dir: Path,
    video_stem: str,
    frames: list[np.ndarray],
    final_masks: list[np.ndarray],
    fps: float,
    inpainted: list[np.ndarray],
    summary: dict,
    remove_masks: list[np.ndarray] | None = None,
    protect_masks: list[np.ndarray] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_uint8 = [(mask.astype(np.uint8) * 255) for mask in final_masks]
    iio.mimwrite(out_dir / f"mask_{video_stem}.mp4", mask_uint8, fps=fps)
    iio.mimwrite(out_dir / f"w_mask_{video_stem}.mp4", overlay_masks(frames, final_masks), fps=fps)
    iio.mimwrite(out_dir / f"w_box_{video_stem}.mp4", box_video(frames, final_masks), fps=fps)
    iio.mimwrite(out_dir / f"removed_w_mask_{video_stem}.mp4", inpainted, fps=fps)
    if remove_masks is not None:
        iio.mimwrite(
            out_dir / f"remove_mask_{video_stem}.mp4",
            [(mask.astype(np.uint8) * 255) for mask in remove_masks],
            fps=fps,
        )
        iio.mimwrite(
            out_dir / f"w_remove_mask_{video_stem}.mp4",
            overlay_masks(frames, remove_masks, color=(255, 120, 0)),
            fps=fps,
        )
    if protect_masks is not None:
        iio.mimwrite(
            out_dir / f"protect_mask_{video_stem}.mp4",
            [(mask.astype(np.uint8) * 255) for mask in protect_masks],
            fps=fps,
        )
        iio.mimwrite(
            out_dir / f"w_protect_mask_{video_stem}.mp4",
            overlay_masks(frames, protect_masks, color=(0, 255, 0)),
            fps=fps,
        )
    (out_dir / "debug_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def stack_point_objects() -> list[tuple[list[tuple[float, float]], list[int]]]:
    green_negative = (306.0, 176.0)
    return [
        ([(150.0, 186.0), green_negative], [1, 0]),
        ([(426.0, 196.0), green_negative], [1, 0]),
        ([(88.0, 364.0), green_negative], [1, 0]),
        ([(544.0, 358.0), green_negative], [1, 0]),
    ]


def run_one_id(
    runner: StackCupDebugRunner,
    input_video: Path,
    output_root: Path,
    sample_id: int,
    max_frames: int,
    tmp_root: Path,
) -> None:
    frames, fps = read_video(input_video, max_frames)
    video_stem = input_video.stem
    first_frame = tmp_root / f"stack_first_{sample_id}.png"
    iio.imwrite(first_frame, frames[0])

    remove_prompt = DinoPrompt(
        "arms, hands, wrists, watch, left light pink cup, right dark red cup.",
        0.35,
        0.25,
        40,
    )
    protect_prompt = DinoPrompt("green cup.", 0.25, 0.20, 12)
    tight_prompt = DinoPrompt(
        "arms, hands, wrists, watch, light pink cup, dark red cup.",
        0.50,
        0.30,
        20,
    )

    remove_boxes, remove_labels = runner.dino_boxes(first_frame, remove_prompt)
    remove_masks_raw = runner.masks_from_boxes(frames, remove_boxes, tmp_root)
    remove_masks = dilate_masks(remove_masks_raw, remove_prompt.dilate)

    protect_boxes, protect_labels = runner.dino_boxes(first_frame, protect_prompt)
    protect_masks_raw = runner.masks_from_boxes(frames, protect_boxes, tmp_root)
    protect_masks = dilate_masks(protect_masks_raw, protect_prompt.dilate)

    variant_a = [r & ~p for r, p in zip(remove_masks, protect_masks)]
    write_variant_outputs(
        output_root / "A_protect_dino" / "stack_cups" / f"id_{sample_id}" / "stage1_human_inpaint",
        video_stem,
        frames,
        variant_a,
        fps,
        runner.inpaint(frames, variant_a),
        {
            "variant": "A_protect_dino",
            "remove_prompt": remove_prompt.__dict__,
            "remove_labels": remove_labels,
            "remove_box_count": int(len(remove_boxes)),
            "protect_prompt": protect_prompt.__dict__,
            "protect_labels": protect_labels,
            "protect_box_count": int(len(protect_boxes)),
            "final_rule": "remove_mask - protect_green_cup_mask",
        },
        remove_masks=remove_masks,
        protect_masks=protect_masks,
    )

    point_masks = dilate_masks(
        runner.masks_from_points(frames, stack_point_objects(), tmp_root), 30
    )
    write_variant_outputs(
        output_root / "B_points_negative" / "stack_cups" / f"id_{sample_id}" / "stage1_human_inpaint",
        video_stem,
        frames,
        point_masks,
        fps,
        runner.inpaint(frames, point_masks),
        {
            "variant": "B_points_negative",
            "point_objects": stack_point_objects(),
            "final_rule": "SAM2 video masks from positive red-cup/hand points with green-cup negative point",
        },
    )

    green_masks = hsv_green_masks(frames, dilate=16)
    variant_c = [r & ~g for r, g in zip(remove_masks, green_masks)]
    write_variant_outputs(
        output_root / "C_hsv_green_protect" / "stack_cups" / f"id_{sample_id}" / "stage1_human_inpaint",
        video_stem,
        frames,
        variant_c,
        fps,
        runner.inpaint(frames, variant_c),
        {
            "variant": "C_hsv_green_protect",
            "remove_prompt": remove_prompt.__dict__,
            "remove_labels": remove_labels,
            "remove_box_count": int(len(remove_boxes)),
            "green_hsv_range": {"lower": [35, 40, 35], "upper": [88, 255, 255], "dilate": 16},
            "final_rule": "remove_mask - HSV_green_mask",
        },
        remove_masks=remove_masks,
        protect_masks=green_masks,
    )

    tight_boxes, tight_labels = runner.dino_boxes(first_frame, tight_prompt)
    tight_masks_raw = runner.masks_from_boxes(frames, tight_boxes, tmp_root)
    tight_masks = dilate_masks(tight_masks_raw, tight_prompt.dilate)
    write_variant_outputs(
        output_root / "D_tight_dino" / "stack_cups" / f"id_{sample_id}" / "stage1_human_inpaint",
        video_stem,
        frames,
        tight_masks,
        fps,
        runner.inpaint(frames, tight_masks),
        {
            "variant": "D_tight_dino",
            "prompt": tight_prompt.__dict__,
            "labels": tight_labels,
            "box_count": int(len(tight_boxes)),
            "final_rule": "stricter DINO prompt and thresholds; no green protection",
        },
        remove_masks=tight_masks,
    )


def parse_ids(ids_text: str) -> list[int]:
    return [int(x) for x in ids_text.replace(",", " ").split() if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="0 1 2 3 4")
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--sam2_root", type=Path, default=Path("/home/zaijia001/ssd/inpainting_sam2_robot"))
    parser.add_argument("--sttn_ckpt", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=300)
    args = parser.parse_args()

    tmp_root = Path(tempfile.mkdtemp(prefix="stack_debug_variants_"))
    try:
        runner = StackCupDebugRunner(args.sam2_root, args.device, args.sttn_ckpt)
        for sample_id in parse_ids(args.ids):
            video = args.input_root / f"rgb_{sample_id}.mp4"
            if not video.exists():
                print(f"[skip] missing {video}")
                continue
            print(f"[run] stack_cups id={sample_id}")
            run_one_id(runner, video, args.output_root, sample_id, args.max_frames, tmp_root)
            print(f"[done] stack_cups id={sample_id}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
