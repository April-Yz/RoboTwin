from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_TOKEN_TO_CAMERA_KEYS = {
    "base_0_rgb": ("base_0_rgb", "cam_high"),
    "left_wrist_0_rgb": ("left_wrist_0_rgb", "cam_left_wrist"),
    "right_wrist_0_rgb": ("right_wrist_0_rgb", "cam_right_wrist"),
}


def _ensure_uint8_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape: {image.shape}")

    # Handle CHW images.
    if image.shape[0] in (1, 3) and image.shape[2] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _best_grid(num_tokens: int) -> tuple[int, int]:
    if num_tokens <= 0:
        return 1, 1

    best_r, best_c = 1, num_tokens
    best_score = abs(best_c - best_r)
    for r in range(1, int(np.sqrt(num_tokens)) + 1):
        c = int(np.ceil(num_tokens / r))
        score = abs(c - r)
        if score < best_score:
            best_r, best_c = r, c
            best_score = score
    return best_r, best_c


def _scores_to_heatmap(scores: np.ndarray, height: int, width: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return np.zeros((height, width), dtype=np.float32)

    rows, cols = _best_grid(scores.size)
    padded = np.zeros(rows * cols, dtype=np.float32)
    padded[:scores.size] = scores
    grid = padded.reshape(rows, cols)

    min_v, max_v = float(np.min(grid)), float(np.max(grid))
    if max_v - min_v > 1e-8:
        grid = (grid - min_v) / (max_v - min_v)
    else:
        grid = np.zeros_like(grid)

    heat = Image.fromarray(np.clip(grid * 255.0, 0, 255).astype(np.uint8))
    heat = heat.resize((width, height), resample=Image.BICUBIC)
    return np.asarray(heat, dtype=np.float32) / 255.0


def _jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    x = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    heat_color = (_jet_colormap(heatmap) * 255.0).astype(np.float32)
    rgb_float = rgb.astype(np.float32)
    out = rgb_float * (1.0 - alpha) + heat_color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _select_image(images: dict[str, np.ndarray], token_name: str) -> np.ndarray | None:
    candidates = _TOKEN_TO_CAMERA_KEYS.get(token_name, (token_name, ))
    for key in candidates:
        if key in images:
            return _ensure_uint8_hwc(images[key])
    return None


def save_attention_visualizations(
    images: dict[str, np.ndarray],
    attention: dict[str, Any],
    save_dir: str | Path,
    file_prefix: str,
    *,
    overlay_alpha: float = 0.45,
) -> list[str]:
    """Save per-camera attention heatmaps and raw attention metadata.

    Returns:
        Paths of written image files.
    """
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attn_scores = np.asarray(attention.get("attn_scores"), dtype=np.float32)
    if attn_scores.ndim != 3:
        raise ValueError(f"Expected attention shape [layers, query, key], got: {attn_scores.shape}")

    prefix_len = int(attention.get("prefix_len", attn_scores.shape[-1]))
    query_start = int(attention.get("action_query_start", 0))
    query_end = int(attention.get("action_query_end", attn_scores.shape[1]))

    num_queries = attn_scores.shape[1]
    query_start = max(0, min(query_start, max(0, num_queries - 1)))
    query_end = max(query_start + 1, min(query_end, num_queries))
    prefix_len = max(1, min(prefix_len, attn_scores.shape[-1]))

    # [key_len], averaged over layers and selected output tokens.
    token_scores = attn_scores[:, query_start:query_end, :prefix_len].mean(axis=(0, 1))

    np.save(output_dir / f"{file_prefix}_token_scores.npy", token_scores)

    summary = {
        "prefix_len": prefix_len,
        "query_range": [query_start, query_end],
        "segments": [],
    }

    written_images: list[str] = []
    panel_images: list[np.ndarray] = []

    for segment in attention.get("prefix_layout", []):
        if segment.get("type") != "image":
            continue

        token_name = str(segment["name"])
        start = int(segment["start"])
        end = int(segment["end"])

        start = max(0, min(start, prefix_len))
        end = max(start, min(end, prefix_len))
        segment_scores = token_scores[start:end]

        summary["segments"].append({
            "name": token_name,
            "start": start,
            "end": end,
            "mean_score": float(segment_scores.mean()) if segment_scores.size else 0.0,
        })

        image = _select_image(images, token_name)
        if image is None:
            continue

        heatmap = _scores_to_heatmap(segment_scores, image.shape[0], image.shape[1])
        overlay = _overlay_heatmap(image, heatmap, overlay_alpha)

        save_path = output_dir / f"{file_prefix}_{token_name}.png"
        Image.fromarray(overlay).save(save_path)
        written_images.append(str(save_path))
        panel_images.append(overlay)

    if panel_images:
        target_h = max(img.shape[0] for img in panel_images)
        resized = []
        for img in panel_images:
            target_w = int(img.shape[1] * target_h / img.shape[0])
            pil_img = Image.fromarray(img).resize((target_w, target_h), resample=Image.BILINEAR)
            resized.append(np.asarray(pil_img, dtype=np.uint8))
        panel = np.concatenate(resized, axis=1)
        panel_path = output_dir / f"{file_prefix}_panel.png"
        Image.fromarray(panel).save(panel_path)
        written_images.append(str(panel_path))

    with open(output_dir / f"{file_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    return written_images
