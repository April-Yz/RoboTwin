"""Utilities for extracting and visualizing action-conditioned attention."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import cm


def summarize_action_patch_attention(
    last_layer_attention: Optional[torch.Tensor],
    all_actions_mask: torch.Tensor,
    *,
    num_patch_tokens: int,
    num_images: int,
    num_patches_per_image: int,
) -> Optional[dict]:
    """Reduce full last-layer attention to per-image patch heatmaps."""
    if last_layer_attention is None:
        return None

    if last_layer_attention.ndim == 4:
        # (B, num_heads, tgt, src)
        attention = last_layer_attention[0].float().mean(dim=0)
    elif last_layer_attention.ndim == 3:
        # (num_heads, tgt, src)
        attention = last_layer_attention.float().mean(dim=0)
    else:
        return None

    if all_actions_mask.ndim == 2:
        action_positions = torch.nonzero(all_actions_mask[0], as_tuple=False).squeeze(-1)
    else:
        action_positions = torch.nonzero(all_actions_mask, as_tuple=False).squeeze(-1)
    if action_positions.numel() == 0:
        return None

    # Patch tokens are inserted immediately after BOS.
    multimodal_action_positions = action_positions + num_patch_tokens
    action_to_source = attention[multimodal_action_positions].mean(dim=0)

    total_image_patch_tokens = num_images * num_patches_per_image
    patch_attention = action_to_source[1 : 1 + total_image_patch_tokens]
    if patch_attention.numel() != total_image_patch_tokens:
        return None

    grid_size = int(math.sqrt(num_patches_per_image))
    if grid_size * grid_size != num_patches_per_image:
        return None

    patch_attention = patch_attention.reshape(num_images, num_patches_per_image)
    patch_attention = patch_attention.cpu().numpy()
    per_image_maps = [patch_attention[idx].reshape(grid_size, grid_size) for idx in range(num_images)]

    return {
        "per_image_patch_maps": per_image_maps,
        "grid_size": grid_size,
        "global_max": float(max(float(attn_map.max()) for attn_map in per_image_maps)),
    }


def tensor_images_to_uint8_list(pixel_values: torch.Tensor) -> List[np.ndarray]:
    """Convert normalized CHW tensors to uint8 images for diagnostics."""
    images = []
    for image_tensor in pixel_values.detach().float().cpu():
        image = image_tensor.permute(1, 2, 0).numpy()
        image = image - image.min()
        denom = image.max()
        if denom > 1e-8:
            image = image / denom
        image = (image * 255.0).clip(0, 255).astype(np.uint8)
        images.append(image)
    return images


def build_attention_frame(
    images: Iterable[np.ndarray],
    patch_attention_maps: Iterable[np.ndarray],
    *,
    title: str,
) -> np.ndarray:
    """Render image overlays for patch attention maps."""
    image_list = [np.asarray(image, dtype=np.uint8) for image in images]
    attention_list = [np.asarray(attn_map, dtype=np.float32) for attn_map in patch_attention_maps]
    if not image_list or len(image_list) != len(attention_list):
        raise ValueError("Image list and attention list must have equal non-zero length.")

    global_max = max(float(attn_map.max()) for attn_map in attention_list)
    global_max = max(global_max, 1e-8)
    cmap = cm.get_cmap("inferno")

    tiles: List[Image.Image] = []
    labels = ["head", "left_wrist", "right_wrist"]
    for idx, (image, attention) in enumerate(zip(image_list, attention_list)):
        heat = np.clip(attention / global_max, 0.0, 1.0)
        heat_img = Image.fromarray((heat * 255.0).astype(np.uint8), mode="L").resize(
            (image.shape[1], image.shape[0]), resample=Image.Resampling.BILINEAR
        )
        heat_arr = np.asarray(heat_img, dtype=np.float32) / 255.0
        color_arr = (cmap(heat_arr)[..., :3] * 255.0).astype(np.uint8)

        base_arr = image.astype(np.float32)
        alpha = 0.55 * heat_arr[..., None]
        overlay_arr = (1.0 - alpha) * base_arr + alpha * color_arr.astype(np.float32)
        overlay_arr = overlay_arr.clip(0, 255).astype(np.uint8)

        tile = Image.fromarray(overlay_arr)
        draw = ImageDraw.Draw(tile)
        label = labels[idx] if idx < len(labels) else f"img{idx}"
        draw.rectangle((0, 0, 130, 20), fill=(0, 0, 0))
        draw.text((6, 4), label, fill=(255, 255, 255))
        tiles.append(tile)

    spacing = 8
    title_height = 28
    width = sum(tile.width for tile in tiles) + spacing * max(len(tiles) - 1, 0)
    height = max(tile.height for tile in tiles) + title_height
    canvas = Image.new("RGB", (width, height), color=(12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(255, 255, 255))

    x_offset = 0
    for tile in tiles:
        canvas.paste(tile, (x_offset, title_height))
        x_offset += tile.width + spacing

    return np.asarray(canvas, dtype=np.uint8)


def save_attention_frame(frame: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, frame)


def save_attention_video(frames: List[np.ndarray], output_path: Path, *, fps: int = 10) -> None:
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def render_and_save_attention_snapshot(
    images: Iterable[np.ndarray],
    patch_attention_maps: Iterable[np.ndarray],
    *,
    title: str,
    output_path: Path,
) -> None:
    frame = build_attention_frame(images, patch_attention_maps, title=title)
    save_attention_frame(frame, output_path)
