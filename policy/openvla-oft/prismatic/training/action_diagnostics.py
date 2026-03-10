"""Utilities for lightweight action-path diagnostics during training and eval."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _flatten_actions(actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim == 1:
        return actions.reshape(1, -1)
    if actions.ndim == 2:
        return actions
    return actions.reshape(-1, actions.shape[-1])


def compute_action_stats(prefix: str, actions: torch.Tensor) -> Dict[str, float]:
    actions = actions.detach().float()
    flat_actions = _flatten_actions(actions)

    metrics = {
        f"{prefix}_mean": flat_actions.mean().item(),
        f"{prefix}_std": flat_actions.std(unbiased=False).item(),
        f"{prefix}_abs_mean": flat_actions.abs().mean().item(),
        f"{prefix}_abs_max": flat_actions.abs().max().item(),
    }

    dim_abs_mean = flat_actions.abs().mean(dim=0)
    for dim_idx, value in enumerate(dim_abs_mean):
        metrics[f"{prefix}_dim_abs_mean_{dim_idx}"] = value.item()

    return metrics


def compute_pairwise_action_metrics(prefix: str, lhs: torch.Tensor, rhs: torch.Tensor) -> Dict[str, float]:
    lhs = lhs.detach().float()
    rhs = rhs.detach().float()

    lhs_flat = lhs.reshape(lhs.shape[0], -1)
    rhs_flat = rhs.reshape(rhs.shape[0], -1)

    cosine_sim = F.cosine_similarity(lhs_flat, rhs_flat, dim=1).mean().item()
    return {
        f"{prefix}_l1": F.l1_loss(lhs, rhs).item(),
        f"{prefix}_mse": F.mse_loss(lhs, rhs).item(),
        f"{prefix}_cosine_sim": cosine_sim,
    }


def compute_curr_next_action_metrics(predicted_actions: torch.Tensor, ground_truth_actions: torch.Tensor) -> Dict[str, float]:
    predicted_actions = predicted_actions.detach().float()
    ground_truth_actions = ground_truth_actions.detach().float()

    metrics = {
        "curr_action_l1_loss": F.l1_loss(predicted_actions[:, 0], ground_truth_actions[:, 0]).item(),
        "curr_action_pred_abs_mean": predicted_actions[:, 0].abs().mean().item(),
        "curr_action_gt_abs_mean": ground_truth_actions[:, 0].abs().mean().item(),
    }

    if predicted_actions.shape[1] > 1:
        metrics.update(
            {
                "next_actions_l1_loss": F.l1_loss(predicted_actions[:, 1:], ground_truth_actions[:, 1:]).item(),
                "next_actions_pred_abs_mean": predicted_actions[:, 1:].abs().mean().item(),
                "next_actions_gt_abs_mean": ground_truth_actions[:, 1:].abs().mean().item(),
            }
        )
    else:
        metrics.update(
            {
                "next_actions_l1_loss": 0.0,
                "next_actions_pred_abs_mean": 0.0,
                "next_actions_gt_abs_mean": 0.0,
            }
        )

    return metrics


def compute_tensor_stats(prefix: str, tensor: torch.Tensor) -> Dict[str, float]:
    tensor = tensor.detach().float()
    flat_tensor = tensor.reshape(tensor.shape[0], -1)
    return {
        f"{prefix}_norm": flat_tensor.norm(dim=1).mean().item(),
        f"{prefix}_mean": flat_tensor.mean().item(),
        f"{prefix}_std": flat_tensor.std(unbiased=False).item(),
    }


def corrupt_future_pixels(future_pixel_values: torch.Tensor) -> torch.Tensor:
    if future_pixel_values.shape[0] > 1:
        return torch.roll(future_pixel_values, shifts=1, dims=0)
    return torch.zeros_like(future_pixel_values)
