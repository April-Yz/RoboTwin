"""Loss helpers for privileged distillation."""

import torch
import torch.nn.functional as F


def compute_action_distill_loss(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    if loss_type != "mse":
        raise ValueError(f"Unsupported distillation loss type: {loss_type}")
    return F.mse_loss(student_actions, teacher_actions, reduction="mean")
