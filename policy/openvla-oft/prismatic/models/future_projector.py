"""Future observation projector used for privileged distillation."""

import torch
import torch.nn as nn


class FutureObservationProjector(nn.Module):
    """Pools projected future patch tokens into a single conditioning token."""

    def __init__(self, llm_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = llm_dim if hidden_dim is None else hidden_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
        )

    def forward(self, future_patch_embeddings: torch.Tensor, future_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            future_patch_embeddings: (B, T, P, D) projected patch embeddings for future frames.
            future_mask: (B, T) bool mask of valid future frames.
        Returns:
            Tensor of shape (B, 1, D) for teacher conditioning.
        """
        pooled_per_frame = future_patch_embeddings.mean(dim=2)

        if future_mask is not None:
            future_mask = future_mask.to(device=pooled_per_frame.device)
            weights = future_mask.to(dtype=pooled_per_frame.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            pooled = (pooled_per_frame * weights).sum(dim=1) / denom
        else:
            pooled = pooled_per_frame.mean(dim=1)

        return self.proj(pooled).unsqueeze(1)
