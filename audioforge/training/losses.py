from __future__ import annotations

import torch
from torch import nn


def multilabel_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stable loss used by FSD50K multilabel classification."""
    if logits.shape != targets.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} != targets {tuple(targets.shape)}")
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


class FocalBCEWithLogitsLoss(nn.Module):
    """Optional imbalance-aware multilabel loss for controlled experiments."""

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor([]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.pos_weight if self.pos_weight.numel() else None
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=weight
        )
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        return ((1 - p_t).pow(self.gamma) * bce).mean()
