from __future__ import annotations

from typing import Literal

import torch
from torch import nn

LossName = Literal["bce", "focal"]


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


def build_loss_fn(
    name: LossName = "bce",
    *,
    focal_gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
) -> nn.Module:
    """Build the FSD50K multilabel loss selected by training config."""

    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if name == "focal":
        return FocalBCEWithLogitsLoss(gamma=focal_gamma, pos_weight=pos_weight)

    raise ValueError(f"Unsupported loss name: {name}")
