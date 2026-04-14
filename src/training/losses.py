"""
Loss functions for multilabel protein localization.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class WeightedBCEWithLogitsLoss(nn.Module):
    """Wrapper around nn.BCEWithLogitsLoss using class-wise pos_weight."""

    def __init__(self, pos_weight: Tensor) -> None:
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.loss_fn(logits, targets)


class FocalLoss(nn.Module):
    """
    Multi-label focal loss with logits input.
    Applies sigmoid internally.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        targets = targets.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * ((1.0 - pt).clamp(min=1e-8) ** self.gamma)
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_function(
    name: str,
    pos_weights: Optional[Tensor],
    device: torch.device | str,
) -> nn.Module:
    """
    Factory for training losses.
    name: "bce" or "focal"
    """
    key = name.strip().lower()
    device = torch.device(device)

    if key == "bce":
        if pos_weights is None:
            raise ValueError("pos_weights is required for 'bce'")
        return WeightedBCEWithLogitsLoss(pos_weights.to(device))
    if key == "focal":
        return FocalLoss()
    raise ValueError(f"Unknown loss '{name}'. Expected one of: bce, focal")
