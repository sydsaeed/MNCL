from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def bpr_loss(
    positive_scores: Tensor,
    negative_scores: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Compute Eq. 21 BPR loss from positive and negative scores."""
    if positive_scores.shape != negative_scores.shape:
        raise ValueError("positive_scores and negative_scores must match.")
    if positive_scores.ndim != 1:
        raise ValueError("BPR scores must be 1D tensors.")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'.")

    losses = F.softplus(negative_scores - positive_scores)

    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    return losses


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking objective."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def forward(self, positive_scores: Tensor, negative_scores: Tensor) -> Tensor:
        """Return BPR loss for paired positive and negative scores."""
        return bpr_loss(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            reduction=self.reduction,
        )
