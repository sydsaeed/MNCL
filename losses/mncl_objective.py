from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .bpr import BPRLoss
from .contrastive import MultiNegativeContrastiveLoss
from .regularization import l2_parameter_loss


@dataclass
class MNCLLossOutput:
    total_loss: Tensor
    bpr_loss: Tensor
    contrastive_loss: Tensor
    l2_loss: Tensor


class MNCLLoss(nn.Module):
    """Complete MNCL objective from Eq. 22."""

    def __init__(
        self,
        alpha: float,
        l2_lambda: float,
        temperature: float,
        omega: float,
        bpr_reduction: str = "mean",
    ) -> None:
        super().__init__()
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if l2_lambda < 0:
            raise ValueError("l2_lambda must be non-negative.")

        self.alpha = float(alpha)
        self.l2_lambda = float(l2_lambda)
        self.bpr = BPRLoss(reduction=bpr_reduction)
        self.contrastive = MultiNegativeContrastiveLoss(
            temperature=temperature,
            omega=omega,
        )

    @staticmethod
    def _select_rows(embeddings: Tensor, ids: Optional[Tensor]) -> Tensor:
        """Select contrastive nodes when IDs are provided."""
        if ids is None:
            return embeddings
        if ids.ndim != 1 or ids.dtype != torch.long:
            raise ValueError("contrastive IDs must be 1D torch.long tensors.")
        return embeddings[ids.to(embeddings.device)]

    def forward(
        self,
        model: nn.Module,
        positive_scores: Tensor,
        negative_scores: Tensor,
        views,
        contrastive_user_ids: Optional[Tensor] = None,
        contrastive_item_ids: Optional[Tensor] = None,
    ) -> MNCLLossOutput:
        """Return BPR, contrastive, L2, and total MNCL loss."""
        bpr_value = self.bpr(positive_scores, negative_scores)

        e_s_user = self._select_rows(views.e_s_user, contrastive_user_ids)
        e_m_user = self._select_rows(views.e_m_user, contrastive_user_ids)
        e_s_item = self._select_rows(views.e_s_item, contrastive_item_ids)
        e_m_item = self._select_rows(views.e_m_item, contrastive_item_ids)
        e_g_item = self._select_rows(views.e_g_item, contrastive_item_ids)

        contrastive_value = self.contrastive(
            e_s_user=e_s_user,
            e_m_user=e_m_user,
            e_s_item=e_s_item,
            e_m_item=e_m_item,
            e_g_item=e_g_item,
        )

        l2_value = l2_parameter_loss(model)
        total = (
            bpr_value
            + self.alpha * contrastive_value
            + self.l2_lambda * l2_value
        )

        return MNCLLossOutput(
            total_loss=total,
            bpr_loss=bpr_value,
            contrastive_loss=contrastive_value,
            l2_loss=l2_value,
        )
