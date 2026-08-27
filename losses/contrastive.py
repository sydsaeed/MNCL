import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_negative_view_loss(
    anchor: torch.Tensor,
    contrast: torch.Tensor,
    temperature: float,
    omega: float,
) -> torch.Tensor:
    """Compute the MNCL multi-negative loss for one pair of views."""
    if anchor.ndim != 2 or contrast.ndim != 2:
        raise ValueError("anchor and contrast must be 2D tensors")
    if anchor.shape != contrast.shape:
        raise ValueError("anchor and contrast must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if omega < 0:
        raise ValueError("omega must be non-negative")

    anchor = F.normalize(anchor, p=2, dim=-1)
    contrast = F.normalize(contrast, p=2, dim=-1)

    same_logits = anchor @ anchor.T / temperature
    cross_logits = anchor @ contrast.T / temperature

    batch_size = anchor.size(0)
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=anchor.device)

    positive_logits = cross_logits.diagonal()
    same_negative_logits = same_logits.masked_fill(diagonal, float("-inf"))
    cross_negative_logits = cross_logits.masked_fill(diagonal, float("-inf"))

    if omega == 0:
        same_negative_logits = torch.full_like(same_negative_logits, float("-inf"))
    else:
        same_negative_logits = same_negative_logits + math.log(omega)

    same_logsumexp = torch.logsumexp(same_negative_logits, dim=1)
    cross_logsumexp = torch.logsumexp(cross_negative_logits, dim=1)

    denominator_terms = torch.stack(
        [positive_logits, same_logsumexp, cross_logsumexp],
        dim=1,
    )
    log_denominator = torch.logsumexp(denominator_terms, dim=1)
    return (log_denominator - positive_logits).mean()


class MultiNegativeContrastiveLoss(nn.Module):
    """MNCL contrastive objective from Eqs. 15-17."""

    def __init__(self, temperature: float, omega: float):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if omega < 0:
            raise ValueError("omega must be non-negative")

        self.temperature = float(temperature)
        self.omega = float(omega)

    def compute_components(
        self,
        e_s_user: torch.Tensor,
        e_m_user: torch.Tensor,
        e_s_item: torch.Tensor,
        e_m_item: torch.Tensor,
        e_g_item: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the three MNCL contrastive terms."""
        L_G_user = multi_negative_view_loss(
            e_s_user,
            e_m_user,
            self.temperature,
            self.omega,
        )
        L_G_item = multi_negative_view_loss(
            e_s_item,
            e_m_item,
            self.temperature,
            self.omega,
        )
        L_KG_item = multi_negative_view_loss(
            e_s_item,
            e_g_item,
            self.temperature,
            self.omega,
        )

        return {
            "L_G_user": L_G_user,
            "L_G_item": L_G_item,
            "L_KG_item": L_KG_item,
        }

    def forward(
        self,
        e_s_user: torch.Tensor,
        e_m_user: torch.Tensor,
        e_s_item: torch.Tensor,
        e_m_item: torch.Tensor,
        e_g_item: torch.Tensor,
    ) -> torch.Tensor:
        """Return the total contrastive loss."""
        components = self.compute_components(
            e_s_user=e_s_user,
            e_m_user=e_m_user,
            e_s_item=e_s_item,
            e_m_item=e_m_item,
            e_g_item=e_g_item,
        )
        return sum(components.values())
