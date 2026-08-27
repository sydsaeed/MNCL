from __future__ import annotations

import torch
from torch import Tensor, nn


def l2_parameter_loss(module: nn.Module) -> Tensor:
    """Return squared L2 norm of trainable parameters."""
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not trainable:
        return torch.tensor(0.0)

    device = trainable[0].device
    dtype = trainable[0].dtype
    total = torch.zeros((), device=device, dtype=dtype)

    for parameter in trainable:
        total = total + parameter.pow(2).sum()

    return total
