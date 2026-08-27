from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PathViewSample:
    user_index: Tensor
    item_index: Tensor
    kg_edge_index: Tensor
    kg_edge_type: Tensor


def _sample_count(size: int, keep_ratio: float) -> int:
    if size == 0 or keep_ratio <= 0:
        return 0
    return max(1, min(size, int(round(size * keep_ratio))))


def sample_aligned_tensors(
    tensors: tuple[Tensor, ...],
    keep_ratio: float,
) -> tuple[Tensor, ...]:
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in [0, 1].")
    if not tensors:
        raise ValueError("At least one tensor is required.")

    size = tensors[0].size(-1)
    if any(t.size(-1) != size for t in tensors):
        raise ValueError("Aligned tensors must have the same last dimension.")

    count = _sample_count(size, keep_ratio)
    if count == size:
        return tensors
    if count == 0:
        return tuple(t[..., :0] for t in tensors)

    device = tensors[0].device
    index = torch.randperm(size, device=device)[:count]
    return tuple(t.index_select(-1, index) for t in tensors)


class PathViewAugmentor:
    def __init__(self, keep_ratio: float = 0.5) -> None:
        if not 0.0 <= keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in [0, 1].")
        self.keep_ratio = keep_ratio

    def __call__(
        self,
        user_index: Tensor,
        item_index: Tensor,
        kg_edge_index: Tensor,
        kg_edge_type: Tensor,
        enabled: bool,
    ) -> PathViewSample:
        if not enabled or self.keep_ratio == 1.0:
            return PathViewSample(
                user_index=user_index,
                item_index=item_index,
                kg_edge_index=kg_edge_index,
                kg_edge_type=kg_edge_type,
            )

        sampled_users, sampled_items = sample_aligned_tensors(
            (user_index, item_index),
            keep_ratio=self.keep_ratio,
        )
        sampled_kg, sampled_type = sample_aligned_tensors(
            (kg_edge_index, kg_edge_type.unsqueeze(0)),
            keep_ratio=self.keep_ratio,
        )

        return PathViewSample(
            user_index=sampled_users,
            item_index=sampled_items,
            kg_edge_index=sampled_kg,
            kg_edge_type=sampled_type.squeeze(0),
        )
