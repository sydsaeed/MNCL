from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class NoiseEnhancedLightGCN(nn.Module):
    def __init__(
        self,
        num_layers: int,
        beta: float,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if beta < 0:
            raise ValueError("beta must be non-negative.")

        self.num_layers = num_layers
        self.beta = beta

    @staticmethod
    def _validate_inputs(
        user_embeddings: Tensor,
        item_embeddings: Tensor,
        edge_index: Tensor,
    ) -> None:
        if user_embeddings.dim() != 2 or item_embeddings.dim() != 2:
            raise ValueError("Embeddings must be 2D tensors.")
        if user_embeddings.size(1) != item_embeddings.size(1):
            raise ValueError("User and item embedding dimensions must match.")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must use torch.long dtype.")

        num_nodes = user_embeddings.size(0) + item_embeddings.size(0)
        if edge_index.numel() > 0:
            min_node = int(edge_index.min().item())
            max_node = int(edge_index.max().item())
            if min_node < 0 or max_node >= num_nodes:
                raise ValueError("edge_index contains an invalid node ID.")

    @staticmethod
    def _normalized_edge_weight(
        edge_index: Tensor,
        num_nodes: int,
        dtype: torch.dtype,
    ) -> Tensor:
        src, dst = edge_index

        degree = torch.bincount(src, minlength=num_nodes).to(dtype=dtype)
        inv_sqrt_degree = degree.pow(-0.5)
        inv_sqrt_degree.masked_fill_(torch.isinf(inv_sqrt_degree), 0.0)

        return inv_sqrt_degree[src] * inv_sqrt_degree[dst]

    @staticmethod
    def _lightgcn_propagate(
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        src, dst = edge_index
        messages = x[src] * edge_weight.unsqueeze(-1)

        out = torch.zeros_like(x)
        out.index_add_(0, dst, messages)
        return out

    def _noise(self, x: Tensor) -> Tensor:
        gamma = torch.rand_like(x)
        gamma = F.normalize(gamma, p=2, dim=1)
        return torch.sign(x) * gamma * self.beta

    def forward(
        self,
        user_embeddings: Tensor,
        item_embeddings: Tensor,
        edge_index: Tensor,
        add_noise: Optional[bool] = None,
        return_layers: bool = False,
    ):
        self._validate_inputs(user_embeddings, item_embeddings, edge_index)

        if add_noise is None:
            add_noise = self.training

        num_users = user_embeddings.size(0)
        x = torch.cat([user_embeddings, item_embeddings], dim=0)

        edge_index = edge_index.to(device=x.device)
        edge_weight = self._normalized_edge_weight(
            edge_index=edge_index,
            num_nodes=x.size(0),
            dtype=x.dtype,
        )

        layer_embeddings = [x]

        for _ in range(self.num_layers):
            propagated = self._lightgcn_propagate(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )

            if add_noise and self.beta > 0:
                propagated = propagated + self._noise(x)

            x = propagated
            layer_embeddings.append(x)

        stacked = torch.stack(layer_embeddings, dim=0)
        final_embeddings = stacked.mean(dim=0)

        user_output = final_embeddings[:num_users]
        item_output = final_embeddings[num_users:]

        if return_layers:
            return user_output, item_output, stacked

        return user_output, item_output
