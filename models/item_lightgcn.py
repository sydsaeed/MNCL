from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ItemEntityLightGCNOutput:
    item_embeddings: Tensor
    entity_embeddings: Tensor
    edge_index: Tensor
    edge_weight: Tensor
    layer_embeddings: List[Tensor]


class ItemEntityLightGCN(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_layers: int,
        topk: int = 10,
        chunk_size: int = 1024,
    ) -> None:
        super().__init__()

        if num_items < 1:
            raise ValueError("num_items must be at least 1.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if topk < 1:
            raise ValueError("topk must be at least 1.")
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")

        self.num_items = num_items
        self.num_layers = num_layers
        self.topk = topk
        self.chunk_size = chunk_size

    @staticmethod
    def _validate_embeddings(entity_embeddings: Tensor) -> None:
        if entity_embeddings.dim() != 2:
            raise ValueError("entity_embeddings must be a 2D tensor.")
        if entity_embeddings.size(0) == 0:
            raise ValueError("entity_embeddings cannot be empty.")

    def build_semantic_graph(
        self,
        entity_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self._validate_embeddings(entity_embeddings)

        num_nodes = entity_embeddings.size(0)
        k = min(self.topk, num_nodes)

        source_parts = []
        target_parts = []
        weight_parts = []

        with torch.no_grad():
            context = F.normalize(
                entity_embeddings.detach(),
                p=2,
                dim=1,
            )

            for start in range(0, num_nodes, self.chunk_size):
                end = min(start + self.chunk_size, num_nodes)
                similarity = context[start:end] @ context.t()
                values, indices = torch.topk(similarity, k=k, dim=1)

                targets = torch.arange(
                    start,
                    end,
                    device=entity_embeddings.device,
                    dtype=torch.long,
                ).repeat_interleave(k)

                source_parts.append(indices.reshape(-1))
                target_parts.append(targets)
                weight_parts.append(values.reshape(-1))

        source = torch.cat(source_parts, dim=0)
        target = torch.cat(target_parts, dim=0)
        edge_weight = torch.cat(weight_parts, dim=0).to(entity_embeddings.dtype)
        edge_index = torch.stack([source, target], dim=0)

        return edge_index, edge_weight

    @staticmethod
    def _normalize_edge_weight(
        edge_index: Tensor,
        edge_weight: Tensor,
        num_nodes: int,
    ) -> Tensor:
        source, target = edge_index

        degree = torch.zeros(
            num_nodes,
            device=edge_weight.device,
            dtype=edge_weight.dtype,
        )
        degree.index_add_(0, target, edge_weight)

        inv_sqrt_degree = degree.clamp_min(1e-12).pow(-0.5)

        return (
            edge_weight
            * inv_sqrt_degree[target]
            * inv_sqrt_degree[source]
        )

    @staticmethod
    def _propagate(
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        source, target = edge_index
        messages = x[source] * edge_weight.unsqueeze(-1)

        out = torch.zeros_like(x)
        out.index_add_(0, target, messages)
        return out

    def forward(
        self,
        entity_embeddings: Tensor,
        return_details: bool = False,
    ):
        self._validate_embeddings(entity_embeddings)

        if self.num_items > entity_embeddings.size(0):
            raise ValueError("num_items exceeds the KG node count.")

        edge_index, edge_weight = self.build_semantic_graph(entity_embeddings)
        edge_weight = self._normalize_edge_weight(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=entity_embeddings.size(0),
        )

        x = entity_embeddings
        layer_embeddings = [x]

        for _ in range(self.num_layers):
            x = self._propagate(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            layer_embeddings.append(x)

        entity_output = torch.stack(layer_embeddings, dim=0).sum(dim=0)
        item_output = entity_output[: self.num_items]

        if return_details:
            return ItemEntityLightGCNOutput(
                item_embeddings=item_output,
                entity_embeddings=entity_output,
                edge_index=edge_index,
                edge_weight=edge_weight,
                layer_embeddings=layer_embeddings,
            )

        return item_output
