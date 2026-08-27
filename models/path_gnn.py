from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from .path_augmentation import PathViewAugmentor


@dataclass
class PathAwareOutput:
    user_embeddings: Tensor
    item_embeddings: Tensor
    entity_embeddings: Tensor
    relation_embeddings: Tensor
    user_layers: List[Tensor]
    entity_layers: List[Tensor]
    sampled_interactions: int
    sampled_kg_edges: int


class PathAwareKGConv(MessagePassing):
    def __init__(self) -> None:
        super().__init__(aggr="add", flow="source_to_target")

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        relation_embeddings: Tensor,
    ) -> Tensor:
        edge_relation = relation_embeddings[edge_type]
        return self.propagate(
            edge_index=edge_index,
            x=x,
            edge_relation=edge_relation,
            size=(x.size(0), x.size(0)),
        )

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_relation: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        # Relation-aware attention.
        query = torch.cat([x_i, edge_relation], dim=-1)
        key = torch.cat([x_j, edge_relation], dim=-1)
        logits = (query * key).sum(dim=-1)
        alpha = softmax(logits, index, ptr, size_i)

        message = x_j * edge_relation
        return message * alpha.unsqueeze(-1)


class UserItemMeanConv(MessagePassing):
    def __init__(self) -> None:
        super().__init__(aggr="mean", flow="source_to_target")

    def forward(
        self,
        user_embeddings: Tensor,
        entity_embeddings: Tensor,
        user_index: Tensor,
        item_index: Tensor,
    ) -> Tensor:
        edge_index = torch.stack([item_index, user_index], dim=0)
        return self.propagate(
            edge_index=edge_index,
            x=(entity_embeddings, user_embeddings),
            size=(entity_embeddings.size(0), user_embeddings.size(0)),
        )

    def message(self, x_j: Tensor) -> Tensor:
        return x_j


class PathAwareGNN(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_relations: int,
        embedding_dim: int,
        num_hops: int = 2,
        structural_keep_ratio: float = 0.5,
        message_dropout_rate: float = 0.1,
        use_internal_relations: bool = True,
    ) -> None:
        super().__init__()

        if num_items < 1:
            raise ValueError("num_items must be at least 1.")
        if num_relations < 1:
            raise ValueError("num_relations must be at least 1.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1.")
        if num_hops < 1:
            raise ValueError("num_hops must be at least 1.")
        if not 0.0 <= structural_keep_ratio <= 1.0:
            raise ValueError("structural_keep_ratio must be in [0, 1].")
        if not 0.0 <= message_dropout_rate < 1.0:
            raise ValueError("message_dropout_rate must be in [0, 1).")

        self.num_items = num_items
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_hops = num_hops
        self.message_dropout_rate = message_dropout_rate

        self.use_internal_relations = use_internal_relations
        if use_internal_relations:
            self.relation_embeddings = nn.Embedding(
                num_relations,
                embedding_dim,
            )
            nn.init.xavier_uniform_(self.relation_embeddings.weight)
        else:
            self.relation_embeddings = None

        self.kg_convs = nn.ModuleList(
            [PathAwareKGConv() for _ in range(num_hops)]
        )
        self.user_convs = nn.ModuleList(
            [UserItemMeanConv() for _ in range(num_hops)]
        )
        self.augmentor = PathViewAugmentor(structural_keep_ratio)

    def _validate_inputs(
        self,
        user_embeddings: Tensor,
        entity_embeddings: Tensor,
        user_index: Tensor,
        item_index: Tensor,
        kg_edge_index: Tensor,
        kg_edge_type: Tensor,
    ) -> None:
        if user_embeddings.dim() != 2 or entity_embeddings.dim() != 2:
            raise ValueError("Embeddings must be 2D tensors.")
        if user_embeddings.size(1) != self.embedding_dim:
            raise ValueError("User embedding dimension does not match the model.")
        if entity_embeddings.size(1) != self.embedding_dim:
            raise ValueError("Entity embedding dimension does not match the model.")
        if entity_embeddings.size(0) < self.num_items:
            raise ValueError("num_items exceeds the KG node count.")
        if user_index.dim() != 1 or item_index.dim() != 1:
            raise ValueError("Interaction indices must be 1D tensors.")
        if user_index.size(0) != item_index.size(0):
            raise ValueError("Interaction indices must have equal lengths.")
        if kg_edge_index.dim() != 2 or kg_edge_index.size(0) != 2:
            raise ValueError("kg_edge_index must have shape [2, num_edges].")
        if kg_edge_type.dim() != 1:
            raise ValueError("kg_edge_type must be a 1D tensor.")
        if kg_edge_type.size(0) != kg_edge_index.size(1):
            raise ValueError("kg_edge_type must match the KG edge count.")

        tensors = [user_index, item_index, kg_edge_index, kg_edge_type]
        if any(t.dtype != torch.long for t in tensors):
            raise TypeError("Graph indices must use torch.long dtype.")

        if user_index.numel() > 0:
            if int(user_index.min()) < 0 or int(user_index.max()) >= user_embeddings.size(0):
                raise ValueError("user_index contains an invalid user ID.")
            if int(item_index.min()) < 0 or int(item_index.max()) >= self.num_items:
                raise ValueError("item_index contains an invalid item ID.")

        if kg_edge_index.numel() > 0:
            if int(kg_edge_index.min()) < 0:
                raise ValueError("kg_edge_index contains a negative node ID.")
            if int(kg_edge_index.max()) >= entity_embeddings.size(0):
                raise ValueError("kg_edge_index contains an invalid node ID.")

        if kg_edge_type.numel() > 0:
            if int(kg_edge_type.min()) < 0:
                raise ValueError("kg_edge_type contains a negative relation ID.")
            if int(kg_edge_type.max()) >= self.num_relations:
                raise ValueError("kg_edge_type contains an invalid relation ID.")

    @staticmethod
    def _make_bidirectional(
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> tuple[Tensor, Tensor]:
        reverse = edge_index.flip(0)
        return (
            torch.cat([edge_index, reverse], dim=1),
            torch.cat([edge_type, edge_type], dim=0),
        )

    def forward(
        self,
        user_embeddings: Tensor,
        entity_embeddings: Tensor,
        user_index: Tensor,
        item_index: Tensor,
        kg_edge_index: Tensor,
        kg_edge_type: Tensor,
        relation_embeddings: Optional[Tensor] = None,
        structural_dropout: Optional[bool] = None,
        message_dropout: Optional[bool] = None,
        return_details: bool = False,
    ):
        self._validate_inputs(
            user_embeddings,
            entity_embeddings,
            user_index,
            item_index,
            kg_edge_index,
            kg_edge_type,
        )

        if structural_dropout is None:
            structural_dropout = self.training
        if message_dropout is None:
            message_dropout = self.training

        device = user_embeddings.device
        entity_embeddings = entity_embeddings.to(device)
        if relation_embeddings is None:
            if self.relation_embeddings is None:
                raise ValueError("relation_embeddings must be provided.")
            relation_embeddings = self.relation_embeddings.weight
        relation_embeddings = relation_embeddings.to(device)
        if relation_embeddings.shape != (self.num_relations, self.embedding_dim):
            raise ValueError("relation_embeddings has an invalid shape.")
        user_index = user_index.to(device)
        item_index = item_index.to(device)
        kg_edge_index = kg_edge_index.to(device)
        kg_edge_type = kg_edge_type.to(device)

        sample = self.augmentor(
            user_index=user_index,
            item_index=item_index,
            kg_edge_index=kg_edge_index,
            kg_edge_type=kg_edge_type,
            enabled=structural_dropout,
        )
        bi_edge_index, bi_edge_type = self._make_bidirectional(
            sample.kg_edge_index,
            sample.kg_edge_type,
        )

        user_x = user_embeddings
        entity_x = entity_embeddings
        user_res = user_x
        entity_res = entity_x
        user_layers = [user_x]
        entity_layers = [entity_x]

        for kg_conv, user_conv in zip(self.kg_convs, self.user_convs):
            entity_next = kg_conv(
                x=entity_x,
                edge_index=bi_edge_index,
                edge_type=bi_edge_type,
                relation_embeddings=relation_embeddings,
            )
            user_next = user_conv(
                user_embeddings=user_x,
                entity_embeddings=entity_x,
                user_index=sample.user_index,
                item_index=sample.item_index,
            )

            if message_dropout and self.message_dropout_rate > 0:
                entity_next = F.dropout(
                    entity_next,
                    p=self.message_dropout_rate,
                    training=True,
                )
                user_next = F.dropout(
                    user_next,
                    p=self.message_dropout_rate,
                    training=True,
                )

            entity_next = F.normalize(entity_next, p=2, dim=-1)
            user_next = F.normalize(user_next, p=2, dim=-1)

            entity_res = entity_res + entity_next
            user_res = user_res + user_next

            entity_x = entity_next
            user_x = user_next
            entity_layers.append(entity_x)
            user_layers.append(user_x)

        item_output = entity_res[: self.num_items]

        if return_details:
            return PathAwareOutput(
                user_embeddings=user_res,
                item_embeddings=item_output,
                entity_embeddings=entity_res,
                relation_embeddings=relation_embeddings,
                user_layers=user_layers,
                entity_layers=entity_layers,
                sampled_interactions=sample.user_index.numel(),
                sampled_kg_edges=sample.kg_edge_type.numel(),
            )

        return user_res, item_output
