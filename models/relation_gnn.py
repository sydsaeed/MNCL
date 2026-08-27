from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing

from .item_lightgcn import ItemEntityLightGCN, ItemEntityLightGCNOutput


@dataclass
class RelationAwareOutput:
    item_embeddings: Tensor
    entity_embeddings: Tensor
    relation_embeddings: Tensor
    relation_layers: List[Tensor]
    semantic_output: ItemEntityLightGCNOutput


class RelationAwareConv(MessagePassing):
    def __init__(self) -> None:
        super().__init__(aggr="mean", flow="source_to_target")

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

    def message(self, x_j: Tensor, edge_relation: Tensor) -> Tensor:
        return x_j * edge_relation


class RelationAwareGNN(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_relations: int,
        embedding_dim: int,
        num_relation_layers: int,
        num_lightgcn_layers: int,
        semantic_topk: int = 10,
        semantic_chunk_size: int = 1024,
        use_internal_relations: bool = True,
    ) -> None:
        super().__init__()

        if num_items < 1:
            raise ValueError("num_items must be at least 1.")
        if num_relations < 1:
            raise ValueError("num_relations must be at least 1.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1.")
        if num_relation_layers < 1:
            raise ValueError("num_relation_layers must be at least 1.")
        if num_lightgcn_layers < 1:
            raise ValueError("num_lightgcn_layers must be at least 1.")

        self.num_items = num_items
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim

        self.use_internal_relations = use_internal_relations
        if use_internal_relations:
            self.relation_embeddings = nn.Embedding(
                num_relations,
                embedding_dim,
            )
            nn.init.xavier_uniform_(self.relation_embeddings.weight)
        else:
            self.relation_embeddings = None

        self.relation_convs = nn.ModuleList(
            [RelationAwareConv() for _ in range(num_relation_layers)]
        )
        self.item_lightgcn = ItemEntityLightGCN(
            num_items=num_items,
            num_layers=num_lightgcn_layers,
            topk=semantic_topk,
            chunk_size=semantic_chunk_size,
        )

    def _validate_inputs(
        self,
        entity_embeddings: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> None:
        if entity_embeddings.dim() != 2:
            raise ValueError("entity_embeddings must be a 2D tensor.")
        if entity_embeddings.size(0) < self.num_items:
            raise ValueError("num_items exceeds the KG node count.")
        if entity_embeddings.size(1) != self.embedding_dim:
            raise ValueError("Embedding dimension does not match the model.")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        if edge_type.dim() != 1 or edge_type.size(0) != edge_index.size(1):
            raise ValueError("edge_type must match the number of edges.")
        if edge_index.dtype != torch.long or edge_type.dtype != torch.long:
            raise TypeError("edge_index and edge_type must use torch.long.")

        if edge_index.numel() > 0:
            if int(edge_index.min()) < 0:
                raise ValueError("edge_index contains a negative node ID.")
            if int(edge_index.max()) >= entity_embeddings.size(0):
                raise ValueError("edge_index contains an invalid node ID.")

        if edge_type.numel() > 0:
            if int(edge_type.min()) < 0:
                raise ValueError("edge_type contains a negative relation ID.")
            if int(edge_type.max()) >= self.num_relations:
                raise ValueError("edge_type contains an invalid relation ID.")

    def forward(
        self,
        entity_embeddings: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        relation_embeddings: Tensor | None = None,
        return_details: bool = False,
    ):
        self._validate_inputs(entity_embeddings, edge_index, edge_type)

        edge_index = edge_index.to(entity_embeddings.device)
        edge_type = edge_type.to(entity_embeddings.device)

        if relation_embeddings is None:
            if self.relation_embeddings is None:
                raise ValueError("relation_embeddings must be provided.")
            relation_embeddings = self.relation_embeddings.weight
        relation_embeddings = relation_embeddings.to(entity_embeddings.device)
        if relation_embeddings.shape != (self.num_relations, self.embedding_dim):
            raise ValueError("relation_embeddings has an invalid shape.")

        x = entity_embeddings
        relation_layers = [x]

        for conv in self.relation_convs:
            x = conv(
                x=x,
                edge_index=edge_index,
                edge_type=edge_type,
                relation_embeddings=relation_embeddings,
            )
            relation_layers.append(x)

        relation_output = torch.stack(relation_layers, dim=0).sum(dim=0)
        semantic_output = self.item_lightgcn(
            relation_output,
            return_details=True,
        )

        if return_details:
            return RelationAwareOutput(
                item_embeddings=semantic_output.item_embeddings,
                entity_embeddings=relation_output,
                relation_embeddings=relation_embeddings,
                relation_layers=relation_layers,
                semantic_output=semantic_output,
            )

        return semantic_output.item_embeddings
