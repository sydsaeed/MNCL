from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from datasets.graph_builder import GraphBundle


@dataclass
class MNCLViewEmbeddings:
    e_s_user: Tensor
    e_s_item: Tensor
    e_m_user: Tensor
    e_m_item: Tensor
    e_g_item: Tensor


@dataclass
class MNCLOutput:
    scores: Optional[Tensor]
    user_embeddings: Tensor
    item_embeddings: Tensor
    views: MNCLViewEmbeddings


def fuse_view_embeddings(views: MNCLViewEmbeddings) -> tuple[Tensor, Tensor]:
    """Fuse MNCL views following Eqs. 18-19."""
    user_embeddings = torch.cat(
        [views.e_s_user, views.e_s_user, views.e_m_user],
        dim=-1,
    )
    item_embeddings = torch.cat(
        [views.e_s_item, views.e_m_item, views.e_g_item],
        dim=-1,
    )
    return user_embeddings, item_embeddings


def pair_scores(
    user_embeddings: Tensor,
    item_embeddings: Tensor,
    user_ids: Tensor,
    item_ids: Tensor,
) -> Tensor:
    """Compute dot-product scores for user-item pairs."""
    if user_ids.ndim != 1 or item_ids.ndim != 1:
        raise ValueError("user_ids and item_ids must be 1D tensors.")
    if user_ids.shape != item_ids.shape:
        raise ValueError("user_ids and item_ids must have the same shape.")
    if user_ids.dtype != torch.long or item_ids.dtype != torch.long:
        raise TypeError("user_ids and item_ids must use torch.long dtype.")

    user_ids = user_ids.to(user_embeddings.device)
    item_ids = item_ids.to(item_embeddings.device)

    selected_users = user_embeddings[user_ids]
    selected_items = item_embeddings[item_ids]
    return (selected_users * selected_items).sum(dim=-1)


class MNCL(nn.Module):
    """Main MNCL model that connects the three graph views."""

    def __init__(self, info: Any, config: Any) -> None:
        super().__init__()

        if info.M < 1 or info.N < 1 or info.num_kg_nodes < info.N:
            raise ValueError("Invalid dataset sizes.")
        if config.embedding_dim < 1:
            raise ValueError("embedding_dim must be at least 1.")

        from .noise_lightgcn import NoiseEnhancedLightGCN
        from .path_gnn import PathAwareGNN
        from .relation_gnn import RelationAwareGNN

        self.M = int(info.M)
        self.N = int(info.N)
        self.num_kg_nodes = int(info.num_kg_nodes)
        self.L_rel = int(info.L)
        self.embedding_dim = int(config.embedding_dim)

        self.user_embeddings = nn.Embedding(self.M, self.embedding_dim)
        self.entity_embeddings = nn.Embedding(
            self.num_kg_nodes,
            self.embedding_dim,
        )
        self.relation_embeddings = nn.Embedding(
            self.L_rel,
            self.embedding_dim,
        )
        self.reset_parameters()

        self.noise_encoder = NoiseEnhancedLightGCN(
            num_layers=config.L,
            beta=config.beta,
        )
        self.relation_encoder = RelationAwareGNN(
            num_items=self.N,
            num_relations=self.L_rel,
            embedding_dim=self.embedding_dim,
            num_relation_layers=config.K,
            num_lightgcn_layers=config.K,
            semantic_topk=config.semantic_topk,
            semantic_chunk_size=config.semantic_chunk_size,
            use_internal_relations=False,
        )
        self.path_encoder = PathAwareGNN(
            num_items=self.N,
            num_relations=self.L_rel,
            embedding_dim=self.embedding_dim,
            num_hops=config.path_hops,
            structural_keep_ratio=config.path_structural_keep_ratio,
            message_dropout_rate=config.path_message_dropout_rate,
            use_internal_relations=False,
        )

    def reset_parameters(self) -> None:
        """Initialize trainable embeddings with Xavier."""
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    @property
    def item_embeddings(self) -> Tensor:
        """Return base item embeddings from the shared KG space."""
        return self.entity_embeddings.weight[: self.N]

    def encode_views(
        self,
        graphs: "GraphBundle",
        add_noise: Optional[bool] = None,
        structural_dropout: Optional[bool] = None,
        message_dropout: Optional[bool] = None,
    ) -> MNCLViewEmbeddings:
        """Encode the user-item, global, and KG views."""
        user_base = self.user_embeddings.weight
        entity_base = self.entity_embeddings.weight
        item_base = entity_base[: self.N]
        relation_base = self.relation_embeddings.weight

        e_s_user, e_s_item = self.noise_encoder(
            user_embeddings=user_base,
            item_embeddings=item_base,
            edge_index=graphs.user_item.edge_index,
            add_noise=add_noise,
        )

        e_g_item = self.relation_encoder(
            entity_embeddings=entity_base,
            edge_index=graphs.kg.bidirectional_edge_index,
            edge_type=graphs.kg.bidirectional_edge_type,
            relation_embeddings=relation_base,
        )

        e_m_user, e_m_item = self.path_encoder(
            user_embeddings=user_base,
            entity_embeddings=entity_base,
            user_index=graphs.user_item_entity.interaction_user_index,
            item_index=graphs.user_item_entity.interaction_item_index,
            kg_edge_index=graphs.user_item_entity.kg_local_edge_index,
            kg_edge_type=graphs.user_item_entity.kg_local_edge_type,
            relation_embeddings=relation_base,
            structural_dropout=structural_dropout,
            message_dropout=message_dropout,
        )

        return MNCLViewEmbeddings(
            e_s_user=e_s_user,
            e_s_item=e_s_item,
            e_m_user=e_m_user,
            e_m_item=e_m_item,
            e_g_item=e_g_item,
        )

    def get_final_embeddings(
        self,
        graphs: "GraphBundle",
        add_noise: Optional[bool] = None,
        structural_dropout: Optional[bool] = None,
        message_dropout: Optional[bool] = None,
    ) -> tuple[Tensor, Tensor, MNCLViewEmbeddings]:
        """Return fused user and item representations."""
        views = self.encode_views(
            graphs=graphs,
            add_noise=add_noise,
            structural_dropout=structural_dropout,
            message_dropout=message_dropout,
        )
        user_embeddings, item_embeddings = fuse_view_embeddings(views)
        return user_embeddings, item_embeddings, views

    def score(
        self,
        graphs: "GraphBundle",
        user_ids: Tensor,
        item_ids: Tensor,
        add_noise: Optional[bool] = None,
        structural_dropout: Optional[bool] = None,
        message_dropout: Optional[bool] = None,
    ) -> Tensor:
        """Return Eq. 20 scores for selected pairs."""
        user_embeddings, item_embeddings, _ = self.get_final_embeddings(
            graphs=graphs,
            add_noise=add_noise,
            structural_dropout=structural_dropout,
            message_dropout=message_dropout,
        )
        return pair_scores(
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            user_ids=user_ids,
            item_ids=item_ids,
        )

    def forward(
        self,
        graphs: "GraphBundle",
        user_ids: Optional[Tensor] = None,
        item_ids: Optional[Tensor] = None,
        add_noise: Optional[bool] = None,
        structural_dropout: Optional[bool] = None,
        message_dropout: Optional[bool] = None,
    ) -> MNCLOutput:
        """Encode all views and optionally score selected pairs."""
        user_embeddings, item_embeddings, views = self.get_final_embeddings(
            graphs=graphs,
            add_noise=add_noise,
            structural_dropout=structural_dropout,
            message_dropout=message_dropout,
        )

        scores = None
        if user_ids is not None or item_ids is not None:
            if user_ids is None or item_ids is None:
                raise ValueError("user_ids and item_ids must be provided together.")
            scores = pair_scores(
                user_embeddings=user_embeddings,
                item_embeddings=item_embeddings,
                user_ids=user_ids,
                item_ids=item_ids,
            )

        return MNCLOutput(
            scores=scores,
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            views=views,
        )
