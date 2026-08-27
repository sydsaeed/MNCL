from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from .loader import DatasetBundle
from .splitter import RatingSplit


@dataclass
class GraphBundle:
    user_item: Data
    kg: Data
    user_item_entity: Data


def _positive_interactions(ratings: np.ndarray) -> np.ndarray:
    return ratings[ratings[:, 2] == 1]


def build_user_item_graph(
    train_ratings: np.ndarray,
    M: int,
    N: int,
) -> Data:
    positive = _positive_interactions(train_ratings)

    users = torch.as_tensor(positive[:, 0], dtype=torch.long)
    items = torch.as_tensor(positive[:, 1], dtype=torch.long) + M

    src = torch.cat([users, items])
    dst = torch.cat([items, users])
    edge_index = torch.stack([src, dst], dim=0)

    graph = Data(edge_index=edge_index, num_nodes=M + N)
    graph.M = M
    graph.N = N
    graph.item_offset = M
    return graph


def build_kg_graph(
    kg: np.ndarray,
    num_kg_nodes: int,
    L: int,
) -> Data:
    head = torch.as_tensor(kg[:, 0], dtype=torch.long)
    relation = torch.as_tensor(kg[:, 1], dtype=torch.long)
    tail = torch.as_tensor(kg[:, 2], dtype=torch.long)

    edge_index = torch.stack([head, tail], dim=0)
    reverse_edge_index = torch.stack([tail, head], dim=0)

    graph = Data(
        edge_index=edge_index,
        edge_type=relation,
        num_nodes=num_kg_nodes,
    )
    graph.bidirectional_edge_index = torch.cat(
        [edge_index, reverse_edge_index],
        dim=1,
    )
    graph.bidirectional_edge_type = torch.cat([relation, relation], dim=0)
    graph.L = L
    return graph


def build_user_item_entity_graph(
    train_ratings: np.ndarray,
    kg: np.ndarray,
    M: int,
    num_kg_nodes: int,
    L: int,
) -> Data:
    positive = _positive_interactions(train_ratings)

    users = torch.as_tensor(positive[:, 0], dtype=torch.long)
    items = torch.as_tensor(positive[:, 1], dtype=torch.long) + M

    ui_forward = torch.stack([users, items], dim=0)
    ui_reverse = torch.stack([items, users], dim=0)
    interaction_edge_index = torch.cat([ui_forward, ui_reverse], dim=1)

    head = torch.as_tensor(kg[:, 0], dtype=torch.long) + M
    relation = torch.as_tensor(kg[:, 1], dtype=torch.long)
    tail = torch.as_tensor(kg[:, 2], dtype=torch.long) + M

    kg_forward = torch.stack([head, tail], dim=0)
    kg_reverse = torch.stack([tail, head], dim=0)
    kg_edge_index = torch.cat([kg_forward, kg_reverse], dim=1)
    kg_edge_type = torch.cat([relation, relation], dim=0)

    edge_index = torch.cat([interaction_edge_index, kg_edge_index], dim=1)

    ui_edge_type = torch.full(
        (interaction_edge_index.size(1),),
        fill_value=L,
        dtype=torch.long,
    )
    edge_type = torch.cat([ui_edge_type, kg_edge_type], dim=0)

    graph = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=M + num_kg_nodes,
    )
    graph.interaction_edge_index = interaction_edge_index
    graph.interaction_user_index = users
    graph.interaction_item_index = torch.as_tensor(positive[:, 1], dtype=torch.long)
    graph.kg_edge_index = kg_edge_index
    graph.kg_edge_type = kg_edge_type
    graph.kg_local_edge_index = torch.stack([head - M, tail - M], dim=0)
    graph.kg_local_edge_type = relation
    graph.user_offset = 0
    graph.kg_offset = M
    graph.ui_relation_id = L
    return graph


def build_graphs(
    data: DatasetBundle,
    split: RatingSplit,
) -> GraphBundle:
    info = data.info

    user_item = build_user_item_graph(
        split.train_ratings,
        M=info.M,
        N=info.N,
    )

    kg_graph = build_kg_graph(
        data.kg,
        num_kg_nodes=info.num_kg_nodes,
        L=info.L,
    )

    user_item_entity = build_user_item_entity_graph(
        split.train_ratings,
        data.kg,
        M=info.M,
        num_kg_nodes=info.num_kg_nodes,
        L=info.L,
    )

    return GraphBundle(
        user_item=user_item,
        kg=kg_graph,
        user_item_entity=user_item_entity,
    )
