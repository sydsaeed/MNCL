from types import SimpleNamespace

import pytest
import torch

from models.mncl import (
    MNCLViewEmbeddings,
    fuse_view_embeddings,
    pair_scores,
)


def _views():
    return MNCLViewEmbeddings(
        e_s_user=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        e_s_item=torch.tensor([[2.0, 1.0], [1.0, 3.0]]),
        e_m_user=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        e_m_item=torch.tensor([[4.0, 2.0], [2.0, 5.0]]),
        e_g_item=torch.tensor([[3.0, 1.0], [6.0, 2.0]]),
    )


def test_fusion_matches_paper_order():
    views = _views()
    user, item = fuse_view_embeddings(views)

    expected_user = torch.cat(
        [views.e_s_user, views.e_s_user, views.e_m_user],
        dim=-1,
    )
    expected_item = torch.cat(
        [views.e_s_item, views.e_m_item, views.e_g_item],
        dim=-1,
    )

    assert torch.equal(user, expected_user)
    assert torch.equal(item, expected_item)
    assert user.shape == (2, 6)
    assert item.shape == (2, 6)


def test_pair_scores_match_dot_product():
    views = _views()
    user, item = fuse_view_embeddings(views)
    user_ids = torch.tensor([0, 1], dtype=torch.long)
    item_ids = torch.tensor([1, 0], dtype=torch.long)

    scores = pair_scores(user, item, user_ids, item_ids)
    expected = torch.tensor([
        torch.dot(user[0], item[1]),
        torch.dot(user[1], item[0]),
    ])

    assert torch.allclose(scores, expected)


def test_pair_scores_validates_ids():
    views = _views()
    user, item = fuse_view_embeddings(views)

    with pytest.raises(TypeError):
        pair_scores(
            user,
            item,
            torch.tensor([0.0]),
            torch.tensor([0], dtype=torch.long),
        )


def test_full_model_shapes_when_pyg_is_available():
    pytest.importorskip("torch_geometric")

    from torch_geometric.data import Data
    from models.mncl import MNCL

    info = SimpleNamespace(M=2, N=2, L=2, num_kg_nodes=4)
    config = SimpleNamespace(
        embedding_dim=8,
        L=2,
        beta=0.5,
        K=1,
        semantic_topk=2,
        semantic_chunk_size=2,
        path_hops=1,
        path_structural_keep_ratio=1.0,
        path_message_dropout_rate=0.0,
    )

    user_item = Data(
        edge_index=torch.tensor(
            [[0, 1, 2, 3], [2, 3, 0, 1]],
            dtype=torch.long,
        ),
        num_nodes=4,
    )

    kg = Data(
        edge_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        edge_type=torch.tensor([0, 1], dtype=torch.long),
        num_nodes=4,
    )
    kg.bidirectional_edge_index = torch.tensor(
        [[0, 1, 2, 3], [2, 3, 0, 1]],
        dtype=torch.long,
    )
    kg.bidirectional_edge_type = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    uie = Data(num_nodes=6)
    uie.interaction_user_index = torch.tensor([0, 1], dtype=torch.long)
    uie.interaction_item_index = torch.tensor([0, 1], dtype=torch.long)
    uie.kg_local_edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    uie.kg_local_edge_type = torch.tensor([0, 1], dtype=torch.long)

    graphs = SimpleNamespace(
        user_item=user_item,
        kg=kg,
        user_item_entity=uie,
    )

    model = MNCL(info, config)
    model.eval()
    out = model(
        graphs,
        user_ids=torch.tensor([0, 1], dtype=torch.long),
        item_ids=torch.tensor([0, 1], dtype=torch.long),
        add_noise=False,
        structural_dropout=False,
        message_dropout=False,
    )

    assert out.user_embeddings.shape == (2, 24)
    assert out.item_embeddings.shape == (2, 24)
    assert out.scores.shape == (2,)
    assert out.views.e_s_user.shape == (2, 8)
    assert out.views.e_s_item.shape == (2, 8)
    assert out.views.e_m_user.shape == (2, 8)
    assert out.views.e_m_item.shape == (2, 8)
    assert out.views.e_g_item.shape == (2, 8)
