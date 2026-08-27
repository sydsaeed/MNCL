import pytest
import torch

pytest.importorskip("torch_geometric")

from models.relation_gnn import RelationAwareGNN


def _toy_kg():
    edge_index = torch.tensor(
        [
            [0, 2, 0, 3, 1, 3, 2, 4],
            [2, 0, 3, 0, 3, 1, 4, 2],
        ],
        dtype=torch.long,
    )
    edge_type = torch.tensor(
        [0, 0, 1, 1, 0, 0, 1, 1],
        dtype=torch.long,
    )
    return edge_index, edge_type


def test_relation_gnn_output_shape():
    torch.manual_seed(7)
    entity_embeddings = torch.randn(5, 8)
    edge_index, edge_type = _toy_kg()

    encoder = RelationAwareGNN(
        num_items=2,
        num_relations=2,
        embedding_dim=8,
        num_relation_layers=2,
        num_lightgcn_layers=2,
        semantic_topk=2,
        semantic_chunk_size=2,
    )
    out = encoder(entity_embeddings, edge_index, edge_type)

    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_relation_gnn_gradient():
    torch.manual_seed(7)
    entity_embeddings = torch.randn(5, 8, requires_grad=True)
    edge_index, edge_type = _toy_kg()

    encoder = RelationAwareGNN(
        num_items=2,
        num_relations=2,
        embedding_dim=8,
        num_relation_layers=1,
        num_lightgcn_layers=1,
        semantic_topk=2,
        semantic_chunk_size=2,
    )
    out = encoder(entity_embeddings, edge_index, edge_type)
    loss = out.square().mean()
    loss.backward()

    assert entity_embeddings.grad is not None
    assert encoder.relation_embeddings.weight.grad is not None
