import pytest
import torch

pytest.importorskip("torch_geometric")

from models.path_gnn import PathAwareGNN


def build_inputs():
    user_embeddings = torch.randn(3, 8, requires_grad=True)
    entity_embeddings = torch.randn(5, 8, requires_grad=True)
    user_index = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    item_index = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    kg_edge_index = torch.tensor(
        [[0, 1, 2, 3], [3, 3, 4, 4]],
        dtype=torch.long,
    )
    kg_edge_type = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    return (
        user_embeddings,
        entity_embeddings,
        user_index,
        item_index,
        kg_edge_index,
        kg_edge_type,
    )


def test_path_gnn_output_shapes():
    inputs = build_inputs()
    model = PathAwareGNN(
        num_items=3,
        num_relations=2,
        embedding_dim=8,
        num_hops=2,
        structural_keep_ratio=1.0,
        message_dropout_rate=0.0,
    )
    model.eval()

    user_out, item_out = model(*inputs)

    assert user_out.shape == (3, 8)
    assert item_out.shape == (3, 8)
    assert torch.isfinite(user_out).all()
    assert torch.isfinite(item_out).all()


def test_eval_is_deterministic():
    inputs = build_inputs()
    model = PathAwareGNN(3, 2, 8, structural_keep_ratio=0.5)
    model.eval()

    first = model(*inputs)
    second = model(*inputs)

    assert torch.allclose(first[0], second[0])
    assert torch.allclose(first[1], second[1])


def test_gradients_flow():
    inputs = build_inputs()
    model = PathAwareGNN(
        3,
        2,
        8,
        structural_keep_ratio=1.0,
        message_dropout_rate=0.0,
    )

    user_out, item_out = model(
        *inputs,
        structural_dropout=False,
        message_dropout=False,
    )
    loss = user_out.square().mean() + item_out.square().mean()
    loss.backward()

    assert inputs[0].grad is not None
    assert inputs[1].grad is not None
    assert model.relation_embeddings.weight.grad is not None
