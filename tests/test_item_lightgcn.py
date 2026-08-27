import torch

from models.item_lightgcn import ItemEntityLightGCN


def test_item_lightgcn_output_shape():
    torch.manual_seed(7)
    x = torch.randn(8, 6)

    encoder = ItemEntityLightGCN(
        num_items=3,
        num_layers=2,
        topk=3,
        chunk_size=4,
    )
    out = encoder(x)

    assert out.shape == (3, 6)
    assert torch.isfinite(out).all()


def test_semantic_graph_shape():
    torch.manual_seed(7)
    x = torch.randn(7, 4)

    encoder = ItemEntityLightGCN(
        num_items=2,
        num_layers=1,
        topk=3,
        chunk_size=3,
    )
    edge_index, edge_weight = encoder.build_semantic_graph(x)

    assert edge_index.shape == (2, 21)
    assert edge_weight.shape == (21,)
    assert torch.isfinite(edge_weight).all()


def test_item_lightgcn_gradient():
    torch.manual_seed(7)
    x = torch.randn(8, 6, requires_grad=True)

    encoder = ItemEntityLightGCN(
        num_items=3,
        num_layers=2,
        topk=3,
        chunk_size=4,
    )
    out = encoder(x)
    loss = out.square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
