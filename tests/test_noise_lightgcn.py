import torch

from models import NoiseEnhancedLightGCN


def make_toy_graph():
    # Users: 0, 1 | Items: 2, 3, 4
    src = torch.tensor([0, 0, 1, 2, 3, 4], dtype=torch.long)
    dst = torch.tensor([2, 3, 4, 0, 0, 1], dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def test_output_shapes():
    torch.manual_seed(7)

    users = torch.randn(2, 8)
    items = torch.randn(3, 8)
    edge_index = make_toy_graph()

    encoder = NoiseEnhancedLightGCN(num_layers=4, beta=1.5)
    encoder.eval()

    user_output, item_output, layers = encoder(
        users,
        items,
        edge_index,
        return_layers=True,
    )

    assert user_output.shape == (2, 8)
    assert item_output.shape == (3, 8)
    assert layers.shape == (5, 5, 8)
    assert torch.isfinite(layers).all()


def test_eval_is_deterministic():
    torch.manual_seed(7)

    users = torch.randn(2, 8)
    items = torch.randn(3, 8)
    edge_index = make_toy_graph()

    encoder = NoiseEnhancedLightGCN(num_layers=2, beta=1.5)
    encoder.eval()

    first = encoder(users, items, edge_index)[0]
    second = encoder(users, items, edge_index)[0]

    assert torch.allclose(first, second)


def test_training_noise_changes_output():
    torch.manual_seed(7)

    users = torch.randn(2, 8)
    items = torch.randn(3, 8)
    edge_index = make_toy_graph()

    encoder = NoiseEnhancedLightGCN(num_layers=2, beta=1.5)
    encoder.train()

    first = encoder(users, items, edge_index)[0]
    second = encoder(users, items, edge_index)[0]

    assert not torch.allclose(first, second)


def test_beta_zero_disables_noise_effect():
    torch.manual_seed(7)

    users = torch.randn(2, 8)
    items = torch.randn(3, 8)
    edge_index = make_toy_graph()

    encoder = NoiseEnhancedLightGCN(num_layers=2, beta=0.0)
    encoder.train()

    first = encoder(users, items, edge_index)[0]
    second = encoder(users, items, edge_index)[0]

    assert torch.allclose(first, second)


def test_gradients_flow():
    torch.manual_seed(7)

    users = torch.randn(2, 8, requires_grad=True)
    items = torch.randn(3, 8, requires_grad=True)
    edge_index = make_toy_graph()

    encoder = NoiseEnhancedLightGCN(num_layers=2, beta=1.5)
    encoder.train()

    user_output, item_output = encoder(users, items, edge_index)
    loss = user_output.square().mean() + item_output.square().mean()
    loss.backward()

    assert users.grad is not None
    assert items.grad is not None
    assert torch.isfinite(users.grad).all()
    assert torch.isfinite(items.grad).all()
