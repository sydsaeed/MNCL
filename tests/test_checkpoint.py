import random

import numpy as np
import torch
from torch import nn

from training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip_restores_state_and_history(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    x = torch.randn(4, 3)
    loss = model(x).pow(2).mean()
    loss.backward()
    optimizer.step()

    saved_weights = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    path = tmp_path / "model.pt"
    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        epoch=7,
        best_test_loss=0.25,
        train_history=[1.0, 0.7],
        test_history=[0.9, 0.6],
        config={"dataset": "last-fm"},
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)

    state = load_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        device="cpu",
    )

    for key, value in model.state_dict().items():
        assert torch.allclose(value, saved_weights[key])
    assert state.epoch == 7
    assert state.best_test_loss == 0.25
    assert state.train_history == [1.0, 0.7]
    assert state.test_history == [0.9, 0.6]
    assert state.config == {"dataset": "last-fm"}


def test_checkpoint_restores_rng_state(tmp_path):
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "rng.pt"
    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        best_test_loss=1.0,
        train_history=[1.0],
        test_history=[1.0],
    )

    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(1))

    random.random()
    np.random.rand()
    torch.rand(1)

    load_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        restore_rng=True,
    )

    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert float(torch.rand(1)) == expected_torch
