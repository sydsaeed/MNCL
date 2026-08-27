from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from training.checkpoint import save_checkpoint
from training.evaluator import evaluate_checkpoint, evaluate_ctr, predict_ctr


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def get_final_embeddings(
        self,
        graphs,
        add_noise=None,
        structural_dropout=None,
        message_dropout=None,
    ):
        users = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            device=self.scale.device,
        ) * self.scale
        items = torch.tensor(
            [[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]],
            device=self.scale.device,
        )
        return users, items, None


def _ratings():
    return np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 2, 1],
            [1, 3, 0],
        ],
        dtype=np.int64,
    )


def test_predict_ctr_returns_all_rows():
    model = FakeModel()
    labels, raw_scores, probabilities = predict_ctr(
        model=model,
        graphs=SimpleNamespace(),
        ratings=_ratings(),
        batch_size=2,
        device=torch.device("cpu"),
        show_progress=False,
    )

    assert labels.tolist() == [1, 0, 1, 0]
    assert raw_scores.shape == (4,)
    assert probabilities.shape == (4,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_evaluate_ctr_perfect_metrics():
    result = evaluate_ctr(
        model=FakeModel(),
        graphs=SimpleNamespace(),
        ratings=_ratings(),
        batch_size=3,
        device=torch.device("cpu"),
        threshold=0.5,
        show_progress=False,
    )

    assert result.metrics.auc == pytest.approx(1.0)
    assert result.metrics.f1 == pytest.approx(1.0)
    assert result.metrics.precision == pytest.approx(1.0)
    assert result.metrics.recall == pytest.approx(1.0)


def test_evaluate_checkpoint_loads_saved_model(tmp_path: Path):
    source_model = FakeModel()
    optimizer = torch.optim.Adam(source_model.parameters(), lr=0.01)
    with torch.no_grad():
        source_model.scale.fill_(2.0)

    checkpoint_path = tmp_path / "best.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=source_model,
        optimizer=optimizer,
        epoch=7,
        best_test_loss=0.3,
        train_history=[0.8, 0.5],
        test_history=[0.7, 0.3],
        config={"dataset": "fake"},
    )

    evaluation_model = FakeModel()
    with torch.no_grad():
        evaluation_model.scale.fill_(0.1)

    result = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        model=evaluation_model,
        graphs=SimpleNamespace(),
        ratings=_ratings(),
        batch_size=2,
        device=torch.device("cpu"),
        threshold=0.5,
        show_progress=False,
    )

    assert float(evaluation_model.scale.detach()) == pytest.approx(2.0)
    assert result.checkpoint is not None
    assert result.checkpoint.epoch == 7
    assert result.metrics.auc == pytest.approx(1.0)
