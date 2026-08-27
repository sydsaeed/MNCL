from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from training.checkpoint import load_checkpoint
from training.trainer import train_epochs


@dataclass
class Views:
    e_s_user: torch.Tensor
    e_s_item: torch.Tensor
    e_m_user: torch.Tensor
    e_m_item: torch.Tensor
    e_g_item: torch.Tensor


@dataclass
class Output:
    user_embeddings: torch.Tensor
    item_embeddings: torch.Tensor
    views: Views


@dataclass
class LossOutput:
    total_loss: torch.Tensor
    bpr_loss: torch.Tensor
    contrastive_loss: torch.Tensor
    l2_loss: torch.Tensor


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.users = nn.Parameter(torch.randn(2, 3))
        self.items = nn.Parameter(torch.randn(4, 3))

    def forward(self, graphs, **kwargs):
        views = Views(self.users, self.items, self.users, self.items, self.items)
        return Output(self.users, self.items, views)


class DummyLoss(nn.Module):
    def forward(
        self,
        model,
        positive_scores,
        negative_scores,
        views,
        contrastive_user_ids=None,
        contrastive_item_ids=None,
    ):
        bpr = torch.nn.functional.softplus(
            negative_scores - positive_scores
        ).mean()
        zero = bpr * 0.0
        return LossOutput(bpr, bpr, zero, zero)


def test_trainer_saves_best_and_latest_checkpoints(tmp_path):
    ratings = np.array(
        [
            [0, 0, 1], [0, 1, 0],
            [1, 2, 1], [1, 3, 0],
        ],
        dtype=np.int64,
    )
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    best_path = tmp_path / "best.pt"
    latest_path = tmp_path / "latest.pt"

    train_loss, test_loss = train_epochs(
        model=model,
        optimizer=optimizer,
        loss_fn=DummyLoss(),
        graphs=object(),
        train_ratings=ratings,
        test_ratings=ratings,
        num_epochs=3,
        batch_size=2,
        device=torch.device("cpu"),
        seed=3,
        best_checkpoint_path=best_path,
        latest_checkpoint_path=latest_path,
        total_train_loss=[],
        total_test_loss=[],
        config={"dataset": "last-fm"},
    )

    assert best_path.exists()
    assert latest_path.exists()

    latest = load_checkpoint(
        latest_path,
        model=model,
        optimizer=optimizer,
        restore_rng=False,
    )
    assert latest.epoch == 3
    assert latest.train_history == train_loss
    assert latest.test_history == test_loss
    assert latest.best_test_loss == min(test_loss)

    best = load_checkpoint(
        best_path,
        model=model,
        optimizer=optimizer,
        restore_rng=False,
    )
    assert best.best_test_loss == min(test_loss)
    assert best.test_history[-1] == min(test_loss)
