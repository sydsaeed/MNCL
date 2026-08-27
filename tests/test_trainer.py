import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


TRAINER_PATH = Path(__file__).parents[1] / "training" / "trainer.py"
spec = importlib.util.spec_from_file_location("trainer_module", TRAINER_PATH)
trainer = importlib.util.module_from_spec(spec)
import sys
sys.modules["trainer_module"] = trainer
# Import dependency without importing datasets package __init__.
BPR_PATH = Path(__file__).parents[1] / "datasets" / "bpr_dataset.py"
bpr_spec = importlib.util.spec_from_file_location("datasets.bpr_dataset", BPR_PATH)
bpr_mod = importlib.util.module_from_spec(bpr_spec)
sys.modules["datasets.bpr_dataset"] = bpr_mod
bpr_spec.loader.exec_module(bpr_mod)
spec.loader.exec_module(trainer)


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
        bpr = torch.nn.functional.softplus(negative_scores - positive_scores).mean()
        zero = bpr * 0.0
        return LossOutput(bpr, bpr, zero, zero)


def test_train_epochs_returns_requested_arrays():
    ratings = np.array(
        [
            [0, 0, 1], [0, 1, 0],
            [1, 2, 1], [1, 3, 0],
        ],
        dtype=np.int64,
    )
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    train_loss, test_loss = trainer.train_epochs(
        model=model,
        optimizer=optimizer,
        loss_fn=DummyLoss(),
        graphs=object(),
        train_ratings=ratings,
        test_ratings=ratings,
        num_epochs=2,
        batch_size=2,
        device=torch.device("cpu"),
        seed=1,
        start_epoch=5,
    )

    assert len(train_loss) == 2
    assert len(test_loss) == 2
    assert all(np.isfinite(train_loss))
    assert all(np.isfinite(test_loss))
