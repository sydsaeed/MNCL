from types import SimpleNamespace

import torch
from torch import nn

from losses import MNCLLoss, l2_parameter_loss
from models.mncl import MNCLViewEmbeddings


def _views():
    return MNCLViewEmbeddings(
        e_s_user=torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
        e_s_item=torch.tensor([[1.0, 1.0], [1.0, -1.0]], requires_grad=True),
        e_m_user=torch.tensor([[0.9, 0.1], [0.1, 0.9]], requires_grad=True),
        e_m_item=torch.tensor([[0.8, 1.2], [1.1, -0.8]], requires_grad=True),
        e_g_item=torch.tensor([[1.2, 0.7], [0.7, -1.2]], requires_grad=True),
    )


def test_l2_parameter_loss_is_squared_norm():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0]]))

    assert torch.allclose(l2_parameter_loss(model), torch.tensor(25.0))


def test_eq22_combines_three_terms():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.5)

    objective = MNCLLoss(
        alpha=0.1,
        l2_lambda=0.01,
        temperature=0.2,
        omega=0.8,
    )
    positive = torch.tensor([2.0, 1.0], requires_grad=True)
    negative = torch.tensor([0.5, 0.2], requires_grad=True)

    out = objective(
        model=model,
        positive_scores=positive,
        negative_scores=negative,
        views=_views(),
    )

    expected = out.bpr_loss + 0.1 * out.contrastive_loss + 0.01 * out.l2_loss
    assert torch.allclose(out.total_loss, expected)


def test_objective_supports_batch_contrastive_ids():
    model = nn.Linear(2, 1)
    objective = MNCLLoss(
        alpha=0.1,
        l2_lambda=0.0,
        temperature=0.2,
        omega=0.8,
    )

    views = MNCLViewEmbeddings(
        e_s_user=torch.randn(4, 3),
        e_s_item=torch.randn(5, 3),
        e_m_user=torch.randn(4, 3),
        e_m_item=torch.randn(5, 3),
        e_g_item=torch.randn(5, 3),
    )

    out = objective(
        model=model,
        positive_scores=torch.tensor([1.0, 2.0]),
        negative_scores=torch.tensor([0.0, 0.5]),
        views=views,
        contrastive_user_ids=torch.tensor([0, 2], dtype=torch.long),
        contrastive_item_ids=torch.tensor([1, 4], dtype=torch.long),
    )

    assert torch.isfinite(out.total_loss)


def test_total_loss_backpropagates():
    model = nn.Linear(2, 1)
    objective = MNCLLoss(
        alpha=0.1,
        l2_lambda=1e-4,
        temperature=0.2,
        omega=0.8,
    )
    positive = torch.tensor([1.0, 0.5], requires_grad=True)
    negative = torch.tensor([0.2, 0.8], requires_grad=True)
    views = _views()

    out = objective(model, positive, negative, views)
    out.total_loss.backward()

    assert torch.isfinite(positive.grad).all()
    assert torch.isfinite(negative.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())
