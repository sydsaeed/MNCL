import torch
import torch.nn.functional as F

from losses import BPRLoss, bpr_loss


def test_bpr_matches_eq21():
    positive = torch.tensor([2.0, 0.5, -0.3])
    negative = torch.tensor([0.5, 0.1, -1.0])

    result = bpr_loss(positive, negative, reduction="sum")
    expected = -torch.log(torch.sigmoid(positive - negative)).sum()

    assert torch.allclose(result, expected)


def test_bpr_mean_matches_softplus():
    positive = torch.tensor([1.0, 2.0])
    negative = torch.tensor([0.0, 3.0])

    result = BPRLoss(reduction="mean")(positive, negative)
    expected = F.softplus(negative - positive).mean()

    assert torch.allclose(result, expected)


def test_bpr_has_finite_gradients():
    positive = torch.tensor([1.0, 0.5], requires_grad=True)
    negative = torch.tensor([0.2, 0.8], requires_grad=True)

    loss = bpr_loss(positive, negative)
    loss.backward()

    assert torch.isfinite(positive.grad).all()
    assert torch.isfinite(negative.grad).all()
