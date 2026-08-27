import math

import pytest
import torch
import torch.nn.functional as F

from losses.contrastive import MultiNegativeContrastiveLoss, multi_negative_view_loss


def brute_force_loss(anchor, contrast, temperature, omega):
    anchor = F.normalize(anchor, dim=-1)
    contrast = F.normalize(contrast, dim=-1)

    losses = []
    for i in range(anchor.size(0)):
        positive = torch.exp(torch.dot(anchor[i], contrast[i]) / temperature)

        same_negative = anchor.new_tensor(0.0)
        cross_negative = anchor.new_tensor(0.0)
        for j in range(anchor.size(0)):
            if i == j:
                continue
            same_negative = same_negative + torch.exp(
                torch.dot(anchor[i], anchor[j]) / temperature
            )
            cross_negative = cross_negative + torch.exp(
                torch.dot(anchor[i], contrast[j]) / temperature
            )

        denominator = positive + omega * same_negative + cross_negative
        losses.append(-torch.log(positive / denominator))

    return torch.stack(losses).mean()


def test_view_loss_matches_direct_equation():
    anchor = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    contrast = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9], [0.8, 1.2]],
        dtype=torch.float32,
    )

    actual = multi_negative_view_loss(anchor, contrast, temperature=0.5, omega=0.8)
    expected = brute_force_loss(anchor, contrast, temperature=0.5, omega=0.8)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_total_loss_is_sum_of_three_components():
    torch.manual_seed(7)
    loss_fn = MultiNegativeContrastiveLoss(temperature=0.2, omega=0.8)

    e_s_user = torch.randn(5, 8)
    e_m_user = torch.randn(5, 8)
    e_s_item = torch.randn(7, 8)
    e_m_item = torch.randn(7, 8)
    e_g_item = torch.randn(7, 8)

    components = loss_fn.compute_components(
        e_s_user,
        e_m_user,
        e_s_item,
        e_m_item,
        e_g_item,
    )
    total = loss_fn(e_s_user, e_m_user, e_s_item, e_m_item, e_g_item)

    assert torch.allclose(total, sum(components.values()))
    assert set(components) == {"L_G_user", "L_G_item", "L_KG_item"}


def test_gradients_are_finite():
    torch.manual_seed(1)
    anchor = torch.randn(6, 4, requires_grad=True)
    contrast = torch.randn(6, 4, requires_grad=True)

    loss = multi_negative_view_loss(anchor, contrast, temperature=0.3, omega=0.8)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(anchor.grad).all()
    assert torch.isfinite(contrast.grad).all()


def test_single_sample_has_zero_loss():
    anchor = torch.tensor([[1.0, 2.0]])
    contrast = torch.tensor([[2.0, 1.0]])

    loss = multi_negative_view_loss(anchor, contrast, temperature=0.5, omega=0.8)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_zero_omega_removes_same_view_negatives():
    anchor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    contrast = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    actual = multi_negative_view_loss(anchor, contrast, temperature=1.0, omega=0.0)
    expected = math.log(1.0 + math.exp(-1.0))

    assert actual.item() == pytest.approx(expected, abs=1e-6)


def test_invalid_shapes_raise():
    anchor = torch.randn(4, 8)
    contrast = torch.randn(5, 8)

    with pytest.raises(ValueError):
        multi_negative_view_loss(anchor, contrast, temperature=0.2, omega=0.8)
