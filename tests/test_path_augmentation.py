import torch

from models.path_augmentation import PathViewAugmentor, sample_aligned_tensors


def test_sample_aligned_tensors_keeps_alignment():
    torch.manual_seed(7)
    a = torch.arange(10)
    b = a + 100

    out_a, out_b = sample_aligned_tensors((a, b), keep_ratio=0.5)

    assert out_a.numel() == 5
    assert torch.equal(out_b - out_a, torch.full_like(out_a, 100))


def test_augmentor_can_be_disabled():
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([3, 4, 5])
    kg_edge_index = torch.tensor([[0, 1], [1, 2]])
    kg_edge_type = torch.tensor([0, 1])

    augmentor = PathViewAugmentor(keep_ratio=0.5)
    sample = augmentor(
        users,
        items,
        kg_edge_index,
        kg_edge_type,
        enabled=False,
    )

    assert torch.equal(sample.user_index, users)
    assert torch.equal(sample.item_index, items)
    assert torch.equal(sample.kg_edge_index, kg_edge_index)
    assert torch.equal(sample.kg_edge_type, kg_edge_type)


def test_zero_keep_ratio_returns_empty_edges():
    users = torch.tensor([0, 1])
    items = torch.tensor([1, 2])
    kg_edge_index = torch.tensor([[0, 1], [1, 2]])
    kg_edge_type = torch.tensor([0, 0])

    sample = PathViewAugmentor(keep_ratio=0.0)(
        users,
        items,
        kg_edge_index,
        kg_edge_type,
        enabled=True,
    )

    assert sample.user_index.numel() == 0
    assert sample.item_index.numel() == 0
    assert sample.kg_edge_index.shape == (2, 0)
    assert sample.kg_edge_type.numel() == 0
