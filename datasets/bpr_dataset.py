from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class BPRTriplets:
    user_ids: Tensor
    positive_item_ids: Tensor
    negative_item_ids: Tensor

    def __len__(self) -> int:
        return int(self.user_ids.numel())


class BPRTripletDataset(Dataset):
    """Pair positive and negative items for the same user."""

    def __init__(self, ratings: np.ndarray, seed: int = 42) -> None:
        self.triplets = build_bpr_triplets(ratings, seed=seed)

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.triplets.user_ids[index],
            self.triplets.positive_item_ids[index],
            self.triplets.negative_item_ids[index],
        )


def _validate_ratings(ratings: np.ndarray) -> None:
    if ratings.ndim != 2 or ratings.shape[1] != 3:
        raise ValueError("ratings must have shape [num_rows, 3].")
    if ratings.shape[0] == 0:
        raise ValueError("ratings cannot be empty.")
    if not np.all(np.isin(np.unique(ratings[:, 2]), [0, 1])):
        raise ValueError("ratings labels must be 0 or 1.")


def build_bpr_triplets(
    ratings: np.ndarray,
    seed: int = 42,
) -> BPRTriplets:
    """Build same-user (positive, negative) BPR triplets."""
    _validate_ratings(ratings)
    rng = np.random.default_rng(seed)

    users_out: list[np.ndarray] = []
    positives_out: list[np.ndarray] = []
    negatives_out: list[np.ndarray] = []

    for user_id in np.unique(ratings[:, 0]):
        user_rows = ratings[ratings[:, 0] == user_id]
        positive_items = user_rows[user_rows[:, 2] == 1, 1]
        negative_items = user_rows[user_rows[:, 2] == 0, 1]

        if len(positive_items) == 0 or len(negative_items) == 0:
            continue

        positive_items = positive_items[rng.permutation(len(positive_items))]
        negative_items = negative_items[rng.permutation(len(negative_items))]

        pair_count = max(len(positive_items), len(negative_items))
        positive_indices = np.arange(pair_count) % len(positive_items)
        negative_indices = np.arange(pair_count) % len(negative_items)

        users_out.append(np.full(pair_count, user_id, dtype=np.int64))
        positives_out.append(positive_items[positive_indices].astype(np.int64, copy=False))
        negatives_out.append(negative_items[negative_indices].astype(np.int64, copy=False))

    if not users_out:
        raise ValueError("No user has both positive and negative samples.")

    users = np.concatenate(users_out)
    positives = np.concatenate(positives_out)
    negatives = np.concatenate(negatives_out)

    order = rng.permutation(len(users))
    users = users[order]
    positives = positives[order]
    negatives = negatives[order]

    return BPRTriplets(
        user_ids=torch.as_tensor(users, dtype=torch.long),
        positive_item_ids=torch.as_tensor(positives, dtype=torch.long),
        negative_item_ids=torch.as_tensor(negatives, dtype=torch.long),
    )


def build_bpr_dataloader(
    ratings: np.ndarray,
    batch_size: int,
    seed: int = 42,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a deterministic BPR DataLoader for one epoch."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    dataset = BPRTripletDataset(ratings=ratings, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        drop_last=False,
    )
