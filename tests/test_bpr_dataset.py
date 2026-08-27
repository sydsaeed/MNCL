import importlib.util
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).parents[1] / "datasets" / "bpr_dataset.py"
spec = importlib.util.spec_from_file_location("bpr_dataset", MODULE_PATH)
bpr_dataset = importlib.util.module_from_spec(spec)
import sys
sys.modules["bpr_dataset"] = bpr_dataset
spec.loader.exec_module(bpr_dataset)


def test_triplets_pair_same_user_labels():
    ratings = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [0, 2, 0],
            [0, 3, 0],
            [1, 1, 1],
            [1, 4, 0],
        ],
        dtype=np.int64,
    )
    triplets = bpr_dataset.build_bpr_triplets(ratings, seed=7)

    assert len(triplets) == 3
    assert triplets.user_ids.dtype == torch.long

    positives = {(int(u), int(i)) for u, i in ratings[ratings[:, 2] == 1, :2]}
    negatives = {(int(u), int(i)) for u, i in ratings[ratings[:, 2] == 0, :2]}

    for u, p, n in zip(
        triplets.user_ids.tolist(),
        triplets.positive_item_ids.tolist(),
        triplets.negative_item_ids.tolist(),
    ):
        assert (u, p) in positives
        assert (u, n) in negatives


def test_triplets_are_deterministic_for_same_seed():
    ratings = np.array(
        [
            [0, 0, 1], [0, 1, 1], [0, 2, 0], [0, 3, 0],
            [1, 4, 1], [1, 5, 1], [1, 6, 0], [1, 7, 0],
        ],
        dtype=np.int64,
    )
    first = bpr_dataset.build_bpr_triplets(ratings, seed=11)
    second = bpr_dataset.build_bpr_triplets(ratings, seed=11)

    assert torch.equal(first.user_ids, second.user_ids)
    assert torch.equal(first.positive_item_ids, second.positive_item_ids)
    assert torch.equal(first.negative_item_ids, second.negative_item_ids)


def test_unequal_counts_cycle_smaller_side():
    ratings = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [0, 2, 1],
            [0, 3, 0],
        ],
        dtype=np.int64,
    )
    triplets = bpr_dataset.build_bpr_triplets(ratings, seed=3)
    assert len(triplets) == 3
    assert set(triplets.negative_item_ids.tolist()) == {3}
