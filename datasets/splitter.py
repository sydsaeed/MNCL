from dataclasses import dataclass

import numpy as np


@dataclass
class RatingSplit:
    train_ratings: np.ndarray
    test_ratings: np.ndarray


def _split_group(
    rows: np.ndarray,
    test_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(len(rows))

    if len(rows) <= 1:
        return rows[order], rows[:0]

    test_count = max(1, int(round(len(rows) * test_ratio)))
    test_count = min(test_count, len(rows) - 1)

    test_rows = rows[order[:test_count]]
    train_rows = rows[order[test_count:]]
    return train_rows, test_rows


def split_ratings(
    ratings: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> RatingSplit:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for user_id in np.unique(ratings[:, 0]):
        user_rows = ratings[ratings[:, 0] == user_id]

        for label in (0, 1):
            group = user_rows[user_rows[:, 2] == label]
            train_group, test_group = _split_group(group, test_ratio, rng)
            train_parts.append(train_group)
            test_parts.append(test_group)

    train_ratings = np.concatenate(train_parts, axis=0)
    test_ratings = np.concatenate(test_parts, axis=0)

    train_ratings = train_ratings[rng.permutation(len(train_ratings))]
    test_ratings = test_ratings[rng.permutation(len(test_ratings))]

    return RatingSplit(
        train_ratings=train_ratings,
        test_ratings=test_ratings,
    )
