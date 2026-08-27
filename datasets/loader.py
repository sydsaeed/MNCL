from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetInfo:
    M: int
    N: int
    K: int
    L: int

    @property
    def num_kg_nodes(self) -> int:
        return self.N + self.K


@dataclass
class DatasetBundle:
    ratings: np.ndarray
    kg: np.ndarray
    info: DatasetInfo


def _load_array(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

    array = np.load(path)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape [num_rows, 3].")

    return array.astype(np.int64, copy=False)


def _validate_ratings(ratings: np.ndarray) -> None:
    labels = np.unique(ratings[:, 2])
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("ratings labels must be 0 or 1.")

    if ratings[:, 0].min() < 0 or ratings[:, 1].min() < 0:
        raise ValueError("ratings IDs must be non-negative.")


def _validate_kg(kg: np.ndarray) -> None:
    if kg.min() < 0:
        raise ValueError("kg IDs must be non-negative.")


def _infer_info(ratings: np.ndarray, kg: np.ndarray) -> DatasetInfo:
    M = int(ratings[:, 0].max()) + 1
    N = int(ratings[:, 1].max()) + 1
    L = int(kg[:, 1].max()) + 1

    max_kg_node = int(max(kg[:, 0].max(), kg[:, 2].max()))
    num_kg_nodes = max_kg_node + 1
    K = num_kg_nodes - N

    if K < 0:
        raise ValueError("KG node range is smaller than the item range.")

    return DatasetInfo(M=M, N=N, K=K, L=L)


def _validate_ranges(
    ratings: np.ndarray,
    kg: np.ndarray,
    info: DatasetInfo,
) -> None:
    user_ids = np.unique(ratings[:, 0])
    item_ids = np.unique(ratings[:, 1])
    relation_ids = np.unique(kg[:, 1])
    kg_nodes = np.unique(np.concatenate([kg[:, 0], kg[:, 2]]))

    if not np.array_equal(user_ids, np.arange(info.M)):
        raise ValueError("user IDs must be continuous from 0 to M-1.")

    if not np.array_equal(item_ids, np.arange(info.N)):
        raise ValueError("item IDs must be continuous from 0 to N-1.")

    if not np.array_equal(relation_ids, np.arange(info.L)):
        raise ValueError("relation IDs must be continuous from 0 to L-1.")

    if not np.array_equal(kg_nodes, np.arange(info.num_kg_nodes)):
        raise ValueError("KG node IDs must be continuous from 0 to N+K-1.")


def load_dataset(config) -> DatasetBundle:
    dataset_dir = config.dataset_dir()
    ratings_path = dataset_dir / "ratings_final.npy"
    kg_path = dataset_dir / "kg_final.npy"

    ratings = _load_array(ratings_path, "ratings_final.npy")
    kg = _load_array(kg_path, "kg_final.npy")

    _validate_ratings(ratings)
    _validate_kg(kg)

    info = _infer_info(ratings, kg)
    _validate_ranges(ratings, kg, info)

    return DatasetBundle(ratings=ratings, kg=kg, info=info)
