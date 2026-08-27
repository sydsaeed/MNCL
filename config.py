from dataclasses import dataclass
from pathlib import Path


@dataclass
class MNCLConfig:
    dataset: str = "last-fm"
    data_root: str = "data"
    seed: int = 40
    test_ratio: float = 0.2

    embedding_dim: int = 64
    batch_size: int = 4096
    lr: float = 0.001
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = False
    checkpoint_dir: str = "checkpoints"

    alpha: float = 0.1
    K: int = 2
    L: int = 4
    beta: float = 1.5
    omega: float = 0.8
    tau: float = 0.6  # Official code default; paper does not report tau
    l2_lambda: float = 1e-6  # Implementation default; tune from the paper grid
    bpr_reduction: str = "mean"
    f1_threshold: float = 0.5

    semantic_topk: int = 10
    semantic_chunk_size: int = 1024

    path_hops: int = 2
    path_structural_keep_ratio: float = 0.5
    path_message_dropout_rate: float = 0.1

    def dataset_dir(self) -> Path:
        return Path(self.data_root) / self.dataset

    def best_checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir) / f"{self.dataset}_best.pt"

    def latest_checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir) / f"{self.dataset}_latest.pt"
