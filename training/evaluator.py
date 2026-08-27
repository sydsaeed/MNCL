from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from training.checkpoint import CheckpointState, load_checkpoint
from utils.metrics import BinaryMetrics, binary_classification_metrics


@dataclass(frozen=True)
class CTREvaluationResult:
    metrics: BinaryMetrics
    checkpoint: CheckpointState | None = None


def _validate_ratings(ratings: np.ndarray) -> np.ndarray:
    ratings = np.asarray(ratings)
    if ratings.ndim != 2 or ratings.shape[1] != 3:
        raise ValueError("ratings must have shape [num_samples, 3].")
    if ratings.shape[0] == 0:
        raise ValueError("ratings must not be empty.")
    if not np.all(np.isin(np.unique(ratings[:, 2]), [0, 1])):
        raise ValueError("rating labels must contain only 0 and 1.")
    return ratings


def _pair_scores(
    user_embeddings: Tensor,
    item_embeddings: Tensor,
    user_ids: Tensor,
    item_ids: Tensor,
) -> Tensor:
    users = user_embeddings[user_ids]
    items = item_embeddings[item_ids]
    return (users * items).sum(dim=-1)


@torch.no_grad()
def predict_ctr(
    model: nn.Module,
    graphs: Any,
    ratings: np.ndarray,
    batch_size: int,
    device: torch.device,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict deterministic CTR probabilities for all rating rows."""
    ratings = _validate_ratings(ratings)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    model.eval()
    user_embeddings, item_embeddings, _ = model.get_final_embeddings(
        graphs=graphs,
        add_noise=False,
        structural_dropout=False,
        message_dropout=False,
    )

    labels = ratings[:, 2].astype(np.int64, copy=True)
    raw_score_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []

    starts = range(0, ratings.shape[0], batch_size)
    progress = tqdm(
        starts,
        desc="CTR evaluation",
        leave=False,
        disable=not show_progress,
    )

    for start in progress:
        end = min(start + batch_size, ratings.shape[0])
        batch = ratings[start:end]

        user_ids = torch.as_tensor(
            batch[:, 0],
            dtype=torch.long,
            device=device,
        )
        item_ids = torch.as_tensor(
            batch[:, 1],
            dtype=torch.long,
            device=device,
        )

        raw_scores = _pair_scores(
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            user_ids=user_ids,
            item_ids=item_ids,
        )
        probabilities = torch.sigmoid(raw_scores)

        raw_score_parts.append(raw_scores.detach().cpu().numpy())
        probability_parts.append(probabilities.detach().cpu().numpy())

    raw_scores = np.concatenate(raw_score_parts, axis=0)
    probabilities = np.concatenate(probability_parts, axis=0)
    return labels, raw_scores, probabilities


@torch.no_grad()
def evaluate_ctr(
    model: nn.Module,
    graphs: Any,
    ratings: np.ndarray,
    batch_size: int,
    device: torch.device,
    threshold: float = 0.5,
    show_progress: bool = True,
) -> CTREvaluationResult:
    """Evaluate AUC and F1 on all CTR test interactions."""
    labels, _, probabilities = predict_ctr(
        model=model,
        graphs=graphs,
        ratings=ratings,
        batch_size=batch_size,
        device=device,
        show_progress=show_progress,
    )
    metrics = binary_classification_metrics(
        labels=labels,
        probabilities=probabilities,
        threshold=threshold,
    )
    return CTREvaluationResult(metrics=metrics)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    graphs: Any,
    ratings: np.ndarray,
    batch_size: int,
    device: torch.device,
    threshold: float = 0.5,
    show_progress: bool = True,
) -> CTREvaluationResult:
    """Load one checkpoint and evaluate CTR metrics."""
    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        device=device,
        restore_rng=False,
    )
    result = evaluate_ctr(
        model=model,
        graphs=graphs,
        ratings=ratings,
        batch_size=batch_size,
        device=device,
        threshold=threshold,
        show_progress=show_progress,
    )
    return CTREvaluationResult(
        metrics=result.metrics,
        checkpoint=checkpoint,
    )
