from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryMetrics:
    auc: float
    f1: float
    precision: float
    recall: float
    threshold: float
    num_samples: int


def _as_1d(array, name: str) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return values


def _validate_binary_labels(labels: np.ndarray) -> None:
    unique = np.unique(labels)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError("labels must contain only 0 and 1.")


def binary_auc(labels, scores) -> float:
    """Compute ROC-AUC with tie-aware average ranks."""
    labels = _as_1d(labels, "labels").astype(np.int64, copy=False)
    scores = _as_1d(scores, "scores").astype(np.float64, copy=False)

    if labels.size != scores.size:
        raise ValueError("labels and scores must have the same length.")
    if labels.size == 0:
        raise ValueError("labels and scores must not be empty.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite.")
    _validate_binary_labels(labels)

    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUC requires both positive and negative samples.")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)

    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1

        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = ranks[labels == 1].sum()
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


def binary_classification_metrics(
    labels,
    probabilities,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Compute AUC, F1, precision, and recall for CTR predictions."""
    labels = _as_1d(labels, "labels").astype(np.int64, copy=False)
    probabilities = _as_1d(probabilities, "probabilities").astype(
        np.float64,
        copy=False,
    )

    if labels.size != probabilities.size:
        raise ValueError("labels and probabilities must have the same length.")
    if labels.size == 0:
        raise ValueError("labels and probabilities must not be empty.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite.")
    _validate_binary_labels(labels)

    predictions = (probabilities >= threshold).astype(np.int64)

    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )
    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )

    return BinaryMetrics(
        auc=binary_auc(labels, probabilities),
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        threshold=float(threshold),
        num_samples=int(labels.size),
    )
