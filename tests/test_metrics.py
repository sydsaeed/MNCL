import numpy as np
import pytest

from utils.metrics import binary_auc, binary_classification_metrics


def test_binary_auc_perfect_ranking():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert binary_auc(labels, scores) == pytest.approx(1.0)


def test_binary_auc_handles_ties():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert binary_auc(labels, scores) == pytest.approx(0.5)


def test_binary_metrics_threshold():
    labels = np.array([0, 1, 1, 0])
    probabilities = np.array([0.1, 0.9, 0.8, 0.7])

    metrics = binary_classification_metrics(
        labels,
        probabilities,
        threshold=0.5,
    )

    assert metrics.auc == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(0.8)
    assert metrics.num_samples == 4


def test_auc_requires_both_classes():
    with pytest.raises(ValueError):
        binary_auc(np.array([1, 1]), np.array([0.2, 0.8]))
