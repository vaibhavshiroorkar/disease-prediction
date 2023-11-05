"""Tests for the scoring helpers.

Metric code that is wrong in the optimistic direction is the easiest kind of bug
to ship, so these check the helpers against hand-computed answers.
"""


import numpy as np
import pandas as pd

from ml.evaluate import per_class_recall, top_k_accuracy, weakest_classes


def test_top_k_accuracy_counts_a_hit_anywhere_in_the_top_k() -> None:
    proba = np.array(
        [
            [0.5, 0.3, 0.2],  # true 'b' ranked 2nd
            [0.1, 0.2, 0.7],  # true 'c' ranked 1st
        ]
    )
    classes = np.array(["a", "b", "c"])
    truth = pd.Series(["b", "c"])

    assert top_k_accuracy(proba, classes, truth, k=1) == 0.5
    assert top_k_accuracy(proba, classes, truth, k=2) == 1.0


def test_top_k_accuracy_is_zero_when_the_truth_never_ranks() -> None:
    proba = np.array([[0.9, 0.05, 0.05]])
    classes = np.array(["a", "b", "c"])
    assert top_k_accuracy(proba, classes, pd.Series(["c"]), k=1) == 0.0


def test_per_class_recall_matches_hand_computed_values() -> None:
    truth = pd.Series(["a", "a", "b", "b", "c"])
    predicted = np.array(["a", "b", "b", "b", "a"])

    recalls = per_class_recall(truth, predicted, ["a", "b", "c"])

    assert recalls["a"] == 0.5  # 1 of 2 caught
    assert recalls["b"] == 1.0  # 2 of 2 caught
    assert recalls["c"] == 0.0  # never predicted


def test_a_class_the_model_never_predicts_reports_zero_not_missing() -> None:
    """The silently-dropped class is exactly the one worth seeing."""
    truth = pd.Series(["a", "b"])
    predicted = np.array(["a", "a"])
    recalls = per_class_recall(truth, predicted, ["a", "b"])
    assert recalls == {"a": 1.0, "b": 0.0}


def test_weakest_classes_lists_the_worst_first_and_skips_perfect_ones() -> None:
    recalls = {"a": 1.0, "b": 0.4, "c": 0.9, "d": 0.0}
    assert weakest_classes(recalls, limit=3) == [
        {"disease": "d", "recall": 0.0},
        {"disease": "b", "recall": 0.4},
        {"disease": "c", "recall": 0.9},
    ]


def test_weakest_classes_is_empty_when_everything_is_perfect() -> None:
    assert weakest_classes({"a": 1.0, "b": 1.0}) == []
