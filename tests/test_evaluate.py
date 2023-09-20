"""Tests for the scoring helpers.

Metric code that is wrong in the optimistic direction is the easiest kind of bug
to ship, so these check the helpers against hand-computed answers.
"""


import numpy as np
import pandas as pd

from ml.evaluate import top_k_accuracy


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
