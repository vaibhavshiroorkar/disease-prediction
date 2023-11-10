"""Tests for the classifier definition and the recall-oriented search."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.model import PARAM_GRID, base_model, recall_search, search_summary


def test_base_model_is_a_balanced_random_forest() -> None:
    model = base_model()
    assert isinstance(model, RandomForestClassifier)
    assert model.class_weight == "balanced"
    assert model.random_state == 42


def test_base_model_accepts_overrides() -> None:
    assert base_model(n_estimators=7).n_estimators == 7


def test_search_refits_on_recall_not_accuracy() -> None:
    """The whole point of the search: the winner is the highest-recall candidate."""
    search = recall_search()
    assert search.refit == "recall_macro"
    assert set(search.scoring) == {"recall_macro", "accuracy"}


def test_param_grid_only_holds_supported_forest_arguments() -> None:
    valid = RandomForestClassifier().get_params()
    assert set(PARAM_GRID) <= set(valid)


def test_search_summary_reports_ties_and_the_accuracy_trade() -> None:
    """A tiny separable problem where most candidates reach the same recall."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.integers(0, 2, size=(40, 4)), columns=list("abcd"))
    y = pd.Series(["x" if row else "y" for row in X["a"]])

    search = recall_search(folds=2)
    search.fit(X, y)
    summary = search_summary(search)

    assert summary["candidates"] == len(search.cv_results_["params"])
    assert 1 <= summary["tied_at_best"] <= summary["candidates"]
    assert 0.0 <= summary["cv_recall_macro"] <= 1.0
    assert summary["worst_candidate_recall"] <= summary["cv_recall_macro"]
    assert set(summary["best_params"]) == set(PARAM_GRID)
