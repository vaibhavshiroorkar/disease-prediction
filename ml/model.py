"""The classifier and its hyperparameter search.

A Random Forest over the binary symptom vector, selected by macro recall rather
than accuracy. In this setting a false negative means telling someone their
condition is not in the ranking at all, which is worse than listing a condition
they do not have. Accuracy would let the search trade away recall on a rare
disease to gain a fraction of a point on common ones; macro recall weights every
one of the 41 classes the same, so it cannot.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from ml.config import RANDOM_SEED

# Kept deliberately small. There are only ~300 unique training patterns, so a
# wider grid overfits the cross-validation rather than finding a better model.
PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [300, 600],
    "max_depth": [None, 20],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.3],
    "class_weight": ["balanced", "balanced_subsample"],
}


def base_model(seed: int = RANDOM_SEED, **overrides: Any) -> RandomForestClassifier:
    """An untuned forest with the settings that never varied.

    class_weight defaults to balanced because the deduplicated training set is
    uneven: some diseases keep 10 distinct symptom patterns, others only 5.
    """
    params: dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": seed,
    }
    params.update(overrides)
    return RandomForestClassifier(**params)


def recall_search(folds: int = 5, seed: int = RANDOM_SEED) -> GridSearchCV:
    """Grid search scoring on macro recall.

    `refit="recall_macro"` means the returned best_estimator_ is the one that
    missed the fewest true conditions, not the one with the best accuracy. Both
    scores are recorded so the trade-off is visible in the training log.
    """
    return GridSearchCV(
        estimator=base_model(seed=seed),
        param_grid=PARAM_GRID,
        scoring={"recall_macro": "recall_macro", "accuracy": "accuracy"},
        refit="recall_macro",
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        n_jobs=-1,
        return_train_score=False,
    )


def search_summary(search: GridSearchCV) -> dict[str, Any]:
    """What the search chose, what accuracy that choice cost, and how close the
    field was.

    `tied_at_best` matters here. This dataset is separable enough that most of
    the grid reaches the same macro recall, so the winner is the first of a tie
    rather than a clear victor. Reporting the tie count stops the best_params
    from being read as a meaningful finding.
    """
    index = int(search.best_index_)
    results = search.cv_results_
    best_recall = float(results["mean_test_recall_macro"][index])
    tied = sum(1 for score in results["mean_test_recall_macro"] if abs(float(score) - best_recall) < 1e-9)
    return {
        "candidates": len(results["params"]),
        "tied_at_best": tied,
        "best_params": search.best_params_,
        "cv_recall_macro": round(best_recall, 4),
        "cv_accuracy": round(float(results["mean_test_accuracy"][index]), 4),
        "worst_candidate_recall": round(float(min(results["mean_test_recall_macro"])), 4),
    }
