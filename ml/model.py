"""The classifier.

A Random Forest over the binary symptom vector. Trees suit this data: the
features are 0/1 flags, the interactions between them are what identify a
disease, and nothing needs scaling.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from ml.config import RANDOM_SEED


def base_model(seed: int = RANDOM_SEED, **overrides: Any) -> RandomForestClassifier:
    """A forest with sensible defaults for this dataset.

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
