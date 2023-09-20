"""Scoring helpers.

The point of this module is to produce numbers that mean something. The symptom
dataset is easy enough that a careless split reports 100% accuracy, so scores
here are always computed on deduplicated patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ml.config import RANDOM_SEED


def top_k_accuracy(proba: np.ndarray, classes: np.ndarray, y_true: pd.Series, k: int) -> float:
    """Share of rows where the true label lands in the top k ranked classes."""
    ranked = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    truth = np.asarray(y_true)
    hits = [truth[i] in classes[ranked[i]] for i in range(len(truth))]
    return float(np.mean(hits))


def cross_validated_scores(model, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> dict[str, object]:
    """Out-of-fold scores over unique symptom patterns.

    Every disease keeps at least 5 distinct patterns after deduplication, so
    5 folds always have one of each class available.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba", n_jobs=None)
    classes = np.unique(y)
    predicted = classes[np.argmax(proba, axis=1)]
    return {
        "cv_folds": folds,
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "macro_f1": round(float(f1_score(y, predicted, average="macro")), 4),
        "top_3_accuracy": round(top_k_accuracy(proba, classes, y, 3), 4),
    }


def holdout_scores(model, X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    """Score a fitted model on the publisher's held-out split."""
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    predicted = classes[np.argmax(proba, axis=1)]
    return {
        "rows": int(len(y)),
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "top_3_accuracy": round(top_k_accuracy(proba, classes, y, 3), 4),
    }
