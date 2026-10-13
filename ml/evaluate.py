"""Scoring helpers.

The point of this module is to produce numbers that mean something. The symptom
dataset is easy enough that a careless split reports 100% accuracy, so scores
here are always computed on deduplicated patterns.

Recall is the headline. Missing a real condition is the expensive error, so
per-class recall is reported in full rather than averaged away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ml.config import RANDOM_SEED


def top_k_accuracy(proba: np.ndarray, classes: np.ndarray, y_true: pd.Series, k: int) -> float:
    """Share of rows where the true label lands in the top k ranked classes."""
    ranked = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    truth = np.asarray(y_true)
    hits = [truth[i] in classes[ranked[i]] for i in range(len(truth))]
    return float(np.mean(hits))


def per_class_recall(y_true: pd.Series, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Recall for every class, keyed by label.

    zero_division=0 so a class the model never predicts reports 0.0 instead of
    vanishing from the report, which is the case worth seeing.
    """
    scores = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {label: round(float(score), 4) for label, score in zip(labels, scores, strict=True)}


def weakest_classes(recalls: dict[str, float], limit: int = 5) -> list[dict[str, float]]:
    """The classes the model misses most often, worst first."""
    ordered = sorted(recalls.items(), key=lambda pair: pair[1])
    return [{"disease": name, "recall": score} for name, score in ordered[:limit] if score < 1.0]


def cross_validated_scores(model, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> dict[str, object]:
    """Out-of-fold scores over unique symptom patterns.

    Every disease keeps at least 5 distinct patterns after deduplication, so
    5 folds always have one of each class available.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba", n_jobs=None)
    classes = np.unique(y)
    predicted = classes[np.argmax(proba, axis=1)]
    recalls = per_class_recall(y, predicted, list(classes))
    return {
        "cv_folds": folds,
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "macro_recall": round(float(recall_score(y, predicted, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y, predicted, average="macro")), 4),
        "top_3_accuracy": round(top_k_accuracy(proba, classes, y, 3), 4),
        # Accuracy is saturated on this data, so this is the score that still
        # separates models: it punishes a right answer given with low confidence.
        "log_loss": round(float(log_loss(y, proba, labels=classes)), 4),
        "weakest_classes": weakest_classes(recalls),
    }


def holdout_scores(model, X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    """Score a fitted model on the publisher's held-out split.

    Reports accuracy and per-class recall together, which is the pair the model
    was selected on.
    """
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    predicted = classes[np.argmax(proba, axis=1)]
    recalls = per_class_recall(y, predicted, list(classes))
    present = sorted(set(y))
    return {
        "rows": int(len(y)),
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "macro_recall": round(float(recall_score(y, predicted, average="macro", zero_division=0)), 4),
        "top_3_accuracy": round(top_k_accuracy(proba, classes, y, 3), 4),
        "classes_with_perfect_recall": sum(1 for c in present if recalls.get(c) == 1.0),
        "classes_evaluated": len(present),
        "per_class_recall": {c: recalls[c] for c in present},
    }


def _perturb(matrix: np.ndarray, drop: int, add: int, rng: np.random.Generator) -> np.ndarray:
    """Switch off `drop` present symptoms and switch on `add` absent ones, per row."""
    for row in range(matrix.shape[0]):
        present = np.flatnonzero(matrix[row] == 1)
        absent = np.flatnonzero(matrix[row] == 0)
        if drop and len(present) > drop:
            matrix[row, rng.choice(present, size=drop, replace=False)] = 0
        if add and len(absent) >= add:
            matrix[row, rng.choice(absent, size=add, replace=False)] = 1
    return matrix


def held_out_noise(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    drop: int,
    add: int,
    folds: int = 5,
    trials: int = 5,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """The noise test on patterns the model never saw.

    `noise_robustness` perturbs the training patterns themselves, so a model
    that memorised them gets a head start. Here each fold is held out, then
    perturbed and scored by a model fit on the other four. That is the fair
    way to compare two models' robustness.
    """
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    truth: list[str] = []
    probas: list[np.ndarray] = []
    classes = np.unique(y)

    for train_index, test_index in cv.split(X, y):
        fitted = clone(model).fit(X.iloc[train_index], y.iloc[train_index])
        order = [list(fitted.classes_).index(c) for c in classes]
        for _ in range(trials):
            perturbed = _perturb(X.iloc[test_index].to_numpy(copy=True), drop, add, rng)
            probas.append(fitted.predict_proba(pd.DataFrame(perturbed, columns=X.columns))[:, order])
            truth.extend(y.iloc[test_index])

    proba = np.vstack(probas)
    truth_series = pd.Series(truth)
    predicted = classes[np.argmax(proba, axis=1)]
    return {
        "dropped_symptoms": drop,
        "added_symptoms": add,
        "accuracy": round(float(accuracy_score(truth_series, predicted)), 4),
        "macro_recall": round(
            float(recall_score(truth_series, predicted, average="macro", zero_division=0)), 4
        ),
        "top_3_accuracy": round(top_k_accuracy(proba, classes, truth_series, 3), 4),
    }


def noise_robustness(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    drop: int = 1,
    add: int = 1,
    trials: int = 20,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Score the model when the reported symptoms are wrong in realistic ways.

    A real user forgets a symptom and volunteers one that is unrelated. Each
    trial removes `drop` present symptoms and switches on `add` absent ones,
    then measures whether the true disease is still ranked first and top-3.
    """
    rng = np.random.default_rng(seed)
    classes = np.asarray(model.classes_)
    columns = list(X.columns)
    top1: list[float] = []
    top3: list[float] = []
    recalls: list[float] = []

    for _ in range(trials):
        perturbed = _perturb(X.to_numpy(copy=True), drop, add, rng)
        proba = model.predict_proba(pd.DataFrame(perturbed, columns=columns))
        predicted = classes[np.argmax(proba, axis=1)]
        top1.append(float(accuracy_score(y, predicted)))
        top3.append(top_k_accuracy(proba, classes, y, 3))
        recalls.append(float(recall_score(y, predicted, average="macro", zero_division=0)))

    return {
        "dropped_symptoms": drop,
        "added_symptoms": add,
        "trials": trials,
        "accuracy": round(float(np.mean(top1)), 4),
        "accuracy_std": round(float(np.std(top1)), 4),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "top_3_accuracy": round(float(np.mean(top3)), 4),
    }
