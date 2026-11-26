"""One repeat of a transfer: split the target, draw n local patients, update, score.

Both studies use this, so a draw means the same thing in each.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.clinical_data import FeatureSpec
from ml.clinical_model import preprocessor
from ml.config import RANDOM_SEED
from study.shared.updating import apply_shift_slope, em_rate, fit_shift_slope, local_rate, reweight

METHODS = ("none", "em", "intercept", "platt", "refit", "rate")
TEST_CAP = 20_000
RELIABLE = 0.90
METRICS = ("log_loss", "brier", "auc", "gap", "sensitivity", "specificity")


def clipped_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Log loss with p kept off 0 and 1, where one confident miss would be infinite."""
    return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))


def score(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    p = np.asarray(p, dtype=float)
    flagged = p >= threshold
    return {
        "log_loss": clipped_log_loss(y, p),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "gap": float(p.mean() - np.mean(y)),
        "sensitivity": float(flagged[y == 1].mean()),
        "specificity": float((~flagged[y == 0]).mean()),
    }


def local_refit(
    features: list[FeatureSpec], X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame
) -> np.ndarray:
    """A logistic model from the local patients alone, with the served preprocessing."""
    # A column with no value at all in the sample cannot be imputed.
    present = [f for f in features if X[f.name].notna().any()]
    model = Pipeline(
        [
            ("prep", preprocessor(present, scale=True, missing_indicators=False)),
            ("model", LogisticRegression(C=1.0, max_iter=5000)),
        ]
    )
    names = [f.name for f in present]
    model.fit(X[names], y)
    return model.predict_proba(X_test[names])[:, 1]


def split_target(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Half the target is the pool local patients come from, half the test, capped."""
    pool, test = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed)
    if len(test) > TEST_CAP:
        _, test = train_test_split(test, test_size=TEST_CAP, stratify=y[test], random_state=seed)
    return pool, test


def draw(y: np.ndarray, repeat: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This repeat's pool, test half, and the order local patients are taken in."""
    seed = RANDOM_SEED + repeat
    pool, test = split_target(y, seed)
    return pool, test, np.random.default_rng(seed).permutation(pool)


def one_repeat(
    features: list[FeatureSpec], X: pd.DataFrame, y: np.ndarray, p: np.ndarray,
    source_rate: float, threshold: float, sizes: list[int], repeat: int,
) -> list[dict[str, Any]]:
    _, test, order = draw(y, repeat)
    rows = []
    for n in sizes:
        local = order[:n]
        y_local, p_local, p_test = y[local], p[local], p[test]
        one_class = len(np.unique(y_local)) == 1
        rate = np.full(len(test), local_rate(y_local))
        updated = {
            "none": p_test,
            "em": reweight(p_test, source_rate, em_rate(p_local, source_rate)),
            "intercept": apply_shift_slope(p_test, *fit_shift_slope(p_local, y_local, False)),
            "platt": apply_shift_slope(p_test, *fit_shift_slope(p_local, y_local, True)),
            "refit": rate if one_class else local_refit(features, X.iloc[local], y_local, X.iloc[test]),
            "rate": rate,
        }
        for method in METHODS:
            rows.append(
                {"repeat": repeat, "n": n, "method": method, "one_class": one_class,
                 "local_cases": int(y_local.sum()),
                 **score(y[test], updated[method], threshold)}
            )
    return rows


def repair_size(share_better: dict[int, float]) -> int | None:
    """The smallest n at which this and every larger n helped in RELIABLE of repeats."""
    sizes = sorted(share_better)
    for i, n in enumerate(sizes):
        if all(share_better[m] >= RELIABLE for m in sizes[i:]):
            return n
    return None


def summarize(rows: pd.DataFrame, sizes: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mean and 5th-95th percentile per method and n, plus paired wins over `none`."""
    curves: dict[str, Any] = {}
    sizes_needed: dict[str, Any] = {}
    base = rows[rows.method == "none"].set_index(["repeat", "n"])["log_loss"]
    for method in METHODS:
        mine = rows[rows.method == method]
        curves[method], share = {}, {}
        for n in sizes:
            at = mine[mine.n == n]
            entry = {
                metric: {
                    "mean": round(float(at[metric].mean()), 5),
                    "p5": round(float(at[metric].quantile(0.05)), 5),
                    "p95": round(float(at[metric].quantile(0.95)), 5),
                }
                for metric in METRICS
            }
            paired = at.set_index(["repeat", "n"])["log_loss"]
            diff = paired - base.loc[paired.index]
            share[n] = float((diff < 0).mean())
            entry["diff_from_none"] = round(float(diff.mean()), 5)
            entry["share_better_than_none"] = round(share[n], 3)
            curves[method][str(n)] = entry
        if method != "none":
            sizes_needed[method] = repair_size(share)
    return curves, sizes_needed
