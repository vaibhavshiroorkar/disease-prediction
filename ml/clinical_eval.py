"""Scoring for the binary risk models.

The test sets here are small (115 to 184 patients), so every headline number
comes with a bootstrap interval. A single AUC from 150 patients can move by
0.05 between two reasonable splits, and quoting it to three decimals without a
range overstates what we know.

The operating threshold follows the same argument as the symptom model. A
missed case costs more than a false alarm, so the threshold is the highest one
that still catches a target share of real cases (90% by default).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ml.config import RANDOM_SEED

TARGET_SENSITIVITY = 0.90


def choose_threshold(y: np.ndarray, proba: np.ndarray, target_sensitivity: float = TARGET_SENSITIVITY) -> float:
    """The highest threshold that still flags `target_sensitivity` of true cases.

    A row is flagged when its probability is >= the threshold. Sorting the
    positives' scores from high to low, the k-th one (k = ceil(target * n_pos))
    is exactly the highest cut that keeps k of them above the line.

    It is chosen on out-of-fold training predictions, never on the test set,
    so the sensitivity the test set reports is a fair check of the choice.
    """
    y = np.asarray(y)
    positives = np.sort(np.asarray(proba)[y == 1])[::-1]
    if len(positives) == 0:
        raise ValueError("cannot choose a sensitivity threshold with no positive rows")
    k = max(1, math.ceil(target_sensitivity * len(positives)))
    return float(positives[k - 1])


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_scores(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    """Every score worth reporting for a risk model, at one threshold.

    The ranking scores (ROC AUC, PR AUC) do not depend on the threshold. Brier
    and log loss judge the probabilities themselves. The rest describe the
    yes/no decision the threshold makes.
    """
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    predicted = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    ppv = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": _safe_div(2 * ppv * sensitivity, ppv + sensitivity),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "accuracy": _safe_div(tp + tn, len(y)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_ci(
    y: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    resamples: int = 1000,
    seed: int = RANDOM_SEED,
) -> dict[str, list[float]]:
    """95% percentile intervals for each score, from stratified resamples.

    Positives and negatives are resampled separately so every draw keeps the
    test set's class balance and AUC is always defined.
    """
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    draws: dict[str, list[float]] = {}
    for _ in range(resamples):
        index = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        scores = binary_scores(y[index], proba[index], threshold)
        for key, value in scores.items():
            if key != "confusion":
                draws.setdefault(key, []).append(value)

    return {
        key: [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]
        for key, values in draws.items()
    }


def calibration_table(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """Reliability by equal-width probability bin, plus expected calibration error.

    ECE is the row-weighted average gap between what the model predicted and
    what happened in each bin. 0.05 means that, on average, a "30%" really
    came true somewhere between 25% and 35% of the time.
    """
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(proba, edges[1:-1], right=False), 0, bins - 1)

    rows = []
    ece = 0.0
    for b in range(bins):
        mask = which == b
        if not mask.any():
            continue
        predicted = float(proba[mask].mean())
        observed = float(y[mask].mean())
        ece += mask.mean() * abs(predicted - observed)
        rows.append(
            {
                "bin": [round(float(edges[b]), 2), round(float(edges[b + 1]), 2)],
                "rows": int(mask.sum()),
                "mean_predicted": round(predicted, 4),
                "observed_rate": round(observed, 4),
            }
        )
    return {"ece": round(float(ece), 4), "bins": rows}


def out_of_fold_proba(model, X: pd.DataFrame, y: pd.Series, folds: int = 5, seed: int = RANDOM_SEED) -> np.ndarray:
    """Positive-class probabilities where each row is scored by a model that never saw it."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")[:, 1]


def leave_one_site_out(
    make_model: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    target_sensitivity: float = TARGET_SENSITIVITY,
) -> list[dict[str, Any]]:
    """Train on every hospital but one, then score the one left out.

    This is the closest the data gets to "what happens at a new hospital".
    Two things can go wrong there and they show up separately:
    - ranking (AUC): can it still tell sick from well?
    - calibration-in-the-large: does its average predicted risk match the new
      site's real prevalence? Prevalence here runs from 36% to 93%, so a model
      trained without a site will be off on average even if it ranks well.
    """
    results = []
    for site in sorted(groups.unique()):
        held = (groups == site).to_numpy()
        X_train, y_train = X[~held], y[~held]
        X_test, y_test = X[held], y[held]

        model = make_model()
        threshold = choose_threshold(y_train.to_numpy(), out_of_fold_proba(model, X_train, y_train),
                                     target_sensitivity)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        row: dict[str, Any] = {
            "site": str(site),
            "rows": int(held.sum()),
            "prevalence": round(float(y_test.mean()), 4),
            "mean_predicted": round(float(proba.mean()), 4),
        }
        if y_test.nunique() == 2:
            scores = binary_scores(y_test.to_numpy(), proba, threshold)
            row.update(
                roc_auc=round(scores["roc_auc"], 4),
                sensitivity=round(scores["sensitivity"], 4),
                specificity=round(scores["specificity"], 4),
            )
        results.append(row)
    return results


def rounded(scores: dict[str, Any], digits: int = 4) -> dict[str, Any]:
    return {k: round(v, digits) if isinstance(v, float) else v for k, v in scores.items()}


def figures(
    name: str,
    y: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    out_dir: Path,
    comparison: dict[str, np.ndarray] | None = None,
) -> list[Path]:
    """ROC, precision-recall, calibration and confusion plots for the test set.

    matplotlib is imported here rather than at module level so the API, which
    never draws anything, does not pay for it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    curves = {"chosen model": proba, **(comparison or {})}
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    ax = axes[0]
    for label, scores in curves.items():
        fpr, tpr, _ = roc_curve(y, scores)
        ax.plot(fpr, tpr, lw=2 if label == "chosen model" else 1,
                label=f"{label} (AUC {roc_auc_score(y, scores):.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
    predicted = proba >= threshold
    ax.scatter([1 - (~predicted & (y == 0)).sum() / max((y == 0).sum(), 1)],
               [(predicted & (y == 1)).sum() / max((y == 1).sum(), 1)],
               color="black", zorder=5, label=f"threshold {threshold:.2f}")
    ax.set(title="ROC", xlabel="False positive rate", ylabel="Sensitivity")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    for label, scores in curves.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        ax.plot(recall, precision, lw=2 if label == "chosen model" else 1,
                label=f"{label} (AP {average_precision_score(y, scores):.3f})")
    ax.axhline(y.mean(), ls=":", color="grey", lw=1, label=f"prevalence {y.mean():.2f}")
    ax.set(title="Precision-recall", xlabel="Recall (sensitivity)", ylabel="Precision (PPV)")
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[2]
    table = calibration_table(y, proba)
    xs = [row["mean_predicted"] for row in table["bins"]]
    ys = [row["observed_rate"] for row in table["bins"]]
    sizes = [20 + 4 * row["rows"] for row in table["bins"]]
    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1)
    ax.plot(xs, ys, marker="o", lw=1.5)
    ax.scatter(xs, ys, s=sizes, alpha=0.3)
    ax.set(title=f"Calibration (ECE {table['ece']:.3f})", xlabel="Predicted risk", ylabel="Observed rate",
           xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(f"{name}: held-out test set, {len(y)} patients")
    fig.tight_layout()
    path = out_dir / f"{name}_curves.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    paths.append(path)
    return paths
