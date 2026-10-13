"""Candidate models for the binary risk tasks, and how one gets picked.

Four families compete on every dataset:

- a dummy that predicts the training prevalence for everyone. It scores 0.5
  AUC by construction. Any model has to clear it to be worth anything.
- logistic regression, the model a clinician could check by hand.
- a random forest.
- histogram gradient boosting.

Selection is nested. The outer loop (5 folds, repeated 3 times) estimates how
well each family does. The inner loop tunes that family's few settings. All of
it runs on the training split, so the test split stays untouched until the
winner is chosen.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml.clinical_data import FeatureSpec
from ml.config import RANDOM_SEED

OUTER_FOLDS = 5
OUTER_REPEATS = 3
INNER_FOLDS = 5

# Rows in the comparison table that measure a design choice rather than compete.
ABLATIONS = {"logistic_no_log"}


def preprocessor(
    features: list[FeatureSpec], scale: bool, missing_indicators: bool, log_skewed: bool = True
) -> ColumnTransformer:
    """Impute, then encode. Fit inside each fold, so no test statistics leak in.

    Skewed lab values go through log1p first. Numbers then get the median and,
    where allowed, a "was missing" column. Binary
    and categorical inputs get the most common value. Categories are one-hot
    encoded so logistic regression does not read chest pain type 4 as "twice"
    type 2.
    """
    skewed = [f.name for f in features if f.kind == "numeric" and f.skewed and log_skewed]
    numeric = [f.name for f in features if f.kind == "numeric" and f.name not in skewed]
    binary = [f.name for f in features if f.kind == "binary"]
    category = [f.name for f in features if f.kind == "category"]

    def numeric_steps(log: bool) -> Pipeline:
        steps: list[tuple[str, Any]] = []
        if log:
            steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        steps.append(("impute", SimpleImputer(strategy="median", add_indicator=missing_indicators)))
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", numeric_steps(log=False), numeric))
    if skewed:
        transformers.append(("skewed", numeric_steps(log=True), skewed))
    if binary:
        transformers.append(("binary", SimpleImputer(strategy="most_frequent"), binary))
    if category:
        transformers.append(
            (
                "category",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                category,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def candidates(
    features: list[FeatureSpec], missing_indicators: bool, seed: int = RANDOM_SEED
) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Each family as (pipeline, small grid).

    The grids are tiny on purpose. With 450 to 730 training rows, a wide grid
    mostly finds settings that suit the particular folds.

    Gradient boosting could handle NaN on its own, but it gets imputed input
    like everyone else. Its native NaN handling learns from missingness, which
    on the heart data means learning the hospital.
    """
    def pipe(model: Any, scale: bool = False, log_skewed: bool = True) -> Pipeline:
        prep = preprocessor(features, scale, missing_indicators, log_skewed)
        return Pipeline([("prep", prep), ("model", model)])

    families = {
        "dummy": (pipe(DummyClassifier(strategy="prior")), {}),
        "logistic": (
            pipe(LogisticRegression(max_iter=5000), scale=True),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "forest": (
            pipe(RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)),
            {"model__min_samples_leaf": [1, 5, 15], "model__max_features": ["sqrt", 0.5]},
        ),
        "boosting": (
            pipe(
                HistGradientBoostingClassifier(
                    random_state=seed, early_stopping=False, l2_regularization=1.0
                )
            ),
            {"model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 3], "model__max_iter": [100, 300]},
        ),
    }
    if any(f.skewed for f in features):
        # Not a contender. It is the same logistic regression without the log
        # step, kept so every training run shows what that step is worth.
        families["logistic_no_log"] = (
            pipe(LogisticRegression(max_iter=5000), scale=True, log_skewed=False),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        )
    return families


def tuned(pipeline: Pipeline, grid: dict[str, list[Any]], seed: int = RANDOM_SEED) -> Any:
    """The family as one estimator: tune on the folds it is given, then refit."""
    if not grid:
        return pipeline
    return GridSearchCV(
        pipeline,
        grid,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed),
        n_jobs=1,
    )


def compare(
    features: list[FeatureSpec],
    X: pd.DataFrame,
    y: pd.Series,
    missing_indicators: bool,
    seed: int = RANDOM_SEED,
    families: tuple[str, ...] | None = None,
    repeats: int = OUTER_REPEATS,
) -> list[dict[str, Any]]:
    """Nested cross-validation of every family, best first.

    Each row carries the mean and spread over the outer folds (15 by default).
    The spread is the part to read: two families whose means sit within one
    standard deviation of each other are not meaningfully different on this
    much data.
    """
    outer = RepeatedStratifiedKFold(n_splits=OUTER_FOLDS, n_repeats=repeats, random_state=seed)
    rows = []
    for family, (pipeline, grid) in candidates(features, missing_indicators, seed).items():
        if families is not None and family not in families:
            continue
        result = cross_validate(
            tuned(pipeline, grid, seed),
            X,
            y,
            cv=outer,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"},
            n_jobs=-1,
        )
        auc = result["test_roc_auc"]
        rows.append(
            {
                "family": family,
                "roc_auc": round(float(np.mean(auc)), 4),
                "roc_auc_std": round(float(np.std(auc)), 4),
                "pr_auc": round(float(np.mean(result["test_pr_auc"])), 4),
                "brier": round(float(-np.mean(result["test_brier"])), 4),
                "fit_seconds": round(float(np.sum(result["fit_time"])), 1),
                "ablation": family in ABLATIONS,
            }
        )
    return sorted(rows, key=lambda row: row["roc_auc"], reverse=True)


def pick(table: list[dict[str, Any]]) -> str:
    """The family with the best mean AUC, preferring logistic regression on a near tie.

    "Near" means within half the winner's fold-to-fold standard deviation. On a
    gap that small the ranking could flip with a different seed, and the
    simpler model is easier to explain and less likely to be fitting noise.
    """
    contenders = [row for row in table if not row["ablation"]]
    best = contenders[0]
    logistic = next(row for row in contenders if row["family"] == "logistic")
    if best["roc_auc"] - logistic["roc_auc"] <= best["roc_auc_std"] / 2:
        return "logistic"
    return best["family"]


def untuned(
    family: str, features: list[FeatureSpec], missing_indicators: bool, seed: int = RANDOM_SEED
) -> CalibratedClassifierCV:
    """The family with its search still inside, so tuning happens on whatever rows it is fit on.

    The leave-one-site-out check uses this. The left-out hospital must not
    influence the settings, and fitting the search afresh guarantees that.
    """
    pipeline, grid = candidates(features, missing_indicators, seed)[family]
    return CalibratedClassifierCV(
        tuned(pipeline, grid, seed),
        method="sigmoid",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
    )


def fixed(
    family: str,
    features: list[FeatureSpec],
    missing_indicators: bool,
    params: dict[str, Any],
    seed: int = RANDOM_SEED,
) -> CalibratedClassifierCV:
    """The family with settings already chosen, ready to fit.

    For data too big to re-run the search inside every check. The settings
    were picked on the training split, so a check that holds out part of that
    split (one region, say) has seen it once through the search. With four
    settings and 160,000 rows that leak is small, and the training log says
    which way the check was run.
    """
    pipeline, _ = candidates(features, missing_indicators, seed)[family]
    pipeline.set_params(**{f"model__{key}": value for key, value in params.items()})
    return CalibratedClassifierCV(
        pipeline, method="sigmoid", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    )


def fit_final(
    family: str,
    features: list[FeatureSpec],
    X: pd.DataFrame,
    y: pd.Series,
    missing_indicators: bool,
    seed: int = RANDOM_SEED,
) -> tuple[CalibratedClassifierCV, dict[str, Any]]:
    """Tune the chosen family on all training rows, then wrap it for calibration.

    Returns the calibrated model unfitted, plus the settings the search chose.
    Sigmoid calibration fits two numbers, which is about what a few hundred rows
    can support. Isotonic fits a step function and overfits at this size.
    """
    pipeline, grid = candidates(features, missing_indicators, seed)[family]
    search = tuned(pipeline, grid, seed)
    search.fit(X, y)
    if isinstance(search, GridSearchCV):
        best = search.best_estimator_
        params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    else:
        best, params = search, {}

    calibrated = CalibratedClassifierCV(
        best, method="sigmoid", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    )
    return calibrated, params
