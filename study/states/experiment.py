"""One state-year target: shift score, repeats, and its report.

Protocol: docs/research/protocol-states.md.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ml import config
from ml.clinical_data import DENGUE_FEATURES
from ml.clinical_eval import choose_threshold, out_of_fold_proba
from study.shared.models import dengue_model
from study.shared.repeats import one_repeat, score, split_target, summarize
from study.states.data import Year, source_rows, target_name

SIZES = [100, 250, 500, 1000, 2500]
REPEATS = 50
SHIFT_ROWS = 20_000
LOW_CASES, HIGH_CASES = 10, 20


def shift_score(X_source: pd.DataFrame, X_target: pd.DataFrame, seed: int = config.RANDOM_SEED) -> float:
    """How well a classifier tells source rows from target rows, without labels.

    Balanced, at most SHIFT_ROWS from each side. Native NaN handling is kept on
    purpose: a change in what goes unrecorded is a change in the inputs too.
    """
    rng = np.random.default_rng(seed)
    take = min(len(X_source), len(X_target), SHIFT_ROWS)
    both = pd.concat(
        [X_source.iloc[rng.choice(len(X_source), take, replace=False)],
         X_target.iloc[rng.choice(len(X_target), take, replace=False)]],
        ignore_index=True,
    )
    side = np.r_[np.zeros(take), np.ones(take)]
    model = HistGradientBoostingClassifier(random_state=seed)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = cross_val_predict(model, both, side, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(side, p))


def sizes_for(target_rows: int) -> list[int]:
    """Only sizes the pool, half the target, can supply."""
    return [n for n in SIZES if n <= target_rows // 2]


def cases_split(rows: pd.DataFrame) -> dict[str, Any]:
    """S1 inputs: hurt shares of `intercept` draws with few and with many local cases."""
    mine = rows[rows.method == "intercept"].set_index(["repeat", "n"])
    base = rows[rows.method == "none"].set_index(["repeat", "n"])["log_loss"]
    hurt = mine["log_loss"] > base.loc[mine.index]
    low, high = mine["local_cases"] < LOW_CASES, mine["local_cases"] >= HIGH_CASES
    return {
        "low_draws": int(low.sum()), "high_draws": int(high.sum()),
        "low_hurt_share": round(float(hurt[low].mean()), 4) if low.any() else None,
        "high_hurt_share": round(float(hurt[high].mean()), 4) if high.any() else None,
    }


@dataclass
class Source:
    """One state's source model, trained on every other state's SOURCE_YEAR cases."""

    X: pd.DataFrame
    y: pd.Series
    model: Any
    oof: np.ndarray | None = None  # out-of-fold predictions on X
    threshold: float | None = None

    @property
    def rate(self) -> float:
        return float(self.y.mean())


def fit_source(source_year: Year, state: str, oof: bool = True) -> Source:
    """Fit the source model. The threshold comes from its out-of-fold predictions.

    oof=False skips those predictions, for checks that need only the fitted
    model. The fitted model is the same either way.
    """
    X_s, y_s = source_rows(source_year, state)
    model = dengue_model(DENGUE_FEATURES)
    source = Source(X_s, y_s, model)
    if oof:
        source.oof = out_of_fold_proba(model, X_s, y_s)
        source.threshold = choose_threshold(y_s.to_numpy(), source.oof)
    model.fit(X_s, y_s)
    return source


class Sources:
    """Source models fitted on first use, one per state, reused across that state's years."""

    def __init__(self, source_year: Year, oof: bool = True) -> None:
        self._year, self._oof, self._fitted = source_year, oof, {}

    def __call__(self, state: str) -> Source:
        if state not in self._fitted:
            self._fitted[state] = fit_source(self._year, state, self._oof)
        return self._fitted[state]


def run_target(
    year: int, state: str, stage: str, source: Source, X_t: pd.DataFrame, y_t: np.ndarray, jobs: int,
    repeats: int = REPEATS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    name = target_name(year, state)
    X_s, y_s, threshold = source.X, source.y, source.threshold
    source_score = score(y_s.to_numpy(), source.oof, threshold)
    X_t = X_t.reset_index(drop=True)
    p = source.model.predict_proba(X_t)[:, 1]
    sizes = sizes_for(len(y_t))
    pool, _ = split_target(y_t, config.RANDOM_SEED)
    shift = shift_score(X_s, X_t.iloc[pool])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        batches = Parallel(n_jobs=jobs)(
            delayed(one_repeat)(DENGUE_FEATURES, X_t, y_t, p, float(y_s.mean()), threshold, sizes, r)
            for r in range(repeats)
        )
    rows = pd.DataFrame([row for batch in batches for row in batch])
    curves, needed = summarize(rows, sizes)
    cases = rows[rows.method == "intercept"].groupby("n")["local_cases"].mean()
    report = {
        "name": name, "state": state, "year": year, "stage": stage,
        "sizes": sizes, "repeats": repeats,
        "source_rows": len(y_s), "target_rows": len(y_t), "target_severe": int(y_t.sum()),
        "threshold": round(threshold, 4),
        "source_rate": round(float(y_s.mean()), 4), "target_rate": round(float(y_t.mean()), 4),
        "mean_predicted": round(float(p.mean()), 4),
        "observed_over_expected": round(float(y_t.mean() / p.mean()), 4),
        "source": {k: round(v, 4) for k, v in source_score.items()},
        "target": {k: round(v, 4) for k, v in score(y_t, p, threshold).items()},
        "shift_score": round(shift, 4),
        "curves": curves,
        "repair_size": needed,
        "mean_local_cases": {str(n): round(float(v), 2) for n, v in cases.items()},
        "cases_split": cases_split(rows),
    }
    rows.insert(0, "target", name)
    return report, rows
