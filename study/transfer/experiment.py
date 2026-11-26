"""The transfer study: train away from a region or site, then repair locally.

Protocol: docs/research/protocol.md
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from ml.clinical_data import ClinicalDataset
from ml.clinical_eval import choose_threshold, out_of_fold_proba
from ml.clinical_model import untuned
from ml.config import RANDOM_SEED
from study.shared.models import dengue_model
from study.shared.repeats import one_repeat, score, summarize
from study.shared.updating import fit_shift_slope

SIZES = {"heart": [10, 25, 50], "dengue": [100, 250, 500, 1000, 2500]}
REPEATS = {"heart": 200, "dengue": 50}
H2_SIZE = {"heart": 50, "dengue": 500}
NEEDED = 8
BOOTSTRAPS = 1000


def bootstrap_shift(
    y_s: np.ndarray, p_s: np.ndarray, y_t: np.ndarray, p_t: np.ndarray
) -> tuple[list[float], list[float]]:
    """95% intervals: source minus target AUC, and target observed/expected."""
    rng = np.random.default_rng(RANDOM_SEED)
    drops, ratios = [], []
    for _ in range(BOOTSTRAPS):
        s, t = rng.integers(0, len(y_s), len(y_s)), rng.integers(0, len(y_t), len(y_t))
        ratios.append(y_t[t].mean() / p_t[t].mean())
        if len(np.unique(y_t[t])) == 2:
            drops.append(roc_auc_score(y_s[s], p_s[s]) - roc_auc_score(y_t[t], p_t[t]))
    return tuple([round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])] for x in (drops, ratios))


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    return fit_shift_slope(p, y, True)[1]


def run_target(
    name: str, data: ClinicalDataset, model: Any, X_s: pd.DataFrame, y_s: pd.Series,
    X_t: pd.DataFrame, y_t: pd.Series, jobs: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    family = data.name
    p_s = out_of_fold_proba(model, X_s, y_s)
    threshold = choose_threshold(y_s.to_numpy(), p_s)
    model.fit(X_s, y_s)
    X_t = X_t.reset_index(drop=True)
    y, y_source = y_t.to_numpy(), y_s.to_numpy()
    p = model.predict_proba(X_t)[:, 1]

    drop_ci, oe_ci = bootstrap_shift(y_source, p_s, y, p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        batches = Parallel(n_jobs=jobs)(
            delayed(one_repeat)(data.features, X_t, y, p, float(y_source.mean()), threshold, SIZES[family], r)
            for r in range(REPEATS[family])
        )
    rows = pd.DataFrame([row for batch in batches for row in batch])
    curves, needed = summarize(rows, SIZES[family])
    one_class = rows[rows.method == "refit"].groupby("n")["one_class"].mean()
    whole = score(y, p, threshold)
    internal = score(y_source, p_s, threshold)
    report = {
        "name": name,
        "family": family,
        "sizes": SIZES[family],
        "repeats": REPEATS[family],
        "source_rows": len(y_source),
        "target_rows": len(y),
        "target_minority": int(min(y.sum(), len(y) - y.sum())),
        "threshold": round(threshold, 4),
        "source_rate": round(float(y_source.mean()), 4),
        "target_rate": round(float(y.mean()), 4),
        "mean_predicted": round(float(p.mean()), 4),
        "source": {k: round(v, 4) for k, v in internal.items()},
        "target": {k: round(v, 4) for k, v in whole.items()},
        "calibration_slope": round(calibration_slope(y, p), 4),
        "observed_over_expected": round(float(y.mean() / p.mean()), 4),
        "auc_drop_ci": drop_ci,
        "oe_ci": oe_ci,
        "curves": curves,
        "repair_size": needed,
        "one_class_share": {str(n): round(float(v), 3) for n, v in one_class.items()},
    }
    rows.insert(0, "target", name)
    return report, rows


def run_heart(data: ClinicalDataset, jobs: int):
    for site in sorted(data.groups.unique()):
        held = data.groups == site
        model = untuned("logistic", data.features, data.missing_indicators)
        yield run_target(
            f"heart: {site}", data, model, data.X[~held].reset_index(drop=True),
            data.y[~held].reset_index(drop=True), data.X[held], data.y[held], jobs,
        )


def run_dengue(data: ClinicalDataset, jobs: int):
    def model():
        return dengue_model(data.features, data.missing_indicators)

    for region in sorted(data.groups.unique()):
        held = data.groups == region
        yield run_target(
            f"dengue: {region}", data, model(), data.X[~held].reset_index(drop=True),
            data.y[~held].reset_index(drop=True), data.X[held], data.y[held], jobs,
        )
    if data.external is None:
        raise FileNotFoundError("the dengue 2025 file is missing. Run: python -m scripts.download_data")
    yield run_target(
        "dengue: 2025", data, model(), data.X, data.y, data.external.X, data.external.y, jobs
    )


def verdicts(transfers: list[dict[str, Any]]) -> dict[str, Any]:
    """The frozen pass rules from docs/research/protocol.md."""

    def loss(t: dict[str, Any], method: str, n: int) -> float:
        return t["curves"][method][str(n)]["log_loss"]["mean"]

    ranking = [t["name"] for t in transfers if t["auc_drop_ci"][0] > 0]
    rates = [t["name"] for t in transfers if not t["oe_ci"][0] <= 1 <= t["oe_ci"][1]]
    h2 = [t["name"] for t in transfers
          if loss(t, "intercept", H2_SIZE[t["family"]]) < loss(t, "none", H2_SIZE[t["family"]])]
    h3 = [t["name"] for t in transfers
          if loss(t, "intercept", t["sizes"][0]) < loss(t, "platt", t["sizes"][0])]
    h4 = [t["name"] for t in transfers
          if all(loss(t, "intercept", n) < loss(t, "refit", n) for n in t["sizes"])]
    h5 = [t["name"] for t in transfers
          if loss(t, "em", t["sizes"][-1]) > loss(t, "none", t["sizes"][-1])]
    total = len(transfers)

    def verdict(names: list[str], needed: int) -> dict[str, Any]:
        return {"count": len(names), "of": total, "needed": needed, "targets": names,
                "supported": len(names) >= needed}

    return {
        "H1": {"ranking_broke": ranking, "rates_broke": rates, "of": total,
               "supported": len(rates) > len(ranking), "exploratory": True},
        "H2": verdict(h2, NEEDED),
        "H3": verdict(h3, NEEDED),
        "H4": verdict(h4, NEEDED),
        "H5": verdict(h5, 2),
    }
