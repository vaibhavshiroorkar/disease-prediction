"""Ten transfers: train away from a target, then repair with n local patients."""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.clinical_data import ClinicalDataset
from ml.clinical_eval import choose_threshold, out_of_fold_proba
from ml.clinical_model import fixed, preprocessor, untuned
from ml.config import MODELS_DIR, RANDOM_SEED
from study.updating import (
    apply_shift_slope,
    em_rate,
    fit_shift_slope,
    local_rate,
    reweight,
)

METHODS = ("none", "em", "intercept", "platt", "refit", "rate")
SIZES = {"heart": [10, 25, 50], "dengue": [100, 250, 500, 1000, 2500]}
REPEATS = {"heart": 200, "dengue": 50}
H2_SIZE = {"heart": 50, "dengue": 500}
TEST_CAP = 20_000
RELIABLE = 0.90
NEEDED = 8
BOOTSTRAPS = 1000


def score(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    p = np.asarray(p, dtype=float)
    flagged = p >= threshold
    return {
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "gap": float(p.mean() - np.mean(y)),
        "sensitivity": float(flagged[y == 1].mean()),
        "specificity": float((~flagged[y == 0]).mean()),
    }


def local_refit(data: ClinicalDataset, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame) -> np.ndarray:
    """A logistic model from the local patients alone, with the served preprocessing."""
    # A column with no value at all in the sample cannot be imputed.
    features = [f for f in data.features if X[f.name].notna().any()]
    model = Pipeline(
        [
            ("prep", preprocessor(features, scale=True, missing_indicators=False)),
            ("model", LogisticRegression(C=1.0, max_iter=5000)),
        ]
    )
    names = [f.name for f in features]
    model.fit(X[names], y)
    return model.predict_proba(X_test[names])[:, 1]


def one_repeat(
    data: ClinicalDataset, X: pd.DataFrame, y: np.ndarray, p: np.ndarray,
    source_rate: float, threshold: float, sizes: list[int], repeat: int,
) -> list[dict[str, Any]]:
    seed = RANDOM_SEED + repeat
    pool, test = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=seed)
    if len(test) > TEST_CAP:
        _, test = train_test_split(test, test_size=TEST_CAP, stratify=y[test], random_state=seed)
    order = np.random.default_rng(seed).permutation(pool)
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
            "refit": rate if one_class else local_refit(data, X.iloc[local], y_local, X.iloc[test]),
            "rate": rate,
        }
        for method in METHODS:
            rows.append(
                {"repeat": repeat, "n": n, "method": method, "one_class": one_class,
                 **score(y[test], updated[method], threshold)}
            )
    return rows


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


def repair_size(share_better: dict[int, float]) -> int | None:
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
                for metric in ("log_loss", "brier", "auc", "gap", "sensitivity", "specificity")
            }
            diff = at.set_index(["repeat", "n"])["log_loss"] - base.loc[at.set_index(["repeat", "n"]).index]
            share[n] = float((diff < 0).mean())
            entry["diff_from_none"] = round(float(diff.mean()), 5)
            entry["share_better_than_none"] = round(share[n], 3)
            curves[method][str(n)] = entry
        if method != "none":
            sizes_needed[method] = repair_size(share)
    return curves, sizes_needed


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
            delayed(one_repeat)(data, X_t, y, p, float(y_source.mean()), threshold, SIZES[family], r)
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
    params = json.loads((MODELS_DIR / "dengue_summary.json").read_text(encoding="utf-8"))["metrics"][
        "chosen_params"
    ]

    def model():
        return fixed("boosting", data.features, data.missing_indicators, params)

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
