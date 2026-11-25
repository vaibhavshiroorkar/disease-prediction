"""Shortcut audits: what a model could score without reading the patient."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ml import config
from ml.clinical_data import HEART_COLUMNS, LOADERS, ClinicalDataset

FOLDS = 5
MIN_PER_CLASS = 10


def _folds(y: np.ndarray) -> np.ndarray:
    folds = np.empty(len(y), dtype=int)
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    for k, (_, test) in enumerate(cv.split(np.zeros(len(y)), y)):
        folds[test] = k
    return folds


def site_rate_scores(y: np.ndarray, sites: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Each row's score is its site's positive rate among the other folds."""
    scores = np.empty(len(y))
    for fold in np.unique(folds):
        rates = pd.Series(y[folds != fold]).groupby(sites[folds != fold]).mean()
        scores[folds == fold] = pd.Series(sites[folds == fold]).map(rates).to_numpy()
    return scores


def within_site_auc(y: np.ndarray, p: np.ndarray, sites: np.ndarray) -> dict[str, Any]:
    """AUC inside each site, and the mean weighted by positive-negative pairs.

    A pooled AUC also counts pairs of patients from different sites, and
    knowing the site is enough to order many of those.
    """
    per_site, weights = {}, {}
    for site in sorted(set(sites)):
        mask = sites == site
        pairs = int((y[mask] == 1).sum() * (y[mask] == 0).sum())
        if pairs:
            per_site[str(site)] = float(roc_auc_score(y[mask], p[mask]))
            weights[str(site)] = pairs
    total = sum(weights.values())
    return {"per_site": per_site, "mean": sum(per_site[s] * weights[s] for s in per_site) / total}


def _auc_from_missingness(missing: pd.DataFrame, target: np.ndarray) -> float:
    """Cross-validated AUC, one-vs-rest macro average when there are several classes."""
    model = HistGradientBoostingClassifier(max_depth=3, random_state=config.RANDOM_SEED)
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    proba = cross_val_predict(model, missing.astype(float), target, cv=cv, method="predict_proba")
    if proba.shape[1] == 2:
        return float(roc_auc_score(target, proba[:, 1]))
    return float(roc_auc_score(target, proba, multi_class="ovr"))


def _raw_heart() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frames = []
    for site in config.HEART_SITES:
        path = config.SOURCES[f"heart_{site}"].path
        frames.append(pd.read_csv(path, names=HEART_COLUMNS, na_values="?").assign(site=site))
    df = pd.concat(frames, ignore_index=True)
    df[["chol", "trestbps"]] = df[["chol", "trestbps"]].replace(0, np.nan)
    inputs = df[[c for c in HEART_COLUMNS if c != "num"]]
    return inputs.isna(), (df["num"] > 0).astype(int).to_numpy(), df["site"].to_numpy()


def missingness_audit(datasets: dict[str, ClinicalDataset]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, data in datasets.items():
        missing = data.X.isna()
        out[name] = {
            "missing_share": round(float(missing.to_numpy().mean()), 4),
            "predicts_label_auc": round(_auc_from_missingness(missing, data.y.to_numpy()), 4),
        }
    raw, y, sites = _raw_heart()
    out["heart"]["predicts_site_auc"] = round(
        _auc_from_missingness(datasets["heart"].X.isna(), datasets["heart"].groups.to_numpy()), 4
    )
    out["heart_raw_13_inputs"] = {
        "missing_share": round(float(raw.to_numpy().mean()), 4),
        "predicts_label_auc": round(_auc_from_missingness(raw, y), 4),
        "predicts_site_auc": round(_auc_from_missingness(raw, sites), 4),
    }
    return out


def site_audit(data: ClinicalDataset, oof: np.ndarray) -> dict[str, Any]:
    y, sites = data.y.to_numpy(), data.groups.astype(str).to_numpy()
    shortcut = site_rate_scores(y, sites, _folds(y))
    within = within_site_auc(y, oof, sites)
    return {
        "site_rate_alone_auc": round(float(roc_auc_score(y, shortcut)), 4),
        "pooled_auc": round(float(roc_auc_score(y, oof)), 4),
        "within_site_auc": round(within["mean"], 4),
        "per_site_auc": {k: round(v, 4) for k, v in within["per_site"].items()},
    }


def load_all() -> dict[str, ClinicalDataset]:
    return {name: load() for name, load in LOADERS.items()}
