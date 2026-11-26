"""The state-by-state dengue study: targets, source models, shift score, verdicts.

Protocol: docs/research/protocol-states.md. The pass marks below are copied
from it and must not change after a run has started.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from ml import config
from ml.clinical_data import DENGUE_COLUMNS, DENGUE_FEATURES, DENGUE_SEVERE, dengue_confirmed, dengue_inputs
from ml.clinical_eval import choose_threshold, out_of_fold_proba
from study.transfer import one_repeat, score, split_target, summarize

SOURCE_YEAR = 2023
STAGE_A = (2023,)
STAGE_B = (2024, 2025)
MIN_CASES, MIN_SEVERE = 1_000, 100
SOURCE_SAMPLE = 200_000
SIZES = [100, 250, 500, 1000, 2500]
REPEATS = 50
SHIFT_ROWS = 20_000
LOW_CASES, HIGH_CASES, MIN_DRAWS = 10, 20, 20
S1_SHARE = 0.75
R1_SIZE, R1_SHARE, R2_SHARE = 500, 0.75, 0.25
STATE_BOOTSTRAPS = 1_000
CHUNK_ROWS = 500_000

# IBGE state codes, for readable target names.
STATES = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP", "41": "PR",
    "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}
FEATURE_NAMES = [f.name for f in DENGUE_FEATURES]


@dataclass
class Year:
    X: pd.DataFrame
    y: np.ndarray
    state: np.ndarray


def read_year(year: int) -> Year:
    """Lab-confirmed cases of one year, built chunk by chunk to keep memory low."""
    source = config.SOURCES[f"dengue_{year}"]
    if not source.path.exists():
        raise FileNotFoundError(f"{source.path} is missing. Run: python -m scripts.download_data")
    parts = []
    for chunk in pd.read_csv(source.path, usecols=DENGUE_COLUMNS, dtype=str, chunksize=CHUNK_ROWS):
        confirmed = dengue_confirmed(chunk)
        X = dengue_inputs(confirmed)[FEATURE_NAMES].astype("float32")
        X["severe"] = confirmed["CLASSI_FIN"].isin(DENGUE_SEVERE).astype("int8")
        X["state"] = confirmed["SG_UF_NOT"]
        parts.append(X)
    rows = pd.concat(parts, ignore_index=True)
    return Year(rows[FEATURE_NAMES], rows["severe"].to_numpy(), rows["state"].to_numpy(dtype=str))


def choose_targets(counts: pd.DataFrame) -> tuple[list[tuple[int, str]], list[dict[str, Any]]]:
    """Apply the inclusion rule to a table with year, state, cases and severe columns."""
    keep = (counts["cases"] >= MIN_CASES) & (counts["severe"] >= MIN_SEVERE)
    chosen = [(int(r.year), str(r.state)) for r in counts[keep].itertuples()]
    excluded = [
        {"year": int(r.year), "state": STATES.get(str(r.state), str(r.state)),
         "cases": int(r.cases), "severe": int(r.severe)}
        for r in counts[~keep].itertuples()
    ]
    return sorted(chosen), excluded


def count_cases(years: dict[int, Year]) -> pd.DataFrame:
    frames = [
        pd.DataFrame({"year": year, "state": data.state, "severe": data.y})
        .groupby(["year", "state"], as_index=False)
        .agg(cases=("severe", "size"), severe=("severe", "sum"))
        for year, data in years.items()
    ]
    return pd.concat(frames, ignore_index=True)


def source_rows(source: Year, state: str, seed: int = config.RANDOM_SEED) -> tuple[pd.DataFrame, pd.Series]:
    """Every other state, stratified down to SOURCE_SAMPLE rows."""
    away = np.flatnonzero(source.state != state)
    if len(away) > SOURCE_SAMPLE:
        away, _ = train_test_split(away, train_size=SOURCE_SAMPLE, stratify=source.y[away], random_state=seed)
        away = np.sort(away)
    return source.X.iloc[away].reset_index(drop=True), pd.Series(source.y[away])


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


def fit_source(model: Any, X_s: pd.DataFrame, y_s: pd.Series) -> tuple[Any, float, dict[str, float]]:
    """Fit the source model. The threshold comes from its out-of-fold predictions."""
    p_s = out_of_fold_proba(model, X_s, y_s)
    threshold = choose_threshold(y_s.to_numpy(), p_s)
    model.fit(X_s, y_s)
    return model, threshold, score(y_s.to_numpy(), p_s, threshold)


def run_target(
    year: int, state: str, stage: str, model: Any, threshold: float, source_score: dict[str, float],
    X_s: pd.DataFrame, y_s: pd.Series, X_t: pd.DataFrame, y_t: np.ndarray, jobs: int,
    repeats: int = REPEATS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    name = f"{STATES.get(state, state)} {year}"
    X_t = X_t.reset_index(drop=True)
    p = model.predict_proba(X_t)[:, 1]
    sizes = sizes_for(len(y_t))
    pool, _ = split_target(y_t, config.RANDOM_SEED)
    shift = shift_score(X_s, X_t.iloc[pool])
    holder = SimpleNamespace(features=DENGUE_FEATURES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        batches = Parallel(n_jobs=jobs)(
            delayed(one_repeat)(holder, X_t, y_t, p, float(y_s.mean()), threshold, sizes, r)
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


# Verdicts. Each takes the per-target reports only, so they can be tested on
# made-up reports and rerun from the saved files.

def em_change(report: dict[str, Any]) -> float:
    """EM's mean paired log-loss change from `none`, at the target's largest n."""
    return report["curves"]["em"][str(report["sizes"][-1])]["diff_from_none"]


def guard_threshold(stage_a: list[dict[str, Any]]) -> float:
    """The shift-score cut that minimizes summed log loss of "EM below it, none above".

    Candidates are 0 (never EM) and just above each target's score. Ties go to
    the lowest cut, the one that uses EM least.
    """
    scores = [r["shift_score"] for r in stage_a]
    candidates = [0.0, *sorted(s + 1e-9 for s in scores)]

    # Summing the change from `none` ranks cuts the same as summing log loss.
    def total(cut: float) -> float:
        return sum(em_change(r) for r in stage_a if r["shift_score"] < cut)

    best = min(total(c) for c in candidates)
    return min(c for c in candidates if total(c) == best)


def _bootstrap_by_state(
    reports: list[dict[str, Any]], statistic, seed: int = config.RANDOM_SEED
) -> dict[str, Any]:
    """95% interval that resamples states, keeping each state's years together."""
    by_state: dict[str, list[dict[str, Any]]] = {}
    for r in reports:
        by_state.setdefault(r["state"], []).append(r)
    states = sorted(by_state)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(STATE_BOOTSTRAPS):
        drawn = [t for s in rng.choice(states, len(states)) for t in by_state[s]]
        value = statistic(drawn)
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        return {"ci": None, "valid": 0}
    low, high = np.percentile(values, [2.5, 97.5])
    return {"ci": [round(float(low), 4), round(float(high), 4)], "valid": len(values)}


def _cv(values: list[float]) -> float | None:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or values.mean() == 0:
        return None
    return float(values.std() / values.mean())


def _cv_ratio(reports: list[dict[str, Any]]) -> float | None:
    repaired = [r for r in reports if r["repair_size"]["intercept"] is not None]
    patients = _cv([r["repair_size"]["intercept"] for r in repaired])
    cases = _cv([r["mean_local_cases"][str(r["repair_size"]["intercept"])] for r in repaired])
    if patients is None or cases is None or patients == 0:
        return None
    return cases / patients


def _spearman(reports: list[dict[str, Any]]) -> float | None:
    if len({r["shift_score"] for r in reports}) < 2 or len({em_change(r) for r in reports}) < 2:
        return None
    return float(spearmanr([r["shift_score"] for r in reports], [em_change(r) for r in reports]).statistic)


def verdicts(stage_a: list[dict[str, Any]], stage_b: list[dict[str, Any]], guard: float) -> dict[str, Any]:
    """The frozen pass rules from docs/research/protocol-states.md."""
    every = stage_a + stage_b

    eligible = [r for r in every
                if r["cases_split"]["low_draws"] >= MIN_DRAWS and r["cases_split"]["high_draws"] >= MIN_DRAWS]
    s1 = [r["name"] for r in eligible
          if r["cases_split"]["low_hurt_share"] > r["cases_split"]["high_hurt_share"]]

    ratio = _cv_ratio(every)
    ratio_ci = _bootstrap_by_state(every, _cv_ratio)
    rho = _spearman(every)
    rho_ci = _bootstrap_by_state(every, _spearman)

    em_worse = [r["name"] for r in stage_b if em_change(r) > 0]
    guarded = [r for r in stage_b if r["shift_score"] < guard]
    guard_worse = [r["name"] for r in guarded if em_change(r) > 0]
    guard_helps = [r["name"] for r in guarded if em_change(r) < 0]

    r1_pool = [r for r in every if R1_SIZE in r["sizes"]]
    r1 = [r["name"] for r in r1_pool if r["curves"]["intercept"][str(R1_SIZE)]["diff_from_none"] < 0]
    r2 = [r["name"] for r in every if em_change(r) > 0]

    def share_rule(names: list[str], of: int, needed: float) -> dict[str, Any]:
        return {"count": len(names), "of": of, "needed_share": needed, "targets": names,
                "supported": of > 0 and len(names) >= needed * of}

    return {
        "S1": {**share_rule(s1, len(eligible), S1_SHARE),
               "eligible": [r["name"] for r in eligible]},
        "S2": {"cv_ratio": None if ratio is None else round(ratio, 4), **ratio_ci,
               "repaired": [r["name"] for r in every if r["repair_size"]["intercept"] is not None],
               "supported": ratio is not None and ratio < 1 and ratio_ci["ci"] is not None
               and ratio_ci["ci"][1] < 1},
        "S3": {"spearman": None if rho is None else round(rho, 4), **rho_ci, "of": len(every),
               "supported": rho is not None and rho > 0 and rho_ci["ci"] is not None and rho_ci["ci"][0] > 0},
        "S4": {"threshold": guard, "em_worse": em_worse, "guard_worse": guard_worse,
               "guard_helps": guard_helps, "of": len(stage_b),
               "supported": len(guard_worse) < len(em_worse) and len(guard_helps) >= 1},
        "R1": share_rule(r1, len(r1_pool), R1_SHARE),
        "R2": share_rule(r2, len(every), R2_SHARE),
    }

