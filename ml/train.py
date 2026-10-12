"""Train the models and write them to models/.

    python -m ml.train                    # the symptom ranker
    python -m ml.train --no-search        # symptom ranker, base settings, no grid search
    python -m ml.train --with-feedback    # fold in corrected predictions from Postgres
    python -m ml.train --model heart      # one clinical risk model
    python -m ml.train --model clinical   # heart, diabetes and liver
    python -m ml.train --model all        # everything

Each artifact is a self-describing dict: the fitted estimator, the exact feature
order the API must rebuild, and the scores this run measured. Alongside it goes
a <name>_summary.json with the same scores in a form you can read or diff.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from ml.clinical_data import LOADERS, ClinicalDataset
from ml.clinical_eval import (
    TARGET_SENSITIVITY,
    binary_scores,
    bootstrap_ci,
    calibration_table,
    choose_threshold,
    figures,
    leave_one_site_out,
    out_of_fold_proba,
    rounded,
)
from ml.clinical_model import compare, fit_final, pick, untuned
from ml.config import ARTIFACT_VERSION, MODELS_DIR, RANDOM_SEED, ROOT, SOURCES
from ml.datasets import Dataset, load_symptoms, unique_symptom_patterns
from ml.evaluate import cross_validated_scores, held_out_noise, holdout_scores, noise_robustness
from ml.model import base_model, baselines, recall_search, search_summary

FIGURES_DIR = ROOT / "docs" / "figures"
CLINICAL = list(LOADERS)


def _log(dataset: Dataset) -> None:
    for note in dataset.notes:
        print(f"    - {note}")


def disease_symptom_map(X: pd.DataFrame, y: pd.Series) -> dict[str, list[str]]:
    """Symptoms that appear in at least one recorded pattern for each disease.

    The API uses this to tell a user which of their reported symptoms actually
    support a ranked prediction, which is cheaper and steadier than per-request
    feature attribution.
    """
    mapping: dict[str, list[str]] = {}
    for disease, group in X.groupby(y.to_numpy()):
        present = group.sum(axis=0)
        mapping[str(disease)] = sorted(present[present > 0].index)
    return mapping


def load_feedback(feature_names: list[str]) -> Dataset | None:
    """Pull reviewer-corrected predictions out of Postgres as extra training rows.

    Every prediction the API serves is logged, and a reviewer can record what the
    condition actually was. Those are cases the model already got wrong, so they
    are worth more per row than anything in the original file.

    Returns None when the database is unreachable or holds no corrections yet.
    Training must not depend on it.
    """
    from app.db import fetch_training_feedback  # late import: training works with no database

    rows = fetch_training_feedback()
    if not rows:
        return None

    index = {name: i for i, name in enumerate(feature_names)}
    matrix: list[list[int]] = []
    labels: list[str] = []
    for symptoms, actual in rows:
        known = [s for s in symptoms if s in index]
        if not known:
            continue
        vector = [0] * len(feature_names)
        for symptom in known:
            vector[index[symptom]] = 1
        matrix.append(vector)
        labels.append(actual)

    if not matrix:
        return None

    X = pd.DataFrame(matrix, columns=feature_names).astype("int8")
    return Dataset(
        X=X,
        y=pd.Series(labels),
        name="feedback",
        notes=[f"{len(X)} corrected predictions pulled from Postgres"],
    )


def train(search: bool = True, with_feedback: bool = False) -> dict[str, Any]:
    print("[symptom] loading")
    raw = load_symptoms("train")
    _log(raw)

    unique = unique_symptom_patterns(raw)
    _log(unique)

    X, y = unique.X, unique.y
    feedback_rows = 0
    if with_feedback:
        feedback = load_feedback(unique.feature_names)
        if feedback is None:
            print("    - no feedback available, training on the base data only")
        else:
            _log(feedback)
            feedback_rows = len(feedback.X)
            X = pd.concat([X, feedback.X], ignore_index=True)
            y = pd.concat([y, feedback.y], ignore_index=True)

    started = time.perf_counter()
    metrics: dict[str, Any] = {}

    if search:
        print("[symptom] grid searching on macro recall")
        grid = recall_search()
        grid.fit(X, y)
        metrics["search"] = search_summary(grid)
        model = grid.best_estimator_
        print(f"    best params: {grid.best_params_}")
        print(
            f"    cv macro recall {metrics['search']['cv_recall_macro']:.4f}"
            f"  accuracy {metrics['search']['cv_accuracy']:.4f}"
        )
    else:
        model = base_model()

    print("[symptom] cross-validating the chosen model")
    metrics["cross_validation"] = cross_validated_scores(model, X, y)

    print("[symptom] fitting on all training rows")
    model.fit(X, y)

    test = load_symptoms("test")
    aligned = test.X.reindex(columns=unique.feature_names, fill_value=0)
    metrics["holdout"] = holdout_scores(model, aligned, test.y)

    print("[symptom] measuring noise robustness")
    metrics["noise_1_dropped_1_added"] = noise_robustness(model, unique.X, unique.y, drop=1, add=1)
    metrics["noise_2_dropped_2_added"] = noise_robustness(model, unique.X, unique.y, drop=2, add=2)

    for label, scores in metrics.items():
        # per_class_recall is 41 entries; it belongs in the artifact, not the log.
        shown = {k: v for k, v in scores.items() if k != "per_class_recall"}
        print(f"    {label}: {shown}")

    print("[symptom] comparing against simpler models on held-out noisy patterns")
    metrics["comparison"] = compare_symptom_models(model, unique.X, unique.y)
    for row in metrics["comparison"]:
        print(f"    {row}")

    elapsed = round(time.perf_counter() - started, 1)

    return {
        "artifact_version": ARTIFACT_VERSION,
        "name": "symptom",
        "kind": "multiclass",
        "model": model,
        "feature_names": unique.feature_names,
        "classes": [str(c) for c in model.classes_],
        "disease_symptoms": disease_symptom_map(unique.X, unique.y),
        "feature_importances": [float(v) for v in model.feature_importances_],
        "metrics": metrics,
        "dataset": {
            "source": SOURCES["symptom_train"].url,
            "description": SOURCES["symptom_train"].description,
            "rows_raw": int(len(raw.X)),
            "rows_unique": int(len(unique.X)),
            "rows_feedback": feedback_rows,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "train_seconds": elapsed,
    }


def _split(data: ClinicalDataset):
    """80/20, stratified on the label, and on the hospital too where there is one.

    Stratifying on the hospital keeps the test set's mix of sites, and with it
    its prevalence, close to the training set's.
    """
    strata = data.y.astype(str)
    if data.groups is not None:
        strata = strata + "_" + data.groups.astype(str)
    return train_test_split(data.X, data.y, test_size=0.2, stratify=strata, random_state=RANDOM_SEED)


def _reference_values(data: ClinicalDataset, X: pd.DataFrame) -> dict[str, float]:
    """A typical value per input: the training median, or the most common category.

    The API swaps each supplied value for this one to see how much it moved the
    risk. That is how a single prediction gets explained.
    """
    values: dict[str, float] = {}
    for spec in data.features:
        column = X[spec.name].dropna()
        values[spec.name] = float(column.median() if spec.kind == "numeric" else column.mode().iloc[0])
    return values


def _odds_ratios(model) -> dict[str, float] | None:
    """Odds ratio per standard deviation (or per category) when the winner is logistic.

    This reads the tuned pipeline that calibration wraps, fitted on the
    training split. Calibration rescales the output but keeps the direction
    each input pushes, so the ratios still say which way that is.
    """
    base = model.estimator
    final = base.named_steps.get("model") if hasattr(base, "named_steps") else None
    if final is None or not hasattr(final, "coef_"):
        return None
    names = base[:-1].get_feature_names_out()
    return {str(n): round(float(np.exp(c)), 3) for n, c in zip(names, final.coef_[0], strict=True)}


def train_clinical(name: str) -> dict[str, Any]:
    print(f"[{name}] loading")
    data = LOADERS[name]()
    for note in data.notes:
        print(f"    - {note}")

    started = time.perf_counter()
    X_train, X_test, y_train, y_test = _split(data)
    print(f"    - split: {len(X_train)} train, {len(X_test)} test")

    print(f"[{name}] comparing model families (nested CV on the training split)")
    table = compare(data.features, X_train, y_train, data.missing_indicators)
    for row in table:
        tag = "  (ablation)" if row["ablation"] else ""
        print(
            f"    {row['family']:16s} AUC {row['roc_auc']:.4f} +/- {row['roc_auc_std']:.4f}"
            f"  PR-AUC {row['pr_auc']:.4f}  Brier {row['brier']:.4f}{tag}"
        )
    family = pick(table)
    print(f"    chose {family}")

    model, params = fit_final(family, data.features, X_train, y_train, data.missing_indicators)

    print(f"[{name}] choosing the threshold on out-of-fold training predictions")
    oof = out_of_fold_proba(model, X_train, y_train)
    threshold = choose_threshold(y_train.to_numpy(), oof, TARGET_SENSITIVITY)
    print(f"    flag at risk >= {threshold:.3f} to catch {TARGET_SENSITIVITY:.0%} of cases")

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    print(f"[{name}] scoring the held-out test set")
    test = rounded(binary_scores(y_test.to_numpy(), proba, threshold))
    intervals = bootstrap_ci(y_test.to_numpy(), proba, threshold)
    calibration = calibration_table(y_test.to_numpy(), proba)
    for key in ["roc_auc", "pr_auc", "brier", "sensitivity", "specificity", "ppv", "npv"]:
        low, high = intervals[key]
        print(f"    {key:12s} {test[key]:.3f}  [{low:.3f}, {high:.3f}]")
    print(f"    ece          {calibration['ece']:.3f}")

    importance = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=RANDOM_SEED
    )
    importances = sorted(
        (
            {"feature": f, "auc_drop": round(float(m), 4), "std": round(float(s), 4)}
            for f, m, s in zip(
                data.feature_names, importance.importances_mean, importance.importances_std, strict=True
            )
        ),
        key=lambda row: row["auc_drop"],
        reverse=True,
    )

    metrics: dict[str, Any] = {
        "comparison": table,
        "chosen_family": family,
        "chosen_params": params,
        "threshold": {
            "value": round(threshold, 4),
            "target_sensitivity": TARGET_SENSITIVITY,
            "chosen_on": "out-of-fold predictions on the training split",
        },
        "test": {**test, "rows": int(len(y_test)), "prevalence": round(float(y_test.mean()), 4)},
        "test_ci95": intervals,
        "calibration": calibration,
        "permutation_importance": importances,
    }
    odds = _odds_ratios(model)
    if odds:
        metrics["odds_ratios"] = odds

    if data.groups is not None:
        print(f"[{name}] leave one hospital out")

        # Uses every row: each hospital takes a turn as the test set, so the
        # 80/20 split above does not apply here.
        metrics["leave_one_site_out"] = leave_one_site_out(
            lambda: untuned(family, data.features, data.missing_indicators), data.X, data.y, data.groups
        )
        for row in metrics["leave_one_site_out"]:
            print(f"    {row}")

    for path in figures(name, y_test.to_numpy(), proba, threshold, FIGURES_DIR):
        print(f"    drew {path.relative_to(ROOT).as_posix()}")

    return {
        "artifact_version": ARTIFACT_VERSION,
        "name": name,
        "kind": "binary",
        "model": model,
        "feature_names": data.feature_names,
        "feature_specs": [spec.as_dict() for spec in data.features],
        "classes": [0, 1],
        "target": data.target,
        "threshold": threshold,
        "reference_values": _reference_values(data, X_train),
        "metrics": metrics,
        "dataset": {
            "sources": [SOURCES[k].url for k in SOURCES if k == name or k.startswith(f"{name}_")],
            "rows": int(len(data.X)),
            "rows_train": int(len(X_train)),
            "rows_test": int(len(X_test)),
            "prevalence": round(data.prevalence, 4),
            "cleaning": data.notes,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "train_seconds": round(time.perf_counter() - started, 1),
    }


def compare_symptom_models(forest, X: pd.DataFrame, y: pd.Series) -> list[dict[str, Any]]:
    """Clean and noisy held-out scores for the forest and each baseline.

    Clean scores tie at or near 1.000 for everyone, so the noisy columns and
    log loss carry the comparison.
    """
    rows = []
    for name, candidate in {"forest": forest, **baselines()}.items():
        clean = cross_validated_scores(candidate, X, y)
        row: dict[str, Any] = {
            "model": name,
            "accuracy": clean["accuracy"],
            "top_3_accuracy": clean["top_3_accuracy"],
            "log_loss": clean["log_loss"],
        }
        for drop_add in (1, 2, 3):
            noisy = held_out_noise(candidate, X, y, drop=drop_add, add=drop_add)
            row[f"noise_{drop_add}_accuracy"] = noisy["accuracy"]
            row[f"noise_{drop_add}_top_3"] = noisy["top_3_accuracy"]
        rows.append(row)
    return rows


def write(bundle: dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    name = bundle["name"]
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(bundle, path, compress=3)
    print(f"    wrote {path.name} ({path.stat().st_size / 1024:.0f} KB)")

    summary = {
        "name": name,
        "kind": bundle["kind"],
        "classes": len(bundle["classes"]),
        "features": len(bundle["feature_names"]),
        "trained_at": bundle["trained_at"],
        "sklearn_version": bundle["sklearn_version"],
        "dataset": bundle["dataset"],
        "metrics": bundle["metrics"],
    }
    summary_path = MODELS_DIR / f"{name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"    wrote {summary_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        choices=["symptom", *CLINICAL, "clinical", "all"],
        default="symptom",
        help="which model to train (default: symptom)",
    )
    parser.add_argument("--no-search", action="store_true", help="symptom model: skip the grid search")
    parser.add_argument(
        "--with-feedback",
        action="store_true",
        help="symptom model: add reviewer-corrected predictions from Postgres to the training data",
    )
    args = parser.parse_args()

    if args.model in {"symptom", "all"}:
        write(train(search=not args.no_search, with_feedback=args.with_feedback))
    if args.model in {"clinical", "all"}:
        names = CLINICAL
    elif args.model in CLINICAL:
        names = [args.model]
    else:
        names = []
    for name in names:
        write(train_clinical(name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
