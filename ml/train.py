"""Tune and train the symptom classifier, then write models/symptom.joblib.

    python -m ml.train                 # grid search, then fit the winner
    python -m ml.train --no-search     # skip the search, use the base settings
    python -m ml.train --with-feedback # fold in corrected predictions from Postgres

The artifact is a self-describing dict: the fitted estimator, the exact feature
order the API must rebuild, the class labels, and the scores this run measured.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
import sklearn

from ml.config import ARTIFACT_VERSION, MODELS_DIR, SOURCES
from ml.datasets import Dataset, load_symptoms, unique_symptom_patterns
from ml.evaluate import cross_validated_scores, holdout_scores, noise_robustness
from ml.model import base_model, recall_search, search_summary


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

    elapsed = round(time.perf_counter() - started, 1)
    for label, scores in metrics.items():
        # per_class_recall is 41 entries; it belongs in the artifact, not the log.
        shown = {k: v for k, v in scores.items() if k != "per_class_recall"}
        print(f"    {label}: {shown}")

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


def write(bundle: dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "symptom.joblib"
    joblib.dump(bundle, path, compress=3)
    print(f"    wrote {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")

    summary = {
        "classes": len(bundle["classes"]),
        "features": len(bundle["feature_names"]),
        "trained_at": bundle["trained_at"],
        "sklearn_version": bundle["sklearn_version"],
        "metrics": bundle["metrics"],
    }
    (MODELS_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("    wrote summary.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-search", action="store_true", help="skip the grid search")
    parser.add_argument(
        "--with-feedback",
        action="store_true",
        help="add reviewer-corrected predictions from Postgres to the training data",
    )
    args = parser.parse_args()

    write(train(search=not args.no_search, with_feedback=args.with_feedback))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
