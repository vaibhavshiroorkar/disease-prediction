"""Train the symptom classifier and write models/symptom.joblib.

    python -m ml.train

The artifact is a self-describing dict: the fitted estimator, the exact feature
order the API must rebuild, the class labels, and the scores this run measured.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
import sklearn

from ml.config import ARTIFACT_VERSION, MODELS_DIR, SOURCES
from ml.datasets import Dataset, load_symptoms, unique_symptom_patterns
from ml.evaluate import cross_validated_scores, holdout_scores
from ml.model import base_model


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


def train() -> dict[str, Any]:
    print("[symptom] loading")
    raw = load_symptoms("train")
    _log(raw)

    unique = unique_symptom_patterns(raw)
    _log(unique)

    started = time.perf_counter()
    model = base_model()

    print("[symptom] cross-validating")
    metrics: dict[str, Any] = {"cross_validation": cross_validated_scores(model, unique.X, unique.y)}

    print("[symptom] fitting on all training rows")
    model.fit(unique.X, unique.y)

    test = load_symptoms("test")
    aligned = test.X.reindex(columns=unique.feature_names, fill_value=0)
    metrics["holdout"] = holdout_scores(model, aligned, test.y)

    elapsed = round(time.perf_counter() - started, 1)
    for label, scores in metrics.items():
        print(f"    {label}: {scores}")

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
    write(train())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
