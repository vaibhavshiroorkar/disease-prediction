"""Loads and holds the trained artifact.

One registry instance is created at import and shared by every request. The model
is read-only after loading, and scikit-learn's predict_proba releases the GIL for
the tree traversal, so no locking is needed around inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from app.config import MODELS_DIR

log = logging.getLogger(__name__)

# Artifacts written by ml/train.py. Must match ml.config.ARTIFACT_VERSION.
SUPPORTED_ARTIFACT_VERSION = 2

MODEL_NAMES = ["symptom"]


class ModelNotAvailable(RuntimeError):
    """Raised when a route needs an artifact that is missing or unreadable."""


@dataclass
class LoadedModel:
    """A trained bundle plus the lookups the routes need."""

    name: str
    kind: str
    estimator: Any
    feature_names: list[str]
    classes: list[str]
    metrics: dict[str, Any]
    dataset: dict[str, Any]
    trained_at: str
    disease_symptoms: dict[str, list[str]]
    feature_importances: list[float] | None
    sklearn_version: str = "unknown"

    @cached_property
    def feature_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.feature_names)}

    def frame(self, values: list[float]) -> pd.DataFrame:
        """Build a one-row frame in the exact column order the model was fit on.

        Passing a DataFrame rather than a bare array keeps scikit-learn from
        warning about missing feature names, and makes a column-order bug fail
        loudly here instead of silently scoring the wrong feature.
        """
        return pd.DataFrame([values], columns=self.feature_names)

    def importance_ranking(self, top: int = 5) -> list[tuple[str, float]]:
        if not self.feature_importances:
            return []
        pairs = zip(self.feature_names, self.feature_importances)
        return sorted(pairs, key=lambda p: p[1], reverse=True)[:top]


def _to_loaded(name: str, bundle: dict[str, Any]) -> LoadedModel:
    version = bundle.get("artifact_version")
    if version != SUPPORTED_ARTIFACT_VERSION:
        raise ModelNotAvailable(
            f"{name}.joblib has artifact_version {version}, expected "
            f"{SUPPORTED_ARTIFACT_VERSION}. Retrain with: python -m ml.train"
        )
    return LoadedModel(
        name=bundle["name"],
        kind=bundle["kind"],
        estimator=bundle["model"],
        feature_names=list(bundle["feature_names"]),
        classes=[str(c) for c in bundle["classes"]],
        metrics=bundle.get("metrics", {}),
        dataset=bundle.get("dataset", {}),
        trained_at=bundle.get("trained_at", "unknown"),
        disease_symptoms=bundle.get("disease_symptoms", {}),
        feature_importances=bundle.get("feature_importances"),
        sklearn_version=str(bundle.get("sklearn_version", "unknown")),
    )


class Registry:
    """Lazy, cached access to the artifacts on disk."""

    def __init__(self, models_dir=MODELS_DIR) -> None:
        self.models_dir = models_dir
        self._cache: dict[str, LoadedModel] = {}
        self._errors: dict[str, str] = {}

    def path_for(self, name: str):
        return self.models_dir / f"{name}.joblib"

    def get(self, name: str) -> LoadedModel:
        if name in self._cache:
            return self._cache[name]

        path = self.path_for(name)
        if not path.exists():
            raise ModelNotAvailable(
                f"{path.name} not found in {self.models_dir}. Run: python -m ml.train"
            )
        try:
            loaded = _to_loaded(name, joblib.load(path))
        except ModelNotAvailable:
            raise
        except (ImportError, AttributeError, ModuleNotFoundError) as exc:
            # scikit-learn moves internal modules between minor releases, so a
            # pickle written by another version fails here with a confusing
            # "No module named '_loss'". Name the real cause instead.
            raise ModelNotAvailable(
                f"could not read {path.name}: {exc}. This usually means the artifact was "
                f"trained on a different scikit-learn than the {sklearn.__version__} running "
                f"here. Retrain with: python -m ml.train"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as a 503
            raise ModelNotAvailable(f"could not read {path.name}: {exc}") from exc

        if loaded.sklearn_version not in {sklearn.__version__, "unknown"}:
            log.warning(
                "%s was trained on scikit-learn %s but %s is installed; predictions load but "
                "may differ",
                name,
                loaded.sklearn_version,
                sklearn.__version__,
            )

        self._cache[name] = loaded
        return loaded

    def preload(self) -> None:
        """Warm every artifact, recording rather than raising on failure.

        A failure here should not stop the process from coming up, so it is
        recorded and reported through /health instead of raising.
        """
        for name in MODEL_NAMES:
            try:
                model = self.get(name)
                log.info("loaded %s (%d features, %d classes)", name, len(model.feature_names), len(model.classes))
            except ModelNotAvailable as exc:
                self._errors[name] = str(exc)
                log.warning("could not load %s: %s", name, exc)

    def status(self) -> dict[str, Any]:
        available = {}
        for name in MODEL_NAMES:
            if name in self._cache:
                model = self._cache[name]
                available[name] = {
                    "loaded": True,
                    "trained_at": model.trained_at,
                    "sklearn_version": model.sklearn_version,
                    "features": len(model.feature_names),
                    "classes": len(model.classes),
                }
            else:
                available[name] = {"loaded": False, "error": self._errors.get(name, "not loaded yet")}
        return available


registry = Registry()


def rank_probabilities(proba: np.ndarray, classes: list[str], top_k: int) -> list[tuple[str, float]]:
    """Return the top_k (label, probability) pairs, highest first."""
    order = np.argsort(proba)[::-1][:top_k]
    return [(classes[i], float(proba[i])) for i in order]
