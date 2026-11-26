"""The served symptom estimator matches the stronger noisy-data baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_symptom_estimator_learns_probabilities_without_forest_attributes():
    from ml.model import symptom_model

    X = pd.DataFrame({"itching": [1, 1, 0, 0], "cough": [0, 0, 1, 1]})
    y = pd.Series(["Skin condition", "Skin condition", "Respiratory", "Respiratory"])
    model = symptom_model().fit(X, y)
    assert model.predict(pd.DataFrame({"itching": [1], "cough": [0]}))[0] == "Skin condition"
    assert np.isclose(model.predict_proba(X)[0].sum(), 1)
    assert hasattr(model, "coef_")
    assert not hasattr(model, "feature_importances_")


def test_committed_artifact_serves_logistic_probabilities():
    from app.registry import Registry

    model = Registry().get("symptom")
    assert model.metrics["chosen_family"] == "logistic"
    assert hasattr(model.estimator, "coef_")
    proba = model.estimator.predict_proba(model.frame([0.0] * len(model.feature_names)))[0]
    assert np.isfinite(proba).all()
    assert np.isclose(proba.sum(), 1)


def test_training_writes_a_logistic_bundle_with_usable_global_coefficients(monkeypatch):
    from itertools import product

    from ml import train as training
    from ml.datasets import Dataset

    X = pd.DataFrame(list(product((0, 1), repeat=5)), columns=list("abcde")).astype("int8")
    y = pd.Series(["Skin" if bit else "Respiratory" for bit in X["a"]])
    dataset = Dataset(X=X, y=y, name="fixture", notes=[])
    monkeypatch.setattr(training, "load_symptoms", lambda split: dataset)
    bundle = training.train(search=False)
    assert bundle["metrics"]["chosen_family"] == "logistic"
    assert bundle["model"].predict(X).tolist() == y.tolist()
    assert len(bundle["feature_importances"]) == 5
    assert np.isclose(sum(bundle["feature_importances"]), 1)
    assert {row["model"] for row in bundle["metrics"]["comparison"]} == {
        "forest", "logistic", "bernoulli_nb",
    }
