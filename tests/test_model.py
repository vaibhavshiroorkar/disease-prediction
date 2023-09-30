"""Tests for the classifier definition and the recall-oriented search."""


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.model import base_model


def test_base_model_is_a_balanced_random_forest() -> None:
    model = base_model()
    assert isinstance(model, RandomForestClassifier)
    assert model.class_weight == "balanced"
    assert model.random_state == 42


def test_base_model_accepts_overrides() -> None:
    assert base_model(n_estimators=7).n_estimators == 7
