"""Tests for the candidate pipelines and the selection rule."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.clinical_data import FeatureSpec
from ml.clinical_model import candidates, pick, preprocessor

FEATURES = [
    FeatureSpec("age", "Age", "numeric", "", "years", 0, 100),
    FeatureSpec("alt", "ALT", "numeric", "", "IU/L", 1, 3000, skewed=True),
    FeatureSpec("sex", "Sex", "binary", "", choices={0: "f", 1: "m"}),
    FeatureSpec("cp", "Pain", "category", "", choices={1: "a", 2: "b", 3: "c"}),
]

X = pd.DataFrame(
    {
        "age": [40.0, np.nan, 60.0, 50.0],
        "alt": [20.0, 2000.0, np.nan, 35.0],
        "sex": [1.0, 0.0, np.nan, 1.0],
        "cp": [1.0, 3.0, 2.0, np.nan],
    }
)


def test_missing_indicators_can_be_switched_off() -> None:
    with_flags = preprocessor(FEATURES, scale=False, missing_indicators=True).fit(X)
    without = preprocessor(FEATURES, scale=False, missing_indicators=False).fit(X)
    assert any("missingindicator" in name for name in with_flags.get_feature_names_out())
    assert not any("missingindicator" in name for name in without.get_feature_names_out())


def test_categories_are_one_hot_encoded() -> None:
    names = preprocessor(FEATURES, scale=False, missing_indicators=False).fit(X).get_feature_names_out()
    assert {"cp_1.0", "cp_2.0", "cp_3.0"} <= set(names)


def test_skewed_values_are_logged_before_imputing() -> None:
    out = preprocessor(FEATURES, scale=False, missing_indicators=False).fit_transform(X)
    names = list(preprocessor(FEATURES, scale=False, missing_indicators=False).fit(X).get_feature_names_out())
    alt = out[:, names.index("alt")]
    assert alt[1] == np.log1p(2000.0)
    # The gap is filled with the median of the logged values, not the log of the median.
    assert alt[2] == np.median(np.log1p([20.0, 2000.0, 35.0]))


def test_every_candidate_fits_and_predicts() -> None:
    y = pd.Series([0, 1, 0, 1])
    for family, (pipeline, _) in candidates(FEATURES, missing_indicators=True).items():
        proba = pipeline.fit(X, y).predict_proba(X)[:, 1]
        assert ((proba >= 0) & (proba <= 1)).all(), family


def _row(family: str, auc: float, std: float = 0.03, ablation: bool = False) -> dict:
    return {"family": family, "roc_auc": auc, "roc_auc_std": std, "ablation": ablation}


def test_pick_prefers_logistic_on_a_near_tie() -> None:
    table = [_row("forest", 0.880), _row("logistic", 0.875), _row("dummy", 0.5, 0.0)]
    assert pick(table) == "logistic"


def test_pick_takes_the_clear_winner() -> None:
    table = [_row("boosting", 0.90, 0.02), _row("logistic", 0.85), _row("dummy", 0.5, 0.0)]
    assert pick(table) == "boosting"


def test_pick_never_chooses_an_ablation_row() -> None:
    table = [_row("logistic_no_log", 0.95, ablation=True), _row("forest", 0.90, 0.01), _row("logistic", 0.80)]
    assert pick(table) == "forest"
