"""The true-rate check against answers known in advance. No data files needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.clinical_data import DENGUE_FEATURES
from study.shared.repeats import one_repeat
from study.shared.updating import to_logit
from study.states.true_rate import mean_matching_shift, one_draw, summarize


def _data(seed: int = 3, rows: int = 4000):
    rng = np.random.default_rng(seed)
    y = (rng.random(rows) < 0.08).astype(int)
    p = np.clip(0.05 + 0.3 * y + rng.normal(0, 0.04, rows), 0.01, 0.99)
    return y, p


def test_mean_matching_shift_hits_the_rate() -> None:
    _, p = _data()
    shift = mean_matching_shift(p, 0.2)
    assert (1 / (1 + np.exp(-(to_logit(p) + shift)))).mean() == pytest.approx(0.2, abs=1e-8)


def test_draw_repeats_the_study_draw() -> None:
    """none and em must be the very numbers one_repeat gave, or the comparison is off."""
    y, p = _data()
    X = pd.DataFrame({"age": np.zeros(len(y))})
    rows = one_repeat(DENGUE_FEATURES[:1], X, y, p, 0.05, 0.2, [500], 4)
    study = {r["method"]: r["log_loss"] for r in rows}
    mine = one_draw(p, y, 0.05, 500, 4)
    assert mine["none"] == pytest.approx(study["none"], abs=1e-12)
    assert mine["em"] == pytest.approx(study["em"], abs=1e-12)


def test_true_prior_is_no_change_when_rates_agree() -> None:
    y, p = _data()
    row = one_draw(p, y, float(y.mean()), 500, 0)
    assert row["true_prior"] == pytest.approx(row["none"], abs=1e-12)


def _target(em: float, prior: float, mean: float = 0.0) -> dict:
    return {"change_from_none": {"em": em, "true_prior": prior, "true_mean": mean}}


def test_summary_applies_the_decision_rule() -> None:
    estimate = [_target(0.01, -0.01)] * 3 + [_target(0.01, 0.01)] + [_target(-0.01, 0.01)]
    assert summarize(estimate)["answer"] == "the estimate failed"
    correction = [_target(0.01, 0.01)] * 4 + [_target(0.01, -0.01)]
    assert summarize(correction)["answer"] == "the correction failed"
    mixed = [_target(0.01, 0.01), _target(0.01, -0.01)]
    assert summarize(mixed)["answer"] == "mixed"
    assert summarize(estimate)["helped_of_all"]["true_prior"] == 3
