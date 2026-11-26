"""The BBSE check against answers known in advance. No data files needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from study.states.bbse import bbse_rates, one_draw, solve_rate, summarize
from study.states.experiment import Source


def _source(rate: float = 0.1, rows: int = 20_000, seed: int = 0) -> Source:
    """A model whose scores depend only on the label, so only the rate can shift."""
    rng = np.random.default_rng(seed)
    y = (rng.random(rows) < rate).astype(int)
    oof = np.clip(0.05 + 0.5 * y + rng.normal(0, 0.1, rows), 0.001, 0.999)
    return Source(X=pd.DataFrame(index=range(rows)), y=pd.Series(y), model=None, oof=oof, threshold=0.3)


def _target(rate: float, rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(rows) < rate).astype(int)
    return y, np.clip(0.05 + 0.5 * y + rng.normal(0, 0.1, rows), 0.001, 0.999)


def test_solve_rate_inverts_the_mixture() -> None:
    assert solve_rate(0.3 * 0.9 + 0.7 * 0.1, 0.9, 0.1) == pytest.approx(0.3)
    assert solve_rate(0.0, 0.9, 0.1) == pytest.approx(1e-6)
    with pytest.raises(ValueError):
        solve_rate(0.5, 0.2, 0.4)


def test_bbse_recovers_a_pure_rate_shift() -> None:
    """Under label shift alone, both versions find the new rate. EM should too."""
    y, p = _target(0.3, 20_000, 1)
    rates = bbse_rates(_source(), p)
    assert rates["bbse"] == pytest.approx(y.mean(), abs=0.01)
    assert rates["bbse_soft"] == pytest.approx(y.mean(), abs=0.01)


def test_draw_scores_every_estimate_on_the_same_test_half() -> None:
    y, p = _target(0.3, 6_000, 2)
    row = one_draw(p, y, _source(), 2500, 0)
    assert set(row) >= {"none", "em", "bbse", "bbse_soft", "em_rate", "bbse_rate", "bbse_soft_rate"}
    assert row["bbse"] < row["none"]  # the rate rose from 0.1 to 0.3, and BBSE saw it


def _result(em: float, bbse: float, soft: float = 0.0) -> dict:
    return {"change_from_none": {"em": em, "bbse": bbse, "bbse_soft": soft},
            "rate_over_true": {"em": 2.0, "bbse": 1.0, "bbse_soft": 1.1}}


def test_summary_applies_the_decision_rule() -> None:
    assert summarize([_result(0.1, -0.01)] * 3 + [_result(0.1, 0.01)])["answer"] == (
        "bbse is a usable label-free repair")
    assert summarize([_result(0.1, 0.01)] * 4)["answer"] == (
        "bbse is a better estimate than EM, but not a usable repair")
    assert summarize([_result(-0.1, 0.01)] * 4)["answer"] == "bbse fails here too"
    assert summarize([_result(0.1, 0.01)])["rate_within_0.8_to_1.25"] == {"em": 0, "bbse": 1, "bbse_soft": 1}
