"""The study's maths, against answers known in advance."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logit

from study.shared.repeats import repair_size, score
from study.shared.updating import (
    apply_shift_slope,
    em_rate,
    fit_shift_slope,
    local_rate,
    reweight,
)
from study.transfer.audits import site_rate_scores, within_site_auc
from study.transfer.experiment import verdicts


@pytest.fixture()
def cohort():
    rng = np.random.default_rng(0)
    return expit(rng.normal(-0.5, 1.5, 40_000)), rng


def test_intercept_recovers_a_known_shift(cohort) -> None:
    p, rng = cohort
    y = rng.random(len(p)) < apply_shift_slope(p, 1.2, 1.0)
    shift, slope = fit_shift_slope(p, y, learn_slope=False)
    assert slope == 1.0
    assert shift == pytest.approx(1.2, abs=0.05)


def test_platt_recovers_a_known_shift_and_slope(cohort) -> None:
    p, rng = cohort
    y = rng.random(len(p)) < apply_shift_slope(p, -0.7, 0.5)
    shift, slope = fit_shift_slope(p, y, learn_slope=True)
    assert (shift, slope) == pytest.approx((-0.7, 0.5), abs=0.05)


@pytest.mark.parametrize("learn_slope", [False, True])
@pytest.mark.parametrize("label", [0, 1])
def test_one_class_samples_stay_finite(label, learn_slope) -> None:
    p = np.array([0.1, 0.4, 0.9])
    updated = apply_shift_slope(p, *fit_shift_slope(p, np.full(3, label), learn_slope))
    assert np.isfinite(updated).all()
    assert ((updated > 0) & (updated < 1)).all()


def test_intercept_never_changes_the_ranking(cohort) -> None:
    p = cohort[0][:500]
    assert (np.argsort(apply_shift_slope(p, 2.0, 1.0)) == np.argsort(p)).all()


def test_local_rate_is_smoothed() -> None:
    assert local_rate(np.zeros(9)) == pytest.approx(0.05)


def test_em_finds_the_rate_when_only_the_rate_changed() -> None:
    rng = np.random.default_rng(1)
    y = rng.random(40_000) < 0.15
    x = rng.normal(np.where(y, 1.0, -1.0), 1.0)
    p_source = expit(2 * x + logit(0.6))  # the exact posterior at a 60% rate
    assert em_rate(p_source, 0.6) == pytest.approx(0.15, abs=0.01)


def test_reweight_to_the_same_rate_changes_nothing() -> None:
    p = np.array([0.2, 0.7])
    assert reweight(p, 0.3, 0.3) == pytest.approx(p)


def test_score_reports_the_gap_between_predicted_and_real() -> None:
    result = score(np.array([0, 0, 1, 1]), np.array([0.2, 0.2, 0.8, 0.4]), threshold=0.5)
    assert result["auc"] == 1.0
    assert result["gap"] == pytest.approx(-0.1)
    assert result["sensitivity"] == 0.5
    assert result["specificity"] == 1.0


def test_repair_size_needs_every_larger_size_to_work_too() -> None:
    assert repair_size({10: 0.95, 25: 0.80, 50: 0.95}) == 50
    assert repair_size({10: 0.95, 25: 0.95, 50: 0.95}) == 10
    assert repair_size({10: 0.95, 25: 0.95, 50: 0.50}) is None


def _transfer(name, family, sizes, losses):
    return {
        "name": name,
        "family": family,
        "sizes": sizes,
        "auc_drop_ci": [0.01, 0.1],
        "oe_ci": [1.1, 1.3],
        "curves": {m: {str(n): {"log_loss": {"mean": losses[m]}} for n in sizes} for m in losses},
    }


def test_verdicts_can_pass_and_fail() -> None:
    good = {"none": 0.6, "em": 0.7, "intercept": 0.5, "platt": 0.55, "refit": 0.58, "rate": 0.6}
    bad = {"none": 0.5, "em": 0.4, "intercept": 0.6, "platt": 0.55, "refit": 0.4, "rate": 0.6}
    names = [("heart", [10, 25, 50])] * 4 + [("dengue", [100, 250, 500, 1000, 2500])] * 6
    ok = verdicts([_transfer(str(i), f, s, good) for i, (f, s) in enumerate(names)])
    no = verdicts([_transfer(str(i), f, s, bad) for i, (f, s) in enumerate(names)])
    for h in ("H2", "H3", "H4", "H5"):
        assert ok[h]["supported"] is True
        assert no[h]["supported"] is False
    # Equal counts do not support H1.
    assert ok["H1"]["supported"] is False


def test_site_rate_shortcut_and_within_site_auc() -> None:
    sites = np.array(["a"] * 4 + ["b"] * 4)
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    folds = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    rates = site_rate_scores(y, sites, folds)
    # Scored from the other fold, so a row's own label cannot leak in.
    assert rates[0] == pytest.approx(0.5)  # site a, fold 1: labels 0 and 1
    assert rates[1] == 0.0  # site a, fold 0: labels 0 and 0
    p = np.array([0.1, 0.1, 0.1, 0.2, 0.8, 0.9, 0.9, 0.9])
    assert within_site_auc(y, p, sites)["mean"] == pytest.approx(1.0)


@pytest.mark.parametrize("spread", [0.5, 1.0, 2.0])
def test_em_matches_a_direct_likelihood_search(spread) -> None:
    """Including where EM's assumption is broken and its answer is far from the truth."""
    p = expit(np.random.default_rng(3).normal(0.8, spread, 2_000))
    source = 0.5
    grid = np.linspace(0.0005, 0.9995, 2_000)
    likelihood = [np.log(g * p / source + (1 - g) * (1 - p) / (1 - source)).sum() for g in grid]
    assert em_rate(p, source) == pytest.approx(grid[int(np.argmax(likelihood))], abs=0.002)
