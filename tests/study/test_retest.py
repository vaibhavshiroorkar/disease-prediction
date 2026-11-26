"""The 2026 retest rules against answers known in advance. No data files needed."""

from __future__ import annotations

from study.states.retest import retest_verdicts

GOOD_CELLS = {"cells": 30, "share_negative": 0.7}
USABLE = "bbse is a usable label-free repair"
FAILS = "bbse fails here too"


def _target(i: int, em: float, true_prior: float, oe: float, repaired: bool, em_ratio: float,
            bbse: float = 0.01) -> dict:
    return {
        "name": f"t{i}", "state": str(i), "oe": oe, "repaired": repaired,
        "change": {"em": em, "true_prior": true_prior},
        "em_rate_over_true": em_ratio,
        "bbse": {"change_from_none": {"em": em, "bbse": bbse, "bbse_soft": bbse},
                 "rate_over_true": {"em": em_ratio, "bbse": 1.0, "bbse_soft": 1.0}},
    }


def _like_before(count: int = 12) -> list[dict]:
    """Targets that look like 2023 to 2025: EM hurts in proportion to its rate error."""
    targets = []
    for i in range(count):
        inside = i < 4
        targets.append(_target(i, em=0.01 * (i + 1), true_prior=-0.002,
                               oe=1.0 if inside else 1.5, repaired=not inside, em_ratio=1.2 + 0.3 * i,
                               bbse=0.02 * (i + 1)))
    return targets


def test_findings_that_hold_again_pass() -> None:
    v = retest_verdicts(_like_before(), GOOD_CELLS, FAILS)
    for test in ("T1", "T2", "T3", "T4", "T5", "T6", "T7"):
        assert v[test]["testable"], test
        assert v[test]["supported"], test


def test_findings_that_do_not_hold_fail() -> None:
    targets = [_target(i, em=-0.01, true_prior=0.01, oe=1.5, repaired=i % 2 == 0, em_ratio=1.0 + i,
                       bbse=-0.01) for i in range(10)]
    targets += [_target(10 + i, em=-0.01, true_prior=0.01, oe=1.0, repaired=True, em_ratio=2.0)
                for i in range(3)]
    v = retest_verdicts(targets, {"cells": 30, "share_negative": 0.4}, FAILS)
    assert not v["T1"]["supported"]
    assert not v["T2"]["testable"]  # EM hurt nowhere, so T2 has nothing to judge
    assert not v["T4"]["supported"]
    assert not v["T5"]["supported"]
    assert not v["T6"]["supported"]
    assert not v["T7"]["supported"] and v["T7"]["after"] == USABLE


def test_too_few_targets_makes_nothing_testable() -> None:
    v = retest_verdicts(_like_before(5), GOOD_CELLS, FAILS)
    assert not any(x["testable"] or x["supported"] for x in v.values())


def test_band_needs_three_targets_on_each_side() -> None:
    targets = _like_before()
    for t in targets:
        t["oe"] = 1.5
    assert not retest_verdicts(targets, GOOD_CELLS, FAILS)["T4"]["testable"]
