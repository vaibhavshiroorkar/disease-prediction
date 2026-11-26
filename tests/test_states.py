"""The state study's rules, against answers known in advance. No data files needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.clinical_data import DENGUE_FEATURES
from study import states
from study.states import (
    Year,
    cases_split,
    choose_targets,
    guard_threshold,
    shift_score,
    sizes_for,
    source_rows,
    verdicts,
)
from study.transfer import one_repeat


def test_inclusion_rule_needs_both_counts() -> None:
    counts = pd.DataFrame({
        "year": [2023, 2023, 2023, 2025],
        "state": ["35", "14", "29", "29"],
        "cases": [1000, 5000, 999, 2000],
        "severe": [100, 99, 500, 150],
    })
    chosen, excluded = choose_targets(counts)
    assert chosen == [(2023, "35"), (2025, "29")]
    assert {(e["state"], e["year"]) for e in excluded} == {("RR", 2023), ("BA", 2023)}


def test_source_rows_never_hold_the_target_state(monkeypatch) -> None:
    monkeypatch.setattr(states, "SOURCE_SAMPLE", 50)
    rng = np.random.default_rng(0)
    state = np.array(["35"] * 100 + ["29"] * 100)
    year = Year(pd.DataFrame({"age": np.arange(200.0)}), (rng.random(200) < 0.3).astype(int), state)
    X, y = source_rows(year, "35")
    assert len(X) == 50
    assert (X["age"] >= 100).all()


def test_shift_score_sees_a_shift_and_no_shift() -> None:
    rng = np.random.default_rng(1)
    a = pd.DataFrame({"x": rng.normal(0, 1, 2000), "z": rng.normal(0, 1, 2000)})
    same = pd.DataFrame({"x": rng.normal(0, 1, 2000), "z": rng.normal(0, 1, 2000)})
    moved = pd.DataFrame({"x": rng.normal(1.5, 1, 2000), "z": rng.normal(0, 1, 2000)})
    assert shift_score(a, same) == pytest.approx(0.5, abs=0.05)
    assert shift_score(a, moved) > 0.8


def test_sizes_fit_in_the_pool() -> None:
    assert sizes_for(1000) == [100, 250, 500]
    assert sizes_for(5000) == [100, 250, 500, 1000, 2500]


def test_each_draw_records_its_local_cases() -> None:
    rng = np.random.default_rng(2)
    y = (rng.random(2000) < 0.1).astype(int)
    p = np.clip(0.1 + 0.3 * y + rng.normal(0, 0.05, 2000), 0.01, 0.99)
    holder = type("H", (), {"features": DENGUE_FEATURES[:1]})()
    X = pd.DataFrame({"age": rng.normal(40, 10, 2000)})
    rows = one_repeat(holder, X, y, p, 0.1, 0.2, [100, 250], 0)
    by_n = {r["n"]: r["local_cases"] for r in rows}
    assert 0 < by_n[100] <= by_n[250] <= 250
    assert len({(r["n"], r["local_cases"]) for r in rows}) == 2


def test_cases_split_counts_hurt_draws_by_local_cases() -> None:
    rows = []
    for repeat, (cases, hurt) in enumerate([(5, True), (5, True), (5, False), (25, False), (25, True)]):
        rows.append({"repeat": repeat, "n": 100, "method": "none", "log_loss": 0.5, "local_cases": cases})
        rows.append({"repeat": repeat, "n": 100, "method": "intercept",
                     "log_loss": 0.6 if hurt else 0.4, "local_cases": cases})
    split = cases_split(pd.DataFrame(rows))
    assert split["low_draws"] == 3 and split["high_draws"] == 2
    assert split["low_hurt_share"] == pytest.approx(2 / 3, abs=1e-4)
    assert split["high_hurt_share"] == 0.5


SPLIT_KEYS = ("low_draws", "high_draws", "low_hurt_share", "high_hurt_share")


def _report(name, state, shift, em, intercept=-0.01, repair=500, cases=None, split=(30, 30, 0.5, 0.2)):
    sizes = [100, 250, 500, 1000]
    curves = {m: {str(n): {"diff_from_none": 0.0} for n in sizes} for m in ("em", "intercept")}
    curves["em"]["1000"]["diff_from_none"] = em
    curves["intercept"]["500"]["diff_from_none"] = intercept
    return {
        "name": name, "state": state, "sizes": sizes, "shift_score": shift, "curves": curves,
        "repair_size": {"intercept": repair},
        "mean_local_cases": {str(n): (cases if cases is not None else n / 40) for n in sizes},
        "cases_split": dict(zip(SPLIT_KEYS, split, strict=True)),
    }


def test_guard_takes_the_cut_with_least_loss_and_ties_go_low() -> None:
    a = [_report("a", "1", 0.55, -0.02), _report("b", "2", 0.60, -0.01), _report("c", "3", 0.80, 0.05)]
    assert guard_threshold(a) == pytest.approx(0.60, abs=1e-6)
    assert guard_threshold([_report("d", "4", 0.7, 0.0)]) == 0.0
    assert guard_threshold([_report("e", "5", 0.7, 0.03)]) == 0.0


def test_verdicts_can_pass() -> None:
    stage_a = [_report(f"a{i}", str(i), 0.5 + i / 100, -0.01 + i / 1000) for i in range(6)]
    stage_b = [_report(f"b{i}", str(i), 0.5 + i / 50, -0.02 + i / 100,
                       repair=[250, 500, 1000][i % 3], cases=12.0 + (i % 2)) for i in range(6)]
    v = verdicts(stage_a, stage_b, guard_threshold(stage_a))
    assert v["S1"]["supported"] and v["S1"]["of"] == 12
    assert v["S3"]["spearman"] > 0
    assert v["S4"]["em_worse"] and v["S4"]["guard_helps"]
    assert v["S4"]["supported"]
    assert v["R1"]["supported"]


def test_verdicts_can_fail() -> None:
    flat = [_report(f"x{i}", str(i), 0.5 + i / 100, 0.05 - i / 100, intercept=0.01,
                    split=(30, 30, 0.1, 0.4)) for i in range(6)]
    v = verdicts(flat[:3], flat[3:], 0.0)
    assert not v["S1"]["supported"]
    assert v["S3"]["spearman"] < 0 and not v["S3"]["supported"]
    assert not v["S4"]["supported"]  # a guard that never uses EM never helps
    assert not v["R1"]["supported"]


def test_s1_needs_enough_draws_in_both_groups() -> None:
    few = [_report("a", "1", 0.5, 0.0, split=(19, 30, 0.9, 0.1))]
    assert verdicts(few, [], 0.0)["S1"]["of"] == 0
    assert not verdicts(few, [], 0.0)["S1"]["supported"]


def test_state_bootstrap_keeps_a_state_together() -> None:
    reports = [_report("a23", "35", 0.5, 0.0), _report("a25", "35", 0.5, 0.0), _report("b", "29", 0.5, 0.0)]
    seen = []

    def record(drawn):
        names = [r["name"] for r in drawn]
        seen.append(names)
        return 0.0

    states._bootstrap_by_state(reports, record)
    assert all(names.count("a23") == names.count("a25") for names in seen)
