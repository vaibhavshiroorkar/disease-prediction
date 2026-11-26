"""The lab page's numbers must match the committed models and study results."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml.config import MODELS_DIR, ROOT
from study.lab_data import TARGET, histogram, threshold_counts, threshold_grid


def test_counts_include_scores_equal_to_the_cutoff() -> None:
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.5, 0.5, 0.1])
    counts = threshold_counts(y, p, np.array([0.0, 0.5, 0.6, 1.0]))
    assert counts["tp"] == [2, 2, 1, 0]
    assert counts["fp"] == [2, 1, 0, 0]


def test_grid_holds_the_served_cutoff_exactly() -> None:
    cutoff = 0.123456789
    assert cutoff in threshold_grid(np.linspace(0, 1, 500), cutoff).tolist()


def test_histogram_loses_no_row() -> None:
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 1000), rng.random(1000)
    h = histogram(y, p)
    assert sum(h["positive"]) + sum(h["negative"]) == 1000


@pytest.fixture(scope="module")
def lab():
    return json.loads(TARGET.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["heart", "diabetes", "liver", "dengue", "malaria"])
def test_page_matches_the_committed_summary(lab, name) -> None:
    panel = lab["models"][name]
    test = json.loads((MODELS_DIR / f"{name}_summary.json").read_text(encoding="utf-8"))["metrics"]["test"]
    index = panel["grid"].index(panel["threshold"])
    assert panel["auc"] == pytest.approx(test["roc_auc"], abs=1e-4)
    assert panel["tp"][index] == test["confusion"]["tp"]
    assert panel["fp"][index] == test["confusion"]["fp"]


def test_page_matches_the_study_report(lab) -> None:
    report = json.loads((ROOT / "results" / "transfer" / "report.json").read_text(encoding="utf-8"))
    assert lab["study_commit"] == report["provenance"]["commit"]
    assert lab["verdicts"] == report["verdicts"]
    assert [t["name"] for t in lab["transfers"]] == [t["name"] for t in report["transfers"]]


def test_page_matches_the_state_study(lab) -> None:
    states = ROOT / "results" / "states"
    report = json.loads((states / "report.json").read_text(encoding="utf-8"))
    panel = lab["states"]
    assert panel["commit"] == report["provenance"]["commit"]
    assert [t["name"] for t in panel["targets"]] == [t["name"] for t in report["targets"]]
    for test, verdict in report["verdicts"].items():
        assert panel["verdicts"][test]["supported"] == verdict["supported"]
    for check in ("true_rate", "bbse"):
        summary = json.loads((states / f"{check}.json").read_text(encoding="utf-8"))["summary"]
        assert panel["checks"][check] == summary
    em_hurt = sum(t["change"]["em"] > 0 for t in panel["targets"])
    assert em_hurt == report["verdicts"]["R2"]["count"]
