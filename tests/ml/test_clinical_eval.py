"""Tests for the binary scoring helpers, against hand-computed answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.clinical_eval import (
    binary_scores,
    bootstrap_ci,
    calibration_table,
    choose_threshold,
    external_scores,
    leave_one_site_out,
)


def test_binary_scores_match_a_hand_built_confusion_matrix() -> None:
    #           tp   tp   fn   fp   tn   tn
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1])
    scores = binary_scores(y, p, threshold=0.5)

    assert scores["confusion"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert scores["sensitivity"] == pytest.approx(2 / 3)
    assert scores["specificity"] == pytest.approx(2 / 3)
    assert scores["ppv"] == pytest.approx(2 / 3)
    assert scores["npv"] == pytest.approx(2 / 3)
    # 8 of the 9 positive-negative pairs are ordered correctly.
    assert scores["roc_auc"] == pytest.approx(8 / 9)
    assert scores["brier"] == pytest.approx(np.mean((p - y) ** 2))


def test_threshold_is_inclusive() -> None:
    scores = binary_scores(np.array([1, 0]), np.array([0.5, 0.2]), threshold=0.5)
    assert scores["sensitivity"] == 1.0


def test_choose_threshold_hits_the_target_sensitivity() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 * y + rng.normal(0.4, 0.2, 500), 0, 1)

    threshold = choose_threshold(y, p, target_sensitivity=0.9)
    assert binary_scores(y, p, threshold)["sensitivity"] >= 0.9
    # It is the highest such threshold: nudging it up loses a true case.
    assert binary_scores(y, p, threshold + 1e-9)["sensitivity"] < 0.9


def test_choose_threshold_on_a_tiny_case() -> None:
    y = np.array([1, 1, 1, 1, 0])
    p = np.array([0.9, 0.7, 0.4, 0.2, 0.5])
    # 90% of 4 positives rounds up to all 4, so the cut must sit at the lowest one.
    assert choose_threshold(y, p, 0.9) == 0.2
    assert choose_threshold(y, p, 0.5) == 0.7


def test_choose_threshold_needs_positives() -> None:
    with pytest.raises(ValueError):
        choose_threshold(np.array([0, 0]), np.array([0.1, 0.2]))


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = np.clip(0.4 * y + rng.normal(0.3, 0.2, 200), 0, 1)
    point = binary_scores(y, p, 0.5)
    ci = bootstrap_ci(y, p, 0.5, resamples=300)
    for key in ["roc_auc", "sensitivity", "specificity", "brier"]:
        low, high = ci[key]
        assert low <= point[key] <= high, key


def test_calibration_error_is_zero_when_predictions_come_true_at_their_rate() -> None:
    # 10 rows predicted 0.3, 3 of them positive; 10 predicted 0.8, 8 positive.
    p = np.array([0.3] * 10 + [0.8] * 10)
    y = np.array([1] * 3 + [0] * 7 + [1] * 8 + [0] * 2)
    table = calibration_table(y, p)
    assert table["ece"] == pytest.approx(0.0)
    assert [row["rows"] for row in table["bins"]] == [10, 10]


def test_calibration_error_catches_overconfidence() -> None:
    p = np.array([0.9] * 10)
    y = np.array([1] * 5 + [0] * 5)
    assert calibration_table(y, p)["ece"] == pytest.approx(0.4)


def test_leave_one_site_out_scores_every_site() -> None:
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"a": rng.normal(size=240)})
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=240) > 0).astype(int))
    groups = pd.Series(["x", "y", "z"] * 80)

    results = leave_one_site_out(lambda: LogisticRegression(), X, y, groups)
    assert [r["site"] for r in results] == ["x", "y", "z"]
    assert all(r["rows"] == 80 for r in results)
    assert all(r["roc_auc"] > 0.7 for r in results)


def test_grouped_bootstrap_is_wider_when_rows_cluster() -> None:
    """Ten copies of each of 20 patients hold the information of 20 rows, not 200."""
    rng = np.random.default_rng(3)
    y = np.repeat(rng.integers(0, 2, 20), 10)
    p = np.clip(0.3 * y + rng.normal(0.35, 0.2, 200), 0, 1)
    groups = np.repeat(np.arange(20), 10)

    def width(ci):
        return ci["roc_auc"][1] - ci["roc_auc"][0]

    by_row = bootstrap_ci(y, p, 0.5, resamples=300)
    by_group = bootstrap_ci(y, p, 0.5, resamples=300, groups=groups)
    assert width(by_group) > width(by_row)


def test_external_scores_report_calibration_drift() -> None:
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    p = np.full(10, 0.1)
    scores = external_scores(y, p, threshold=0.5)
    assert scores["prevalence"] == 0.3
    assert scores["mean_predicted"] == 0.1
    assert scores["rows"] == 10
    assert scores["sensitivity"] == 0.0
