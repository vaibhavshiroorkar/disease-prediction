"""Prediction logging against a real Postgres.

These run only when TEST_DATABASE_URL is set. They use a real database rather
than a mock because the parts worth testing are the parts a mock would fake: the
text[] round trip, the jsonb column, and the partial indexes behind the review
queue.
"""

from __future__ import annotations

import pytest

from app import db
from tests.conftest import requires_db

pytestmark = requires_db

RANKED = [
    {"disease": "Fungal infection", "probability": 0.91, "confidence": "high"},
    {"disease": "Acne", "probability": 0.04, "confidence": "low"},
]


def test_schema_is_idempotent(database) -> None:
    """Startup applies the schema every boot, so it must survive re-running."""
    database.apply_schema()
    database.apply_schema()
    with database.cursor() as cur:
        cur.execute("select to_regclass('public.predictions')")
        assert cur.fetchone()[0] == "predictions"


def test_log_prediction_returns_an_id_and_round_trips(clean_table) -> None:
    prediction_id = clean_table.log_prediction(
        symptoms=["itching", "skin_rash"],
        unknown=["hedache"],
        top_k=5,
        ranked=RANKED,
        model_trained_at="2024-02-01T00:00:00+00:00",
        response_ms=12.5,
    )
    assert isinstance(prediction_id, int)

    rows = clean_table.recent_predictions(10)
    assert len(rows) == 1
    row = rows[0]
    assert row["symptoms"] == ["itching", "skin_rash"]
    assert row["predicted_disease"] == "Fungal infection"
    assert row["confidence"] == pytest.approx(0.91, abs=1e-4)
    assert row["was_correct"] is None


def test_empty_ranking_is_not_logged(clean_table) -> None:
    assert clean_table.log_prediction([], [], 5, []) is None
    assert clean_table.recent_predictions(10) == []


def test_recent_predictions_are_newest_first(clean_table) -> None:
    for disease in ["A", "B", "C"]:
        clean_table.log_prediction(["itching"], [], 5, [{"disease": disease, "probability": 0.5}])
    assert [r["predicted_disease"] for r in clean_table.recent_predictions(10)] == ["C", "B", "A"]


def test_review_marks_a_prediction_wrong_with_the_correct_answer(clean_table) -> None:
    prediction_id = clean_table.log_prediction(["itching"], [], 5, RANKED)

    assert clean_table.record_review(prediction_id, "Acne", was_correct=False) is True

    wrong = clean_table.wrong_predictions()
    assert len(wrong) == 1
    assert wrong[0]["actual_disease"] == "Acne"
    assert wrong[0]["was_correct"] is False


def test_review_of_a_missing_row_reports_failure(clean_table) -> None:
    assert clean_table.record_review(999_999, "Acne", was_correct=False) is False


def test_correct_predictions_stay_out_of_the_wrong_queue(clean_table) -> None:
    prediction_id = clean_table.log_prediction(["itching"], [], 5, RANKED)
    clean_table.record_review(prediction_id, "Fungal infection", was_correct=True)
    assert clean_table.wrong_predictions() == []


def test_stats_counts_reviews_and_observed_accuracy(clean_table) -> None:
    first = clean_table.log_prediction(["itching"], [], 5, RANKED)
    second = clean_table.log_prediction(["skin_rash"], [], 5, RANKED)
    clean_table.log_prediction(["chills"], [], 5, RANKED)  # left unreviewed

    clean_table.record_review(first, "Fungal infection", was_correct=True)
    clean_table.record_review(second, "Acne", was_correct=False)

    stats = clean_table.stats()
    assert stats["logging_enabled"] is True
    assert stats["total"] == 3
    assert stats["reviewed"] == 2
    assert stats["correct"] == 1
    assert stats["wrong"] == 1
    assert stats["observed_accuracy"] == pytest.approx(0.5)


def test_observed_accuracy_is_null_before_any_review(clean_table) -> None:
    clean_table.log_prediction(["itching"], [], 5, RANKED)
    assert clean_table.stats()["observed_accuracy"] is None


def test_feedback_only_returns_rows_with_a_known_answer(clean_table) -> None:
    """A prediction marked wrong without a correct label cannot be trained on."""
    usable = clean_table.log_prediction(["itching", "skin_rash"], [], 5, RANKED)
    unusable = clean_table.log_prediction(["chills"], [], 5, RANKED)

    clean_table.record_review(usable, "Acne", was_correct=False)
    clean_table.record_review(unusable, None, was_correct=False)

    feedback = clean_table.fetch_training_feedback()
    assert len(feedback) == 1
    symptoms, actual = feedback[0]
    assert sorted(symptoms) == ["itching", "skin_rash"]
    assert actual == "Acne"


def test_logging_survives_a_dead_pool() -> None:
    """A database that disappears must not raise into the request path."""
    db.close_pool()
    try:
        assert db.log_prediction(["itching"], [], 5, RANKED) is None
        assert db.recent_predictions() == []
        assert db.wrong_predictions() == []
        assert db.fetch_training_feedback() == []
        assert db.stats()["logging_enabled"] is False
        assert db.record_review(1, "Acne", was_correct=False) is False
    finally:
        from tests.conftest import TEST_DATABASE_URL

        db.init_pool(TEST_DATABASE_URL)


def test_risk_prediction_round_trips_its_inputs(clean_table) -> None:
    prediction_id = clean_table.log_risk_prediction(
        model="heart",
        inputs={"age": 54.0, "sex": 1, "cp": 4},
        imputed=["chol", "trestbps"],
        probability=0.71,
        threshold=0.4,
        flagged=True,
    )
    rows = clean_table.recent_risk_predictions(10)
    assert rows[0]["id"] == prediction_id
    assert rows[0]["inputs"] == {"age": 54.0, "sex": 1, "cp": 4}
    assert rows[0]["flagged"] is True
    assert rows[0]["had_condition"] is None


def test_recent_risk_predictions_filter_by_model(clean_table) -> None:
    for model in ["heart", "liver", "heart"]:
        clean_table.log_risk_prediction(model, {}, [], 0.5, 0.4, True)
    assert [r["model"] for r in clean_table.recent_risk_predictions(10, "heart")] == ["heart", "heart"]
    assert len(clean_table.recent_risk_predictions(10)) == 3


def test_risk_stats_measure_the_threshold_on_reviewed_rows(clean_table) -> None:
    """Two sick patients, one caught; two well patients, one cleared."""
    cases = [(0.9, True, True), (0.2, False, True), (0.7, True, False), (0.1, False, False)]
    for probability, flagged, had in cases:
        prediction_id = clean_table.log_risk_prediction("liver", {}, [], probability, 0.5, flagged)
        clean_table.record_risk_review(prediction_id, had)
    clean_table.log_risk_prediction("liver", {}, [], 0.6, 0.5, True)  # not reviewed yet

    stats = clean_table.risk_stats()["models"]["liver"]
    assert stats["total"] == 5
    assert stats["reviewed"] == 4
    assert stats["sensitivity"] == 0.5
    assert stats["specificity"] == 0.5
    assert stats["observed_rate"] == 0.5
    assert stats["mean_predicted"] == pytest.approx(0.475, abs=1e-4)
