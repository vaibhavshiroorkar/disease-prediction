"""Logging when the pool is open but the server behind it has gone away.

No database is needed: the pool is replaced by one whose acquire always times
out, which is what psycopg_pool does once Postgres stops answering.
"""

from __future__ import annotations

import pytest

from app import db
from tests.conftest import requires_model


class UnreachablePool:
    def connection(self, timeout=None):
        raise TimeoutError("couldn't get a connection after 2.00 sec")


@pytest.fixture
def unreachable(monkeypatch):
    monkeypatch.setattr(db, "_pool", UnreachablePool())


def test_logging_calls_return_empty_instead_of_raising(unreachable) -> None:
    ranked = [{"disease": "Acne", "probability": 0.9, "confidence": "high"}]
    assert db.log_prediction(["itching"], [], 5, ranked) is None
    assert db.recent_predictions() == []
    assert db.record_review(1, "Acne", was_correct=False) is False


@requires_model
def test_prediction_succeeds_when_the_database_is_unreachable(client, unreachable) -> None:
    response = client.post("/predict/symptoms", json={"symptoms": ["itching", "skin_rash"]})
    assert response.status_code == 200
    response = client.post("/predict/heart", json={"age": 54, "sex": 1, "cp": 4})
    assert response.status_code == 200
