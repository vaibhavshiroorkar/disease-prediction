"""Shared fixtures.

The API tests run against the real artifact in models/. They are skipped rather
than failed when it is absent, so a fresh clone can run the data and unit tests
before spending time on training.

Database tests need a throwaway Postgres. Point TEST_DATABASE_URL at one to
enable them; without it they skip and the rest of the suite still runs, which is
the same "works with no database" path the service itself supports.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import app
from app.registry import MODEL_NAMES, registry

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "").strip()


def artifacts_present() -> bool:
    return all(registry.path_for(name).exists() for name in MODEL_NAMES)


requires_model = pytest.mark.skipif(
    not artifacts_present(),
    reason="trained artifact missing, run: python -m ml.train",
)

requires_db = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="set TEST_DATABASE_URL to a throwaway Postgres to run database tests",
)


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def database():
    """Open a pool against the test database and start from an empty table."""
    if not TEST_DATABASE_URL:
        pytest.skip("no TEST_DATABASE_URL")

    db.close_pool()
    db.init_pool(TEST_DATABASE_URL)
    if not db.enabled():
        pytest.skip("could not connect to TEST_DATABASE_URL")

    yield db
    db.close_pool()


@pytest.fixture()
def clean_table(database):
    """Truncate between tests so counts are predictable."""
    with database.cursor() as cur:
        cur.execute("truncate table predictions, risk_predictions restart identity")
    yield database


@pytest.fixture(scope="session")
def symptom_keys(client) -> list[str]:
    response = client.get("/symptoms")
    if response.status_code != 200:
        pytest.skip("symptom model unavailable")
    return [entry["key"] for entry in response.json()["symptoms"]]
