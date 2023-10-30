"""Shared fixtures.

The API tests run against the real artifact in models/. They are skipped rather
than failed when it is absent, so a fresh clone can run the data and unit tests
before spending time on training.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.registry import MODEL_NAMES, registry


def artifacts_present() -> bool:
    return all(registry.path_for(name).exists() for name in MODEL_NAMES)


requires_model = pytest.mark.skipif(
    not artifacts_present(),
    reason="trained artifact missing, run: python -m ml.train",
)


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def symptom_keys(client) -> list[str]:
    response = client.get("/symptoms")
    if response.status_code != 200:
        pytest.skip("symptom model unavailable")
    return [entry["key"] for entry in response.json()["symptoms"]]
