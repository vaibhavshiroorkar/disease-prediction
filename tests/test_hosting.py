"""Public deployment boundaries: privacy, resource limits and readiness."""

from __future__ import annotations

import pytest

from app import config
from app.main import app
from app.registry import registry


@pytest.mark.parametrize("path", [
    "/predictions", "/predictions/wrong", "/predictions/stats",
    "/risk-predictions", "/risk-predictions/stats",
])
def test_history_requires_an_admin_key(client, monkeypatch, path):
    monkeypatch.setattr(config, "ADMIN_API_KEY", "test-admin-key", raising=False)
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"X-API-Key": "wrong"}).status_code == 401


@pytest.mark.parametrize("path,payload", [
    ("/predictions/1/review", {"was_correct": True}),
    ("/risk-predictions/1/review", {"had_condition": True}),
])
def test_reviews_require_an_admin_key(client, monkeypatch, path, payload):
    monkeypatch.setattr(config, "ADMIN_API_KEY", "test-admin-key", raising=False)
    assert client.post(path, json=payload).status_code == 401


def test_unconfigured_admin_access_is_closed(client, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_KEY", "", raising=False)
    assert client.get("/predictions").status_code == 503


def test_admin_key_opens_history_without_becoming_a_cookie(client, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_KEY", "test-admin-key", raising=False)
    response = client.get("/predictions", headers={"X-API-Key": "test-admin-key"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert "set-cookie" not in response.headers
    assert response.headers["cache-control"] == "no-store"


def test_public_health_does_not_publish_review_statistics(client):
    database = client.get("/health").json()["database"]
    assert "logging_enabled" in database
    assert "total" not in database
    assert "observed_accuracy" not in database


def test_homepage_is_html_and_metadata_has_its_own_url(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<main" in response.text
    assert client.get("/api").json()["service"] == "disease-prediction-api"
    assert client.get("/static/app.js").status_code == 200


def test_browser_responses_prevent_framing_and_external_scripts(client):
    response = client.get("/")
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_prediction_limit_is_enforced_without_trusting_forwarded_headers(client, monkeypatch):
    from app.security import SlidingWindow

    monkeypatch.setattr(app.state, "prediction_limiter", SlidingWindow(limit=2))
    assert client.post("/predict/symptoms", json={}).status_code == 422
    assert client.post("/predict/symptoms", json={}).status_code == 422
    response = client.post("/predict/symptoms", json={}, headers={"X-Forwarded-For": "other"})
    assert response.status_code == 429
    assert int(response.headers["retry-after"]) > 0
    assert client.get("/health").status_code == 200


def test_limiter_recovers_after_the_window_and_bounds_client_storage():
    from app.security import SlidingWindow

    now = [0.0]
    limiter = SlidingWindow(limit=1, window=60, max_clients=2, clock=lambda: now[0])
    assert limiter.retry_after("one") == 0
    assert limiter.retry_after("one") == 60
    assert limiter.retry_after("two") == 0
    assert limiter.retry_after("three") > 0
    now[0] = 60.1
    assert limiter.retry_after("three") == 0
    assert limiter.retry_after("one") == 0


def test_large_streamed_upload_is_rejected_before_image_decoding(client):
    response = client.post(
        "/predict/malaria", content=(b"x" * 1_000_000 for _ in range(3)),
        headers={"Content-Type": "image/png"},
    )
    assert response.status_code == 413


def test_compressed_image_with_excessive_pixel_count_is_rejected(client):
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2500, 2500), "white").save(buffer, format="PNG")
    response = client.post(
        "/predict/malaria", content=buffer.getvalue(), headers={"Content-Type": "image/png"},
    )
    assert response.status_code == 422
    assert "dimensions" in response.json()["detail"]


def test_ready_fails_when_a_model_is_unavailable(client, monkeypatch):
    monkeypatch.setattr(registry, "status", lambda: {"symptom": {"loaded": False}})
    assert client.get("/ready").status_code == 503


def test_ready_succeeds_when_models_are_loaded(client):
    assert client.get("/ready").status_code == 200


def test_ready_with_lazy_loading_checks_the_files_on_disk(client, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "PRELOAD_MODELS", False)
    monkeypatch.setattr(registry, "_cache", {})
    assert client.get("/ready").status_code == 200
    monkeypatch.setattr(registry, "_errors", {"heart": "could not read heart.joblib"})
    assert client.get("/ready").status_code == 503


def test_limiter_keys_on_the_entry_the_trusted_proxy_appended(client, monkeypatch):
    from app.security import SlidingWindow

    monkeypatch.setattr(config, "TRUSTED_PROXY_HOPS", 1)
    monkeypatch.setattr(app.state, "prediction_limiter", SlidingWindow(limit=2))
    # The client controls everything before the last entry, so changing it
    # must not buy a fresh budget.
    for forged in ["1.1.1.1", "2.2.2.2"]:
        headers = {"X-Forwarded-For": f"{forged}, 203.0.113.9"}
        assert client.post("/predict/symptoms", json={}, headers=headers).status_code == 422
    headers = {"X-Forwarded-For": "3.3.3.3, 203.0.113.9"}
    assert client.post("/predict/symptoms", json={}, headers=headers).status_code == 429
    # A different real client still has its own budget.
    headers = {"X-Forwarded-For": "198.51.100.4"}
    assert client.post("/predict/symptoms", json={}, headers=headers).status_code == 422


def test_client_address_falls_back_to_the_socket_without_enough_entries():
    from app.security import client_address

    scope = {"client": ("10.0.0.1", 1234), "headers": [(b"x-forwarded-for", b"9.9.9.9")]}
    assert client_address(scope, 0) == "10.0.0.1"
    assert client_address(scope, 1) == "9.9.9.9"
    assert client_address(scope, 2) == "10.0.0.1"
    assert client_address({"client": None, "headers": []}, 1) == "unknown"
